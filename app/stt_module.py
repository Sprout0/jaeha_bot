"""STT: faster-whisper 기반 음성 -> 텍스트 (VAD 자동 녹음 포함).

목표(가이드 4-2): 인식률보다 전체 반응 안정성 우선.
운영 구성은 `configs/model_paths.yaml` 에 있다(젯슨: large-v3-turbo / cuda / float16).
🔴 **한국어 파인튜닝판(whisper-small-ko)은 반복 환각의 원인이라 폐기했다.** 순정 다국어
   모델을 쓴다 — 여기 디코딩 옵션 대부분이 그 파인튜닝의 상처를 덮으려던 장치였고,
   순정에서는 그것들 없이도 무너지지 않는다(그래도 무해해서 유지 중).
- listen(): 마이크에서 말이 시작되면 자동 녹음, 조용해지면 자동 종료 -> 텍스트.
- transcribe(): numpy 오디오(또는 wav 경로) -> (텍스트, 처리시간초).

VAD 는 별도 라이브러리 없이 에너지(RMS) 기반으로 구현.
주변 소음을 먼저 재서 임계값을 자동 보정하므로 환경마다 손댈 일이 적다.
REPL 테스트: python -m app.stt_module
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .audio_source import FRAME
from .text_norm import correct_stt, dedupe_repeats

log = logging.getLogger("jaeha_bot.stt")

SAMPLE_RATE = 16000  # faster-whisper 표준 입력


class STTModule:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        *,
        cpu_threads: int | None = None,  # ctranslate2 디코딩 스레드. None 이면 라이브러리 기본(4).
                                         # 젯슨은 6코어인데 기본 4만 써서 놀았다 — 실측 4.56s -> 3.84s.
                                         # 디코딩 방식이 아니라 병렬도만 바꾸므로 인식 품질은 불변.
        language: str = "ko",
        # --- VAD(에너지 기반) 파라미터 ---
        silence_ratio: float = 2.0,     # 소음 대비 몇 배 넘으면 '말'로 볼지
        min_start_rms: float = 0.005,   # 임계값 하한. 0.010 은 흐린/작은 발화를 놓쳐(입력없음)
                                        # 0.005 로 낮춤(소음 바닥 ~0.0004 의 ~12배라 오작동 여유).
                                        # 여전히 놓치면 아래 '입력 없음' 로그의 최대 RMS 보고 더 낮출 것.
        silence_duration: float = 1.5,  # 이만큼 연속 조용하면 종료(초). 유아는 뚝뚝 끊어 말해 넉넉히.
        pre_roll: float = 0.3,          # 말 시작 직전 이만큼을 미리 담아 첫 음절 안 잘리게(초)
        max_duration: float = 10.0,     # 안전장치: 최대 녹음 길이(초)
        start_timeout: float = 15.0,    # 말이 없으면 이 시간 후 포기(초)
        # --- 오인식 사후 교정(고유명사 OOV: 재하봇->재하보사 등) ---
        keywords: list | None = None,   # 자모 편집거리로 가까우면 이 단어로 교정
        aliases: dict | None = None,    # 확정 오인식 정확 매핑 {"제아부사": "재하봇"}
        initial_prompt: str | None = None,  # Whisper 디코딩 힌트(고유명사 원천 보정).
                                            # 환각 위험 있어 기본 off. 마이크로 검증 후 켤 것.
        # --- 반복 환각 가드 ---
        max_chars_per_sec: float | None = None,  # 오디오 1초당 이 글자수를 넘으면 폭주로 보고 버린다.
                                                 # None 이면 끔(옛 동작). 근거는 tests/ 와 config 주석.
        min_chars_to_reject: int = 20,   # 이 길이 미만은 아무리 빨라도 버리지 않는다.
                                         # 짧은 유아 발화 보호용(진짜 폭주는 본질적으로 길다).
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self.silence_ratio = silence_ratio
        self.min_start_rms = min_start_rms
        self.silence_duration = silence_duration
        self.pre_roll = pre_roll
        self.max_duration = max_duration
        self.start_timeout = start_timeout
        self.keywords = keywords or []
        self.aliases = aliases or {}
        self.initial_prompt = initial_prompt
        self.max_chars_per_sec = max_chars_per_sec
        self.min_chars_to_reject = min_chars_to_reject
        # 직전 transcribe 가 '환각이라 버림'이었는지. main 이 이걸 봐야 '말이 없었다'와
        # 구분해 되물을 수 있다(빈 문자열만으로는 구분 불가 → 침묵하게 된다).
        self.last_rejected = False
        # 직전 녹음의 'VAD 꼬리' — 마지막 말소리부터 녹음 종료까지(초).
        # 아이가 순전히 기다리는 구간이라 지연 예산에서 따로 봐야 한다.
        # 🔴 silence_duration 을 그대로 쓰면 안 된다. quiet 카운터가 말소리에서
        #    감쇠하므로 실제 꼬리는 설정값보다 짧을 수도 길 수도 있다.
        self.last_vad_tail_s = 0.0
        self._model = None

    # ------------------------------------------------------------------ 모델
    def load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            kw = {}
            if self.cpu_threads:
                kw["cpu_threads"] = self.cpu_threads
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type,
                **kw,
            )
        return self._model

    # -------------------------------------------------------------- 인식(STT)
    def transcribe(self, audio, initial_prompt: str | None = None) -> tuple[str, float]:
        """오디오(float32 numpy [-1,1] 또는 wav 경로) -> (텍스트, 처리시간초).

        빈 입력/잡음은 '' 반환하고 복구는 상위(main)에서 처리한다.

        initial_prompt: 이 호출에만 쓰는 디코딩 힌트. None 이면 인스턴스 기본값.
          🔴 호출어 2단계 검증에 필수다. 실측(2026-08-12, 실음성 44건, 자모컷 0.45):
             힌트 없음 41% / '재하봇' 64% / **"재하봇아, 재하봇이, 재하봇 불러." 98%**
             (긍정 자모거리 중앙 0.00 / 부정 0.86). 부정 통과는 2% 로 유지된다.
          ⚠️ **대화용 STT 에는 절대 쓰지 말 것.** 아이가 무슨 말을 하든 '재하봇' 쪽으로
             편향된다. 그래서 인스턴스 고정값이 아니라 호출별 인자로 받는다.
             (모델을 두 벌 올리는 건 불가 — turbo 적재 후 여유가 2GB 뿐이다.)
        """
        model = self.load()
        self.last_rejected = False
        # 가드는 '오디오 길이'가 있어야 계산된다. wav 경로 입력은 길이를 모르므로 건너뛴다
        # (_trim_edges 와 같은 제약 — 실파이프라인은 numpy 라 무영향).
        duration = audio.size / SAMPLE_RATE if isinstance(audio, np.ndarray) else 0.0
        # numpy 오디오면 앞뒤 무음을 잘라 환각/반복을 막는다. Whisper 는 무음을 만나면
        # 학습된 문장을 게워내는 성질이 있고(특히 좁게 파인튜닝된 한국어 모델), 뒤 침묵을
        # 제거하면 이 환각·반복·fallback 폭주가 사라진다. (실측: 0/5 -> 4/5, 11s -> 2s)
        if isinstance(audio, np.ndarray):
            audio = _trim_edges(audio)
        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,   # whisper 내장 VAD 로 무음 구간 한 번 더 제거
            beam_size=5,       # 그리디(beam=1)는 반복 환각에 잘 빠진다. 빔서치가 이를 크게 억제.
                               # 실측(small): beam1=환각2·정답2 -> beam5=환각0·정답4, 2.0->2.4s.
                               # (beam1 은 실아동 음성서 CER 0.29 로 붕괴 확인 2026-07-27, 유지.)
            no_repeat_ngram_size=3,  # 같은 3-gram 반복 금지 → "하하하…"·"아웃아웃…" 무한루프를
                               # 디코딩 단계서 조기 종료(짧은발화+끝모음 폭주로 18~62s 튀던 것 차단).
                               # 실측(AI-Hub 30): CER 0.213->0.120(정확도 오히려↑, 후처리 dedupe와 동급).
            # 영어/불분명 발음이 들어오면 한국어 강제 모델이 확신을 못 해 온도를 올려
            # 재디코딩(fallback)한다. 기본 6단계면 beam5 와 겹쳐 7~8s 로 튄다.
            # 2단계로 캡을 씌워 최악을 ~2.6s 로 묶는다(깨끗한 한국어는 첫 시도에 성공해 무영향).
            temperature=[0.0, 0.2],
            condition_on_previous_text=False,  # 턴제 대화라 이전 문맥 이월/반복 방지
            # 호출별 인자가 있으면 그것을, 없으면 인스턴스 기본값을 쓴다.
            initial_prompt=(initial_prompt if initial_prompt is not None
                            else self.initial_prompt),
        )
        text = " ".join(s.text for s in segments).strip()
        # ① 반복 환각 제거(같은 구절 무한반복) — 실아동 음성서 유일 치명오류(24%). 모델무관 후처리.
        text = dedupe_repeats(text)
        # ② 고유명사 OOV 오인식(재하봇->재하보사)을 자모 편집거리·정확매핑으로 사후 교정.
        text = correct_stt(text, self.keywords, self.aliases)
        # ③ 남은 폭주(①이 못 잡는 '매번 다른 토큰' 형태)를 말속도로 거른다.
        if self._is_runaway(text, duration):
            self.last_rejected = True
            log.warning("환각으로 판단해 버림(%.1f글자/초 > %.1f): %s...",
                        _chars(text) / duration, self.max_chars_per_sec, text[:40])
            return "", time.perf_counter() - t0
        return text, time.perf_counter() - t0

    def _is_runaway(self, text: str, duration: float) -> bool:
        """오디오 길이 대비 글자수가 사람 말속도를 벗어나면 환각으로 본다.

        정답 길이와 비교하는 편이 훨씬 정확하지만 운영에선 정답을 모른다. 런타임에
        아는 것은 오디오 길이뿐이라 '글자/초'를 쓴다. (2026-08-04 실측: 실제 아동
        발화의 말속도는 중앙값 3.03, p95 5.44, 최대 6.88 글자/초.)
        """
        if not self.max_chars_per_sec or duration <= 0 or not text:
            return False
        n = _chars(text)
        if n < self.min_chars_to_reject:
            return False
        return n / duration > self.max_chars_per_sec

    # --------------------------------------------------------- 마이크 녹음(VAD)
    def record_until_silence(self, verbose: bool = False, *,
                             source=None, prefix=None) -> np.ndarray:
        """말이 시작되면 녹음, silence_duration 만큼 조용해지면 종료.

        source: AudioSource(공유 스트림). None 이면 예전처럼 자체 스트림을 연다.
                호출어 감지기와 마이크를 나눠 쓸 때 넘긴다(장치가 하나뿐이라 필수).
        prefix: 이미 확보한 앞부분 오디오(호출어 프리롤). 주면 '말 시작 대기'를
                건너뛰고 바로 무음 판정으로 들어간다(한 숨에 말한 경우).

        반환: float32 numpy 오디오(모노, 16kHz). 말이 없으면 빈 배열.
        """
        self.last_vad_tail_s = 0.0   # 이전 턴 값이 새 턴에 새지 않게
        if source is None:
            return self._record_own_stream(verbose)
        return self._record_from_source(source, verbose, prefix)

    def _record_from_source(self, source, verbose: bool, prefix) -> np.ndarray:
        """공유 스트림에서 녹음. 소음 바닥은 source 가 이미 재 뒀다(재측정 안 함).

        시간 계산은 source 가 실제로 공급하는 프레임 크기(``source.frame``, 없으면
        모듈 기본 FRAME)를 기준으로 한다 — 비기본 프레임 크기의 소스가 조용히
        시간 계산을 어긋내지 않도록.
        """
        from collections import deque

        frame_size = getattr(source, "frame", FRAME)
        threshold = max(getattr(source, "noise_floor", 0.0) * self.silence_ratio,
                        self.min_start_rms)

        collected: list[np.ndarray] = []
        if prefix is not None and np.asarray(prefix).size:
            collected.append(np.asarray(prefix, dtype=np.float32).reshape(-1))
            started = True
        else:
            started = False

        # 1) 말 시작 대기(prefix 가 있으면 건너뜀). own-stream 과 동일하게 직전
        #    pre_roll 초를 링버퍼에 담아 말 시작 직전 소리부터 포함한다(첫 음절 안 잘리게).
        if not started:
            pre_roll_frames = max(1, int(self.pre_roll * SAMPLE_RATE / frame_size))
            ring: deque = deque(maxlen=pre_roll_frames)
            waited = 0
            max_wait = int(self.start_timeout * SAMPLE_RATE / frame_size)
            max_seen = 0.0  # 대기 중 관측된 최대 RMS(입력없음 진단용)
            while waited < max_wait:
                waited += 1
                try:
                    block = source.read()
                except StopIteration:
                    return np.zeros(0, dtype=np.float32)
                r = _rms(block)
                if r > max_seen:
                    max_seen = r
                if r >= threshold:
                    collected = list(ring)  # 말 시작 직전 소리부터 포함
                    collected.append(block)
                    started = True
                    break
                ring.append(block)
            if not started:
                if verbose:
                    print(f"  (입력 없음 — 종료. 최대 관측 RMS={max_seen:.4f}, 시작 임계값={threshold:.4f})")
                return np.zeros(0, dtype=np.float32)

        # 2) 무음이 이어지면 종료. 임계는 절대값과 '직전 최고음량의 12%' 중 큰 값.
        #    quiet 감쇠는 own-stream 과 같은 시간폭(약 80ms 프레임 1개)이 되도록 -1
        #    (own-stream 은 30ms 프레임이라 -3 = 90ms). -3 을 그대로 쓰면 블립 하나가
        #    지우는 무음 예산이 2.67배로 뛰어, 말끝을 흘리는 아이의 녹음이 quiet
        #    카운터로 못 끊기고 max_duration 캡까지 끌려간다(예전에 고쳤던 실패 재발).
        quiet_needed = max(1, int(self.silence_duration * SAMPLE_RATE / frame_size))
        # 샘플 수 기준 상한: prefix 가 여러 프레임을 하나의 배열로 합쳐 넘길 수 있어
        # (예: AudioSource 의 preroll) 리스트 길이로 세면 캡이 밀린다.
        max_samples = int(self.max_duration * SAMPLE_RATE)
        total_samples = sum(int(np.asarray(b).size) for b in collected)
        quiet = 0
        speech_peak = threshold
        last_loud_samples = total_samples   # 마지막 말소리까지 모인 양(꼬리 계산 기준)
        while total_samples < max_samples:
            try:
                block = source.read()
            except StopIteration:
                break
            collected.append(block)
            total_samples += int(np.asarray(block).size)
            r = _rms(block)
            speech_peak = max(speech_peak, r)
            if r < max(threshold, speech_peak * 0.12):
                quiet += 1
                if quiet >= quiet_needed:
                    break
            else:
                quiet = max(0, quiet - 1)
                last_loud_samples = total_samples
        self.last_vad_tail_s = (total_samples - last_loud_samples) / SAMPLE_RATE

        if not collected:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(collected).astype(np.float32).reshape(-1)
        return _normalize(audio)

    def _record_own_stream(self, verbose: bool = False) -> np.ndarray:
        """자체 스트림을 열어 녹음한다(예전 동작)."""
        import sounddevice as sd

        from collections import deque

        frame = int(SAMPLE_RATE * 0.03)  # 30ms 프레임
        silence_frames = int(self.silence_duration / 0.03)
        max_frames = int(self.max_duration / 0.03)
        pre_roll_frames = int(self.pre_roll / 0.03)
        start_deadline = time.time() + self.start_timeout

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=frame
        ) as stream:
            # 1) 주변 소음 측정(0.3초) -> 시작 임계값 자동 보정
            noise = []
            for _ in range(10):
                block, _ = stream.read(frame)
                noise.append(_rms(block))
            noise_floor = float(np.mean(noise)) if noise else 0.0
            threshold = max(noise_floor * self.silence_ratio, self.min_start_rms)
            if verbose:
                print(f"  (소음 바닥={noise_floor:.4f}, 시작 임계값={threshold:.4f})")
                print("  말해보세요...")

            # 2) 말 시작 대기 (직전 pre_roll 초를 링버퍼에 담아 첫 음절 보존)
            ring = deque(maxlen=pre_roll_frames)
            max_seen = 0.0  # 대기 중 관측된 최대 RMS(입력없음 진단용)
            while True:
                block, _ = stream.read(frame)
                r = _rms(block)
                if r > max_seen:
                    max_seen = r
                if r >= threshold:
                    collected = list(ring)  # 말 시작 직전 소리부터 포함
                    collected.append(block)
                    break
                ring.append(block)
                if time.time() > start_deadline:
                    if verbose:
                        # 최대 관측 RMS 가 임계값에 근접했다면 min_start_rms 를 그만큼 낮추면 잡힌다.
                        print(f"  (입력 없음 — 종료. 최대 관측 RMS={max_seen:.4f}, 시작 임계값={threshold:.4f})")
                    return np.zeros(0, dtype=np.float32)

            # 3) 녹음 지속: 말이 잦아들면 종료.
            #    말끝을 '흘리면' 음량이 임계값 근처에서 떨려, 한 프레임 블립에 통째로
            #    리셋하면 종료가 안 잡혀 최대 길이까지 끌려간다(=처리 지연). 그래서:
            #    - 종료 임계값을 절대값 + '직전 말소리 최대치의 12%' 중 큰 값으로(상대 종료).
            #      흘리는 말끝은 자기 최고음량보다 훨씬 작아지므로 확실히 잡힌다.
            #    - loud 프레임엔 카운터를 0으로 리셋하지 않고 조금만 감산(블립에 관대).
            quiet = 0
            speech_peak = threshold
            last_loud_frames = len(collected)
            while len(collected) < max_frames:
                block, _ = stream.read(frame)
                collected.append(block)
                r = _rms(block)
                speech_peak = max(speech_peak, r)
                end_thr = max(threshold, speech_peak * 0.12)
                if r < end_thr:
                    quiet += 1
                    if quiet >= silence_frames:
                        break
                else:
                    quiet = max(0, quiet - 3)
                    last_loud_frames = len(collected)
            self.last_vad_tail_s = ((len(collected) - last_loud_frames)
                                    * frame / SAMPLE_RATE)

        audio = np.concatenate(collected).astype(np.float32).flatten()
        return _normalize(audio)

    # ------------------------------------------------- 녹음 + 인식 한 번에
    def listen(self, verbose: bool = False, *, source=None, prefix=None) -> tuple[str, float]:
        """마이크 -> (인식 텍스트, 인식 처리시간초). 말이 없으면 ('', 0.0)."""
        t0 = time.perf_counter()
        audio = self.record_until_silence(verbose=verbose, source=source, prefix=prefix)
        rec_dt = time.perf_counter() - t0
        if audio.size == 0:
            # 여기서 transcribe() 를 안 거치므로 플래그를 직접 지운다. 안 지우면 직전
            # 거부가 남아 main 이 무음 턴마다 되묻고, 영영 잠들지 않는다.
            self.last_rejected = False
            return "", 0.0
        text, tr_dt = self.transcribe(audio)
        if verbose:
            # '말끝 흘림→지연/미인식' 진단용: 녹음대기 vs 오디오길이 vs 트림후길이 vs 인식.
            # 짧게 말했는데 오디오가 길면(예: 10s) VAD 종료 늦음. 트림후가 너무 짧으면 끝말 잘림.
            trimmed_len = _trim_edges(audio).size / SAMPLE_RATE
            print(f"  [타이밍] 녹음대기 {rec_dt:.2f}s, 오디오 {audio.size / SAMPLE_RATE:.2f}s"
                  f"(트림후 {trimmed_len:.2f}s), 인식 {tr_dt:.2f}s")
            if not text:
                # 빈 인식이면 녹음본을 저장해 원인(잘림/버려짐/불명확)을 직접 분석한다.
                import soundfile as sf
                fn = f"logs/empty_{int(time.time())}.wav"
                try:
                    sf.write(fn, audio, SAMPLE_RATE)
                    print(f"  [진단] 빈 인식 → 녹음본 저장: {fn}")
                except Exception as e:
                    print(f"  [진단] 녹음본 저장 실패: {e}")
        return text, tr_dt


def _chars(text: str) -> int:
    """공백을 뺀 글자수. 띄어쓰기 습관이 말속도 판정을 흔들면 안 된다."""
    return len("".join(text.split()))


def _trim_edges(audio: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """앞뒤 무음을 제거해 Whisper 환각/반복을 막는다. 말 앞뒤로 pad 초만 남긴다.

    임계값은 신호 최대치의 2%로 잡아, _normalize 로 볼륨을 키운 녹음에도 맞게 적응한다.
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:  # 사실상 무음
        return audio
    thr = max(2e-3, 0.02 * peak)
    nz = np.where(np.abs(audio) > thr)[0]
    if nz.size == 0:
        return audio
    start = max(0, int(nz[0]) - int(pad * SAMPLE_RATE))
    end = min(audio.size, int(nz[-1]) + int(pad * SAMPLE_RATE))
    return audio[start:end]


def _normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """작게 말한 목소리도 모델이 듣기 쉽게 최대 진폭을 peak 로 끌어올린다."""
    if audio.size == 0:
        return audio
    m = float(np.max(np.abs(audio)))
    if m < 1e-6:  # 사실상 무음이면 증폭하지 않음(잡음 폭주 방지)
        return audio
    return (audio * (peak / m)).astype(np.float32)


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))


def _repl() -> None:
    """터미널 마이크 테스트. Ctrl+C 로 종료."""
    from .config import settings

    stt = STTModule(**settings.models.get("stt", {}))
    print("STT 모델 로딩 중... (첫 실행은 모델 다운로드로 느릴 수 있음)")
    stt.load()
    print("준비 완료! 마이크에 대고 말해보세요. (Ctrl+C 종료)\n")
    try:
        while True:
            text, dt = stt.listen(verbose=True)
            if text:
                print(f"  인식: {text}   (처리 {dt:.2f}s)\n")
            else:
                print("  (인식된 말 없음)\n")
    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    _repl()
