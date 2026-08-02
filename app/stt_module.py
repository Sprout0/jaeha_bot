"""STT: faster-whisper 기반 음성 -> 텍스트 (VAD 자동 녹음 포함).

목표(가이드 4-2): 인식률보다 전체 반응 안정성 우선. base/int8 로 시작.
- listen(): 마이크에서 말이 시작되면 자동 녹음, 조용해지면 자동 종료 -> 텍스트.
- transcribe(): numpy 오디오(또는 wav 경로) -> (텍스트, 처리시간초).

VAD 는 별도 라이브러리 없이 에너지(RMS) 기반으로 구현.
주변 소음을 먼저 재서 임계값을 자동 보정하므로 환경마다 손댈 일이 적다.
REPL 테스트: python -m app.stt_module
"""
from __future__ import annotations

import time

import numpy as np

from .text_norm import correct_stt, dedupe_repeats

SAMPLE_RATE = 16000  # faster-whisper 표준 입력


class STTModule:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        *,
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
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
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
        self._model = None

    # ------------------------------------------------------------------ 모델
    def load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    # -------------------------------------------------------------- 인식(STT)
    def transcribe(self, audio) -> tuple[str, float]:
        """오디오(float32 numpy [-1,1] 또는 wav 경로) -> (텍스트, 처리시간초).

        빈 입력/잡음은 '' 반환하고 복구는 상위(main)에서 처리한다.
        """
        model = self.load()
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
            initial_prompt=self.initial_prompt,  # None 이면 무영향. 고유명사 원천 보정용 힌트.
        )
        text = " ".join(s.text for s in segments).strip()
        # ① 반복 환각 제거(같은 구절 무한반복) — 실아동 음성서 유일 치명오류(24%). 모델무관 후처리.
        text = dedupe_repeats(text)
        # ② 고유명사 OOV 오인식(재하봇->재하보사)을 자모 편집거리·정확매핑으로 사후 교정.
        text = correct_stt(text, self.keywords, self.aliases)
        return text, time.perf_counter() - t0

    # --------------------------------------------------------- 마이크 녹음(VAD)
    def record_until_silence(self, verbose: bool = False) -> np.ndarray:
        """말이 시작되면 녹음, silence_duration 만큼 조용해지면 종료.

        반환: float32 numpy 오디오(모노, 16kHz). 말이 없으면 빈 배열.
        """
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

        audio = np.concatenate(collected).astype(np.float32).flatten()
        return _normalize(audio)

    # ------------------------------------------------- 녹음 + 인식 한 번에
    def listen(self, verbose: bool = False) -> tuple[str, float]:
        """마이크 -> (인식 텍스트, 인식 처리시간초). 말이 없으면 ('', 0.0)."""
        t0 = time.perf_counter()
        audio = self.record_until_silence(verbose=verbose)
        rec_dt = time.perf_counter() - t0
        if audio.size == 0:
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
