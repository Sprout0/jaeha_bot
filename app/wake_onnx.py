"""ONNX 호출어 감지기 — 세션 트리거의 '감지기' 부분(교체 가능).

기존 STT 감지기(app/wake.py)는 whisper 가 받아쓴 '텍스트'를 '재하봇'과 자모 비교했다.
그 방식은 (1)'재하봇'이 OOV 라 유아 웅얼거림에서 딴판으로 적히고 (2)받아쓰는 4~6초 동안
호출을 놓쳐 구조적 한계가 있었다. 이 감지기는 받아쓰기를 하지 않고 음향 패턴만 본다.

체인(openWakeWord 규격, livekit-wakeword 호환):
    80ms 프레임(1280샘플)
      -> melspectrogram.onnx -> 멜 5프레임 -> [x/10+2] -> 멜 링버퍼(76)
      -> embedding_model.onnx(창 76, 보폭 5) -> 임베딩(96) -> 임베딩 링버퍼(16)
      -> <호출어>.onnx -> score

조용한 실패 지점 2가지(예외가 안 나고 점수만 망가진다):
  1) 멜 입력은 int16 '범위' 값을 float32 로 준다. 우리 오디오는 [-1,1] 이라 32767 배 해야 한다.
  2) 멜 출력에 x/10+2 정규화를 반드시 적용한다.
[[jaeha-bot-progress]]
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio_source import FRAME, SAMPLE_RATE

log = logging.getLogger("jaeha_bot.wake_onnx")

MEL_WINDOW = 76     # 임베딩 하나가 보는 멜 프레임 수
MEL_BANDS = 32
MEL_PER_FRAME = 5   # 1280샘플이 만드는 멜 프레임 수 = 보폭과 같음.
                    # 실측(2026-08-02, livekit-wakeword melspectrogram.onnx): 5행.
                    # openWakeWord 규격 문서의 8이 아니다(hop 길이가 다름) — 실측값을 쓴다.
                    # push() 는 나온 행 수만큼 넣으므로 동작엔 영향 없고, WARMUP_FRAMES 계산에만 쓰인다.
EMB_WINDOW = 16     # 분류기가 보는 임베딩 개수
EMB_DIM = 96
INT16_SCALE = 32767.0


def _peak_rms(audio: np.ndarray, win: int = FRAME) -> float:
    """구간 안에서 가장 큰 프레임의 RMS. 2초 평균은 앞뒤 무음에 희석되므로 최대를 본다."""
    if audio.size < win:
        return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return max(float(np.sqrt(np.mean(np.square(audio[i:i + win]))))
               for i in range(0, audio.size - win + 1, win))


@dataclass
class WakeResult:
    """호출어가 걸렸을 때 상위(main)에 넘기는 것."""
    preroll: np.ndarray   # 감지 직전 오디오(+이어진 발화). continued=False 면 빈 배열
    continued: bool       # 감지 직후에도 말이 이어졌는가(인사말 생략 판단)
    score: float = 0.0


class OnnxWakeDetector:
    # 멜 76프레임 채우기 = ceil(76/8) = 10 프레임,
    # 그 뒤 임베딩 16개 채우기 = 15 프레임 더. 총 25번째 push 에서 첫 점수.
    WARMUP_FRAMES = -(-MEL_WINDOW // MEL_PER_FRAME) + EMB_WINDOW - 1

    def __init__(self, model_dir, classifier: str = "jaehabot.onnx",
                 threshold: float = 0.5, trigger_frames: int = 2,
                 providers=None, source=None,
                 continuation_window: float = 0.5,
                 verifier=None, verify_cooldown_s: float = 1.0,
                 verify_min_rms: float = 0.005) -> None:
        import pathlib

        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        d = pathlib.Path(model_dir)

        def _load(name):
            p = d / name
            if not p.exists():
                raise FileNotFoundError(f"ONNX 없음: {p}")
            return ort.InferenceSession(str(p), providers=providers)

        self._mel_sess = _load("melspectrogram.onnx")
        self._emb_sess = _load("embedding_model.onnx")
        self._cls_sess = _load(classifier)
        log.info("호출어 ONNX 로드: %s (providers=%s)", classifier, providers)

        self._init_state(threshold, trigger_frames, continuation_window, source,
                         verifier, verify_cooldown_s, verify_min_rms)

    def _init_state(self, threshold, trigger_frames, continuation_window, source,
                    verifier=None, verify_cooldown_s: float = 1.0,
                    verify_min_rms: float = 0.005):
        """__init__ 과 테스트가 공유하는 순수 상태 초기화(ONNX 로드 없음).

        ⚠️ 새 상태는 **반드시 여기에** 둔다. __init__ 에만 두면 _init_state 로 만든
           객체에서 AttributeError 가 난다(실제로 겪었다).
        """
        self.threshold = float(threshold)
        self.trigger_frames = int(trigger_frames)
        self.continuation_window = float(continuation_window)
        self.source = source
        self._mel: deque[np.ndarray] = deque(maxlen=MEL_WINDOW)
        self._emb: deque[np.ndarray] = deque(maxlen=EMB_WINDOW)
        self._hits = 0
        # 2단계 검증기: audio(np.float32) -> bool. None 이면 1단계만으로 깨어난다(= 옛 동작).
        self.verifier = verifier
        self.verify_cooldown_s = float(verify_cooldown_s)
        self.verify_min_rms = float(verify_min_rms)   # 에너지 게이트 하한(튜닝 손잡이)
        self._armed = True       # 히스테리시스: 기각 후에는 점수가 임계 아래로 내려가야 재무장
        self._last_verify = 0.0  # 쿨다운 기준 시각

    # ------------------------------------------------------------------ 체인
    def reset(self) -> None:
        """링버퍼를 비운다. 깨어난 뒤 다시 대기로 갈 때 호출."""
        self._mel.clear()
        self._emb.clear()
        self._hits = 0
        self._armed = True

    def push(self, frame: np.ndarray) -> float | None:
        """80ms 프레임 하나를 넣고 점수를 얻는다. 워밍업 중이면 None."""
        # 1) 멜: int16 범위 float32 로 넣고, 출력에 x/10+2 정규화
        audio = np.asarray(frame, dtype=np.float32).reshape(1, -1) * INT16_SCALE
        name = self._mel_sess.get_inputs()[0].name
        mel = np.squeeze(self._mel_sess.run(None, {name: audio})[0])
        mel = mel.reshape(-1, MEL_BANDS) / 10.0 + 2.0
        for row in mel:
            self._mel.append(row.astype(np.float32))
        if len(self._mel) < MEL_WINDOW:
            return None

        # 2) 임베딩: 멜 76프레임 창 -> 96차원 하나
        window = np.stack(list(self._mel))[None, :, :, None].astype(np.float32)
        name = self._emb_sess.get_inputs()[0].name
        emb = np.squeeze(self._emb_sess.run(None, {name: window})[0])
        self._emb.append(emb.reshape(EMB_DIM).astype(np.float32))
        if len(self._emb) < EMB_WINDOW:
            return None

        # 3) 분류기: 임베딩 16개 -> 점수
        feats = np.stack(list(self._emb))[None, :, :].astype(np.float32)
        name = self._cls_sess.get_inputs()[0].name
        return float(np.squeeze(self._cls_sess.run(None, {name: feats})[0]))

    # -------------------------------------------------------------- 대기 루프
    def wait_for_wake(self, max_frames: int | None = None) -> WakeResult | None:
        """호출어가 걸릴 때까지 프레임을 읽는다. 걸리면 WakeResult, 아니면 None.

        감지기 교체 가능한 표면은 **인자 없는 호출**뿐이다(`wait_for_wake()`).
        `max_frames` 는 테스트·진단 전용이며, `SttWakeDetector.wait_for_wake`의
        `max_turns`와 단위가 다르다(프레임 수 vs 턴 수). 감지기 종류를 모르는
        호출부(main.py)는 이 인자를 절대 넘기면 안 된다. None 이면 걸릴 때까지 계속 듣는다.
        """
        if self.source is None:
            raise RuntimeError("source 가 없다 — AudioSource 를 넘겨야 한다")

        n = 0
        best = 0.0
        while max_frames is None or n < max_frames:
            n += 1
            score = self.push(self.source.read())
            if score is None:      # 워밍업
                continue
            best = max(best, score)

            if score < self.threshold:
                self._hits = 0
                self._armed = True   # 점수가 내려왔다 = 다음 상승은 '새 발화'다
                # 진단: 임계값 튜닝 근거. 대기 중 주기적으로 최고 점수를 남긴다.
                if n % 250 == 0:   # 250프레임 = 20초
                    log.info("[대기] 최근 20초 최고 점수 %.3f (임계 %.2f)", best, self.threshold)
                    best = 0.0
                continue

            self._hits += 1
            if self._hits < self.trigger_frames:
                continue

            # ── 1단계 통과 = '후보' ──
            if self.verifier is not None and not self._verify(score):
                continue

            # ── 깨움 확정 ──
            log.info("[호출] 점수 %.3f → 깨어남", score)
            pre = self.source.preroll()          # 감지 직전 0.5초(호출어 포함)
            tail, continued = self._observe_continuation()
            self._hits = 0
            if continued:
                audio = np.concatenate([pre, tail]) if tail.size else pre
            else:
                # 부르고 기다리는 패턴 — 인사말을 하는 사이 낡으므로 버린다.
                audio = np.zeros(0, dtype=np.float32)
                self.source.clear_preroll()
            return WakeResult(preroll=audio.astype(np.float32),
                              continued=continued, score=score)
        return None

    def _verify(self, score: float) -> bool:
        """2단계 — 후보 구간을 검증기에 넘겨 진짜 호출인지 본다.

        1단계 임계를 낮게(0.03) 두는 대신 여기서 헛깨움을 걷어낸다. 실측(2026-08-12,
        실음성 44건): 1단계만 0.25 -> 재현율 30%, 캐스케이드(0.03+검증) -> **80%**,
        헛깨움 추정 0.47 -> 0.27회·시간. 둘이 같이 좋아지는 건 재현율과 헛깨움을
        **서로 다른 손잡이**로 분리했기 때문이다.

        🔴 폭주 방지가 이 함수의 절반이다. 임계 0.03 은 TV 소리 한 문장에도 여러 프레임
           연속으로 넘는다. 그대로 두면 **초당 12번** whisper 를 부른다. 그래서:
             ① 히스테리시스 — 기각했으면 점수가 임계 아래로 내려갔다 와야 다시 부른다
             ② 쿨다운      — 그래도 최소 verify_cooldown_s 는 쉰다
        """
        import time

        if not self._armed:
            return False
        now = time.monotonic()
        if now - self._last_verify < self.verify_cooldown_s:
            return False

        self._armed = False          # 판정 전에 내린다 — 예외가 나도 폭주하지 않게
        self._last_verify = now
        audio = self.source.verify_window()

        # 에너지 게이트 — **디지털 무음만** 거른다. whisper 를 헛되이 부르지 않기 위한 것뿐이다.
        #
        # 🔴 2026-08-12 실기에서 이 게이트가 진짜 호출을 막았다. 원래는 소음 바닥의 2배를
        #    기준으로 삼았는데, 유튜브를 틀면 바닥이 0.005 -> **0.024** 로 뛰어 기준이
        #    0.0485 가 된다. 진짜 호출의 최대프레임 RMS 는 최소 0.0515 라 경계에 걸리고,
        #    실제로 점수 0.103 짜리 후보가 RMS 0.0344 로 잘려 나갔다(로그 14:03:40).
        #    ➡️ **소음 바닥에 비례시키지 않는다.** 시끄러울수록 기준이 올라가면 정확히
        #       가장 필요한 순간에 귀를 닫는 꼴이 된다.
        #    ➡️ 환각 방어는 이제 게이트가 아니라 '두 번 대조'가 한다(app/wake.py). 게이트는
        #       무음에 whisper 를 안 쓰는 절약 장치로만 남긴다.
        loudest = _peak_rms(audio)
        if loudest < self.verify_min_rms:
            log.info("[검증] 무음이라 건너뜀 — 최대 RMS %.4f < %.4f (점수 %.3f)",
                     loudest, self.verify_min_rms, score)
            return False

        try:
            ok = bool(self.verifier(audio))
        except Exception as e:
            # whisper 가 죽어도 봇은 계속 들어야 한다. 안전하게 '기각'으로 본다.
            log.warning("[검증] 실패(기각 처리): %s: %s", type(e).__name__, e)
            return False
        if not ok:
            # 실기 튜닝의 근거가 되는 줄이다. 실제 가정 소음에서 무엇이 후보로 뜨는지
            # 이 로그로만 알 수 있다(합성 부정으로는 확인이 안 된다).
            log.info("[검증] 후보 기각 — 점수 %.3f, 오디오 %.2fs",
                     score, audio.size / SAMPLE_RATE)
        self._hits = 0
        return ok

    def _observe_continuation(self) -> tuple[np.ndarray, bool]:
        """감지 직후 잠깐 들어 말이 이어지는지 본다. (읽은 오디오, 이어짐 여부)."""
        n = max(1, int(self.continuation_window * SAMPLE_RATE / FRAME))
        frames = []
        for _ in range(n):
            try:
                frames.append(self.source.read())
            except StopIteration:
                break
        if not frames:
            return np.zeros(0, dtype=np.float32), False
        tail = np.concatenate(frames).astype(np.float32)
        # 말소리 판정: STT VAD 와 같은 기준(소음 바닥의 2배, 하한 0.005)을 쓴다.
        floor = getattr(self.source, "noise_floor", 0.0)
        thr = max(floor * 2.0, 0.005)
        loudest = max(float(np.sqrt(np.mean(np.square(f)))) for f in frames)
        return tail, loudest >= thr
