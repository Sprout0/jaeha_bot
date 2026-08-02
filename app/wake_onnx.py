"""ONNX 호출어 감지기 — 세션 트리거의 '감지기' 부분(교체 가능).

기존 STT 감지기(app/wake.py)는 whisper 가 받아쓴 '텍스트'를 '재하봇'과 자모 비교했다.
그 방식은 (1)'재하봇'이 OOV 라 유아 웅얼거림에서 딴판으로 적히고 (2)받아쓰는 4~6초 동안
호출을 놓쳐 구조적 한계가 있었다. 이 감지기는 받아쓰기를 하지 않고 음향 패턴만 본다.

체인(openWakeWord 규격, livekit-wakeword 호환):
    80ms 프레임(1280샘플)
      -> melspectrogram.onnx -> 멜 8프레임 -> [x/10+2] -> 멜 링버퍼(76)
      -> embedding_model.onnx(창 76, 보폭 8) -> 임베딩(96) -> 임베딩 링버퍼(16)
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
MEL_PER_FRAME = 8   # 1280샘플이 만드는 멜 프레임 수 = 보폭과 같음
EMB_WINDOW = 16     # 분류기가 보는 임베딩 개수
EMB_DIM = 96
INT16_SCALE = 32767.0


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
                 continuation_window: float = 0.5) -> None:
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

        self._init_state(threshold, trigger_frames, continuation_window, source)

    def _init_state(self, threshold, trigger_frames, continuation_window, source):
        """__init__ 과 테스트가 공유하는 순수 상태 초기화(ONNX 로드 없음)."""
        self.threshold = float(threshold)
        self.trigger_frames = int(trigger_frames)
        self.continuation_window = float(continuation_window)
        self.source = source
        self._mel: deque[np.ndarray] = deque(maxlen=MEL_WINDOW)
        self._emb: deque[np.ndarray] = deque(maxlen=EMB_WINDOW)
        self._hits = 0

    # ------------------------------------------------------------------ 체인
    def reset(self) -> None:
        """링버퍼를 비운다. 깨어난 뒤 다시 대기로 갈 때 호출."""
        self._mel.clear()
        self._emb.clear()
        self._hits = 0

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
                # 진단: 임계값 튜닝 근거. 대기 중 주기적으로 최고 점수를 남긴다.
                if n % 250 == 0:   # 250프레임 = 20초
                    log.info("[대기] 최근 20초 최고 점수 %.3f (임계 %.2f)", best, self.threshold)
                    best = 0.0
                continue

            self._hits += 1
            if self._hits < self.trigger_frames:
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
