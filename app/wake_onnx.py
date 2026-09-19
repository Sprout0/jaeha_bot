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
from dataclasses import dataclass, field

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
    # 호출어 **뒤** 소리만(프리롤 제외). 전면 API 는 이것만 보낸다 — 프리롤째 보내면
    # 서버가 호출어를 아이 말로 받아 적는다(09-19 실기: '하이즈들' → "안녕! 잘 지냈어?").
    tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class OnnxWakeDetector:
    # 멜 76프레임 채우기 = ceil(76/8) = 10 프레임,
    # 그 뒤 임베딩 16개 채우기 = 15 프레임 더. 총 25번째 push 에서 첫 점수.
    WARMUP_FRAMES = -(-MEL_WINDOW // MEL_PER_FRAME) + EMB_WINDOW - 1

    def __init__(self, model_dir, classifier: str = "jaehabot.onnx",
                 threshold: float = 0.5, trigger_frames: int = 2,
                 providers=None, source=None,
                 continuation_window: float = 0.5,
                 verifier=None, verify_cooldown_s: float = 1.0,
                 verify_min_rms: float = 0.005,
                 verify_rearm_delta: float = 0.05,
                 verify_bypass: float = 1.01,
                 verify_settle_s: float = 0.0,
                 embed_rescue=None,
                 verify_embed_settle_s: float | None = None,
                 continuation_min_rms: float | None = None) -> None:
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
                         verifier, verify_cooldown_s, verify_min_rms,
                         verify_rearm_delta, verify_bypass, verify_settle_s,
                         embed_rescue, verify_embed_settle_s, continuation_min_rms)

    def _init_state(self, threshold, trigger_frames, continuation_window, source,
                    verifier=None, verify_cooldown_s: float = 1.0,
                    verify_min_rms: float = 0.005,
                    verify_rearm_delta: float = 0.05,
                    verify_bypass: float = 1.01,
                    verify_settle_s: float = 0.0,
                    embed_rescue=None, verify_embed_settle_s=None,
                    continuation_min_rms=None):
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
        self.verify_min_rms = float(verify_min_rms)   # 에너지 게이트 하한(무음만 거른다)
        # 뒷말 판정 기준. None 이면 위 게이트 값을 같이 쓴다(옛 동작).
        # 🔴 2026-09-19 게이트 값(0.005)은 '디지털 무음' 기준이라 방 소음(젯슨 ~0.007)보다
        #    낮다 — 뒷말 판정에 쓰면 호출어만 말해도 늘 '이어짐'이 된다. 따로 둔다.
        #    소음 바닥에 비례시키지 않는 원칙(아래 _observe_continuation)은 그대로다.
        self.continuation_min_rms = (None if continuation_min_rms is None
                                     else float(continuation_min_rms))
        # 직전 검증보다 이만큼 높은 점수는 '새 사건'으로 보고 재무장한다.
        # 소음이 계속돼 점수가 임계 아래로 안 내려가는 상황의 유일한 탈출구다.
        self.verify_rearm_delta = float(verify_rearm_delta)
        # 🟢 2026-08-26 우회컷. 이 점수를 넘으면 whisper 를 부르지 않고 바로 깨어난다.
        #    근거: 실제 거실 녹음(유튜브 켠 3분)의 **최고 점수가 0.173** 이었다. 그보다
        #    확실히 위인 점수는 소음이 만들 수 있는 값이 아니다. 반면 진짜 호출은 실음성
        #    실측에서 0.39·0.66·0.87 까지 나온다 — 그런 걸 whisper 에게 물어 봐야
        #    손해만 본다(0.873 짜리가 '하이치 루' 로 읽혀 기각된 실례가 있다).
        #    1.01 = 사실상 끔(점수는 1.0 을 못 넘는다). 설정에서 켠다.
        self.verify_bypass = float(verify_bypass)
        # 🔴 2026-08-27 후보가 뜬 **그 순간**의 창은 호출어를 자르고 있을 수 있다.
        #    1단계는 '하이' 까지만 듣고도 임계(0.05)를 넘기고, 그러면 검증창의 끝이
        #    '티'·'드' 앞이라 whisper 가 그대로 '하이' 라고 받아쓴다 — 자모거리 0.50 이라
        #    기각된다. 실기 로그 11:51:53 이 그 모습이다(점수 0.272 인데 전사가 '하이').
        #    이만큼 더 듣고 나서 창을 뜬다. 검증창은 '최근 N초' 라 읽는 만큼 뒤로 따라온다.
        #    ⚠️ 이 시간은 그대로 깨움 지연에 더해진다 — 짧게 잡을 것.
        self.verify_settle_s = float(verify_settle_s)
        # 🔴 2026-09-11 임베딩 대조만 따로 더 듣는다(None = 위와 같음). 1단계 최고점은
        #    첫 임계 넘김보다 0.3~0.5초 뒤인데 본보기는 최고점 자리로 만들었다 — 0.24초만
        #    듣고 창을 뜨면 호출어 끝이 잘린다(젯슨 A/B, 3초 창: +0.24초 6/11 -> +0.48초 10/11).
        #    임베딩은 0.08초에 끝나서 더 기다려도 whisper(0.24+0.78초)보다 먼저 깨운다.
        self.verify_embed_settle_s = (None if verify_embed_settle_s is None
                                      else float(verify_embed_settle_s))
        # 🔴 whisper 가 지어낸 글 때문에 죽은 진짜 호출을 소리로 건지는 장치(없으면 None).
        #    app/wake_embed.py 참고. **whisper 를 대체하지 않고 OR 로 붙는다.**
        self.embed_rescue = embed_rescue
        self._armed = True       # 히스테리시스: 기각 후에는 점수가 임계 아래로 내려가야 재무장
        self._last_verify = 0.0  # 쿨다운 기준 시각
        self._last_score = 0.0   # 직전 검증 때의 점수(재무장 판단용)

    # --------------------------------------------------------- 임베딩 대조용 체인
    def embed_sequence(self, audio) -> np.ndarray:
        """오디오 한 덩어리를 임베딩 열(N, 96)로. 분류기는 안 탄다.

        🔴 **살아 있는 링버퍼를 건드리면 안 된다.** self._mel/_emb 는 지금 흐르는
           소리를 담고 있고, 검증은 그 흐름 도중에 끼어든다. 여기서 그걸 쓰면
           검증 한 번이 감지 상태를 통째로 날려 다음 호출을 놓친다.
           그래서 지역 deque 를 새로 만든다.
        """
        mel: deque[np.ndarray] = deque(maxlen=MEL_WINDOW)
        out = []
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        mel_name = self._mel_sess.get_inputs()[0].name
        emb_name = self._emb_sess.get_inputs()[0].name
        for i in range(0, a.size - FRAME + 1, FRAME):
            chunk = a[i:i + FRAME].reshape(1, -1) * INT16_SCALE
            rows = np.squeeze(self._mel_sess.run(None, {mel_name: chunk})[0])
            for row in rows.reshape(-1, MEL_BANDS) / 10.0 + 2.0:
                mel.append(row.astype(np.float32))
            if len(mel) < MEL_WINDOW:
                continue
            window = np.stack(list(mel))[None, :, :, None].astype(np.float32)
            e = np.squeeze(self._emb_sess.run(None, {emb_name: window})[0])
            out.append(e.reshape(EMB_DIM).astype(np.float32))
        return (np.stack(out) if out
                else np.zeros((0, EMB_DIM), dtype=np.float32))

    # ------------------------------------------------------------------ 체인
    def reset(self) -> None:
        """링버퍼를 비운다. 깨어난 뒤 다시 대기로 갈 때 호출."""
        self._mel.clear()
        self._emb.clear()
        self._hits = 0
        self._armed = True
        self._last_score = 0.0

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
            # 검증 장치가 **둘 중 하나라도** 있으면 태운다. verifier 만 보면 임베딩
            # 단독 구성(S2S)에서 검증이 통째로 건너뛰어져 1단계 단독이 된다.
            if (self.verifier is not None or self.embed_rescue is not None) \
                    and not self._verify(score):
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
                              continued=continued, score=score,
                              tail=(tail if continued else np.zeros(0)).astype(np.float32))
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

        # ── 0) 우회 — 1단계가 확신하면 whisper 를 안 부른다 ──────────────
        # 🔴 여기가 **검증 전에** 와야 한다. 쿨다운·히스테리시스 아래에 두면 확신하는
        #    호출이 '방금 기각했다'는 이유로 같이 막힌다.
        # 근거(2026-08-26 실측): 실제 거실 3분 최고 점수 0.173 vs 진짜 호출 0.39~0.87.
        #    캐스케이드 재현율이 실음성 20건에서 11/20 -> 14/20 으로 오른다.
        # ⚠️ 3분 표본의 최고치다. 더 길게 재면 더 높은 순간이 나온다 — 우회컷은
        #    그 측정 뒤에 확정할 값이다. 낮출수록 whisper 를 건너뛴 헛깨움이 는다.
        if score >= self.verify_bypass:
            log.info("[검증] 건너뜀 — 점수 %.3f >= 우회컷 %.2f (방 소음 실측 최고 0.173)",
                     score, self.verify_bypass)
            self._hits = 0
            return True

        # 🔴 재무장 조건이 둘이다. '임계 아래로 내려갔다 오기'만으로는 소음 속에서 영영
        #    재무장이 안 된다 — 유튜브가 계속 나오면 점수가 임계(0.05) 아래로 안 떨어져
        #    첫 기각 이후로 귀를 닫는다. 실기 로그(2026-08-12 14:11)가 그 모습이다:
        #      14:11:14 [검증] 후보 기각 — 점수 0.060
        #      14:11:29 [대기] 최근 20초 최고 점수 **0.226**   <- 검증조차 안 했다
        #    0.226 은 진짜 호출인데(성공한 호출이 0.186) 통째로 흘려보냈다.
        #    → **점수가 직전 검증 때보다 뚜렷이 높으면 그건 새 사건**이다. 그때도 재무장한다.
        if not self._armed and score < self._last_score + self.verify_rearm_delta:
            return False
        now = time.monotonic()
        if now - self._last_verify < self.verify_cooldown_s:
            return False

        self._armed = False          # 판정 전에 내린다 — 예외가 나도 폭주하지 않게
        self._last_verify = now
        self._last_score = score
        # 호출어 끝을 창 안에 넣는다(위 verify_settle_s 주석 참고). 읽는 프레임은
        # 버려지지 않는다 — 프리롤·검증창 링버퍼로 그대로 들어간다.
        # 임베딩 단독이면 임베딩 쪽 기다림을 쓴다(whisper 쪽은 그대로).
        settle = self._embed_settle() if self.verifier is None else self.verify_settle_s
        self._read_ahead(self._n_frames(settle))
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

        # 🔴 2026-09-02 whisper 없이도 돌 수 있어야 한다. 전면 API(S2S) 로 가면 로컬
        #    STT 를 안 올리는데, 그때 verifier 가 None 이 되어 1단계 단독으로 떨어지면
        #    실제 거실에서 시간당 160회 깨어난다(이미 기각된 길이다).
        #    ➡️ verifier 가 없으면 임베딩 대조가 **단독 관문**이 된다.
        #    비용은 whisper 1.22초가 아니라 수 밀리초다(모델은 1단계가 이미 올려 뒀다).
        if self.verifier is None:
            if self.embed_rescue is None:
                return True          # 검증 장치가 아예 없다 = 옛 동작(1단계 단독)
            try:
                ok, sim = self.embed_rescue.passes(
                    self.embed_sequence(self._embed_audio(audio)))
            except Exception as e:      # noqa: BLE001 — 검증 장치가 봇을 죽이면 안 된다
                log.warning("[검증] 임베딩 단독 실패(기각 처리): %s: %s",
                            type(e).__name__, e)
                return False
            finally:
                self._last_verify = time.monotonic()
            log.info("[검증] 임베딩 단독 %s — 유사도 %.3f %s %.2f (점수 %.3f)",
                     "통과" if ok else "기각", sim, ">=" if ok else "<",
                     self.embed_rescue.min_similarity, score)
            if ok:
                self._hits = 0
            return ok

        try:
            ok = bool(self.verifier(audio))
        except Exception as e:
            # whisper 가 죽어도 봇은 계속 들어야 한다. 안전하게 '기각'으로 본다.
            log.warning("[검증] 실패(기각 처리): %s: %s", type(e).__name__, e)
            return False
        finally:
            # 🔴🔴 2026-08-27 쿨다운을 **끝난 시각**으로 다시 잡는다.
            #    시작 시각으로만 재면 whisper 가 도는 1.2~2.0초가 쿨다운을 통째로 먹어
            #    **쿨다운이 사실상 0** 이 된다. 실기 로그가 그 모습이다:
            #      11:13:26.841 검증 끝 → 11:13:26.939 다음 검증 (0.10초 뒤)
            #    같은 발화 하나를 3~5번 검증하고, 그동안 읽기 루프가 멈춰 마이크 버퍼가
            #    넘친다(그 세션 넘침 29회). 넘친 오디오는 다음 창을 망가뜨려 전사가
            #    빈 문자열로 나오고, 그게 또 후보를 만든다 — **스스로를 먹이는 고리**다.
            #    ⚠️ 예외가 나도 갱신해야 한다(그래서 finally). 안 그러면 검증기가
            #       고장난 동안 폭주가 더 심해진다.
            self._last_verify = time.monotonic()
        if not ok and self.embed_rescue is not None:
            # 🔴 whisper 가 **소리와 무관한 글을 지어내서** 죽은 진짜 호출을 여기서 건진다.
            #    증거: data/wake_real/.../빠르게_00.wav 는 어른이 '하이 티드'라고 부른
            #    녹음인데 whisper 가 '안녕히계세요'로 적는다. 글자로는 손쓸 방법이 없다.
            #    자세한 근거·한계는 app/wake_embed.py 머리말에 있다.
            try:
                # 임베딩은 whisper 보다 더 들어야 한다(verify_embed_settle_s). whisper 가
                # 도는 동안 쌓인 소리라 읽는 데 시간이 거의 안 든다.
                self._read_ahead(self._n_frames(self._embed_settle())
                                 - self._n_frames(self.verify_settle_s))
                saved, sim = self.embed_rescue.passes(
                    self.embed_sequence(self._embed_audio(audio)))
            except Exception as e:      # noqa: BLE001 — 구제 장치가 봇을 죽이면 안 된다
                log.warning("[검증] 임베딩 대조 실패(무시): %s: %s", type(e).__name__, e)
            else:
                if saved:
                    log.info("[검증] 임베딩으로 건짐 — 유사도 %.3f >= %.2f "
                             "(whisper 는 기각했다, 점수 %.3f)",
                             sim, self.embed_rescue.min_similarity, score)
                    self._hits = 0
                    return True
                log.info("[검증] 임베딩도 기각 — 유사도 %.3f < %.2f",
                         sim, self.embed_rescue.min_similarity)
        if not ok:
            # 실기 튜닝의 근거가 되는 줄이다. 실제 가정 소음에서 무엇이 후보로 뜨는지
            # 이 로그로만 알 수 있다(합성 부정으로는 확인이 안 된다).
            # 🔴 2026-08-27 RMS 를 같이 남긴다. 여태 '점수'와 '길이'만 남겨서, 기각된
            #    후보에 **사람 목소리가 있었는지조차** 알 수가 없었다. 조용한 호출과
            #    시끄러운 TV 는 처방이 정반대인데 로그가 둘을 구분해 주지 않았다.
            log.info("[검증] 후보 기각 — 점수 %.3f, 오디오 %.2fs, 최대 RMS %.4f "
                     "(소음바닥 %.4f)", score, audio.size / SAMPLE_RATE, loudest,
                     float(getattr(self.source, "noise_floor", 0.0) or 0.0))
        self._hits = 0
        return ok

    # ------------------------------------------------------ 검증창 도우미
    @staticmethod
    def _n_frames(seconds: float) -> int:
        return int(seconds * SAMPLE_RATE / FRAME)

    def _embed_settle(self) -> float:
        s = self.verify_embed_settle_s
        return self.verify_settle_s if s is None else s

    def _read_ahead(self, n: int) -> None:
        """호출어 끝을 창 안에 넣으려고 n 프레임 더 읽는다. 읽은 프레임은 버려지지 않는다 —
        프리롤·검증창·임베딩창 링버퍼로 그대로 들어간다(verify_settle_s 주석 참고)."""
        for _ in range(max(0, n)):
            try:
                self.source.read()
            except StopIteration:
                break

    def _embed_audio(self, fallback: np.ndarray) -> np.ndarray:
        """임베딩 대조에 쓸 소리 — 소스에 임베딩 창이 있으면 그것, 없으면 검증창."""
        grab = getattr(self.source, "embed_window", None)
        return grab() if grab is not None else fallback

    # 검증 동안 밀린 오디오를 거두되 상한을 둔다. 무한정 거두면 몇 초 전 TV 소리까지
    # 딸려와 whisper 가 그걸 받아쓴다(2026-08-12 에 프리롤을 0.5초로 묶은 것과 같은 이유).
    _MAX_BACKLOG_S = 2.0

    def _observe_continuation(self) -> tuple[np.ndarray, bool]:
        """감지 직후 잠깐 들어 말이 이어지는지 본다. (읽은 오디오, 이어짐 여부)."""
        n = max(1, int(self.continuation_window * SAMPLE_RATE / FRAME))
        frames = []
        # ① 🔴 2026-08-27 **밀린 것부터 거둔다.** 2단계 검증(whisper)이 도는 1.2~1.5초
        #    동안 읽기 루프가 멈춰 있고, 아이가 뒷말을 하는 게 정확히 그 구간이다.
        #    그 오디오는 이미 버퍼에 있는데 여태 0.5초만 읽고 나머지를 버렸다 —
        #    실기(11:41)에서 STT 에 1.12초만 넘어갔고 전사가 비어 봇이 침묵했다.
        #    이미 들어와 있는 것이라 **읽는 시간이 0** 이다(인사말 경로도 안 느려진다).
        grab = getattr(self.source, "read_buffered", None)
        if grab is not None:
            frames.extend(grab(int(self._MAX_BACKLOG_S * SAMPLE_RATE / FRAME)))
        # ② 그 다음 실시간으로 조금 더 듣는다 — 아직 말하는 중일 수 있다.
        #    🔴 2026-08-27 **밀린 것이 이미 창을 채웠으면 더 안 듣는다.** 여태는 무조건
        #    0.5초를 더 기다렸는데, 그 앞에서 whisper 가 1.26초 도는 동안 그만큼이 이미
        #    버퍼에 들어와 있다. 즉 있는 소리를 두고 같은 길이를 한 번 더 기다린 셈이고,
        #    그게 그대로 깨움 지연이었다(실측 11:52:47.709 -> 48.124 = 0.42초).
        #    판정에 쓸 오디오는 똑같고 기다림만 사라진다 — 대가가 없다.
        for _ in range(max(0, n - len(frames))):
            try:
                frames.append(self.source.read())
            except StopIteration:
                break
        if not frames:
            return np.zeros(0, dtype=np.float32), False
        tail = np.concatenate(frames).astype(np.float32)
        # 🔴🔴 2026-08-27 **소음 바닥에 비례시키지 않는다.** 원래 `max(floor*2, 0.005)`
        #    였는데, 이건 08-12 에 에너지 게이트에서 똑같이 잡아낸 함정이다(커밋 038c620).
        #    거기만 고치고 **여기는 같은 코드가 그대로 남아 있었다.** 아무도 못 본 이유는
        #    이 판정이 로그를 한 줄도 안 남겼기 때문이다 — 실패해도 흔적이 없다.
        #    실측: 소음바닥 0.0081 -> 기준 0.0162 로 고정값의 3.2배. 유튜브를 틀면 더 오른다.
        #    즉 **시끄러울수록 "이어 말했다"를 못 알아본다** — 인사말을 하고, 그 사이
        #    '이거 뭐야?'가 통째로 날아간다. 정확히 게이트 때 겪은 그 고장이다.
        thr = (self.verify_min_rms if self.continuation_min_rms is None
               else self.continuation_min_rms)
        loudest = max(float(np.sqrt(np.mean(np.square(f)))) for f in frames)
        continued = loudest >= thr
        # 🔴 판정을 **항상** 남긴다. 이 줄이 없어서 '호출 직후 뒷말'이 언제부터 안 되는지
        #    아무도 몰랐다. 숫자가 없으면 다음에 또 추측만 하게 된다.
        log.info("[호출] 뒷말 %s — 최대 RMS %.4f %s 기준 %.4f (%.2fs 들음)",
                 "이어짐" if continued else "없음", loudest,
                 ">=" if continued else "<", thr, tail.size / SAMPLE_RATE)
        return tail, continued
