"""OnnxWakeDetector 의 링버퍼·워밍업 로직. ONNX 세션은 가짜로 대체해 모델 없이 검증한다."""
import numpy as np
import pytest

from app.audio_source import FRAME
from app.wake_onnx import MEL_BANDS, MEL_PER_FRAME, OnnxWakeDetector, WakeResult

# 가짜 멜 세션의 출력 shape. 실제 모델과 같은 행 수를 내야 워밍업 계산이 일치한다
# (실측 2026-08-02: livekit-wakeword melspectrogram.onnx 는 push 당 5행).
MEL_OUT = (1, 1, MEL_PER_FRAME, MEL_BANDS)


class FakeSession:
    """onnxruntime InferenceSession 흉내. 정해진 shape 을 돌려준다."""

    def __init__(self, in_name, out_shape):
        self._in = in_name
        self._out_shape = out_shape
        self.calls = []

    def get_inputs(self):
        class _I:
            name = self._in
        return [_I()]

    def run(self, _out, feed):
        self.calls.append(feed[self._in])
        return [np.zeros(self._out_shape, dtype=np.float32)]


class ConstMelSession(FakeSession):
    """멜 세션 대역 — 항상 지정된 0이 아닌 상수를 채운 배열을 돌려준다.

    정규화(x/10+2)를 실제로 검증하려면 멜 출력이 0이면 안 된다(0/10+2=2 는
    정규화를 통째로 지워도 우연히 값이 갈리지 않는 경우가 있어 스케일 오류를
    못 잡는다). 30.0 을 쓰면 정규화 시 5.0, 정규화가 빠지면 30.0 으로
    확실히 갈린다.
    """

    def __init__(self, in_name, out_shape, value):
        super().__init__(in_name, out_shape)
        self._value = value

    def run(self, _out, feed):
        self.calls.append(feed[self._in])
        return [np.full(self._out_shape, self._value, dtype=np.float32)]


def make_detector(score=0.0, **kw):
    """ONNX 로드를 건너뛰고 가짜 세션을 꽂은 감지기.

    `score` 는 상수(int/float) 이거나, 호출(=push)마다 하나씩 순서대로 돌려줄
    점수 시퀀스(list 등 iterable)일 수 있다. 시퀀스가 소진되면 마지막 값을
    계속 돌려준다(연속 hit 시나리오를 이어가기 쉽도록). 상수 방식은 기존
    테스트들과 그대로 호환된다.
    """
    d = OnnxWakeDetector.__new__(OnnxWakeDetector)
    d._mel_sess = FakeSession("x", MEL_OUT)
    d._emb_sess = FakeSession("x", (1, 1, 1, 96))

    if isinstance(score, (int, float)):
        score_seq = None
        const_score = float(score)
    else:
        score_seq = list(score)
        const_score = None

    class ScoreSession(FakeSession):
        def __init__(self, in_name, out_shape):
            super().__init__(in_name, out_shape)
            self._idx = 0

        def run(self, _out, feed):
            self.calls.append(feed[self._in])
            if score_seq is not None:
                s = score_seq[min(self._idx, len(score_seq) - 1)]
                self._idx += 1
            else:
                s = const_score
            return [np.array([[s]], dtype=np.float32)]

    d._cls_sess = ScoreSession("embeddings", (1, 1))
    d._init_state(threshold=kw.get("threshold", 0.5),
                  trigger_frames=kw.get("trigger_frames", 2),
                  continuation_window=kw.get("continuation_window", 0.5),
                  source=kw.get("source"))
    return d


def _frame(v=0.1):
    return np.full(FRAME, v, dtype=np.float32)


def test_warmup_returns_none_until_buffers_fill():
    d = make_detector()
    # 멜 76프레임 채우는 데 ceil(76/MEL_PER_FRAME) 프레임, 그 뒤 임베딩 16개 채우는 데
    # 15 프레임 더 = WARMUP_FRAMES 프레임째에 첫 점수(실측 5행 기준 16+15=31).
    scores = [d.push(_frame()) for _ in range(OnnxWakeDetector.WARMUP_FRAMES - 1)]
    assert all(s is None for s in scores), "워밍업 중엔 점수를 내면 안 됨"


def test_first_score_after_warmup():
    d = make_detector(score=0.9)
    for _ in range(OnnxWakeDetector.WARMUP_FRAMES - 1):
        d.push(_frame())
    s = d.push(_frame())
    assert s is not None
    assert s == pytest.approx(0.9)


def test_classifier_receives_16_embeddings_of_96():
    d = make_detector(score=0.1)
    for _ in range(OnnxWakeDetector.WARMUP_FRAMES):
        d.push(_frame())
    fed = d._cls_sess.calls[-1]
    assert fed.shape == (1, 16, 96), f"분류기 입력 형상 오류: {fed.shape}"
    assert fed.dtype == np.float32


def test_melspectrogram_gets_int16_scaled_float():
    """[-1,1] 을 그대로 넣으면 조용히 실패한다. 32767 배율 확인."""
    d = make_detector()
    d.push(np.full(FRAME, 1.0, dtype=np.float32))
    fed = d._mel_sess.calls[0]
    assert fed.dtype == np.float32
    assert abs(float(np.max(np.abs(fed))) - 32767.0) < 1.0, \
        f"int16 범위로 스케일되지 않음: max={np.max(np.abs(fed))}"


def test_mel_normalization_applied():
    """멜 출력에 x/10+2 정규화가 실제로 적용되는지 확인(30.0 -> 5.0).

    FakeSession 은 항상 0을 돌려주므로 다른 테스트들은 이 정규화 자체를
    잡지 못한다(/10.0 + 2.0 을 지워도 통과함). 여기선 0이 아닌 알려진 값을
    돌려주는 세션을 꽂고 링버퍼 d._mel 에 저장된 값을 직접 읽어 확인한다.
    """
    d = OnnxWakeDetector.__new__(OnnxWakeDetector)
    d._mel_sess = ConstMelSession("x", MEL_OUT, 30.0)
    d._emb_sess = FakeSession("x", (1, 1, 1, 96))
    d._cls_sess = FakeSession("embeddings", (1, 1))
    d._init_state(threshold=0.5, trigger_frames=2, continuation_window=0.5, source=None)

    d.push(_frame())

    assert len(d._mel) == MEL_PER_FRAME, \
        f"멜 프레임 {MEL_PER_FRAME}개가 링버퍼에 쌓여야 함: {len(d._mel)}"
    for row in d._mel:
        assert row.shape == (MEL_BANDS,)
        assert np.allclose(row, 5.0), \
            f"멜 정규화(x/10+2) 값 불일치: {row[0]} (30.0 이 그대로면 정규화 누락)"


def test_reset_clears_buffers():
    d = make_detector(score=0.9)
    for _ in range(OnnxWakeDetector.WARMUP_FRAMES):
        d.push(_frame())
    d.reset()
    assert d.push(_frame()) is None, "reset 후엔 다시 워밍업해야 함"


def test_wake_result_fields():
    r = WakeResult(preroll=np.zeros(4, dtype=np.float32), continued=True, score=0.8)
    assert r.preroll.size == 4
    assert r.continued is True
    assert r.score == pytest.approx(0.8)


# ── wait_for_wake / 인사말 분기 ────────────────────────────────────────────

class ScriptedSource:
    """AudioSource 흉내 — 정해진 프레임을 돌려주고 프리롤을 흉내낸다."""

    def __init__(self, frames, preroll_frames=6):
        self._frames = list(frames)
        self._i = 0
        self._ring = []
        self.preroll_frames = preroll_frames
        self.noise_floor = 0.001

    def read(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        self._ring.append(f)
        self._ring = self._ring[-self.preroll_frames:]
        return f

    def preroll(self):
        if not self._ring:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._ring).astype(np.float32)

    def clear_preroll(self):
        self._ring = []


def test_sustained_high_score_eventually_wakes():
    """점수가 계속(상수로) 임계값 이상이면 결국 깨어난다.

    주의: 이 테스트는 "연속" hit 요구를 검증하지 않는다(원래 이름/문서가
    그렇게 주장했지만 실제로는 아니었다) — `ScoreSession` 이 매 프레임
    똑같은 상수 점수만 돌려주므로, 점수가 임계값 아래로 떨어졌다가 다시
    올라오는 시나리오 자체가 이 테스트엔 없다. 즉 `trigger_frames=1` 이어도,
    `trigger_frames` 를 통째로 무시해도, `_hits` 를 리셋하지 않아도 이
    테스트는 그대로 통과한다. "연속 hit 요구"의 실제 검증은
    `test_non_consecutive_hits_do_not_wake_but_consecutive_hits_do` 가 맡는다.
    """
    src = ScriptedSource([_frame()] * 200)
    d = make_detector(score=0.9, trigger_frames=2, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None, "점수가 계속 임계값 이상이면 깨어나야 함"


def test_non_consecutive_hits_do_not_wake_but_consecutive_hits_do():
    """hit, miss, hit(연속 아님) 로는 안 깨고, 그 뒤 연속 hit 2회에서 깨어난다.

    `trigger_frames=2` 인데 hit 사이에 miss 가 끼면 `_hits` 카운터가
    리셋돼야 한다(`app/wake_onnx.py` 의 `self._hits = 0` 분기). 이 테스트는
    점수 시퀀스를 워밍업(25프레임) 직후부터 [hit, miss, hit, hit] 순서로
    고정해, 처음 세 프레임(hit-miss-hit)만으로는 절대 깨지 않는다는 것과
    그 다음 hit 이 와서 비로소 연속 2회가 됐을 때 깨어난다는 것을 함께
    검증한다.
    """
    src = ScriptedSource([_frame()] * 200)
    scores = [0.9, 0.1, 0.9, 0.9]   # 워밍업 이후: hit, miss(리셋), hit, hit(트리거)
    d = make_detector(score=scores, trigger_frames=2, threshold=0.5, source=src)

    # 워밍업 + hit,miss,hit(3) 직전까지만 허용. 이 구간엔 연속 2회가
    # 없으므로(hit-miss-hit) 절대 깨면 안 된다.
    r_early = d.wait_for_wake(max_frames=OnnxWakeDetector.WARMUP_FRAMES + 2)
    assert r_early is None, "hit-miss-hit 은 연속이 아니므로 깨면 안 됨"

    # 이어서 읽으면 다음 점수는 hit 이고, 방금 hit 뒤라 연속 2회가 되어 깨어나야 함
    r = d.wait_for_wake(max_frames=10)
    assert r is not None, "miss 이후라도 hit 이 연속 2회가 되면 깨어나야 함"


def test_below_threshold_never_wakes():
    src = ScriptedSource([_frame()] * 200)
    d = make_detector(score=0.1, threshold=0.5, source=src)
    assert d.wait_for_wake(max_frames=150) is None


def test_continued_true_when_speech_follows():
    """감지 직후 큰 소리가 이어지면 continued=True 이고 오디오가 실려온다."""
    loud = np.full(FRAME, 0.5, dtype=np.float32)
    src = ScriptedSource([loud] * 200)
    d = make_detector(score=0.9, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None
    assert r.continued is True
    assert r.preroll.size > 0, "한 숨 패턴이면 오디오를 넘겨야 함"


def test_continued_false_when_silence_follows():
    """부르고 멈추면 continued=False 이고 프리롤은 버린다(인사말 경로)."""
    # 참고: 이 테스트의 가짜 분류기는 프레임 내용과 무관하게 항상 score=0.9 를
    # 반환하므로, 워밍업(25프레임) + trigger_frames(2) 로 깨움은 항상 26번째
    # read() 에서 확정되고, 이어서 continuation_window(기본 0.5초=6프레임)를
    # 더 읽는다. "부르는 소리"가 그 32프레임을 넘어서까지 이어지면 이어짐
    # 관찰 구간이 여전히 큰 소리를 보게 되어 의도한 시나리오(부르고 멈춤)를
    # 검증하지 못한다. 그래서 큰 소리 구간을 트리거 지점(26프레임)보다
    # 확실히 짧게(15프레임) 잡아, 이어짐 관찰 구간이 조용한 구간에 들어가게 한다.
    loud = np.full(FRAME, 0.5, dtype=np.float32)
    quiet = np.zeros(FRAME, dtype=np.float32)
    src = ScriptedSource([loud] * 15 + [quiet] * 185)
    d = make_detector(score=0.9, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None
    assert r.continued is False
    assert r.preroll.size == 0, "인사말 경로에선 프리롤을 버려야 함"
