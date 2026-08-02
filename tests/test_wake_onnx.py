"""OnnxWakeDetector 의 링버퍼·워밍업 로직. ONNX 세션은 가짜로 대체해 모델 없이 검증한다."""
import numpy as np
import pytest

from app.audio_source import FRAME
from app.wake_onnx import OnnxWakeDetector, WakeResult


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
    """ONNX 로드를 건너뛰고 가짜 세션을 꽂은 감지기."""
    d = OnnxWakeDetector.__new__(OnnxWakeDetector)
    d._mel_sess = FakeSession("x", (1, 1, 8, 32))
    d._emb_sess = FakeSession("x", (1, 1, 1, 96))

    class ScoreSession(FakeSession):
        def run(self, _out, feed):
            self.calls.append(feed[self._in])
            return [np.array([[score]], dtype=np.float32)]

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
    # 멜 76프레임 채우는 데 76/8 = 9.5 -> 10 프레임,
    # 그 뒤 임베딩 16개 채우는 데 15 프레임 더 = 총 25 프레임째에 첫 점수.
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
    d._mel_sess = ConstMelSession("x", (1, 1, 8, 32), 30.0)
    d._emb_sess = FakeSession("x", (1, 1, 1, 96))
    d._cls_sess = FakeSession("embeddings", (1, 1))
    d._init_state(threshold=0.5, trigger_frames=2, continuation_window=0.5, source=None)

    d.push(_frame())

    assert len(d._mel) == 8, f"멜 프레임 8개가 링버퍼에 쌓여야 함: {len(d._mel)}"
    for row in d._mel:
        assert row.shape == (32,)
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


def test_requires_trigger_frames_consecutive_hits():
    """단발 점수 튐으로는 안 깨어난다."""
    src = ScriptedSource([_frame()] * 200)
    d = make_detector(score=0.9, trigger_frames=2, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None, "연속 2프레임이면 깨어나야 함"


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
