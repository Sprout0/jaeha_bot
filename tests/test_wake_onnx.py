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
