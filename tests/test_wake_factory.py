"""감지기 선택과 폴백. ONNX 파일이 없어도 봇이 죽지 않아야 한다."""
import numpy as np
import pytest

from app.wake import SttWakeDetector, make_detector
from app.wake_onnx import WakeResult


class FakeStt:
    def __init__(self, texts):
        self._texts = list(texts)
        self._i = 0

    def listen(self, **kw):
        if self._i >= len(self._texts):
            return "", 0.0
        t = self._texts[self._i]
        self._i += 1
        return t, 0.1


def test_stt_detector_wakes_on_wake_word():
    d = SttWakeDetector(FakeStt(["엄마 어디 있어", "재하봇"]), word="재하봇",
                        threshold=0.68, aliases=[])
    r = d.wait_for_wake(max_turns=5)
    assert isinstance(r, WakeResult)
    assert r.continued is False
    assert r.preroll.size == 0, "STT 감지기는 오디오를 넘기지 않는다"


def test_stt_detector_returns_none_when_never_called():
    d = SttWakeDetector(FakeStt(["엄마 어디 있어", "사과 먹고 싶어"]), word="재하봇",
                        threshold=0.68, aliases=[])
    assert d.wait_for_wake(max_turns=2) is None


def test_factory_selects_stt_when_configured():
    d = make_detector({"detector": "stt", "word": "재하봇", "threshold": 0.68},
                      stt=FakeStt([]), source=None)
    assert isinstance(d, SttWakeDetector)


def test_factory_falls_back_to_stt_when_onnx_missing(caplog):
    """분류기 파일이 없어도 예외 없이 STT 감지기로 내려와야 한다."""
    cfg = {"detector": "onnx", "word": "재하봇", "threshold": 0.68,
           "onnx": {"model_dir": "models/does-not-exist",
                    "classifier": "nope.onnx"}}
    d = make_detector(cfg, stt=FakeStt([]), source=object())
    assert isinstance(d, SttWakeDetector), "폴백하지 않았다"
