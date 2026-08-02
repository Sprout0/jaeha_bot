"""감지기 선택과 폴백. ONNX 파일이 없어도 봇이 죽지 않아야 한다."""
import logging

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


def test_stt_detector_raises_when_stt_missing():
    """OnnxWakeDetector(source=None) 이 RuntimeError 를 내는 것과 같은 방식으로 실패해야
    호출부(main.py)가 감지기 종류를 몰라도 하나의 예외 형태만 다루면 된다."""
    d = SttWakeDetector(None, word="재하봇", threshold=0.68, aliases=[])
    with pytest.raises(RuntimeError):
        d.wait_for_wake(max_turns=1)


def test_factory_selects_stt_when_configured():
    d = make_detector({"detector": "stt", "word": "재하봇", "threshold": 0.68},
                      stt=FakeStt([]), source=None)
    assert isinstance(d, SttWakeDetector)


def test_factory_falls_back_to_stt_when_onnx_missing(caplog):
    """melspectrogram.onnx 부터 없어서(가장 먼저 로드) FileNotFoundError 가 나고
    (classifier="nope.onnx" 는 애초에 melspectrogram 단계에서 걸리므로 검사되지 않는다),
    예외 없이 STT 감지기로 내려와야 한다."""
    cfg = {"detector": "onnx", "word": "재하봇", "threshold": 0.68,
           "onnx": {"model_dir": "models/does-not-exist",
                    "classifier": "nope.onnx"}}
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.wake"):
        d = make_detector(cfg, stt=FakeStt([]), source=object())
    assert isinstance(d, SttWakeDetector), "폴백하지 않았다"

    assert len(caplog.records) == 1, "폴백 로그가 정확히 한 번 남아야 한다"
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "FileNotFoundError" in record.message, (
        "예외 타입이 로그에 없으면 '모델이 안 됐나?' 와 '코드 버그인가?' 를 구분할 수 없다"
    )
    assert record.exc_info is not None, "트레이스백이 없으면 현장에서 원인을 못 찾는다"


def test_factory_fallback_log_reports_actual_exception_type(monkeypatch, caplog):
    """폴백은 FileNotFoundError 뿐 아니라 코드 버그(TypeError 등)에서도 일어난다.
    이때도 로그에 실제 예외 타입이 남아야 '모델 미착륙'과 '우리 버그'를 구분할 수 있다."""
    import app.wake_onnx as wake_onnx

    def _boom(*args, **kwargs):
        raise TypeError("kwarg 오타 같은 코드 버그를 흉내")

    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", _boom)

    cfg = {"detector": "onnx", "word": "재하봇", "threshold": 0.68, "onnx": {}}
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.wake"):
        d = make_detector(cfg, stt=FakeStt([]), source=object())

    assert isinstance(d, SttWakeDetector), "코드 버그여도 봇이 죽으면 안 된다"
    assert any("TypeError" in r.message for r in caplog.records)
