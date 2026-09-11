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


# ── 검증 방식 스위치 (2026-09-02) ────────────────────────────────────────
# 🔴 왜: 전면 API(S2S) 로 가면 로컬 STT 가 없다. 그때 whisper 검증기를 아예 안
#    만들도록 설정으로 고를 수 있어야 한다. 기본은 반드시 현행(whisper)이다.

def _capture_kwargs(monkeypatch):
    """OnnxWakeDetector 에 실제로 넘어간 인자를 잡아 둔다."""
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)

    import app.wake_onnx as wake_onnx
    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", Spy)
    return seen


def test_mode가_없으면_현행대로_whisper를_쓴다(monkeypatch):
    # 🔴 기본값이 바뀌면 아무도 모르게 동작이 달라진다. 여기서 못 박는다.
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True}}}
    make_detector(cfg, stt=FakeStt([]), source=object())
    assert seen["verifier"] is not None


def test_mode가_embed면_whisper를_안_만든다(monkeypatch):
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed"}}}
    make_detector(cfg, stt=FakeStt([]), source=object())
    assert seen["verifier"] is None


def test_mode가_embed면_stt가_없어도_만들어진다(monkeypatch):
    # S2S 구성에서는 stt 인스턴스 자체가 없다. 그때 죽으면 안 된다.
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed"}}}
    make_detector(cfg, stt=None, source=object())
    assert seen["verifier"] is None


def test_모르는_mode는_죽는다(monkeypatch):
    # 오타가 조용히 '현행'으로 떨어지면 켠 줄 알고 안 켜진다.
    _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embedding"}}}
    with pytest.raises(ValueError):
        make_detector(cfg, stt=FakeStt([]), source=object())


# ── stt 없이 부팅 (2026-09-02) ───────────────────────────────────────────

def test_stt가_없는데_onnx가_죽으면_조용히_넘어가지_않는다(monkeypatch):
    # 🔴 조용한 폴백이 제일 나쁘다. SttWakeDetector 는 stt 로 듣는데 stt 가 None 이면
    #    부를 수 없는 객체다. 그걸 돌려주면 봇이 영영 안 깨어나면서 로그엔 '폴백함'
    #    한 줄만 남아 원인을 못 찾는다.
    import app.wake_onnx as wake_onnx

    def _boom(**kw):
        raise RuntimeError("모델 없음")

    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", _boom)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed"}}}
    with pytest.raises(RuntimeError):
        make_detector(cfg, stt=None, source=object())


def test_stt가_있으면_예전처럼_폴백한다(monkeypatch):
    import app.wake_onnx as wake_onnx

    def _boom(**kw):
        raise RuntimeError("모델 없음")

    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", _boom)
    cfg = {"detector": "onnx", "word": "하이티드", "onnx": {"verify": {"enabled": True}}}
    d = make_detector(cfg, stt=FakeStt([]), source=object())
    assert isinstance(d, SttWakeDetector)


def test_detector가_stt인데_stt가_없으면_죽는다():
    with pytest.raises(RuntimeError):
        make_detector({"detector": "stt", "word": "하이티드"}, stt=None, source=object())


# ── 임베딩 쪽 기다림 (2026-09-11) ────────────────────────────────────────
def test_embed_settle_s_가_감지기로_넘어간다(monkeypatch):
    # 임베딩은 후보 뒤 0.48초를 더 들어야 호출어 끝이 창에 들어온다(젯슨 A/B).
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed", "embed_settle_s": 0.5}}}
    make_detector(cfg, stt=None, source=object())
    assert seen["verify_embed_settle_s"] == 0.5


def test_embed_settle_s_가_없으면_None_이라_whisper_기다림을_따른다(monkeypatch):
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True}}}
    make_detector(cfg, stt=FakeStt([]), source=object())
    assert seen["verify_embed_settle_s"] is None
