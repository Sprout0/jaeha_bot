"""cpu_threads 가 실제로 ctranslate2 까지 전달되는지 — 모델 파일·젯슨 불필요.

안 넘어가도 예외가 없고 인식 결과도 같다. 그냥 조용히 느려질 뿐이라(기본 4스레드)
설정만 고쳐두고 배선을 빠뜨리면 아무도 못 알아챈다.
"""
import sys
import types

import pytest

from app.stt_module import STTModule


@pytest.fixture
def captured_kwargs(monkeypatch):
    """faster_whisper.WhisperModel 을 가로채 생성 인자를 잡아둔다."""
    seen = {}

    class FakeWhisperModel:
        def __init__(self, model_size, **kw):
            seen["model_size"] = model_size
            seen.update(kw)

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return seen


def test_cpu_threads_reaches_whisper_model(captured_kwargs):
    STTModule(model_size="tiny", cpu_threads=6).load()

    assert captured_kwargs.get("cpu_threads") == 6, \
        f"cpu_threads 가 WhisperModel 까지 전달돼야 함: {captured_kwargs}"


def test_cpu_threads_omitted_when_not_set(captured_kwargs):
    # 미지정이면 라이브러리 기본값이 쓰이도록 인자를 아예 넘기지 않는다
    # (0 이나 None 을 넘기면 버전에 따라 해석이 달라진다).
    STTModule(model_size="tiny").load()

    assert "cpu_threads" not in captured_kwargs, \
        f"미지정이면 인자를 넘기지 말아야 함: {captured_kwargs}"


def test_device_and_compute_type_still_passed(captured_kwargs):
    STTModule(model_size="tiny", device="cpu", compute_type="int8", cpu_threads=6).load()

    assert captured_kwargs["device"] == "cpu"
    assert captured_kwargs["compute_type"] == "int8"
