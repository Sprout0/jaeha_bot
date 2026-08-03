"""TTS 실행 프로바이더(GPU 가속) 배선 검증. 모델 로드·GPU 불필요.

여기서 지키려는 건 딱 하나 — supertonic 은 providers 인자를 안 받아서
DEFAULT_ONNX_PROVIDERS 를 덮어써야 하는데, loader.py 가 `from .config import ...` 로
이름을 바인딩해두기 때문에 **config 쪽을 고치면 조용히 무시된다.**
틀려도 예외가 없고 CPU 로 잘 돌아가서, GPU 가 안 붙은 걸 알아채기 어렵다.
"""
import sys
import types

import pytest

from app.tts_module import TTSModule

GPU = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def fake_supertonic(monkeypatch):
    """supertonic 패키지를 흉내낸다 — loader/config 둘 다 같은 상수를 들고 있다."""
    pkg = types.ModuleType("supertonic")
    loader = types.ModuleType("supertonic.loader")
    config = types.ModuleType("supertonic.config")
    loader.DEFAULT_ONNX_PROVIDERS = ["CPUExecutionProvider"]
    config.DEFAULT_ONNX_PROVIDERS = ["CPUExecutionProvider"]
    pkg.loader = loader
    pkg.config = config
    for name, mod in [("supertonic", pkg), ("supertonic.loader", loader),
                      ("supertonic.config", config)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return loader, config


def test_providers_patch_loader_not_config(fake_supertonic):
    loader, config = fake_supertonic
    TTSModule(providers=GPU)._apply_providers()

    assert loader.DEFAULT_ONNX_PROVIDERS == GPU, \
        "loader 의 이름을 덮어써야 한다 — config 만 고치면 조용히 무시된다"


def test_providers_none_leaves_supertonic_untouched(fake_supertonic):
    loader, _ = fake_supertonic
    TTSModule()._apply_providers()

    assert loader.DEFAULT_ONNX_PROVIDERS == ["CPUExecutionProvider"], \
        "providers 미지정이면 supertonic 기본값을 건드리면 안 됨"


# ------------------------------------------------------- 조용한 CPU 폴백 감지
class _FakeSession:
    def __init__(self, providers):
        self._p = providers

    def get_providers(self):
        return self._p


def _tts_with_session(providers, bound):
    t = TTSModule(providers=providers)
    t._tts = types.SimpleNamespace(model=types.SimpleNamespace(
        vocoder_ort=_FakeSession(bound)))
    return t


def test_warns_when_available_provider_did_not_bind(monkeypatch, caplog):
    # CUDA 가 이 기기에 있는데도 세션엔 CPU 만 붙음 = 위 함정에 빠진 상태.
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: GPU)

    with caplog.at_level("WARNING", logger="jaeha_bot.tts"):
        _tts_with_session(GPU, ["CPUExecutionProvider"])._report_providers()

    msgs = [r.getMessage() for r in caplog.records]
    assert any("붙지 않음" in m and "CUDAExecutionProvider" in m for m in msgs), \
        f"GPU 가 있는데 안 붙었으면 경고해야 함: {msgs}"


def test_no_warning_when_gpu_absent_on_this_machine(monkeypatch, caplog):
    # 노트북 — CUDA 자체가 없다. 같은 config 를 공유하므로 여기선 조용해야 한다.
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    with caplog.at_level("WARNING", logger="jaeha_bot.tts"):
        _tts_with_session(GPU, ["CPUExecutionProvider"])._report_providers()

    assert not caplog.records, "CUDA 가 없는 기기에선 경고하면 안 됨(노트북 잡음)"
