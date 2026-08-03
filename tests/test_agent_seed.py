"""seed 가 llama.cpp 까지 전달되는지 — GGUF 파일 불필요.

안 넘어가도 예외가 없고 답변도 그럴듯하게 나온다. 다만 매 실행 답이 달라져
설정 A/B 가 노이즈에 묻힌다(2026-08-03에 실제로 겪음: 같은 설정 26.5~49.5자).
"""
import sys
import types

import pytest

from app.agent import LLMAgent


@pytest.fixture
def captured_kwargs(monkeypatch, tmp_path):
    """llama_cpp.Llama 를 가로채 생성 인자를 잡아둔다."""
    seen = {}

    class FakeLlama:
        def __init__(self, **kw):
            seen.update(kw)

    mod = types.ModuleType("llama_cpp")
    mod.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)

    gguf = tmp_path / "fake.gguf"     # 존재 검사만 통과하면 된다
    gguf.write_bytes(b"")
    seen["_path"] = str(gguf)
    return seen


def _agent(captured, **kw):
    return LLMAgent(model_path=captured["_path"], system_prompt="테스트", **kw)


def test_seed_reaches_llama(captured_kwargs):
    _agent(captured_kwargs, seed=777)._ensure_loaded()

    assert captured_kwargs.get("seed") == 777, \
        f"seed 가 Llama 까지 전달돼야 함: {captured_kwargs}"


def test_seed_omitted_when_none(captured_kwargs):
    # 미지정이면 인자를 아예 넘기지 않는다(라이브러리 기본=랜덤을 그대로 둔다).
    _agent(captured_kwargs)._ensure_loaded()

    assert "seed" not in captured_kwargs, \
        f"미지정이면 seed 를 넘기지 말아야 함: {captured_kwargs}"


def test_other_llama_args_still_passed(captured_kwargs):
    _agent(captured_kwargs, n_ctx=2048, n_gpu_layers=-1, seed=777)._ensure_loaded()

    assert captured_kwargs["n_ctx"] == 2048
    assert captured_kwargs["n_gpu_layers"] == -1
