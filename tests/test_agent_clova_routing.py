"""운영(app/agent.py)이 클로바로 갈 때 **평가 도구와 같은 규칙**을 쓰는지.

🔴 규칙이 두 개면 평가에서 잰 품질이 운영에서 재현되지 않는다. tools/eval_llm.py 는
   '모델 이름이 HCX- 로 시작하면 클로바'로 갈래를 판단한다. 운영도 같아야 한다.
   그러면 백엔드 교체가 `api_model: HCX-005` 한 줄이 된다.

🔴 이 파일이 지키는 가장 중요한 것: **키가 없을 때 조용히 로컬로 새지 않는 것.**
   _complete() 는 원격 실패를 잡아 로컬로 폴백한다 — 아이가 침묵을 듣지 않게 하려는
   설계다. 그런데 '키 없음'은 네트워크 장애가 아니라 **설정 오류**다. 폴백에 삼켜지면
   "클로바를 켰다"고 믿는데 실제로는 EXAONE 이 2.3초씩 답하는 상태가 되고,
   로그 한 줄 말고는 아무 신호가 없다. 2026-08-10 에 실제로 이 종류로 당했다.
"""
import sys
import types

import pytest

from app.agent import CLOVA_BASE_URL, LLMAgent


@pytest.fixture
def spy_openai(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

    mod = types.ModuleType("openai")
    mod.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", mod)
    return seen


def _agent(**kw):
    return LLMAgent(model_path="없어도된다.gguf", system_prompt="시스템", **kw)


def test_hcx_model_goes_to_clova_with_its_own_key(spy_openai, monkeypatch):
    monkeypatch.setenv("CLOVA_STUDIO_KEY", "clova-key")

    _agent(backend="openai", api_model="HCX-005")._api_client()

    assert spy_openai["base_url"] == CLOVA_BASE_URL
    assert spy_openai["api_key"] == "clova-key"


def test_gpt_model_keeps_the_env_convention(spy_openai, monkeypatch):
    """OpenAI 는 예전처럼 인자 없이 만든다 — OPENAI_API_KEY 관례를 깨면 안 된다."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    _agent(backend="openai", api_model="gpt-4o-mini")._api_client()

    assert "base_url" not in spy_openai, "OpenAI 모델을 클로바로 보내고 있다"
    assert "api_key" not in spy_openai


@pytest.mark.parametrize("model", ["gpt-4o-mini", "HCX-005"])
def test_retry_and_timeout_survive_the_switch(spy_openai, monkeypatch, model):
    """🔴 백엔드를 바꾸다 재시도가 되살아나면 p95 가 13초로 돌아간다(08-18 실측)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CLOVA_STUDIO_KEY", "clova-key")

    _agent(backend="openai", api_model=model, api_timeout=3.5)._api_client()

    assert spy_openai["max_retries"] == 0, f"{model} 에 재시도가 살아 있다"
    assert spy_openai["timeout"] == 3.5


def test_missing_clova_key_fails_at_startup_not_silently(monkeypatch):
    """키가 없으면 **기동 때** 터진다. 폴백에 삼켜져 조용히 EXAONE 이 되면 안 된다."""
    monkeypatch.delenv("CLOVA_STUDIO_KEY", raising=False)

    with pytest.raises(ValueError, match="CLOVA_STUDIO_KEY"):
        _agent(backend="openai", api_model="HCX-005")


def test_local_backend_does_not_need_any_key(monkeypatch):
    """로컬만 쓰는 기계는 키가 없어도 떠야 한다."""
    monkeypatch.delenv("CLOVA_STUDIO_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _agent(backend="local", api_model="HCX-005")   # 예외가 나면 실패


def test_clova_key_check_is_skipped_for_openai_models(monkeypatch):
    monkeypatch.delenv("CLOVA_STUDIO_KEY", raising=False)

    _agent(backend="openai", api_model="gpt-4o-mini")   # 예외가 나면 실패
