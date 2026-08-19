"""지연 벤치 도구가 **올바른 서버에, 재시도 없이** 붙는지.

이 도구의 결론은 '어느 서버가 더 가까운가'다. 그래서 두 가지가 틀리면 결론이 통째로 뒤집힌다:
- HCX 를 OpenAI 로 보내면(또는 반대) 재는 대상이 아예 다르다.
- SDK 자동 재시도가 살아 있으면 '느린 호출'과 '세 번 친 호출'을 구분할 수 없다
  (2026-08-18 에 실제로 p95 가 3.757s -> 13.233s 로 보였다).
"""
import sys
import types

import pytest

from tools.bench_llm_latency import CLOVA_BASE_URL, make_client


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


def test_hcx_goes_to_clova_with_its_own_key(spy_openai, monkeypatch):
    monkeypatch.setenv("CLOVA_STUDIO_KEY", "clova-key")
    make_client("HCX-DASH-002")

    assert spy_openai["base_url"] == CLOVA_BASE_URL
    assert spy_openai["api_key"] == "clova-key"


def test_openai_model_does_not_get_clova_base_url(spy_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    make_client("gpt-4o-mini")

    assert "base_url" not in spy_openai, "OpenAI 모델을 클로바로 보내고 있다"


@pytest.mark.parametrize("model", ["gpt-4o-mini", "HCX-DASH-002"])
def test_sdk_retry_disabled_for_every_backend(spy_openai, monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CLOVA_STUDIO_KEY", "clova-key")
    make_client(model)

    assert spy_openai["max_retries"] == 0, f"{model} 에 재시도가 살아 있다"


def test_missing_key_exits_with_message(monkeypatch, capsys):
    monkeypatch.delenv("CLOVA_STUDIO_KEY", raising=False)

    with pytest.raises(SystemExit):
        make_client("HCX-DASH-002")
    assert "CLOVA_STUDIO_KEY" in capsys.readouterr().out


# ── 로컬(EXAONE) 팔 ──────────────────────────────────────────────────────────
# 🔴 왜 지연 벤치가 로컬까지 재야 하는가 (2026-08-19):
#    eval_llm.py 도 지연을 찍지만 **모델을 하나씩 끝내고 다음으로 넘어간다.** 그래서
#    로컬 값과 원격 값 사이에 회선이 변하면 그 변화가 '모델 차이'로 둔갑한다.
#    로컬은 회선을 안 타니 상관없다고 생각하기 쉬운데, 비교 대상인 **원격 쪽이 흔들려서**
#    똑같이 오염된다. 세 팔을 문항마다 교차시켜야 같은 조건에서 잰 값이 된다.


class _FakeLlama:
    """_complete_local 이 부르는 것만 흉내낸다."""

    def __init__(self):
        self.calls = []

    def create_chat_completion(self, messages, max_tokens, **kw):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return {"choices": [{"message": {"content": "응 좋아!"}}]}


@pytest.fixture
def local_agent(monkeypatch):
    """LLMAgent 를 진짜로 만들되 GGUF 로드만 가짜로 바꾼다.

    백엔드 분기(_complete)는 진짜 코드를 타야 한다 — 우리가 확인하려는 게 바로 그 분기다.
    """
    from app import agent as agent_mod

    fake = _FakeLlama()
    monkeypatch.setattr(agent_mod.LLMAgent, "_ensure_loaded", lambda self: fake)
    return fake


def test_local_does_not_build_an_openai_client(local_agent, monkeypatch):
    """로컬 팔이 openai 를 import 하는 순간 재는 대상이 달라진다."""
    monkeypatch.setitem(sys.modules, "openai", None)  # 건드리면 TypeError 로 터진다

    client = make_client("local")
    client.chat.completions.create(
        model="local", max_completion_tokens=80,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "안녕"}])

    assert local_agent.calls, "로컬 추론이 한 번도 안 불렸다"


def test_local_never_falls_back_to_the_api(local_agent, monkeypatch):
    """🔴 설정이 backend: openai 여도 'local' 팔은 무조건 로컬이어야 한다.

    안 그러면 표에 'local' 이라 찍히면서 실제로는 gpt 를 잰 값이 들어간다.
    """
    from app.config import settings

    monkeypatch.setitem(settings.models["llm"], "backend", "openai")
    called = []

    from app import agent as agent_mod
    monkeypatch.setattr(agent_mod.LLMAgent, "_complete_api",
                        lambda self, m: called.append(m) or "원격 답")

    client = make_client("local")
    out = client.chat.completions.create(
        model="local", max_completion_tokens=80,
        messages=[{"role": "user", "content": "안녕"}])

    assert not called, "local 팔이 원격 API 를 쳤다"
    assert out.choices[0].message.content == "응 좋아!"


def test_local_uses_the_prompt_it_was_given(local_agent):
    """팔마다 시스템 프롬프트가 다르다([바닥] vs 실제). 넘긴 걸 그대로 써야 한다."""
    client = make_client("local")
    client.chat.completions.create(
        model="local", max_completion_tokens=80,
        messages=[{"role": "system", "content": "바닥 프롬프트"},
                  {"role": "user", "content": "안녕"}])

    assert local_agent.calls[0]["messages"][0]["content"] == "바닥 프롬프트"
    assert local_agent.calls[0]["max_tokens"] == 80


def test_local_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLOVA_STUDIO_KEY", raising=False)

    from tools.bench_llm_latency import required_key
    assert required_key("local") is None
