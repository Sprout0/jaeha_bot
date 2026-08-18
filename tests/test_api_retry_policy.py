"""OpenAI SDK 의 **자동 재시도**를 끄고, 타임아웃을 실측 분산 위에 두는지.

🔴 2026-08-18 실측이 드러낸 것: 타임아웃이 지연을 **줄이는 게 아니라 늘리고 있었다.**

    `OpenAI(timeout=3.0)` 만 넘기면 `max_retries` 는 SDK 기본값 2 가 산다.
    3초를 넘긴 호출은 폴백으로 안 가고 **처음부터 두 번 더** 쳐진다.

      운영(재시도 켜짐)  중앙 2.580s / p95 **13.233s**
      재시도 끔          중앙 1.610s / p95   3.757s

    `api_timeout: 3.0` 은 08-10 의 p95 1.72s 를 보고 정한 값인데,
    2026-08-18 실측 p95 는 3.7~4.4s 다. **타임아웃이 정상 분산 안쪽으로 들어왔다** —
    평범한 호출이 매번 문턱을 스치고, 스칠 때마다 3배로 부풀었다.

여기서 지키려는 것:
- 재시도는 SDK 가 아니라 **우리가** 정한다. 느린 호출은 다시 치는 게 아니라 폴백으로 간다
  (그래야 `api_timeout + 로컬시간` 이라는 [app/agent.py] 의 침묵 계산이 성립한다).
- 타임아웃은 실측 p95 **위**에 있어야 한다. 아래로 내려가면 헛폴백만 는다.
"""
import inspect
import sys
import types

import pytest

from app.agent import LLMAgent
from app.config import settings

# 2026-08-18 젯슨 실측(gpt-4o-mini, 14문항×3회, 재시도 끔): p95 3.757s / 최대 4.444s.
MEASURED_P95_S = 3.76


@pytest.fixture
def spy_openai(monkeypatch):
    """OpenAI() 에 넘어간 인자를 잡아 둔다. 실제 호출은 하지 않는다."""
    seen = {}

    class FakeCompletions:
        def create(self, **kw):
            msg = types.SimpleNamespace(content="응!")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    mod = types.ModuleType("openai")
    mod.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", mod)
    return seen


def test_sdk_auto_retry_is_disabled(spy_openai):
    agent = LLMAgent(model_path="dummy.gguf", system_prompt="테스트", backend="openai")
    agent.respond("안녕")

    assert spy_openai.get("max_retries") == 0, (
        "SDK 기본 재시도(2회)가 살아 있다 — 타임아웃 3초짜리 호출이 13초가 된다. "
        f"넘어간 인자: {spy_openai}"
    )


def test_timeout_still_passed(spy_openai):
    """재시도를 끄면서 타임아웃까지 놓치면, 느린 호출이 영영 안 끝난다."""
    agent = LLMAgent(model_path="dummy.gguf", system_prompt="테스트",
                     backend="openai", api_timeout=4.5)
    agent.respond("안녕")

    assert spy_openai.get("timeout") == 4.5


def test_configured_timeout_is_above_measured_p95():
    """설정값이 실측 p95 아래면, 정상 호출이 실패로 취급돼 헛폴백이 는다."""
    configured = settings.models["llm"]["api_timeout"]

    assert configured > MEASURED_P95_S, (
        f"api_timeout {configured}s <= 실측 p95 {MEASURED_P95_S}s. "
        "정상 범위의 호출이 타임아웃으로 잘린다."
    )


def test_default_timeout_is_above_measured_p95():
    default = inspect.signature(LLMAgent.__init__).parameters["api_timeout"].default

    assert default > MEASURED_P95_S, (
        f"생성자 기본 api_timeout {default}s 가 실측 p95 {MEASURED_P95_S}s 아래다"
    )
