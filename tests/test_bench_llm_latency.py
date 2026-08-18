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
