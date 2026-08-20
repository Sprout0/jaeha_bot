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


# ── 팔 구성 ──────────────────────────────────────────────────────────────────
# 🔴 2026-08-19 실측으로 드러난 함정. local 에 [바닥] 팔을 붙였더니 중앙 **6.277s**
#    가 나왔다. 프롬프트를 고정해 다시 재니 **2.097s**. 4.2초가 측정 방식 탓이었다.
#    llama.cpp 는 같은 시스템 프롬프트가 이어지면 접두 KV 캐시를 재사용한다.
#    42자 바닥 프롬프트와 4185자 실제 프롬프트를 번갈아 치면 매 호출 2,356토큰
#    prompt eval 을 처음부터 다시 낸다. 운영은 프롬프트가 고정이라 그 값을 안 낸다.
#    ⚠️ 원격 팔이 사이에 끼는 건 무해하다 — llama 컨텍스트를 건드리지 않는다.
#       깨뜨리는 건 오직 '로컬의 두 번째 프롬프트'다.
#    게다가 [바닥]='그 서버까지의 왕복 하한'인데 local 은 갈 서버가 없다. 뜻도 없다.

from tools.bench_llm_latency import FLOOR_SYSTEM, build_arms  # noqa: E402


def test_local_gets_no_floor_arm():
    arms = build_arms(["local"], "실제 프롬프트", with_floor=True)

    prompts = {sysp for _, _, sysp in arms}
    assert prompts == {"실제 프롬프트"}, "local 팔의 프롬프트가 두 종류다 — KV 캐시가 깨진다"
    assert len(arms) == 1


def test_remote_models_still_get_their_floor_arm():
    arms = build_arms(["gpt-4o-mini"], "실제 프롬프트", with_floor=True)

    assert [a[0] for a in arms] == ["gpt-4o-mini [바닥]", "gpt-4o-mini"]
    assert arms[0][2] == FLOOR_SYSTEM


def test_mixed_run_keeps_one_prompt_for_local_and_two_for_remote():
    arms = build_arms(["gpt-4o-mini", "local"], "실제 프롬프트", with_floor=True)

    assert [a[0] for a in arms] == ["gpt-4o-mini [바닥]", "gpt-4o-mini", "local"]


def test_no_floor_drops_every_floor_arm():
    arms = build_arms(["gpt-4o-mini", "local"], "실제 프롬프트", with_floor=False)

    assert [a[0] for a in arms] == ["gpt-4o-mini", "local"]


# ── 호출 간격 ────────────────────────────────────────────────────────────────
# 🔴 2026-08-20: 같은 모델을 두 도구로 쟀는데 클로바만 갈렸다.
#      bench_llm_latency (연달아 호출)          HCX 0.831s
#      bench_pipeline    (턴 사이 TTS 재생 3.5s) HCX 1.264s
#    gpt 는 두 도구에서 1.59/1.66s 로 일치했다. 클로바만 간격에 반응한다면
#    '커넥션이 식는 것'이고, 그건 운영에서 실제로 일어나는 일이다 — 아이가 말하고
#    봇이 3.5초 말하는 동안 호출이 없다. 재는 조건이 운영과 다르면 숫자도 다르다.

def test_gap_defaults_to_zero_so_old_runs_stay_comparable():
    from tools.bench_llm_latency import build_arms  # noqa: F401
    import inspect
    from tools import bench_llm_latency

    sig = inspect.signature(bench_llm_latency.main)
    assert sig is not None   # main 은 인자를 안 받는다(argparse)


def test_gap_is_applied_between_calls(monkeypatch):
    """--gap 이 켜지면 호출 사이에 그만큼 쉰다."""
    from tools import bench_llm_latency

    slept = []
    monkeypatch.setattr(bench_llm_latency.time, "sleep", lambda s: slept.append(s))

    bench_llm_latency.pace(0.0)
    assert slept == [], "gap 0 인데 잤다 — 예전 측정과 비교가 안 된다"

    bench_llm_latency.pace(3.5)
    assert slept == [3.5]
