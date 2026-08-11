"""LLM API 백엔드 + 로컬 EXAONE 폴백. 네트워크·GGUF 불필요.

배경(2026-08-10 젯슨 실측): LLM 을 API 로 빼면 체인 중앙 2.642s -> 1.816s.
이득이 큰 만큼, 네트워크가 끊겼을 때 아이가 아무 답도 못 듣는 일이 없어야 한다.

여기서 지키려는 것:
- 기본값은 로컬이다(설정 없이 과금되면 안 된다).
- API 답변도 **로컬과 똑같은 후처리**를 거친다. 이모지·마크다운·'재하봇:' 이름표는
  TTS 로 읽으면 치명적이라, 백엔드가 바뀌었다고 빠지면 안 된다.
- 실패하면 로컬로 내려가고, 반드시 기록을 남긴다.
"""
import sys
import types

import pytest

from app.agent import LLMAgent

LOCAL_REPLY = "로컬이 대답했어!"


@pytest.fixture
def fake_llama(monkeypatch, tmp_path):
    """llama_cpp 를 흉내낸다. loaded 로 '로컬이 실제로 불렸는지'를 본다."""
    state = {"loaded": 0, "calls": []}

    class FakeLlama:
        def __init__(self, **kw):
            state["loaded"] += 1

        def create_chat_completion(self, messages, **kw):
            state["calls"].append(messages)
            return {"choices": [{"message": {"content": LOCAL_REPLY}}]}

    mod = types.ModuleType("llama_cpp")
    mod.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)

    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"")
    state["path"] = str(gguf)
    return state


@pytest.fixture
def fake_openai(monkeypatch):
    """openai 를 흉내낸다. reply 를 바꿔 후처리를 검사하고, fail 로 폴백을 검사한다."""
    state = {"calls": [], "fail": None, "reply": "API 가 대답했어!"}

    class FakeCompletions:
        def create(self, **kw):
            state["calls"].append(kw)
            if state["fail"]:
                raise state["fail"]
            msg = types.SimpleNamespace(content=state["reply"])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    mod = types.ModuleType("openai")
    mod.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", mod)
    return state


def _agent(llama, **kw):
    return LLMAgent(model_path=llama["path"], system_prompt="너는 재하봇이야.", **kw)


# ------------------------------------------------------------------ 백엔드 선택
def test_default_backend_is_local(fake_llama, fake_openai):
    reply = _agent(fake_llama).respond("안녕")["text"]

    assert fake_openai["calls"] == [], "기본값은 로컬이어야 한다(설정 없이 과금되면 안 됨)"
    assert reply == LOCAL_REPLY


def test_api_backend_answers_without_loading_local_model(fake_llama, fake_openai):
    reply = _agent(fake_llama, backend="openai", api_model="gpt-4o-mini").respond("안녕")["text"]

    assert reply == "API 가 대답했어!"
    assert len(fake_openai["calls"]) == 1
    assert fake_openai["calls"][0]["model"] == "gpt-4o-mini"
    assert fake_llama["loaded"] == 0, "API 로 답했으면 GGUF 를 메모리에 올릴 이유가 없다"


def test_api_receives_system_prompt_and_user_text(fake_llama, fake_openai):
    _agent(fake_llama, backend="openai").respond("칼 어딨어?")

    msgs = fake_openai["calls"][0]["messages"]
    assert msgs[0]["role"] == "system" and "재하봇" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "칼 어딨어?"}


def test_api_keeps_conversation_history(fake_llama, fake_openai):
    agent = _agent(fake_llama, backend="openai")
    agent.respond("안녕")
    agent.respond("뭐해?")

    msgs = fake_openai["calls"][1]["messages"]
    assert {"role": "user", "content": "안녕"} in msgs, "이력이 API 에도 실려야 한다"


# ------------------------------------------------------------------- 후처리 공유
def test_api_reply_gets_the_same_postprocessing(fake_llama, fake_openai):
    # 이모지·굵게·화자이름표·3번째 문장 — 전부 TTS 로 읽으면 안 되는 것들.
    fake_openai["reply"] = "재하봇: **멍멍!** 😀 같이 놀자! 세 번째 문장이야."

    reply = _agent(fake_llama, backend="openai").respond("멍멍")["text"]

    assert reply == "멍멍! 같이 놀자!", f"로컬과 같은 후처리를 거쳐야 한다: {reply!r}"


# -------------------------------------------------------------------- 폴백
def test_falls_back_to_local_when_api_raises(fake_llama, fake_openai):
    fake_openai["fail"] = RuntimeError("connection reset")

    reply = _agent(fake_llama, backend="openai").respond("안녕")["text"]

    assert reply == LOCAL_REPLY, "API 가 죽으면 로컬이 답해야 한다"
    assert fake_llama["loaded"] == 1


def test_fallback_logs_warning(fake_llama, fake_openai, caplog):
    fake_openai["fail"] = RuntimeError("timeout")

    with caplog.at_level("WARNING", logger="jaeha_bot.agent"):
        _agent(fake_llama, backend="openai").respond("안녕")

    assert any("폴백" in r.getMessage() for r in caplog.records), \
        f"폴백은 기록을 남겨야 한다: {[r.getMessage() for r in caplog.records]}"


def test_missing_api_key_falls_back_without_crashing(fake_llama, monkeypatch):
    mod = types.ModuleType("openai")

    def boom(**kw):
        raise RuntimeError("api_key must be set")

    mod.OpenAI = boom
    monkeypatch.setitem(sys.modules, "openai", mod)

    reply = _agent(fake_llama, backend="openai").respond("안녕")["text"]

    assert reply == LOCAL_REPLY


# --------------------------------------------------------------------- render
def test_render_also_uses_api(fake_llama, fake_openai):
    fake_openai["reply"] = "같이 소리 맞추기 하자!"

    out = _agent(fake_llama, backend="openai").render("놀이를 시작하라고 말해")

    assert out == "같이 소리 맞추기 하자!"
    assert len(fake_openai["calls"]) == 1


def test_warm_preloads_local_model_when_backend_is_remote(fake_llama, fake_openai):
    # 폴백이 차가우면 그 순간 6초를 쓴다(2026-08-10 젯슨 실측).
    # 아이 앞에서 6초 침묵은 폴백이 아니라 실패다.
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert fake_llama["loaded"] == 1, "원격 백엔드일수록 로컬 폴백을 미리 데워야 한다"


def test_warm_runs_a_real_inference_not_just_load(fake_llama, fake_openai):
    # 🔴 로드만 데우면 안 된다 — 젯슨 실측에서 로드는 1.84s 인데 첫 폴백은 7.18s 였다.
    # 비용의 대부분은 2165자 시스템 프롬프트의 prompt eval + CUDA 커널 초기화라,
    # 실제로 한 번 추론을 돌려야 그게 사라진다.
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert fake_llama["calls"], "예열은 추론을 한 번 돌려야 한다(로드만으로는 부족)"
    assert fake_llama["calls"][0][0]["content"] == agent.system_prompt, \
        "실제 시스템 프롬프트로 데워야 prompt eval 이 캐시된다"


def test_warm_opens_the_api_connection_when_remote(fake_llama, fake_openai):
    """🔴 첫 턴이 3.1~3.6초인데 그 다음부터 0.5~0.6초다 — 차이는 TLS 최초 수립이다.

    아이의 **첫 질문**이 늘 이 값을 치른다. 기동 때 미리 한 번 다녀오면 사라진다.
    """
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert len(fake_openai["calls"]) == 1, "원격이면 커넥션도 데워야 한다"
    assert fake_openai["calls"][0]["max_completion_tokens"] == 1, \
        "예열은 연결만 맺으면 된다 — 토큰을 더 쓸 이유가 없다"


def test_warm_does_not_call_api_when_local(fake_llama, fake_openai):
    _agent(fake_llama).warm()

    assert fake_openai["calls"] == [], "로컬 백엔드가 API 를 부르면 안 된다(과금)"


def test_api_warm_failure_still_warms_local(fake_llama, fake_openai):
    # 네트워크가 없는 채로 켜질 수도 있다. 기동이 막히면 안 되고, 로컬은 데워져야 한다.
    fake_openai["fail"] = RuntimeError("no network")
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert fake_llama["loaded"] == 1
    assert fake_llama["calls"], "원격 예열이 실패해도 로컬 예열은 끝나야 한다"


def test_warm_does_not_pollute_history(fake_llama, fake_openai):
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert agent.history == [], "예열 대화가 이력에 남으면 안 된다"


def test_warm_is_idempotent(fake_llama, fake_openai):
    agent = _agent(fake_llama, backend="openai")

    agent.warm()
    agent.warm()

    assert fake_llama["loaded"] == 1, "두 번 불러도 한 번만 로드해야 한다"


def test_warmed_fallback_does_not_reload_on_failure(fake_llama, fake_openai):
    fake_openai["fail"] = RuntimeError("connection reset")
    agent = _agent(fake_llama, backend="openai")
    agent.warm()

    reply = agent.respond("안녕")["text"]

    assert reply == LOCAL_REPLY
    assert fake_llama["loaded"] == 1, "데워둔 모델을 그대로 써야 한다(재로드 금지)"


def test_render_does_not_leak_into_history(fake_llama, fake_openai):
    agent = _agent(fake_llama, backend="openai")
    agent.render("놀이를 시작하라고 말해")
    agent.respond("안녕")

    msgs = fake_openai["calls"][1]["messages"]
    assert not any("놀이를 시작하라고" in m["content"] for m in msgs), \
        "render 는 stateless 여야 한다 — 대화 이력을 오염시키면 안 된다"
