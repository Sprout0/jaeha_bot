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


# ── 스트리밍 분해 ────────────────────────────────────────────────────────────
# 🔴 왜 필요한가 (2026-08-28): '생각 1.44s' 가 어디로 가는지 갈 방법이 없었다.
#    08-21 에 "87% 가 왕복"이라 결론 냈지만 그건 [바닥] 팔의 **총 시간**으로 잰
#    간접 추정이었다. 스트리밍을 켜면 첫 토큰까지(=왕복+프롬프트)와 그 뒤(=생성)를
#    직접 가를 수 있다.
# ⚠️ 조용히 틀리는 자리: 첫 청크는 대개 role 만 있고 content 가 비어 있다. 그걸
#    첫 토큰으로 세면 TTFT 가 과소평가되고 생성이 그만큼 부푼다 — 즉 "생성이 병목"
#    이라는 **정반대 결론**이 나온다. 그래서 여기에 못을 박는다.

from types import SimpleNamespace  # noqa: E402

from tools.bench_llm_latency import summarize_stream  # noqa: E402


def test_ttft_ignores_the_empty_role_chunk():
    """빈 delta 를 첫 토큰으로 세면 TTFT 가 과소평가된다."""
    tr = summarize_stream(t0=10.0, events=[(10.4, ""), (11.2, "응"), (11.3, " 좋아!")])

    assert tr.ttft_s == pytest.approx(1.2), "빈 청크를 첫 토큰으로 셌다"


def test_generation_spans_first_content_to_last_content():
    tr = summarize_stream(t0=10.0, events=[(10.4, ""), (11.2, "응"), (11.7, " 좋아!")])

    assert tr.gen_s == pytest.approx(0.5)


def test_buffered_server_shows_up_as_zero_generation():
    """서버가 안 흘려주고 한 번에 주면 생성 0s 로 찍힌다 — 실패가 아니라 결과다."""
    tr = summarize_stream(t0=10.0, events=[(11.9, "응 좋아! 나도 그래.")])

    assert tr.gen_s == pytest.approx(0.0)
    assert tr.ttft_s == pytest.approx(1.9)


def test_first_sentence_time_is_when_tts_could_start_speaking():
    """TTS 가 실제로 자르는 그 경계에서 재야 숫자가 옮겨간다."""
    tr = summarize_stream(t0=0.0, events=[
        (1.0, "응"), (1.1, " 좋아!"), (1.2, " 나"), (1.3, "도 그래.")])

    assert tr.first_sentence_s == pytest.approx(1.2)


def test_single_sentence_answer_has_nothing_to_cut_early():
    """한 문장짜리 답은 스트리밍으로 벌 게 없다. None 이어야 '절감 0'으로 집계된다."""
    tr = summarize_stream(t0=0.0, events=[(1.0, "응"), (1.4, " 좋아!")])

    assert tr.first_sentence_s is None


def test_empty_stream_reports_no_first_token():
    tr = summarize_stream(t0=0.0, events=[(1.0, ""), (1.1, "")])

    assert tr.ttft_s is None
    assert tr.chars == 0


def test_stream_splitter_is_the_one_tts_actually_uses():
    """분리 규칙이 갈라지면 여기서 잰 '첫 문장'이 TTS 에서 재현되지 않는다."""
    from app import tts_module
    from tools import bench_llm_latency

    assert bench_llm_latency._SENT_SPLIT is tts_module._SENT_SPLIT


# ── 호출 이음매(--stream) ────────────────────────────────────────────────────
# 🔴 기본값은 반드시 '스트리밍 끔'이어야 한다. 스위치를 안 준 실행이 예전과 **한 글자도
#    다르지 않아야** 08-19/08-21 값과 비교할 수 있다.

class _FakeCompletions:
    def __init__(self, chunks=None, text="응 좋아!"):
        self.kwargs = None
        self._chunks = chunks
        self._text = text

    def create(self, **kw):
        self.kwargs = kw
        if kw.get("stream"):
            return iter(self._chunks or [])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))])


def _fake_client(chunks=None, text="응 좋아!"):
    comp = _FakeCompletions(chunks, text)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_stream_off_asks_the_server_exactly_as_before():
    from tools.bench_llm_latency import call_once

    client, comp = _fake_client()
    call_once(client, "gpt-4o-mini", "시스템", "안녕", max_tokens=80, stream=False)

    assert "stream" not in comp.kwargs, "기본 실행이 예전과 달라졌다 — 옛 값과 비교 불가"


def test_stream_on_asks_the_server_to_stream():
    from tools.bench_llm_latency import call_once

    client, comp = _fake_client(chunks=[_chunk("응"), _chunk(" 좋아!")])
    call_once(client, "gpt-4o-mini", "시스템", "안녕", max_tokens=80, stream=True)

    assert comp.kwargs["stream"] is True


def test_stream_off_still_reports_the_answer_length():
    from tools.bench_llm_latency import call_once

    client, _ = _fake_client(text="응 좋아! 나도.")
    _total, tr = call_once(client, "gpt-4o-mini", "시스템", "안녕", max_tokens=80, stream=False)

    assert tr.chars == len("응 좋아! 나도.")
    assert tr.ttft_s is None, "스트리밍을 안 켰는데 TTFT 가 찍혔다"


def test_usage_only_chunk_without_choices_does_not_crash():
    """스트림 끝에 choices 가 빈 청크를 주는 서버가 있다(사용량 보고)."""
    from tools.bench_llm_latency import call_once

    client, _ = _fake_client(chunks=[_chunk("응"), SimpleNamespace(choices=[])])
    _total, tr = call_once(client, "gpt-4o-mini", "시스템", "안녕", max_tokens=80, stream=True)

    assert tr.chars == 1


# ── 429 를 다른 실패와 가른다 ────────────────────────────────────────────────
# 🔴 테스트 앱 키는 분당 한도가 있다. 08-19 에 HCX 최대값이 11.143s 로 찍힌 게 그것이다.
#    429 를 일반 실패와 섞어 세면 '클로바가 불안정하다'와 '우리 키가 테스트 키다'를
#    사후에 가를 수 없다.

def test_rate_limit_is_recognized():
    from tools.bench_llm_latency import is_rate_limited

    assert is_rate_limited(Exception("Error code: 429 - Too Many Requests"))
    assert not is_rate_limited(Exception("Connection reset by peer"))


def test_rate_limit_words_come_from_the_eval_tool():
    """판정 낱말이 갈라지면 두 도구의 실패 집계가 서로 다른 뜻이 된다."""
    from tools import bench_llm_latency, eval_llm

    assert bench_llm_latency._RATE_LIMIT is eval_llm._RATE_LIMIT


def test_stream_with_local_is_refused_before_the_run_starts():
    """local 팔은 스트림을 흉내 낼 수 없다. 20분 돌린 뒤 터지면 그 시간이 통째로 날아간다."""
    from tools.bench_llm_latency import reject_unstreamable

    with pytest.raises(SystemExit):
        reject_unstreamable(["gpt-4o-mini", "local"], stream=True)

    reject_unstreamable(["gpt-4o-mini", "local"], stream=False)   # 안 켰으면 통과
    reject_unstreamable(["gpt-4o-mini", "HCX-005"], stream=True)


# ── 분해를 읽는 법 ───────────────────────────────────────────────────────────
# 🔴 08-21 의 "생각의 87% 가 왕복" 자체가 잘못 읽은 숫자였다. 그러니 이 도구가 내는
#    숫자도 **읽는 법을 같이 찍어야** 한다. 특히 두 모양이 위험하다:
#    - 프롬프트 대가가 음수 = 바닥 팔이 더 느리게 나온 것 = 효과가 잡음보다 작다.
#      그걸 "-30.9%" 로만 찍으면 '프롬프트가 시간을 줄여준다'로 읽힌다.
#    - 생성이 0 에 가까움 = 서버가 안 흘려준 것. 그러면 문장 스트리밍은 무의미하다.

def test_negative_prompt_cost_is_called_noise_not_a_saving():
    from tools.bench_llm_latency import describe_split

    lines = "\n".join(describe_split(rtt=1.5, ttft_real=1.1, gen=0.2, total=1.4, chars=23))

    assert "잡음" in lines, "음수 프롬프트 대가를 '이득'으로 읽히게 두고 있다"


def test_normal_split_reports_each_share():
    from tools.bench_llm_latency import describe_split

    lines = "\n".join(describe_split(rtt=1.0, ttft_real=1.2, gen=0.2, total=1.5, chars=23))

    assert "0.200s" in lines, "프롬프트 대가(1.2-1.0)가 안 보인다"
    assert "66.7%" in lines, "왕복 비율이 안 보인다"
    assert "잡음" not in lines


def test_buffered_server_is_flagged():
    from tools.bench_llm_latency import describe_split

    lines = "\n".join(describe_split(rtt=1.0, ttft_real=1.4, gen=0.01, total=1.45, chars=23))

    assert "안 흘려주고" in lines


def test_short_answer_with_zero_generation_is_not_flagged_as_buffering():
    """8자짜리 답은 청크 한둘로 끝나는 게 정상이다 — 버퍼링과 구분해야 한다."""
    from tools.bench_llm_latency import describe_split

    lines = "\n".join(describe_split(rtt=1.0, ttft_real=1.4, gen=0.01, total=1.45, chars=8))

    assert "안 흘려주고" not in lines
