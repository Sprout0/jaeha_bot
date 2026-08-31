"""few-shot 을 담는 그릇: system 안의 '예시 텍스트' vs 진짜 주고받은 대화.

🔴 왜 스위치인가 (2026-08-31):
   - 08-28 측정: 진짜 메시지쌍으로 주면 **HCX 의 30자 초과가 25% -> 6%** 로 잡힌다
     (gpt 무해). 클로바 전환을 막고 있던 게 이 말투 문제였다.
   - 그런데 **반대 근거가 이미 기록돼 있다**: 예전에 가짜 대화이력으로 주입했을 때
     이전 예시의 '주제'가 실제 답에 샜다(무의미 입력에 '빨간색 게임' confabulation).
     그래서 지금의 텍스트 방식으로 바꾼 것이다.
   ➡️ 어느 쪽이 맞는지는 모델마다 다를 수 있다(누수는 로컬 EXAONE 에서 관찰됐고,
      길이 이득은 API 모델에서 쟀다). **설정 한 줄로 되돌아올 수 있어야 한다.**

⚠️ 여기서 재는 건 '그릇'뿐이다. 두 형식이 모델에 보여주는 **예시 내용은 같다.**
"""
import json

import pytest

from app.agent import LLMAgent

BASE_SYSTEM = "너는 티드야."


@pytest.fixture
def seed(tmp_path):
    """few_shot 두 쌍짜리 씨앗 파일. 실제 파일과 형식이 같다."""
    rows = [
        {"few_shot": True, "messages": [{"role": "user", "content": "졸려요"},
                                        {"role": "assistant", "content": "졸리구나!"}]},
        {"few_shot": True, "messages": [{"role": "user", "content": "배고파"},
                                        {"role": "assistant", "content": "배고프구나!"}]},
        # few_shot 이 아닌 줄은 주입되지 않는다(QLoRA 학습용으로만 쓰인다).
        {"few_shot": False, "messages": [{"role": "user", "content": "학습전용"},
                                         {"role": "assistant", "content": "새면안된다"}]},
    ]
    p = tmp_path / "seed.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(p)


def _agent(seed, form):
    return LLMAgent(model_path="없어도된다.gguf", system_prompt=BASE_SYSTEM,
                    fewshot_path=seed, fewshot_form=form)


def test_turns_form_sends_examples_as_real_messages(seed):
    """turns 형식은 예시를 **주고받은 대화**로 앞에 깐다 — 이게 08-28 이 잰 그릇이다."""
    agent = _agent(seed, "turns")

    assert agent._build_messages("안녕", "") == [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": "졸려요"},
        {"role": "assistant", "content": "졸리구나!"},
        {"role": "user", "content": "배고파"},
        {"role": "assistant", "content": "배고프구나!"},
        {"role": "user", "content": "안녕"},
    ]


def test_turns_form_leaves_the_system_prompt_alone(seed):
    """예시가 messages 로 갔으면 system 에는 **없어야 한다.** 둘 다 들어가면 예시가 두 번
    실려 프롬프트가 커지고, 08-28 이 잰 것과 다른 것을 운영하게 된다."""
    agent = _agent(seed, "turns")

    assert agent.system_prompt == BASE_SYSTEM
    assert "졸리구나" not in agent.system_prompt


def test_text_form_keeps_the_examples_inside_the_system_prompt(seed):
    """현행(text)은 그대로여야 한다 — 되돌릴 곳이 있어야 스위치다."""
    agent = _agent(seed, "text")

    assert "졸리구나!" in agent.system_prompt
    assert agent._build_messages("안녕", "") == [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": "안녕"},
    ]


def test_examples_stay_ahead_of_the_conversation_history(seed):
    """예시는 늘 이력보다 앞이다. 뒤로 가면 '방금 한 말'로 읽혀 아이 말을 덮는다."""
    agent = _agent(seed, "turns")
    agent._remember("사자 하자", "좋아!")

    roles = [m["role"] for m in agent._build_messages("안녕", "")]
    contents = [m.get("content") for m in agent._build_messages("안녕", "")]
    assert roles == ["system", "user", "assistant", "user", "assistant",
                     "user", "assistant", "user"]
    assert contents.index("배고프구나!") < contents.index("좋아!")


def test_examples_that_are_not_marked_fewshot_never_ship(seed):
    """씨앗 파일은 QLoRA 학습분까지 같이 들고 있다. few_shot=true 만 나가야 한다."""
    agent = _agent(seed, "turns")

    sent = json.dumps(agent._build_messages("안녕", ""), ensure_ascii=False)
    assert "학습전용" not in sent and "새면안된다" not in sent


def test_a_typo_in_the_form_is_caught_at_startup(seed):
    """설정 오타가 조용히 한쪽으로 폴백하면, 무엇을 운영 중인지 아무도 모른다."""
    with pytest.raises(ValueError, match="fewshot_form"):
        _agent(seed, "turn")     # 's' 하나 빠뜨린 흔한 오타


def test_local_warmup_carries_the_same_examples_the_real_turn_will(seed, monkeypatch):
    """폴백 예열이 실제 턴보다 짧은 프롬프트로 돌면, n_ctx 초과를 예열이 못 잡는다.

    🔴 2026-08-11~18 에 이 모양으로 로컬 폴백이 일주일간 죽어 있었다. 예열은 경고 로그로만
       나가는 자리라 **인터넷이 끊기는 날에야** 발견될 종류다. 예열은 실제 턴과 같은 크기로
       돌아야 뜻이 있다.
    """
    agent = _agent(seed, "turns")
    sent = []
    monkeypatch.setattr(agent, "_complete_local", lambda msgs: sent.append(msgs) or "응")

    agent._warm_local()

    assert sent, "예열이 로컬을 아예 안 쳤다"
    assert [m["content"] for m in sent[0]][:5] == [
        BASE_SYSTEM, "졸려요", "졸리구나!", "배고파", "배고프구나!"]


def test_warmup_does_not_leave_the_examples_in_the_history(seed, monkeypatch):
    """예열 대화가 이력에 남으면 첫 답변의 맥락이 오염된다(warm() 이 지키는 규칙)."""
    agent = _agent(seed, "turns")
    monkeypatch.setattr(agent, "_complete_local", lambda msgs: "응")

    agent._warm_local()

    assert agent.history == []


# ── 도구와 운영이 같은 앞머리를 쓰는가 ────────────────────────────────────────
# 🔴 측정 도구가 운영과 다른 프롬프트를 조립하면 **틀린 것을 재고도 모른다.**
#    이 프로젝트에서 그 종류로 여러 번 당했다(2026-08-11 채점 대상, 08-20 측정 도구 셋).
#    그래서 앞머리 조립은 함수 하나(build_prefix)로 두고 양쪽이 그걸 쓴다.

def test_build_prefix_text_form_is_one_system_message(seed):
    from app.agent import _augment_system, build_prefix, load_fewshot

    fs = load_fewshot(seed)
    assert build_prefix(BASE_SYSTEM, fs, "text") == [
        {"role": "system", "content": _augment_system(BASE_SYSTEM, fs)}]


def test_build_prefix_turns_form_keeps_system_clean_and_appends_the_talk(seed):
    from app.agent import build_prefix, load_fewshot

    fs = load_fewshot(seed)
    assert build_prefix(BASE_SYSTEM, fs, "turns") == \
        [{"role": "system", "content": BASE_SYSTEM}] + fs


def test_the_agent_and_the_tools_assemble_the_very_same_prefix(seed):
    """도구가 이걸 쓰는 한 운영과 어긋날 수 없다. 어긋나면 여기서 걸린다."""
    from app.agent import build_prefix, load_fewshot

    fs = load_fewshot(seed)
    for form in ("text", "turns"):
        agent = _agent(seed, form)
        assert agent._build_messages("안녕", "") == \
            build_prefix(BASE_SYSTEM, fs, form) + [{"role": "user", "content": "안녕"}]


def test_build_prefix_rejects_a_form_it_does_not_know(seed):
    from app.agent import build_prefix, load_fewshot

    with pytest.raises(ValueError, match="fewshot_form"):
        build_prefix(BASE_SYSTEM, load_fewshot(seed), "turn")
