"""파이프라인 벤치가 **아이가 실제로 기다리는 시간**을 재는지.

🔴 2026-08-19 까지 이 도구의 '합계'는 `LLM + 첫 소리` 였다. 그런데 speak() 는 재생이
   끝날 때까지 블로킹한다 — 아이가 다음 말을 할 수 있게 되는 시점은 **말이 끝난 뒤**다.
   즉 답변이 길어지면 그 대가가 이 표에 **아예 안 잡혔다.**
   하필 그게 오늘 클로바 판정의 갈림길이었다: HCX-005 는 gpt 보다 0.84초 빠른데
   답변의 12.5%가 40자를 넘고 최대 106자다(gpt·EXAONE 은 40자 초과 0건).
   번 시간을 말하는 데 도로 쓰는지 아닌지를 이 도구가 답해야 한다.

또 하나: 조합 표기가 `local|openai` 뿐이라 **어느 원격 모델인지** 못 적었다.
gpt 와 HCX 를 같은 표에서 비교하려면 모델 이름을 조합에 넣을 수 있어야 한다.
"""
import pytest

from tools.bench_pipeline import felt_total, parse_combo


def test_plain_combo_has_no_model_override():
    assert parse_combo("local+supertonic") == ("local", None, "supertonic")


def test_model_can_be_pinned_on_the_llm_side():
    assert parse_combo("openai:HCX-005+supertonic") == ("openai", "HCX-005", "supertonic")


def test_gpt_can_be_pinned_too_so_both_arms_are_explicit():
    """한쪽만 모델을 적으면 다른 쪽이 설정값을 따라가 '무엇과 비교했는지'가 흐려진다."""
    assert parse_combo("openai:gpt-4o-mini+supertonic") == (
        "openai", "gpt-4o-mini", "supertonic")


@pytest.mark.parametrize("bad", ["local", "openai:HCX-005", "a+b+c", ""])
def test_malformed_combo_is_rejected(bad):
    with pytest.raises(SystemExit):
        parse_combo(bad)


# ── 체감 총시간 ───────────────────────────────────────────────────────────────

def test_felt_total_includes_the_speaking_time():
    """LLM 1.0s + 첫 소리 1.1s + 재생 2.4s = 4.5s. 재생을 빼면 답변 길이가 안 보인다."""
    assert felt_total(1.0, 1.1, 2.4) == pytest.approx(4.5)


def test_a_longer_answer_costs_more_even_when_the_llm_is_faster():
    """🔴 이 도구가 답해야 하는 바로 그 질문.

    빠른 모델(LLM 0.83s)이 말을 길게 하면(재생 3.4s), 느린 모델(LLM 1.67s)이
    짧게 말하는 것(재생 2.0s)보다 아이를 더 기다리게 할 수 있다.
    """
    fast_but_wordy = felt_total(0.83, 1.07, 3.40)
    slow_but_brief = felt_total(1.67, 1.07, 2.00)

    assert fast_but_wordy > slow_but_brief


def test_synthesis_only_mode_has_no_playback_to_add():
    """--play 없이 재면 재생 시간이 없다. 0 을 넣어도 계산이 깨지지 않아야 한다."""
    assert felt_total(1.0, 0.9, 0.0) == pytest.approx(1.9)
