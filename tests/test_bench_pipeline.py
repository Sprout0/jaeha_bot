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
    assert parse_combo("local+supertonic") == ("local", None, None, "supertonic")


def test_model_can_be_pinned_on_the_llm_side():
    assert parse_combo("openai:HCX-005+supertonic") == (
        "openai", "HCX-005", None, "supertonic")


def test_gpt_can_be_pinned_too_so_both_arms_are_explicit():
    """한쪽만 모델을 적으면 다른 쪽이 설정값을 따라가 '무엇과 비교했는지'가 흐려진다."""
    assert parse_combo("openai:gpt-4o-mini+supertonic") == (
        "openai", "gpt-4o-mini", None, "supertonic")


# ── 팔마다 답변 길이 상한 ────────────────────────────────────────────────────
# 🔴 2026-08-28 실측: HCX 가 LLM 에서 0.40초를 벌고 말하기에서 1.06초를 도로 뱉었다.
#    그런데 글자당 말하기 속도는 네 팔 모두 0.167~0.169s 로 같았다 — TTS 가 느린 게
#    아니라 **HCX 가 글자를 더 쓴 것**이다(20~22자 vs 26~29자).
#    그러면 남는 질문은 하나다: **길이를 맞추면 뒤집히는가.**
#    그걸 재려면 같은 실행 안에서 팔마다 문장 상한을 다르게 걸 수 있어야 한다.

def test_sentence_cap_can_be_pinned_per_arm():
    assert parse_combo("openai:HCX-005/1+supertonic") == (
        "openai", "HCX-005", 1, "supertonic")


def test_sentence_cap_works_without_a_model_name():
    assert parse_combo("local/1+supertonic") == ("local", None, 1, "supertonic")


@pytest.mark.parametrize("bad", ["openai:HCX-005/많이+supertonic",
                                 "openai:HCX-005/0+supertonic",
                                 "openai:HCX-005/+supertonic"])
def test_bad_sentence_cap_is_rejected(bad):
    """조용히 무시하면 '길이를 걸었다'고 믿는데 안 걸린 값이 표에 들어간다."""
    with pytest.raises(SystemExit):
        parse_combo(bad)


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


# ── 조합 사이에 앞 것을 놓는다 ───────────────────────────────────────────────
# 🔴 2026-08-28 실측으로 드러났다. build() 가 조합마다 TTSModule 을 새로 만드는데
#    앞 것을 **놓기 전에** 만든다. TRT 엔진이 하나에 ~2.9GB 라 젯슨 8GB 에서 두 개가
#    겹치면 터진다. 실제로 팔 3번째부터 TRT 가 236MB 할당에 실패해 CUDA 세션으로
#    폴백했고, 그 팔들의 '첫 소리'가 0.73s -> 1.43s 로 두 배가 됐다.
#    ⚠️ 조용히 느려진다 — 표에는 그냥 '그 조합이 느리다'로 찍힌다. 그래서 못을 박는다.

def test_previous_combo_is_released_before_the_next_is_built(monkeypatch):
    import gc
    import weakref

    from tools import bench_pipeline

    refs: list = []
    live_at_build: list[int] = []

    class _FakeTTS:
        pass

    class _FakeAgent:
        pass

    def fake_build(*a, **kw):
        gc.collect()
        live_at_build.append(sum(1 for r in refs if r() is not None))
        tts = _FakeTTS()
        refs.append(weakref.ref(tts))
        return _FakeAgent(), tts

    monkeypatch.setattr(bench_pipeline, "build", fake_build)
    monkeypatch.setattr(bench_pipeline, "run_combo", lambda *a, **kw: 1.0)

    for combo in ("openai:gpt-4o-mini+supertonic", "openai:HCX-005+supertonic"):
        bench_pipeline.measure_combo(combo, "시스템", rows=[], repeat=1,
                                     play=False, out_dir=None)

    assert live_at_build == [0, 0], (
        f"조합을 만들 때 앞 조합이 아직 살아 있다({live_at_build}) — TRT 엔진이 겹친다")


# ── 길이 지시 강화 ───────────────────────────────────────────────────────────
# 🔴 2026-08-28: 길이 지시는 **이미 시스템 프롬프트에 있다**
#    (configs/prompt_templates.yaml: "스무 글자 안팎으로, 길어도 두 문장까지만").
#    gpt 는 20자로 지키고 HCX 는 27~32자에 최대 74~97자로 안 지킨다.
#    그래서 재려는 건 '길이 유도를 넣으면'이 아니라 **'지시를 강화하면 따르는가'** 다.
# ⚠️ 반드시 **끝에** 붙인다. 앞에 끼우면 원문 규칙 사이를 갈라 놓고, 최신 지시가
#    더 세게 먹는 성질도 못 쓴다.

def test_no_hint_leaves_the_prompt_untouched():
    from tools.bench_pipeline import apply_length_hint

    assert apply_length_hint("원래 프롬프트", None) == "원래 프롬프트"


def test_hint_is_appended_at_the_end_and_keeps_the_original():
    from tools.bench_pipeline import apply_length_hint

    out = apply_length_hint("원래 프롬프트", "열 글자 이내로 답한다.")

    assert out.startswith("원래 프롬프트")
    assert out.rstrip().endswith("열 글자 이내로 답한다.")


def test_hint_is_separated_so_it_does_not_glue_onto_the_last_rule():
    """붙여 쓰면 마지막 규칙과 한 줄이 돼 둘 다 흐려진다."""
    from tools.bench_pipeline import apply_length_hint

    out = apply_length_hint("...마지막 규칙이다.", "열 글자 이내로 답한다.")

    assert "\n" in out[len("...마지막 규칙이다."):]


def test_empty_hint_is_treated_as_no_hint():
    from tools.bench_pipeline import apply_length_hint

    assert apply_length_hint("원래 프롬프트", "") == "원래 프롬프트"
    assert apply_length_hint("원래 프롬프트", "   ") == "원래 프롬프트"
