"""토너먼트가 단계마다 **한 축만** 바꾸는지.

이 도구의 결론은 '어느 목소리가 아이에게 좋은가'다. 한 단계에서 두 축을 동시에 바꾸면
(예: 리듬과 속도를 같이) 무엇 때문에 좋아졌는지 알 수 없고, 고른 값도 재현이 안 된다.
호출어 v3 때 증강 축이 붙어 있어 실패 원인을 못 가렸던 것과 같은 실수다.
"""
import pytest

from tools.voice_tournament import PRESETS, build_explore, build_stage


def test_stage1_is_plain_presets():
    _, c = build_stage(1, {})

    assert [x["voice"] for x in c] == PRESETS
    assert len({x["seed"] for x in c}) == 1, "1단계에서 시드가 흔들리면 음색 비교가 아니다"
    assert len({x["speed"] for x in c}) == 1


def test_stage2_blends_only_the_stage1_picks():
    _, c = build_stage(2, {"stage1": ["F2", "F5"]})
    voices = [x["voice"] for x in c]

    assert "F2:0.7+F5:0.3" in voices and "F2:0.3+F5:0.7" in voices
    assert "F2+F5" in voices
    for v in voices:
        assert set(v.replace(":", "+").split("+")) <= {"F2", "F5", "0.7", "0.3"}, \
            f"고르지 않은 프리셋이 섞였다: {v}"


def test_stage2_keeps_singles_as_control():
    """블렌드가 단일보다 낫다는 보장이 없다. 대조군이 없으면 그걸 모른다."""
    _, c = build_stage(2, {"stage1": ["F2", "F5"]})

    assert "F2" in [x["voice"] for x in c] and "F5" in [x["voice"] for x in c]


def test_stage3_holds_timbre_and_varies_rhythm():
    _, c = build_stage(3, {"stage2": "F2:0.7+F5:0.3"})

    for x in c:
        timbre, _, rhythm = x["voice"].partition("|")
        assert timbre == "F2:0.7+F5:0.3", "3단계에서 음색이 바뀌면 안 된다"
        assert rhythm
    assert len({x["voice"].split("|")[1] for x in c}) == len(c), "리듬 후보가 중복이다"


def test_stage4_varies_seed_only():
    _, c = build_stage(4, {"stage3": "F2|F4"})

    assert len({x["seed"] for x in c}) == len(c)
    assert len({x["voice"] for x in c}) == 1
    assert len({x["speed"] for x in c}) == 1


def test_stage5_varies_speed_only_and_keeps_earlier_picks():
    _, c = build_stage(5, {"stage3": "F2|F4", "stage4_seed": 42})

    assert len({x["speed"] for x in c}) == len(c)
    assert {x["voice"] for x in c} == {"F2|F4"}
    assert {x["seed"] for x in c} == {42}, "4단계에서 고른 시드가 5단계로 안 넘어왔다"


def test_unknown_stage_refuses():
    with pytest.raises(SystemExit):
        build_stage(9, {})


def test_explore1_keeps_current_pick_as_control():
    """더 밀어붙인 게 오히려 나쁠 수 있다. 기준이 없으면 그걸 모른다."""
    _, c = build_explore(1, "F1:0.3+F4:0.7", {})

    assert c[0]["voice"] == "F1:0.3+F4:0.7"


def test_explore1_goes_outside_the_presets():
    """외삽(음수 가중치)이 없으면 프리셋 '사이'만 뒤지게 된다."""
    _, c = build_explore(1, "F1:0.3+F4:0.7", {})
    voices = [x["voice"] for x in c]

    assert any(":-" in v for v in voices), "프리셋 밖으로 나가는 후보가 없다"
    assert len({x["voice"] for x in c}) == len(c), "후보가 중복이다"


def test_explore2_adds_a_third_voice_including_male():
    _, c = build_explore(2, "F1:0.3+F4:0.7", {})
    added = [x["voice"].rsplit("+", 1)[-1] for x in c[1:]]

    assert any(a.startswith("M") for a in added), "M 계열이 빠졌다(두께를 못 얻는다)"
    assert all(a.endswith(":0.2") for a in added), "제3의 목소리는 소량이어야 한다"
    assert not any(a.startswith(("F1:", "F4:")) for a in added), \
        "이미 들어 있는 목소리를 또 더하고 있다"


def test_explore_varies_nothing_but_voice():
    for rnd in (1, 2):
        _, c = build_explore(rnd, "F1:0.3+F4:0.7", {})
        assert len({x["seed"] for x in c}) == 1
        assert len({x["speed"] for x in c}) == 1


def test_unknown_explore_round_refuses():
    with pytest.raises(SystemExit):
        build_explore(9, "F1+F4", {})


def test_explore1_is_centred_on_the_given_base():
    """외삽 지점이 기준일 때 0.3 근처를 훑으면 우승한 자리에서 멀어지기만 한다."""
    from app.tts_module import TTSModule

    _, c = build_explore(1, "F3:-0.2+F1:1.2", {})
    got = [dict(TTSModule()._parse_blend(x["voice"]))["F3"] for x in c[1:]]

    assert all(abs(w - (-0.2)) <= 0.45 + 1e-9 for w in got), f"기준에서 멀다: {got}"
    assert min(got) < -0.2 < max(got), "한쪽으로만 움직였다"


# ── --from 을 3~5단계에도 ─────────────────────────────────────────────────────
# 🔴 2026-08-19: 블라인드 토너먼트(voice_knockout)로 우승자를 뽑고 나면, 그 값은
#    이 도구의 상태 파일에 없다. 그런데 시드·속도 단계는 저장된 stage2/stage3 만
#    보고 있어서 우승자를 넘길 방법이 없었다 — 손으로 상태 파일을 고치는 수밖에.
#    `--from` 은 이미 있는데 --explore 에만 걸려 있었다. 뜻 그대로 '기준 목소리'다.

def test_stage3_can_take_an_explicit_base():
    _, c = build_stage(3, {"stage2": "저장된값"}, base="F2")

    assert all(x["voice"].startswith("F2") for x in c), "--from 을 무시하고 저장값을 썼다"


def test_stage4_can_take_an_explicit_base():
    _, c = build_stage(4, {"stage3": "저장된값"}, base="F2:0.85+F5:0.15")

    assert {x["voice"] for x in c} == {"F2:0.85+F5:0.15"}
    assert len({x["seed"] for x in c}) == len(c), "시드 단계인데 시드가 안 흔들린다"


def test_stage5_can_take_an_explicit_base():
    _, c = build_stage(5, {"stage3": "저장된값"}, base="F2|F1")

    assert {x["voice"] for x in c} == {"F2|F1"}
    assert len({x["speed"] for x in c}) == len(c), "속도 단계인데 속도가 안 흔들린다"


def test_saved_state_is_still_used_when_no_base_given():
    """--from 을 안 주면 예전처럼 저장된 결과를 따라간다."""
    _, c = build_stage(4, {"stage3": "F2|F4"})

    assert {x["voice"] for x in c} == {"F2|F4"}
