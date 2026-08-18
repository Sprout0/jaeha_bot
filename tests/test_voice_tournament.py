"""토너먼트가 단계마다 **한 축만** 바꾸는지.

이 도구의 결론은 '어느 목소리가 아이에게 좋은가'다. 한 단계에서 두 축을 동시에 바꾸면
(예: 리듬과 속도를 같이) 무엇 때문에 좋아졌는지 알 수 없고, 고른 값도 재현이 안 된다.
호출어 v3 때 증강 축이 붙어 있어 실패 원인을 못 가렸던 것과 같은 실수다.
"""
import pytest

from tools.voice_tournament import PRESETS, build_stage


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
