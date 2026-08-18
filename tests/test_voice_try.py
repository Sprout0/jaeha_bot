"""'주변 훑기'가 진짜로 **기준값 주변**인지.

🔴 이게 이 도구의 존재 이유다. voice_tournament 의 --explore 1 은 고정된 비율
   (0.15/0.2/0.4)로 후보를 만들어서, 기준이 'F3:-0.2+F1:1.2' 처럼 외삽 지점일 때
   0.3 근처를 훑는다 — 우승한 자리에서 멀어지는 쪽만 들려준다.
   주변 탐색은 언제나 '지금 값에서 ±' 여야 한다.
"""
import pytest

from app.tts_module import TTSModule
from tools.voice_try import neighbors, rhythm_variants


def _w(spec: str) -> dict[str, float]:
    return dict(TTSModule()._parse_blend(spec.split("|")[0]))


def test_neighbors_are_centred_on_the_base():
    base = "F3:-0.2+F1:1.2"
    got = [_w(s)["F3"] for s in neighbors(base)]

    assert all(abs(w - (-0.2)) <= 0.2 + 1e-9 for w in got), \
        f"기준 -0.2 에서 멀리 떨어진 후보가 있다: {got}"
    assert min(got) < -0.2 < max(got), "한쪽 방향으로만 움직였다"


def test_neighbors_keep_weights_summing_to_one():
    for s in neighbors("F3:-0.2+F1:1.2"):
        assert sum(_w(s).values()) == pytest.approx(1.0)


def test_neighbors_keep_the_same_presets():
    """주변을 본다면서 다른 목소리를 끌어오면 그건 주변이 아니다."""
    for s in neighbors("F3:-0.2+F1:1.2"):
        assert set(_w(s)) == {"F3", "F1"}


def test_neighbors_preserve_the_rhythm_side():
    for s in neighbors("F3:-0.2+F1:1.2|F4"):
        assert s.endswith("|F4"), f"리듬이 사라졌다: {s}"


def test_neighbors_keep_third_component_proportional():
    """3개 조합에서도 나머지 비율이 유지돼야 '한 축만 움직였다'가 된다."""
    for s in neighbors("F1:0.5+F2:0.3+M3:0.2"):
        w = _w(s)
        assert w["F2"] / w["M3"] == pytest.approx(0.3 / 0.2)


def test_neighbors_never_returns_the_base_itself():
    base = "F3:-0.2+F1:1.2"

    assert base not in neighbors(base)


def test_single_preset_has_no_neighbours():
    """F1 하나뿐이면 움직일 축이 없다 — 조용히 이상한 값을 만들면 안 된다."""
    assert neighbors("F1") == []


def test_rhythm_variants_hold_the_timbre():
    for s in rhythm_variants("F3:-0.2+F1:1.2|F5"):
        assert s.split("|")[0] == "F3:-0.2+F1:1.2"


@pytest.mark.parametrize("base", ["F3:-0.2+F1:1.2", "F1+F4", "F1:0.5+F2:0.3+M3:0.2",
                                  "F2:0.7+F5:0.3|F1"])
def test_every_neighbour_is_synthesisable(base):
    m = TTSModule()
    for s in neighbors(base) + rhythm_variants(base):
        for side in s.split("|"):
            assert m._parse_blend(side)
