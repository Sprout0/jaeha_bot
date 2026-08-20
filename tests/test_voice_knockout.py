"""블라인드 토너먼트가 '공정한 비교'인지.

이 도구의 결론은 귀로 고른 목소리 하나다. 그래서 다음이 깨지면 결론이 통째로 무의미해진다:
- 후보가 실제로 합성 가능한 문자열이 아니면 (판이 진행되다 터진다)
- 현행 목소리가 후보에 없으면 ('바꿀 이유가 없다'는 결론이 아예 안 나온다)
- 판을 나눌 때 후보가 새거나 겹치면 (진 후보가 올라가거나 이긴 후보가 사라진다)
"""
import random

import pytest

from app.tts_module import TTSModule
from tools.voice_knockout import make_heats, make_pool, random_spec


def _parseable(spec: str) -> None:
    """운영 코드가 실제로 읽는 경로로 검사한다 — 여기서 통과하면 합성도 된다."""
    m = TTSModule()
    for side in spec.split("|"):
        parts = m._parse_blend(side)
        assert parts
        assert abs(sum(w for _, w in parts) - 1.0) < 1e-6, f"가중치 합이 1 이 아니다: {spec}"
        for name, _ in parts:
            assert name[0] in "FM" and name[1:].isdigit(), f"없는 프리셋: {name} in {spec}"


@pytest.mark.parametrize("seed", range(30))
def test_random_specs_are_always_synthesisable(seed):
    _parseable(random_spec(random.Random(seed)))


def test_pool_always_contains_the_current_voice():
    """대조군이 빠지면 '현행이 제일 낫다'는 답이 나올 수 없다."""
    pool = make_pool(12, random.Random(0), "F1+F4")

    assert "F1+F4" in pool


def test_pool_has_no_duplicates():
    """같은 목소리가 두 번 나오면 무승부 통계가 오염된다."""
    pool = make_pool(15, random.Random(1), "F1+F4")

    assert len(pool) == len(set(pool))


def test_heats_cover_every_candidate_exactly_once():
    pool = [f"c{i}" for i in range(15)]
    heats = make_heats(pool, 3)
    flat = [c for h in heats for c in h]

    assert sorted(flat) == sorted(pool), "후보가 새거나 겹쳤다"


@pytest.mark.parametrize("n,k", [(15, 3), (9, 2), (7, 3), (4, 3), (2, 2), (10, 3)])
def test_heats_are_never_bigger_than_k(n, k):
    heats = make_heats([f"c{i}" for i in range(n)], k)

    assert all(1 <= len(h) <= k for h in heats)
    assert sum(len(h) for h in heats) == n


def test_tournament_terminates():
    """이긴 것만 올리므로 후보는 매 회전 줄어야 한다 — 안 줄면 무한 루프다."""
    pool = [f"c{i}" for i in range(15)]
    rounds = 0
    while len(pool) > 1:
        before = len(pool)
        pool = [h[0] for h in make_heats(pool, 3)]      # 항상 첫 번째가 이긴다고 가정
        assert len(pool) < before, "회전을 돌아도 후보가 안 줄었다"
        rounds += 1
        assert rounds < 20
    assert len(pool) == 1


# ── 후보를 직접 넣기 ──────────────────────────────────────────────────────────
# 사람이 "F1 과 F4 가 좋더라" 까지 좁힌 뒤에는 무작위 후보가 오히려 방해가 된다.
# 그때는 손으로 고른 목록을 **같은 블라인드 절차**에 태워야 한다 — 목록을 하나씩
# 듣는 절대평가로 돌아가면 2026-08-18 의 "솔직히 잘 모르겠어"를 반복한다.

def test_given_specs_are_used_instead_of_random():
    rng = random.Random(0)
    given = ["F1:0.6+F4:0.6+F3:-0.2", "F1:0.6+F4:0.6+M1:-0.2"]

    pool = make_pool(99, rng, "F1+F4", specs=given)

    assert set(pool) == set(given) | {"F1+F4"}, "준 목록 말고 다른 게 섞였다"


def test_control_is_still_smuggled_in_when_specs_are_given():
    """🔴 현행 목소리는 손으로 고른 목록에도 반드시 들어가야 한다.

    블라인드에서 현행이 이기면 '바꿀 이유가 없다'가 결론이고, 그것도 결과다.
    사람이 고른 목록만 돌리면 그 결론이 나올 길 자체가 없어진다.
    """
    pool = make_pool(99, random.Random(0), "현행목소리", specs=["F1+F4"])

    assert "현행목소리" in pool


def test_control_is_not_duplicated_if_already_listed():
    pool = make_pool(99, random.Random(0), "F1+F4", specs=["F1+F4", "F2"])

    assert sorted(pool) == ["F1+F4", "F2"]


def test_duplicates_in_the_given_list_collapse():
    pool = make_pool(99, random.Random(0), "F1+F4", specs=["F2", "F2", "F3"])

    assert sorted(pool) == ["F1+F4", "F2", "F3"]


def test_random_pool_still_works_when_no_specs_given():
    pool = make_pool(6, random.Random(1), "F1+F4")

    assert len(pool) == 6 and "F1+F4" in pool
