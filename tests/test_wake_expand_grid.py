"""증강 격자가 지켜야 할 성질. 오디오·모델 불필요.

왜 필요한가 — 이 격자는 두 번 틀렸고 두 번 다 **실기에서야** 알았다:
  v3: 리샘플 하나로 만들어 '빠르다'와 '높다'가 붙어 다녔다 -> 모델이 피치를 골랐다.
  v4: 피치를 **올리는 쪽으로만** 줘서 성인 남성(104~116Hz)이 분포의 얇은 꼬리에 남았다.
격자는 코드가 아니라 상수라 리뷰에서 눈으로 넘어가기 쉽다. 성질을 못으로 박아 둔다.
"""
import numpy as np

from tools.gen_wake_supertonic import (EXPAND_GRID, EXPAND_PER_CLIP, pitch_shift,
                                       time_stretch)

RATES = {r for r, _ in EXPAND_GRID}
PITCHES = {p for _, p in EXPAND_GRID}


def test_axes_are_independent():
    """🔴 v3 실패. 대각선만 있으면 모델이 두 축을 구분할 수 없다.

    '빠르지만 안 높은' 칸과 '느리지만 높은' 칸이 둘 다 있어야 한다.
    """
    assert any(r > 1.0 and p == 1.0 for r, p in EXPAND_GRID), "빠르기만 한 칸이 없다"
    assert any(r == 1.0 and p != 1.0 for r, p in EXPAND_GRID), "피치만 바뀐 칸이 없다"


def test_pitch_goes_both_ways():
    """🔴 v4 실패. 올리는 쪽만 있으면 낮은 목소리가 분포 밖에 남는다.

    실측(2026-08-12): 학습 긍정 F0 중앙 165Hz vs 성인 남성 104~116Hz.
    녹음을 +15~35% 올리면 점수가 살아났다 = 학습은 0.87·0.74 쪽을 채워야 한다.
    """
    assert any(p > 1.0 for p in PITCHES), "피치를 올리는 칸이 없다(아이 음역)"
    assert any(p < 1.0 for p in PITCHES), "피치를 내리는 칸이 없다(성인 남성 음역)"


def test_low_pitch_range_matches_measurement():
    """내리는 칸이 실측 최적 구간(1/1.15≈0.87, 1/1.35≈0.74)을 덮어야 한다."""
    low = sorted(p for p in PITCHES if p < 1.0)
    assert low, "내리는 칸이 없다"
    assert min(low) <= 0.78, f"충분히 낮은 칸이 없다: {low}"
    assert max(low) >= 0.82, f"완만하게 낮춘 칸이 없다: {low}"


def test_low_pitch_is_crossed_with_speed():
    """낮은 목소리로 '빠르게' 부르는 경우도 있어야 두 축이 함께 학습된다."""
    assert any(r > 1.0 and p < 1.0 for r, p in EXPAND_GRID), \
        "낮은 피치 × 빠른 속도 칸이 없다"


def test_grid_has_no_duplicates_and_no_identity():
    assert len(EXPAND_GRID) == len(set(EXPAND_GRID)), "중복 칸이 있다"
    assert (1.0, 1.0) not in EXPAND_GRID, "원본과 같은 칸은 증강이 아니다"


def test_per_clip_draw_keeps_total_count_stable():
    """개수가 v3·v4(원본×4)와 같아야 학습 시간이 같고 비교가 성립한다."""
    assert EXPAND_PER_CLIP == 3
    assert EXPAND_PER_CLIP <= len(EXPAND_GRID), "격자보다 많이 뽑을 수 없다"


def test_time_stretch_actually_changes_length():
    """🔴 WSOLA 는 탐색 반경을 잘못 잡으면 **예외 없이** 아무 일도 안 한다."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    for rate in sorted(r for r in RATES if r != 1.0):
        got = time_stretch(x, rate).size / x.size
        assert abs(got - 1.0 / rate) < 0.08, \
            f"배속 {rate}: 길이비 {got:.3f} (기대 {1/rate:.3f})"


def test_pitch_shift_keeps_length():
    """피치 칸은 길이를 바꾸면 안 된다 — 바꾸면 시간축과 다시 섞인다."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    for p in sorted(v for v in PITCHES if v != 1.0):
        got = pitch_shift(x, p).size / x.size
        assert abs(got - 1.0) < 0.08, f"피치 {p}: 길이가 {got:.3f} 배로 변했다"


def test_pitch_shift_moves_pitch_in_the_right_direction():
    """0.85 는 낮추고 1.15 는 높여야 한다. 부호가 뒤집히면 v4 실패를 반복한다."""
    sr = 16000
    t = np.arange(sr) / sr
    x = ((2 * ((150 * t) % 1.0) - 1.0) * 0.3).astype(np.float32)

    def f0(y):
        n = 2048
        seg = y[len(y) // 2 - n // 2:len(y) // 2 + n // 2].astype(np.float64)
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[n - 1:]
        lo, hi = sr // 400, sr // 70
        return sr / (int(np.argmax(ac[lo:hi])) + lo)

    assert f0(pitch_shift(x, 0.85)) < 150 < f0(pitch_shift(x, 1.15))
