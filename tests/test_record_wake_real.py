"""실음성 진단 도구의 계산부 검증. 마이크·모델 불필요.

왜 필요한가 (2026-08-10):
  이 도구는 "음역 문제냐 발음 문제냐"를 가르는 데 쓴다. 판정이 틀리면 다음 학습 데이터를
  통째로 잘못 만든다. 특히 score() 의 무음 덧대기는 빠뜨려도 예외가 안 나고
  **점수가 전부 0.000 으로 조용히 죽는다**(실제로 배선 점검 때 겪었다).
"""
import numpy as np

from tools.record_wake_real import FRAME, PAD_S, SR, f0_median, score


class _FakeDet:
    """25프레임을 채우기 전에는 None 을 돌려주는 진짜 감지기의 성질만 흉내낸다."""

    WARMUP = 25

    def __init__(self):
        self.n = 0
        self.seen = 0

    def reset(self):
        self.n = 0

    def push(self, frame):
        self.n += 1
        self.seen += 1
        if self.n < self.WARMUP:
            return None
        return float(np.abs(frame).max())


def test_score_pads_so_short_clips_still_get_scored():
    """1초짜리(12프레임) 클립도 점수가 나와야 한다 — 덧대기가 빠지면 0.0 이 된다."""
    det = _FakeDet()
    y = np.full(SR, 0.5, dtype=np.float32)
    assert score(det, y) > 0.0, "무음 덧대기가 빠져 워밍업을 못 채웠다"


def test_score_padding_is_silent_so_it_does_not_raise_the_score():
    """덧댄 구간은 무음이어야 한다. 소리가 섞이면 점수를 부풀린다."""
    det = _FakeDet()
    y = np.full(SR, 0.5, dtype=np.float32)
    assert abs(score(det, y) - 0.5) < 1e-6


def test_score_resets_between_calls():
    """이전 발화의 상태가 남으면 다음 클립 점수가 오염된다."""
    det = _FakeDet()
    y = np.full(SR, 0.5, dtype=np.float32)
    first = score(det, y)
    assert abs(score(det, y) - first) < 1e-6


def test_pad_covers_warmup():
    """앞 무음만으로 워밍업(25프레임=2.0s)이 채워져야 클립의 첫 음절부터 채점된다."""
    assert PAD_S * SR / FRAME >= 25


def test_f0_reads_a_known_pitch():
    t = np.arange(SR) / SR
    saw = ((2 * ((150 * t) % 1.0) - 1.0) * 0.3).astype(np.float32)
    assert abs(f0_median(saw) - 150) < 8


def test_f0_separates_male_from_child_range():
    """판정이 이 대소 관계에 기대므로 뒤집히지 않는지 본다."""
    t = np.arange(SR) / SR
    def saw(f):
        return ((2 * ((f * t) % 1.0) - 1.0) * 0.3).astype(np.float32)
    assert f0_median(saw(110)) < f0_median(saw(300))


def test_f0_returns_zero_on_silence():
    """무음에서 억지 값을 내면 F0 표가 거짓말을 한다."""
    assert f0_median(np.zeros(SR, dtype=np.float32)) == 0.0
