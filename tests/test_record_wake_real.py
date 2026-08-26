"""실음성 진단 도구의 계산부 검증. 마이크·모델 불필요.

왜 필요한가 (2026-08-10):
  이 도구는 "음역 문제냐 발음 문제냐"를 가르는 데 쓴다. 판정이 틀리면 다음 학습 데이터를
  통째로 잘못 만든다. 특히 score() 의 무음 덧대기는 빠뜨려도 예외가 안 나고
  **점수가 전부 0.000 으로 조용히 죽는다**(실제로 배선 점검 때 겪었다).
"""
import numpy as np

from tools.record_wake_real import (FRAME, PAD_S, SILENT_RMS, SR, f0_median,
                                    rms, score)


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


# ── 무음 관문 ────────────────────────────────────────────────────────────
# 🔴 2026-08-12: 이 관문이 없어서 무음 20개를 받아 놓고 '발음 문제'라는 결론까지 냈다.
#    젯슨 기본 입력(35번)은 에러 없이 0.0 만 주기 때문에 예외로는 절대 안 잡힌다.

def test_silence_threshold_sits_between_dead_mic_and_quiet_room():
    """죽은 마이크(0.0)와 조용한 방(실측 0.0035) 사이에 선이 있어야 한다."""
    assert 0.0 < SILENT_RMS < 0.0035


def test_rms_flags_dead_mic_and_passes_quiet_room():
    assert rms(np.zeros(SR, dtype=np.float32)) < SILENT_RMS
    rng = np.random.default_rng(0)
    room = (rng.standard_normal(SR) * 0.0035).astype(np.float32)
    assert rms(room) > SILENT_RMS


def test_rms_handles_empty_input():
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


def _fake_sd(monkeypatch, value):
    import sounddevice as sd
    monkeypatch.setattr(
        sd, "rec",
        lambda n, **k: np.full((n, 1), value, dtype=np.float32))


def test_check_mic_aborts_when_mic_is_dead(monkeypatch):
    """무음이면 20번 부르게 하기 전에 여기서 멈춰야 한다."""
    import pytest

    from tools.record_wake_real import check_mic
    _fake_sd(monkeypatch, 0.0)
    with pytest.raises(SystemExit) as e:
        check_mic(0)
    assert "봇" in str(e.value), "무엇을 하라는 안내가 있어야 한다"


def test_check_mic_passes_on_quiet_room(monkeypatch):
    from tools.record_wake_real import check_mic
    _fake_sd(monkeypatch, 0.0035)
    check_mic(0)          # 예외가 안 나면 통과


def test_check_mic_explains_when_device_is_busy(monkeypatch):
    """봇이 마이크를 쥐고 있으면 열기 자체가 실패한다 — 그 경우도 안내해야 한다."""
    import pytest
    import sounddevice as sd

    from tools.record_wake_real import check_mic

    def boom(*a, **k):
        raise RuntimeError("Device unavailable")

    monkeypatch.setattr(sd, "rec", boom)
    with pytest.raises(SystemExit) as e:
        check_mic(0)
    assert "봇" in str(e.value)


# ══════════════════════════════════════════════════════════════════════
#  아이 모드 (2026-08-26)
# ══════════════════════════════════════════════════════════════════════
# 🔴 왜 여기에 테스트가 필요한가: 아이 모드는 **한 번 녹음하면 다시 못 찍는다.**
#    2세를 다시 앉히는 비용이 크고, 구간 분리가 틀리면 그 세션이 통째로 버려진다.
#    마이크 없이 확인할 수 있는 건 전부 여기서 확인한다.

from tools.record_wake_real import (ADULT_CONDITIONS, CHILD_ELICIT,  # noqa: E402
                                    F0_MAX_ADULT, F0_MAX_CHILD,
                                    segment_utterances)


def _tone(f: float, sec: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return ((2 * ((f * t) % 1.0) - 1.0) * amp).astype(np.float32)


def _quiet(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def test_segments_split_on_long_gaps():
    """1초 쉬면 다른 발화다 — 두 번 부른 걸 하나로 묶으면 표본이 반으로 준다."""
    y = np.concatenate([_quiet(1.0), _tone(300, 0.8), _quiet(1.0),
                        _tone(300, 0.8), _quiet(0.5)])
    assert len(segment_utterances(y)) == 2


def test_segments_merge_across_short_pauses():
    """아이 내부 쉼 p90 은 0.32s(2026-08-24 실측). '하이…티드'가 갈리면 안 된다."""
    y = np.concatenate([_quiet(0.5), _tone(300, 0.5), _quiet(0.32),
                        _tone(300, 0.5), _quiet(0.5)])
    assert len(segment_utterances(y)) == 1


def test_segments_drop_short_clicks_even_though_padding_lengthens_them():
    """🔴 길이 판정은 덧대기 **전**에. 0.1초 기침도 앞뒤 0.3초가 붙으면 0.7초가 된다."""
    y = np.concatenate([_quiet(1.0), _tone(300, 0.08), _quiet(1.0)])
    assert segment_utterances(y) == []


def test_segments_pad_so_the_first_consonant_survives():
    """앞을 바짝 자르면 감지기가 못 잡는다 — 모델 탓이 아니라 녹음 탓이 된다."""
    y = np.concatenate([_quiet(1.0), _tone(300, 0.8), _quiet(1.0)])
    (s0, s1), = segment_utterances(y)
    assert s0 < SR * 1.0, "발화 시작 앞에 여유가 없다"
    assert s1 > SR * 1.8, "발화 끝 뒤에 여유가 없다"


def test_segments_handle_silence_and_empty():
    assert segment_utterances(np.zeros(0, dtype=np.float32)) == []
    assert segment_utterances(_quiet(3.0)) == []


def test_segment_threshold_follows_the_measured_noise_floor():
    """시끄러운 방에서 바닥이 높으면 그 위만 발화로 봐야 한다."""
    rng = np.random.default_rng(0)
    noisy = (rng.standard_normal(int(3 * SR)) * 0.02).astype(np.float32)
    assert segment_utterances(noisy, noise_floor=0.02) == []


# ── F0 상한 ──────────────────────────────────────────────────────────
# 🔴 재하 F0 는 이 프로젝트에서 **한 번도 측정된 적이 없다.** 그런데 EXPAND_GRID 의
#    아이 음역 칸(43%)이 그 미측정 값 위에 서 있다. 처음 재는 값이 천장에 눌리면
#    다음 학습 데이터를 통째로 잘못 만든다.

def test_child_ceiling_is_above_adult_ceiling():
    assert F0_MAX_CHILD > F0_MAX_ADULT >= 400


def test_excited_child_pitch_needs_the_raised_ceiling():
    """500Hz 는 흥분한 2세에게 흔하다. 성인 상한(400)으로는 못 읽는다."""
    saw = _tone(500, 1.0)
    assert abs(f0_median(saw, f_max=F0_MAX_CHILD) - 500) < 25
    assert abs(f0_median(saw, f_max=F0_MAX_ADULT) - 500) > 25, \
        "성인 상한에서도 읽힌다면 이 상한 구분은 의미가 없다"


def test_adult_range_still_reads_the_same_with_either_ceiling():
    """상한을 올린 게 성인 음역 판독을 망치면 부모 녹음이 틀어진다."""
    saw = _tone(110, 1.0)
    assert abs(f0_median(saw, f_max=F0_MAX_CHILD)
               - f0_median(saw, f_max=F0_MAX_ADULT)) < 5


# ── 대본 ─────────────────────────────────────────────────────────────

def test_no_stale_wake_word_anywhere_in_the_prompts():
    """🔴 08-25 에 호출어를 바꿨는데 이 대본에 '재하봇'이 남아 있었다(08-26 발견).
    부모가 그대로 읽으면 **틀린 호출어로 녹음한 세션 하나를 통째로 버린다.**"""
    for text in [g for _, g in ADULT_CONDITIONS] + CHILD_ELICIT:
        assert "재하" not in text and "잰봇" not in text, text


def test_child_script_tells_the_parent_not_the_child():
    """2세는 '빠르게 말하되 다 발음하라'를 수행할 수 없다. 조건은 부모가 만들어 준다."""
    assert len(CHILD_ELICIT) >= 4
    assert any("깨워" in s for s in CHILD_ELICIT), "빠르고 높은 소리를 끌어낼 유도가 필요"
    assert any("멀리" in s for s in CHILD_ELICIT), "실제 호출과 가장 비슷한 조건"
