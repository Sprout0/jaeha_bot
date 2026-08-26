"""재생 앞 무음(PLAY_PAD_S)이 실제로 얼마나 필요한지 재는 절차. 장치 없이 검증한다.

🔴 왜 재나. `PLAY_PAD_S = 0.15` 는 **한 번도 측정된 적이 없다**. 도입 커밋(5c1892c)이
   '잘림 방지용'으로 넣은 뒤 그대로다(`git log -S` 결과가 그 커밋 하나뿐).
   sounddevice 는 스트림이 실제로 열리기까지 앞쪽 샘플을 흘리고, 그래서 첫 음절이
   잘린다. 그걸 무음으로 흡수하는 값인데 — **그 앞 무음 동안은 아무 소리도 안 난다.**
   지금 첫 소리 721ms 중 150ms(21%)가 이 여유분이다.

측정 방식: 주파수가 각각 다른 짧은 삑을 줄줄이 이어 붙여 **패딩 없이** 재생하고
마이크로 되받는다. 잘려나간 삑은 그 주파수가 녹음에 **아예 없다**.
⚠️ 시간 정렬로 재지 않는 이유: 마이크 경로에도 지연이 있어 '녹음의 0초'와 '재생의
   0초'가 다르다. 주파수 유무로 세면 정렬이 아예 필요 없고, 방 울림·음량 차이에도
   안 흔들린다.
"""
import numpy as np
import pytest

from tools.measure_play_pad import (ANCHOR_HZ, add_anchor, calibrate_noise_floor,
                                    classify_magnitudes, make_probe, present,
                                    present_with_floor, truncation_ms)

SR = 16000


def test_probe_length_matches_the_burst_plan():
    sig, freqs = make_probe(SR, n=20, burst_ms=10.0)

    assert len(freqs) == 20
    assert len(sig) == pytest.approx(20 * 0.010 * SR, abs=2)


def test_each_burst_gets_its_own_frequency():
    _, freqs = make_probe(SR, n=5, f0=600.0, df=200.0)

    assert freqs == [600.0, 800.0, 1000.0, 1200.0, 1400.0]


def test_frequencies_stay_below_nyquist():
    """젯슨 USB 오디오는 16kHz 전용이다. 8kHz 를 넘으면 접혀서 엉뚱하게 잡힌다."""
    _, freqs = make_probe(SR, n=20, f0=600.0, df=200.0)

    assert max(freqs) < SR / 2 * 0.9


# ── 검출 ─────────────────────────────────────────────────────────────────────

def test_all_bursts_are_found_in_the_untouched_signal():
    sig, freqs = make_probe(SR, n=10)

    assert present(sig, freqs, SR) == [True] * 10


def test_pure_silence_finds_nothing():
    _, freqs = make_probe(SR, n=10)

    assert present(np.zeros(SR, dtype=np.float32), freqs, SR) == [False] * 10


def test_a_cut_start_shows_up_as_missing_leading_bursts():
    """🔴 이게 이 도구의 본론이다 — 앞이 잘리면 앞쪽 삑이 사라진다."""
    sig, freqs = make_probe(SR, n=10, burst_ms=10.0)
    cut = sig[int(0.030 * SR):]          # 앞 30ms = 삑 3개 삭제

    got = present(cut, freqs, SR)

    assert got[:3] == [False, False, False], f"잘린 삑이 잡혔다: {got}"
    assert all(got[3:]), f"남아 있는 삑을 놓쳤다: {got}"


def test_detection_survives_noise_and_low_volume():
    """마이크로 되받으면 작아지고 잡음이 낀다. 그 정도로 판정이 흔들리면 못 쓴다."""
    sig, freqs = make_probe(SR, n=10)
    rng = np.random.default_rng(0)
    dirty = (0.05 * sig + 0.002 * rng.standard_normal(len(sig))).astype(np.float32)

    assert present(dirty, freqs, SR) == [True] * 10


# ── 결론 내기 ────────────────────────────────────────────────────────────────

def test_truncation_is_the_run_of_missing_bursts_at_the_front():
    assert truncation_ms([False, False, True, True, True], 10.0) == 20.0


def test_a_gap_in_the_middle_is_not_counted_as_truncation():
    """중간에 하나 안 잡힌 건 검출 실패지 잘림이 아니다. 앞쪽 연속만 센다."""
    assert truncation_ms([True, False, True, True], 10.0) == 0.0


def test_nothing_missing_means_no_padding_needed():
    assert truncation_ms([True] * 10, 10.0) == 0.0


def test_everything_missing_is_reported_as_beyond_the_probe():
    """전부 사라졌으면 잘림이 탐침 길이보다 길다 — '적어도 이만큼'이라고 알려야 한다."""
    assert truncation_ms([False] * 10, 10.0) == 100.0


# ── 기준음: '녹음 실패'와 '진짜 다 잘림'을 가른다 ────────────────────────────
# 🔴 첫 실기 측정이 여기서 틀렸다. sounddevice 의 간편 API(sd.rec/sd.play)는 전역
#    스트림 하나를 공유해서, 재생을 시작하면 녹음이 끊긴다. 그래서 삑을 하나도 못
#    들은 회차가 나왔는데, 도구는 그걸 '200ms 잘림'으로 집계해 최댓값을 오염시켰다.
#    둘은 반드시 구분해야 한다 — 하나는 버릴 데이터고 하나는 진짜 결과다.
# ➡️ 탐침 **끝**에 기준음을 붙인다. 앞이 아무리 잘려도 끝은 남으므로,
#    기준음이 안 들리면 그건 잘림이 아니라 녹음 실패다.

def test_anchor_is_appended_at_the_end():
    sig, _ = make_probe(SR, n=5)
    out = add_anchor(sig, SR, ms=30.0)

    assert len(out) == pytest.approx(len(sig) + 0.030 * SR, abs=2)
    assert np.array_equal(out[: len(sig)], sig), "탐침을 건드렸다"


def test_anchor_survives_a_cut_start():
    sig, freqs = make_probe(SR, n=10, burst_ms=10.0)
    out = add_anchor(sig, SR, ms=30.0)

    cut = out[int(0.050 * SR):]           # 앞 50ms 삭제 = 삑 5개

    got = present(cut, freqs + [ANCHOR_HZ], SR)
    assert got[-1] is True, "기준음은 끝에 있으니 살아 있어야 한다"
    assert got[:5] == [False] * 5


def test_a_failed_recording_loses_the_anchor_too():
    """녹음이 통째로 실패하면 기준음도 없다 — 그때는 그 회차를 버려야 한다."""
    _, freqs = make_probe(SR, n=10)

    got = present(np.zeros(SR, dtype=np.float32), freqs + [ANCHOR_HZ], SR)

    assert got[-1] is False


def test_anchor_frequency_does_not_collide_with_the_bursts():
    _, freqs = make_probe(SR, n=12)

    assert all(abs(f - ANCHOR_HZ) > 150 for f in freqs)


# ── CLI 기본값도 나이퀴스트를 지켜야 한다 ────────────────────────────────────
# 🔴 실기에서 8,200Hz 짜리 삑을 만들어 16kHz 녹음(나이퀴스트 8,000Hz)에서 접혔다.
#    make_probe 만 검사하고 CLI 경로를 안 봐서 놓쳤다.

def test_default_probe_stays_under_the_recording_nyquist():
    from tools.measure_play_pad import REC_RATE

    _, freqs = make_probe(SR)

    assert max(freqs) < REC_RATE / 2 * 0.9, f"녹음에서 접힌다: {max(freqs)}Hz"


# ── 잡음바닥 대비 판정 ────────────────────────────────────────────────────────
# 🔴 2026-08-24 젯슨 실측이 present() 의 결함을 드러냈다. **재생을 전혀 안 했는데도**
#    600Hz·5000Hz·기준음(400Hz)이 8/8 회 '들렸다'로 오판됐다. peak-relative 판정은
#    '이 녹음 안의 최댓값'을 기준으로 삼는데, 마이크 자체잡음은 주파수마다 균일하지
#    않아서(낮은/높은 극단에서 우연히 크다) 잡음이 제일 큰 주파수가 그냥 통과해버린다.
#    심지어 안전장치인 기준음(400Hz)까지 뚫려서, 녹음이 통째로 실패해도 '성공'으로
#    오판할 수 있었다 — 그 다음에 '잘림 없음' 이라는 결론까지 나올 뻔했다.
# ➡️ 각 주파수를 **그 주파수 자신의 사전 측정 잡음바닥**과 비교한다. 잡음바닥이
#    원래 높은 주파수는 문턱도 그만큼 높아져 오판을 막는다.

def test_classify_rejects_a_naturally_noisy_bin_that_looks_like_the_loudest():
    """🔴 젯슨에서 실제로 벌어진 그대로: 잡음바닥이 높은 주파수가 이 녹음의 최댓값이다."""
    mags = [5.0, 0.3, 0.05]     # bin0 = 우연히 큰 잡음 / bin1 = 진짜 약한 톤 / bin2 = 무음
    floor = [4.8, 0.05, 0.05]   # bin0 은 원래 잡음바닥이 높다(마이크 자체특성)

    got = classify_magnitudes(mags, floor, margin_db=12.0)

    assert got == [False, True, False], (
        "peak-relative 였다면 bin0(최댓값)이 통과하고 bin1(진짜 톤)은 떨어졌을 것이다")


def test_classify_finds_a_weak_real_tone_even_when_peak_is_just_noise():
    """구 방식(peak-relative)이 정확히 여기서 틀렸다 — 재현해서 대조한다."""
    mags = [5.0, 0.3, 0.05]
    peak = max(mags)
    old_way = [20 * __import__("math").log10(m / peak + 1e-12) > -22.0 for m in mags]
    assert old_way == [True, False, False], "구 방식이 진짜 재연되는지 확인(대조군)"

    new_way = classify_magnitudes(mags, [4.8, 0.05, 0.05], margin_db=12.0)
    assert new_way == [False, True, False], "새 방식은 반대로 판정해야 한다"


def test_classify_without_floor_falls_back_to_peak_relative():
    """기존 present() 와 같은 동작을 보존한다(마른 신호로 만든 합성 테스트들이 이미 이걸 검증)."""
    mags = [1.0, 0.5, 0.01]

    got = classify_magnitudes(mags, None)

    assert got == [True, True, False]


def test_classify_zero_floor_does_not_crash():
    """잡음바닥이 0(진짜 완전 무음 측정)이어도 0 으로 나누면 안 된다.

    0 을 넘는 값(0.1)만 통과해야 한다 — 0 자체는 0 보다 크지 않으니 여전히 무음이다.
    """
    got = classify_magnitudes([0.1, 0.0], [0.0, 0.0], margin_db=12.0)

    assert got == [True, False]


def test_present_with_floor_uses_the_calibrated_reference():
    """파형 입력 경로도 classify_magnitudes 와 같은 결론을 내야 한다."""
    sig, freqs = make_probe(SR, n=3, f0=600.0, df=1000.0)  # 600/1600/2600Hz, 진짜 3톤
    quiet = 0.001 * np.ones(len(sig), dtype=np.float32)     # 잡음바닥 계산용 무음 근사

    floor = calibrate_noise_floor(quiet, freqs, SR)
    got = present_with_floor(sig, freqs, SR, floor, margin_db=12.0)

    assert got == [True, True, True], "진짜 톤 셋 다 낮은 잡음바닥 위에서 잡혀야 한다"


def test_present_with_floor_rejects_pure_silence_against_a_real_calibration():
    freqs = [600.0, 1600.0, 2600.0]
    quiet_a = (0.001 * np.random.default_rng(1).standard_normal(SR)).astype(np.float32)
    quiet_b = (0.001 * np.random.default_rng(2).standard_normal(SR)).astype(np.float32)

    floor = calibrate_noise_floor(quiet_a, freqs, SR)
    got = present_with_floor(quiet_b, freqs, SR, floor, margin_db=12.0)

    assert got == [False, False, False], "잡음 대 잡음은 문턱을 넘으면 안 된다"


# ── 꼬리 잘림 ────────────────────────────────────────────────────────────────
# 🔴 2026-08-26 젯슨 청취: 필러도 답변도 **말끝이 뚝 끊긴다**. 노트북에서는 파형·재생
#    어느 쪽에도 잘릴 자리가 없었다(버퍼 끝 무음 485~798ms, 콜백 미소비 0프레임).
#    남은 건 젯슨의 출력 경로뿐이라, 앞이 아니라 **뒤**를 재는 짝이 필요하다.

from tools.measure_play_pad import tail_truncation_ms


def test_tail_truncation_is_the_run_of_missing_bursts_at_the_back():
    assert tail_truncation_ms([True, True, False, False], 15.0) == 30.0


def test_a_gap_in_the_middle_is_not_counted_as_tail_truncation():
    """중간 구멍은 검출 실패지 잘림이 아니다 — 앞쪽 짝과 같은 규칙."""
    assert tail_truncation_ms([True, False, True, True], 15.0) == 0.0


def test_nothing_missing_at_the_back_means_the_tail_survived():
    assert tail_truncation_ms([True, True, True], 15.0) == 0.0


def test_everything_missing_is_the_whole_probe():
    assert tail_truncation_ms([False, False, False], 15.0) == 45.0


def test_head_and_tail_are_counted_independently():
    """앞뒤가 함께 잘린 녹음에서 서로를 오염시키면 안 된다."""
    from tools.measure_play_pad import truncation_ms

    found = [False, True, True, False]

    assert truncation_ms(found, 10.0) == 10.0
    assert tail_truncation_ms(found, 10.0) == 10.0


def test_the_anchor_can_go_at_the_front_instead():
    """꼬리를 잴 때는 기준음이 **앞**에 있어야 한다 — 뒤에 두면 같이 잘려서,
    '녹음 실패'와 '진짜 다 잘림'을 가르는 장치가 그 순간 무용지물이 된다."""
    sig = np.ones(100, dtype=np.float32)

    out = add_anchor(sig, 16000, ms=2.0, at_end=False)

    m = int(round(16000 * 2.0 / 1000))
    assert out.size == sig.size + m
    assert np.array_equal(out[m:], sig), "원래 신호가 뒤에 그대로 있어야 한다"
