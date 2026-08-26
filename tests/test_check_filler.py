"""젯슨 필러 점검 도구의 판정 로직.

장치가 필요한 부분은 여기서 못 잰다. 대신 **판정이 틀리지 않는지**를 지킨다 —
젯슨에서 나온 숫자를 이 함수들이 잘못 읽으면 멀쩡한 걸 기각하거나 그 반대가 된다.
"""
import numpy as np
import pytest

from tools.check_filler import (ANSWER_AT_FAST_S, ANSWER_AT_TYPICAL_S,
                                audible_span, overlap_verdict)


def _buf(rate=16000, pad_s=0.15, speech_s=0.6, tail_s=0.4):
    """앞무음 + 소리 + 뒤무음. 필러 캐시 파일과 같은 모양."""
    return np.concatenate([
        np.zeros(int(pad_s * rate), dtype=np.float32),
        np.full(int(speech_s * rate), 0.3, dtype=np.float32),
        np.zeros(int(tail_s * rate), dtype=np.float32),
    ])


# ── 들리는 구간 찾기 ─────────────────────────────────────────────────────────

def test_the_padding_is_not_counted_as_sound():
    start, end = audible_span(_buf(pad_s=0.15, speech_s=0.6, tail_s=0.4), 16000)

    assert start == 0.15
    assert end == 0.75


def test_a_silent_buffer_has_no_span():
    assert audible_span(np.zeros(1000, dtype=np.float32), 16000) == (0.0, 0.0)


def test_near_silence_below_the_threshold_does_not_count():
    """-60dB 이하를 소리로 세면 꼬리 무음이 '말소리'가 돼 판정이 뒤집힌다."""
    a = np.concatenate([np.full(1000, 1e-5, dtype=np.float32),
                        np.full(1000, 0.3, dtype=np.float32)])

    start, _ = audible_span(a, 16000)

    assert start == 1000 / 16000


# ── 답이 필러를 끊나 ─────────────────────────────────────────────────────────
# 🔴 sd.play() 는 내부에서 먼저 stop() 을 부른다 — 앞 재생을 닫는다. 답이 먼저 오면
#    필러는 그 자리에서 잘린다. 꼬리 무음만 잘리면 아무도 모르고, 말소리를 자르면 '뚝'이다.

def test_a_cut_that_only_takes_the_silent_tail_is_not_audible():
    v = overlap_verdict(buffer_s=1.35, audible_end_s=0.85, answer_at_s=1.20)

    assert v["cut"] is True, "버퍼는 잘렸다"
    assert v["audible_cut"] is False, "그런데 잘린 건 무음이다 — 안 들린다"


def test_a_cut_into_the_speech_is_the_one_that_gets_heard():
    v = overlap_verdict(buffer_s=1.35, audible_end_s=0.85, answer_at_s=0.60)

    assert v["audible_cut"] is True


def test_an_answer_after_the_whole_buffer_cuts_nothing():
    v = overlap_verdict(buffer_s=1.35, audible_end_s=0.85, answer_at_s=2.20)

    assert v["cut"] is False
    assert v["audible_cut"] is False


def test_the_margin_says_how_much_room_is_left():
    v = overlap_verdict(buffer_s=1.35, audible_end_s=0.85, answer_at_s=1.72)

    assert v["margin_s"] == 1.72 - 0.85


def test_the_fast_turn_is_the_one_that_decides():
    """보통 턴만 보면 통과한다 — 최악의 턴에서 잘리는 걸 놓치면 안 된다."""
    assert ANSWER_AT_FAST_S < ANSWER_AT_TYPICAL_S

    late = overlap_verdict(1.35, 1.90, ANSWER_AT_TYPICAL_S)
    fast = overlap_verdict(1.35, 1.90, ANSWER_AT_FAST_S)

    assert late["audible_cut"] is False
    assert fast["audible_cut"] is True


def test_the_current_fillers_survive_the_fastest_turn():
    """지금 문구(말소리 최대 0.85s)는 최악의 턴에도 안 잘려야 한다.

    ⚠️ 문구를 늘리다 이 선을 넘으면 필러가 제 답에 잘려 '뚝' 소리가 난다.
    """
    v = overlap_verdict(buffer_s=1.39, audible_end_s=0.85, answer_at_s=ANSWER_AT_FAST_S)

    assert v["audible_cut"] is False, "필러가 제 답에 잘린다 — 문구를 줄일 것"


# ── 재생 레이트 고르기 ───────────────────────────────────────────────────────
# 🔴 젯슨 /etc/asound.conf: `pcm.!default = plug -> hw:APE,0 @48000`.
#    `plug` 는 뭐든 받아주므로 check_output_settings(44100) 이 성공하고, 운영 경로는
#    44100 을 고른다 → ALSA 가 매 재생마다 44100->48000 을 변환한다.
#    그 변환을 빼고 재보려면 레이트를 강제할 수 있어야 한다.

from tools.check_filler import resolve_play_rate


class _FakeSd:
    def __init__(self, dev_rate=48000):
        self._dev_rate = dev_rate

    def query_devices(self, i):
        return {"default_samplerate": self._dev_rate}


class _FakeTTS:
    def _resolve_play_rate(self, sd):
        return 44100        # 운영 경로가 고르는 값


def test_a_forced_rate_wins_over_everything():
    assert resolve_play_rate(_FakeSd(), _FakeTTS(), out_dev=3, forced=48000) == 48000


def test_a_named_device_uses_its_own_rate():
    assert resolve_play_rate(_FakeSd(48000), _FakeTTS(), out_dev=34, forced=None) == 48000


def test_without_either_the_operating_path_decides():
    """운영 경로를 그대로 재현하는 게 기본이어야 한다 — 안 그러면 재현이 아니다."""
    assert resolve_play_rate(_FakeSd(), _FakeTTS(), out_dev=None, forced=None) == 44100


# ── 어느 장치가 진짜 스피커인가 ──────────────────────────────────────────────
# 🔴 젯슨 기본 출력(`plug -> hw:APE,0`)은 Tegra 오디오 패브릭의 DMA 입구다. XBAR
#    라우팅과 코덱이 없으면 **sd.play 는 성공하고 소리만 안 난다.** 그래서 후보를
#    하나씩 재보는 모드가 필요한데, APE 내부 링크가 24개나 있어 그냥 다 재면 못 듣는다.

from tools.check_filler import output_candidates


def _dev(name, out=2, inp=0):
    return {"name": name, "max_output_channels": out, "max_input_channels": inp}


def test_input_only_devices_are_not_candidates():
    devs = [_dev("마이크", out=0, inp=6), _dev("스피커", out=2)]

    assert output_candidates(devs) == [(1, "스피커")]


def test_the_ape_dma_links_are_left_out():
    """이걸 안 빼면 사람이 24번 삑 소리를 듣고 있어야 한다 — 아무것도 안 들리는 채로."""
    devs = [_dev("ReSpeaker (hw:0,0)"),
            _dev("NVIDIA Jetson Orin Nano APE: - (hw:2,0)"),
            _dev("NVIDIA Jetson Orin Nano APE: - (hw:2,1)"),
            _dev("HDMI 0 (hw:1,3)", out=8)]

    assert output_candidates(devs) == [(0, "ReSpeaker (hw:0,0)"), (3, "HDMI 0 (hw:1,3)")]


def test_the_index_is_the_device_index_not_the_candidate_number():
    """돌려주는 번호를 --out-device 에 그대로 넣는다 — 어긋나면 엉뚱한 장치를 지정한다."""
    devs = [_dev("mic", out=0, inp=2), _dev("mic2", out=0, inp=2), _dev("speaker")]

    assert output_candidates(devs)[0][0] == 2


# ── 미뤄 내면 여유가 그만큼 준다 ─────────────────────────────────────────────
# 🔴 delay_s 를 키우면 필러가 늦게 시작하므로 **말소리 끝도 그만큼 뒤로 밀린다.**
#    판정이 그걸 안 세면 "안 잘림"이라 해놓고 실기에선 잘린다.

def test_a_delayed_filler_loses_that_much_margin():
    at_zero = overlap_verdict(1.79, 0.91 + 0.0, ANSWER_AT_FAST_S)
    delayed = overlap_verdict(1.79, 0.91 + 0.6, ANSWER_AT_FAST_S)

    assert at_zero["margin_s"] == pytest.approx(0.81, abs=1e-9)
    assert delayed["margin_s"] == pytest.approx(0.21, abs=1e-9)
    assert delayed["audible_cut"] is False, "0.6 은 아직 안전해야 한다"


def test_pushing_the_delay_past_the_ceiling_cuts_the_filler():
    """상한 0.81 을 넘기면 필러 말소리가 답에 잘린다 — 도구가 그걸 말해줘야 한다."""
    v = overlap_verdict(1.99, 0.91 + 0.9, ANSWER_AT_FAST_S)

    assert v["audible_cut"] is True
