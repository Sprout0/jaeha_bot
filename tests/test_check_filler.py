"""젯슨 필러 점검 도구의 판정 로직.

장치가 필요한 부분은 여기서 못 잰다. 대신 **판정이 틀리지 않는지**를 지킨다 —
젯슨에서 나온 숫자를 이 함수들이 잘못 읽으면 멀쩡한 걸 기각하거나 그 반대가 된다.
"""
import numpy as np

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
