"""Smart Turn 오프라인 채점기의 결정적인 부분들.

🔴 왜 이걸 만드나: 실기 체감 4.49s 중 **VAD 꼬리 1.20s(27%)가 정의상 무음**이다.
   아무 말도 안 들어오는데 "혹시 더 말하려나" 하고 기다리는 시간이다. Smart Turn 은
   '말이 문법적/의미적으로 끝났나'를 오디오만 보고 판정하므로, 끝났다고 하면 꼬리를
   1.2s -> 0.2s 로 줄일 수 있다. 벤더 실측(v3.2, 한국어 889샘플)은 23개 언어 중 1위:
     정확도 96.96% / 재현율 0.984 / 오탐률 2.25%
   ⚠️ 그 889샘플은 **성인**이다. 우리 대상은 3~6세다. 아이는 문장 중간에 훨씬 오래
      쉬므로 오탐(말 끊기)이 성인보다 나쁠 수 있다 — 그게 이 채점기로 확인할 것이다.

여기 테스트는 **모델이 아니라 채점 절차**를 검증한다. 모델 자체의 성적은 실측이지
단위테스트가 아니다. 절차가 틀리면 실측 숫자가 통째로 거짓말이 되므로 이쪽을 굳힌다.
"""
import numpy as np
import pytest

from tools.eval_turn_detect import (
    _recall_at_capped_fpr,
    internal_gaps,
    fit_to_window,
    make_incomplete,
    speech_span,
    append_room_tone,
    append_silence,
    summarize,
    trim_tail,
)

SR = 16000


def _tone(seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ── 8초 창 맞추기 ────────────────────────────────────────────────────────────
# 공식 지침: "짧으면 **앞쪽에** 0 을 채워 오디오가 벡터의 끝에 오게 하라".
# 뒤에 채우면 모델이 보는 '마지막 순간'이 무음이 돼 판정이 통째로 뒤틀린다.

def test_short_audio_is_padded_at_the_front():
    audio = _tone(2.0)

    out = fit_to_window(audio, SR, seconds=8.0)

    assert len(out) == 8 * SR
    assert np.array_equal(out[-len(audio):], audio), "오디오가 끝에 붙지 않았다"
    assert not out[: 6 * SR].any(), "앞쪽이 0 이 아니다 — 뒤에 패딩했다"


def test_long_audio_keeps_the_tail():
    """8초를 넘으면 **최근** 8초를 남긴다. 판정 대상은 말의 끝이다."""
    audio = np.concatenate([_tone(6.0, 0.2), _tone(6.0, 0.9)])

    out = fit_to_window(audio, SR, seconds=8.0)

    assert len(out) == 8 * SR
    assert np.array_equal(out, audio[-8 * SR:])


def test_exact_length_passes_through():
    audio = _tone(8.0)
    assert np.array_equal(fit_to_window(audio, SR, seconds=8.0), audio)


# ── 말소리 구간 찾기 ─────────────────────────────────────────────────────────

def test_speech_span_finds_the_loud_region():
    audio = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])

    start, end = speech_span(audio, SR)

    assert start == pytest.approx(1.0 * SR, abs=SR // 40)
    assert end == pytest.approx(3.0 * SR, abs=SR // 40)


def test_speech_span_of_pure_silence_is_empty():
    start, end = speech_span(_silence(2.0), SR)
    assert start == end


# ── '아직 안 끝난 말' 만들기 ─────────────────────────────────────────────────
# 미완 케이스가 있어야 오탐률(=아이 말을 끊는 비율)을 잴 수 있다.

def test_make_incomplete_cuts_inside_the_speech():
    audio = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])

    out = make_incomplete(audio, SR, fraction=0.5)

    assert len(out) == pytest.approx(1.5 * SR, abs=SR // 40), (
        "말소리 2초의 절반(1초)에서 잘라야 한다 -> 0.5s 앞 무음 + 1.0s")


def test_make_incomplete_ignores_trailing_silence():
    """🔴 파일 길이의 비율로 자르면 안 된다 — 뒤 무음이 길수록 엉뚱한 데를 자른다.

    아래는 5.5초 파일이지만 말소리는 0.5~2.5초뿐이다. 파일 기준 50% 는 2.75초라
    **이미 말이 다 끝난 지점**이고, 그러면 '미완'이라 이름 붙인 표본이 실은 완결이라
    오탐률이 통째로 거짓이 된다.
    """
    audio = np.concatenate([_silence(0.5), _tone(2.0), _silence(3.0)])

    out = make_incomplete(audio, SR, fraction=0.5)

    assert len(out) == pytest.approx(1.5 * SR, abs=SR // 40)
    assert len(out) < 2.5 * SR, "말이 끝난 뒤를 잘랐다 — 미완 표본이 아니다"


def test_make_incomplete_returns_none_when_there_is_no_speech():
    assert make_incomplete(_silence(2.0), SR, fraction=0.5) is None


# ── 집계 ─────────────────────────────────────────────────────────────────────
# 양성 = '말이 끝났다(complete)'. 벤더 표와 같은 정의로 맞춰야 비교가 가능하다.

def test_summarize_computes_recall_and_false_positive_rate():
    rows = [
        {"kind": "complete", "prob": 0.9},    # 맞힘
        {"kind": "complete", "prob": 0.8},    # 맞힘
        {"kind": "complete", "prob": 0.2},    # 놓침(느려짐)
        {"kind": "incomplete", "prob": 0.1},  # 맞힘
        {"kind": "incomplete", "prob": 0.7},  # 오탐(말 끊음)
    ]

    m = summarize(rows, threshold=0.5)

    assert m["n_complete"] == 3
    assert m["n_incomplete"] == 2
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_summarize_respects_the_threshold():
    """문턱을 올리면 말을 덜 끊는 대신 시간을 덜 번다 — 그 맞바꿈을 조절할 수 있어야 한다."""
    rows = [{"kind": "complete", "prob": 0.6}, {"kind": "incomplete", "prob": 0.7}]

    strict = summarize(rows, threshold=0.8)

    assert strict["recall"] == 0.0
    assert strict["fpr"] == 0.0


def test_summarize_without_samples_does_not_divide_by_zero():
    m = summarize([], threshold=0.5)
    assert m["recall"] is None and m["fpr"] is None


# ── 끝 무음 정규화 ───────────────────────────────────────────────────────────
# 🔴 이걸 안 하면 실측 재현율이 통째로 부풀려진다.

def test_trim_tail_keeps_only_a_short_silence_after_speech():
    """운영에선 VAD 가 0.2초 무음에서 모델을 부른다. 파일에 붙은 3초 꼬리를 그대로
    먹이면 모델에 '이만큼 조용했다'는 공짜 힌트를 주는 셈이라 성적이 부풀려진다."""
    audio = np.concatenate([_silence(0.5), _tone(2.0), _silence(3.0)])

    out = trim_tail(audio, SR, keep_ms=200.0)

    assert len(out) == pytest.approx(2.7 * SR, abs=SR // 40)


def test_trim_tail_does_not_lengthen_a_clip_that_is_already_short():
    audio = np.concatenate([_tone(1.0), _silence(0.05)])

    out = trim_tail(audio, SR, keep_ms=200.0)

    assert len(out) == len(audio), "없는 무음을 만들어 붙였다"


def test_trim_tail_never_cuts_into_the_speech():
    audio = np.concatenate([_silence(0.2), _tone(1.5)])

    out = trim_tail(audio, SR, keep_ms=0.0)

    assert len(out) >= 1.7 * SR - SR // 40, "말소리를 잘라먹었다"


def test_trim_tail_leaves_pure_silence_alone():
    audio = _silence(1.0)
    assert len(trim_tail(audio, SR, keep_ms=200.0)) == len(audio)


# ── 무음 맞추기 ──────────────────────────────────────────────────────────────
# 🔴 2026-08-24. 첫 실험이 여기서 틀렸다. '완결+200ms' 대 '미완+0ms' 를 비교했는데,
#    이 모델은 뒤 무음 길이를 크게 참고한다(아이 발화 30개 실측: 뒤 무음 0.0s->중앙
#    0.187, 0.4s->0.366, 0.8s->0.765). 두 클래스의 무음이 다르면 모델은 무음만 보고도
#    가를 수 있어, 정작 재려던 '말이 끝나게 들리나'가 아니라 '무음이 기냐'를 재게 된다.
#    운영에서도 두 경우 모두 같은 시간만큼 조용하다 — 그래야 공정하다.

def test_append_silence_adds_exactly_that_much():
    audio = _tone(1.0)

    out = append_silence(audio, SR, seconds=0.4)

    assert len(out) == pytest.approx(1.4 * SR, abs=2)
    assert np.array_equal(out[: len(audio)], audio), "원본을 건드렸다"
    assert not out[len(audio):].any(), "붙인 게 무음이 아니다"


def test_append_silence_of_zero_is_a_no_op():
    audio = _tone(1.0)
    assert np.array_equal(append_silence(audio, SR, 0.0), audio)


def test_both_classes_can_be_matched_to_the_same_silence():
    """완결·미완 표본이 같은 무음을 달고 나오는지 — 비교의 전제다."""
    audio = np.concatenate([_silence(0.2), _tone(2.0), _silence(3.0)])

    complete = append_silence(trim_tail(audio, SR, keep_ms=0.0), SR, 0.5)
    incomplete = append_silence(make_incomplete(audio, SR, 0.5), SR, 0.5)

    assert len(complete) - 2.2 * SR == pytest.approx(0.5 * SR, abs=SR // 40)
    assert len(incomplete) - 1.2 * SR == pytest.approx(0.5 * SR, abs=SR // 40)


# ── 룸톤 패딩 ────────────────────────────────────────────────────────────────
# 🔴 디지털 0 은 실제 무음이 아니다. 진짜 조용한 순간에도 마이크는 룸톤을 담는다.
#    모델은 그런 오디오로 학습됐으니 순수 0 은 분포 밖 입력일 수 있다. 0 으로 잰
#    숫자가 실기와 다를 위험이 있어, 그 녹음 자체의 조용한 부분을 잘라 붙여 대조한다.

def _hiss(seconds: float, amp: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def test_append_room_tone_gets_the_length_right():
    audio = np.concatenate([_hiss(0.3), _tone(1.0), _hiss(0.3)])

    out = append_room_tone(audio, SR, seconds=0.5)

    assert len(out) == pytest.approx(len(audio) + 0.5 * SR, abs=2)
    assert np.array_equal(out[: len(audio)], audio), "원본을 건드렸다"


def test_append_room_tone_is_not_digital_zero():
    audio = np.concatenate([_hiss(0.3), _tone(1.0), _hiss(0.3)])

    tail = append_room_tone(audio, SR, seconds=0.5)[len(audio):]

    assert tail.any(), "0 을 붙였다 — 룸톤을 쓰는 의미가 없다"


def test_append_room_tone_is_quiet_not_speech():
    """붙인 게 말소리 크기면 모델이 '아직 말하는 중'으로 읽는다 — 정반대가 된다."""
    audio = np.concatenate([_hiss(0.3), _tone(1.0, amp=0.5), _hiss(0.3)])

    tail = append_room_tone(audio, SR, seconds=0.5)[len(audio):]
    speech = audio[int(0.3 * SR): int(1.3 * SR)]

    assert _rms(tail) < _rms(speech) / 20, "붙인 게 너무 크다"


def test_append_room_tone_falls_back_to_zeros_on_pure_silence():
    out = append_room_tone(_silence(1.0), SR, seconds=0.3)
    assert len(out) == pytest.approx(1.3 * SR, abs=2)


def test_append_room_tone_of_zero_is_a_no_op():
    audio = _tone(1.0)
    assert np.array_equal(append_room_tone(audio, SR, 0.0), audio)


# ── 오탐 상한 아래의 최고 재현율 ─────────────────────────────────────────────

def test_recall_at_capped_fpr_picks_the_best_usable_threshold():
    rows = [
        {"kind": "complete", "prob": 0.95},
        {"kind": "complete", "prob": 0.60},
        {"kind": "incomplete", "prob": 0.70},
        {"kind": "incomplete", "prob": 0.10},
    ]

    recall, fpr, th = _recall_at_capped_fpr(rows, cap=0.5)

    assert recall == pytest.approx(1.0), "오탐 50% 를 허용하면 둘 다 잡을 수 있다"
    assert fpr <= 0.5


def test_recall_at_capped_fpr_says_impossible_instead_of_lying():
    """🔴 상한을 못 지키면 '불가'라고 해야 한다.

    예전엔 초기값을 그대로 돌려줘 표에 '재현율 0% / 오탐 100%' 로 찍혔다.
    100% 는 잰 값이 아니라 초기값인데, 읽는 사람은 실측으로 오해한다.
    """
    rows = [
        {"kind": "complete", "prob": 0.9},
        {"kind": "incomplete", "prob": 0.99},   # 무슨 문턱을 써도 완결보다 높다
    ]

    recall, fpr, th = _recall_at_capped_fpr(rows, cap=0.0)

    assert recall == 0.0
    assert fpr is None and th is None, "달성 불가를 실측치처럼 돌려줬다"


# ── 발화 내부의 쉼 ───────────────────────────────────────────────────────────
# 🔴 왜 재나: Smart Turn 이 안 먹혀도 **고정 꼬리를 줄이는 길**은 남는다. 지금 1.2s 를
#    쓰는 이유는 '아이가 문장 중간에 쉬어도 안 끊기려고' 인데, 그 쉼이 실제로 얼마나
#    긴지는 재본 적이 없다. 재보면 꼬리를 얼마까지 줄여도 되는지 데이터로 정해진다.
# ⚠️ 앞뒤 무음은 쉼이 아니다 — 그건 녹음 여백이다. 그걸 세면 값이 통째로 부풀려진다.

def test_internal_gaps_finds_a_pause_between_two_words():
    audio = np.concatenate([_tone(0.5), _silence(0.4), _tone(0.5)])

    gaps = internal_gaps(audio, SR)

    assert len(gaps) == 1
    assert gaps[0] == pytest.approx(0.4, abs=0.05)


def test_internal_gaps_ignores_leading_and_trailing_silence():
    """🔴 녹음 여백을 쉼으로 세면 '아이가 3초씩 쉰다'는 거짓 결론이 나온다."""
    audio = np.concatenate([_silence(1.5), _tone(0.5), _silence(0.3),
                            _tone(0.5), _silence(2.0)])

    gaps = internal_gaps(audio, SR)

    assert len(gaps) == 1, f"앞뒤 여백까지 셌다: {gaps}"
    assert gaps[0] == pytest.approx(0.3, abs=0.05)


def test_internal_gaps_finds_several():
    audio = np.concatenate([_tone(0.3), _silence(0.2), _tone(0.3),
                            _silence(0.5), _tone(0.3)])

    gaps = internal_gaps(audio, SR)

    assert len(gaps) == 2
    assert sorted(gaps)[0] == pytest.approx(0.2, abs=0.05)
    assert sorted(gaps)[1] == pytest.approx(0.5, abs=0.05)


def test_internal_gaps_of_continuous_speech_is_empty():
    assert internal_gaps(_tone(2.0), SR) == []


def test_internal_gaps_of_pure_silence_is_empty():
    assert internal_gaps(_silence(2.0), SR) == []
