"""`app/text_norm.py` — 자모 분해·거리·STT 사후교정.

왜 뒤늦게 쓰나 (2026-08-25 전수 점검에서 발견):
  이 모듈에는 **직접 테스트가 하나도 없었다.** 그런데 호출어 판정 전체가 여기 위에 서 있다.
  `app/wake.best_wake_ratio` -> `jamo_ratio` 이고, 2단계 검증 컷(`plain_max_ratio: 0.35`)은
  이 함수가 내는 숫자다. 즉 **깨지면 호출어가 조용히 안 걸리거나 아무 말에나 걸린다.**
  실기에서야 드러나는 종류라 못으로 박아 둔다.

여기 적힌 거리값은 tools/probe_verify_precision.py 실측과 같은 계산이다.
"""
import pytest

from app.text_norm import correct_stt, decompose, dedupe_repeats, jamo_ratio

WORD = "하이티드"


# ── 자모 분해 ────────────────────────────────────────────────────────
def test_decompose_splits_syllable_into_parts():
    assert decompose("하") == ["ㅎ", "ㅏ"]          # 받침 없으면 2개
    assert decompose("한") == ["ㅎ", "ㅏ", "ㄴ"]     # 받침 있으면 3개


def test_decompose_keeps_non_hangul_as_is():
    assert decompose("hi 1!") == list("hi 1!")


def test_decompose_empty():
    assert decompose("") == []


# ── 자모 거리 ────────────────────────────────────────────────────────
def test_ratio_is_zero_for_identical():
    assert jamo_ratio(WORD, WORD) == 0.0


def test_ratio_is_symmetric():
    assert jamo_ratio("하이티드", "하이드") == jamo_ratio("하이드", "하이티드")


def test_ratio_never_exceeds_one():
    assert 0.0 <= jamo_ratio("엄마 어디", WORD) <= 1.0


def test_ratio_empty_does_not_divide_by_zero():
    assert jamo_ratio("", "") == 0.0
    assert jamo_ratio("", WORD) == 1.0


@pytest.mark.parametrize("other, expected", [
    ("하이티드", 0.000),   # 그대로
    ("하이티들", 0.111),   # 받침 하나
    ("하이치드", 0.125),   # 자음 하나
    ("하이드", 0.250),     # 음절 하나 빠짐 — 실측상 가장 가까운 함정
    ("하이브리드", 0.300),  # '하이브리드 차' 가 유일하게 컷을 넘은 말
    ("하이", 0.500),       # 인사말. **여기가 컷 밖이어야 한다**
])
def test_measured_distances_hold(other, expected):
    """실측으로 정한 컷(0.35)이 성립하려면 이 거리들이 유지돼야 한다.

    이 값이 흔들리면 `configs/model_paths.yaml` 의 plain_max_ratio 를 다시 재야 한다.
    """
    assert jamo_ratio(other, WORD) == pytest.approx(expected, abs=0.005)


def test_wake_word_and_traps_are_separated_by_the_cut():
    """호출어 변형은 컷 안, 음운적 함정은 컷 밖 — 이 분리가 2단계 검증의 전부다."""
    cut = 0.35
    for near in ("하이티드", "하이 티드야".replace(" ", ""), "하이티들", "하이티브"):
        assert jamo_ratio(near, WORD) <= cut, near
    for far in ("하이", "티드", "하이킹", "하이라이트", "다이어트", "엄마"):
        assert jamo_ratio(far, WORD) > cut, far


# ── STT 사후교정 ─────────────────────────────────────────────────────
def test_correct_stt_uses_exact_alias_first():
    assert correct_stt("하이티들", keywords=[], aliases={"하이티들": WORD}) == WORD


def test_correct_stt_snaps_near_token_to_keyword():
    assert correct_stt("하이티들", keywords=[WORD]) == WORD


def test_correct_stt_leaves_far_token_alone():
    assert correct_stt("엄마 어디 있어", keywords=[WORD]) == "엄마 어디 있어"


def test_correct_stt_protects_very_short_tokens():
    """자모 4개 미만은 손대지 않는다. '하'는 ㅎ+ㅏ 로 2개뿐이다."""
    assert correct_stt("하", keywords=["하이드"]) == "하"


def test_short_keyword_over_corrects_nearby_words():
    """🔴 **짧은 키워드는 멀쩡한 말을 끌어간다.** 봇 이름을 keywords 에서 뺀 이유다.

    '하이'는 자모 4개라 보호선(>=4)을 **간신히 넘어** 교정 대상이 된다. 그리고
    jamo_ratio('하이','하이드') = 2/6 = 0.333 으로 기본 컷 0.34 안에 든다.
    그래서 아이가 "하이" 라고만 해도 '하이드'로 바뀐다.

    2026-08-25 에 `stt.keywords` 에서 봇 이름을 빼고 `["하연"]` 만 남긴 근거가 이것이다.
    옛 이름 '재하봇'은 whisper 가 못 읽는 OOV 라 이 교정이 꼭 필요했지만,
    '하이티드'는 whisper 가 그대로 읽으니 교정이 이득 없이 위험만 남는다.
    ⚠️ 이 테스트가 깨졌다면 = 동작이 바뀐 것이다. keywords 설정을 다시 검토할 것.
    """
    assert jamo_ratio("하이", "하이드") == pytest.approx(0.333, abs=0.005)
    assert correct_stt("하이", keywords=["하이드"]) == "하이드"      # 끌려간다
    assert correct_stt("하이", keywords=["하연"]) == "하이"          # 운영 설정에선 안전


def test_correct_stt_preserves_punctuation():
    assert correct_stt("하이티들!", keywords=[WORD]) == f"{WORD}!"


def test_correct_stt_empty_is_passthrough():
    assert correct_stt("", keywords=[WORD]) == ""


def test_correct_stt_without_config_is_identity():
    """운영 설정은 keywords=['하연'] / aliases={} 다 — 호출어는 교정하지 않는다."""
    text = "하이 티드 이거 뭐야"
    assert correct_stt(text, keywords=[], aliases={}) == text


# ── 반복 환각 접기 ───────────────────────────────────────────────────
def test_dedupe_collapses_repeated_words():
    out = dedupe_repeats("안녕 안녕 안녕 안녕 안녕")
    assert out.count("안녕") < 5


def test_dedupe_keeps_normal_sentence():
    text = "엄마 어디 있어"
    assert dedupe_repeats(text) == text
