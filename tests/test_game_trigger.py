"""놀이 시작 트리거의 오탐과 탈출구. 실기(2026-08-10)에서 드러난 결함이다.

실제로 일어난 일 — 아이가 "와 목소리가 왜그래?" 라고 했는데 동물 소리 놀이가 시작됐고,
그 뒤 "다른 놀이 하자" 에도 "돼지는 꿀꿀 하고 운다!" 로 답하며 4턴을 끌고 갔다.

원인은 두 가지다:
  1) match_trigger 의 구성 단서 규칙이 '동물이름 + 소리/놀이' 를 찾는데,
     ANIMAL_ITEMS 에 **한 글자 이름('소','양')** 이 있다.
     "목소리" 안에 '소' 와 '소리' 가 다 들어 있어 조건이 성립한다.
     claims.py 가 _MIN_NAME 으로 막아둔 것과 같은 부류인데 여기엔 가드가 없었다.
  2) 놀이 중에 빠져나올 말이 '그만' 계열뿐이라, 아이가 흔히 쓰는
     "다른 놀이 하자" 로는 탈출이 안 된다.
"""
from app.education_modes import GameManager, _is_stop, match_trigger


# ------------------------------------------------------------------ 오탐 방지
def test_moksori_does_not_start_animal_game():
    # 실기에서 실제로 놀이를 시작시킨 발화. '목소리' = 목 + 소(동물) + 소리.
    assert match_trigger("와 목소리가 왜그래?") is None


def test_talking_about_the_voice_never_starts_a_game():
    for utt in ["목소리 이상해", "너 목소리가 좀 심각한데?", "목소리 좀 수정이 될 것 같아"]:
        assert match_trigger(utt) is None, f"놀이가 시작되면 안 됨: {utt}"


def test_common_words_containing_one_letter_animal_names():
    # '소'(소), '양'(양)이 흔한 낱말 속에 숨어 있다. 여기에 '놀이/소리'가 겹치면 오탐.
    for utt in ["소파에서 놀이하자", "양말 소리 나", "모양 놀이 재밌어"]:
        assert match_trigger(utt) is None, f"놀이가 시작되면 안 됨: {utt}"


def test_two_letter_animal_name_inside_another_word():
    """🔴 한 글자 이름만 문제가 아니다 — '오리'가 '오리기' 안에 들어 있다(2026-08-31).

    _MIN_ANIMAL_NAME 은 이름 '길이'만 보므로 두 글자 이름이 흔한 낱말에 박힌 건 못 막는다.
    종이 오리기는 2세가 실제로 하는 놀이라 "가위로 오리기 놀이 하자"가 그대로 온다.
    audio_player.AudioLibrary.find() 의 오탐과 같은 병이다 — 낱말 경계를 안 본다.
    """
    for utt in ["가위로 오리기 놀이 하자", "종이 오리기 하고 놀이하자"]:
        assert match_trigger(utt) is None, f"놀이가 시작되면 안 됨: {utt}"


def test_animal_name_with_a_particle_still_starts():
    """한국어는 이름에 조사가 붙어 한 낱말이 된다. 경계를 너무 빡빡하게 잡으면
    '오리기'를 막다가 '고양이랑'까지 잃는다."""
    assert match_trigger("고양이랑 소리 놀이 하자") == "animal"
    assert match_trigger("강아지는 어떻게 우는지 놀이하자") == "animal"


# --------------------------------------------------------------- 정상 트리거 유지
def test_explicit_animal_request_still_starts():
    assert match_trigger("동물 소리 놀이 하자") == "animal"
    assert match_trigger("동물 놀이 하고 싶어") == "animal"


def test_named_animal_plus_sound_still_starts():
    # 두 글자 이상 이름은 그대로 동작해야 한다 — 오탐만 막고 기능은 남긴다.
    assert match_trigger("강아지 소리 내봐") == "animal"
    assert match_trigger("고양이 흉내 내볼까") == "animal"


def test_repeat_game_trigger_unaffected():
    assert match_trigger("따라 말하기 하자") == "repeat"


# ------------------------------------------------------------------- 탈출구
def test_asking_for_a_different_game_counts_as_stop():
    assert _is_stop("다른 놀이 하자"), "아이가 흔히 쓰는 탈출 표현이다"
    assert _is_stop("딴 거 하자")


def test_asking_for_another_animal_does_not_stop():
    # '다른 동물' 은 놀이를 계속하겠다는 뜻이다 — 여기서 끊기면 안 된다.
    assert not _is_stop("다른 동물 하자")


def test_child_can_leave_an_active_game():
    mgr = GameManager(render=None)
    assert mgr.maybe_start("동물 소리 놀이 하자") is not None
    assert mgr.active is not None

    mgr.handle("다른 놀이 하자")

    assert mgr.active is None, "탈출 요청이면 놀이를 끝내고 평소 대화로 돌아가야 한다"
