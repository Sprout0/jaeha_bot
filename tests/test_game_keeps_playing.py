"""아이가 "양 말고 다른 동물" 하면 놀이가 **끝나면 안 된다.**

🔴 2026-08-31 실기(젯슨 10턴)에서 실제로 났다:
     [아이] 약 말고 다른 놀이 다른 동물      (STT 가 '양'을 '약'으로 받아썼다)
     [티드] 재밌었지? 다음에 또 같이 놀자!   <- 놀이가 끝나버렸다
   `_STOP_WORDS` 에 '다른놀이'가 있어서 걸렸다. 아이 뜻은 정반대였다 —
   **그만두겠다는 게 아니라 이 동물이 싫으니 다음 걸 달라는 것**이다.

   그 바로 위 주석이 이미 이렇게 경고하고 있었다:
       "⚠️ '다른 동물'은 계속하겠다는 뜻이라 넣지 않는다"
   대비는 돼 있었는데, 아이가 '다른 놀이'와 '다른 동물'을 **한 문장에 같이** 말하니
   그만두는 쪽이 이겼다. 두 신호가 부딪히면 **계속하는 쪽이 이겨야 한다** —
   잘못 끝내면 아이가 놀이를 잃고, 잘못 계속하면 봇이 한 번 더 물어볼 뿐이다.
"""
from app.education_modes import AnimalSoundGame, RepeatWordGame


def _game(batch, state="await_answer"):
    g = AnimalSoundGame()
    g.batch = list(batch)
    g.bi = 0
    g.state = state
    return g


def _text(beat):
    return beat["fallback"] + " " + (beat["instruction"] or "")


def test_the_exact_utterance_that_ended_the_game_no_longer_does():
    """2026-08-31 실기 로그 그대로. 이 한 줄이 이 파일의 존재 이유다."""
    g = _game([("양", "매애"), ("강아지", "멍멍")])

    beat = g.step("약 말고 다른 놀이 다른 동물")

    assert not g.done, "'다른 동물'이라고 했는데 놀이가 끝났다"
    assert "강아지" in _text(beat), "다음 동물을 물어야 한다"


def test_asking_for_another_animal_moves_on_instead_of_repeating():
    """되묻기('같이 해보자')로 같은 동물에 머물면, 싫다는 걸 한 번 더 시키는 셈이다."""
    g = _game([("양", "매애"), ("강아지", "멍멍")])

    beat = g.step("양 말고 다른 동물")

    assert not g.done
    assert g.bi == 1, "다음 항목으로 넘어가야 한다"
    assert "양" not in beat["fallback"], "방금 싫다고 한 동물을 또 물었다"


def test_wanting_a_different_game_still_ends_it():
    """'다른 놀이'만 있으면 진짜로 나가겠다는 뜻이다 — 이건 그대로 끝나야 한다."""
    g = _game([("양", "매애"), ("강아지", "멍멍")])

    g.step("다른 놀이 하자")

    assert g.done


def test_a_plain_stop_still_stops():
    g = _game([("양", "매애"), ("강아지", "멍멍")])

    g.step("그만할래")

    assert g.done


def test_another_animal_at_the_checkpoint_starts_a_new_round():
    """'더 할래?' 에 '다른 동물!' 은 **하겠다**는 뜻이다."""
    g = _game([("양", "매애"), ("강아지", "멍멍")], state="await_continue")
    g.bi = 1

    beat = g.step("다른 동물 하고 싶어")

    assert not g.done, "체크포인트에서 계속하겠다고 했는데 끝났다"
    assert beat["fallback"].strip(), "다음 질문이 나와야 한다"


def test_the_word_game_understands_its_own_wording():
    """따라 말하기에서 '다른 말'은 낱말을 바꿔 달라는 뜻이다."""
    g = RepeatWordGame()
    g.batch = [("사과", "사과"), ("바나나", "바나나")]
    g.bi = 0
    g.state = "await_answer"

    beat = g.step("다른 말 해줘")

    assert not g.done
    assert "바나나" in _text(beat)


def test_saying_next_time_is_goodbye_not_next_item():
    """'다음에 하자' 는 그만하겠다는 말이다. '다음 거'와 헷갈리면 안 끝나는 놀이가 된다."""
    g = _game([("양", "매애"), ("강아지", "멍멍")])

    g.step("다음에 하자 그만")

    assert g.done
