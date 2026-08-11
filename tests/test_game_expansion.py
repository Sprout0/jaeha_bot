"""놀이 중 확장(expansion)과 모방(imitation). 마이크·모델 불필요.

문헌에서 효과크기가 가장 큰 개입인데(recast 메타분석 0.76~0.96 SD) 구조적으로
불가능했다 — step() 이 아이 발화를 받고도 beat 에는 '맞았나' 불리언만 넘겼다.
설계: docs/superpowers/specs/2026-08-11-play-interaction-redesign-design.md
"""
from app.education_modes import AnimalSoundGame


def _text(beat):
    """검사 대상은 템플릿과 LLM 지시문이다(LLM 출력은 오프라인에서 재현 불가)."""
    return beat["fallback"] + " " + (beat["instruction"] or "")


def _game(batch):
    g = AnimalSoundGame()
    g.batch = list(batch)
    g.bi = 0
    g.state = "await_answer"
    return g


def test_misheard_text_never_reaches_the_reply():
    # STT 가 '멍멍'을 '명명'으로 받아써도(자모거리 0.333 이라 인정된다) 봇은
    # 카드의 정답 낱말로 되받아야 한다. 원문을 되받으면 잘못된 발음을 가르친다.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    out = _text(g.step("명명"))

    assert "명명" not in out, f"오인식 원문이 새어 나왔다: {out}"
    assert "멍멍" in out


def test_reply_imitates_then_expands():
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    fb = g.step("멍멍")["fallback"]

    assert fb.startswith("멍멍"), f"아이 말을 먼저 그대로 따라 해야 한다: {fb}"
    assert fb.count("멍멍") >= 2, f"따라 한 뒤 늘려 말해야 한다: {fb}"
    assert "강아지" in fb, f"확장 절에 대상 이름이 들어가야 한다: {fb}"


def test_require_forces_the_child_word():
    # LLM 이 아이 말을 빠뜨리면 검증에 걸려 템플릿으로 폴백된다.
    # 확장이 'LLM 의 선의'가 아니라 구조로 보장되는 지점.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    assert "멍멍" in g.step("멍멍")["require"]


def test_checkpoint_also_expands():
    # 배치의 마지막 항목이면 '더 할래?' 체크포인트로 간다. 거기서도 확장은 유지된다.
    g = _game([("강아지", "멍멍")])

    fb = g.step("멍멍")["fallback"]

    assert "멍멍" in fb and "강아지" in fb and "더 할래?" in fb


from app.education_modes import RepeatWordGame


def _repeat_game(batch):
    g = RepeatWordGame()
    g.batch = list(batch)
    g.bi = 0
    g.state = "await_answer"
    return g


def test_repeat_game_echoes_the_child_word():
    g = _repeat_game([("바나나", "바나나"), ("딸기", "딸기")])

    fb = g.step("바나나")["fallback"]

    assert fb.startswith("바나나"), f"아이 말을 먼저 따라 해야 한다: {fb}"
    assert "딸기" in fb, f"다음 낱말로 이어야 한다: {fb}"


def test_repeat_game_misheard_uses_the_card_word():
    # '바나'만 말해도 인정된다(자모거리 0.333). 되받는 건 카드의 '바나나'여야 한다.
    g = _repeat_game([("바나나", "바나나"), ("딸기", "딸기")])

    out = _text(g.step("바나"))

    assert "바나나" in out
    assert out.count("바나나") >= 2, f"따라 한 뒤 다시 써야 한다: {out}"
