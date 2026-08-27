"""`GameManager._voice` — 이 브랜치의 중심 보증이 실제로 걸리는지.

설계(docs/superpowers/specs/2026-08-11-play-interaction-redesign-design.md §3-3)의 주장은
**확장이 LLM 의 선의가 아니라 구조로 보장된다**는 것이다. 근거는 딱 한 군데,
`_voice` 의 require 검증이다 — LLM 이 아이 말(또는 다음 항목)을 빠뜨리면 검증에 걸려
결정적인 템플릿으로 내려온다. 그런데 그 한 군데를 덮는 테스트가 없었다(2026-08-11 리뷰).

🔴 이건 예외 경로가 아니다. 리액션 beat 는 세 마디를 요구하는데 렌더 결과는
   agent._postprocess 가 두 문장으로 자른다. 즉 세 번째 마디(다음 질문)가 사라져
   require 가 깨지는 일이 **상시로** 일어난다. 템플릿 폴백이 평상시 경로다.
"""
from app.education_modes import GameManager, _beat


def _beat_with_two_tokens():
    """리액션 beat 의 모양: 확장 절 + 다음 질문, require 두 개."""
    return _beat("멍멍! 강아지는 멍멍 하고 울어! 그럼 고양이는 야옹?",
                 "아무 지시문",
                 ["멍멍", "고양이"])


def test_voice_falls_back_when_render_drops_a_required_token():
    # LLM 이 확장은 했지만 다음 질문(=고양이)을 빠뜨린 경우.
    # 두 문장 클램프에 잘리면 실제로 이렇게 된다.
    mgr = GameManager(render=lambda ins: "멍멍! 강아지는 멍멍 하고 울어!")
    beat = _beat_with_two_tokens()

    assert mgr._voice(beat) == beat["fallback"]


def test_voice_falls_back_when_render_drops_the_child_word():
    # 아이 말을 빠뜨리면(확장 없음) 역시 템플릿으로. 확장 보장의 핵심 분기.
    mgr = GameManager(render=lambda ins: "강아지 소리는 그렇구나! 그럼 고양이는?")
    beat = _beat_with_two_tokens()

    assert mgr._voice(beat) == beat["fallback"]


def test_voice_uses_render_output_when_every_token_is_present():
    # 반대 방향. 지키면 LLM 문장이 그대로 나가야 한다 — 안 그러면 LLM 경로가 무의미하다.
    good = "멍멍, 강아지가 멍멍 하고 울어! 그럼 고양이는 야옹?"
    mgr = GameManager(render=lambda ins: good)

    assert mgr._voice(_beat_with_two_tokens()) == good


def test_voice_falls_back_on_empty_render():
    mgr = GameManager(render=lambda ins: "")

    assert mgr._voice(_beat_with_two_tokens()) == _beat_with_two_tokens()["fallback"]


def test_voice_falls_back_when_render_raises():
    # 네트워크·모델 실패로 render 가 터져도 놀이는 절대 멈추지 않아야 한다.
    def boom(_ins):
        raise RuntimeError("LLM 죽음")

    mgr = GameManager(render=boom)

    assert mgr._voice(_beat_with_two_tokens()) == _beat_with_two_tokens()["fallback"]


def test_voice_uses_template_when_there_is_no_renderer():
    # render=None 이면 LLM 없이 템플릿만. 마이크·모델 없는 검증 경로의 전제다.
    mgr = GameManager(render=None)

    assert mgr._voice(_beat_with_two_tokens()) == _beat_with_two_tokens()["fallback"]


def test_voice_returns_none_for_no_beat():
    assert GameManager(render=lambda ins: "뭐라도")._voice(None) is None


# ── A5: 폴백 문장 수 (2026-08-12) ────────────────────────────────────────────
from app.education_modes import AnimalSoundGame, RepeatWordGame   # noqa: E402


def _walk(cls, correct: bool) -> list[dict]:
    """놀이를 끝까지 굴려 나오는 모든 beat 를 모은다(인트로·리액션·재시도·체크포인트)."""
    g = cls()
    beats = [g.start()]
    for _ in range(16):
        if g.done:
            break
        if g.state == "await_continue":
            text = "응"
        else:
            _subj, tgt = g.batch[g.bi]
            text = tgt if correct else "몰라"
        beats.append(g.step(text))
    beats.append(g._bye_beat())
    return beats


def _all_beats() -> list[dict]:
    out = []
    for cls in (AnimalSoundGame, RepeatWordGame):
        for correct in (True, False):
            out += _walk(cls, correct)
    return out


def test_every_fallback_survives_the_production_sentence_cap():
    """🔴 A5. LLM 경로는 두 문장으로 잘리는데 폴백은 클램프를 안 거친다.

    그래서 같은 상황인데 **LLM 이 성공하면 두 문장, 폴백이면 세 문장**을 말했다.
    폴백은 예외 경로가 아니라 평상시 경로다(이 파일 맨 위 주석).

    고치는 방향은 클램프를 거는 게 아니다 — 그러면 세 번째 마디인 '다음 질문'이
    잘려 놀이가 끊긴다. 따라 하는 말을 느낌표로 끊지 말고 확장 문장 **안에** 넣어
    두 문장으로 만든다. 지시문이 LLM 에게 이미 요구하는 바로 그 형태다
    ("따라 하는 말을 느낌표로 끊어 따로 한 문장으로 만들지 않는다").
    """
    from app.agent import _clamp_sentences
    from app.config import settings

    cap = settings.models["llm"]["max_sentences"]
    for beat in _all_beats():
        fb = beat["fallback"]
        assert _clamp_sentences(fb, cap) == fb, \
            f"폴백이 {cap}문장을 넘는다(LLM 경로였다면 잘렸을 문장): {fb}"


def test_clamped_fallback_still_carries_every_required_token():
    """문장을 합치다가 require 토큰을 잃으면 '검증이 요구하는 것'과 갈린다."""
    for beat in _all_beats():
        assert all(tok in beat["fallback"] for tok in beat["require"]), beat


# ── 대화 도중에 "안녕!" 이 나오면 안 된다 ────────────────────────────────────
# 🔴 2026-08-26 젯슨 실기, 대화 4턴째:
#      아이 "동물 소리도 이리 밖에 못해?" -> 티드 "**안녕!** 강아지는 멍?"
#    가드가 뚫린 게 아니라 **지시문이 그렇게 시켰다**: "첫 문장은 놀이를 하자는 밝은 인사".
#    LLM 은 '인사'를 '안녕'으로 읽었고, 시킨 대로 한 것이다. 폴백 템플릿
#    ("좋아, 동물 소리 놀이 하자!")이 뜻하던 건 첫인사가 아니라 **신난 대답**이었다.
# ⚠️ _bye_beat 의 "마무리하는 인사"는 작별이라 맞다 — 그건 건드리지 않는다.

import pytest

from app.education_modes import AnimalSoundGame, RepeatWordGame


@pytest.mark.parametrize("game", [AnimalSoundGame(), RepeatWordGame()])
def test_the_intro_forbids_the_greeting_it_used_to_ask_for(game):
    """`"인사" not in instruction` 으로 재려다 실패했다 — 금지 문구에도 '첫인사'가
    들어가서 자기 자신을 잡는다. 볼 것은 낱말의 유무가 아니라 **금지했는가**다."""
    instruction = game.start()["instruction"]

    assert "안녕" in instruction and "쓰지 마" in instruction,         f"'안녕' 을 쓰지 말라고 명시해야 한다: {instruction}"
    assert "밝은 인사" not in instruction, "이 문구가 '안녕!'을 부른 그 지시다"


@pytest.mark.parametrize("game", [AnimalSoundGame(), RepeatWordGame()])
def test_the_intro_shows_what_to_say_instead(game):
    """금지만 주면 모델이 뭘 할지 모른다 — 대신 할 말을 예시로 줘야 한다."""
    instruction = game.start()["instruction"]

    assert "하자!" in instruction, f"대신 할 말의 예시가 없다: {instruction}"


def test_the_farewell_may_still_greet():
    """놀이를 끝낼 때의 인사는 맞는 자리다 — 같이 지우면 안 된다."""
    game = AnimalSoundGame()
    game.start()
    beat = game._bye_beat()

    assert "인사" in beat["instruction"]
