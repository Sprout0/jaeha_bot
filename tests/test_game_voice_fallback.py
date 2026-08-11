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
