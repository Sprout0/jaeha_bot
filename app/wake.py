"""호출어(웨이크워드) 판정 — 세션 트리거의 '감지기' 부분.

지금은 STT 텍스트를 '재하봇'과 자모 유사도로 비교하는 '명확발화 감지기'다(임계값 0.6).
실측(2026-07-27, wake_samples): 또박또박 재하봇=0.25~0.57, 무관한 말=0.71~1.00 →
0.6 컷이면 명확발화는 깨우고 무관한 말엔 안 깨움(오작동 0). 유아 웅얼거림은 이 방식으론
못 잡음 → 나중에 openWakeWord(ONNX, 오디오 직접감지)로 이 감지기만 교체(세션 뼈대는 그대로).

주의: STT transcribe 는 이미 correct_stt(keywords)로 가까운 변형(≤0.34)을 '재하봇'으로
스냅한다. 여기 0.6 은 그보다 관대해 0.34~0.6 변형(재하/제하모 등)까지 깨운다.
마이크 없이 문자열로 검증 가능(REPL: python -m app.wake).
"""
from __future__ import annotations

from .text_norm import jamo_ratio

_PUNCT = " .,!?~\"'…"


def is_wake_word(
    text: str,
    word: str = "재하봇",
    threshold: float = 0.6,
    aliases: list[str] | None = None,
) -> bool:
    """text 안에 호출어(word)로 볼 만한 토큰이 있으면 True.

    - aliases: 확정 호출 오인식(정확 일치 시 즉시 True). 예: ["개하복","제하모"].
    - threshold: 자모 유사도 컷(작을수록 엄격). 단일 토큰 + 인접 2토큰 결합을 모두 검사
      (STT가 '재하 봇'처럼 쪼개 적어도 결합해서 '재하봇'으로 잡히게).
    """
    if not text:
        return False
    aliases = set(aliases or [])
    toks = [t.strip(_PUNCT) for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return False

    # 1) 단일 토큰: 별칭 정확일치 or 자모 유사도 ≤ threshold
    for t in toks:
        if t in aliases or t == word:
            return True
        if jamo_ratio(t, word) <= threshold:
            return True

    # 2) 인접 2토큰 결합('재하 봇' -> '재하봇')도 검사
    for a, b in zip(toks, toks[1:]):
        if jamo_ratio(a + b, word) <= threshold:
            return True

    return False


def best_wake_ratio(text: str, word: str = "재하봇") -> float:
    """text 안에서 호출어(word)에 가장 가까운 자모거리(0=동일, 1=완전다름). 진단·튜닝용.

    대기 모드에서 '안 깨움' 시 이 값을 로그로 남기면, 실제로 뭐라고 들렸고 얼마나
    가까웠는지 보여 임계값·별칭을 실측으로 조정할 수 있다.
    """
    if not text:
        return 1.0
    toks = [t.strip(_PUNCT) for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return 1.0
    cands = list(toks) + [a + b for a, b in zip(toks, toks[1:])]  # 단일 + 인접결합
    return min(jamo_ratio(c, word) for c in cands)


def is_sleep_command(text: str, words: list[str] | None = None) -> bool:
    """대화를 끝내고 다시 대기(잠듦)로 갈 발화인지. 놀이의 '그만'과 겹치지 않게 별도 단어."""
    if not text:
        return False
    words = words or ["잘자", "잘 자", "코자자", "코 자자", "바이바이", "안녕히"]
    t = text.strip(_PUNCT)
    return any(w in t for w in words)


def _repl() -> None:
    """문자열로 판정 확인(마이크 불필요)."""
    aliases = ["개하복", "제하모", "재하모사", "재보소"]
    pos = ["재하봇", "재하봇아", "재하", "제하모", "개하복", "재하 봇", "재하보사"]
    neg = ["엄마 어디 있어", "사과 먹고 싶어", "이게 뭐야", "강아지 멍멍 해봐", "제가 웃어"]
    print("[깨워야 함]")
    for s in pos:
        print(f"  {is_wake_word(s, aliases=aliases)!s:5} <- {s!r}")
    print("[안 깨야 함]")
    for s in neg:
        print(f"  {is_wake_word(s, aliases=aliases)!s:5} <- {s!r}")


if __name__ == "__main__":
    _repl()
