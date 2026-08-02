"""STT 오인식 사후 교정.

문제: '재하봇' 같은 고유명사는 Whisper 학습에 없어(OOV) 발음대로 적힌다
      (재하봇아 -> 재하보사, 제아부사 등). 뜻이 뭉개져 LLM 이 헷갈린다.

해결(2층):
  1) aliases: 자주 나오는 확정 오인식을 정확 매핑(빠르고 100%).
  2) keywords: 한글을 '자모'로 분해해 편집거리(Levenshtein)가 가까운 토큰을
     해당 키워드로 교정. 고유명사는 자모열이 독특해 오탐이 적다.

마이크 없이도 가짜 오타 문자열로 검증 가능(REPL: python -m app.text_norm).
"""
from __future__ import annotations

import re

# 한글 자모 분해용 표 (유니코드 조합 규칙)
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def decompose(text: str) -> list[str]:
    """문자열을 자모 리스트로 분해한다(한글 음절만 분해, 나머지는 그대로)."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:  # 완성형 한글
            i = code - 0xAC00
            out.append(_CHO[i // (21 * 28)])
            out.append(_JUNG[(i % (21 * 28)) // 28])
            jong = i % 28
            if jong:
                out.append(_JONG[jong])
        else:
            out.append(ch)
    return out


def _edit_distance(a: list[str], b: list[str]) -> int:
    """두 자모열의 Levenshtein 편집거리."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def jamo_ratio(w1: str, w2: str) -> float:
    """두 단어의 자모 편집거리를 긴 쪽 길이로 나눈 비율(0=동일, 클수록 다름)."""
    j1, j2 = decompose(w1), decompose(w2)
    denom = max(len(j1), len(j2)) or 1
    return _edit_distance(j1, j2) / denom


def dedupe_repeats(text: str, min_word_run: int = 3) -> str:
    """Whisper 반복 환각 제거: 같은 단어/구절이 연달아 반복되면 한 번만 남긴다.

    실제 아동 음성 평가(2026-07-23)에서 발화 뒤 잔여음에 whisper 가 헛것을 게워내며
    같은 구절을 무한 반복하는 게 유일한 치명 오류(24%)로 확인됨. 예:
      "일찍 일찍 일찍 ..."(×40)                         -> "일찍"
      "저 위에 꺼 내려주세요 저 위에 꺼 내려주세요"       -> "저 위에 꺼 내려주세요"
      "...나요 그건 ...나요 그건 ...나요"                 -> "...나요 그건 ...나요"
    모델 무관 후처리라 젯슨에도 그대로 이득. 자연스러운 2회 반복(멍멍 멍멍)은
    보존하려 단어 루프는 min_word_run 회 이상만, 구절 반복은 2단어 이상 블록만 축약한다.
    """
    if not text:
        return text

    # ⓪ 글자 단위 반복 축약: 같은 글자가 3회 이상 연속이면 1개로("모사하하하하"->"모사하").
    #    짧은발화 끝모음서 whisper 가 한 글자를 수백번 게워내는 폭주 대응(단어단위 dedupe 로는
    #    한 토큰 내부 반복을 못 잡음). 한국어 정상어는 같은 글자 3연속이 거의 없어 오탐 낮음.
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # ⓪-2 한 토큰 내부가 '짧은 단위의 3회 이상 반복'이면 단위 하나만("아웃아웃아웃"->"아웃").
    #     단일글자 폭주는 위에서, 2글자↑ 단위 폭주는 여기서. 2회 반복은 보존(자연 반복 여지).
    def _collapse_tok(tok: str) -> str:
        n = len(tok)
        for p in range(1, n // 3 + 1):        # 반복 단위 길이
            for s in range(0, n - 3 * p + 1):  # 반복 시작 위치(앞에 접두어 있어도 잡게)
                unit = tok[s:s + p]
                reps = 1
                while tok[s + reps * p:s + (reps + 1) * p] == unit:
                    reps += 1
                if reps >= 3:                 # 단위가 3회 이상 반복 → 1개만 남기고 나머지 재귀
                    return tok[:s + p] + _collapse_tok(tok[s + reps * p:])
        return tok
    text = " ".join(_collapse_tok(t) for t in text.split())

    words = text.split()
    if len(words) < 2:
        return text

    # ① 연속 동일 '단어'가 min_word_run 회 이상이면 1개로("일찍"×40). 2회는 보존(멍멍 멍멍).
    collapsed: list[str] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j] == words[i]:
            j += 1
        run = j - i
        collapsed.append(words[i]) if run >= min_word_run else collapsed.extend(words[i:j])
        i = j
    words = collapsed

    # ② 연속 반복 '구절'(2단어 이상 블록이 2회 이상 이어짐)을 1회로. 위치 무관.
    changed = True
    while changed:
        changed = False
        n = len(words)
        for p in range(2, n // 2 + 1):
            for s in range(0, n - 2 * p + 1):
                if words[s:s + p] == words[s + p:s + 2 * p]:
                    block = words[s:s + p]
                    k = 2
                    while words[s + k * p:s + (k + 1) * p] == block:
                        k += 1
                    words = words[:s + p] + words[s + k * p:]  # k회 -> 1회
                    changed = True
                    break
            if changed:
                break

    return " ".join(words)


def correct_stt(
    text: str,
    keywords: list[str] | None = None,
    aliases: dict[str, str] | None = None,
    max_ratio: float = 0.34,
) -> str:
    """STT 텍스트의 각 토큰을 aliases(정확)·keywords(자모 유사)로 교정한다.

    - aliases: {"재하보사": "재하봇"} 형태의 확정 매핑(토큰 전체 일치 시 치환).
    - keywords: 이 목록의 단어와 자모비율 <= max_ratio 로 가까우면 그 키워드로 치환.
      짧은 토큰(자모 4개 미만)은 오탐이 커서 건드리지 않는다.
    빈 입력이면 그대로 반환.
    """
    if not text:
        return text
    aliases = aliases or {}
    keywords = keywords or []

    out_tokens: list[str] = []
    for tok in text.split():
        # 문장부호를 분리해 보존(예: "재하보사!" -> 코어 "재하보사" + 꼬리 "!")
        core = tok.strip(" .,!?~\"'…")
        tail = tok[len(tok.rstrip(" .,!?~\"'…")):] if core else ""
        head = tok[: len(tok) - len(tok.lstrip(" .,!?~\"'…"))] if core else ""

        fixed = core
        if core in aliases:
            fixed = aliases[core]
        elif core and len(decompose(core)) >= 4:
            best, best_ratio = None, max_ratio
            for kw in keywords:
                r = jamo_ratio(core, kw)
                if r <= best_ratio:
                    best, best_ratio = kw, r
            if best is not None:
                fixed = best
        out_tokens.append(f"{head}{fixed}{tail}")
    return " ".join(out_tokens)


def _repl() -> None:
    """가짜 오인식 문자열로 교정 확인(마이크 불필요)."""
    keywords = ["재하봇", "하연"]
    aliases = {"제아부사": "재하봇", "재아부사": "재하봇"}
    samples = [
        "재하봇아 안녕",
        "재하보사 뭐하고 놀자",
        "재하보싸 이리와",
        "제아부사 사랑해",
        "재하봇 오늘 뭐해",
        "하여니 어디갔어",
        "재밌어 우리 놀자",       # 교정되면 안 됨(오탐 체크)
        "사과 먹고 싶어",         # 교정되면 안 됨
        "빨간색 찾기 하자",       # 교정되면 안 됨
    ]
    for s in samples:
        print(f"{s!r:32} -> {correct_stt(s, keywords, aliases)!r}")


if __name__ == "__main__":
    _repl()
