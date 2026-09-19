"""아이가 자주 찾는 노래·캐릭터 이름 — 받아쓰기 힌트와 검색 전 이름 바로잡기에 같이 쓴다.

🔴 2026-09-19 실기: "티니핑" 을 받아쓰기가 "비니닝" 으로 적어 유튜브에서 못 찾았다.
   두 겹으로 막는다.
     ① 받아쓰기 모델에 이 이름들을 힌트로 준다(Realtime transcription.prompt).
     ② 그래도 틀리면, 검색 전에 **확실히 비슷할 때만** 사전 이름으로 바꾼다.

기준(09-19 실측, text_norm.jamo_ratio):
  틀린 받아쓰기  비니닝→티니핑 0.29 · 뽀르르→뽀로로 0.33 · 아기사어→아기상어 0.11
  평범한 말      핑크→핑크퐁 0.38 · 토끼→산토끼 0.43 · 코끼리→캐리 0.50
  → 컷 0.34. 두 글자는 안 바꾼다 — '파요'→'타요' 가 0.25 라 평범한 말이 끌려온다.
오탐이 미탐보다 나쁘다(music.py 머리말): 엉뚱한 노래는 되돌릴 수 없다.
"""
from __future__ import annotations

import logging
import re

from .text_norm import jamo_ratio

log = logging.getLogger("jaeha_bot.song_names")

NAMES = (
    # 캐릭터·시리즈
    "티니핑", "캐치티니핑", "하츄핑", "핑크퐁", "아기상어", "상어가족", "뽀로로", "타요",
    "콩순이", "로보카폴리", "베베핀", "신비아파트", "헬로카봇", "또봇", "미니특공대",
    "슈퍼윙스", "코코몽", "엄마까투리", "번개맨", "브레드이발소", "공룡메카드", "캐리",
    "라바", "치로", "핑크퐁원더스타", "꼬마버스타요", "아기공룡둘리", "짱구",
    # 동요
    "곰세마리", "산토끼", "나비야", "올챙이송", "반짝반짝작은별", "둥글게둥글게",
    "학교종", "머리어깨무릎발", "생일축하노래", "개구리노래", "숲속을걸어요",
    "리틀스타", "자장가", "우리집에왜왔니", "퐁당퐁당", "텔레토비",
)

CUT = 0.34
MIN_LEN = 3
_NOUNS = ("노래", "음악", "동요", "송")
_HANGUL = re.compile(r"[가-힣]")


def transcribe_prompt(names=NAMES) -> str:
    """받아쓰기 힌트. 목록만 준다 — 문장을 주면 그 문장을 받아 적는 일이 있다."""
    return "하이 티드, " + ", ".join(names)


def _best(word: str, names) -> tuple[str, float]:
    best = min(names, key=lambda n: jamo_ratio(word, n))
    return best, jamo_ratio(word, best)


def correct(query: str, names=NAMES) -> str:
    """노래 검색어의 이름 부분을 사전 이름으로 바로잡는다. 확실하지 않으면 그대로."""
    toks = (query or "").split()
    if not toks:
        return query
    out, i = [], 0
    while i < len(toks):
        hit = None
        for span in (3, 2, 1):                       # 긴 것부터 — '티니 핑' 을 한 덩어리로
            win = toks[i:i + span]
            if len(win) < span or any(t in _NOUNS for t in win):
                continue
            word = "".join(win)
            if len(_HANGUL.findall(word)) < MIN_LEN:
                continue
            name, r = _best(word, names)
            if r <= CUT:
                hit = (span, name, r)
                break
        if hit:
            span, name, r = hit
            if r > 0 or span > 1:
                log.info("[노래] 이름 바로잡음: '%s' → '%s' (자모거리 %.2f)",
                         " ".join(toks[i:i + span]), name, r)
            out.append(name)
            i += span
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)
