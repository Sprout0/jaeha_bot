"""놀이 턴 대본 이탈 — 어디서 자르고 무엇을 이어 붙일지(2026-09-22). 소리·소켓을 모른다.

spec: docs/superpowers/specs/2026-09-22-realtime-turn-fixes-design.md §3
실측(09-21 실서버 놀이 8턴): 동물 이름 글자가 스피커보다 중앙 0.98s·최소 0.14s 앞서 온다.
  첫 문장 안의 틀린 이름 → 글자가 온 즉시(check_stream)
  둘째 문장의 틀린 이름·질문 빠짐 → done 에서(check_done) — 둘째 문장은 1~2s 뒤에 들린다
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_PARTICLE = "은는이가을를도랑야와과의에한"
_END = re.compile(r"[.!?](\s|$)")


@dataclass
class Repair:
    cut_char: int
    lines: list[str] = field(default_factory=list)
    reason: str = ""
    in_first: bool = False


def find_name(text: str, name: str) -> int:
    """낱말 경계로 찾는다(조사 허용). '소'가 '소리'에 걸리지 않게."""
    m = re.search(rf"(?<![가-힣]){re.escape(name)}(?=[{_PARTICLE}]?(?![가-힣]))", text or "")
    return m.start() if m else -1


def first_sentence_end(text: str) -> int:
    m = _END.search(text or "")
    return m.start() + 1 if m else -1


def _wrong_name(said: str, beat: dict, *, partial: bool = False) -> int:
    """허용 밖 이름의 첫 위치. partial=True(글자가 아직 오는 중)면 **끝에 걸린 이름은 미룬다** —
    09-22 실서버에서 "좋아, 동물 소" 의 '소'(= 소리의 앞 글자)를 틀린 이름으로 봤다."""
    allow = set(beat.get("allow") or [])
    end = len(said.rstrip())
    hits = [p for n in beat.get("names") or []
            if n not in allow and (p := find_name(said, n)) >= 0
            and not (partial and p + len(n) >= end)]
    return min(hits) if hits else -1


def _lines(beat: dict, in_first: bool) -> list[str]:
    lines = [beat.get("fix_react", ""), beat.get("fix_next", "")] if in_first else [beat.get("fix_next", "")]
    return [x for x in lines if x]


def check_stream(said: str, beat: dict) -> Repair | None:
    """글자가 들어오는 도중 — 첫 문장 안의 틀린 이름만 본다(여유가 0.14s 뿐이라 기다릴 수 없다)."""
    pos = _wrong_name(said, beat, partial=True)
    if pos < 0:
        return None
    end = first_sentence_end(said)
    if end != -1 and pos >= end:
        return None                       # 둘째 문장 — done 에서 정확히 자른다
    lines = _lines(beat, True)
    return Repair(pos, lines, "wrong_name", True) if lines else None


def check_done(said: str, beat: dict) -> Repair | None:
    """글자가 다 왔다 — 틀린 이름(어디든) 또는 다음 질문 빠짐."""
    end = first_sentence_end(said)
    pos = _wrong_name(said, beat)
    if pos >= 0:
        in_first = end == -1 or pos < end
        lines = _lines(beat, in_first)
        return Repair(pos if in_first else end, lines, "wrong_name", in_first) if lines else None
    need = beat.get("next_need") or []
    if need and not all(tok in said for tok in need):
        lines = _lines(beat, False)
        return Repair(end if end != -1 else len(said), lines, "missing_next") if lines else None
    return None


def quiet_cut(audio: np.ndarray, near: int, sr: int, *, before_s: float = 0.5,
              after_s: float = 0.5) -> int:
    """near 근처(앞 before_s·뒤 after_s)에서 가장 조용한 30ms 의 가운데 샘플."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    win = max(1, int(0.03 * sr))
    lo = max(0, near - int(before_s * sr))
    hi = min(a.size, near + int(after_s * sr))
    if hi - lo < win:
        return max(0, min(near, a.size))
    seg = a[lo:hi]
    n = (seg.size - win) // win + 1
    energy = [float(np.mean(seg[i * win:(i + 1) * win] ** 2)) for i in range(n)]
    best = int(np.argmin(energy))
    return lo + best * win + win // 2
