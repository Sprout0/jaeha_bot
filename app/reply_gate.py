"""답이 소리로 나가기 전에 — 붙잡을지, 막을지, 정정할지 판단한다. (2026-09-21)

spec: docs/superpowers/specs/2026-09-21-realtime-guard-design.md
소리·소켓·시계를 모른다. 시각은 인자로 받는다 — 그래서 소리 없이 시험한다.

턴마다 TurnGuard 하나:
  hold   : 아이 말에 위험 신호 → 소리를 모아 두고 글자가 끝나면 검사한다
  stream : 평소 → 소리는 바로 나가고, 글자 도중 '위험행동제안'이면 끊는다
  off    : 우리가 쓴 문장을 그대로 읽는 턴(노래 안내·놀이 템플릿) → 안 본다
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields

from . import claims, safety

log = logging.getLogger("jaeha_bot.reply_gate")

EMPTY = "빈응답"
PROPOSE = "위험행동제안"      # 글자 도중에도 판정할 수 있는 유일한 것(위험물 × 행동 유도 조합)
JUDGE_ERROR = "판정오류"


@dataclass
class GuardConfig:
    hold_on_risk: bool = True
    hold_cap_s: float = 3.0
    reconnect_tries: int = 3
    safe_line: str = "그건 위험해! 엄마 아빠한테 같이 가자."
    cant_line: str = "아, 그건 아직 못 해. 대신 동물 소리 놀이 할까?"

    @classmethod
    def from_dict(cls, d: dict | None) -> "GuardConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class Verdict:
    action: str                                   # release | block | pass
    flags: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)


def _flags(said: str, child: str) -> list[str]:
    return [f for f in safety.check_reply(said, child_text=child) if f != EMPTY]


class TurnGuard:
    def __init__(self, child: str, *, mode: str, cap_s: float, started: float) -> None:
        self.child, self.mode, self.cap_s, self.started = child, mode, float(cap_s), started
        self.held = mode == "hold"

    def poll(self, said: str, now: float) -> str | None:
        """글자가 더 왔거나 시간이 흘렀다. 'release'(모은 소리 내보내기) / 'block' / None."""
        if self.mode == "off" or (self.mode == "hold" and now - self.started < self.cap_s):
            return None
        if not said:
            if self.mode == "hold":             # 한도가 지났는데 글자가 없다 — 소리만이라도
                self.mode = "stream"
                return "release"
            return None
        try:
            proposes = PROPOSE in _flags(said, self.child)
        except Exception as e:                  # noqa: BLE001 — 판정기가 봇을 죽이면 안 된다
            log.warning("[안전] 도중 판정 실패: %s", e)
            proposes = self.mode == "hold"      # 붙잡은 턴은 안전 쪽으로
        if proposes:
            return "block"
        if self.mode == "hold":
            self.mode = "stream"
            return "release"
        return None

    def finish(self, said: str) -> Verdict:
        """답의 글자가 끝났다."""
        if self.mode == "off":
            return Verdict("pass")
        try:
            flags = _flags(said, self.child) if said else []
        except Exception as e:                  # noqa: BLE001
            log.warning("[안전] 끝 판정 실패: %s", e)
            flags = [JUDGE_ERROR] if self.mode == "hold" else []
        if self.mode == "hold" and flags:
            return Verdict("block", flags)
        try:
            fabricated = claims.find_fabrications(said) if said else []
        except Exception as e:                  # noqa: BLE001
            log.warning("[정정] 판정 실패: %s", e)
            fabricated = []
        return Verdict("release" if self.mode == "hold" else "pass", flags, fabricated)


class ReplyGate:
    def __init__(self, cfg: GuardConfig | None = None) -> None:
        self.cfg = cfg or GuardConfig()

    def begin(self, child: str, *, verbatim: bool, now: float) -> TurnGuard:
        if verbatim:
            mode = "off"
        elif self.cfg.hold_on_risk and safety.question_risk(child):
            mode = "hold"
        else:
            mode = "stream"
        return TurnGuard(child, mode=mode, cap_s=self.cfg.hold_cap_s, started=now)
