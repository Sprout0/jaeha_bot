"""부모 기록 — 막은 일·정정한 일·끊김 복구 실패를 남긴다. (2026-09-21)

logs/guardian_YYYYMMDD.jsonl 에 한 줄씩. **소리는 남기지 않는다** — 글자만.
notify() 는 나중에 텔레그램을 끼우는 자리다. 기본은 아무것도 안 보낸다.
무엇을 기기 밖으로 내보낼지(원문/요약)는 그걸 붙일 때 정한다.
기록·전송이 실패해도 대화는 멈추지 않는다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("jaeha_bot.guardian")


class NullNotifier:
    def notify(self, event: dict) -> None:
        return None


class GuardianLog:
    def __init__(self, log_dir, notifier=None, now=datetime.now) -> None:
        self.dir = Path(log_dir)
        self.notifier = notifier or NullNotifier()
        self._now = now

    def record(self, kind: str, **fields) -> dict:
        t = self._now()
        ev = {"ts": t.isoformat(timespec="seconds"), "kind": kind, **fields}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f"guardian_{t:%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception as e:                  # noqa: BLE001
            log.warning("부모 기록을 못 썼다: %s", e)
        try:
            self.notifier.notify(ev)
        except Exception as e:                  # noqa: BLE001
            log.warning("부모 알림을 못 보냈다: %s", e)
        log.info("[부모 기록] %s %s", kind, {k: v for k, v in fields.items() if k != "reply"})
        return ev
