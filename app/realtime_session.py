"""Realtime 연결 하나. 소켓 라이브러리는 여기에만 닿는다 — 나머지는 dict 만 주고받는다."""
from __future__ import annotations

import base64
import json
import logging

import numpy as np

from .realtime_audio import PcmAccumulator
from .realtime_protocol import (READER, URL, RealtimeConfig, event_kind, history_items,
                                say_exactly, session_update)

log = logging.getLogger("jaeha_bot.realtime")


class ConnectionLost(Exception):
    """소켓이 닫혔거나 보내기가 실패했다. 대화 구간은 이걸 받으면 대기로 간다."""


async def _ws_connect(url: str, headers: dict):
    import websockets
    return await websockets.connect(url, additional_headers=headers, max_size=None)


class RealtimeSession:
    def __init__(self, cfg: RealtimeConfig, api_key: str, connect=None) -> None:
        self.cfg = cfg
        self._key = api_key
        self._connect = connect or _ws_connect
        self._ws = None

    async def open(self, instructions: str, history: list[tuple[str, str]]) -> None:
        try:
            self._ws = await self._connect(URL.format(model=self.cfg.model),
                                           {"Authorization": f"Bearer {self._key}"})
        except Exception as e:
            raise ConnectionLost(f"연결 실패: {type(e).__name__}: {e}") from e
        await self.send(session_update(self.cfg, instructions))
        for item in history_items(history, self.cfg.history_turns):
            await self.send(item)

    async def send(self, msg: dict) -> None:
        if self._ws is None:
            raise ConnectionLost("열리지 않았다")
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            raise ConnectionLost(f"보내기 실패: {type(e).__name__}: {e}") from e

    async def events(self):
        if self._ws is None:
            raise ConnectionLost("열리지 않았다")
        try:
            async for raw in self._ws:
                yield json.loads(raw)
        except Exception as e:
            raise ConnectionLost(f"받기 실패: {type(e).__name__}: {e}") from e
        raise ConnectionLost("서버가 연결을 닫았다")

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass


async def synthesize(cfg: RealtimeConfig, api_key: str, line: str, connect=None) -> np.ndarray:
    """문장 하나를 marin 목소리 24k float32 로. 고정 문구 캐시를 만들 때 쓴다."""
    s = RealtimeSession(cfg, api_key, connect=connect)
    acc, chunks = PcmAccumulator(), []
    try:
        await s.open(READER, [])
        await s.send(say_exactly(line))
        try:
            async for ev in s.events():
                k = event_kind(ev)
                if k == "audio":
                    chunks.append(acc.feed(base64.b64decode(ev["delta"])))
                elif k == "error":
                    raise RuntimeError(f"만들기 실패: {ev.get('error')}")
                elif k == "done":
                    break
        except ConnectionLost:
            pass
    finally:
        await s.close()
    total = sum(c.size for c in chunks)
    if not total:
        raise RuntimeError(f"소리가 안 왔다: {line}")
    return np.concatenate(chunks).astype(np.float32)
