import asyncio
import base64
import json

import numpy as np
import pytest

from app.realtime_protocol import RealtimeConfig
from app.realtime_session import ConnectionLost, RealtimeSession, synthesize


class FakeWS:
    def __init__(self, incoming=(), fail_send=False):
        self.sent, self.closed = [], False
        self._in = list(incoming)
        self.fail_send = fail_send

    async def send(self, raw):
        if self.fail_send:
            raise OSError("끊김")
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._in:
            raise StopAsyncIteration
        return json.dumps(self._in.pop(0))


def _connect(ws, seen):
    async def connect(url, headers):
        seen.update(url=url, headers=headers)
        return ws
    return connect


def test_열면_세션설정과_이력을_보낸다():
    ws, seen = FakeWS(), {}
    s = RealtimeSession(RealtimeConfig(), "키", connect=_connect(ws, seen))
    asyncio.run(s.open("지시", [("안녕", "안녕!")]))
    assert "gpt-realtime-mini" in seen["url"] and seen["headers"]["Authorization"] == "Bearer 키"
    assert [m["type"] for m in ws.sent] == ["session.update", "conversation.item.create",
                                           "conversation.item.create"]


def test_보내기_실패는_끊김이다():
    s = RealtimeSession(RealtimeConfig(), "키", connect=_connect(FakeWS(fail_send=True), {}))
    with pytest.raises(ConnectionLost):
        asyncio.run(s.open("지시", []))


def test_이벤트가_끝나면_끊김이다():
    s = RealtimeSession(RealtimeConfig(), "키", connect=_connect(FakeWS([{"type": "session.updated"}]), {}))

    async def go():
        await s.open("지시", [])
        got = []
        with pytest.raises(ConnectionLost):
            async for ev in s.events():
                got.append(ev["type"])
        return got

    assert asyncio.run(go()) == ["session.updated"]


def test_문장_하나를_소리로_만든다():
    pcm = (np.full(480, 0.25) * 32767).astype("<i2").tobytes()
    ws = FakeWS([{"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()},
                 {"type": "response.done"}])
    audio = asyncio.run(synthesize(RealtimeConfig(), "키", "안녕", connect=_connect(ws, {})))
    assert audio.dtype == np.float32 and audio.size == 480
    assert ws.sent[-1]["response"]["conversation"] == "none" and ws.closed


def test_소리가_안_오면_실패다():
    with pytest.raises(RuntimeError):
        asyncio.run(synthesize(RealtimeConfig(), "키", "안녕",
                               connect=_connect(FakeWS([{"type": "response.done"}]), {})))
