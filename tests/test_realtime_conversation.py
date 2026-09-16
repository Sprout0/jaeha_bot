import asyncio
import base64

import numpy as np

from app.education_modes import GameManager
from app.music import MusicReply
from app.realtime_conversation import Conversation
from app.realtime_protocol import RealtimeConfig
from app.realtime_session import ConnectionLost
from app.realtime_turn import PHRASES


class FakeSession:
    def __init__(self):
        self.sent, self.closed, self.opened = [], False, None
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def open(self, instructions, history):
        self.opened = (instructions, list(history))

    async def send(self, msg):
        self.sent.append(msg)

    async def events(self):
        while True:
            ev = await self.inbox.get()
            if ev is None:
                raise ConnectionLost("끝")
            yield ev

    async def close(self):
        self.closed = True

    def requests(self):
        return [m for m in self.sent if m["type"] == "response.create"]


class FakeSpeaker:
    busy = False

    def __init__(self):
        self.pushed = []

    def push(self, samples):
        self.pushed.append(np.asarray(samples))


class FakeCache:
    def __init__(self):
        self.asked = []

    def get(self, phrase):
        self.asked.append(phrase)
        return np.full(10, 0.1, dtype=np.float32)


def _cfg():
    return RealtimeConfig(filler_after_s=0.05, response_timeout_s=0.3, wake_listen_s=0.2)


def _audio_delta():
    pcm = (np.full(240, 0.2) * 32767).astype("<i2").tobytes()
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}


def _transcript(text):
    return {"type": "conversation.item.input_audio_transcription.completed", "transcript": text}


def _conv(session, *, music=None, cache=None, speaker=None, sleep_timeout=5.0, history=None):
    return Conversation(session=session, mic=asyncio.Queue(), speaker=speaker or FakeSpeaker(),
                        cache=cache or FakeCache(), cfg=_cfg(), music=music,
                        games=GameManager(render=None), sleep_words=None, instructions="지시",
                        history=history if history is not None else [],
                        sleep_timeout=sleep_timeout, filler_phrases=["음..."], tick_s=0.01)


async def _feed(session, events, gap=0.02):
    for ev in events:
        await asyncio.sleep(gap)
        await session.inbox.put(ev)


def test_자유대화는_답을_요청하고_이력에_남긴다():
    async def go():
        s, history = FakeSession(), []
        c = _conv(s, history=history, sleep_timeout=0.3)
        asyncio.create_task(_feed(s, [
            {"type": "input_audio_buffer.speech_stopped"}, _transcript("안녕"), _audio_delta(),
            {"type": "response.output_audio_transcript.delta", "delta": "안녕!"},
            {"type": "response.done", "response": {"usage": {}}}]))
        return s, history, await c.run(greet=False)

    s, history, reason = asyncio.run(go())
    assert s.requests() == [{"type": "response.create"}]
    assert history == [("안녕", "안녕!")] and reason == "idle" and s.closed


def test_잠들기는_문구를_틀고_끝난다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache)
        asyncio.create_task(_feed(s, [_transcript("바이바이")]))
        return await c.run(greet=False), s, cache

    reason, s, cache = asyncio.run(go())
    assert reason == "sleep" and s.requests() == [] and PHRASES["sleep"] in cache.asked


def test_노래는_안내를_그대로_말하고_틀고_대기로():
    played = []

    class Music:
        def handle(self, text):
            return MusicReply("상어가족 틀어 줄게!", action=lambda: played.append(1) or True,
                              standby=True)

    async def go():
        s = FakeSession()
        c = _conv(s, music=Music())
        asyncio.create_task(_feed(s, [_transcript("상어가족 틀어줘"), _audio_delta(),
                                      {"type": "response.done", "response": {}}]))
        return await c.run(greet=False), s

    reason, s = asyncio.run(go())
    req = s.requests()[0]["response"]
    assert req["conversation"] == "none" and "상어가족 틀어 줄게!" in req["input"][0]["content"][0]["text"]
    assert played == [1] and reason == "music"


def test_놀이_시작은_지시를_붙여_요청한다():
    async def go():
        s = FakeSession()
        c = _conv(s, sleep_timeout=0.3)
        asyncio.create_task(_feed(s, [_transcript("동물 소리 놀이 하자"), _audio_delta(),
                                      {"type": "response.done", "response": {}}]))
        await c.run(greet=False)
        return s

    assert "상황:" in asyncio.run(go()).requests()[0]["response"]["instructions"]


def test_답이_늦으면_맞장구를_튼다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache, sleep_timeout=0.4)
        asyncio.create_task(_feed(s, [_transcript("안녕")]))
        asyncio.create_task(_feed(s, [_audio_delta(), {"type": "response.done", "response": {}}],
                                  gap=0.15))
        await c.run(greet=False)
        return cache

    assert "음..." in asyncio.run(go()).asked


def test_답이_안_오면_취소하고_되묻는다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache, sleep_timeout=0.6)
        asyncio.create_task(_feed(s, [_transcript("안녕")]))
        await c.run(greet=False)
        return s, cache

    s, cache = asyncio.run(go())
    assert {"type": "response.cancel"} in s.sent and PHRASES["recovery"] in cache.asked


def test_끊기면_문구를_틀고_끝난다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache)
        asyncio.create_task(_feed(s, [None]))
        return await c.run(greet=False), cache

    reason, cache = asyncio.run(go())
    assert reason == "lost" and PHRASES["lost"] in cache.asked


def test_노래_중_호출에_말이_없으면_노래로_돌아간다():
    class Music:
        resumed = 0

        def handle(self, text):
            return None

        def resume_after_wake(self):
            Music.resumed += 1
            return True

    async def go():
        return await _conv(FakeSession(), music=Music()).run(greet=False, woke_during_music=True)

    assert asyncio.run(go()) == "music_resumed" and Music.resumed == 1


async def _pump_once(conv):
    await conv.mic.put((np.zeros(1280, dtype=np.float32), 16000))
    t = asyncio.create_task(conv._pump_mic())
    await asyncio.sleep(0.05)
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)


def test_마이크는_재생_중에는_안_보낸다():
    class BusySpeaker(FakeSpeaker):
        busy = True

    async def go():
        s = FakeSession()
        await _pump_once(_conv(s, speaker=BusySpeaker()))
        return s

    assert asyncio.run(go()).sent == []


def test_마이크는_조용하면_24k_로_보낸다():
    async def go():
        s = FakeSession()
        await _pump_once(_conv(s))
        return s

    sent = asyncio.run(go()).sent
    assert len(sent) == 1 and sent[0]["type"] == "input_audio_buffer.append"


def test_호출_직후_이어진_말을_먼저_보낸다():
    async def go():
        s = FakeSession()
        await _conv(s, sleep_timeout=0.1).run(preroll=(np.zeros(1600, dtype=np.float32), 16000),
                                              greet=False)
        return s

    assert asyncio.run(go()).sent[0]["type"] == "input_audio_buffer.append"
