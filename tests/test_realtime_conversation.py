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

    def clear(self):
        self.cleared = getattr(self, "cleared", 0) + 1

    def mark(self):
        self.marks = getattr(self, "marks", 0) + 1

    def played(self):
        return getattr(self, "played_n", 0)

    def truncate(self, keep):
        self.truncated = keep


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


# ── 말끝 시각 (2026-09-16 실측) ─────────────────────────────────────────
# 🔴 speech_stopped 는 무음 대기(1.2초)가 **끝난 뒤** 온다. audio_end_ms 도 그 대기를
#    포함한다(실측: 실제 말끝 4,209ms → audio_end_ms 5,504ms). 이벤트 도착 시각부터 재면
#    체감이 1.2초 넘게 좋게 나와 로컬 4.14초와 비교할 수 없다.

def test_보낸_오디오_위치를_벽시계로_바꾼다():
    c = _conv(FakeSession())
    c._sent_log = [(80.0, 10.08), (160.0, 10.16), (240.0, 10.24)]
    assert abs(c._wall_at(120.0) - 10.12) < 1e-9
    assert abs(c._wall_at(240.0) - 10.24) < 1e-9
    assert c._wall_at(500.0) is None and c._wall_at(-5.0) is None


def test_체감은_실제_말끝부터_잰다():
    class Clock:
        t = 100.0
        def __call__(self):
            return self.t

    class Rec:
        def __init__(self):
            self.kw = None
        def record_realtime_turn(self, **kw):
            self.kw = kw

    async def go():
        clock, rec, s = Clock(), Rec(), FakeSession()
        c = Conversation(session=s, mic=asyncio.Queue(), speaker=FakeSpeaker(), cache=FakeCache(),
                         cfg=_cfg(), music=None, games=GameManager(render=None), sleep_words=None,
                         instructions="지시", history=[], metrics=rec, clock=clock, tick_s=0.01)
        c._sent_log = [(1000.0, 101.0), (2000.0, 102.0), (3300.0, 103.3)]
        clock.t = 103.4
        await c._on_event({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 3200})
        clock.t = 104.0
        await c._on_transcript("안녕", clock.t)
        clock.t = 104.5
        await c._on_event(_audio_delta())
        await c._on_event({"type": "response.done", "response": {}})
        return rec.kw

    kw = asyncio.run(go())
    # 실제 말끝 = 3200 - 1200 = 2000ms → 102.0 / 첫 소리 104.5
    assert abs(kw["perceived_s"] - 2.5) < 1e-9
    assert abs(kw["vad_tail_s"] - 1.4) < 1e-9          # 103.4 - 102.0
    assert abs(kw["transcribe_s"] - 0.6) < 1e-9        # 104.0 - 103.4


class FakeGuardian:
    def __init__(self):
        self.records = []

    def record(self, kind, **fields):
        self.records.append((kind, fields))


def _text(t):
    return {"type": "response.output_audio_transcript.delta", "delta": t}


def _done():
    return {"type": "response.done", "response": {"usage": {}}}


def _guard_conv(s, speaker, guardian, history=None, **kw):
    return Conversation(session=s, mic=asyncio.Queue(), speaker=speaker, cache=FakeCache(),
                        cfg=_cfg(), music=None, games=GameManager(render=None), sleep_words=None,
                        instructions="지시", history=history if history is not None else [],
                        sleep_timeout=0.4, filler_phrases=[], tick_s=0.01, guardian=guardian, **kw)


def test_위험_신호_턴은_글자가_끝나기_전엔_소리를_안_낸다():
    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        c = _guard_conv(s, sp, g)
        await c._on_transcript("칼 어딨어?", 0.0)
        await c._on_event(_audio_delta())
        before = len(sp.pushed)
        await c._on_event(_text("칼은 위험해! 엄마한테 말하자."))
        await c._on_event(_done())
        return before, sp, g

    before, sp, g = asyncio.run(go())
    assert before == 0 and len(sp.pushed) == 1 and g.records == []


def test_위험_신호_턴이_걸리면_소리는_버리고_안전_문장과_부모_기록():
    async def go():
        s, sp, g, history = FakeSession(), FakeSpeaker(), FakeGuardian(), []
        c = _guard_conv(s, sp, g, history=history)
        await c._on_transcript("칼 어딨어?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("칼은 부엌에 있어! 같이 찾아볼까?"))
        await c._on_event(_done())
        return c, sp, g, history

    c, sp, g, history = asyncio.run(go())
    assert PHRASES["safe"] in c.cache.asked and len(sp.pushed) == 1   # 안전 문장 하나만
    assert g.records[0][0] == "safety_block" and g.records[0][1]["leaked"] is False
    assert history == [("칼 어딨어?", PHRASES["safe"])]


def test_흘려보내는_턴이_도중에_걸리면_취소하고_스피커를_비운다():
    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        c = _guard_conv(s, sp, g)
        await c._on_transcript("뭐 하고 놀까?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("우리 칼 같이 찾아볼까"))
        await c._on_event(_audio_delta())                  # 막은 뒤 오는 조각은 버린다
        await c._on_event(_done())
        return s, c, sp, g

    s, c, sp, g = asyncio.run(go())
    assert {"type": "response.cancel"} in s.sent and sp.cleared == 1
    assert PHRASES["safe"] in c.cache.asked and len(sp.pushed) == 2      # 새어 나간 1 + 안전 문장
    assert g.records[0][1]["leaked"] is True


def test_못_하는_것이면_답_뒤에_정정한다():
    async def go():
        s, sp, g, history = FakeSession(), FakeSpeaker(), FakeGuardian(), []
        c = _guard_conv(s, sp, g, history=history)
        await c._on_transcript("뭐 하고 놀까?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("좋아! 색칠 놀이 하자!"))
        await c._on_event(_done())
        return c, g, history

    c, g, history = asyncio.run(go())
    assert PHRASES["cant"] in c.cache.asked and g.records[0][0] == "fabrication"
    assert history[0][1].endswith(PHRASES["cant"])


def test_붙잡기_한도가_지나면_시계가_내보낸다():
    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        from app.reply_gate import GuardConfig, ReplyGate
        c = _guard_conv(s, sp, g, gate=ReplyGate(GuardConfig(hold_cap_s=0.05)))
        asyncio.create_task(_feed(s, [_transcript("칼 어딨어?"), _audio_delta(),
                                      _text("칼은")], gap=0.01))
        asyncio.create_task(_feed(s, [_done()], gap=0.25))
        await c.run(greet=False)
        return sp

    assert len(asyncio.run(go()).pushed) >= 1


def test_계측에_붙잡기와_막기가_남는다():
    class Rec:
        kw = None

        def record_realtime_turn(self, **kw):
            Rec.kw = kw

    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        c = _guard_conv(s, sp, g)
        c.metrics = Rec()
        await c._on_transcript("칼 어딨어?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("칼은 부엌에 있어."))
        await c._on_event(_done())

    asyncio.run(go())
    assert Rec.kw["held"] is True and Rec.kw["blocked"] is True and Rec.kw["reconnects"] == 0


class DeadSession(FakeSession):
    async def open(self, instructions, history):
        raise ConnectionLost("안 붙는다")


def test_끊기면_말없이_다시_붙고_답하던_말을_다시_묻는다():
    s1, s2 = FakeSession(), FakeSession()
    made = iter([s2])

    async def go():
        history = [("전", "후")]
        c = _conv(s1, history=history, sleep_timeout=0.4)
        c.make_session, c.reconnect_delays = (lambda: next(made)), (0.01,)
        asyncio.create_task(_feed(s1, [_transcript("공룡 좋아해?"), None]))
        asyncio.create_task(_feed(s2, [_audio_delta(), _text("응 좋아!"), _done()], gap=0.1))
        reason = await c.run(greet=False)
        return c, history, reason

    c, history, reason = asyncio.run(go())
    assert c.reconnects == 1 and reason == "idle"
    assert s2.opened[1][0] == ("전", "후")                      # 기록을 다시 넣었다
    items = [m for m in s2.sent if m["type"] == "conversation.item.create"]
    assert items[0]["item"]["content"][0]["text"] == "공룡 좋아해?"
    assert s2.requests() == [{"type": "response.create"}]
    assert history[-1] == ("공룡 좋아해?", "응 좋아!")
    assert PHRASES["lost"] not in c.cache.asked


def test_다시_붙기가_다_실패하면_lost_와_부모_기록():
    g = FakeGuardian()

    async def go():
        c = _conv(FakeSession())
        c.guardian, c.make_session, c.reconnect_delays = g, DeadSession, (0.01, 0.01, 0.01)
        asyncio.create_task(_feed(c.session, [None]))
        return await c.run(greet=False), c

    reason, c = asyncio.run(go())
    assert reason == "lost" and PHRASES["lost"] in c.cache.asked
    assert g.records[-1][0] == "connection_lost" and g.records[-1][1]["tries"] == 3


def test_첫_연결이_실패해도_다시_시도한다():
    s2 = FakeSession()

    async def go():
        c = _conv(DeadSession(), sleep_timeout=0.1)
        c.make_session, c.reconnect_delays = (lambda: s2), (0.01,)
        return await c.run(greet=False), c

    reason, c = asyncio.run(go())
    assert reason == "idle" and s2.opened is not None


def test_연결_함수가_없으면_옛_동작():
    async def go():
        c = _conv(FakeSession())
        asyncio.create_task(_feed(c.session, [None]))
        return await c.run(greet=False)

    assert asyncio.run(go()) == "lost"


def _game_conv(s, sp, history=None):
    return _guard_conv(s, sp, FakeGuardian(), history=history)


def _start_game(c):
    return c._on_transcript("동물 소리 놀이 하자", 0.0)


def test_놀이_첫_문장에_틀린_이름이면_바로_자르고_조각을_잇는다():
    async def go():
        s, sp, history = FakeSession(), FakeSpeaker(), []
        c = _game_conv(s, sp, history)
        await _start_game(c)
        beat = c._pending.route.beat
        wrong = next(n for n in beat["names"] if n not in beat["allow"])
        await c._on_event(_audio_delta())
        await c._on_event(_text(f"{wrong}는 "))
        await c._on_event(_done())
        return s, sp, c, beat, history

    s, sp, c, beat, history = asyncio.run(go())
    assert {"type": "response.cancel"} in s.sent and hasattr(sp, "truncated")
    assert beat["fix_react"] in c.cache.asked and beat["fix_next"] in c.cache.asked
    assert history[-1][1].endswith(beat["fix_next"])


def test_놀이_질문이_빠지면_끝에서_질문만_잇는다():
    async def go():
        s, sp = FakeSession(), FakeSpeaker()
        c = _game_conv(s, sp)
        await _start_game(c)
        beat = c._pending.route.beat
        await c._on_event(_audio_delta())
        await c._on_event(_text("좋아, 동물 소리 놀이 하자! 이제 따라 해볼까?"))
        await c._on_event(_done())
        return s, sp, c, beat

    s, sp, c, beat = asyncio.run(go())
    assert {"type": "response.cancel"} not in s.sent
    assert beat["fix_next"] in c.cache.asked and beat["fix_react"] not in c.cache.asked
    assert hasattr(sp, "truncated")


def test_놀이가_제대로면_아무것도_안_한다():
    async def go():
        s, sp = FakeSession(), FakeSpeaker()
        c = _game_conv(s, sp)
        await _start_game(c)
        beat = c._pending.route.beat
        await c._on_event(_audio_delta())
        await c._on_event(_text(f"좋아, 동물 소리 놀이 하자! {beat['fix_next']}"))
        await c._on_event(_done())
        return sp, c, beat

    sp, c, beat = asyncio.run(go())
    assert not hasattr(sp, "truncated") and beat["fix_next"] not in c.cache.asked
