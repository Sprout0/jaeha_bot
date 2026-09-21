"""깨어 있는 한 구간 — 연결을 열고, 턴을 돌리고, 대기로 갈 이유가 생기면 닫는다.

세 일이 동시에 돈다: 마이크 보내기 / 이벤트 받기 / 시계(맞장구·응답 제한·무응답).
끝나는 이유: sleep(잠들기 명령) · idle(무응답) · music(노래 틀고 대기) ·
music_resumed(노래 중 호출에 답했거나 말이 없었다) · lost(끊김).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .realtime_audio import MicGate, PcmAccumulator, to_pcm16
from .realtime_protocol import (append_audio, cached_tokens, cancel, cost_usd, event_kind,
                                respond, say_exactly, user_text)
from .realtime_session import ConnectionLost
from .realtime_turn import PHRASES, Route, missing_required, route
from .reply_gate import PROPOSE, ReplyGate, TurnGuard

log = logging.getLogger("jaeha_bot.realtime")


@dataclass
class _Pending:
    route: Route
    child: str
    stopped_at: float | None          # 서버가 말끝을 판정한 순간(무음 대기 뒤)
    speech_end_at: float | None       # 실제 말끝 — 체감의 시작점
    transcript_at: float
    requested_at: float
    first_audio_at: float | None = None
    said: str = ""
    filler: bool = False
    acc: PcmAccumulator = field(default_factory=PcmAccumulator)
    # 2026-09-21 안전 가드(spec 2026-09-21-realtime-guard)
    msg: dict | None = None                         # 다시 붙었을 때 같은 방식으로 다시 요청
    guard: TurnGuard | None = None
    held: list = field(default_factory=list)        # 붙잡은 소리 조각
    first_delta_at: float | None = None
    hold_s: float | None = None
    blocked: bool = False
    blocked_at: float | None = None
    block_flags: list = field(default_factory=list)
    reconnects: int = 0


class Conversation:
    def __init__(self, *, session, mic: asyncio.Queue, speaker, cache, cfg, music, games,
                 sleep_words, instructions: str, history: list, metrics=None,
                 sleep_timeout: float = 30.0, filler_phrases=(), clock=time.monotonic,
                 tick_s: float = 0.05, gate=None, guardian=None) -> None:
        self.session, self.mic, self.speaker, self.cache = session, mic, speaker, cache
        self.cfg, self.music, self.games = cfg, music, games
        self.sleep_words, self.instructions = sleep_words, instructions
        self.history, self.metrics = history, metrics
        self.sleep_timeout = float(sleep_timeout)
        self.filler_phrases = list(filler_phrases)
        self.clock, self.tick_s = clock, tick_s
        self.mic_gate = MicGate(cfg.mic_pad_s)
        self.gate = gate or ReplyGate()                 # 답을 소리 전에 붙잡을지·막을지
        self.guardian = guardian                        # 부모 기록(없으면 안 남긴다)
        self._pending: _Pending | None = None
        self._stopped_at: float | None = None
        self._speech_end_at: float | None = None
        # 보낸 오디오 누적 ms → 보낸 벽시계. 서버의 audio_end_ms 를 벽시계로 되돌린다.
        self._sent_log: list[tuple[float, float]] = []
        self._sent_ms = 0.0
        self._last_active = clock()
        self._started = clock()
        self._heard_speech = False
        self._woke_during_music = False
        self._filler_i = 0
        self._done: asyncio.Future | None = None

    # ── 밖에서 부르는 것 ──────────────────────────────────────────────────
    async def run(self, *, preroll=None, greet: bool = True,
                  woke_during_music: bool = False) -> str:
        self._done = asyncio.get_running_loop().create_future()
        self._woke_during_music = woke_during_music
        self._started = self._last_active = self.clock()
        try:
            await self.session.open(self.instructions, self.history)
        except ConnectionLost as e:
            log.warning("Realtime 연결 실패: %s", e)
            self._say_cached("lost")
            await self._wait_quiet()
            return "lost"
        if preroll is not None and np.asarray(preroll[0]).size:
            await self._send_audio(preroll[0], preroll[1])
            log.info("호출 직후 이어진 말 %.2fs 를 먼저 보냄", np.asarray(preroll[0]).size / preroll[1])
        elif greet:
            self._say_cached("wake")
        tasks = [asyncio.create_task(self._pump_mic()),
                 asyncio.create_task(self._read_events()),
                 asyncio.create_task(self._tick())]
        try:
            reason = await self._done
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._wait_quiet()
            await self.session.close()
        log.info("대화 구간 끝 → %s", reason)
        return reason

    # ── 내부 ────────────────────────────────────────────────────────────
    def _finish(self, reason: str) -> None:
        if self._done is not None and not self._done.done():
            self._done.set_result(reason)

    async def _send(self, msg: dict) -> None:
        try:
            await self.session.send(msg)
        except ConnectionLost as e:
            self._lost(e)

    def _lost(self, e) -> None:
        if self._done is not None and self._done.done():
            return
        log.warning("Realtime 끊김: %s", e)
        self._pending = None
        self._say_cached("lost")
        self._finish("lost")

    def _guardian_record(self, kind: str, **fields) -> None:
        if self.guardian is not None:
            self.guardian.record(kind, **fields)

    def _say_cached(self, key: str) -> None:
        audio = self.cache.get(PHRASES[key])
        if audio is None:
            log.warning("고정 문구 캐시가 없다(%s) — 말 없이 넘어간다", key)
            return
        self.speaker.push(audio)

    async def _wait_quiet(self, cap_s: float = 15.0) -> None:
        end = self.clock() + cap_s
        while self.speaker.busy and self.clock() < end:
            await asyncio.sleep(self.tick_s)

    async def _send_audio(self, frame, rate: int) -> None:
        pcm = to_pcm16(frame, rate)
        await self._send(append_audio(pcm))
        self._sent_ms += len(pcm) / 2 / 24000 * 1000
        self._sent_log.append((self._sent_ms, self.clock()))
        if len(self._sent_log) > 4000:
            del self._sent_log[:2000]

    def _wall_at(self, ms: float) -> float | None:
        """보낸 오디오의 ms 위치가 **보내진** 벽시계 시각. 기록 밖이면 None."""
        prev_ms = 0.0
        for cum_ms, wall in self._sent_log:
            if prev_ms <= ms <= cum_ms and ms >= 0:
                return wall - (cum_ms - ms) / 1000.0
            prev_ms = cum_ms
        return None

    async def _pump_mic(self) -> None:
        while True:
            frame, rate = await self.mic.get()
            now = self.clock()
            self.mic_gate.sync(self.speaker.busy, now)
            if self.mic_gate.should_send(now):
                await self._send_audio(frame, rate)

    async def _read_events(self) -> None:
        try:
            async for ev in self.session.events():
                await self._on_event(ev)
        except ConnectionLost as e:
            self._lost(e)

    async def _on_event(self, ev: dict) -> None:
        k, now, p = event_kind(ev), self.clock(), self._pending
        if k == "speech_started":
            self._last_active, self._heard_speech = now, True
        elif k == "speech_stopped":
            self._stopped_at, self._last_active = now, now
            # 🔴 audio_end_ms 는 무음 대기를 포함한다(09-16 실측 +1,295ms). 대기만큼 빼야 실제 말끝.
            end_ms = ev.get("audio_end_ms")
            self._speech_end_at = (self._wall_at(float(end_ms) - self.cfg.silence_ms)
                                   if end_ms is not None else None)
        elif k == "transcript":
            await self._on_transcript((ev.get("transcript") or "").strip(), now)
        elif k == "audio" and p is not None:
            if p.blocked:
                return                                  # 막은 뒤 오는 조각은 버린다
            samples = p.acc.feed(base64.b64decode(ev.get("delta", "")))
            if p.first_delta_at is None:
                p.first_delta_at = now
            if p.guard is not None and p.guard.mode == "hold":
                p.held.append(samples)
            else:
                if p.first_audio_at is None:
                    p.first_audio_at = now
                self.speaker.push(samples)
        elif k == "text" and p is not None:
            if p.blocked:
                return
            p.said += ev.get("delta", "")
            await self._poll_guard(p, now)
        elif k == "done" and p is not None:
            await self._on_done(p, ev.get("response", {}) or {})
        elif k == "error":
            log.warning("Realtime 오류 이벤트: %s", ev.get("error"))

    async def _on_transcript(self, text: str, now: float) -> None:
        self._last_active = now
        if self._pending is not None:
            log.info("[아이] %s (답하는 중이라 이번 말은 넘긴다)", text)
            return
        r = route(text, music=self.music, games=self.games, sleep_words=self.sleep_words)
        if r.kind == "empty":
            log.info("받아 적기 결과 없음 → 계속 듣는다")
            return
        log.info("[아이] %s (%s)", text, r.kind)
        if r.kind == "sleep":
            self._say_cached("sleep")
            self._finish("sleep")
            return
        verbatim = r.kind == "music" or (r.kind.startswith("game") and not r.instructions)
        if verbatim:
            msg = say_exactly(r.say)
        elif r.kind.startswith("game"):
            msg = respond(r.instructions)
        else:
            msg = respond()
        t = self.clock()
        self._pending = _Pending(r, text, self._stopped_at, self._speech_end_at, now, t, msg=msg,
                                 guard=self.gate.begin(text, verbatim=verbatim, now=t))
        if self._pending.guard.held:
            log.info("[안전] 위험 신호 — 답 소리를 붙잡는다: %s", text)
        await self._send(msg)

    # ── 안전 가드 ────────────────────────────────────────────────────────
    async def _poll_guard(self, p: _Pending, now: float) -> None:
        act = p.guard.poll(p.said, now) if p.guard is not None else None
        if act == "release":
            self._release(p, now)
        elif act == "block":
            await self._block(p, now, [PROPOSE], cancel_response=True)

    def _release(self, p: _Pending, now: float) -> None:
        for s in p.held:
            self.speaker.push(s)
        if p.held and p.first_audio_at is None:
            p.first_audio_at = now
        p.held.clear()
        if p.first_delta_at is not None and p.hold_s is None:
            p.hold_s = now - p.first_delta_at

    async def _block(self, p: _Pending, now: float, flags: list, *, cancel_response: bool) -> None:
        leaked = p.first_audio_at is not None
        p.blocked, p.blocked_at, p.block_flags = True, now, list(flags)
        p.held.clear()
        if leaked:
            self.speaker.clear()
        if cancel_response:
            await self._send(cancel())
        self._say_cached("safe")
        if p.first_audio_at is None:
            p.first_audio_at = now          # 안전 문장이 첫 소리 — 응답 제한에 안 걸린다
        log.warning("[안전] 막음 %s ← %s", flags, p.said.strip())
        self._guardian_record("safety_block", child=p.child, reply=p.said.strip(),
                              flags=list(flags), held=bool(p.guard and p.guard.held),
                              leaked=leaked)

    async def _on_done(self, p: _Pending, response: dict) -> None:
        self._pending = None
        self._last_active = self.clock()
        reply, kind, now = p.said.strip(), p.route.kind, self.clock()
        corrected = False
        if p.blocked:
            flags = p.block_flags
        else:
            v = p.guard.finish(reply) if p.guard is not None else None
            flags = v.flags if v else []
            if v is not None and v.action == "release":
                self._release(p, now)
            elif v is not None and v.action == "block":
                await self._block(p, now, v.flags, cancel_response=False)
            elif flags:
                log.warning("[안전] %s ← %s (다 나간 뒤라 기록만)", flags, reply)
                self._guardian_record("safety_late", child=p.child, reply=reply, flags=flags)
            if v is not None and v.fabricated and not p.blocked:
                self._say_cached("cant")
                corrected = True
                self._guardian_record("fabrication", child=p.child, reply=reply,
                                      items=list(v.fabricated))
        missing = missing_required(reply, p.route.require) if p.route.instructions else []
        if missing:
            log.warning("[놀이] 필수 낱말 빠짐 %s ← %s (R2)", missing, reply)
        log.info("[티드] %s (%s)", reply, kind)
        usage = response.get("usage", {}) or {}
        if self.metrics is not None:
            got_audio = p.first_audio_at is not None
            self.metrics.record_realtime_turn(
                kind=kind,
                perceived_s=(p.first_audio_at - p.speech_end_at
                             if got_audio and p.speech_end_at is not None else None),
                vad_tail_s=(p.stopped_at - p.speech_end_at
                            if p.stopped_at is not None and p.speech_end_at is not None else None),
                transcribe_s=(p.transcript_at - p.stopped_at if p.stopped_at is not None else None),
                respond_first_s=(p.first_audio_at - p.requested_at if got_audio else None),
                filler=p.filler, reply=reply, child_text=p.child,
                cost_usd=cost_usd(self.cfg.model, usage), cached_tokens=cached_tokens(usage),
                safety=flags, game_missing=missing,
                held=bool(p.guard and p.guard.held), hold_s=p.hold_s, blocked=p.blocked,
                corrected=corrected, reconnects=p.reconnects)
        if kind in ("chat", "game", "game_start") and (reply or p.blocked):
            said = PHRASES["safe"] if p.blocked else (
                f"{reply} {PHRASES['cant']}" if corrected else reply)
            self.history.append((p.child, said))
            if len(self.history) > self.cfg.history_turns:
                del self.history[:len(self.history) - self.cfg.history_turns]
        if kind == "music":
            await self._wait_quiet()
            mr = p.route.music
            ok = mr.action() if mr.action is not None else True
            if not ok:
                self._say_cached("music_failed")
            if ok and mr.standby:
                self._finish("music")
            return
        if self._woke_during_music and self.music is not None:
            await self._wait_quiet()
            if self.music.resume_after_wake():
                self._finish("music_resumed")

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.tick_s)
            now, p = self.clock(), self._pending
            if p is not None and p.blocked and now - (p.blocked_at or now) >= 2.0:
                await self._on_done(p, {})      # 취소 뒤 done 이 안 와도 턴을 닫는다
                continue
            if p is not None and not p.blocked and p.guard is not None and p.guard.mode == "hold":
                await self._poll_guard(p, now)
            if p is not None and p.first_audio_at is None:
                waited = now - p.requested_at
                if (p.route.kind == "chat" and not p.filler and self.filler_phrases
                        and waited >= self.cfg.filler_after_s):
                    phrase = self.filler_phrases[self._filler_i % len(self.filler_phrases)]
                    self._filler_i += 1
                    audio = self.cache.get(phrase)
                    if audio is not None:
                        self.speaker.push(audio)
                    p.filler = True
                if waited >= self.cfg.response_timeout_s:
                    log.warning("답이 %.1fs 동안 안 왔다 → 취소하고 되묻는다", waited)
                    self._pending = None
                    await self._send(cancel())
                    self._say_cached("recovery")
                    self._last_active = now
                continue
            if p is not None or self.speaker.busy:
                continue
            if (self._woke_during_music and not self._heard_speech
                    and now - self._started >= self.cfg.wake_listen_s):
                if self.music is not None and self.music.resume_after_wake():
                    log.info("노래 중 호출 뒤 말이 없음 → 노래 이어서, 대기로")
                    self._finish("music_resumed")
                    return
            if now - self._last_active >= self.sleep_timeout:
                log.info("무응답 %.0f초 → 대기 모드로", self.sleep_timeout)
                self._say_cached("sleep")
                self._finish("idle")
                return
