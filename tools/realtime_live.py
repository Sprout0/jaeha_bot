"""OpenAI Realtime 라이브 루프 — 마이크 ↔ 소켓 ↔ 스피커.

09-02 에 파일을 소켓에 밀어 넣어 지연·CER·비용·안전여유는 이미 쟀다. 여기서 재는 건
**라이브로 돌렸을 때만 드러나는 것** 셋이다: 에코 / 바지인 / 리샘플.
"""
from __future__ import annotations

import numpy as np

REMAINDER = 2          # pcm16 한 샘플 = 2 바이트
FULL_SCALE = 32768.0


class MicGate:
    """반이중. 봇이 말하는 동안(+뒤로 pad 만큼) 마이크를 서버에 보내지 않는다.

    서버 VAD 는 우리가 보낸 것만 듣는다. 재생 중에 열어 두면 봇이 제 말에 반응한다.
    pad 는 스피커 잔향과 장치 버퍼 몫 — 08-24 실측으로 정한 0.150s 를 쓴다.
    """

    def __init__(self, pad_s: float = 0.15) -> None:
        self.pad_s = pad_s
        self._playing = False
        self._ended_at: float | None = None

    def playing_started(self, at: float) -> None:
        self._playing = True
        self._ended_at = None

    def playing_ended(self, at: float) -> None:
        self._playing = False
        self._ended_at = at

    def sync(self, playing: bool, now: float) -> None:
        """스피커 상태를 그대로 넘긴다. **가장자리에서만** 시계를 잡는다 —
        매번 잡으면 pad 가 영원히 안 지나 마이크가 다시 안 열린다."""
        if playing:
            self.playing_started(now)
        elif self._playing:
            self.playing_ended(now)

    def should_send(self, now: float) -> bool:
        if self._playing:
            return False
        if self._ended_at is None:
            return True
        return now - self._ended_at >= self.pad_s


class PcmAccumulator:
    """pcm16 리틀엔디언 바이트를 float32(-1..1) 로. 델타는 샘플 중간에서 잘려 온다."""

    def __init__(self) -> None:
        self._tail = b""

    def feed(self, chunk: bytes) -> np.ndarray:
        buf = self._tail + chunk
        n = len(buf) - len(buf) % REMAINDER
        self._tail = buf[n:]
        if not n:
            return np.zeros(0, dtype=np.float32)
        return (np.frombuffer(buf[:n], dtype="<i2").astype(np.float32) / FULL_SCALE)


def blind_labels(voices: list[str], seed: int) -> dict[str, str]:
    """목소리 -> 정체를 숨긴 파일명. 씨앗만 있으면 되살릴 수 있다."""
    import random

    order = list(range(len(voices)))
    random.Random(seed).shuffle(order)
    return {v: f"voice_{chr(ord('A') + order[i])}" for i, v in enumerate(voices)}


# ── 여기서부터는 장치·소켓이라 단위 시험이 안 붙는다. 실제로 돌려서 본다. ──────
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio_player import _resample                      # noqa: E402
from tools.realtime_probe import (CHUNK_MS, PRICE, SR, connect,  # noqa: E402
                                  cost_usd, session_config)

AUDIO_DELTA = ("response.audio.delta", "response.output_audio.delta")
AUDIO_DONE = ("response.audio.done", "response.output_audio.done")
TEXT_DELTA = ("response.audio_transcript.delta",
              "response.output_audio_transcript.delta")
VOICES = ["alloy", "ash", "ballad", "cedar", "coral", "echo", "marin", "sage",
          "shimmer", "verse"]
READER = ("너는 낭독자다. 사용자가 준 문장을 **글자 그대로** 한 번만 읽어라. "
          "덧붙이거나 줄이거나 바꾸지 마라.")
SAMPLE_LINE = "안녕! 나는 티드야. 오늘 뭐 하고 놀까? 같이 노래도 부를 수 있어."


def resolve_rate(kind: str, want: int) -> int:
    """장치가 want 를 받으면 그대로, 아니면 장치 기본값.

    젯슨 ReSpeaker 는 16000 전용이고 Realtime 은 24000 규격이라 양방향 리샘플이
    필요하다. 노트북은 보통 24000 을 그대로 받아 리샘플이 0 회다 — 그래서 이 값을
    반드시 찍어 본다. 노트북에서 안 겪은 문제를 젯슨에서 처음 만나면 안 된다.
    """
    import sounddevice as sd

    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    try:
        check(samplerate=want)
        return want
    except Exception:
        dev = sd.default.device
        idx = dev[0 if kind == "input" else 1] if isinstance(dev, (list, tuple)) else dev
        try:
            return int(sd.query_devices(idx, kind)["default_samplerate"])
        except Exception:
            return want


class Speaker:
    """소켓에서 오는 조각을 끊김 없이 낸다.

    `SoundDeviceSink.play()` 는 호출마다 앞 재생을 끊어서 못 쓴다(조각마다 부르면
    마지막 조각만 들린다). 콜백 스트림 + 대기열로 이어 붙인다.
    """

    def __init__(self, src_rate: int = SR) -> None:
        import sounddevice as sd

        self.src_rate = src_rate
        self.rate = resolve_rate("output", src_rate)
        self.resampled = self.rate != src_rate
        self._q: deque = deque()
        self._left = np.zeros(0, dtype=np.float32)
        self._stream = sd.OutputStream(samplerate=self.rate, channels=1,
                                       dtype="float32", callback=self._cb)
        self._stream.start()

    def _cb(self, out, frames, time_info, status) -> None:
        need, got = frames, []
        while need > 0:
            if self._left.size == 0:
                if not self._q:
                    break
                self._left = self._q.popleft()
            take = min(need, self._left.size)
            got.append(self._left[:take])
            self._left = self._left[take:]
            need -= take
        block = np.concatenate(got) if got else np.zeros(0, dtype=np.float32)
        if block.size < frames:                       # 남으면 무음으로 채운다
            block = np.concatenate([block, np.zeros(frames - block.size, dtype=np.float32)])
        out[:, 0] = block

    def push(self, samples: np.ndarray) -> None:
        if self.resampled:
            samples = _resample(samples, self.src_rate, self.rate)
        self._q.append(samples)

    @property
    def busy(self) -> bool:
        return bool(self._q) or self._left.size > 0

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class FileSink:
    """스피커가 없을 때 쓰는 대역. 봇 목소리를 재생 대신 wav 로 받는다.

    ⚠️ 이 모드로는 **에코를 못 잰다** — 소리가 안 나가니 마이크가 되들을 것도 없다.
       08-24 에 이어폰으로 대신해 보려다 마이크가 8회 중 0회 검출해 이미 닫힌 길이다.
       여기서 확인되는 건 마이크 쪽 절반(포착·전송·서버 VAD·전사·지연)뿐이다.
    """

    rate = SR
    resampled = False

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._chunks: list = []

    def push(self, samples: np.ndarray) -> None:
        self._chunks.append(samples)

    @property
    def busy(self) -> bool:
        return False        # 재생이 없다. True 를 주면 마이크가 영영 안 열린다.

    def close(self) -> None:
        if not self._chunks:
            return
        import soundfile as sf
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.path, np.concatenate(self._chunks), SR)


async def sample_voices(model: str, voices: list[str], line: str, out: Path,
                        seed: int) -> dict:
    """1단계. 목소리마다 같은 문장을 말하게 하고 wav 로 남긴다 — **블라인드로**.

    Realtime 목소리는 고정이라 못 바꾼다. 아이가 싫어하면 나머지 측정이 전부
    무의미해지므로 여기부터 본다. 08-25 에 목소리 F2 를 블라인드 4라운드로 골랐던
    것과 같은 규율: 파일명에 목소리 이름을 안 적는다.
    """
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    labels = blind_labels(voices, seed)
    rows: dict = {}
    for v in voices:
        cfg = session_config(1200, "gpt-4o-transcribe", v)
        # 🔴 세션 기본 인격은 "30자 안쪽으로 짧게"라 이 문장과 싸운다. 실제로 10개 중
        #    9개가 뒷문장을 잘라 먹어 표본 길이가 3.0~5.8s 로 벌어졌다 — 목소리가 아니라
        #    말 길이를 비교하게 된다. 표본 뽑을 때만 낭독자로 바꾼다.
        cfg["session"]["instructions"] = READER
        ws = await connect(model, cfg)
        acc = PcmAccumulator()
        chunks, said = [], ""
        try:
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": f"정확히 이렇게만 말해: {line}"}]}}))
            await ws.send(json.dumps({"type": "response.create"}))
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t in AUDIO_DELTA:
                    chunks.append(acc.feed(base64.b64decode(ev["delta"])))
                elif t in TEXT_DELTA:
                    said += ev.get("delta", "")
                elif t == "response.done":
                    break
                elif t == "error":
                    print(f"  ❌ {v}: {ev.get('error')}")
                    break
        finally:
            await ws.close()
        if not chunks:
            print(f"  ❌ {v}: 소리가 안 왔다")
            continue
        path = out / f"{labels[v]}.wav"
        samples = np.concatenate(chunks)
        sf.write(path, samples, SR)
        said = said.strip()
        rows[labels[v]] = {"voice": v, "seconds": round(samples.size / SR, 2),
                           "said": said}
        short = "  ⚠️ 문장을 다 안 말했다" if said and said != line else ""
        print(f"  {path.name}  ({samples.size / SR:.1f}s){short}")

    key = out / "정답표.json"
    key.write_text(json.dumps({"seed": seed, "line": line, "samples": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  정답표: {key}  ← 다 듣고 고른 뒤에 열 것")
    return labels


async def live(model: str, cfg: dict, seconds: float, duplex: str,
               pad_s: float, out_wav: Path | None = None) -> dict:
    """2단계. 마이크 ↔ 소켓 ↔ 스피커 한 바퀴.

    duplex=half : 봇이 말하는 동안 마이크를 안 보낸다(지금 봇과 같은 구조).
    duplex=full : 계속 보낸다. **에코가 나는지 재는 모드** — 봇이 말하는 중에
                  서버가 발화 시작을 잡으면 그게 에코다. 그 횟수를 센다.
    out_wav     : 주면 재생하지 않고 파일로 받는다(스피커 없는 기계용).
    """
    import sounddevice as sd

    in_rate = resolve_rate("input", SR)
    spk = FileSink(out_wav) if out_wav else Speaker(SR)
    gate = MicGate(pad_s)
    print(f"  마이크 {in_rate}Hz{' → 24000 리샘플' if in_rate != SR else ''} | "
          f"스피커 {spk.rate}Hz{' ← 24000 리샘플' if spk.resampled else ''} | {duplex}")
    if out_wav:
        print(f"  ⚠️ 재생 없음 → {out_wav} 로 받는다. 이 모드로는 에코를 못 잰다.")

    ws = await connect(model, cfg)
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stats = {"echo": 0, "turns": 0, "sent_s": 0.0, "dropped_s": 0.0,
             "in_rate": in_rate, "out_rate": spk.rate, "duplex": duplex,
             "cost_usd": 0.0, "lead_s": [], "wav": str(out_wav) if out_wav else None}

    def on_mic(indata, frames, tinfo, status):
        loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

    stream = sd.InputStream(samplerate=in_rate, channels=1, dtype="float32",
                            blocksize=int(in_rate * CHUNK_MS / 1000), callback=on_mic)
    stream.start()

    async def pump_mic():
        while True:
            block = await q.get()
            now = time.monotonic()
            gate.sync(spk.busy, now)
            dur = block.size / in_rate
            if duplex == "half" and not gate.should_send(now):
                stats["dropped_s"] += dur
                continue
            if in_rate != SR:
                block = _resample(block, in_rate, SR)
            pcm = (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm).decode()}))
            stats["sent_s"] += dur

    async def read_events():
        acc = PcmAccumulator()
        said, first_text = "", None
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t in AUDIO_DELTA:
                spk.push(acc.feed(base64.b64decode(ev["delta"])))
                if first_text is not None:
                    stats["lead_s"].append(time.monotonic() - first_text)
                    first_text = None
            elif t in TEXT_DELTA:
                if first_text is None and not said:
                    first_text = time.monotonic()      # 전사가 소리보다 먼저 온다
                said += ev.get("delta", "")
            elif t == "input_audio_buffer.speech_started":
                if spk.busy:
                    stats["echo"] += 1
                    print("  🔴 에코 — 봇이 말하는 중에 서버가 발화를 잡았다")
            elif t.endswith("input_audio_transcription.completed"):
                print(f"  아이: {(ev.get('transcript') or '').strip()}")
            elif t == "response.done":
                stats["turns"] += 1
                print(f"  티드: {said.strip()}")
                said, first_text = "", None
                c = cost_usd(model, ev.get("response", {}).get("usage", {}) or {})
                if c:
                    stats["cost_usd"] += c
            elif t == "error":
                print(f"  ❌ {ev.get('error')}")

    tasks = [asyncio.create_task(pump_mic()), asyncio.create_task(read_events())]
    try:
        await asyncio.wait(tasks, timeout=seconds)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        stream.stop()
        stream.close()
        while spk.busy:                    # 하던 말은 끝내고 닫는다
            await asyncio.sleep(0.05)
        spk.close()
        await ws.close()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Realtime 라이브 루프 (1·2단계)")
    p.add_argument("--voices", action="store_true",
                   help="1단계: 목소리 표본을 블라인드로 뽑는다(마이크 필요 없음)")
    p.add_argument("--voice-list", default=",".join(VOICES))
    p.add_argument("--line", default=SAMPLE_LINE)
    p.add_argument("--out", default="data/voice_blind", help="1단계 wav 를 둘 곳")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", default="gpt-realtime-mini", choices=sorted(PRICE))
    p.add_argument("--voice", default="marin",
                   help="2단계에서 쓸 목소리. 09-07 블라인드에서 marin(voice_A) 선택")
    p.add_argument("--duplex", default="half", choices=("half", "full"),
                   help="full 은 마이크를 계속 열어 둔다 — 에코를 재는 모드")
    p.add_argument("--pad", type=float, default=0.15,
                   help="재생이 끝나고 마이크를 다시 열기까지(초). 08-24 실측 0.150")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--no-speaker", metavar="WAV", nargs="?",
                   const="reports/bench/live_reply.wav",
                   help="스피커 없이. 봇 목소리를 재생 대신 이 wav 로 받는다")
    p.add_argument("--silence-ms", type=int, default=1200)
    p.add_argument("--transcribe-model", default="gpt-4o-transcribe")
    p.add_argument("--json")
    a = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY 가 없다")

    if a.voices:
        voices = [v.strip() for v in a.voice_list.split(",") if v.strip()]
        print(f"목소리 {len(voices)}개 | 씨앗 {a.seed}\n")
        asyncio.run(sample_voices(a.model, voices, a.line, Path(a.out), a.seed))
        return

    cfg = session_config(a.silence_ms, a.transcribe_model, a.voice)
    print(f"{a.model} | 목소리 {a.voice} | 꼬리 {a.silence_ms}ms | {a.seconds:.0f}초")
    print("  말을 걸어 보세요. Ctrl+C 로 끝냅니다.\n")
    try:
        stats = asyncio.run(live(a.model, cfg, a.seconds, a.duplex, a.pad,
                                 Path(a.no_speaker) if a.no_speaker else None))
    except KeyboardInterrupt:
        print("\n중단")
        return

    lead = stats.pop("lead_s")
    wav = stats.pop("wav")
    print(f"\n턴 {stats['turns']} | 에코 {stats['echo']} | "
          f"보냄 {stats['sent_s']:.1f}s / 버림 {stats['dropped_s']:.1f}s | "
          f"${stats['cost_usd']:.4f}")
    if lead:
        print(f"전사가 소리보다 앞선 시간: 중앙 {float(np.median(lead)):.3f}s "
              f"(n={len(lead)}) — 안전 검사에 쓸 수 있는 여유")
    if wav:
        print(f"봇 목소리: {wav}  (에코는 이 모드로 못 쟀다)")
    if stats["echo"]:
        print("🔴 에코가 잡혔다. 이 조건에서는 마이크를 열어 둘 수 없다.")
    if a.json:
        Path(a.json).write_text(json.dumps({**stats, "lead_s": lead},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  저장: {a.json}")


if __name__ == "__main__":
    main()
