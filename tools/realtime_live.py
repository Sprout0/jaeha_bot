"""OpenAI Realtime 라이브 루프 — 마이크 ↔ 소켓 ↔ 스피커.

09-02 에 파일을 소켓에 밀어 넣어 지연·CER·비용·안전여유는 이미 쟀다. 여기서 재는 건
**라이브로 돌렸을 때만 드러나는 것** 셋이다: 에코 / 바지인 / 리샘플.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.realtime_audio import (FULL_SCALE, REMAINDER, MicGate,  # noqa: E402,F401 — 도구·시험이 이 이름으로 쓴다
                                PcmAccumulator)


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
from app.realtime_audio import StreamSpeaker as Speaker, resolve_rate  # noqa: E402,F401

AUDIO_DELTA = ("response.audio.delta", "response.output_audio.delta")
AUDIO_DONE = ("response.audio.done", "response.output_audio.done")
TEXT_DELTA = ("response.audio_transcript.delta",
              "response.output_audio_transcript.delta")
VOICES = ["alloy", "ash", "ballad", "cedar", "coral", "echo", "marin", "sage",
          "shimmer", "verse"]
READER = ("너는 낭독자다. 사용자가 준 문장을 **글자 그대로** 한 번만 읽어라. "
          "덧붙이거나 줄이거나 바꾸지 마라.")
SAMPLE_LINE = "안녕! 나는 티드야. 오늘 뭐 하고 놀까? 같이 노래도 부를 수 있어."


PROMPT_YAML = Path(__file__).resolve().parent.parent / "configs" / "prompt_templates.yaml"


def real_prompt(path: Path = PROMPT_YAML) -> str:
    """실기가 쓰는 system 프롬프트 그대로.

    시험용 인격은 60자뿐이라 "할 수 있는 놀이는 둘뿐"이라는 목록이 없다. 그래서
    Realtime 이 없는 놀이를 지어냈다(09-07: 숫자 맞추기·그림 맞추기). 실기에서는
    이 부류를 4d571da·7a488f3 에서 **기능이 아니라 프롬프트로** 잡았다.
    """
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))["system"]


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
    p.add_argument("--speed", type=float, default=1.0,
                   help="말 속도 0.25~1.5(API 한계). 봇은 configs realtime.speed")
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
    p.add_argument("--real-prompt", action="store_true",
                   help="시험용 60자 인격 대신 실기 프롬프트를 꽂는다(규칙을 지키는지)")
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
    cfg["session"]["audio"]["output"]["speed"] = a.speed
    if a.real_prompt:
        cfg["session"]["instructions"] = real_prompt()
    print(f"{a.model} | 목소리 {a.voice} ×{a.speed:g} | 꼬리 {a.silence_ms}ms | {a.seconds:.0f}초 | "
          f"프롬프트 {'실기' if a.real_prompt else '시험용 60자'}")
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
