"""말 끝 판정 방식 비교 — 아이 녹음(3~6세 153발화)을 실서버에 실시간으로 흘린다. (2026-09-22)

spec: docs/superpowers/specs/2026-09-22-realtime-turn-fixes-design.md §5
재는 것 (발화마다):
  빨라짐 latency_s : 실제 말 끝을 보낸 시각 → 그 뒤 첫 speech_stopped 도착
  잘림   premature : 실제 말 끝보다 0.1s 넘게 **먼저** speech_stopped 가 옴 = 말 중간에 끝났다고 봄
답은 만들지 않는다(create_response false, 받아 적기 끔) — 입력 소리만 보낸다.

실행:
  python -m tools.turn_end_probe --n 10            # 먼저 비용·동작 확인
  python -m tools.turn_end_probe --json reports/turn/realtime_turn_end.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
SR = 24000
CHUNK = 960                         # 40ms
LEAD_S, TAIL_S = 0.5, 6.0           # 앞 무음 / 뒤 무음(semantic low 는 오래 기다린다)
EARLY_S = 0.1


def _vad(kind: str, **kw) -> dict:
    return {"type": kind, "create_response": False, "interrupt_response": False, **kw}


VARIANTS = {
    "server1200": _vad("server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=1200),
    "server900": _vad("server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=900),
    "server600": _vad("server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=600),
    "semantic_low": _vad("semantic_vad", eagerness="low"),
    "semantic_medium": _vad("semantic_vad", eagerness="medium"),
    "semantic_high": _vad("semantic_vad", eagerness="high"),
}


def analyze(stops: list[float], t_end: float) -> dict:
    early = [s for s in stops if s < t_end - EARLY_S]
    after = [s for s in stops if s >= t_end - EARLY_S]
    return {"latency_s": (after[0] - t_end) if after else None,
            "premature": bool(early), "splits": len(early)}


def summarize(rows: list[dict]) -> list[str]:
    out = ["| 방식 | n | 빨라짐 중앙 | p90 | 잘림 |", "|---|---|---|---|---|"]
    for name in VARIANTS:
        rs = [r for r in rows if r.get("variant") == name]
        if not rs:
            continue
        lat = sorted(r["latency_s"] for r in rs if r.get("latency_s") is not None)
        pre = sum(1 for r in rs if r.get("premature"))
        med = f"{statistics.median(lat):.2f}s" if lat else "-"
        p90 = f"{lat[int(0.9 * (len(lat) - 1))]:.2f}s" if lat else "-"
        out.append(f"| {name} | {len(rs)} | {med} | {p90} | {pre}/{len(rs)} = {100 * pre / len(rs):.0f}% |")
    return out


def _items(n: int) -> list[dict]:
    idx = json.loads((BASE / "reports/stt/young/audio_index.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in
            (BASE / "reports/stt/young/manifest.jsonl").read_text(encoding="utf-8").splitlines() if x]
    items = [dict(r, path=idx[r["file"]]) for r in rows if r["file"] in idx and os.path.exists(idx[r["file"]])]
    return items[:n] if n else items


async def _run_variant(name: str, items: list[dict], model: str, sink: Path | None) -> list[dict]:
    from tools.realtime_probe import connect, load_pcm
    cfg = {"type": "session.update", "session": {
        "type": "realtime", "output_modalities": ["text"],
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": SR},
                            "turn_detection": VARIANTS[name]}}}}
    ws = await connect(model, cfg)
    stops: list[float] = []

    async def reader():
        async for raw in ws:
            ev = json.loads(raw)
            if ev.get("type") == "input_audio_buffer.speech_stopped":
                stops.append(time.monotonic())
            elif ev.get("type") == "error":
                print(f"   [{name}] 오류 {ev.get('error')}", flush=True)

    rt = asyncio.create_task(reader())
    rows = []
    silence = np.zeros(CHUNK, dtype="<i2").tobytes()

    async def send(pcm: bytes) -> None:
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(pcm).decode()}))
        await asyncio.sleep(len(pcm) / 2 / SR)

    try:
        for i, it in enumerate(items, 1):
            for _ in range(int(LEAD_S * SR / CHUNK)):
                await send(silence)
            stops.clear()
            pcm = load_pcm(Path(it["path"]), "speech")
            for k in range(0, len(pcm), CHUNK * 2):
                await send(pcm[k:k + CHUNK * 2])
            t_end = time.monotonic()
            for _ in range(int(TAIL_S * SR / CHUNK)):
                await send(silence)
            r = {"variant": name, "file": it["file"], "age": it.get("age"),
                 "sec": round(len(pcm) / 2 / SR, 2), **analyze(list(stops), t_end)}
            rows.append(r)
            if sink:
                with sink.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"   [{name}] {i}/{len(items)}", flush=True)
    finally:
        rt.cancel()
        await ws.close()
    return rows


async def _main(a) -> list[dict]:
    items = _items(a.n)
    names = a.only.split(",") if a.only else list(VARIANTS)
    sink = Path(a.json).with_suffix(".jsonl") if a.json else None
    print(f"발화 {len(items)}개 × 방식 {len(names)}개 — 동시에 흘린다", flush=True)
    res = await asyncio.gather(*(_run_variant(n, items, a.model, sink) for n in names))
    return [r for rs in res for r in rs]


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    p = argparse.ArgumentParser(description="말 끝 판정 방식 비교(아이 녹음, 실서버)")
    p.add_argument("--n", type=int, default=0, help="0 이면 전부(153)")
    p.add_argument("--only", help="쉼표로 방식 고르기")
    p.add_argument("--model", default="gpt-realtime-mini")
    p.add_argument("--json")
    a = p.parse_args()
    rows = asyncio.run(_main(a))
    print("\n".join(summarize(rows)))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
