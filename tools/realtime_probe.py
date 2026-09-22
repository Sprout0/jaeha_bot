"""OpenAI Realtime(S2S)을 **우리 조건으로** 잰다 — 지연 · 과금 · 아이 말 알아듣기.

🔴 왜 (2026-09-02):
   "전부 API 로 가면 어떤가"에 답하려면 세 가지를 알아야 하는데, 벤더 수치는 전부
   영어·성인 기준이라 하나도 쓸 수 없다. 그래서 우리 오디오로 직접 잰다.

무엇을 재나
  [지연]  t_vad    말이 끝난 순간 -> 서버가 '끝났다'고 판단할 때까지 (= 우리 꼬리 1.2s 자리)
          t_resp   그 판단 -> 첫 오디오 조각 (= 우리 STT+LLM+TTS 자리)
          t_text   그 판단 -> 첫 **출력 전사** 조각
                   🔴 이게 t_resp 보다 빠르면, 소리를 잠깐 물고 텍스트로 먼저 안전 검사를
                      할 수 있다. S2S 를 막는 제일 큰 이유가 '안전이 사후가 된다'인데,
                      그 이유가 살아 있는지 죽는지가 이 한 값에 달렸다.
  [과금]  response.done 의 usage 를 그대로 읽어 곱한다. 추정하지 않는다.
          ⚠️ 입력이 **턴마다 누적된다**(실측: 턴1 19토큰 -> 턴8 210토큰). 한 세션을
             길게 끌면 비용이 제곱으로 는다. --fresh 는 발화마다 새 세션을 열어
             이 누적을 없앤다(CER 측정처럼 발화끼리 독립이어야 할 때 쓴다).
  [인식]  입력 전사를 정답과 대 본다. 우리 로컬 medium 과 **같은 자**로 채점하려면
          `--baseline reports/stt/young/detail_medium.json` 을 준다.

⚠️ 이 도구가 재는 '입력 전사'는 모델이 실제로 들은 것이 아니라 **곁다리로 돌린 STT** 다.
   그래도 우리에겐 이게 중요하다 — 안전 필터·claims·논문 로그가 전부 이 전사를 먹는다.

사용:
    # 지연·과금 (어른 실음성, 우리 꼬리와 같은 1200ms)
    python tools/realtime_probe.py --wav-dir data/wake_real/adult_20260826_1445 \
        --silence-ms 1200 --trim speech

    # 아이 말 알아듣기 (AI-Hub 3~6세 153발화)
    python tools/realtime_probe.py --manifest reports/stt/young/manifest.jsonl \
        --audio-root <AI-Hub 루트> --audio-index young_index.json \
        --baseline reports/stt/young/detail_medium.json \
        --fresh --trim edges --silence-ms 1200 --n 40 --json out.json

⚠️ 오래 도는 실행이다. --json 을 주면 그 옆에 .jsonl 로 **한 줄씩** 쌓이므로 중간에
   터져도 거기까지는 남는다(실제로 40발화 중 28번째에서 서버가 끊어 27턴을 잃었다).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

# 편집거리는 운영 코드 것을 빌려 온다 — 자가 둘이면 측정이 거짓말을 한다.
from app.text_norm import _edit_distance  # noqa: E402

URL = "wss://api.openai.com/v1/realtime?model={model}"
SR = 24000            # Realtime pcm16 규격
CHUNK_MS = 40         # 실기 마이크 프레임과 비슷한 크기로 흘린다

from app.realtime_protocol import PRICE, cost_usd  # noqa: E402,F401 — 옮겼다. 도구·시험이 이 이름으로 쓴다

INSTRUCTIONS = (
    "너는 다섯 살 아이의 친구 로봇 '티드'야. 밝은 반말로 한두 문장만, 30자 안쪽으로 "
    "짧게 말해. 이모지나 기호는 쓰지 마."
)

# ── 채점 ────────────────────────────────────────────────────────────────────
# AI-Hub 정답의 (SP:떠) 같은 태그는 **속 내용을 남긴다**. 153개 중 114개의 기록된 CER 이
# 이 규칙으로 재현된다(다른 규칙은 55개 안팎). 공백·문장부호는 양쪽에서 지운다 —
# whisper 는 붙이고 정답엔 없어서, 안 지우면 표기 차이가 오류로 잡힌다.
# ⚠️ 그래서 여기 절대값은 README 의 31.46%(공백 포함)와 다르다. **같은 자를 양쪽에
#    대므로 A/B 는 유효하다** — 반드시 --baseline 을 같이 돌려 나란히 볼 것.
_TAG = re.compile(r"\(([A-Z]{2}):([^)]*)\)")
_PUNCT = re.compile(r"[.,!?~…·\"'’“”\-–—()\[\]{}]")


def clean_ref(s: str) -> str:
    return _PUNCT.sub("", _TAG.sub(r"\2", s or "")).replace(" ", "").strip()


def clean_hyp(s: str) -> str:
    return _PUNCT.sub("", s or "").replace(" ", "").strip()


def cer(hyp: str, ref: str) -> float:
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 100.0
    return 100.0 * _edit_distance(h, r) / len(r)


# ── 오디오 ──────────────────────────────────────────────────────────────────
def trim_speech(a: np.ndarray, pad_s: float = 0.1) -> np.ndarray:
    """20ms 프레임 RMS 로 **말소리 끝에서 딱 끊는다**. 지연 측정용.

    🔴 이걸 안 하면 측정이 거짓말이 된다. 녹음이 고정 길이면 뒤 무음까지 '말소리'로
       흘리게 되고, 서버 VAD 가 파일이 끝나기 전에 발화 종료를 잡아 t_vad 가 0 으로
       무너진다(실제로 8턴 중 4턴이 그랬다).
    ⚠️ 대신 말끝을 조금 깎을 수 있다. **CER 측정에는 쓰지 말 것** — trim_edges 를 쓴다.
    """
    if a.size == 0:
        return a
    win = int(0.02 * SR)
    n = a.size // win
    if n < 2:
        return a
    rms = np.sqrt((a[:n * win].reshape(n, win) ** 2).mean(axis=1))
    if float(rms.max()) < 1e-6:
        return a
    loud = np.where(rms > 0.10 * float(rms.max()))[0]
    if loud.size == 0:
        return a
    start = max(0, int(loud[0]) * win - int(pad_s * SR))
    end = min(a.size, (int(loud[-1]) + 1) * win + int(0.03 * SR))
    return a[start:end]


def trim_edges(a: np.ndarray, pad_s: float = 0.1) -> np.ndarray:
    """운영 규칙(app/stt_module._trim_edges)과 같은 자. 넉넉히 남긴다. CER 측정용."""
    if a.size == 0:
        return a
    peak = float(np.max(np.abs(a)))
    if peak < 1e-6:
        return a
    nz = np.where(np.abs(a) > max(2e-3, 0.02 * peak))[0]
    if nz.size == 0:
        return a
    pad = int(pad_s * SR)
    return a[max(0, int(nz[0]) - pad):min(a.size, int(nz[-1]) + pad)]


TRIMS = {"speech": trim_speech, "edges": trim_edges, "none": lambda a: a}


def load_pcm(path: Path, trim: str) -> bytes:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly   # 파일 읽을 때만 — 젯슨엔 scipy 가 없다(realtime_live 는 안 쓴다)
        g = gcd(sr, SR)
        mono = resample_poly(mono, SR // g, sr // g)
    return (np.clip(TRIMS[trim](mono), -1.0, 1.0) * 32767).astype("<i2").tobytes()


# ── 도구(function calling) ──────────────────────────────────────────────────
# 🔴 우리 봇은 지금 LLM function calling 을 **안 쓴다**(`agent.respond` 의 function_call
#    은 늘 None). 소리 재생은 `app/audio_player.AudioLibrary.find` 의 낱말 규칙이 고른다.
#    그러니 여기서 물을 것은 "우리 도구를 옮길 수 있나"가 아니라
#    **"모델이 부르는 게 그 낱말 규칙보다 나은가"** 다. 그래서 둘을 나란히 채점한다.
# ⚠️ 도구 목록은 `configs/audio_assets.yaml` 에서 만든다 — 봇이 실제로 낼 수 있는
#    소리와 시험이 어긋나면, 통과해도 옮길 수 있다는 뜻이 아니다.
def build_tools() -> list[dict]:
    import yaml
    d = yaml.safe_load((BASE / "configs/audio_assets.yaml").read_text(encoding="utf-8"))
    sounds = [a["id"] for a in (d.get("sounds") or [])]
    songs = [a["id"] for a in (d.get("songs") or [])]
    return [
        {"type": "function", "name": "play_sound",
         "description": "아이가 동물 소리를 **들려 달라고 요청했을 때만** 부른다. "
                        "동물 이름이 나왔다고 요청인 것은 아니다 "
                        "('고양이 봤어'는 요청이 아니다).",
         "parameters": {"type": "object", "properties": {
             "sound_id": {"type": "string", "enum": sounds}}, "required": ["sound_id"]}},
        {"type": "function", "name": "play_song",
         "description": "아이가 노래를 틀어 달라고 요청했을 때만 부른다.",
         "parameters": {"type": "object", "properties": {
             "song_id": {"type": "string", "enum": songs}}, "required": ["song_id"]}},
    ]


# 시험 문항. 정답은 우리 규칙이 아니라 **사람이 봐서 옳은 것**이다.
# 앞의 둘은 실제로 났던 오작동이고(커밋 ad022ba·c3cf296 에서 고침), 뒤의 둘은
# 지금 낱말 규칙이 여전히 놓치는 자리다.
TOOL_CASES = [
    ("고양이 소리 들려줘", "cat"), ("멍멍이 소리 내봐", "dog"),
    ("개구리 소리", "frog"), ("병아리 소리", "chick"),
    ("강아지 소리 들려줘", "dog"), ("돼지 소리 내봐", "pig"),
    ("왈왈 하는 소리 내봐", "dog"),      # 낱말 규칙 놓침
    ("야옹 해봐", "cat"),               # 낱말 규칙 놓침
    ("고양이 봤어", None),               # 예전 오작동
    ("무슨 소리야", None),               # 예전 오작동
    ("오늘 하루 어땠어", None), ("양말 신었어", None),
    ("개미 봤어", None), ("엄마랑 과자를 만들어요", None),
]


async def one_text_turn(ws, text: str) -> dict:
    """글로 한 턴. 도구를 부르는지만 보므로 오디오를 안 태운다(싸고 빠르다)."""
    await drain(ws, 0.3)
    await ws.send(json.dumps({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}}))
    await ws.send(json.dumps({"type": "response.create"}))
    m: dict = {"text": text}
    async for raw in ws:
        ev = json.loads(raw)
        t = ev.get("type", "")
        if t.endswith("function_call_arguments.done"):
            m["tool"] = ev.get("name")
            try:
                m["args"] = json.loads(ev.get("arguments") or "{}")
            except json.JSONDecodeError:
                m["args"] = {"_raw": ev.get("arguments")}
        elif t == "response.done":
            for item in ev.get("response", {}).get("output", []):
                if item.get("type") == "function_call":
                    m.setdefault("tool", item.get("name"))
                    m.setdefault("args", json.loads(item.get("arguments") or "{}"))
                for c in item.get("content", []):
                    if c.get("transcript") or c.get("text"):
                        m["said"] = (c.get("transcript") or c.get("text")).strip()
            break
        elif t == "error":
            m["error"] = ev.get("error", {})
            break
    return m


# ── 세션 ────────────────────────────────────────────────────────────────────
def session_config(silence_ms: int, transcribe_model: str, voice: str,
                   tools: list[dict] | None = None,
                   modalities: list[str] | None = None) -> dict:
    """GA 스키마(2026-09 확인). 베타의 평평한 input_audio_format/turn_detection 은 거부된다."""
    s: dict = {
        "type": "realtime",
        "instructions": INSTRUCTIONS,
        "output_modalities": modalities or ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SR},
                "transcription": {"model": transcribe_model, "language": "ko"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": silence_ms,
                                   "threshold": 0.5, "prefix_padding_ms": 300},
            },
            "output": {"format": {"type": "audio/pcm", "rate": SR}, "voice": voice},
        },
    }
    if tools:
        s |= {"tools": tools, "tool_choice": "auto"}
    return {"type": "session.update", "session": s}


async def drain(ws, quiet_s: float = 0.6) -> None:
    """지난 턴 이벤트를 비운다. 안 하면 이전 턴의 델타를 '첫 소리'로 세어 0.001s 가 나온다."""
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=quiet_s)
        except asyncio.TimeoutError:
            return


async def one_turn(ws, pcm: bytes, pad_s: float) -> dict:
    """오디오를 **실시간 속도로** 흘리고(한꺼번에 부으면 서버 VAD 가 실제와 달리 돈다),
    뒤에 무음을 붙여 서버가 발화 끝을 잡게 한다."""
    step = SR * CHUNK_MS // 1000 * 2
    dt = CHUNK_MS / 1000.0
    await drain(ws)

    t0 = time.perf_counter()
    for i in range(0, len(pcm), step):
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(pcm[i:i + step]).decode()}))
        await asyncio.sleep(max(0.0, t0 + (i // step + 1) * dt - time.perf_counter()))
    t_end = time.perf_counter()          # <- 말이 끝난 순간

    quiet = b"\x00" * step
    m: dict = {}

    async def feeder():
        for k in range(int(pad_s / dt)):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(quiet).decode()}))
            await asyncio.sleep(max(0.0, t_end + (k + 1) * dt - time.perf_counter()))

    feed = asyncio.create_task(feeder())
    try:
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            now = time.perf_counter() - t_end
            if t == "input_audio_buffer.speech_stopped":
                m.setdefault("t_vad", now)
            elif t in ("response.audio.delta", "response.output_audio.delta"):
                m.setdefault("t_first_audio", now)
            elif t in ("response.audio_transcript.delta",
                       "response.output_audio_transcript.delta"):
                m.setdefault("t_first_text", now)
            elif t.endswith("input_audio_transcription.completed"):
                m["heard"] = (ev.get("transcript") or "").strip()
            elif t.endswith("input_audio_transcription.failed"):
                m["heard"] = ""
                m["transcribe_error"] = ev.get("error", {})
            elif t == "response.done":
                r = ev.get("response", {})
                m["usage"] = r.get("usage", {})
                for item in r.get("output", []):
                    for c in item.get("content", []):
                        if c.get("transcript"):
                            m["said"] = c["transcript"].strip()
                # 입력 전사는 response.done 뒤에 오는 일이 잦다. 잠깐 더 받아 준다.
                if "heard" not in m:
                    end = time.perf_counter() + 2.5
                    while time.perf_counter() < end:
                        try:
                            e2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                        except asyncio.TimeoutError:
                            break
                        if e2.get("type", "").endswith("input_audio_transcription.completed"):
                            m["heard"] = (e2.get("transcript") or "").strip()
                            break
                break
            elif t == "error":
                m["error"] = ev.get("error", {})
                break
    finally:
        feed.cancel()
    return m


async def connect(model: str, cfg: dict):
    import websockets
    ws = await websockets.connect(
        URL.format(model=model),
        additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        max_size=None)
    await ws.send(json.dumps(cfg))
    return ws


async def run(model: str, items: list[dict], cfg: dict, trim: str, pad_s: float,
              fresh: bool, sink: Path | None = None, retries: int = 2) -> list[dict]:
    """fresh=True 면 발화마다 새 세션. 이력 누적이 없어져 발화끼리 독립이 된다(CER 용).

    🔴 한 줄 끝날 때마다 **바로 파일에 붙인다.** 40발화 중 28번째에서 서버가 연결을
       끊었는데(ConnectionClosedError), 끝에 한 번에 저장하는 구조라 앞의 27턴을
       통째로 잃었다. 20분짜리 실행이 끝물에 터져 아무것도 안 남는 일은 한 번이면 족하다.
    ⚠️ 서버가 끊는 건 드물지 않다. 그 발화만 다시 붙어 재시도하고, 그래도 안 되면
       그 줄만 버리고 **계속 간다** — 한 발화 때문에 집합 전체를 잃지 않는다.
    """
    import websockets

    rows: list[dict] = []
    ws = None if fresh else await connect(model, cfg)

    async def attempt(it) -> dict:
        nonlocal ws
        pcm = load_pcm(Path(it["path"]), trim)
        for k in range(retries + 1):
            try:
                if fresh:
                    ws = await connect(model, cfg)
                try:
                    return await one_turn(ws, pcm, pad_s)
                finally:
                    if fresh and ws is not None:
                        await ws.close()
            except websockets.exceptions.WebSocketException as e:
                if k == retries:
                    return {"error": {"type": type(e).__name__, "message": str(e)[:120]}}
                print(f"      연결이 끊겼다({type(e).__name__}) — 다시 붙는다", flush=True)
                if not fresh:                      # 한 세션 모드면 새로 열어 이어 간다
                    ws = await connect(model, cfg)
                await asyncio.sleep(1.0 + k)
        return {}

    try:
        for i, it in enumerate(items, 1):
            m = await attempt(it)
            m |= {"file": Path(it["path"]).name, "ref": it.get("ref", ""),
                  "age": it.get("age")}
            rows.append(m)
            if sink:
                with sink.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            if "error" in m:
                print(f"  [{i}/{len(items)}] 버림 — {m['error']}", flush=True)
                continue
            print(f"  [{i}/{len(items)}] VAD {m.get('t_vad', float('nan')):.3f} "
                  f"소리 {m.get('t_first_audio', float('nan')):.3f} "
                  f"글 {m.get('t_first_text', float('nan')):.3f} | "
                  f"들음 {m.get('heard','')!r} | 답 {m.get('said','')!r}", flush=True)
    finally:
        if not fresh and ws is not None:
            await ws.close()
    return rows


# ── 읽는 법 ─────────────────────────────────────────────────────────────────
def summarize(model: str, rows: list[dict], baseline: dict[str, str] | None) -> list[str]:
    ok = [r for r in rows if "t_first_audio" in r]
    if not ok:
        return ["첫 오디오를 한 번도 못 받았다 — 위 오류를 볼 것."]
    med = statistics.median
    out = [f"n={len(ok)}"]

    # 🔴 t_vad 가 0 인 줄은 시간에서 뺀다. 그건 '엄청 빨랐다'가 아니라 **서버가 우리
    #    스트리밍이 끝나기 전에 발화 종료를 잡은 것**이다(파일 뒤에 무음이 남아 있으면
    #    그렇게 된다). 섞어서 중앙값을 내면 지연이 실제보다 훨씬 좋아 보인다.
    timed = [r for r in ok if r.get("t_vad", 0.0) > 0.05]
    if len(timed) < len(ok):
        out.append(f"  ⚠️ 시간 통계에서 {len(ok) - len(timed)}개 제외"
                   " (서버가 스트리밍 도중 발화 종료를 잡음 — 그 줄의 시간은 무의미하다)")
    vad = [r["t_vad"] for r in timed]
    resp = [r["t_first_audio"] - r["t_vad"] for r in timed]
    txt = [r["t_first_text"] - r["t_vad"] for r in timed if "t_first_text" in r]
    if vad:
        out.append(f"  t_vad   (말끝->끝판단)   중앙 {med(vad):.3f}s")
    if resp:
        out.append(f"  t_resp  (끝판단->첫소리) 중앙 {med(resp):.3f}s")
    if txt:
        lead = med(resp) - med(txt)
        out.append(f"  t_text  (끝판단->첫글)   중앙 {med(txt):.3f}s"
                   f"   -> 글이 소리보다 {lead:+.3f}s 앞선다")
        out.append("     읽는 법: 이 값이 **양수로 충분히 크면** 소리를 물고 텍스트로"
                   " 먼저 안전 검사를 할 수 있다. 0 이하면 S2S 에서 안전은 사후다.")

    costs = [c for r in ok if (c := cost_usd(model, r.get("usage", {}))) is not None]
    if costs:
        out.append(f"  턴당 비용 중앙 ${med(costs):.5f}")

    heard = [r.get("heard", "") for r in ok]
    empty = sum(1 for h in heard if not h)
    out.append(f"  입력 전사 빈 것 {empty}/{len(heard)} = {100*empty/len(heard):.0f}%"
               "   <- 안전 필터·로그가 먹을 재료")

    said = [r.get("said", "") for r in ok if r.get("said")]
    if said:
        over = sum(1 for s in said if len(s) > 30)
        out.append(f"  답 길이 중앙 {med([len(s) for s in said]):.0f}자, "
                   f"30자 초과 {over}/{len(said)} = {100*over/len(said):.0f}%")

    scored = [(clean_hyp(r.get("heard", "")), clean_ref(r["ref"])) for r in ok if r.get("ref")]
    if scored:
        vals = [cer(h, rf) for h, rf in scored if rf]
        out.append("")
        out.append(f"  [아이 말 알아듣기] Realtime 입력전사 CER 평균 {statistics.mean(vals):.2f}%"
                   f" · 중앙 {med(vals):.2f}% · 완벽일치 {sum(1 for v in vals if v == 0)}/{len(vals)}")
        if baseline:
            bvals, n = [], 0
            for r in ok:
                b = baseline.get(r["file"])
                if b is not None and r.get("ref"):
                    bvals.append(cer(clean_hyp(b), clean_ref(r["ref"])))
                    n += 1
            if bvals:
                out.append(f"  [같은 자·같은 발화] 로컬 medium      CER 평균 {statistics.mean(bvals):.2f}%"
                           f" · 중앙 {med(bvals):.2f}% · 완벽일치 {sum(1 for v in bvals if v == 0)}/{n}")
                out.append(f"  ➡️ 차이 {statistics.mean(vals) - statistics.mean(bvals):+.2f}%p"
                           "  (음수면 Realtime 이 낫다)")
        out.append("  ⚠️ 절대값은 README 의 31.46%(공백 포함 채점)와 다르다."
                   " 같은 자를 양쪽에 댔으므로 **비교만** 유효하다.")
    return out


def load_items(a) -> list[dict]:
    if a.manifest:
        # ⚠️ AI-Hub 뿌리에는 wav 가 27만 개 있다. 매번 훑으면 몇 분이 그냥 날아가므로
        #    이름->경로 색인을 파일로 남겨 두고 다시 쓴다.
        cache = Path(a.audio_index) if a.audio_index else None
        if cache and cache.exists():
            index = {k: Path(v) for k, v in
                     json.loads(cache.read_text(encoding="utf-8")).items()}
        else:
            print("오디오 색인을 만드는 중… (27만 파일이면 몇 분 걸린다)", flush=True)
            index = {}
            for p in Path(a.audio_root).rglob("*.wav"):
                index.setdefault(p.name, p)
            if cache:
                cache.write_text(json.dumps({k: str(v) for k, v in index.items()},
                                            ensure_ascii=False), encoding="utf-8")
        items = []
        for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            p = index.get(j["file"])
            if p is None:
                continue
            items.append({"path": str(p), "ref": j.get("stt", ""), "age": j.get("age")})
        missing = sum(1 for line in Path(a.manifest).read_text(encoding="utf-8").splitlines()
                      if line.strip()) - len(items)
        if missing:
            print(f"⚠️ 오디오를 못 찾은 발화 {missing}개 — 건너뛴다")
        return items[:a.n] if a.n else items
    wavs = sorted(Path(a.wav_dir).glob("*.wav"))
    return [{"path": str(w)} for w in (wavs[:a.n] if a.n else wavs)]


async def run_tool_check(model: str, cases: list[tuple]) -> list[dict]:
    """도구를 물려 놓고 글로 물어본다. 한 세션에서 이어 가되 앞 턴이 안 새게 비운다."""
    cfg = session_config(1200, "gpt-4o-transcribe", "marin",
                         tools=build_tools(), modalities=["audio"])
    ws = await connect(model, cfg)
    rows = []
    try:
        for text, want in cases:
            m = await one_text_turn(ws, text)
            m["want"] = want
            rows.append(m)
            if "error" in m:
                print(f"  오류 {m['error']}")
                break
    finally:
        await ws.close()
    return rows


def report_tool_check(rows: list[dict]) -> list[str]:
    """모델의 호출과 **우리 낱말 규칙**을 나란히 놓는다. 규칙보다 나은지가 물음이다."""
    from app.audio_player import default_library
    lib = default_library()

    out = [f"{'아이 말':22} {'우리 규칙':>10} {'Realtime':>12} {'정답':>8}   판정", "-" * 68]
    ours_ok = model_ok = 0
    for r in rows:
        if "error" in r:
            continue
        got = (r.get("args") or {}).get("sound_id") if r.get("tool") == "play_sound" else None
        a = lib.find(r["text"])
        ours = a.id if a else None
        want = r["want"]
        ours_ok += (ours == want)
        model_ok += (got == want)
        mark = {(True, True): "둘 다 맞음", (True, False): "규칙만 맞음",
                (False, True): "**모델만 맞음**", (False, False): "둘 다 틀림"}[
            (ours == want, got == want)]
        out.append(f"{r['text']:22} {str(ours):>10} {str(got):>12} {str(want):>8}   {mark}")
    n = sum(1 for r in rows if "error" not in r)
    out += ["-" * 68,
            f"우리 낱말 규칙 {ours_ok}/{n}   Realtime function calling {model_ok}/{n}",
            "",
            "  읽는 법: '모델만 맞음'이 있어야 옮길 값이 있다. 규칙이 다 맞으면",
            "           지금 구조를 바꿀 이유가 없다는 뜻이다."]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI Realtime 실측")
    p.add_argument("--tool-check", action="store_true",
                   help="function calling 이 되는지 + 우리 낱말 규칙보다 나은지")
    p.add_argument("--model", default="gpt-realtime-mini", choices=sorted(PRICE))
    p.add_argument("--wav-dir", default="data/wake_real/adult_20260826_1445")
    p.add_argument("--manifest", help="{file, stt, age} jsonl — 정답이 있는 집합")
    p.add_argument("--audio-root", help="--manifest 의 wav 를 찾을 뿌리 디렉터리")
    p.add_argument("--audio-index", help="이름->경로 색인 캐시 json (없으면 만든다)")
    p.add_argument("--baseline", help="같은 발화의 로컬 STT 결과 json (detail_*.json)")
    p.add_argument("--transcribe-model", default="whisper-1")
    p.add_argument("--voice", default="marin")
    p.add_argument("--silence-ms", type=int, default=1200,
                   help="서버 VAD 무음 임계. 우리 꼬리와 맞추려면 1200")
    p.add_argument("--trim", default="speech", choices=sorted(TRIMS),
                   help="speech=말끝에서 끊는다(지연용) / edges=넉넉히(CER용) / none")
    p.add_argument("--pad", type=float, default=5.0, help="발화 뒤 무음(초)")
    p.add_argument("--n", type=int, default=8, help="0 이면 전부")
    p.add_argument("--fresh", action="store_true",
                   help="발화마다 새 세션(이력 누적 없음). CER 측정에 쓴다")
    p.add_argument("--json")
    a = p.parse_args()

    if a.tool_check:
        rows = asyncio.run(run_tool_check(a.model, TOOL_CASES))
        print()
        print("\n".join(report_tool_check(rows)))
        if a.json:
            Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
            print(f"\n  저장: {a.json}")
        return

    if a.manifest and not a.audio_root:
        sys.exit("--manifest 를 쓰면 --audio-root 도 필요하다")
    items = load_items(a)
    if not items:
        sys.exit("돌릴 오디오가 없다")

    baseline = None
    if a.baseline:
        rows = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        # detail_*.json 은 파일명이 없고 순서가 manifest 와 같다. ref 로 짝짓는다.
        baseline = {}
        by_ref = {r.get("ref"): r.get("hyp", "") for r in rows}
        for it in items:
            if it.get("ref") in by_ref:
                baseline[Path(it["path"]).name] = by_ref[it["ref"]]
        print(f"기준선 {len(baseline)}/{len(items)} 발화 짝지음")

    cfg = session_config(a.silence_ms, a.transcribe_model, a.voice)
    print(f"{a.model} | 발화 {len(items)}개 | VAD {a.silence_ms}ms | trim {a.trim} | "
          f"전사 {a.transcribe_model} | {'새 세션마다' if a.fresh else '한 세션'}")
    # 한 줄씩 붙일 곳. --json 옆에 .jsonl 로 둔다 — 중간에 터져도 여기까지는 남는다.
    sink = Path(a.json).with_suffix(".jsonl") if a.json else None
    if sink and sink.exists():
        sink.unlink()
    rows = asyncio.run(run(a.model, items, cfg, a.trim, a.pad, a.fresh, sink))

    print()
    print("\n".join(summarize(a.model, rows, baseline)))

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "silence_ms": a.silence_ms, "trim": a.trim,
             "transcribe_model": a.transcribe_model, "fresh": a.fresh, "rows": rows},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  저장: {a.json}")


if __name__ == "__main__":
    main()
