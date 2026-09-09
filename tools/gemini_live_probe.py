"""Gemini Live 를 **OpenAI Realtime 과 같은 자로** 잰다.

09-02 에 `tools/realtime_probe.py` 로 OpenAI 를 쟀다. 여기서 바꾸는 건 벤더 하나뿐이고,
말끝 자르기·실시간 흘리기·채점은 그 도구 것을 그대로 쓴다 — 다른 자로 재면 비교가 아니다.

🔴 재는 방법이 결과를 만든다. OpenAI 쪽에서 두 번 데었다:
  - 오디오를 한꺼번에 부으면 서버 VAD 가 실제와 다르게 돌아 지연이 0 이 된다.
  - 무음을 실시간보다 **느리게** 흘리면 그 지연이 그대로 측정값에 더해진다.
    (첫 탐색에서 드리프트 보정 없이 sleep(0.04) 만 했더니 8.8 초가 나왔다.)
  그래서 아래 feeder 는 벽시계 기준으로 보정하며 흘린다.

⚠️ 무료 티어는 구글이 학습에 쓴다(.env.example 참고). 아이 녹취와 실기 프롬프트는
   이 도구로 보내지 않는다. 어른 실음성만 쓴다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.realtime_probe import CHUNK_MS, trim_edges, trim_speech  # noqa: E402

IN_SR = 16000          # Gemini 입력 규격. OpenAI 는 24000 이다.
OUT_SR = 24000         # 출력은 24000

# 100만 토큰당 USD. https://ai.google.dev/gemini-api/docs/pricing (2026-09 확인)
# 🔴 캐시 단가는 공개돼 있지 않다 — 그래서 여기 안 적는다. cached_tokens() 로 수만 드러낸다.
PRICE = {
    "gemini-2.5-flash-native-audio-preview-12-2025":
        {"text_in": 0.50, "audio_in": 3.00, "text_out": 2.00, "audio_out": 12.00},
    "gemini-2.5-flash-native-audio-latest":
        {"text_in": 0.50, "audio_in": 3.00, "text_out": 2.00, "audio_out": 12.00},
    "gemini-3.1-flash-live-preview":
        {"text_in": 0.75, "audio_in": 3.00, "text_out": 4.50, "audio_out": 12.00},
}
_MOD = {"TEXT": "text", "AUDIO": "audio"}     # 모르는 모달리티는 세지 않는다


def _modal(details, suffix: str, price: dict) -> float | None:
    """[{modality, token_count}] 를 값으로. 아는 모달리티만 센다."""
    total = 0.0
    for d in details or []:
        mod = getattr(d.modality, "value", d.modality)
        key = _MOD.get(str(mod))
        if key:
            total += (d.token_count or 0) * price[f"{key}_{suffix}"]
    return total


def cost_usd(model: str, usage) -> float | None:
    """usage 를 그대로 값으로. 모양이 낯설면 None — 추측해서 채우지 않는다."""
    p = PRICE.get(model)
    if not p or usage is None:
        return None
    pin = getattr(usage, "prompt_tokens_details", None)
    pout = getattr(usage, "response_tokens_details", None)
    if not pin and not pout:
        return None
    return (_modal(pin, "in", p) + _modal(pout, "out", p)) / 1e6


def cached_tokens(usage) -> int:
    """캐시로 처리된 토큰 수. 단가가 공개 안 돼 있어 값 대신 수를 드러낸다."""
    if usage is None:
        return 0
    return getattr(usage, "cached_content_token_count", None) or 0


def load_env() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def session_config(silence_ms: int, instructions: str,
                   lang: str | None = None) -> dict:
    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": instructions,
        "realtime_input_config": {
            "automatic_activity_detection": {
                "silence_duration_ms": silence_ms, "prefix_padding_ms": 300}},
        "input_audio_transcription": {"language_codes": [lang]} if lang else {},
        "output_audio_transcription": {},
    }


def load_pcm(path: Path, trim: str) -> bytes:
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a[:, 0]
    a = {"speech": trim_speech, "edges": trim_edges}.get(trim, lambda x: x)(a)
    if sr != IN_SR:
        import soxr
        a = soxr.resample(a, sr, IN_SR).astype("float32")
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


async def one_turn(session, pcm: bytes, pad_s: float) -> dict:
    """OpenAI one_turn 과 같은 절차. 실시간 속도로 흘리고 뒤에 무음을 붙인다."""
    from google.genai import types

    step = IN_SR * CHUNK_MS // 1000 * 2
    dt = CHUNK_MS / 1000.0
    blob = lambda b: types.Blob(data=b, mime_type=f"audio/pcm;rate={IN_SR}")

    t0 = time.perf_counter()
    for i in range(0, len(pcm), step):
        await session.send_realtime_input(audio=blob(pcm[i:i + step]))
        await asyncio.sleep(max(0.0, t0 + (i // step + 1) * dt - time.perf_counter()))
    t_end = time.perf_counter()          # 말이 끝난 순간

    quiet = bytes(step)
    m: dict = {}

    async def feeder():
        for k in range(int(pad_s / dt)):
            await session.send_realtime_input(audio=blob(quiet))
            await asyncio.sleep(max(0.0, t_end + (k + 1) * dt - time.perf_counter()))

    feed = asyncio.create_task(feeder())
    try:
        async for r in session.receive():
            now = time.perf_counter() - t_end
            sc = r.server_content
            if sc:
                it = getattr(sc, "input_transcription", None)
                if it and it.text:
                    m["heard"] = (m.get("heard", "") + it.text)
                ot = getattr(sc, "output_transcription", None)
                if ot and ot.text:
                    m.setdefault("t_first_text", now)
                    m["said"] = (m.get("said", "") + ot.text)
                mt = getattr(sc, "model_turn", None)
                if mt:
                    for part in mt.parts or []:
                        if part.inline_data and part.inline_data.data:
                            m.setdefault("t_first_audio", now)
                            m["out_bytes"] = m.get("out_bytes", 0) + len(part.inline_data.data)
            um = getattr(r, "usage_metadata", None)
            if um:
                m["usage"] = um
            if sc and sc.turn_complete:
                break
    finally:
        feed.cancel()
    for k in ("heard", "said"):
        if k in m:
            m[k] = m[k].strip()
    return m


async def run(model: str, items: list[Path], cfg: dict, trim: str,
              pad_s: float, sink: Path | None) -> list[dict]:
    import google.genai as genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rows: list[dict] = []
    for n, path in enumerate(items, 1):
        pcm = load_pcm(path, trim)
        row: dict = {"file": path.name, "sec": len(pcm) / 2 / IN_SR}
        try:
            # 발화마다 새 세션 — 이력 누적을 없애 발화끼리 독립으로 만든다.
            async with client.aio.live.connect(model=model, config=cfg) as s:
                row |= await one_turn(s, pcm, pad_s)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        usage = row.pop("usage", None)
        row["cost_usd"] = cost_usd(model, usage)
        row["cached_tokens"] = cached_tokens(usage)
        if usage is not None:
            row["tokens"] = {
                "prompt": usage.prompt_token_count, "response": usage.response_token_count,
                "detail_in": [(str(getattr(d.modality, "value", d.modality)), d.token_count)
                              for d in (usage.prompt_tokens_details or [])],
                "detail_out": [(str(getattr(d.modality, "value", d.modality)), d.token_count)
                               for d in (usage.response_tokens_details or [])]}
        rows.append(row)
        print(f"  [{n}/{len(items)}] {path.name:22} "
              f"첫소리 {row.get('t_first_audio', float('nan')):.3f}s  "
              f"들은말 {row.get('heard', '')[:24]!r}")
        if sink:
            with sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return rows


def summarize(model: str, rows: list[dict]) -> list[str]:
    ok = [r for r in rows if r.get("t_first_audio")]
    out = [f"== {model} · {len(ok)}/{len(rows)} 성공 =="]
    if not ok:
        return out + ["  전부 실패 — " + str(rows[0].get("error", ""))[:160]]
    for label, key in (("말끝→첫 소리", "t_first_audio"), ("말끝→첫 글자", "t_first_text")):
        v = [r[key] for r in ok if r.get(key)]
        if v:
            out.append(f"  {label}: 중앙 {st.median(v):.3f}s  "
                       f"최소 {min(v):.3f}  최대 {max(v):.3f}")
    lead = [r["t_first_audio"] - r["t_first_text"] for r in ok
            if r.get("t_first_text") and r.get("t_first_audio")]
    if lead:
        out.append(f"  전사가 소리보다 앞섬: 중앙 {st.median(lead):.3f}s")
    c = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    if c:
        out.append(f"  턴당 비용: 중앙 ${st.median(c):.5f}  합계 ${sum(c):.4f}")
    cached = sum(r.get("cached_tokens", 0) for r in ok)
    out.append(f"  캐시된 토큰 {cached} — " +
               ("0 이므로 위 비용은 실값이다." if not cached else
                "🔴 캐시 단가가 공개 안 돼 위 비용은 **상한**이다."))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Gemini Live 를 OpenAI 와 같은 자로 잰다")
    p.add_argument("--model", default="gemini-2.5-flash-native-audio-preview-12-2025",
                   choices=sorted(PRICE))
    p.add_argument("--wav-dir", default="data/wake_real/adult_20260826_1445")
    p.add_argument("--silence-ms", type=int, default=1200,
                   help="우리 꼬리와 맞추려면 1200. OpenAI 측정과 같은 값")
    p.add_argument("--trim", default="speech", choices=("speech", "edges", "none"))
    p.add_argument("--pad", type=float, default=5.0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--instructions",
                   default="너는 다섯 살 아이의 친구 로봇 '티드'야. 밝은 반말로 한두 문장만, "
                           "30자 안쪽으로 짧게 말해. 이모지나 기호는 쓰지 마.")
    p.add_argument("--lang", default=None,
                   help="입력 전사 언어(예: ko-KR). 안 주면 자동 — 09-09 실측에서 "
                        "'하이 티드'를 hated/はい てる 로 적었다")
    p.add_argument("--json")
    a = p.parse_args()

    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY 가 없다 (.env 확인)")

    items = sorted(Path(a.wav_dir).glob("*.wav"))[:a.n or None]
    if not items:
        sys.exit(f"{a.wav_dir} 에 wav 가 없다")
    cfg = session_config(a.silence_ms, a.instructions, a.lang)
    print(f"{a.model} | 발화 {len(items)}개 | VAD {a.silence_ms}ms | "
          f"입력 {IN_SR}Hz | trim {a.trim} | 전사언어 {a.lang or '자동'}")

    sink = Path(a.json).with_suffix(".jsonl") if a.json else None
    if sink and sink.exists():
        sink.unlink()
    rows = asyncio.run(run(a.model, items, cfg, a.trim, a.pad, sink))
    print()
    print("\n".join(summarize(a.model, rows)))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "silence_ms": a.silence_ms, "in_sr": IN_SR,
             "trim": a.trim, "rows": rows}, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"\n  저장: {a.json}")


if __name__ == "__main__":
    main()
