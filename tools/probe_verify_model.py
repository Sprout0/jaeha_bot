"""2단계 검증을 어느 whisper 크기로 할 것인가 — 실음성으로 가른다.

🔴 왜 이게 지금 유일한 손잡이인가 (2026-08-26):
  실음성 20건에서 1단계는 19/20 을 잡는데 캐스케이드는 11/20 이다. 8건을 whisper 가
  죽인다. 그런데 다른 길이 전부 막혔다:
    - 단일단계 전환  -> 기각. 실제 방 소음이 0.05 에서 **160회·시간**이다(합성 예측의 23배).
    - 우회컷(확신하면 검증 생략) -> 기각. 방 소음이 0.725 까지 올라 진짜 호출(최고 0.873)과
      분포가 겹친다. 안전한 컷이 없다.
  ➡️ 남은 건 **whisper 가 더 잘 읽게 하는 것**뿐이다.

무엇을 재나: 같은 wav 를 크기별로 전사해 호출어와의 자모거리를 낸다. 거리가
`plain_max_ratio` 아래여야 2단계를 통과한다. 즉 이 표가 곧 캐스케이드 재현율이다.

⚠️ 검증은 **2초 클립 하나**다. 대화 STT 와 비용 구조가 다르다 — 대화가 medium 이라고
   검증까지 medium 일 이유는 없다. 다만 젯슨 메모리는 같이 쓴다(모델을 둘 올리면
   그만큼 더 먹는다). 그래서 속도와 함께 잰다.

사용법 (젯슨에서, 봇을 끄고):
    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/probe_verify_model.py --dir data/wake_real/adult_20260826_1445
    python tools/probe_verify_model.py --dir ... --sizes medium,large-v3-turbo
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models  # noqa: E402
from app.wake import best_wake_ratio  # noqa: E402


def load_clips(d: str) -> list[dict]:
    """녹음 폴더에서 wav 를 읽는다. takes.jsonl 이 있으면 조건·점수도 같이."""
    meta = {}
    mp = os.path.join(d, "takes.jsonl")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    meta[os.path.basename(r["path"])] = r
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
        name = os.path.basename(p)
        if name == "session.wav":          # 통짜 녹음은 건너뛴다
            continue
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        m = meta.get(name, {})
        out.append({"name": name, "audio": y, "sr": sr,
                    "cond": m.get("cond", "?"), "score": m.get("score", float("nan"))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="녹음 폴더(data/wake_real/...)")
    ap.add_argument("--sizes", default="medium,large-v3-turbo",
                    help="쉼표로 구분한 whisper 크기")
    ap.add_argument("--hint", action="store_true",
                    help="힌트(initial_prompt)를 주고도 재본다 — 2패스의 2관문")
    args = ap.parse_args()

    wcfg = load_models()["wake"]
    word = wcfg["word"]
    vcfg = (wcfg["onnx"].get("verify") or {})
    cut = float(vcfg.get("plain_max_ratio", 0.35))
    thr = float(wcfg["onnx"]["threshold"])
    prompt = vcfg.get("initial_prompt") or None

    clips = load_clips(args.dir)
    if not clips:
        print(f"🔴 wav 가 없다: {args.dir}")
        return 1
    print(f"클립 {len(clips)}개 · 호출어 '{word}' · 2단계 컷 {cut} · 1단계 임계 {thr}\n")

    from app.stt_module import STTModule
    base = load_models().get("stt", {})
    results: dict[str, list[dict]] = {}

    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        print(f"── {size} 로드 중...")
        cfg = dict(base)
        cfg["model_size"] = size
        stt = STTModule(**cfg)
        stt.load()
        rows, t0 = [], time.time()
        for c in clips:
            a = time.time()
            plain, _ = stt.transcribe(c["audio"], initial_prompt=None)
            dt = time.time() - a
            row = {"name": c["name"], "cond": c["cond"], "score": c["score"],
                   "plain": (plain or "").strip(),
                   "r": best_wake_ratio((plain or "").strip(), word), "sec": dt}
            if args.hint:
                h, _ = stt.transcribe(c["audio"], initial_prompt=prompt)
                row["hinted"] = (h or "").strip()
                row["rh"] = best_wake_ratio(row["hinted"], word)
            rows.append(row)
        results[size] = rows
        print(f"   {len(rows)}개 전사 완료 ({time.time()-t0:.0f}초, "
              f"클립당 {np.mean([r['sec'] for r in rows]):.2f}초)")
        del stt

    sizes = list(results)
    print(f"\n{'클립':<16}{'조건':<7}{'1단계':>7}   " +
          "   ".join(f"{s:<22}" for s in sizes))
    print("-" * (32 + 25 * len(sizes)))
    for i, c in enumerate(clips):
        line = f"{c['name'][:15]:<16}{c['cond']:<7}{c['score']:>7.3f}   "
        for s in sizes:
            r = results[s][i]
            mark = "○" if r["r"] <= cut else "✗"
            line += f"{mark} {r['r']:.2f} {r['plain'][:12]:<12}  "
        print(line)

    print("-" * (32 + 25 * len(sizes)))
    print(f"\n{'크기':<18}{'정확전사':>9}{'2단계통과':>10}{'캐스케이드':>11}{'클립당':>9}")
    for s in sizes:
        rows = results[s]
        exact = sum(1 for r in rows if r["r"] == 0.0)
        pass2 = sum(1 for r in rows if r["r"] <= cut)
        casc = sum(1 for r in rows if r["score"] >= thr and r["r"] <= cut)
        print(f"{s:<18}{exact:>6}/{len(rows)}{pass2:>7}/{len(rows)}"
              f"{casc:>8}/{len(rows)}{np.mean([r['sec'] for r in rows]):>8.2f}초")

    if args.hint:
        print(f"\n힌트를 주면(2패스의 2관문, 컷 {vcfg.get('max_ratio', 0.45)}):")
        for s in sizes:
            rows = results[s]
            print(f"  {s:<18}정확 {sum(1 for r in rows if r['rh'] == 0.0)}/{len(rows)}"
                  f"   중앙거리 {np.median([r['rh'] for r in rows]):.3f}")

    print(f"""
⚠️ 이 표는 **조용한 방·성인 남성 1명**이다. 아이도 소음도 없다.
⚠️ 검증은 2초 클립 하나라 대화 STT 와 비용 구조가 다르다. 다만 젯슨 메모리는
   같이 쓴다 — 대화용과 검증용을 다른 크기로 두면 모델을 둘 올려야 한다.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
