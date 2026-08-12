"""학습 데이터를 Colab 에 올리기 전에 점검한다. 어느 버전(output_vN)이든 쓴다.

왜 필요한가:
  증강은 **에러 없이 조용히 망가진다.** v3 는 배속이 안 걸린 데이터로 학습해 놓고
  오프라인 recall 91% 로 통과했고, 실기에서야 실패를 알았다. Colab 에 올리기 전에
  '내가 만들려던 것이 실제로 만들어졌는지'를 파일로 확인한다.

보는 것:
  1. 개수        — 학습 시간과 노트북 assert 가 여기 걸려 있다
  2. 격자 칸별   — 칸이 비면 그 조건은 실기에서 실패한다
  3. 길이비      — WSOLA 가 실제로 배속을 걸었는가(안 걸려도 예외가 안 난다)
  4. 길이·클리핑 — 2초 넘는 폭주, 진폭이 깎인 파일
  5. STT 통과율  — 칸별로 '사람이 알아들을 수 있는 소리'인지 (--no-stt 로 생략)

사용법:
    python tools/sanity_wake_data.py --dir output_v5
    python tools/sanity_wake_data.py --dir output_v5 --no-stt      # 빠름
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIRS = ("positive_train", "positive_test", "negative_train", "negative_test")
MAX_DUR = 2.0        # 이보다 길면 폭주를 의심한다
STT_PER_CELL = 8     # 칸당 이만큼만 전사해 본다(전수는 너무 느리다)
QC_MAX_RATIO = 0.45  # 자모거리 컷. tools/wake_qc.py 기본값과 같게 둔다


def load(dirpath: str) -> list[dict]:
    p = os.path.join(dirpath, "manifest.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def check_counts(root: str) -> bool:
    print("1. 개수")
    ok = True
    for d in DIRS:
        n_wav = len(glob.glob(os.path.join(root, d, "clip_*.wav")))
        rows = load(os.path.join(root, d))
        orig = sum(1 for r in rows if "src" not in r)
        mark = "OK " if n_wav == len(rows) else "🔴"
        ok = ok and n_wav == len(rows)
        print(f"   {mark} {d:<16} wav {n_wav:>6} / 매니페스트 {len(rows):>6}"
              f"  (원본 {orig} + 증강 {len(rows)-orig})")
    return ok


def check_grid(root: str) -> bool:
    """칸별 개수와 실제 길이비. 긍정·부정이 **같은 증강**을 받았는지도 본다."""
    print("\n2~3. 격자 칸별 개수와 실제 길이비")
    ok = True
    cells_per_dir = {}
    for d in DIRS:
        rows = [r for r in load(os.path.join(root, d)) if "src" in r]
        if not rows:
            continue
        by = defaultdict(list)
        for r in rows:
            by[(r["rate"], r.get("pitch", 1.0))].append(r)
        cells_per_dir[d] = set(by)
        print(f"   [{d}]")
        src_dur = {r["clip"]: r["dur"] for r in load(os.path.join(root, d))}
        for cell in sorted(by):
            g = by[cell]
            ratios = [r["dur"] / src_dur[r["src"]] for r in g
                      if src_dur.get(r["src"], 0) > 0]
            got = float(np.mean(ratios)) if ratios else 0.0
            want = 1.0 / cell[0]
            off = abs(got - want)
            ok = ok and off <= 0.08
            print(f"      시간 {cell[0]:<5} 피치 {cell[1]:<5} n={len(g):>5}"
                  f"   길이비 {got:.3f} (기대 {want:.3f})"
                  f"{'  🔴 배속이 안 걸렸다' if off > 0.08 else ''}")

    # 클래스 대칭: 긍정에만 걸린 증강이 있으면 모델은 '그 변형 = 호출어'를 배운다
    pos = cells_per_dir.get("positive_train", set())
    neg = cells_per_dir.get("negative_train", set())
    if pos != neg:
        print(f"   🔴 긍정과 부정의 격자가 다르다. 긍정만: {sorted(pos-neg)} / "
              f"부정만: {sorted(neg-pos)}")
        ok = False
    else:
        print(f"   OK  긍정·부정이 같은 {len(pos)}칸을 받았다(클래스 대칭)")
    return ok


def check_audio(root: str) -> bool:
    print("\n4. 길이·클리핑")
    import soundfile as sf
    ok = True
    rng = random.Random(0)
    for d in DIRS:
        rows = load(os.path.join(root, d))
        if not rows:
            continue
        longs = [r for r in rows if r.get("dur", 0) > MAX_DUR]
        pick = rng.sample(rows, min(200, len(rows)))
        clipped = 0
        for r in pick:
            y, _ = sf.read(os.path.join(root, d, r["clip"]), dtype="float32")
            if np.mean(np.abs(y) >= 0.999) > 0.001:
                clipped += 1
        mark = "OK " if not longs and not clipped else "🔴"
        ok = ok and not longs and not clipped
        print(f"   {mark} {d:<16} {MAX_DUR}s 초과 {len(longs)}개 "
              f"/ 표본 {len(pick)}개 중 클리핑 {clipped}개"
              f"  (최대 {max((r.get('dur',0) for r in rows), default=0):.2f}s)")
    return ok


def check_stt(root: str, word: str) -> bool:
    """칸별로 whisper 가 알아듣는지. **사람이 들을 수 있는 소리인가**의 대리 지표다."""
    print(f"\n5. 칸별 STT 통과율 (칸당 {STT_PER_CELL}개, 긍정만)")
    from app.config import load_models
    from app.stt_module import STTModule
    from app.wake import best_wake_ratio   # 검수(tools/wake_qc.py)와 같은 판정을 쓴다

    stt = STTModule(**load_models().get("stt", {}))
    stt.load()
    rng = random.Random(0)
    ok = True
    d = "positive_train"
    rows = load(os.path.join(root, d))
    by = defaultdict(list)
    for r in rows:
        by[(r.get("rate", 1.0), r.get("pitch", 1.0))].append(r)
    for cell in sorted(by):
        pick = rng.sample(by[cell], min(STT_PER_CELL, len(by[cell])))
        hit = 0
        for r in pick:
            text, _ = stt.transcribe(os.path.join(root, d, r["clip"]))
            if best_wake_ratio(text or "", word) <= QC_MAX_RATIO:
                hit += 1
        rate = hit / len(pick)
        ok = ok and rate >= 0.5
        print(f"   시간 {cell[0]:<5} 피치 {cell[1]:<5} {hit}/{len(pick)}"
              f" = {rate*100:>3.0f}%{'   🔴 낮다' if rate < 0.5 else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="예: output_v5")
    ap.add_argument("--word", default="재하봇")
    ap.add_argument("--no-stt", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"없는 폴더: {args.dir}")

    results = [check_counts(args.dir), check_grid(args.dir), check_audio(args.dir)]
    if not args.no_stt:
        try:
            results.append(check_stt(args.dir, args.word))
        except Exception as e:
            print(f"   ⚠️ STT 점검 실패(무시): {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if all(results):
        print("✅ 전부 통과. Colab 에 올려도 된다.")
        return 0
    print("🔴 통과하지 못한 항목이 있다. 위 🔴 를 먼저 볼 것.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
