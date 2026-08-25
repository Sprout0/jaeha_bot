"""후보 호출어가 **합성 속도를 얼마나 견디는가** + Supertonic 화자의 F0 분포.

왜 이걸 먼저 재나 (2026-08-25):
  v2~v5 가 실패한 자리가 여기다. '재하봇'은 Supertonic 합성 속도 1.15 부터 음절을 삼켰다
  (1.25~1.40 통과 7%). 그래서 **정속으로 합성한 뒤 WSOLA 로 빠르기를 얹는** 우회를 썼는데,
  실기에선 그게 안 통했다 — 실음성 44건에서 '빠르게' 통과 10%.
  결론은 하나였다: **합성 배속과 진짜 빠른 말은 다르다.**

  ➡️ 후보가 높은 합성 속도를 견디면, 우회 없이 **진짜 빠른 발화**를 학습셋에 넣을 수 있다.
     견디는 상한이 곧 --expand(WSOLA) 로 메워야 할 구간의 시작점이다.

두 번째로 F0 를 본다:
  v4 실기 실패의 원인이 분포 불일치였다 — 학습 긍정 F0 중앙 165Hz vs 사용자 104~116Hz.
  v5 는 이걸 '피치를 낮추는 증강'으로 풀려다 실패했다(다른 칸을 깎아 먹음).
  화자를 골라서 푸는 게 맞는지 보려면 **Supertonic 화자별 F0 를 알아야 한다.**

사용:
  python tools/probe_wake_speed.py --phrase "하이 티드" --device cpu --compute-type int8
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models              # noqa: E402
from app.stt_module import STTModule            # noqa: E402
from app.tts_module import TTSModule            # noqa: E402
from app.wake import best_wake_ratio            # noqa: E402
from tools.record_wake_real import f0_median    # noqa: E402

TARGET_SR = 16000
VOICES = ["F1", "F2", "F3", "F5", "M1", "M2", "M3", "M5", "F2+F5", "M1+M3", "F1+M2", "M2+M5"]
SPEEDS = [0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45]
QC_CUT = 0.25   # 힌트 없이 whisper 가 읽어 주는가 (하이 티드 실측 최대 0.222)


def main() -> int:
    ap = argparse.ArgumentParser(description="호출어 후보의 합성 속도 내성 + 화자 F0")
    ap.add_argument("--phrase", default="하이 티드")
    ap.add_argument("--device", default=None)
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--seed-base", type=int, default=9100)
    args = ap.parse_args()

    word = args.phrase.replace(" ", "")
    models = load_models()
    tcfg, scfg = models.get("tts", {}), models.get("stt", {})
    tts = TTSModule(model=tcfg.get("model", "supertonic-3"), voice=VOICES[0],
                    language=tcfg.get("language", "ko"),
                    total_steps=int(tcfg.get("total_steps", 24)),
                    threads=int(tcfg.get("threads", 4)),
                    providers=["CPUExecutionProvider"])
    tts.load()
    src_sr = tts.sample_rate
    stt = STTModule(model_size=scfg.get("model_size", "medium"),
                    device=args.device or scfg.get("device", "cpu"),
                    compute_type=args.compute_type or scfg.get("compute_type", "int8"),
                    language=scfg.get("language", "ko"), keywords=[], aliases={})
    stt.load()

    by_speed = collections.defaultdict(list)
    by_voice_f0 = collections.defaultdict(list)
    for si, speed in enumerate(SPEEDS):
        for vi, voice in enumerate(VOICES):
            tts._style = tts._resolve_style(voice)
            tts.speed = speed
            tts.seed = args.seed_base + si * 100 + vi
            a = tts._trim(tts._infer(args.phrase), keep_tail=0.15)
            a = tts._resample(a, src_sr, TARGET_SR)
            peak = float(np.abs(a).max()) if a.size else 0.0
            if peak > 0:
                a = (a / peak * 0.95).astype(np.float32)
            txt, _ = stt.transcribe(a, initial_prompt=None)
            r = best_wake_ratio(txt or "", word)
            dur = a.size / TARGET_SR
            by_speed[speed].append((r, dur, txt or "", voice))
            if speed == 1.05:
                by_voice_f0[voice].append(f0_median(a, TARGET_SR))
            print(f"  x{speed:.2f} {voice:6} {dur:.2f}s  {r:.2f}  '{(txt or '없음')[:22]}'")

    print()
    print("=" * 72)
    print(f"[A] 합성 속도 내성 — '{args.phrase}' (컷 {QC_CUT})")
    print(f"{'속도':>6} {'통과율':>8} {'자모거리중앙':>12} {'길이중앙':>9}   삼킨 예")
    print("-" * 72)
    for speed in SPEEDS:
        rows = by_speed[speed]
        rr = np.array([r for r, _, _, _ in rows])
        dd = np.array([d for _, d, _, _ in rows])
        bad = [t for r, _, t, _ in rows if r > QC_CUT][:2]
        print(f"{speed:>6.2f} {100 * (rr <= QC_CUT).mean():>7.0f}% {np.median(rr):>12.3f} "
              f"{np.median(dd):>8.2f}s   {' / '.join(b[:14] for b in bad)}")
    print("=" * 72)
    print("[B] Supertonic 화자별 F0 (x1.05) — 사용자 실측 104~116Hz 와 비교")
    print("-" * 72)
    vals = []
    for voice in VOICES:
        f = by_voice_f0[voice]
        if f and f[0] > 0:
            vals.append(f[0])
            print(f"  {voice:8} {f[0]:>6.0f} Hz")
    if vals:
        v = np.array(vals)
        print(f"  {'전체':8} 중앙 {np.median(v):.0f} Hz / 최저 {v.min():.0f} Hz / 최고 {v.max():.0f} Hz")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
