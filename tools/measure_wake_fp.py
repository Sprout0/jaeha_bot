"""실제 소음(TV·유튜브·대화)으로 **헛깨움을 실측**한다. 합성 부정으로는 두 번 틀렸다.

왜 필요한가 (2026-08-12):
  2단계 검증을 켰더니 유튜브를 틀어 놓기만 해도 깨어났다. 원인은 initial_prompt 가
  만든 환각인데, 나는 그 확률을 **합성 부정(깨끗한 단어 한 개)** 으로만 재고 2% 라고
  판단했다. 실제 말소리에서는 전혀 다른 값이었다. 부정 표본이 현실과 달랐던 것이다.
  → 이제 실제 방 소리를 녹음해서 다음을 한 번에 잰다:
      ① 1단계 임계별 **후보 발생률**(회/시간)
      ② **환각률 H** = 후보 중 2단계(힌트 있음)를 통과하는 비율
      ③ **두 번 전사 대조**(힌트 있음 AND 없음)가 H 를 얼마나 낮추는지
  헛깨움 = 후보율 × H 이므로, 이 셋이면 임계값을 계산으로 정할 수 있다.

사용법 (젯슨에서, **봇을 끄고**, 평소처럼 TV·유튜브를 틀어 둔 채):
    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/measure_wake_fp.py --seconds 180
    python tools/measure_wake_fp.py --wav logs/noise_xxx.wav   # 녹음 없이 재분석
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio_source import FRAME, SAMPLE_RATE  # noqa: E402
from app.config import load_models  # noqa: E402
from app.wake import best_wake_ratio  # noqa: E402
from app.wake_onnx import OnnxWakeDetector, _peak_rms  # noqa: E402

THRS = [0.25, 0.15, 0.10, 0.05, 0.03]
WIN_S = 2.0          # 검증에 넘기는 창(런타임과 같게)
COOLDOWN_S = 1.0     # 기각 후 최소 대기(런타임과 같게)


def record(seconds: int, outdir: str) -> tuple[np.ndarray, str]:
    import sounddevice as sd
    import soundfile as sf

    name = (load_models().get("audio") or {}).get("device")
    dev = None
    for i, d in enumerate(sd.query_devices()):
        if name and name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            dev, _ = i, print(f"입력 장치: [{i}] {d['name']}")
            break
    print(f"{seconds}초 녹음 — 평소처럼 소리를 틀어 두세요...")
    y = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
               dtype="float32", device=dev, blocking=True).reshape(-1)
    if _peak_rms(y) < 2e-4:
        raise SystemExit("🔴 무음이다. 봇이 돌고 있거나 마이크가 안 잡힌다.")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"noise_{datetime.now():%Y%m%d_%H%M}.wav")
    sf.write(path, y, SAMPLE_RATE)
    print(f"저장: {path}  (최대프레임 RMS {_peak_rms(y):.4f})")
    return y, path


def candidates(det: OnnxWakeDetector, frames, thr: float) -> list[tuple[int, float]]:
    """런타임과 같은 규칙(히스테리시스+쿨다운)으로 후보 시점을 모은다."""
    det.reset()
    out, armed, last = [], True, -1e9
    for i, f in enumerate(frames):
        s = det.push(f)
        if s is None:
            continue
        if s < thr:
            armed = True
            continue
        t = i * FRAME / SAMPLE_RATE
        if not armed or t - last < COOLDOWN_S:
            continue
        armed, last = False, t
        out.append((i, s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--wav", help="이미 녹음한 파일로 재분석")
    ap.add_argument("--max-verify", type=int, default=60, help="whisper 호출 상한")
    args = ap.parse_args()

    import soundfile as sf

    if args.wav:
        y, _ = sf.read(args.wav, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        path = args.wav
    else:
        y, path = record(args.seconds, "logs")
    dur_h = y.size / SAMPLE_RATE / 3600
    print(f"\n분석 대상 {y.size/SAMPLE_RATE:.0f}초\n")

    cfg = load_models()
    o = cfg["wake"]["onnx"]
    v = o["verify"]
    det = OnnxWakeDetector(model_dir=o["model_dir"], classifier=o["classifier"],
                           threshold=0.03, trigger_frames=1)
    frames = [y[i:i + FRAME] for i in range(0, y.size - FRAME + 1, FRAME)]

    # ── ① 임계별 후보율 ──────────────────────────────────────────────
    cands = {t: candidates(det, frames, t) for t in THRS}
    print("① 1단계 후보 발생률 (이 소리 기준)")
    print(f"{'임계':>6}{'후보':>7}{'회/시간':>10}   참고: 합성 코퍼스 값")
    ref = {0.25: 0.47, 0.15: 0.80, 0.10: 2.21, 0.05: 5.78, 0.03: 13.62}
    for t in THRS:
        n = len(cands[t])
        print(f"{t:>6.2f}{n:>7}{n/dur_h:>10.1f}   {ref[t]:>6.2f}")

    # ── ②③ 후보를 두 방식으로 검증 ─────────────────────────────────
    from app.stt_module import STTModule
    stt = STTModule(**cfg.get("stt", {}))
    stt.load()
    prompt, cut = v["initial_prompt"], float(v["max_ratio"])
    word = cfg["wake"]["word"]
    win = int(WIN_S * SAMPLE_RATE)

    seen: dict[int, tuple[float, float, str, str]] = {}
    todo = sorted({i for t in THRS for i, _ in cands[t]})[: args.max_verify]
    print(f"\n후보 {len(todo)}곳을 두 방식으로 전사 중(whisper {len(todo)*2}회)...")
    t0 = time.perf_counter()
    for i in todo:
        end = (i + 1) * FRAME
        seg = y[max(0, end - win):end]
        a, _ = stt.transcribe(seg, initial_prompt=prompt)
        b, _ = stt.transcribe(seg, initial_prompt=None)
        seen[i] = (best_wake_ratio(a or "", word), best_wake_ratio(b or "", word),
                   (a or "").strip(), (b or "").strip())
    print(f"  ({time.perf_counter()-t0:.0f}초)\n")

    print("②③ 검증 방식별 통과율 = 환각률 H")
    print(f"{'임계':>6}{'후보':>6}{'힌트만':>9}{'H':>7}{'두번대조':>10}{'H':>7}"
          f"{'헛깨움 추정':>13}")
    for t in THRS:
        idx = [i for i, _ in cands[t] if i in seen]
        if not idx:
            print(f"{t:>6.2f}{0:>6}{'-':>9}{'-':>7}{'-':>10}{'-':>7}{'-':>13}")
            continue
        one = sum(1 for i in idx if seen[i][0] <= cut)
        two = sum(1 for i in idx if seen[i][0] <= cut and seen[i][1] <= 0.65)
        h1, h2 = one / len(idx), two / len(idx)
        rate = len(cands[t]) / dur_h
        print(f"{t:>6.2f}{len(idx):>6}{one:>8}건{h1*100:>6.0f}%{two:>9}건{h2*100:>6.0f}%"
              f"{rate*h1:>7.2f}/{rate*h2:.2f}")

    print("\n🔴 힌트만으로 통과한 것들 — 환각의 실제 모습")
    shown = 0
    for i in todo:
        r1, r2, a, b = seen[i]
        if r1 <= cut and shown < 8:
            shown += 1
            print(f"   {i*FRAME/SAMPLE_RATE:>6.1f}s  힌트있음 {r1:.2f} '{a[:26]}'")
            print(f"           힌트없음 {r2:.2f} '{b[:34]}'")
    if not shown:
        print("   (없음 — 이 소리에서는 환각이 안 났다)")
    print(f"\n원본: {path}  — 다음 실험에도 같은 소리로 비교할 수 있다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
