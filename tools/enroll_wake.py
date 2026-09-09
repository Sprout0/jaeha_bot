"""호출어 '본보기'를 등록하고, 소음 녹음으로 컷을 정한다.

임베딩 대조(app/wake_embed.py)는 미리 녹음해 둔 본보기와 소리를 견준다. 이 도구가
그 본보기를 만들고, 컷을 **추측이 아니라 측정**으로 정하게 해 준다.

## 쓰는 순서

1) 봇을 쓸 사람이 자기 목소리로 등록한다 (ReSpeaker 로 직접 녹음)
      python tools/enroll_wake.py --record 10
   🔴 **봇이 돌고 있으면 끌 것** — ReSpeaker 는 장치가 하나라 동시에 못 연다.

2) 거실을 30~60분 녹음해 두고, 그 파일로 컷을 정한다
      python tools/enroll_wake.py --score-noise logs/거실.wav
   유사도 분포와 '컷별 헛깨움/시간'이 나온다. **소음 최고치보다 확실히 위**를 고른다.

3) 정한 값을 configs/model_paths.yaml 의
      wake.onnx.verify.embed_rescue.{enabled, min_similarity}
   에 넣는다.

## 이미 있는 녹음으로 만들기

    python tools/enroll_wake.py --from-dir data/wake_real/adult_20260826_1445

🔴 그 폴더는 **아버지 한 분** 것이다. 다른 사람이 봇을 쓸 거라면 그 사람 목소리로
   다시 등록해야 한다 — 화자가 다를 때도 되는지는 아직 아무도 재지 않았다.
[[jaeha-bot-progress]]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.wake_embed import (TEMPLATE_FRAMES, best_similarity,  # noqa: E402
                            make_template, save_templates)
from app.wake_onnx import FRAME, SAMPLE_RATE, OnnxWakeDetector  # noqa: E402

DEFAULT_MODEL_DIR = "models/wake/v6"
DEFAULT_CLASSIFIER = "jaehabot_v6.onnx"
DEFAULT_OUT = "models/wake/v6/templates_haitid.npy"
PAD_S = 2.5     # 링버퍼를 채우려면 앞뒤 무음이 필요하다(record_wake_real 과 같은 값)


def _load_detector(args) -> OnnxWakeDetector:
    return OnnxWakeDetector(args.model_dir, classifier=args.classifier,
                            threshold=args.threshold)


def _scores(det: OnnxWakeDetector, y: np.ndarray) -> np.ndarray:
    """임베딩 열과 같은 길이의 1단계 점수열. 아직 워밍업이면 0."""
    det.reset()
    out = []
    for i in range(0, y.size - FRAME + 1, FRAME):
        s = det.push(y[i:i + FRAME])
        out.append(0.0 if s is None else float(s))
    return np.asarray(out, dtype=np.float32)


def _padded(y: np.ndarray) -> np.ndarray:
    pad = np.zeros(int(PAD_S * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([pad, y.astype(np.float32), pad])


def _read(path: str) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{path}: {sr}Hz 다 — {SAMPLE_RATE}Hz 로 맞춰 줘")
    return y


def _template_from_audio(det: OnnxWakeDetector, y: np.ndarray):
    """녹음 하나 -> 본보기 하나. 점수열을 같이 줘서 감지기가 반응한 자리를 고르게 한다."""
    y = _padded(y)
    embs = det.embed_sequence(y)
    sc = _scores(det, y)
    # embed_sequence 와 push 는 워밍업 길이가 같아 길이가 맞는다. 어긋나면 짧은 쪽에 맞춘다.
    n = min(len(embs), len(sc))
    return make_template(embs[:n], sc[len(sc) - n:]), (float(sc.max()) if sc.size else 0.0)


# ─────────────────────────────────────────────────────────────── 등록
def build_from_dir(det, d: str):
    wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
    if not wavs:
        raise SystemExit(f"wav 가 없다: {d}")
    temps, rows = [], []
    for p in wavs:
        t, peak = _template_from_audio(det, _read(p))
        if t is None:
            print(f"  건너뜀(너무 짧다): {os.path.basename(p)}")
            continue
        temps.append(t)
        rows.append((os.path.basename(p), peak))
    for name, peak in rows:
        print(f"  {name:<24} 1단계 최고 {peak:.3f}")
    return temps


def record_takes(det, n: int):
    """ReSpeaker 로 직접 녹음. 봇이 돌고 있으면 장치를 못 연다."""
    import sounddevice as sd
    print(f"\n'하이 티드' 를 {n}번 부릅니다. 매번 신호가 뜨면 **또박또박** 한 번.")
    print("🔴 봇이 돌고 있으면 먼저 끄세요 — ReSpeaker 는 장치가 하나입니다.\n")
    temps = []
    for i in range(1, n + 1):
        input(f"  [{i}/{n}] 준비되면 Enter -> 2초 녹음")
        y = sd.rec(int(2.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
        sd.wait()
        y = y.reshape(-1)
        t, peak = _template_from_audio(det, y)
        rms = float(np.sqrt(np.mean(np.square(y))))
        if t is None:
            print("      ✗ 너무 짧다 — 다시")
            continue
        if rms < 0.005:
            print(f"      ⚠️ 소리가 거의 없다(RMS {rms:.4f}) — 마이크·거리 확인. 그래도 담는다")
        print(f"      ✓ 담음 (1단계 최고 {peak:.3f}, RMS {rms:.4f})")
        temps.append(t)
    return temps


# ─────────────────────────────────────────────────────── 컷 정하기(소음 채점)
def score_noise(det, temps, path: str, thr: float):
    """긴 소음 녹음에서 **1단계 후보가 뜬 자리만** 유사도를 잰다.

    전 구간을 재면 실제보다 나쁘게 나온다 — 대조는 후보에만 돌기 때문이다.
    """
    y = _read(path)
    hours = y.size / SAMPLE_RATE / 3600
    print(f"\n{path}  ({y.size/SAMPLE_RATE/60:.1f}분)")
    embs = det.embed_sequence(y)
    sc = _scores(det, y)
    n = min(len(embs), len(sc))
    embs, sc = embs[:n], sc[len(sc) - n:]
    k = TEMPLATE_FRAMES
    T = np.stack(temps)
    sims = [best_similarity(embs[i - k + 1:i + 1], T)
            for i in range(k - 1, n) if sc[i] >= thr]
    print(f"1단계 후보 프레임 {len(sims)}개 "
          f"({len(sims)/hours:.1f}프레임·시간, 임계 {thr})")
    if not sims:
        print("🔴 후보가 하나도 없다 — 이 녹음으로는 컷을 정할 수 없다. "
              "실제로 봇이 켜져 있던 환경을 녹음할 것")
        return
    a = np.asarray(sims)
    print(f"유사도  중앙 {np.median(a):.3f}  p99 {np.quantile(a, 0.99):.3f}  "
          f"최대 {a.max():.3f}")
    print("\n컷별 헛깨움(이 녹음 기준)")
    for c in (0.80, 0.82, 0.85, 0.87, 0.90, 0.92):
        k_ = int((a >= c).sum())
        print(f"  {c:.2f}   {k_:>4}프레임  {k_/hours:>7.1f}회·시간")
    print("\n➡️ **소음 최대치보다 확실히 위**를 고를 것. 진짜 호출은 실측 0.87~0.96 이었다.")
    print("   둘이 겹치면 안전한 컷이 없다는 뜻이다 — 그때는 켜지 말 것"
          "(08-26 우회컷이 정확히 그래서 죽었다).")


# ────────────────────────────────────────── 컷 정하기(진짜 호출 채점)
def score_calls(det, temps, d: str, thr: float) -> dict:
    """진짜 호출 녹음들을 본보기에 대조한다 — **컷의 상한**을 준다.

    score_noise 가 하한(소음이 어디까지 올라오나)을, 이쪽이 상한(진짜 호출이
    어디까지 내려가나)을 준다. **둘이 겹치면 안전한 컷이 없다** — 그때는
    임베딩 단독을 켜면 안 된다(08-26 우회컷이 정확히 그래서 죽었다).

    🔴 1단계를 못 넘은 녹음은 유사도 통계에서 **뺀다.** 후보가 떠야 검증기가
       불리므로, 그런 녹음은 2단계의 실패가 아니라 1단계의 실패다. 섞어 세면
       컷을 실제보다 낮게 잡게 되고 소음이 뚫린다.

    ⚠️ 등록에 쓴 녹음을 다시 채점하면 유사도가 1.0 근처로 나온다. **등록에 안 쓴
       녹음 10건 이상**으로 재야 의미가 있다.
    """
    wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
    if not wavs:
        raise SystemExit(f"wav 가 없다: {d}")
    T = np.stack([np.asarray(t, dtype=np.float32) for t in temps])

    rows, sims, miss = [], [], 0
    for p in wavs:
        y = _padded(_read(p))
        embs = det.embed_sequence(y)
        sc = _scores(det, y)
        n = min(len(embs), len(sc))
        peak = float(sc[len(sc) - n:].max()) if n else 0.0
        sim = best_similarity(embs[:n], T)
        ok = peak >= thr
        if ok:
            sims.append(sim)
        else:
            miss += 1
        rows.append((os.path.basename(p), peak, sim, ok))

    print()
    print(f"{d}  ({len(wavs)}건, 1단계 임계 {thr})")
    for name, peak, sim, ok in rows:
        tail = "" if ok else "   <- 1단계에서 못 잡음(2단계 통계에서 뺀다)"
        print(f"  {'  ' if ok else '🔴'} {name:<24} "
              f"1단계 {peak:.3f}  유사도 {sim:.3f}{tail}")

    out = {"rows": rows, "sims": sims, "low": None, "stage1_miss": miss}
    if miss:
        print()
        print(f"🔴 {miss}건이 1단계를 못 넘었다 — 임베딩 컷으로는 못 고치는 실패다.")
    if not sims:
        print("🔴 채점할 게 하나도 없다 — 컷의 상한을 정할 수 없다.")
        return out

    a = np.asarray(sims)
    out["low"] = float(a.min())
    print()
    print(f"유사도  최저 {a.min():.3f}  중앙 {np.median(a):.3f}  최대 {a.max():.3f}")
    print()
    print("컷별 재현율(이 녹음 기준)")
    for c in (0.80, 0.82, 0.85, 0.87, 0.90, 0.92):
        k_ = int((a >= c).sum())
        print(f"  {c:.2f}   {k_:>3}/{len(a)}   {100 * k_ / len(a):>5.0f}%")
    print()
    print(f"➡️ 컷은 **{a.min():.3f} 아래**여야 진짜 호출이 안 죽는다.")
    print("   --score-noise 의 '최대'와 겹치면 안전한 컷이 없다는 뜻이다 — 켜지 말 것.")
    return out


def _load_templates(out: str):
    if not os.path.exists(out):
        raise SystemExit(f"본보기가 없다: {out} — 먼저 등록할 것")
    t = np.load(out)
    return list(t[None, :] if t.ndim == 1 else t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=int, metavar="N", help="마이크로 N번 녹음해 등록")
    ap.add_argument("--from-dir", metavar="DIR", help="폴더의 wav 들로 등록")
    ap.add_argument("--score-noise", metavar="WAV", help="긴 소음 녹음으로 컷의 하한 정하기")
    ap.add_argument("--score-calls", metavar="DIR",
                    help="진짜 호출 wav 들로 컷의 상한 정하기(등록에 안 쓴 것으로)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="1단계 임계 — 설정의 wake.onnx.threshold 와 같게 둘 것")
    ap.add_argument("--append", action="store_true",
                    help="기존 본보기에 덧붙인다(화자를 추가할 때)")
    args = ap.parse_args()

    if not (args.record or args.from_dir or args.score_noise or args.score_calls):
        ap.print_help()
        return 2

    det = _load_detector(args)

    temps = []
    if args.from_dir:
        temps += build_from_dir(det, args.from_dir)
    if args.record:
        temps += record_takes(det, args.record)

    if temps:
        if args.append and os.path.exists(args.out):
            old = np.load(args.out)
            old = old[None, :] if old.ndim == 1 else old
            temps = list(old) + temps
            print(f"\n기존 {len(old)}개에 덧붙인다")
        save_templates(args.out, np.stack(temps))
        print(f"\n✅ 본보기 {len(temps)}개 저장: {args.out}")
        print("   설정에 넣을 것: wake.onnx.verify.embed_rescue.templates")

    if args.score_noise or args.score_calls:
        temps = temps or _load_templates(args.out)
        if args.score_noise:
            score_noise(det, temps, args.score_noise, args.threshold)
        if args.score_calls:
            if args.from_dir and (os.path.abspath(args.from_dir)
                                  == os.path.abspath(args.score_calls)):
                print("⚠️ 등록에 쓴 폴더를 그대로 채점한다 — 유사도가 1.0 근처로 나온다. "
                      "등록에 안 쓴 녹음으로 재야 의미가 있다.")
            score_calls(det, temps, args.score_calls, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
