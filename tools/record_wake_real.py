"""실제 목소리로 호출어를 녹음하고, 배포된 감지기가 왜 못 잡는지 가른다.

왜 필요한가 (2026-08-10):
  v4 실기에서 "빠르게·흘려서 말하면 안 된다"가 남았다. 원인 후보가 둘인데
  둘의 처방이 정반대다:
    (가) 내 목소리가 학습 음역 밖이다      -> 피치 칸을 넓히면 된다(합성으로 해결)
    (나) 흘려 말하면 음소가 빠진다          -> 축약형 문구·실음성이 필요하다(합성으로 안 됨)
  추측으로 다음 데이터를 만들면 또 헛돈다. **같은 녹음 하나로 둘을 가른다.**

가르는 방법 — 실패한 녹음을 피치만 올려 다시 채점한다:
  점수가 뛰면  -> (가) 음역 문제. 낮은 피치 칸을 넣고 재학습하면 된다.
  그대로면     -> (나) 발음 문제. 피치로는 못 고친다.
whisper 전사는 대조군이다. whisper 가 '재하봇'으로 읽는데 감지기 점수가 낮으면
**음성은 멀쩡하고 모델만 못 듣는 것**이다(v3 때 이 방법으로 원인을 잡았다).

녹음물은 버리지 않고 data/wake_real/ 에 쌓는다 — 실음성 재학습의 씨앗이다.

사용법 (젯슨에서, 봇을 끄고):
    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/record_wake_real.py            # 조건당 5회
    python tools/record_wake_real.py --n 10     # 조건당 10회
⚠️ ReSpeaker 는 장치가 하나뿐이라 봇이 돌고 있으면 마이크를 못 연다. 먼저 끌 것.
   (그 상태로 돌리면 젯슨 기본 입력 35번이 잡히는데, 이 장치는 **에러 없이 무음**을 준다.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models  # noqa: E402
from app.wake_onnx import OnnxWakeDetector  # noqa: E402
from tools.gen_wake_supertonic import pitch_shift  # noqa: E402

SR = 16000
FRAME = 1280          # 80ms — 런타임과 같은 단위
TAKE_S = 3.0          # 한 번 녹음 길이
# 감지기는 25프레임(2.0s)을 채워야 첫 점수를 낸다. 마이크는 끊기지 않고 흐르지만
# 잘라낸 파일은 그렇지 않으므로, 채점할 때 앞뒤에 무음을 덧대 같은 조건으로 만든다.
PAD_S = 2.5

CONDITIONS = [
    ("또박또박", "또박또박 천천히  ―  '재 하 봇'"),
    ("보통", "평소처럼  ―  '재하봇'"),
    ("빠르게", "빠르게, 그래도 다 발음해서  ―  '재하봇!'"),
    ("흘려서", "대충 흘려서, 실제로 부를 때처럼  ―  '잰봇' 처럼 돼도 그대로"),
]


# 무음 판정 기준. 젯슨 실측(2026-08-12): 조용한 방의 소음 바닥이 RMS 0.0035 정도고,
# 캡처가 아예 안 되면 **정확히 0.0** 이 나온다. 그 사이에 선을 긋는다.
SILENT_RMS = 2e-4


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))) if y.size else 0.0


def pick_device() -> int | None:
    """설정의 audio.device(이름 부분일치)에 해당하는 **입력 장치 번호**를 돌려준다.

    🔴 sd.default.device 에 넣고 끝내면 안 된다. 젯슨의 시스템 기본 입력은 35번인데
       이 장치는 **에러 없이 0.0 만 준다**(실측 2026-08-12). 전역 기본값은 다른 코드가
       바꿀 수도 있으므로, 번호를 직접 들고 다니며 sd.rec 에 매번 넘긴다.
    """
    name = (load_models().get("audio") or {}).get("device")
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            print(f"입력 장치: [{i}] {d['name']}")
            return i
    print(f"⚠️ '{name}' 을(를) 못 찾음 — 시스템 기본 마이크를 쓴다(USB 연결 확인)")
    return None


def check_mic(device: int | None) -> None:
    """녹음을 시작하기 전에 마이크가 실제로 소리를 잡는지 1초로 확인한다.

    🔴 이 관문이 없어서 무음 20개를 받아 놓고 '발음 문제'라는 엉뚱한 결론까지 냈다
       (2026-08-12). 못 잡으면 여기서 멈추는 게 20번 부르게 하는 것보다 낫다.
    """
    print("마이크 확인 중(1초, 아무 말 안 해도 됨)...")
    try:
        y = sd.rec(SR, samplerate=SR, channels=1, dtype="float32",
                   device=device, blocking=True).reshape(-1)
    except Exception as e:
        raise SystemExit(
            f"🔴 마이크를 열 수 없다: {type(e).__name__}: {e}\n"
            "   봇이 돌고 있으면 끄세요 — ReSpeaker 는 장치가 하나라 동시에 못 엽니다.")
    r = rms(y)
    print(f"  주변 소음 RMS {r:.5f}")
    if r < SILENT_RMS:
        raise SystemExit(
            "🔴 마이크가 소리를 전혀 잡지 못한다(무음). 녹음해도 의미가 없어 여기서 멈춘다.\n"
            "   1) 봇이 돌고 있으면 끄세요 (Ctrl+C) — 마이크는 하나뿐입니다.\n"
            "   2) ReSpeaker USB 연결 확인:  arecord -l\n"
            "   3) 음소거/게인 확인:  alsamixer -c 0  (F4 로 Capture 탭)")


def record_take(device: int | None) -> np.ndarray:
    for c in (3, 2, 1):
        print(f"\r  {c}...", end="", flush=True)
        time.sleep(0.7)
    print("\r  🔴 말하세요!   ", end="", flush=True)
    buf = sd.rec(int(TAKE_S * SR), samplerate=SR, channels=1, dtype="float32",
                 device=device)
    sd.wait()
    print("\r  (끝)          ")
    return buf.reshape(-1)


def score(det: OnnxWakeDetector, y: np.ndarray) -> float:
    pad = np.zeros(int(PAD_S * SR), dtype=np.float32)
    y = np.concatenate([pad, y.astype(np.float32), pad])
    det.reset()
    best = 0.0
    for i in range(0, len(y) - FRAME, FRAME):
        s = det.push(y[i:i + FRAME])
        if s is not None:
            best = max(best, s)
    return best


def f0_median(y: np.ndarray, sr: int = SR) -> float:
    """유성 구간의 기본주파수 중앙값(Hz). 자기상관 — numpy 만 쓴다.

    성인 남성 85~180 / 여성 165~255 / 유아 250~400 대역이 대략의 기준이다.
    """
    n, hop = 1024, 256
    lo, hi = sr // 400, sr // 70          # 70~400Hz 만 본다
    out = []
    for i in range(0, max(0, len(y) - n), hop):
        seg = y[i:i + n].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:   # 무음 건너뜀
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        peak = int(np.argmax(ac[lo:hi])) + lo
        if ac[peak] / ac[0] > 0.3:              # 충분히 주기적일 때만 유성음으로 인정
            out.append(sr / peak)
    return float(np.median(out)) if out else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="조건당 녹음 횟수")
    ap.add_argument("--outdir", default="data/wake_real")
    ap.add_argument("--no-stt", action="store_true", help="whisper 전사 생략(빠름)")
    args = ap.parse_args()

    cfg = load_models()["wake"]["onnx"]
    thr = float(cfg["threshold"])
    det = OnnxWakeDetector(model_dir=cfg["model_dir"], classifier=cfg["classifier"],
                           threshold=thr, trigger_frames=int(cfg.get("trigger_frames", 1)))
    print(f"감지기: {cfg['model_dir']}/{cfg['classifier']}  임계 {thr}")
    device = pick_device()
    check_mic(device)

    day = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = os.path.join(args.outdir, day)
    os.makedirs(outdir, exist_ok=True)

    takes = []
    for label, guide in CONDITIONS:
        print(f"\n{'='*60}\n[{label}]  {guide}\n{'='*60}")
        for k in range(args.n):
            print(f"({k+1}/{args.n})", end="")
            y = record_take(device)
            path = os.path.join(outdir, f"{label}_{k:02d}.wav")
            sf.write(path, y, SR)
            r = rms(y)
            if r < SILENT_RMS:
                # 중간에 마이크를 빼앗기는 경우가 있다. 조용히 넘어가면 그 표본이
                # '못 알아들은 발화'로 둔갑해 결론을 뒤집는다.
                print(f"      🔴 무음(RMS {r:.5f}) — 이 회차는 버린다")
                takes.append({"cond": label, "path": path, "score": 0.0,
                              "rms": r, "silent": True})
                continue
            s = score(det, y)
            mark = "○" if s >= thr else "✗"
            print(f"      {mark} 점수 {s:.3f}  (RMS {r:.4f})")
            takes.append({"cond": label, "path": path, "score": s,
                          "rms": r, "silent": False})

    live = [t for t in takes if not t["silent"]]
    if not live:
        print("\n🔴 전부 무음이라 판정할 수 없다. 봇을 끄고 마이크를 확인한 뒤 다시 하세요.")
        return 1
    if len(live) < len(takes):
        print(f"\n⚠️ {len(takes)-len(live)}회가 무음이라 표에서 뺐다.")
    takes = live

    # ── 전사(대조군): whisper 가 읽으면 음성은 멀쩡하다는 뜻 ──────────────
    if not args.no_stt:
        print("\nwhisper 로 전사 중(모델 로드에 몇 초)...")
        try:
            from app.stt_module import STTModule
            stt = STTModule(**load_models().get("stt", {}))
            stt.load()
            for t in takes:
                y, _ = sf.read(t["path"], dtype="float32")
                text, _sec = stt.transcribe(y)      # (텍스트, 처리시간) 을 돌려준다
                t["text"] = (text or "").strip()
        except Exception as e:                       # 전사는 보조 정보다 — 실패해도 계속
            print(f"⚠️ 전사 실패(무시): {e}")

    # ── 피치를 올려 재채점: 음역 문제인가 발음 문제인가 ────────────────
    print("피치를 올려 재채점 중...")
    for t in takes:
        y, _ = sf.read(t["path"], dtype="float32")
        t["f0"] = f0_median(y)
        t["up15"] = score(det, pitch_shift(y, 1.15))
        t["up35"] = score(det, pitch_shift(y, 1.35))

    # ── 결과 ────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"{'조건':<8}{'n':>3}{'통과':>6}{'점수중앙':>9}{'F0':>7}"
          f"{'피치+15%':>9}{'피치+35%':>9}   전사(다수)")
    print("-" * 74)
    for label, _ in CONDITIONS:
        g = [t for t in takes if t["cond"] == label]
        if not g:
            continue
        ss = sorted(t["score"] for t in g)
        txts = [t.get("text", "") for t in g if t.get("text")]
        top = max(set(txts), key=txts.count)[:14] if txts else "-"
        print(f"{label:<8}{len(g):>3}{sum(1 for t in g if t['score'] >= thr):>5}회"
              f"{ss[len(ss)//2]:>9.3f}{np.median([t['f0'] for t in g]):>6.0f}Hz"
              f"{np.median([t['up15'] for t in g]):>9.3f}"
              f"{np.median([t['up35'] for t in g]):>9.3f}   {top}")

    fails = [t for t in takes if t["score"] < thr]
    print("-" * 74)
    # 🔴 판정 전에 표본이 판정할 만한 것인지 먼저 본다. 2026-08-12 에 무음 20개를 놓고
    #    '발음 문제'라는 결론을 내는 사고가 났다 — 결론은 늘 '왜 그렇게 볼 수 있는지'가
    #    성립할 때만 낸다.
    voiced = [t for t in takes if t["f0"] > 0]
    if len(voiced) < len(takes) * 0.5:
        print(f"🔴 {len(takes)-len(voiced)}/{len(takes)} 건에서 사람 목소리(F0)가 안 잡힌다.")
        print("   녹음이 제대로 안 됐다는 뜻이라 판정하지 않는다. 마이크부터 확인할 것.")
    elif not fails:
        print("➡️ 전부 통과했다. 실패 사례를 더 모아야 한다(더 빠르게·더 흘려서).")
    else:
        gain = np.median([max(t["up15"], t["up35"]) - t["score"] for t in fails])
        print(f"실패 {len(fails)}건의 피치 상승 이득 중앙값: {gain:+.3f}")
        if gain > 0.15:
            print("➡️ **음역 문제**. 낮은 피치 칸(0.85·0.75)을 격자에 넣고 재학습하면 된다.")
        else:
            print("➡️ **발음 문제**. 피치로는 안 고쳐진다. 축약형 문구 추가 + 실음성 재학습이 답이다.")
        if not any(t.get("text") for t in fails):
            print("   ⚠️ 단, 실패 건의 전사가 전부 비어 있다. whisper 도 못 읽는 소리라면"
                  " 모델 탓이 아니라 녹음 탓일 수 있으니 wav 를 직접 들어볼 것.")

    meta = os.path.join(outdir, "takes.jsonl")
    with open(meta, "w", encoding="utf-8") as f:
        for t in takes:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n녹음 {len(takes)}개 저장: {outdir}  (실음성 재학습에 그대로 쓴다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
