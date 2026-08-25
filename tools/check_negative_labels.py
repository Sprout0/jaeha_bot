"""부정 클립에 **호출어가 섞여 있지 않은지** 검사한다 — 라벨 오류 감지.

왜 필요한가:
  부정 문구에는 '하이 준비' '하이브리드' 처럼 호출어와 가까운 것이 일부러 들어 있다.
  TTS 가 이걸 뭉개면 실제로 '하이티드'로 들리는 클립이 나올 수 있고, 그건 **부정 라벨이
  붙은 긍정**이다. 모델에겐 "이 소리는 호출어가 아니다"라고 가르치는 셈이라, 진짜 호출을
  놓치는 쪽으로 학습이 밀린다. 오프라인 지표로는 거의 안 보이는 종류의 오염이다.

왜 전수를 안 하나:
  2,500개 x whisper(CPU) = 3시간. 그런데 무관한 문장('밥 먹자')은 어차피 안 걸린다.
  **음운적으로 가까운 문구의 클립만** 골라 본다 — 위험이 거기에만 있다.

사용:
  python tools/check_negative_labels.py output/negative_v6 -n 250
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models          # noqa: E402
from app.stt_module import STTModule        # noqa: E402
from app.wake import best_wake_ratio        # noqa: E402

# 위험 문구를 고르는 기준: 호출어와 음절을 공유하는가.
RISKY_TOKENS = ("하이", "티드", "티", "타이", "다이", "하이드")


def is_risky(text: str, word: str) -> bool:
    t = text.replace(" ", "")
    if any(tok in t for tok in RISKY_TOKENS):
        return True
    return best_wake_ratio(text, word) <= 0.6      # 표기상 이미 가까운 것


def main() -> int:
    ap = argparse.ArgumentParser(description="부정 클립의 라벨 오류 검사")
    ap.add_argument("directory")
    ap.add_argument("--word", default="하이티드")
    ap.add_argument("-n", "--count", type=int, default=250)
    ap.add_argument("--cut", type=float, default=0.35,
                    help="운영 plain_max_ratio. 이보다 가까우면 라벨 오류 의심")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()

    rows = [json.loads(x) for x in
            open(os.path.join(args.directory, "manifest.jsonl"), encoding="utf-8") if x.strip()]
    risky = [r for r in rows if is_risky(r.get("text", ""), args.word)]
    print(f"전체 {len(rows)}개 중 위험 문구 클립 {len(risky)}개 "
          f"({100 * len(risky) / len(rows):.0f}%) — 여기서 {args.count}개 표본\n")

    scfg = load_models().get("stt", {})
    stt = STTModule(model_size=scfg.get("model_size", "medium"), device=args.device,
                    compute_type=args.compute_type, language="ko", keywords=[], aliases={})
    stt.load()

    pick = random.Random(19).sample(risky, min(args.count, len(risky)))
    # 🔴 두 가지를 반드시 구분한다. 안 하면 겁나는 숫자만 나오고 판단이 안 선다:
    #   라벨 오류  — TTS 가 문구를 뭉개서 **실제로 호출어처럼 들린다**. 빼야 한다.
    #   hard negative — TTS 는 제대로 말했는데 **그 단어 자체가 가깝다**(하이드·하이브리드).
    #                   빼면 안 된다. 모델에게 가장 값진 자료다.
    # 가르는 기준: 전사가 의도한 문구와 일치하는가.
    errors, hard, dists = [], [], []
    for i, r in enumerate(pick, 1):
        y, sr = sf.read(os.path.join(args.directory, r["clip"]))
        if y.ndim > 1:
            y = y.mean(axis=1)
        txt, _ = stt.transcribe(y.astype(np.float32), initial_prompt=None)
        d = best_wake_ratio(txt or "", args.word)
        dists.append(d)
        if d <= args.cut:
            said_it_right = best_wake_ratio(txt or "", r.get("text", "").replace(" ", "")) <= 0.2
            (hard if said_it_right else errors).append(
                (r["clip"], r.get("text", ""), r.get("speed"), d, txt or ""))
        if i % 50 == 0:
            print(f"     ... {i}/{len(pick)}")

    a = np.array(dists)
    print("\n" + "=" * 68)
    print(f"표본 {len(a)}개  |  🔴 라벨 오류 {len(errors)}개  |  "
          f"hard negative {len(hard)}개")
    print(f"자모거리 최소 {a.min():.3f} / 5%tile {np.percentile(a, 5):.3f} "
          f"/ 중앙 {np.median(a):.3f}  (컷 {args.cut})")
    if errors:
        print("\n🔴 라벨 오류 — TTS 가 뭉개서 호출어처럼 들린다. **부정에서 뺄 것**:")
        for c, t, sp, d, got in errors:
            print(f"   {c}  '{t}' x{sp}  ->  '{got}'  {d:.3f}")
    else:
        print("\n✅ 라벨 오류 없음 — TTS 가 뭉개서 호출어가 된 클립은 없다.")
    if hard:
        seen = sorted({t for _, t, _, _, _ in hard})
        print(f"\n💪 hard negative {len(hard)}개 — 그대로 둔다(가장 값진 자료다):")
        print(f"   문구: {', '.join(seen)}")
        print("   TTS 는 제대로 말했고 그 단어 자체가 호출어와 가깝다.")
        print("   ⚠️ 다만 이 말들은 2단계 whisper 도 못 거른다(컷 안에 든다).")
        print("      1단계가 걸러 줘야 하고, 그래서 학습셋에 있어야 한다.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
