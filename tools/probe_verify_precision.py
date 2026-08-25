"""2단계 검증의 **정밀도**를 잰다 — 호출어가 아닌 말이 통과하는가.

오늘 잰 것은 재현율뿐이다(진짜 호출어 36개 중 36개 통과, 자모거리 중앙 0.000).
그것만으로는 컷을 정할 수 없다. **엉뚱한 말이 얼마나 새는지**를 같이 봐야 한다.
이 값이 낮으면 1단계를 훨씬 낮은 임계값으로 열어도 된다 — 캐스케이드 설계의 핵심 근거다.

무엇을 넣나:
  ① 아이가 실제로 할 법한 말   ② 부모가 할 법한 말
  ③ 🔴 **음운적으로 가까운 함정** — 하이/티드/하이킹/타이드/하이라이트/다이어트…
     이게 진짜 시험이다. 무관한 문장은 어차피 안 걸린다.

한계: 합성음이다. TV·유튜브처럼 여러 사람이 겹쳐 말하는 실제 소음은 여기 없다.
      실측(2026-08-12, 재하봇)에서는 그쪽이 훨씬 어려웠다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models          # noqa: E402
from app.stt_module import STTModule        # noqa: E402
from app.tts_module import TTSModule        # noqa: E402
from app.wake import best_wake_ratio        # noqa: E402

TARGET_SR = 16000
VOICES = ["M2+M5", "M5", "M3", "M2", "F5", "M1", "F3", "F2", "F1", "F2+F5"]

# ① 아이 말 (2세)
CHILD = ["엄마 어디 있어", "이거 뭐야", "물 줘", "안 해", "더 줘", "무서워",
         "빵빵 자동차", "멍멍이다", "아빠 안아줘", "쉬 마려워", "같이 놀자", "싫어 싫어"]
# ② 부모 말
PARENT = ["밥 먹자 이리 와", "이거 정리하고 자자", "손 씻고 오세요", "오늘 어린이집 재밌었어",
          "티비 그만 보고", "신발 신자", "잠깐만 기다려", "물 마실래"]
# ③ 🔴 음운적으로 가까운 함정 — 진짜 시험은 여기다
TRAPS = ["하이", "하이요", "안녕 하이", "하이 하이", "티드", "티드가",
         "하이킹 갈까", "하이라이트 봤어", "타이드 세제", "다이어트 해야지",
         "하이브리드 차", "티비", "하이텐션", "네 하이", "그 다음에 티",
         "하이든 음악", "차이 나는", "하이파이브 하자", "티도 안 나", "하이고"]


def main() -> int:
    ap = argparse.ArgumentParser(description="2단계 검증 정밀도(오통과율) 측정")
    ap.add_argument("--word", default="하이티드")
    ap.add_argument("--cut", type=float, default=0.35,
                    help="운영 plain_max_ratio (configs/model_paths.yaml)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--seed-base", type=int, default=77000)
    args = ap.parse_args()

    models = load_models()
    tcfg, scfg = models.get("tts", {}), models.get("stt", {})
    tts = TTSModule(model=tcfg.get("model", "supertonic-3"), voice=VOICES[0],
                    language="ko", total_steps=12,
                    threads=int(tcfg.get("threads", 4)),
                    providers=["CPUExecutionProvider"])
    tts.load()
    src_sr = tts.sample_rate
    stt = STTModule(model_size=scfg.get("model_size", "medium"), device=args.device,
                    compute_type=args.compute_type, language="ko",
                    keywords=[], aliases={})
    stt.load()

    groups = [("아이 말", CHILD), ("부모 말", PARENT), ("가까운 함정", TRAPS)]
    out = {}
    i = 0
    for name, phrases in groups:
        rows = []
        for phrase in phrases:
            voice = VOICES[i % len(VOICES)]
            tts._style = tts._resolve_style(voice)
            tts.speed = [0.95, 1.05, 1.15][i % 3]
            tts.seed = args.seed_base + i
            i += 1
            a = tts._trim(tts._infer(phrase), keep_tail=0.15)
            a = tts._resample(a, src_sr, TARGET_SR)
            peak = float(np.abs(a).max()) if a.size else 0.0
            if peak > 0:
                a = (a / peak * 0.95).astype(np.float32)
            txt, _ = stt.transcribe(a, initial_prompt=None)
            r = best_wake_ratio(txt or "", args.word)
            rows.append((phrase, r, txt or ""))
            flag = "🔴 통과" if r <= args.cut else ""
            print(f"  [{name}] {phrase:16} {r:.3f}  '{(txt or '없음')[:22]}' {flag}")
        out[name] = rows

    print("\n" + "=" * 70)
    print(f"2단계 오통과율 — 호출어 '{args.word}', 컷 {args.cut}")
    print(f"{'묶음':12} {'개수':>5} {'오통과':>7} {'최소거리':>9}   샌 것")
    print("-" * 70)
    tot = bad = 0
    for name, rows in out.items():
        rr = np.array([r for _, r, _ in rows])
        leaked = [p for p, r, _ in rows if r <= args.cut]
        tot += len(rr)
        bad += len(leaked)
        print(f"{name:12} {len(rr):>5} {len(leaked):>7} {rr.min():>9.3f}   "
              f"{', '.join(leaked[:3])}")
    print("-" * 70)
    print(f"{'합계':12} {tot:>5} {bad:>7}  ({100 * bad / tot:.1f}%)")
    print("=" * 70)
    print("⚠️ 합성음이다. TV처럼 여러 사람이 겹쳐 말하는 실제 소음은 여기 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
