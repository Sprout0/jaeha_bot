"""v4 데이터에서 **정속 원본만** 남기고 v5 격자로 증강만 다시 건다.

왜 재합성을 안 하나:
  정속 원본(TTS 산출물)은 이미 STT 검수를 통과한 것들이다. v4 -> v5 에서 바뀌는 건
  증강 격자뿐이므로 합성을 다시 할 이유가 없다. 22분이 아니라 0분이고, 무엇보다
  **원본이 같아야** 결과 차이를 증강 탓으로 좁힐 수 있다.

무엇이 원본인가:
  매니페스트에서 `src` 키가 없는 줄이 원본이다(합성 단계가 쓴 줄: text/voice/speed).
  `src` 가 있으면 증강본이다. 원본은 clip_000000 부터 연속이라 번호를 다시 매길 필요가 없다.

사용법 (젯슨에서):
    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/build_wake_v5.py --src output_v4 --dst output_v5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.gen_wake_supertonic import (EXPAND_GRID, EXPAND_PER_CLIP,  # noqa: E402
                                       expand)

DIRS = ("positive_train", "positive_test", "negative_train", "negative_test")
# v3·v4 와 같아야 하는 최종 개수. 학습 시간과 노트북 assert 가 여기에 걸려 있다.
EXPECT = {"positive_train": 7276, "positive_test": 1816,
          "negative_train": 8000, "negative_test": 2000}


def copy_originals(src_dir: str, dst_dir: str) -> int:
    """`src` 키가 없는 줄 = 정속 원본만 골라 복사하고 매니페스트를 새로 쓴다."""
    os.makedirs(dst_dir, exist_ok=True)
    man_in = os.path.join(src_dir, "manifest.jsonl")
    rows = [json.loads(x) for x in open(man_in, encoding="utf-8") if x.strip()]
    orig = [r for r in rows if "src" not in r]

    # 원본은 정속·정피치여야 한다. 아니면 이전 단계에서 뭔가 섞인 것이다.
    odd = [r for r in orig
           if abs(r.get("rate", 1.0) - 1.0) > 1e-6 or abs(r.get("pitch", 1.0) - 1.0) > 1e-6]
    if odd:
        raise SystemExit(f"🔴 {src_dir}: 원본인데 배속/피치가 걸린 줄 {len(odd)}개 — 확인 필요")

    with open(os.path.join(dst_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in orig:
            p = os.path.join(src_dir, r["clip"])
            if not os.path.exists(p):
                raise SystemExit(f"🔴 매니페스트에 있는데 파일이 없다: {p}")
            shutil.copy2(p, os.path.join(dst_dir, r["clip"]))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  원본 {len(orig)}개 복사 (전체 {len(rows)}줄 중 증강본 {len(rows)-len(orig)}개 버림)")
    return len(orig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="output_v4")
    ap.add_argument("--dst", default="output_v5")
    args = ap.parse_args()

    import soundfile as sf

    if os.path.exists(args.dst):
        raise SystemExit(f"🔴 {args.dst} 가 이미 있다. 지우거나 다른 이름을 쓸 것 "
                         "(덮어쓰면 어떤 격자로 만든 데이터인지 알 수 없게 된다)")

    print(f"격자 {len(EXPAND_GRID)}칸 × 클립당 {EXPAND_PER_CLIP}개")
    for rate, pitch in EXPAND_GRID:
        print(f"    시간 {rate:<5} 피치 {pitch}")
    print()

    bad = False
    for d in DIRS:
        src, dst = os.path.join(args.src, d), os.path.join(args.dst, d)
        print(f"[{d}]")
        n = copy_originals(src, dst)
        if expand(dst, sf) != 0:
            bad = True
        total = n + n * EXPAND_PER_CLIP
        want = EXPECT[d]
        mark = "OK " if total == want else "🔴"
        print(f"  {mark} 합계 {total} (기대 {want})\n")
        bad = bad or total != want

    if bad:
        print("🔴 문제가 있다. 이대로 학습하지 말 것.")
        return 1
    print(f"완료 -> {args.dst}")
    print(f"다음: python tools/sanity_wake_data.py --dir {args.dst} 로 점검한 뒤"
          " tar 로 묶어 Drive 에 올린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
