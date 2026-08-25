"""QC 를 통과하고 **살아남은** 클립을 합성 격자 칸별로 센다 — expand 직전 점검.

왜 따로 있나:
  sanity_wake_data.py 는 **expand 이후**의 칸(시간배속 x 피치)을 본다. 그런데 수율은
  expand 전 단계인 **합성 속도**에서 갈린다 — x1.15 는 통과율이 67~70% 라 한 번
  돌려서는 다른 칸만큼 안 찬다(gen_wake_supertonic.SPEEDS 주석 참고).
  칸이 덜 찬 채로 증강하면 그 불균형이 **곱해져서** 넘어간다.

왜 manifest 와 실제 파일을 교집합하나:
  wake_qc --move-rejects 는 탈락분을 rejects/ 로 옮기지만 **manifest 는 안 고친다.**
  그래서 "manifest 에 있고 + 폴더에 아직 있는" 것이 생존분이다.

사용:
  python tools/wake_cell_balance.py output/positive_v6
  python tools/wake_cell_balance.py output/positive_v6 --target 475   # 목표를 정해 부족분 계산
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def load_manifest(dirpath: str) -> list[dict]:
    p = os.path.join(dirpath, "manifest.jsonl")
    if not os.path.exists(p):
        raise SystemExit(f"manifest 가 없다: {p}")
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="합성 격자 칸별 생존 개수")
    ap.add_argument("directory")
    ap.add_argument("--target", type=int, default=0,
                    help="칸당 목표 개수. 0 이면 가장 많이 찬 칸을 목표로 본다.")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="이 비율 안쪽의 차이는 보충하지 않는다(기본 5%%). 생성은 시드 "
                         "변주라 칸마다 몇 개씩 어긋나는 게 정상인데, 그걸 부족분으로 "
                         "잡으면 2~6개짜리 무의미한 보충 계획이 나온다.")
    args = ap.parse_args()

    rows = load_manifest(args.directory)
    alive = {f for f in os.listdir(args.directory) if f.endswith(".wav")}

    by_speed: dict = collections.defaultdict(lambda: [0, 0])          # [생존, 전체]
    by_cell: dict = collections.defaultdict(lambda: [0, 0])           # (speed, text)
    for r in rows:
        ok = r["clip"] in alive
        for key, d in ((r["speed"], by_speed), ((r["speed"], r["text"]), by_cell)):
            d[key][1] += 1
            d[key][0] += 1 if ok else 0

    total_alive = sum(v[0] for v in by_speed.values())
    print(f"클립 {len(rows)}개 중 생존 {total_alive}개 ({100 * total_alive / len(rows):.1f}%)\n")

    print("[속도별]")
    print(f"{'속도':>6} {'생존':>7} {'전체':>7} {'수율':>7}   {'부족분':>7}")
    print("-" * 46)
    target = args.target or max(v[0] for v in by_speed.values())
    plan = {}
    for speed in sorted(by_speed):
        got, tot = by_speed[speed]
        yield_ = got / tot if tot else 0.0
        short = max(0, target - got)
        if short <= target * args.tolerance:   # 잔차는 보충하지 않는다
            short = 0
        # 부족분을 그 칸의 **실측 수율로 나눠야** 생성 개수가 나온다.
        need = int(round(short / yield_)) if short and yield_ > 0 else 0
        plan[speed] = (short, need, yield_)
        print(f"{speed:>6.2f} {got:>7} {tot:>7} {100 * yield_:>6.0f}%   {short:>7}")
    print("-" * 46)
    print(f"목표 = 칸당 {target}개"
          f"{' (가장 많이 찬 칸)' if not args.target else ''}"
          f" | 허용오차 {100 * args.tolerance:.0f}% 안쪽은 무시\n")

    print("[속도 x 문구]")
    for speed in sorted(by_speed):
        cells = [(t, by_cell[(speed, t)]) for (s, t) in by_cell if s == speed]
        line = "  x%.2f  " % speed
        line += "  ".join(f"{t} {g}/{n}" for t, (g, n) in sorted(cells))
        print(line)

    print("\n[보충 생성 계획]")
    any_need = False
    for speed in sorted(plan):
        short, need, yield_ = plan[speed]
        if not short:
            continue
        any_need = True
        print(f"  x{speed:.2f}: {short}개 부족, 수율 {100 * yield_:.0f}% "
              f"-> **{need}개 생성** (= {short} / {yield_:.2f})")
        print(f"      python tools/gen_wake_supertonic.py --out {args.directory} "
              f"--speeds {speed} -n {need} \\\n"
              f"          --start-index {len(rows)} --seed-base 50000 --total-steps 12")
        print("      ⚠️ --seed-base 를 반드시 바꿀 것 — 탈락 원인이 시드 편차다.")
    if not any_need:
        print("  없음 — 칸이 고르다. --expand 로 진행해도 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
