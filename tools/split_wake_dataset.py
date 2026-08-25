"""긍정·부정을 train/test 로 나눠 학습용 4개 폴더를 만든다.

🔴 **화자를 통째로 빼서 나눈다. 무작위로 자르지 않는다.**

  v1~v5 는 무작위로 잘랐고, 그래서 시험지와 교과서가 같은 곳에서 나왔다 — 같은 생성기,
  같은 화자, 같은 문구. 그 결과 오프라인 재현율 91% 가 실기 30% 였다. 석 달간
  '호출어를 알아듣는가'가 아니라 **'합성 음성을 외웠는가'** 를 재고 있었던 것이다.

  화자를 빼면 오프라인 점수가 **'처음 듣는 목소리에도 되는가'** 를 재게 된다.
  여전히 합성음이라 실기를 대신하지 못한다 — 그러나 최소한 자기채점은 아니다.

🔴 **증강본은 원본과 반드시 같은 쪽에 있어야 한다.**
  clip_000005 가 train 에 있는데 그걸 피치만 바꾼 클립이 test 에 있으면, 그건 시험지에
  교과서를 그대로 베껴 넣은 것이다. 증강 매니페스트 줄에는 voice 가 없고 `src` 만 있으므로
  **src 를 따라 원본까지 거슬러 올라가** 화자를 확인한다.

시험용 화자 고르기(실측 F0 기준):
  M3 122Hz      — 저역. 아버님(104~116Hz)에게 가장 가깝다. v4 가 무너진 자리라 여기를 시험한다.
                  학습에는 더 낮은 M5(97) · M2+M5(99) 가 남아 아래에서 감싼다.
  F2+F5 258Hz   — 고역. 재하 음역대(미측정, 추정 250~400Hz)를 시험한다.

사용:
  python tools/split_wake_dataset.py --positive output/positive_v6 \
      --negative output/negative_v6 --out output/wake_v6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

HOLDOUT_VOICES = ("M3", "F2+F5")


def load(dirpath: str) -> tuple[list[dict], dict]:
    rows = [json.loads(x) for x in
            open(os.path.join(dirpath, "manifest.jsonl"), encoding="utf-8") if x.strip()]
    return rows, {r["clip"]: r for r in rows}


def origin_of(row: dict, by_clip: dict) -> dict:
    """증강본이면 src 를 따라 원본까지 간다. 원본이면 자기 자신."""
    seen = set()
    cur = row
    while "src" in cur:
        if cur["src"] in seen:
            raise SystemExit(f"🔴 src 순환: {cur['clip']}")
        seen.add(cur["src"])
        nxt = by_clip.get(cur["src"])
        if nxt is None:
            raise SystemExit(f"🔴 src 를 못 찾음: {cur['clip']} -> {cur['src']}")
        cur = nxt
    return cur


def split_one(src_dir: str, out_root: str, name: str, holdout: set,
              max_dur: float = 0.0) -> dict:
    rows, by_clip = load(src_dir)
    buckets: dict[str, list[tuple[dict, dict]]] = {"train": [], "test": []}

    # 🔴 길이로 뺄 때는 **계보째** 빼야 한다. 원본만 빼고 그 증강본을 남기면 증강본의
    #    src 가 없는 클립을 가리켜 매니페스트가 깨진다(실제로 KeyError 로 터졌다).
    #    반대로 증강본만 긴 경우는 그것만 빼도 된다 — 원본이 남아 있으니 매달릴 곳이 있다.
    too_long = {r["clip"] for r in rows if max_dur and (r.get("dur") or 0) > max_dur}
    dead_origins = {c for c in too_long if "src" not in by_clip[c]}
    dropped = 0
    for r in rows:
        if r["clip"] in too_long or origin_of(r, by_clip)["clip"] in dead_origins:
            dropped += 1
            continue
        og = origin_of(r, by_clip)
        side = "test" if og.get("voice") in holdout else "train"
        buckets[side].append((r, og))
    if dropped:
        print(f"  ({name}: {max_dur}s 초과 {len(too_long)}개 + 그 증강본 "
              f"{dropped - len(too_long)}개 = {dropped}개 제외)")

    counts = {}
    for side, items in buckets.items():
        d = os.path.join(out_root, f"{name}_{side}")
        os.makedirs(d, exist_ok=True)

        # 🔴 원본을 먼저, 증강본을 나중에 번호 매긴다. 두 가지 이유가 있다:
        #   ① build_wake_v5 가 "원본은 clip_000000 부터 연속"을 가정한다.
        #   ② 증강본 줄에 `src`(원본의 **새 이름**)를 넣으려면 원본 이름이 먼저 정해져야 한다.
        # ⚠️ `src` 키가 없으면 sanity_wake_data 가 증강본을 하나도 못 알아본다. 실제로 한 번
        #    그랬고, '긍정·부정이 같은 0칸을 받았다'에 OK 가 붙었다 — **빈 검사가 통과했다.**
        origs = [(r, og) for r, og in items if "src" not in r]
        augs = [(r, og) for r, og in items if "src" in r]
        newname = {}
        ordered = []
        for i, (r, og) in enumerate(origs + augs):
            new = f"clip_{i:06d}.wav"
            newname[r["clip"]] = new
            ordered.append((new, r, og))

        with open(os.path.join(d, "manifest.jsonl"), "w", encoding="utf-8") as f:
            for new, r, og in ordered:
                shutil.copy2(os.path.join(src_dir, r["clip"]), os.path.join(d, new))
                # 계보를 남긴다 — '어느 화자·문구·증강 칸에서 왔나'를 못 되짚으면
                # 칸별 채점(9-4)도 실패 분석도 못 한다.
                row = {"clip": new, "voice": og.get("voice"), "text": og.get("text"),
                       "speed": og.get("speed"), "dur": r.get("dur")}
                if "src" in r:
                    row["src"] = newname[r["src"]]      # 증강본 표시 + 원본 추적
                    row["rate"] = r.get("rate", 1.0)
                    row["pitch"] = r.get("pitch", 1.0)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[side] = len(items)
        print(f"  {name}_{side:5} {len(items):>6}개  (원본 {len(origs)} + 증강 {len(augs)})")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="화자 홀드아웃으로 train/test 분할")
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default=",".join(HOLDOUT_VOICES),
                    help="시험에만 쓸 화자(쉼표 구분)")
    ap.add_argument("--max-dur", type=float, default=2.0,
                    help="이 길이를 넘는 클립은 뺀다(초). 분류기 수용영역은 2.42초지만 "
                         "파이프라인이 2.0초를 기준으로 본다. 기본값에서 부정 2개가 걸린다.")
    args = ap.parse_args()

    holdout = {v.strip() for v in args.holdout.split(",") if v.strip()}
    print(f"시험 전용 화자: {sorted(holdout)}\n")

    pos = split_one(args.positive, args.out, "positive", holdout, args.max_dur)
    neg = split_one(args.negative, args.out, "negative", holdout, args.max_dur)

    print()
    for name, c in (("긍정", pos), ("부정", neg)):
        tot = c["train"] + c["test"]
        print(f"{name}: train {c['train']} / test {c['test']} "
              f"(시험 비중 {100 * c['test'] / tot:.1f}%)")

    # 누수 검사 — 같은 원본에서 나온 클립이 train 과 test 에 갈라져 있으면 안 된다.
    print("\n[누수 검사] 같은 원본이 양쪽에 갈라졌나")
    for name in ("positive", "negative"):
        sides = {}
        for side in ("train", "test"):
            p = os.path.join(args.out, f"{name}_{side}", "manifest.jsonl")
            sides[side] = {(json.loads(x)["voice"], json.loads(x)["text"])
                           for x in open(p, encoding="utf-8") if x.strip()}
        both = {v for v, _ in sides["train"]} & {v for v, _ in sides["test"]}
        print(f"  {name}: 양쪽에 걸친 화자 {sorted(both) if both else '없음 ✅'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
