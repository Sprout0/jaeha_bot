"""QC 를 통과한 보충분을 본 폴더에 합친다 — 이름 충돌 없이, manifest 를 맞춰서.

왜 도구가 필요한가:
  `wake_qc --move-rejects` 는 생존분을 **clip_000000 부터 재번호**한다. 본 폴더에도
  clip_000000 이 이미 있으므로 **그냥 복사하면 덮어쓴다.** 소리 없이 데이터를 잃는다.
  그래서 받는 쪽의 max(idx)+1 부터 새 번호를 붙이고 manifest 행도 같이 고친다.

왜 보충분을 따로 만들었나:
  같은 폴더에 넣고 QC 를 다시 돌리면 이미 검수한 것까지 전부 재전사한다
  (2,000개 = CPU 2.4시간). 보충분만 따로 검수하고 합치는 게 맞다.

사용:
  python tools/merge_wake_clips.py --src output/topup_115 --dst output/positive_v6
  python tools/merge_wake_clips.py --src ... --dst ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

IDX = re.compile(r"clip_(\d+)\.wav$")


def read_manifest(d: str) -> list[dict]:
    p = os.path.join(d, "manifest.jsonl")
    if not os.path.exists(p):
        raise SystemExit(f"manifest 가 없다: {p}")
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def next_index(d: str) -> int:
    """이제까지 **한 번이라도 쓰인** 번호보다 큰 값. 현재 파일만 보면 안 된다.

    🔴 여기서 한 번 틀렸다(2026-08-25). 폴더에 있는 파일만 보고 max+1 을 잡았는데,
       `wake_qc --move-rejects` 가 생존분을 0..N 으로 **재번호**하는 바람에 폴더의
       최대 번호(1883)가 과거에 쓰인 최대 번호(1999)보다 작았다. 그래서 새 클립에
       **이미 쓴 이름을 다시 붙였고**, manifest 에 같은 이름이 두 번 나왔다.
       그중 8개는 _rejected/ 의 실제 파일과도 이름이 겹쳤다 — 오디오는 멀쩡한데
       '이 클립이 어느 조건에서 나왔나'가 조용히 두 갈래가 된다.
    """
    hi = -1
    seen = list(os.listdir(d))
    bad = os.path.join(d, "_rejected")
    if os.path.isdir(bad):
        seen += os.listdir(bad)
    for name in ("manifest.jsonl", "manifest.pre_qc.jsonl"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                seen += [json.loads(x)["clip"] for x in f if x.strip()]
    for f in seen:
        m = IDX.search(f)
        if m:
            hi = max(hi, int(m.group(1)))
    return hi + 1


def main() -> int:
    ap = argparse.ArgumentParser(description="QC 통과한 보충분을 본 폴더에 합친다")
    ap.add_argument("--src", required=True, help="보충분 폴더(QC 완료 상태)")
    ap.add_argument("--dst", required=True, help="본 폴더")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_rows = read_manifest(args.src)
    dst_rows = read_manifest(args.dst)
    start = next_index(args.dst)

    # manifest 에 있는데 파일이 없으면 이상하다 — 조용히 넘기지 말고 세운다.
    missing = [r["clip"] for r in src_rows
               if not os.path.exists(os.path.join(args.src, r["clip"]))]
    if missing:
        raise SystemExit(f"src manifest 의 파일 {len(missing)}개가 없다: {missing[:3]}")

    print(f"src {len(src_rows)}개 -> dst {len(dst_rows)}개 (다음 번호 clip_{start:06d})")
    new_rows = []
    for i, r in enumerate(src_rows):
        name = f"clip_{start + i:06d}.wav"
        tgt = os.path.join(args.dst, name)
        if os.path.exists(tgt):      # 있을 수 없지만, 있으면 멈춘다
            raise SystemExit(f"이름 충돌: {tgt}")
        if not args.dry_run:
            shutil.copy2(os.path.join(args.src, r["clip"]), tgt)
        new_rows.append({**r, "clip": name})

    if args.dry_run:
        print("(dry-run) 실제 복사·기록 안 함")
        for r in new_rows[:3]:
            print("   ", r["clip"], r["text"], r["speed"])
        return 0

    with open(os.path.join(args.dst, "manifest.jsonl"), "a", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 균형 재점검이 원본 manifest 를 보므로 백업에도 이어 붙인다.
    pre = os.path.join(args.dst, "manifest.pre_qc.jsonl")
    if os.path.exists(pre):
        with open(pre, "a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   manifest.pre_qc.jsonl 에도 {len(new_rows)}행 추가"
              " (생존분만 넣으므로 이 칸의 수율은 100% 로 보인다 — 의도한 것)")

    total = len([f for f in os.listdir(args.dst) if f.endswith(".wav")])
    print(f"합침 완료: {len(new_rows)}개 추가 -> {args.dst} 에 wav {total}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
