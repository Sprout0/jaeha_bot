"""호출어 학습용 클립 품질검수(QC) — 학습에 넣기 전에 '진짜 재하봇이라 말하는지' 센다.

왜 필요한가 (2026-08-07):
  v2 생성 중 positive 클립을 들어보니 6개 중 5개가 '재하봇'으로 들리지 않았다.
  25초짜리 폭주(TTS 가 멈추지 못하고 헛소리를 이어감)도 있었다. augment 는 이걸
  2초로 잘라 쓰므로, 그대로 두면 **아무 말이나 = 호출어**로 학습된다.
  v1 의 recall 40% 도 같은 오염 때문이었을 가능성이 크다.

두 축으로 거른다:
  1) 길이 — 호출어 한 마디가 3초를 넘을 이유가 없다. 넘으면 폭주다(STT 없이도 잡힘).
  2) 발음 — STT 로 전사해 호출어와의 자모거리를 잰다. 런타임 판정과 같은 함수를 쓴다
     (app.wake.best_wake_ratio: 0=동일, 클수록 다름. **작을수록 통과**).

사용:
  python tools/wake_qc.py <클립폴더>                     # 검수만(읽기 전용)
  python tools/wake_qc.py <폴더> --move-rejects          # 탈락본을 _rejected/ 로 격리
  python tools/wake_qc.py <폴더> --model small --device cpu   # 노트북에서 가볍게

젯슨/Colab 은 GPU 라 기본값(large-v3-turbo)으로 3000개도 몇 분이면 끝난다.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.wake import best_wake_ratio  # noqa: E402  런타임과 같은 판정을 쓰려고 재사용

# v3 에서 실제로 합성하는 문구(gen_wake_supertonic.DEFAULT_PHRASES 와 같은 갈래).
# 어느 문구로 만든 클립인지는 manifest 를 봐야 알므로, 전부와 대보고 가장 가까운
# 값을 쓴다(하나라도 맞으면 정상 클립).
# ⚠️ 목록을 늘릴수록 판정이 관대해진다 — v2 의 비어휘 변형(재하보사·재하부사·재하보시)은
#    이제 합성하지 않으므로 넣지 않는다. 넣으면 뭉개진 발음까지 통과한다.
DEFAULT_WORDS = ["하이티드", "하이 티드", "하이티드야"]


def best_ratio_any(text: str, words: list[str]) -> tuple[float, str]:
    """여러 문구 중 가장 가까운 (거리, 문구)."""
    best, hit = 1.0, ""
    for w in words:
        r = best_wake_ratio(text, w)
        if r < best:
            best, hit = r, w
    return best, hit


def main() -> int:
    ap = argparse.ArgumentParser(description="호출어 클립 품질검수")
    ap.add_argument("directory", help="clip_*.wav 가 있는 폴더")
    ap.add_argument("--words", default=",".join(DEFAULT_WORDS),
                    help="쉼표로 구분한 목표 문구들")
    ap.add_argument("--max-ratio", type=float, default=0.45,
                    help="자모거리 컷(작을수록 엄격). 기본 0.45")
    ap.add_argument("--max-dur", type=float, default=3.0,
                    help="이 길이를 넘으면 폭주로 본다(초). 기본 3.0")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만(빠른 표본검사)")
    ap.add_argument("--move-rejects", action="store_true",
                    help="탈락본을 _rejected/ 로 옮기고 clip 인덱스를 다시 매긴다")
    ap.add_argument("--quiet", action="store_true", help="개별 줄 없이 요약만")
    args = ap.parse_args()

    import soundfile as sf
    from faster_whisper import WhisperModel

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    clips = sorted(glob.glob(os.path.join(args.directory, "clip_*.wav")))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"clip_*.wav 없음: {args.directory}")
        return 1

    device = args.device
    compute = "float16"
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if device == "cpu":
        compute = "int8"

    print(f"클립 {len(clips)}개 | STT {args.model}/{device} | "
          f"컷: 길이>{args.max_dur}s 또는 자모거리>{args.max_ratio}\n")
    model = WhisperModel(args.model, device=device, compute_type=compute)

    rows, n_long, n_far = [], 0, 0
    for p in clips:
        dur = sf.info(p).duration
        if dur > args.max_dur:
            # 폭주는 전사할 가치가 없다(그리고 25초짜리는 전사도 느리다).
            rows.append((p, dur, "", 1.0, "", False, "길이"))
            n_long += 1
            continue
        segs, _ = model.transcribe(p, language="ko", beam_size=5,
                                   condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        ratio, hit = best_ratio_any(text, words)
        ok = ratio <= args.max_ratio
        if not ok:
            n_far += 1
        rows.append((p, dur, text, ratio, hit, ok, "" if ok else "발음"))

    if not args.quiet:
        for p, dur, text, ratio, hit, ok, why in rows:
            mark = "OK  " if ok else f"탈락({why})"
            print(f"{mark:>9} {os.path.basename(p):>18} {dur:5.2f}s "
                  f"거리 {ratio:4.2f} {('~' + hit) if hit else '':>8}  {text[:38]}")

    n = len(rows)
    n_ok = sum(1 for r in rows if r[5])
    print(f"\n통과 {n_ok}/{n} ({n_ok / n * 100:.1f}%)  "
          f"| 탈락: 길이 {n_long} · 발음 {n_far}")

    # manifest.jsonl 이 있으면 '어느 축이 범인인지' 바로 갈라 준다.
    # (voxcpm 때는 클립→생성조건 매핑이 없어 이 진단 자체가 불가능했다.)
    man_path = os.path.join(args.directory, "manifest.jsonl")
    if os.path.exists(man_path):
        import json
        man = {}
        with open(man_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    man[rec["clip"]] = rec
        ok_by = {}
        for p, _dur, _t, _r, _hit, ok, _why in rows:
            rec = man.get(os.path.basename(p))
            if not rec:
                continue
            for key in ("text", "voice", "speed", "rate"):
                ok_by.setdefault(key, {}).setdefault(rec[key], []).append(ok)
        for key, groups in ok_by.items():
            print(f"\n  [{key}] 별 통과율")
            for val in sorted(groups, key=lambda v: (isinstance(v, str), v)):
                v = groups[val]
                bar = "#" * int(sum(v) / len(v) * 20)
                print(f"    {str(val):<10} {sum(v)/len(v)*100:5.0f}% "
                      f"({sum(v)}/{len(v)}) {bar}")
    if n_ok / n < 0.8:
        print("→ 80% 미만. 이 데이터로 학습하면 안 된다 — 생성 설정을 먼저 고칠 것.")

    if args.move_rejects:
        bad_dir = os.path.join(args.directory, "_rejected")
        os.makedirs(bad_dir, exist_ok=True)
        keep = [r[0] for r in rows if r[5]]
        for r in rows:
            if not r[5]:
                shutil.move(r[0], os.path.join(bad_dir, os.path.basename(r[0])))
        # 인덱스 연속성 복구 — 피치/배속 증강이 max(idx)+1 부터 이어 붙이기 때문.
        # 오름차순 처리라 목표 이름은 항상 이미 비어 있다(충돌 없음).
        renamed = {}
        for i, p in enumerate(keep):
            tgt = os.path.join(args.directory, f"clip_{i:06d}.wav")
            if os.path.abspath(p) != os.path.abspath(tgt):
                shutil.move(p, tgt)
            renamed[os.path.basename(p)] = os.path.basename(tgt)
        print(f"격리 {n - n_ok}개 -> {bad_dir} / 남은 {len(keep)}개 재번호 완료")

        # manifest 도 같이 고쳐야 한다. 안 그러면 재번호 후 '클립->생성조건' 매핑이
        # 조용히 어긋나 진단이 거짓말을 하기 시작한다.
        if os.path.exists(man_path):
            import json
            out_lines = []
            with open(man_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    new = renamed.get(rec["clip"])
                    if new:
                        rec["clip"] = new
                        out_lines.append(json.dumps(rec, ensure_ascii=False))
            with open(man_path, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            print(f"manifest 갱신: {len(out_lines)}줄")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
