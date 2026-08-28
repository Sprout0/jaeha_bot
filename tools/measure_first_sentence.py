"""문장 스트리밍이 첫 소리를 얼마나 당기는가 — **재생 없이** 합성만 잰다(소리 안 남).

🔴 무엇을 재나 (2026-08-28):
   지금은 통짜 합성이다 — 답변 전체를 다 만든 뒤에야 첫 소리가 난다.
   문장 스트리밍이면 첫 문장만 만들고 바로 틀 수 있으니, 이득 = 통짜 − 첫문장.

⚠️ 그런데 이득만 재면 안 된다. 첫 문장을 트는 동안 나머지를 만들어야 하는데
   **나머지 합성이 첫 문장 재생보다 오래 걸리면 문장 사이가 비어 '뚝' 끊긴다.**
   그래서 나머지 합성시간과 첫 문장 재생시간을 같이 재서 여유를 본다.
   (이게 음수면 문장 스트리밍은 그 문장에서 침묵을 만든다.)

⚠️ 입력은 **오늘 실제로 gpt 가 낸 답 32개**를 쓴다. 손으로 만든 문장으로 재면
   길이 분포가 운영과 달라져 숫자가 옮겨가지 않는다.
⚠️ 문장 분리는 운영과 같은 TTSModule._split_sentences 를 쓴다.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path("/home/jaeha_bot/jaeha_bot")
sys.path.insert(0, str(REPO))

from app.config import settings          # noqa: E402
from app.tts_module import TTSModule     # noqa: E402


def main():
    src = json.load(open(REPO / "eval_20260828_145622.json", encoding="utf-8"))
    replies = [r["reply"] for a in src if a["model"] == "gpt-4o-mini"
               for r in a["records"] if r.get("reply")]

    tts = TTSModule(**settings.models["tts"])
    tts.load()
    tts.warm()
    for _ in range(2):
        tts._infer("워밍업 문장이야. 조금 더 길게 데운다.")   # 첫 합성은 이상치다

    rows = []
    for text in replies:
        parts = TTSModule._split_sentences(text)
        if len(parts) < 2:
            rows.append({"text": text, "n": len(parts), "skip": True})
            continue
        first, rest = parts[0], " ".join(parts[1:])

        t = time.perf_counter(); whole = tts._infer(text);  t_whole = time.perf_counter() - t
        t = time.perf_counter(); a1 = tts._infer(first);    t_first = time.perf_counter() - t
        t = time.perf_counter(); a2 = tts._infer(rest);     t_rest = time.perf_counter() - t

        rate = tts.sample_rate
        rows.append({
            "text": text, "n": len(parts), "skip": False,
            "chars": len(text), "t_whole": t_whole, "t_first": t_first, "t_rest": t_rest,
            "dur_whole": len(whole) / rate, "dur_first": len(a1) / rate,
            "gain": t_whole - t_first,             # 첫 소리가 이만큼 빨라진다
            "slack": len(a1) / rate - t_rest,      # 음수면 문장 사이가 빈다
        })
        print(f"  {len(text):3d}자 {len(parts)}문장 | 통짜 {t_whole:.3f}s -> 첫문장 {t_first:.3f}s "
              f"| 이득 {t_whole - t_first:+.3f}s | 여유 {len(a1) / rate - t_rest:+.3f}s",
              flush=True)

    ok = [r for r in rows if not r["skip"]]
    skipped = [r for r in rows if r["skip"]]
    print(f"\n대상 {len(ok)}개 / 한 문장이라 건너뜀 {len(skipped)}개 (총 {len(rows)})")
    if ok:
        print(f"  통짜 합성   중앙 {st.median([r['t_whole'] for r in ok]):.3f}s")
        print(f"  첫문장 합성 중앙 {st.median([r['t_first'] for r in ok]):.3f}s")
        print(f"  ➡️ 첫 소리 이득 중앙 {st.median([r['gain'] for r in ok]):+.3f}s "
              f"(최소 {min(r['gain'] for r in ok):+.3f} / 최대 {max(r['gain'] for r in ok):+.3f})")
        neg = [r for r in ok if r["slack"] < 0]
        print(f"  ⚠️ 여유 중앙 {st.median([r['slack'] for r in ok]):+.3f}s · "
              f"**음수(문장 사이가 빈다) {len(neg)}/{len(ok)}개**")
        for r in sorted(ok, key=lambda x: x["slack"])[:3]:
            print(f"     여유 {r['slack']:+.3f}s ({r['chars']}자): {r['text']}")

    out = Path("/tmp/tts_first_sentence.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n상세 -> {out}")


if __name__ == "__main__":
    main()
