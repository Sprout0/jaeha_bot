"""few-shot 을 **텍스트로 설명하는 것** vs **진짜 메시지쌍으로 보여주는 것**.

🔴 왜 (2026-08-28): _augment_system 이 few-shot 18개를 시스템 프롬프트 안의
   `- 아이가 "X" 하면 → "Y"` 텍스트 줄로 넣는다. 모델 입장에선 '예시를 설명하는
   문장'이지 **자기가 그렇게 말한 기록이 아니다.** gpt 는 그래도 따르는데
   HCX 는 안 따른다(같은 질문에 90자 백과사전식 설명).
   말투 모방은 진짜 user/assistant 메시지쌍일 때 훨씬 강하게 걸린다 — 그 가설을 잰다.

⚠️ 길이는 기계·회선과 무관하므로 노트북에서 재도 된다. 여기서 재는 건 **글자 수**다.
⚠️ 채점은 반드시 spoken() 을 거친 문자열로 한다 — 운영이 _postprocess 를 거치므로
   API 원문으로 재면 아이가 듣지 않는 문자열을 재게 된다(2026-08-11 에 당한 일).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(BASE / ".env")

from app.agent import _augment_system, load_fewshot  # noqa: E402
from app.config import settings  # noqa: E402
from tools.bench_llm_latency import make_client  # noqa: E402
from tools.eval_llm import load_eval_set, spoken  # noqa: E402

GAP_S = 0.7          # 테스트 앱 키의 분당 한도를 피한다(연달아 치면 429 가 뜬다)


def build_messages(mode: str, base_system: str, fewshot: list[dict], text: str):
    """mode='text'  = 현행. 예시를 시스템 프롬프트 안에 텍스트로 설명한다.
       mode='turns' = 예시를 **진짜 주고받은 대화**로 앞에 깐다.
    ⚠️ 두 팔이 보는 예시 내용은 **완전히 같다.** 다른 건 담는 그릇 하나뿐이다.
    """
    if mode == "text":
        return [{"role": "system", "content": _augment_system(base_system, fewshot)},
                {"role": "user", "content": text}]
    return ([{"role": "system", "content": base_system}] + list(fewshot)
            + [{"role": "user", "content": text}])


def run(model: str, mode: str, base_system, fewshot, rows, max_tokens: int):
    client = make_client(model)
    out = []
    for r in rows:
        time.sleep(GAP_S)
        for attempt in range(4):
            try:
                res = client.chat.completions.create(
                    model=model, max_completion_tokens=max_tokens,
                    messages=build_messages(mode, base_system, fewshot, r["text"]))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"    재시도({type(e).__name__}) {2 ** attempt * 4}s", flush=True)
                time.sleep(2 ** attempt * 4)
        reply = spoken(res.choices[0].message.content or "")
        out.append({"id": r["id"], "category": r["category"],
                    "ask": r["text"], "reply": reply, "chars": len(reply)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="HCX-005,gpt-4o-mini", help="쉼표 구분")
    ap.add_argument("--eval-set", default="data/eval_set.jsonl",
                    help="⚠️ 안전 14문항 셋은 답이 정형화돼 길이 차이가 안 드러난다")
    ap.add_argument("--out", default="", help="상세 json 저장 경로(비우면 저장 안 함)")
    args = ap.parse_args()

    base_system = settings.prompts["system"]
    fewshot = load_fewshot(settings.models["llm"].get("fewshot_path"))
    rows = load_eval_set(BASE / args.eval_set)
    max_tokens = settings.models["llm"].get("max_tokens", 80)
    print(f"{len(rows)}문항 · few-shot {len(fewshot) // 2}쌍 · max_tokens {max_tokens}\n")

    results = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for mode in ("text", "turns"):
            label = f"{model}/{mode}"
            print(f"  {label} 돌리는 중...", flush=True)
            results[label] = run(model, mode, base_system, fewshot, rows, max_tokens)

    print(f"\n{'팔':22} {'중앙':>6} {'평균':>6} {'최대':>6} {'30자초과':>9}")
    for label, recs in results.items():
        c = [r["chars"] for r in recs]
        print(f"{label:22} {st.median(c):6.1f} {st.mean(c):6.1f} {max(c):6d} "
              f"{sum(1 for x in c if x > 30):5d}/{len(c):<3}")

    print("\n[가장 길었던 답 — 팔별 상위 3개]")
    for label, recs in results.items():
        print(f"  {label}")
        for r in sorted(recs, key=lambda x: -x["chars"])[:3]:
            print(f"    {r['chars']:3d}자 [{r['id']}] {r['ask']} -> {r['reply']}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n상세 -> {p}")


if __name__ == "__main__":
    main()
