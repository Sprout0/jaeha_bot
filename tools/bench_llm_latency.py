"""LLM **지연만** 재는 도구. 품질은 eval_llm.py 가 잰다 — 여기는 시계만 본다.

🔴 왜 따로 만들었나 (2026-08-18):
   같은 날 같은 모델을 두 번 쟀는데 0.855s 와 2.580s 가 나왔다. 코드는 한 글자도 안 바뀌었다.
   **회선이 바뀐 것**이었다. 그래서 지연 비교는 반드시 이렇게 재야 한다:

   1) **팔을 번갈아 친다.** 모델 A 를 다 돌리고 B 를 돌리면, 그 사이 회선이 변한 만큼이
      그대로 'B 가 빠르다'로 둔갑한다. 문항마다 순서를 회전시켜 회선을 공유시킨다.
   2) **네트워크 바닥을 같이 잰다.** `--floor` 는 42자짜리 프롬프트로 "안녕"만 묻는다.
      이게 그 모델 엔드포인트까지의 왕복 하한이다. 프롬프트·모델을 아무리 손봐도 못 내려간다.
      (2026-08-18 실측: gpt-4o-mini 는 젯슨에서 바닥이 **1.735s** 였다. 3초 예산의 절반이다.)
   3) **SDK 자동 재시도를 끈다.** 켜두면 느린 호출 하나가 3배로 부풀어 중앙값·p95 를 둘 다 오염시킨다.

   ⚠️ **반드시 젯슨에서 돌릴 것.** 재려는 게 '그 기계에서 그 서버까지'의 왕복이다.
      노트북에서 잰 숫자는 아무 의미가 없다.

사용:
    python tools/bench_llm_latency.py --models gpt-4o-mini
    python tools/bench_llm_latency.py --models gpt-4o-mini,HCX-DASH-002 --repeat 3
    python tools/bench_llm_latency.py --list-models HCX     # 쓸 수 있는 모델 이름 확인
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

from app.agent import _augment_system, load_fewshot  # noqa: E402
from app.config import settings  # noqa: E402
from tools.eval_llm import CLOVA_BASE_URL, load_eval_set, required_key  # noqa: E402

# 네트워크 왕복 하한을 재기 위한 최소 프롬프트. 내용은 중요하지 않다 — 짧다는 것만 중요하다.
FLOOR_SYSTEM = "너는 다섯 살 아이의 친구 로봇 재하봇이야. 밝은 반말로 한두 문장만 말해."


def make_client(model: str):
    """모델 이름으로 어느 서버에 붙을지 정한다. eval_llm.py 와 같은 규칙을 쓴다.

    🔴 max_retries=0 은 이 도구의 존재 이유와 직결된다 — 재시도가 켜져 있으면
       '느린 호출'과 '세 번 친 호출'을 구분할 수 없다.
    """
    from openai import OpenAI

    key_name = required_key(model)
    if key_name and not os.environ.get(key_name):
        print(f"{key_name} 가 없습니다. 프로젝트 루트 .env 에 넣으세요.")
        sys.exit(1)
    if model.startswith("HCX-"):
        return OpenAI(base_url=CLOVA_BASE_URL, api_key=os.environ[key_name],
                      timeout=30.0, max_retries=0)
    return OpenAI(timeout=30.0, max_retries=0)


def list_models(prefix: str) -> None:
    """엔드포인트가 실제로 주는 모델 이름을 찍는다 — 이름을 추측하지 않기 위해."""
    model = f"{prefix}-list" if prefix.startswith("HCX") else prefix
    client = make_client(model if prefix.startswith("HCX") else "gpt-4o-mini")
    for m in sorted(x.id for x in client.models.list().data):
        if m.upper().startswith(prefix.upper()):
            print(" ", m)


def p95(v: list[float]) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-4o-mini", help="쉼표 구분")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--eval-set", default="data/eval_set_safety.jsonl")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--no-floor", action="store_true", help="네트워크 바닥 측정을 건너뛴다")
    ap.add_argument("--list-models", metavar="PREFIX",
                    help="쓸 수 있는 모델 이름만 찍고 끝낸다(예: HCX)")
    args = ap.parse_args()

    if args.list_models:
        list_models(args.list_models)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    system = _augment_system(settings.prompts["system"],
                             load_fewshot(settings.models["llm"].get("fewshot_path")))
    rows = load_eval_set(BASE / args.eval_set)

    # 팔 = (라벨, 모델, 프롬프트). floor 는 같은 서버까지의 왕복 하한이다.
    arms = []
    for m in models:
        if not args.no_floor:
            arms.append((f"{m} [바닥]", m, FLOOR_SYSTEM))
        arms.append((m, m, system))

    clients = {m: make_client(m) for m in models}
    for m, c in clients.items():   # 커넥션·TLS 예열(첫 호출은 언제나 이상치다)
        c.chat.completions.create(model=m, max_completion_tokens=8,
                                  messages=[{"role": "system", "content": FLOOR_SYSTEM},
                                            {"role": "user", "content": "안녕"}])

    lat: dict[str, list[float]] = {a[0]: [] for a in arms}
    lens: dict[str, list[int]] = {a[0]: [] for a in arms}
    fail: dict[str, int] = {a[0]: 0 for a in arms}
    print(f"{len(rows)}문항 × {args.repeat}회 · 팔 {len(arms)}개 교차 · 재시도 끔 · "
          f"프롬프트 {len(system)}자\n")
    for rep in range(args.repeat):
        for i, r in enumerate(rows):
            k = i % len(arms)
            for label, model, sysp in arms[k:] + arms[:k]:   # 문항마다 순서 회전
                t = time.perf_counter()
                try:
                    out = clients[model].chat.completions.create(
                        model=model, max_completion_tokens=args.max_tokens,
                        messages=[{"role": "system", "content": sysp},
                                  {"role": "user", "content": r["text"]}])
                    lat[label].append(time.perf_counter() - t)
                    lens[label].append(len(out.choices[0].message.content or ""))
                except Exception as e:
                    fail[label] += 1
                    print(f"  실패 {label}: {type(e).__name__}", flush=True)
        print(f"  ...{rep + 1}/{args.repeat}회", flush=True)

    print(f"\n{'팔':28} {'n':>4} {'중앙':>8} {'평균':>8} {'p95':>8} {'최대':>8} {'답변':>6} {'실패':>5}")
    for label, _, _ in arms:
        v = lat[label]
        if not v:
            print(f"{label:28} (전부 실패 {fail[label]}회)")
            continue
        print(f"{label:28} {len(v):4} {statistics.median(v):7.3f}s {statistics.mean(v):7.3f}s "
              f"{p95(v):7.3f}s {max(v):7.3f}s {statistics.median(lens[label]):5.0f}자 {fail[label]:5}")

    print("\n  '[바닥]' = 42자 프롬프트로 \"안녕\"만 물은 값 = 그 서버까지의 왕복 하한.")
    print("  프롬프트·모델을 손봐도 이 아래로는 못 내려간다. 여기가 크면 서버를 옮기는 수밖에 없다.")


if __name__ == "__main__":
    main()
