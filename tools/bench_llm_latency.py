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
from types import SimpleNamespace

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


class LocalClient:
    """로컬 EXAONE 을 OpenAI 클라이언트 모양으로 감싼다.

    측정 루프를 백엔드마다 갈라 쓰지 않으려는 것이다 — 갈라 쓰면 '어느 쪽에만 적용된
    조건'이 생기고, 그게 곧 모델 차이로 둔갑한다.

    🔴 **backend 를 local 로 강제한다.** 설정이 `backend: openai` 인 기계에서 이걸
       빠뜨리면 표에는 'local' 이라 찍히면서 실제로는 gpt 를 잰 값이 들어간다.
       조용히 틀리는 종류의 오류라 테스트로 못 박아 뒀다.
    ⚠️ respond() 가 아니라 `_complete_local` 을 직접 부른다. 이유 둘:
       - 팔마다 시스템 프롬프트가 다르다([바닥] vs 실제). respond() 는 생성자에
         박힌 프롬프트만 쓴다.
       - respond() 는 대화 이력을 쌓는다. 뒤 문항일수록 프롬프트가 길어져서
         '나중 문항이 느리다'가 되는데, 그건 모델 성질이 아니라 우리가 만든 기울기다.
       원격 팔도 후처리 전 원문을 재므로 잣대는 같다(품질 채점은 eval_llm.py 담당).
    """

    def __init__(self):
        from app.agent import LLMAgent

        cfg = dict(settings.models["llm"])
        cfg["backend"] = "local"
        self._agent = LLMAgent(model_path=cfg.pop("model_path"),
                               system_prompt=settings.prompts["system"], **cfg)
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model: str, max_completion_tokens: int, messages: list[dict]):
        self._agent.max_tokens = max_completion_tokens
        text = self._agent._complete_local(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def make_client(model: str):
    """모델 이름으로 어느 서버에 붙을지 정한다. eval_llm.py 와 같은 규칙을 쓴다.

    🔴 max_retries=0 은 이 도구의 존재 이유와 직결된다 — 재시도가 켜져 있으면
       '느린 호출'과 '세 번 친 호출'을 구분할 수 없다.
    """
    if model == "local":
        return LocalClient()   # 서버가 없다. openai 를 import 조차 하지 않는다.

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


def build_arms(models: list[str], system: str,
               with_floor: bool) -> list[tuple[str, str, str]]:
    """팔 = (라벨, 모델, 시스템 프롬프트).

    🔴 **local 에는 [바닥] 팔을 붙이지 않는다.** 2026-08-19 에 붙였다가 크게 틀렸다:
         팔 교차(바닥 42자 ↔ 실제 4185자)   중앙 6.277s
         프롬프트 고정                      중앙 2.097s
       llama.cpp 는 같은 시스템 프롬프트가 이어지면 접두 KV 캐시를 재사용한다.
       두 프롬프트를 번갈아 치면 매 호출 2,356토큰 prompt eval 을 처음부터 다시 낸다.
       운영은 프롬프트가 고정이라 그 4.2초를 안 낸다 — **잰 값이 운영에 없는 값이었다.**
       ⚠️ 원격 팔이 사이에 끼는 건 무해하다(llama 컨텍스트를 안 건드린다).
          깨뜨리는 건 오직 '로컬의 두 번째 프롬프트'다.
       그리고 [바닥] 은 정의상 '그 서버까지의 왕복 하한'인데 local 은 갈 서버가 없다.
    """
    arms: list[tuple[str, str, str]] = []
    for m in models:
        if with_floor and m != "local":
            arms.append((f"{m} [바닥]", m, FLOOR_SYSTEM))
        arms.append((m, m, system))
    return arms


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

    arms = build_arms(models, system, with_floor=not args.no_floor)

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

    print()
    print("  '[바닥]' = 42자 프롬프트로 \"안녕\"만 물은 값.")
    print("  원격 팔 = 그 서버까지의 왕복 하한. 프롬프트·모델을 손봐도 이 아래로는 못 내려간다.")
    print("           여기가 크면 서버를 옮기는 수밖에 없다.")
    print("  local 팔 = 왕복이 없으니 **프롬프트 길이의 대가**만 남는다. 원격은 길이가")
    print("           지연과 무관했지만(08-18 A/B) 로컬은 prompt eval 을 직접 치른다.")


if __name__ == "__main__":
    main()
