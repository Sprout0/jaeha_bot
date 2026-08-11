"""LLM 비교 평가 — 같은 질문을 여러 모델에 던져 숫자로 비교한다.

왜 필요한가: "API 로 바꾸니 좋아진 것 같다"로는 교수님께 설명할 수 없다.
같은 평가셋(data/eval_set.jsonl)을 같은 프롬프트로 돌려서 아래를 재고 비교한다.

  날조율   실제로 못 하는 놀이·노래를 하고 있다/해주겠다고 말한 비율 (app/claims.py)
  답변 길이 2세는 길면 못 알아듣는다. 목표 20~34자
  지연     p50 과 **p95**. 평균보다 편차가 문제다 — 가끔 4초 걸리면 아이가 못 기다린다
  캐시     프롬프트 캐시 적중 토큰. 안 걸리면 비용이 10배인데 조용히 그렇게 된다

⚠️ 날조율은 규칙 기반 자동 판정이라 기존 수동 측정치(15%)와 **직접 비교하면 안 된다.**
   모델 간 비교에만 쓴다(모두 같은 잣대로 재므로 상대 비교는 유효).
⚠️ 평가셋은 전부 가상의 유아 발화다. 실제 아이 녹취를 API 로 보내지 않는다.

사용:
  python tools/eval_llm.py --models gpt-5.6-luna
  python tools/eval_llm.py --models gpt-5.6-luna,gpt-5.6-terra --repeat 2
  python tools/eval_llm.py --models local          # 로컬 EXAONE 기준선
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# 프로젝트 루트 .env 에서 키를 읽는다(있으면). setx 는 새로 뜨는 프로그램부터
# 적용돼서 앱을 재시작해야 하는데, .env 는 그럴 필요가 없다.
# .env 는 .gitignore 에 있으므로 커밋되지 않는다.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

from app.agent import (                                      # noqa: E402
    _augment_system, _clamp_sentences, _flatten_markdown,
    _strip_emoji, _strip_speaker_prefix, load_fewshot,
)
from app.claims import find_fabrications                     # noqa: E402
from app.safety import check_reply                           # noqa: E402
from app.config import settings                              # noqa: E402


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def build_system_prompt() -> str:
    """agent.py 와 **똑같이** 조립한다 — 프롬프트가 다르면 비교가 무의미하다."""
    cfg = settings.models["llm"]
    fewshot = load_fewshot(cfg.get("fewshot_path"))
    return _augment_system(settings.prompts["system"], fewshot)


def spoken(text: str) -> str:
    """모델 원문 -> **아이가 실제로 듣는 문자열**. 채점 대상은 항상 이쪽이다.

    🔴 왜 이게 여기 있어야 하는가 (2026-08-11 에 실제로 당한 일):
       운영은 app/agent.py `_postprocess` 를 반드시 거친다 — 이름표 제거,
       마크다운 평탄화, 이모지 제거, 그리고 **최대 max_sentences 문장으로 잘라내기**.
       그런데 평가는 API 원문을 그대로 채점했다. 그래서 안전 평가셋이 42/42 통과로
       찍혔는데, 실제로 입 밖으로 나가는 문장은 세 번째 절이 잘린 것이었다:
         원문  "그건 아직 못 해! 위험해! 엄마 아빠한테 물어보자."
         실제  "그건 아직 못 해! 위험해!"            ← 어른 유도가 사라진다
       42문항 중 30문항이 그랬다. 즉 **아이가 듣지 않는 문자열로 안전을 인증했다.**
    ⚠️ 이 함수를 지우거나 '어차피 원문이 더 정확하다'며 되돌리면 그 구멍이 그대로
       다시 열린다. 백엔드가 무엇이든(local/openai/gemini) 채점은 말해지는 문자열로.
       make_local 은 respond() 안에서 이미 같은 후처리를 거치므로 두 번 걸어도 무해하다
       (후처리는 멱등이다) — 중요한 건 '모든 백엔드가 같은 잣대'라는 것.
    """
    text = _strip_speaker_prefix(text)
    text = _flatten_markdown(text)
    text = _strip_emoji(text)
    return _clamp_sentences(text, _max_sentences())


def _max_sentences() -> int:
    """운영이 쓰는 하드캡을 config 에서 그대로 읽는다(값을 여기 베끼지 않는다)."""
    return int(settings.models["llm"].get("max_sentences", 2))


# ── 백엔드 ────────────────────────────────────────────────────────────────────
# 각 백엔드는 (답변, 캐시적중토큰) 을 돌려주는 callable 을 만든다.
# 새 제공사를 붙일 때 여기만 늘리면 되고 지표·리포트는 그대로 쓴다.
# 🔴 새 백엔드도 반드시 spoken() 을 통과시킬 것 — 위 주석의 이유.

def make_openai(model: str, system: str, max_tokens: int):
    from openai import OpenAI
    client = OpenAI()

    def ask(text: str) -> tuple[str, int]:
        r = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": text}],
        )
        usage = r.usage
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return spoken((r.choices[0].message.content or "").strip()), cached

    return ask


def make_gemini(model: str, system: str, max_tokens: int):
    """Gemini. 사고를 최소로 낮춘다(34자 답변에 사고 시간을 쓸 이유가 없다).

    🔴 세대별로 파라미터가 다르다:
      - 2.5 계열: thinking_config.thinking_budget=0 으로 끈다
      - 3.x 계열: thinking_budget 은 **400 INVALID_ARGUMENT**. thinking_level 을 쓴다
        (THINKING_LEVEL_UNSPECIFIED / MINIMAL / LOW / MEDIUM / HIGH)
    실제로 겪은 일: 3.6-flash 에 budget=0 을 보내 전부 400 이 났는데, 진행 표시줄이
    오류를 안 찍어서 '모델이 빈 답을 낸다'고 오진했다.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()
    is_v3 = model.startswith("gemini-3")
    thinking = (types.ThinkingConfig(thinking_level="MINIMAL") if is_v3
                else types.ThinkingConfig(thinking_budget=0))

    def ask(text: str) -> tuple[str, int]:
        r = client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                thinking_config=thinking,
            ),
        )
        u = r.usage_metadata
        cached = getattr(u, "cached_content_token_count", 0) or 0
        thoughts = getattr(u, "thoughts_token_count", 0) or 0
        if thoughts:
            log_once(f"⚠️ {model}: thinking 이 꺼지지 않았다(사고 {thoughts}토큰)")
        return spoken((r.text or "").strip()), cached

    return ask


_warned: set[str] = set()


def log_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(msg)


def make_local(system: str):
    """로컬 EXAONE 기준선. 비교 대상이 있어야 '좋아졌다'를 말할 수 있다."""
    from app.agent import LLMAgent
    cfg = dict(settings.models["llm"])
    agent = LLMAgent(model_path=cfg.pop("model_path"), system_prompt=system, **cfg)

    def ask(text: str) -> tuple[str, int]:
        return (agent.respond(text).get("text") or "").strip(), 0

    return ask


_RATE_LIMIT = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota")


def _ask_with_retry(ask, text: str, tries: int = 4) -> tuple[str, int, str | None, float]:
    """속도 제한이면 기다렸다 다시. Gemini 무료 티어는 분당 5~15회라 그냥 돌리면 죽는다.

    지연 측정에는 **대기 시간을 빼고** 마지막 시도만 센다 — 우리가 재고 싶은 건
    모델의 응답 속도이지 우리가 얼마나 기다렸는지가 아니다.
    """
    for attempt in range(tries):
        t0 = time.perf_counter()
        try:
            reply, cached = ask(text)
            return reply, cached, None, time.perf_counter() - t0
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if attempt < tries - 1 and any(k.lower() in msg.lower() for k in _RATE_LIMIT):
                wait = 8 * (attempt + 1)
                print(f"    (속도 제한 — {wait}s 대기 후 재시도)")
                time.sleep(wait)
                continue
            return "", 0, msg, time.perf_counter() - t0
    return "", 0, "재시도 소진", 0.0


def evaluate(name: str, ask, rows: list[dict], repeat: int) -> dict:
    records = []
    for _ in range(repeat):
        for row in rows:
            reply, cached, error, elapsed = _ask_with_retry(ask, row["text"])
            records.append({
                "id": row["id"], "category": row["category"], "ask": row["text"],
                "want": row.get("want", ""), "reply": reply, "error": error,
                "latency_s": round(elapsed, 3), "chars": len(reply),
                "cached_tokens": cached,
                "fabrications": find_fabrications(reply),
                "safety_flags": check_reply(reply),
            })
            # 🔴 오류를 반드시 표시한다. 예전엔 답변 길이만 찍어서 429 로 실패한 요청이
            # '0자 응답'처럼 보였고, 모델이 빈 답을 낸다고 오진했다.
            # flush: 백그라운드/파이프로 돌릴 때 버퍼링되면 진행 상황이 안 보인다.
            shown = f"❌ {error[:60]}" if error else reply[:42]
            print(f"  {row['id']:7} {elapsed:5.2f}s {len(reply):3d}자  {shown}", flush=True)
    return {"model": name, "records": records, "summary": summarize(records)}


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if not r["error"]]
    lat = sorted(r["latency_s"] for r in ok)
    chars = [r["chars"] for r in ok]
    fab = [r for r in ok if r["fabrications"]]
    by_cat: dict[str, dict] = {}
    for r in ok:
        c = by_cat.setdefault(r["category"], {"n": 0, "fab": 0})
        c["n"] += 1
        c["fab"] += bool(r["fabrications"])
    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "fabrication_rate": round(len(fab) / len(ok), 3) if ok else None,
        "chars_median": statistics.median(chars) if chars else None,
        "chars_max": max(chars) if chars else None,
        "latency_p50": _pct(lat, 0.50),
        "latency_p95": _pct(lat, 0.95),
        "cache_hit_turns": sum(1 for r in ok if r["cached_tokens"] > 0),
        # 🔴 오류 없이 '빈 답변'을 낸 횟수. 아이에게 아무 말도 안 한 것이라
        # 오류만큼 심각한데, 오류로 안 잡혀서 표에서 조용히 사라졌었다.
        "empty_replies": sum(1 for r in ok if not r["reply"]),
        "unsafe": sum(1 for r in ok if r.get("safety_flags")),
        "unsafe_invite": sum(1 for r in ok if "위험행동제안" in (r.get("safety_flags") or [])),
        "by_category": by_cat,
    }


def _pct(values: list[float], q: float):
    if not values:
        return None
    return round(values[min(int(q * len(values)), len(values) - 1)], 3)


def report(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    # 🔴 헤더와 아래 행은 값 9개가 1:1로 대응해야 한다 — 헤더 라벨을 빼먹으면
    # 뒤의 모든 값이 한 칸씩 밀려서 엉뚱한 헤더 아래 찍힌다(예: unsafe 가 '오류' 밑에).
    # 열을 추가/삭제할 때는 반드시 헤더와 행을 같이 고칠 것.
    print(f"{'모델':22} {'날조율':>7} {'글자중앙':>8} {'최대':>5} "
          f"{'지연p50':>8} {'지연p95':>8} {'캐시턴':>7} {'빈답':>5} {'위험답':>5} {'오류':>5}")
    print("-" * 78)
    for res in results:
        s = res["summary"]
        fr = "-" if s["fabrication_rate"] is None else f"{s['fabrication_rate']*100:.1f}%"
        print(f"{res['model']:22} {fr:>7} {str(s['chars_median']):>8} "
              f"{str(s['chars_max']):>5} {str(s['latency_p50']):>8} "
              f"{str(s['latency_p95']):>8} {s['cache_hit_turns']:>7} "
              f"{s.get('empty_replies', 0):>5} {s.get('unsafe', 0):>5} {s['errors']:>5}")
    print("=" * 78)
    print("\n[카테고리별 날조]")
    for res in results:
        cats = res["summary"]["by_category"]
        line = "  ".join(f"{k}:{v['fab']}/{v['n']}" for k, v in sorted(cats.items()))
        print(f"  {res['model']:22} {line}")
    print("\n⚠️ 날조율은 규칙 기반 자동 판정이다. 모델 간 비교에만 쓰고,")
    print("   기존 수동 측정치(15%)와 직접 비교하지 말 것. 실제 답변은 저장된 json 에서 검토.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.6-luna",
                    help="쉼표 구분. 'local' 이면 로컬 EXAONE 기준선")
    ap.add_argument("--repeat", type=int, default=1,
                    help="지연 편차(p95)를 보려면 2 이상 권장")
    ap.add_argument("--eval-set", default="data/eval_set.jsonl")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--rescore", metavar="LOG.json",
                    help="저장된 결과를 지금 판정기로 다시 채점한다(API 재호출 없음). "
                         "날조 판정 규칙을 고쳤을 때 옛 실행을 같은 잣대로 다시 보려고.")
    ap.add_argument("--ids-from", metavar="EVALSET.jsonl",
                    help="--rescore 와 함께. 이 평가셋에 있는 id 만 집계한다. "
                         "문항 수가 다른 실행을 같은 문항으로 맞춰 비교할 때.")
    args = ap.parse_args()

    if args.rescore:
        results = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        keep = None
        if args.ids_from:
            keep = {r["id"] for r in load_eval_set(BASE / args.ids_from)}
            print(f"{args.ids_from} 의 {len(keep)}문항으로만 집계")
        for res in results:
            if keep is not None:
                res["records"] = [r for r in res["records"] if r["id"] in keep]
            for rec in res["records"]:
                # 옛 로그는 API 원문이 그대로 저장돼 있다(spoken() 이전 실행).
                # 같은 잣대로 다시 보려면 여기서도 '말해지는 문자열'로 바꿔 채점한다 —
                # 그러지 않으면 옛 실행은 계속 '아이가 듣지 않는 문장'으로 채점된다.
                # 새 로그는 이미 후처리된 문자열이고 spoken() 은 멱등이라 안전하다.
                rec["reply"] = spoken(rec["reply"])
                rec["chars"] = len(rec["reply"])
                rec["fabrications"] = find_fabrications(rec["reply"])
                rec["safety_flags"] = check_reply(rec["reply"])
            res["summary"] = summarize(res["records"])
        report(results)
        return

    rows = load_eval_set(BASE / args.eval_set)
    system = build_system_prompt()
    max_tokens = args.max_tokens or settings.models["llm"].get("max_tokens", 80)
    print(f"평가셋 {len(rows)}문항 × {args.repeat}회 | 시스템 프롬프트 {len(system)}자")

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    for name, var in (("gpt", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        if any(n.startswith(name) for n in names) and not os.environ.get(var):
            print(f"\n⚠️ {var} 가 없습니다. 프로젝트 루트 .env 에 한 줄 넣거나,")
            print("   setx 로 등록한 뒤 터미널(또는 앱)을 새로 띄우세요.")
            sys.exit(1)

    results = []
    for name in names:
        print(f"\n=== {name} ===")
        if name == "local":
            ask = make_local(system)
        elif name.startswith("gemini"):
            ask = make_gemini(name, system, max_tokens)
        else:
            ask = make_openai(name, system, max_tokens)
        results.append(evaluate(name, ask, rows, args.repeat))

    report(results)
    out = BASE / "logs" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n상세(답변 전문 포함): {out}")


if __name__ == "__main__":
    main()
