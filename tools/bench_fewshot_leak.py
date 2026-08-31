"""few-shot 을 진짜 대화로 주면 예시의 '주제'가 실제 답에 새는가.

🔴 왜 (2026-08-31): 08-28 에 few-shot 을 메시지쌍(turns)으로 주면 HCX 의 길이가
   잡힌다는 걸 쟀다. 그런데 **반대 근거가 이미 기록돼 있다** — 예전에 가짜 대화이력으로
   주입했을 때 이전 예시의 주제가 실제 답에 샜다(무의미 입력에 '빨간색 게임'
   confabulation). 그래서 지금의 텍스트 방식으로 바꾼 것이다.
   08-28 측정은 **길이만** 쟀고 누수는 안 봤다. 그릇을 바꾸기 전에 그 하나를 확인한다.

⚠️ 재현 조건이 중요하다. 누수는 **말할 거리가 없는 입력**에서 난다 — 모델이 지어내야
   하는 자리에서 눈앞의 예시를 집어 온다. 그래서 입력은 무의미/모호한 말만 쓴다.
   (평범한 질문으로 재면 누수가 안 보인다 — 답할 거리가 이미 있으니까.)
⚠️ 운영 코드(LLMAgent.respond)를 그대로 태운다. 메시지 조립을 여기서 다시 짜면
   운영과 다른 것을 재게 된다.
⚠️ 턴마다 이력을 비운다. 안 그러면 앞 문항의 답이 다음 문항의 맥락이 돼
   무엇이 few-shot 에서 왔는지 갈리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(BASE / ".env")

from app.agent import LLMAgent  # noqa: E402
from app.config import settings  # noqa: E402

# 말할 거리가 없는 입력. 앞 다섯은 whisper 가 소음에 붙이는 실제 환각 문구다
# (reports 기록: '다음 영상에서 만나요' 등). 나머지는 아이의 웅얼거림.
NONSENSE = [
    "다음 영상에서 만나요",
    "오늘도 시청해주셔서 감사합니다",
    "ㅁㄴㅇㄹ",
    "아아아",
    "어 그",
    "음...",
    "그거",
    "저기",
    "우와",
    "몰라",
    "이거",
    "응?",
    # 2026-08-31: n=12 로는 누수 1건이 우연인지 안 갈렸다. 표본을 늘렸다.
    "어어어",
    "아니 그게",
    "빠빠빠",
    "흐음",
    "뭐지",
    "있잖아",
    "그래서",
    "네네",
    "아 진짜",
    "오",
    "됐어",
    "잠깐만",
]

# few-shot 답변에만 나오는 구체 명사. 무의미 입력의 답에 이게 뜨면 예시에서 샌 것이다.
# ⚠️ 손으로 고른 목록이다 — 숫자만 믿지 말고 아래 전문 출력도 같이 볼 것.
LEAK_WORDS = ["공룡", "딸기", "자동차", "빵빵", "하연", "강아지", "멍멍",
              "아기", "코 자", "빨개", "엄마", "아빠"]


def make_agent(model: str, form: str) -> LLMAgent:
    cfg = dict(settings.models["llm"])
    model_path = cfg.pop("model_path")
    cfg["fewshot_form"] = form
    if model == "local":
        cfg.update(backend="local")
    else:
        cfg.update(backend="openai", api_model=model)
    return LLMAgent(model_path=str(BASE / model_path),
                    system_prompt=settings.prompts["system"], **cfg)


def run(model: str, form: str, asks: list[str]) -> list[dict]:
    agent = make_agent(model, form)
    out = []
    for ask in asks:
        agent.history.clear()          # 앞 문항이 맥락으로 남으면 원인이 안 갈린다
        reply = agent.respond(ask)["text"]
        out.append({"ask": ask, "reply": reply,
                    "leaks": [w for w in LEAK_WORDS if w in reply]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-4o-mini",
                    help="쉼표 구분. 'local' 이면 폴백 GGUF(EXAONE) — 누수가 관찰됐던 모델")
    ap.add_argument("--forms", default="text,turns", help="쉼표 구분")
    ap.add_argument("--out", default="", help="상세 json 저장 경로(비우면 저장 안 함)")
    args = ap.parse_args()

    results = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for form in [f.strip() for f in args.forms.split(",") if f.strip()]:
            label = f"{model}/{form}"
            print(f"\n=== {label} ===", flush=True)
            recs = run(model, form, NONSENSE)
            results[label] = recs
            for r in recs:
                mark = "🔴" if r["leaks"] else "  "
                print(f"{mark} [{r['ask']}] -> {r['reply']}"
                      + (f"   << {','.join(r['leaks'])}" if r["leaks"] else ""), flush=True)

    print(f"\n{'팔':22} {'누수 답변':>10} {'샌 낱말':>10}")
    for label, recs in results.items():
        n = sum(1 for r in recs if r["leaks"])
        words = sorted({w for r in recs for w in r["leaks"]})
        print(f"{label:22} {n:5d}/{len(recs):<4} {' '.join(words) or '-'}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n상세 -> {p}")


if __name__ == "__main__":
    main()
