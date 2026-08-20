"""백엔드 조합별 응답 지연 측정. LLM·TTS 후보가 바뀔 때마다 같은 잣대로 재려고 만들었다.

실제 설정(configs/)·실제 시스템 프롬프트·실제 평가셋을 쓴다 — 손으로 만든 문장으로 재면
운영과 다른 숫자가 나온다. 모델 로드와 커넥션은 재기 전에 예열한다(첫 호출은 이상치다).

    python tools/bench_pipeline.py                          # 기본 3조합
    python tools/bench_pipeline.py --combos openai+openai
    python tools/bench_pipeline.py --play                   # 실제 재생 + 말하는 시간까지

조합 표기는 `<llm>+<tts>` — llm 은 local|openai, tts 는 supertonic|openai.
llm 쪽에 `:모델`을 붙이면 그 모델로 고정한다 — gpt 와 HCX 를 같은 표에서 비교할 때:
    --combos "openai:gpt-4o-mini+supertonic,openai:HCX-005+supertonic" --play

🔴 `--play` 로 재면 '체감' 열이 **LLM + 첫 소리 + 말하는 시간**이다. speak() 는 재생이
   끝날 때까지 블로킹하므로 아이가 다음 말을 할 수 있게 되는 시점은 말이 끝난 뒤다.
   2026-08-19 까지 합계가 `LLM + 첫 소리` 였고, 그래서 **답변이 길어지는 대가가 아예
   안 잡혔다.** 답변 길이가 다른 모델을 비교할 때는 이 열만 봐야 한다.
⚠️ 길이 대가를 보려면 평가셋을 `data/eval_set.jsonl`(37문항)로 쓸 것. 기본값인
   안전 14문항은 답이 짧게 정형화돼 있어("~위험해! 엄마 아빠한테 물어보자!")
   모델 간 길이 차이가 드러나지 않는다.

⚠️ `--play` 없이는 **합성 완료까지**를 잰다. OpenAI TTS 스트리밍은 speak() 안에서만
   동작하므로, 스트리밍 이득을 보려면 반드시 `--play` 로 재야 한다(소리가 난다).

기준선 (젯슨 Orin Nano, 2026-08-10, 안전 14문항×2회, 프롬프트 2165자):
  조합                     LLM      TTS     합계
  local+supertonic       1.769s   0.885s  2.619s   (답변 41자)
  openai+supertonic      0.855s   0.880s  1.735s   (답변 18자)  <- 채택
  openai+openai(통짜)     0.989s   1.432s  2.370s   (답변 20자)
  --play 기준 첫 소리: OpenAI 스트리밍 0.694s / Supertonic 1.071s / OpenAI 통짜 1.641s
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.agent import LLMAgent, _augment_system, load_fewshot  # noqa: E402
from app.config import settings  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402
from tools.eval_llm import load_eval_set  # noqa: E402

DEFAULT_COMBOS = "local+supertonic,openai+supertonic,openai+openai"


def pct(values: list[float], p: int) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def parse_combo(combo: str) -> tuple[str, str | None, str]:
    """'openai:HCX-005+supertonic' -> ('openai', 'HCX-005', 'supertonic').

    모델 이름을 조합에 적을 수 있어야 gpt 와 HCX 를 **같은 표에서** 비교할 수 있다.
    안 적으면 설정값(api_model)을 따라가므로 '무엇과 비교했는지'가 흐려진다.
    """
    parts = combo.split("+")
    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"조합은 '<llm>+<tts>' 다: {combo!r}")
    llm, _, model = parts[0].partition(":")
    if not llm:
        raise SystemExit(f"조합은 '<llm>+<tts>' 다: {combo!r}")
    return llm, (model or None), parts[1]


def felt_total(llm_s: float, first_audio_s: float, play_s: float) -> float:
    """아이가 실제로 기다리는 시간.

    🔴 재생 시간을 반드시 더한다. speak() 는 재생이 끝날 때까지 블로킹하므로 아이가
       다음 말을 할 수 있게 되는 시점은 '말이 끝난 뒤'다. 2026-08-19 까지 이 도구의
       합계는 `LLM + 첫 소리` 여서 **답변이 길어지는 대가가 아예 안 잡혔다.**
       하필 그게 클로바 판정의 갈림길이었다 — HCX-005 는 gpt 보다 0.84초 빠른데
       답변의 12.5%가 40자를 넘는다(gpt·EXAONE 은 0건).
    """
    return llm_s + first_audio_s + play_s


def build(llm_backend: str, tts_backend: str, system: str, api_model: str | None = None):
    """조합대로 만들고 예열까지 끝낸 (agent, tts) 를 돌려준다."""
    lcfg = {**settings.models["llm"], "backend": llm_backend}
    if api_model:
        lcfg["api_model"] = api_model
    agent = LLMAgent(model_path=lcfg.pop("model_path"), system_prompt=system, **lcfg)
    tts = TTSModule(**{**settings.models["tts"], "backend": tts_backend})
    agent.warm()          # 로컬 폴백 + API 커넥션
    tts.load()
    tts.warm()
    tts._infer("워밍업")   # 첫 합성은 이상치다(모델 초기화·CUDA 커널)
    return agent, tts


def run_combo(label: str, agent, tts, rows, repeat: int, play: bool, out_dir: Path):
    llm_s, tts_s, play_s, lens = [], [], [], []
    for _ in range(repeat):
        for r in rows:
            t = time.perf_counter()
            reply = agent.respond(r["text"])["text"]
            llm_s.append(time.perf_counter() - t)
            lens.append(len(reply))
            text = reply or "응?"
            if play:
                tm = tts.speak(text)
                tts_s.append(tm.first_audio_s)
                play_s.append(tm.play_s)      # 🔴 말하는 시간. 답변 길이의 대가가 여기 있다
            else:
                t = time.perf_counter()
                tts.synthesize(text, str(out_dir / f"{label}_{r['id']}.wav"))
                tts_s.append(time.perf_counter() - t)
                play_s.append(0.0)
    total = [felt_total(a, b, c) for a, b, c in zip(llm_s, tts_s, play_s)]
    stage = "첫소리" if play else "합성"
    line = (f"  {label:26} LLM {statistics.median(llm_s):6.3f}s | "
            f"TTS({stage}) {statistics.median(tts_s):6.3f}s | ")
    if play:
        line += f"말하기 {statistics.median(play_s):6.3f}s | "
    line += (f"체감 {statistics.median(total):6.3f}s | "
             f"p95 {pct(total, 95):6.3f}s | 답변 {statistics.median(lens):.0f}자 "
             f"(최대 {max(lens)}자)")
    print(line)
    return statistics.median(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combos", default=DEFAULT_COMBOS, help=f"쉼표 구분. 기본 {DEFAULT_COMBOS}")
    ap.add_argument("--eval-set", default="data/eval_set_safety.jsonl")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--play", action="store_true",
                    help="실제로 재생해 '첫 소리까지'를 잰다(소리가 난다). 스트리밍은 이걸로만 측정된다")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  [경고] %(message)s")
    if args.play:      # 재생하려면 운영과 같은 장치 선택이 필요하다
        from app.main import _setup_audio_device
        _setup_audio_device()

    rows = load_eval_set(BASE / args.eval_set)
    system = _augment_system(settings.prompts["system"],
                             load_fewshot(settings.models["llm"].get("fewshot_path")))
    out_dir = BASE / "logs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(rows)}문항 × {args.repeat}회 | 프롬프트 {len(system)}자 | "
          f"{'재생(첫 소리)' if args.play else '합성까지'}\n")
    results = {}
    for combo in args.combos.split(","):
        combo = combo.strip()
        llm_b, api_model, tts_b = parse_combo(combo)
        agent, tts = build(llm_b, tts_b, system, api_model=api_model)
        results[combo] = run_combo(combo, agent, tts, rows,
                                   args.repeat, args.play, out_dir)

    if len(results) > 1:
        best = min(results, key=results.get)
        print(f"\n  가장 빠른 조합: {best} ({results[best]:.3f}s)")
    if not args.play:
        print("\n  ⚠️ 합성까지만 잰 값이다. OpenAI TTS 스트리밍 이득은 --play 로만 보인다.")
    print(f"  wav -> {out_dir}")


if __name__ == "__main__":
    main()
