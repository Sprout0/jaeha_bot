"""백엔드 조합별 응답 지연 측정. LLM·TTS 후보가 바뀔 때마다 같은 잣대로 재려고 만들었다.

실제 설정(configs/)·실제 시스템 프롬프트·실제 평가셋을 쓴다 — 손으로 만든 문장으로 재면
운영과 다른 숫자가 나온다. 모델 로드와 커넥션은 재기 전에 예열한다(첫 호출은 이상치다).

    python tools/bench_pipeline.py                          # 기본 3조합
    python tools/bench_pipeline.py --combos openai+openai
    python tools/bench_pipeline.py --play                   # 실제 재생 + 말하는 시간까지

조합 표기는 `<llm>[:모델][/문장상한]+<tts>` — llm 은 local|openai, tts 는 supertonic|openai.
`:모델`을 붙이면 그 모델로 고정한다 — gpt 와 HCX 를 같은 표에서 비교할 때:
    --combos "openai:gpt-4o-mini+supertonic,openai:HCX-005+supertonic" --play
`/문장상한`을 붙이면 그 팔만 답변 길이를 조인다 — 말하기 시간은 글자 수에 정확히
비례하므로(2026-08-28 실측 0.167~0.169초/자, 모델 무관), 길이를 맞추면 뒤집히는지:
    --combos "openai:HCX-005/1+supertonic,openai:gpt-4o-mini+supertonic" --play

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
import gc
import logging
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.agent import LLMAgent, build_prefix, load_fewshot  # noqa: E402
from app.config import settings  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402
from tools.eval_llm import load_eval_set  # noqa: E402

DEFAULT_COMBOS = "local+supertonic,openai+supertonic,openai+openai"


def pct(values: list[float], p: int) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def parse_combo(combo: str) -> tuple[str, str | None, int | None, str]:
    """'openai:HCX-005/1+supertonic' -> ('openai', 'HCX-005', 1, 'supertonic').

    표기는 `<llm>[:<모델>][/<문장상한>]+<tts>` 다.

    모델 이름을 조합에 적을 수 있어야 gpt 와 HCX 를 **같은 표에서** 비교할 수 있다.
    안 적으면 설정값(api_model)을 따라가므로 '무엇과 비교했는지'가 흐려진다.

    🔴 문장 상한이 팔마다 필요한 이유 (2026-08-28 실측): HCX 가 LLM 에서 0.40초를
       벌고 말하기에서 1.06초를 도로 뱉었다. 그런데 글자당 말하기 속도는 네 팔 모두
       0.167~0.169s 로 같았다 — TTS 가 느린 게 아니라 HCX 가 글자를 더 쓴 것이다.
       남는 질문은 "길이를 맞추면 뒤집히는가" 하나이고, 그걸 재려면 **같은 실행 안에서**
       팔마다 상한을 다르게 걸 수 있어야 한다(실행을 나누면 회선 변화가 섞인다).
    """
    parts = combo.split("+")
    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"조합은 '<llm>+<tts>' 다: {combo!r}")
    left, _, cap_text = parts[0].partition("/")
    llm, _, model = left.partition(":")
    if not llm:
        raise SystemExit(f"조합은 '<llm>+<tts>' 다: {combo!r}")
    cap = None
    if "/" in parts[0]:
        # 조용히 무시하면 '길이를 걸었다'고 믿는데 안 걸린 값이 표에 들어간다.
        if not cap_text.isdigit() or int(cap_text) < 1:
            raise SystemExit(f"문장 상한은 1 이상 정수다: {combo!r}")
        cap = int(cap_text)
    return llm, (model or None), cap, parts[1]


DEFAULT_LENGTH_HINT = """⚠️ 길이 규칙을 다시 확인한다. 한 문장으로, 스무 글자를 넘지 않게 답한다.
스무 글자를 넘으면 실패다. 되묻는 말을 붙이고 싶으면 그것까지 스무 글자 안에 넣는다."""


def apply_length_hint(system: str, hint: str | None) -> str:
    """길이 지시를 시스템 프롬프트 **끝에** 덧붙인다.

    🔴 왜 필요한가 (2026-08-28): 길이 지시는 이미 프롬프트에 있다
       ("스무 글자 안팎으로, 길어도 두 문장까지만"). gpt 는 20자로 지키고
       **HCX 는 27~32자에 최대 74~97자로 안 지킨다.** 그래서 재려는 건
       '길이 유도를 넣으면'이 아니라 **'지시를 강화하면 따르는가'** 다.
    ⚠️ 반드시 끝에 붙인다. 앞에 끼우면 원문 규칙 사이를 갈라 놓는다.
    """
    if not (hint or "").strip():
        return system
    return f"{system}\n\n{hint.strip()}"


def felt_total(llm_s: float, first_audio_s: float, play_s: float) -> float:
    """아이가 실제로 기다리는 시간.

    🔴 재생 시간을 반드시 더한다. speak() 는 재생이 끝날 때까지 블로킹하므로 아이가
       다음 말을 할 수 있게 되는 시점은 '말이 끝난 뒤'다. 2026-08-19 까지 이 도구의
       합계는 `LLM + 첫 소리` 여서 **답변이 길어지는 대가가 아예 안 잡혔다.**
       하필 그게 클로바 판정의 갈림길이었다 — HCX-005 는 gpt 보다 0.84초 빠른데
       답변의 12.5%가 40자를 넘는다(gpt·EXAONE 은 0건).
    """
    return llm_s + first_audio_s + play_s


def build(llm_backend: str, tts_backend: str, system: str, api_model: str | None = None,
          max_sentences: int | None = None):
    """조합대로 만들고 예열까지 끝낸 (agent, tts) 를 돌려준다.

    🔴 `system` 은 **예시가 붙기 전의 원본**이어야 한다. LLMAgent 가 fewshot_path 를
       들고 있어서 예시를 스스로 붙인다 — 여기에 이미 붙은 것을 넘기면 **두 번 실린다.**
       2026-08-31 에 발견: 그동안 4,182자인 줄 알았던 프롬프트가 실제로는 4,981자였다.
    """
    lcfg = {**settings.models["llm"], "backend": llm_backend}
    if api_model:
        lcfg["api_model"] = api_model
    if max_sentences is not None:
        lcfg["max_sentences"] = max_sentences
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


def measure_combo(combo: str, system: str, rows, repeat: int, play: bool, out_dir):
    """조합 하나를 재고 **쓴 것을 놓고** 돌아온다.

    🔴 놓는 게 이 함수의 존재 이유다 (2026-08-28 실측). build() 가 조합마다
       TTSModule 을 새로 만드는데, 앞 것을 놓기 전에 만들면 TRT 엔진 두 개가 겹친다
       (하나에 ~2.9GB, 젯슨은 8GB). 실제로 팔 3번째부터 TRT 가 236MB 할당에 실패해
       CUDA 세션으로 폴백했고 그 팔들의 '첫 소리'가 0.73s -> 1.43s 로 두 배가 됐다.
    ⚠️ 조용히 느려진다 — 표에는 그냥 '그 조합이 느리다'로 찍힌다.
    """
    llm_b, api_model, cap, tts_b = parse_combo(combo)
    agent, tts = build(llm_b, tts_b, system, api_model=api_model, max_sentences=cap)
    try:
        return run_combo(combo, agent, tts, rows, repeat, play, out_dir)
    finally:
        del agent, tts
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combos", default=DEFAULT_COMBOS, help=f"쉼표 구분. 기본 {DEFAULT_COMBOS}")
    ap.add_argument("--eval-set", default="data/eval_set_safety.jsonl")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--play", action="store_true",
                    help="실제로 재생해 '첫 소리까지'를 잰다(소리가 난다). 스트리밍은 이걸로만 측정된다")
    ap.add_argument("--length-hint", nargs="?", const=DEFAULT_LENGTH_HINT, default=None,
                    metavar="문구",
                    help="시스템 프롬프트 끝에 길이 지시를 덧붙인다(모든 팔에 똑같이). "
                         "값을 안 주면 기본 문구를 쓴다. 길이 지시는 이미 프롬프트에 있는데 "
                         "HCX 만 안 지켜서, 강화하면 따르는지 보려는 것이다")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  [경고] %(message)s")
    if args.play:      # 재생하려면 운영과 같은 장치 선택이 필요하다
        from app.main import _setup_audio_device
        _setup_audio_device()

    rows = load_eval_set(BASE / args.eval_set)
    # 예시는 붙이지 않는다 — LLMAgent 가 붙인다(build() 주석 참조). 길이 지시만 얹는다.
    system = apply_length_hint(settings.prompts["system"], args.length_hint)
    out_dir = BASE / "logs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 실제로 나가는 앞머리 크기를 찍는다(운영과 같은 조립 = build_prefix).
    llm_cfg = settings.models["llm"]
    prefix = build_prefix(system, load_fewshot(llm_cfg.get("fewshot_path")),
                          llm_cfg.get("fewshot_form", "text"))
    sent_chars = sum(len(m["content"]) for m in prefix)

    print(f"{len(rows)}문항 × {args.repeat}회 | 프롬프트 {sent_chars}자"
          f"({llm_cfg.get('fewshot_form', 'text')}) | "
          f"{'재생(첫 소리)' if args.play else '합성까지'}"
          f"{' | 길이 지시 강화' if args.length_hint else ''}\n")
    if args.length_hint:
        print(f"  덧붙인 지시: {args.length_hint.strip()}\n")
    results = {}
    for combo in args.combos.split(","):
        combo = combo.strip()
        results[combo] = measure_combo(combo, system, rows,
                                       args.repeat, args.play, out_dir)

    if len(results) > 1:
        best = min(results, key=results.get)
        print(f"\n  가장 빠른 조합: {best} ({results[best]:.3f}s)")
    if not args.play:
        print("\n  ⚠️ 합성까지만 잰 값이다. OpenAI TTS 스트리밍 이득은 --play 로만 보인다.")
    print(f"  wav -> {out_dir}")


if __name__ == "__main__":
    main()
