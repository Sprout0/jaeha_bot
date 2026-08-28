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
from dataclasses import dataclass
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
from tools.eval_llm import (  # noqa: E402
    _RATE_LIMIT, CLOVA_BASE_URL, load_eval_set, required_key)

# 네트워크 왕복 하한을 재기 위한 최소 프롬프트. 내용은 중요하지 않다 — 짧다는 것만 중요하다.
FLOOR_SYSTEM = "너는 다섯 살 아이의 친구 로봇 티드야. 밝은 반말로 한두 문장만 말해."

# 문장 경계는 **TTS 가 실제로 자르는 그 규칙**을 그대로 빌려 온다. 여기서 따로 정의하면
# "첫 문장이 0.6s 에 준비된다"고 재 놓고 정작 TTS 는 다른 자리에서 잘라, 잰 값이
# 운영에서 재현되지 않는다. 규칙이 둘이면 측정은 거짓말이 된다.
from app.tts_module import _SENT_SPLIT  # noqa: E402


@dataclass
class StreamTrace:
    """스트림 한 번을 조각낸 결과.

    ttft_s            첫 **내용 있는** 청크까지 = 왕복 + 프롬프트 처리
    gen_s             첫 내용 청크 -> 마지막 내용 청크 = 순수 생성
    first_sentence_s  TTS 가 첫 문장을 넘겨받을 수 있게 되는 순간(한 문장짜리 답이면 None)
    """

    ttft_s: float | None
    gen_s: float
    first_sentence_s: float | None
    chars: int
    text: str


def summarize_stream(t0: float, events) -> StreamTrace:
    """(시각, delta) 목록 -> 조각난 시간. **네트워크를 안 탄다** — 그래서 시험할 수 있다.

    🔴 빈 delta 는 버린다. 첫 청크는 대개 role 만 담고 content 가 비어 있는데, 그걸
       첫 토큰으로 세면 TTFT 가 과소평가되고 그만큼 생성이 부푼다. 즉 "병목은 생성"
       이라는 **정반대 결론**이 나온다. 이 도구가 답하려는 질문이 딱 그거라 치명적이다.
    ⚠️ 첫 문장 시각은 최대 청크 하나만큼 **늦게** 잡힌다. _SENT_SPLIT 이 종결부호
       **뒤의 공백**으로 가르기 때문에 다음 문장의 첫 글자가 와야 두 조각이 된다.
       늦게 잡히는 쪽이라 스트리밍 이득을 과장하지 않는다 — 그래서 그냥 둔다.
    """
    content = [(t, d) for t, d in events if d]
    if not content:
        return StreamTrace(ttft_s=None, gen_s=0.0, first_sentence_s=None, chars=0, text="")

    buf = ""
    first_sentence = None
    for t, d in content:
        buf += d
        if first_sentence is None and len([p for p in _SENT_SPLIT.split(buf) if p.strip()]) >= 2:
            first_sentence = t - t0
    return StreamTrace(
        ttft_s=content[0][0] - t0,
        gen_s=content[-1][0] - content[0][0],
        first_sentence_s=first_sentence,
        chars=len(buf),
        text=buf,
    )


def is_rate_limited(exc: BaseException) -> bool:
    """429 인가. 판정 낱말은 eval_llm 것을 그대로 쓴다 — 두 도구의 '실패'가 같은 뜻이라야 한다."""
    msg = str(exc).lower()
    return any(k.lower() in msg for k in _RATE_LIMIT)


def reject_unstreamable(models: list[str], stream: bool) -> None:
    """local 은 흘려줄 수 없다. 20분짜리 실행이 끝물에 터지지 않도록 **시작 전에** 막는다."""
    if stream and "local" in models:
        print("--stream 은 원격 팔에만 됩니다. local 을 빼거나 --stream 을 끄세요.")
        sys.exit(1)


def call_once(client, model: str, sysp: str, text: str, max_tokens: int,
              stream: bool = False) -> tuple[float, StreamTrace]:
    """한 번 치고 (총 시간, 조각) 을 돌려준다.

    🔴 stream 을 안 켜면 `stream` 인자를 **아예 넘기지 않는다.** 넘기면(False 라도)
       요청 모양이 달라져 예전 측정과 비교할 수 없다.
    """
    messages = [{"role": "system", "content": sysp}, {"role": "user", "content": text}]
    kw = {"stream": True} if stream else {}

    t0 = time.perf_counter()
    out = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens, messages=messages, **kw)

    if not stream:
        total = time.perf_counter() - t0
        reply = out.choices[0].message.content or ""
        return total, StreamTrace(None, 0.0, None, len(reply), reply)

    events = []
    for chunk in out:
        choices = getattr(chunk, "choices", None)
        delta = ""
        if choices:   # 끝에 choices 가 빈 사용량 청크를 주는 서버가 있다
            delta = getattr(choices[0].delta, "content", None) or ""
        events.append((time.perf_counter(), delta))
    return time.perf_counter() - t0, summarize_stream(t0, events)


def describe_split(rtt: float, ttft_real: float, gen: float, total: float,
                   chars: float, floor_gen: float | None = None) -> list[str]:
    """분해를 사람이 읽을 줄로 만든다. **읽는 법까지 같이 찍는다.**

    🔴 왜 읽는 법이 필요한가: 08-21 의 "생각의 87% 가 왕복"이 잘못 읽은 숫자였다.
       숫자만 던져 놓으면 같은 일이 또 난다. 특히 두 모양을 그냥 두면 안 된다.
       - 프롬프트 대가가 음수: 바닥 팔이 더 느리게 나온 것이다. '프롬프트가 시간을
         줄여준다'가 아니라 **효과가 잰 잡음보다 작다**는 뜻이다.
       - 생성이 0 에 가까움: 서버가 안 흘려주고 한 번에 준 것이다. 그 서버로는
         문장 스트리밍으로 벌 게 없다. 단, **짧은 답은 원래 청크 한둘**이라 제외한다.
    """
    prompt_cost = ttft_real - rtt
    out = [
        f"      왕복      {rtt:6.3f}s ({rtt / total * 100:4.1f}%)  <- [바닥] 팔의 TTFT",
        f"      프롬프트  {prompt_cost:6.3f}s ({prompt_cost / total * 100:4.1f}%)"
        f"  <- 실제 TTFT - 바닥 TTFT",
        f"      생성      {gen:6.3f}s ({gen / total * 100:4.1f}%)",
        f"      총        {total:6.3f}s",
    ]
    if prompt_cost <= 0:
        out.append("      -> 프롬프트 대가가 0 이하 = 잰 잡음보다 작다는 뜻이다."
                   " 시간을 벌어준다는 뜻이 아니다.")
    if gen < 0.05 and chars > 10:
        if floor_gen is not None and floor_gen >= 0.05:
            # [바닥] 팔이 흘렸다면 서버는 흘려줄 줄 안다. 짧은 답을 서버 탓으로 돌리면
            # '클로바로는 문장 스트리밍 불가'라는 없는 제약을 만들어 낸다.
            out.append(f"      -> 생성이 0 에 가깝지만 [바닥] 팔은 흘렸다(생성 {floor_gen:.3f}s)."
                       f" 서버는 흘려준다.")
            out.append(f"         우리 답이 {chars:.0f}자로 짧아 청크가 사실상 하나인 것뿐이다.")
        else:
            out.append("      ⚠️ 생성이 0 에 가깝다 = 서버가 안 흘려주고 한 번에 줬다.")
            out.append("         이 서버로는 문장 스트리밍으로 벌 게 없다.")
    return out


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


def pace(gap: float) -> None:
    """호출 사이에 쉰다. 운영은 연달아 치지 않는다 — 아이가 말하고, 봇이 3.5초 말한다.

    🔴 2026-08-20 에 필요해진 이유: 같은 HCX-005 를 연달아 치면 0.831s, 파이프라인
       벤치(턴 사이에 TTS 재생 3.5s)에서는 1.264s 였다. gpt 는 두 조건에서 같았다.
       클로바만 간격에 반응한다면 커넥션이 식는 것이고, 그건 keep-alive 로 되찾을 수
       있는 0.4초다. 재는 조건이 운영과 다르면 숫자도 다르다.
    """
    if gap > 0:
        time.sleep(gap)


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
    ap.add_argument("--gap", type=float, default=0.0, metavar="SEC",
                    help="호출 사이에 쉬는 시간. 운영은 봇이 3.5초 말하는 동안 호출이 없다. "
                         "기본 0(연달아 치기 — 예전 측정과 비교 가능)")
    ap.add_argument("--stream", action="store_true",
                    help="스트리밍으로 받아 첫 토큰까지(TTFT)와 생성 시간을 가른다. "
                         "기본 꺼짐 — 켜면 요청 모양이 달라져 예전 값과 직접 비교가 안 된다")
    ap.add_argument("--list-models", metavar="PREFIX",
                    help="쓸 수 있는 모델 이름만 찍고 끝낸다(예: HCX)")
    args = ap.parse_args()

    if args.list_models:
        list_models(args.list_models)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    reject_unstreamable(models, args.stream)
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
    ttft: dict[str, list[float]] = {a[0]: [] for a in arms}
    gen: dict[str, list[float]] = {a[0]: [] for a in arms}
    early: dict[str, list[float]] = {a[0]: [] for a in arms}   # 첫 문장으로 당길 수 있는 시간
    rate: dict[str, int] = {a[0]: 0 for a in arms}             # 그중 429
    gap_note = f" · 호출 간격 {args.gap}s" if args.gap else " · 연달아"
    print(f"{len(rows)}문항 × {args.repeat}회 · 팔 {len(arms)}개 교차 · 재시도 끔 · "
          f"프롬프트 {len(system)}자{gap_note}\n")
    for rep in range(args.repeat):
        for i, r in enumerate(rows):
            k = i % len(arms)
            for label, model, sysp in arms[k:] + arms[:k]:   # 문항마다 순서 회전
                pace(args.gap)
                try:
                    total, tr = call_once(clients[model], model, sysp, r["text"],
                                          args.max_tokens, stream=args.stream)
                    lat[label].append(total)
                    lens[label].append(tr.chars)
                    if tr.ttft_s is not None:
                        ttft[label].append(tr.ttft_s)
                        gen[label].append(tr.gen_s)
                        # 첫 문장이 안 닫혔으면(한 문장짜리 답) 스트리밍으로 벌 게 **0** 이다.
                        # None 을 빼고 세면 '2문장짜리 답만' 모아 평균 내는 셈이라 이득이 부푼다.
                        early[label].append(0.0 if tr.first_sentence_s is None
                                            else max(0.0, total - tr.first_sentence_s))
                except Exception as e:
                    fail[label] += 1
                    if is_rate_limited(e):
                        rate[label] += 1
                    print(f"  실패 {label}: {type(e).__name__}"
                          f"{' (429)' if is_rate_limited(e) else ''}", flush=True)
        print(f"  ...{rep + 1}/{args.repeat}회", flush=True)

    print(f"\n{'팔':28} {'n':>4} {'중앙':>8} {'평균':>8} {'p95':>8} {'최대':>8} {'답변':>6} {'실패':>5}")
    for label, _, _ in arms:
        v = lat[label]
        if not v:
            print(f"{label:28} (전부 실패 {fail[label]}회)")
            continue
        print(f"{label:28} {len(v):4} {statistics.median(v):7.3f}s {statistics.mean(v):7.3f}s "
              f"{p95(v):7.3f}s {max(v):7.3f}s {statistics.median(lens[label]):5.0f}자 {fail[label]:5}")

    if any(rate.values()):
        hit = ", ".join(f"{k} {v}회" for k, v in rate.items() if v)
        print(f"\n  ⚠️ 429(분당 한도) {hit} — 테스트 앱 키의 한도다.")
        print("     이 실행의 p95·최대에는 그 대기가 섞여 있다. 중앙값만 믿을 것.")

    if args.stream:
        print(f"\n{'팔':28} {'TTFT중앙':>10} {'생성중앙':>10} {'첫문장절감':>12} {'2문장이상':>11}")
        for label, _, _ in arms:
            if not ttft[label]:
                continue
            e = early[label]
            multi = sum(1 for x in e if x > 0)
            print(f"{label:28} {statistics.median(ttft[label]):9.3f}s "
                  f"{statistics.median(gen[label]):9.3f}s "
                  f"{statistics.median(e):11.3f}s "
                  f"{multi:6}/{len(e):<4}")

        print("\n  생각을 세 조각으로 (각각의 중앙값):")
        for m in models:
            floor_label, real = f"{m} [바닥]", m
            if not ttft.get(floor_label) or not ttft.get(real):
                continue
            rtt = statistics.median(ttft[floor_label])
            t_real = statistics.median(ttft[real])
            g = statistics.median(gen[real])
            tot = statistics.median(lat[real])
            print(f"    {m}")
            floor_gen = statistics.median(gen[floor_label]) if gen[floor_label] else None
            for line in describe_split(rtt, t_real, g, tot, statistics.median(lens[real]),
                                       floor_gen=floor_gen):
                print(line)
        print("\n  ⚠️ 조각은 **각각의 중앙값**이라 서로 더해도 총 중앙값과 딱 맞지 않는다.")
        print("  '첫문장절감' = 총 시간 - 첫 문장이 닫힌 시각. 한 문장짜리 답은 0 으로 센다")
        print("               (빼고 세면 2문장 답만 모아 평균 내는 셈이라 이득이 부푼다).")

    print()
    print("  '[바닥]' = 42자 프롬프트로 \"안녕\"만 물은 값.")
    print("  원격 팔 = 그 서버까지의 왕복 하한. 프롬프트·모델을 손봐도 이 아래로는 못 내려간다.")
    print("           여기가 크면 서버를 옮기는 수밖에 없다.")
    print("  local 팔 = 왕복이 없으니 **프롬프트 길이의 대가**만 남는다. 원격은 길이가")
    print("           지연과 무관했지만(08-18 A/B) 로컬은 prompt eval 을 직접 치른다.")


if __name__ == "__main__":
    main()
