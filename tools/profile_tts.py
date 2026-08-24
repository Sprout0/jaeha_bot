"""TTS 첫 소리 1.09초가 **어디서** 쓰이는지 — 고치기 전에 먼저 잰다.

실행(젯슨):
    python -m tools.profile_tts
    python -m tools.profile_tts --shape-test      # shape 가설만

🔴 왜 재나. 체감 4.49s 중 TTS 첫 소리가 1.09s(24%)다. '동적 shape 때문에 느리다,
   버킷팅하면 47% 준다'는 건 **NVIDIA 문서의 남의 사례**다. 우리 파이프라인에서
   그 전제(=길이가 바뀔 때마다 느려진다)가 참인지부터 확인해야 한다.

Supertonic 은 ONNX 세션 4개를 쓴다:
    dp          지속시간 예측      1회
    text_enc    텍스트 인코더      1회
    vector_est  확산              **total_steps 회**(현재 12)
    vocoder     보코더             1회
확산이 12번이라 여기가 지배적일 가능성이 크다. 그렇다면 손댈 곳은 shape 이 아니라
스텝 수이고, 버킷팅은 애초에 무의미하다 — 그 판단을 이 도구로 내린다.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict


# 길이를 골고루: 실제 답변은 18~33자다(2026-08-10 실측)
SENTENCES = [
    "응!",
    "좋아!",
    "그건 파란색이야.",
    "우리 같이 놀자!",
    "고양이는 야옹 하고 울어.",
    "재하야, 오늘 뭐 하고 놀았어?",
    "칼은 위험해! 엄마 아빠한테 물어보자!",
    "빨간 공은 동그랗고 말랑말랑해. 한번 만져볼래?",
]


def _instrument(model) -> dict:
    """네 세션의 run 을 감싸 시간을 모은다. 원본 동작은 안 건드린다."""
    took: dict[str, list[float]] = defaultdict(list)
    for name in ("dp_ort", "text_enc_ort", "vector_est_ort", "vocoder_ort"):
        sess = getattr(model, name)
        if getattr(sess, "_profiled", False):
            continue
        orig = sess.run

        def wrapped(*a, _orig=orig, _key=name, **kw):
            t = time.perf_counter()
            out = _orig(*a, **kw)
            took[_key].append(time.perf_counter() - t)
            return out

        sess.run = wrapped
        sess._profiled = True
    return took


def _build():
    from app.config import settings
    from app.tts_module import TTSModule

    tts = TTSModule(**settings.models.get("tts", {}))
    tts.load()
    return tts


def _row(name: str, xs: list[float], total: float) -> str:
    if not xs:
        return f"  {name:14s}      —"
    s = sum(xs)
    return (f"  {name:14s} {s * 1000:8.1f}ms  {s / total:6.1%}   "
            f"({len(xs)}회, 1회 {statistics.median(xs) * 1000:6.1f}ms)")


def stage_profile(tts, repeats: int) -> None:
    model = tts._tts.model
    print(f"확산 스텝 {tts.total_steps} / 프로바이더 {tts.providers}")
    print()
    print("문장별 내역 (합성만. 재생·리샘플 제외)")
    print(f"  {'문장':28s} {'합계':>9s}  " + "  ".join(
        f"{k:>9s}" for k in ("dp", "text_enc", "확산x" + str(tts.total_steps), "vocoder")))

    took = _instrument(model)          # 한 번만 감싼다 — 감싼 놈이 이 dict 를 채운다
    per_stage: dict[str, list[float]] = defaultdict(list)
    wall: list[float] = []
    for s in SENTENCES:
        for i in range(repeats):
            took.clear()
            t = time.perf_counter()
            tts._infer_local(s)
            w = time.perf_counter() - t
            wall.append(w)
            for k, v in took.items():
                per_stage[k].append(sum(v))
            if i == repeats - 1:
                short = (s[:26] + "..") if len(s) > 26 else s
                print(f"  {short:28s} {w * 1000:8.1f}ms  "
                      + "  ".join(f"{sum(took[k]) * 1000:8.1f}ms" for k in
                                  ("dp_ort", "text_enc_ort", "vector_est_ort",
                                   "vocoder_ort")))

    total = sum(wall)
    print()
    print("합계 내역")
    for k, label in (("dp_ort", "dp"), ("text_enc_ort", "text_enc"),
                     ("vector_est_ort", "확산(전체)"), ("vocoder_ort", "vocoder")):
        print(_row(label, per_stage[k], total))
    covered = sum(sum(per_stage[k]) for k in per_stage)
    print(f"  {'그 외(파이썬·G2P·노이즈)':14s} {(total - covered) * 1000:8.1f}ms  "
          f"{(total - covered) / total:6.1%}")
    print(f"  {'─' * 40}")
    print(f"  {'합성 전체':14s} {total * 1000:8.1f}ms  "
          f"(문장 1개당 중앙 {statistics.median(wall) * 1000:.1f}ms)")


def shape_test(tts, repeats: int) -> None:
    """🔴 버킷팅의 전제 검증 — 길이가 바뀌면 정말 느려지나?

    ⚠️ **문장을 자기 자신과 비교해야 한다.** 서로 다른 문장끼리 재면 '길이가 길어서
       느린 것'과 '모양이 바뀌어서 느린 것'이 섞여 구분이 안 된다. 그래서 A/B 두
       문장을 정하고, 각각을
         (가) 연달아만 낸다            -> 모양이 고정된다
         (나) A,B,A,B 로 번갈아 낸다    -> 매번 새 모양이다
       같은 문장의 (가) 대비 (나) 차이가 순수한 '모양 바뀜' 값이다.
    """
    a, b = "응!", "빨간 공은 동그랗고 말랑말랑해. 한번 만져볼래?"

    def timed(s: str) -> float:
        t = time.perf_counter()
        tts._infer_local(s)
        return time.perf_counter() - t

    stable: dict[str, list[float]] = {a: [], b: []}
    for s in (a, b):
        for _ in range(3):
            timed(s)                                   # 그 모양으로 예열
        for _ in range(repeats * 3):
            stable[s].append(timed(s))

    alt: dict[str, list[float]] = {a: [], b: []}
    timed(a)
    for _ in range(repeats * 3):                       # 번갈아 — 매 호출이 새 모양
        alt[a].append(timed(a))
        alt[b].append(timed(b))

    print(f"  {'문장':30s} {'연달아':>9s} {'번갈아':>9s} {'차이':>9s}")
    for s in (a, b):
        st = statistics.median(stable[s]) * 1000
        al = statistics.median(alt[s]) * 1000
        short = (s[:26] + "..") if len(s) > 26 else s
        print(f"  {short:30s} {st:7.1f}ms {al:7.1f}ms {al - st:+8.1f}ms")
    print(f"  (n = 각 {repeats * 3}회, 예열 후 중앙값)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TTS 단계별 시간 프로파일")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--shape-test", action="store_true",
                   help="동적 shape 가설만 검증")
    a = p.parse_args(argv)

    tts = _build()
    tts._infer_local("예열")          # 첫 호출은 CUDA 초기화가 섞여 못 쓴다
    tts._infer_local("예열 한 번 더")

    if a.shape_test:
        print("동적 shape 가 실제로 비용인가")
        shape_test(tts, a.repeats)
    else:
        stage_profile(tts, a.repeats)
        print()
        print("동적 shape 가 실제로 비용인가")
        shape_test(tts, a.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
