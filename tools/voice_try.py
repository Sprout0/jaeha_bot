"""목소리를 **직접 쳐서** 들어 보고, 그 주변을 훑는 도구. 젯슨에서 돌릴 것.

voice_knockout.py 가 우승자를 뽑아 주면, 그 근처를 손으로 더듬는 게 여기다.
토너먼트는 넓게 훑고(탐색), 이 도구는 좁게 판다(활용).

문법 (app/tts_module.py `_resolve_style`):
    F1                  단일. F1~F5 / M1~M5
    F1+F4               균등 블렌드
    F1:0.7+F4:0.3       가중 블렌드 ('F1:3+F4:1' 처럼 비율로도 된다)
    F3:-0.2+F1:1.2      🔴 외삽 — F1 에서 F3 성분을 **빼는** 방향.
                           어떤 프리셋보다도 더 F1 다운 목소리가 된다.
    F1:0.7+F2:0.3|F4    음색은 왼쪽, 리듬은 오른쪽

🔴 직전에 들은 것과 **바로 비교**할 수 있게 `c` 를 뒀다. 목소리는 혼자 들으면
   좋은지 모른다 — 붙여 들어야 안다(voice_knockout.py 를 짝비교로 만든 이유와 같다).

사용:
    python tools/voice_try.py "F3:-0.2+F1:1.2"
    python tools/voice_try.py            # 현행 목소리부터 시작
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import settings  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402

PRESETS = [f"F{i}" for i in range(1, 6)] + [f"M{i}" for i in range(1, 6)]
DEFAULT_TEXT = "멍멍! 이건 무슨 동물 소리일까? 재하야, 우리 같이 놀자!"
DELTAS = (-0.2, -0.1, 0.1, 0.2)


def _fmt(parts: list[tuple[str, float]]) -> str:
    return "+".join(f"{n}:{round(w, 3)}" for n, w in parts)


def neighbors(spec: str, deltas: tuple[float, ...] = DELTAS) -> list[str]:
    """가중치를 기준값 **주변으로** 조금씩 움직인 후보들.

    🔴 고정된 비율(0.3/0.5/0.7)을 쓰면 안 된다. 기준이 -0.2 인데 0.3 근처를 훑으면
       엉뚱한 데를 뒤지게 된다. 항상 '지금 값에서 ±' 로 움직여야 한다.
    첫 항목의 무게를 옮기고 나머지는 비율을 유지한 채 줄이거나 늘린다(합은 1).
    """
    m = TTSModule()
    timbre, sep, rhythm = spec.partition("|")
    parts = m._parse_blend(timbre)
    if len(parts) < 2:
        return []
    out = []
    old_rest = sum(w for _, w in parts[1:])
    if abs(old_rest) < 1e-9:
        return []
    for d in deltas:
        w0 = parts[0][1] + d
        rest = 1.0 - w0
        new = [(parts[0][0], w0)] + [(n, w / old_rest * rest) for n, w in parts[1:]]
        cand = _fmt(new) + (f"|{rhythm}" if sep else "")
        if cand != spec:
            out.append(cand)
    return out


def rhythm_variants(spec: str) -> list[str]:
    """음색은 그대로 두고 리듬만 바꾼 후보들. '목소리는 좋은데 말투가 어색할' 때."""
    timbre = spec.split("|")[0]
    return [f"{timbre}|{p}" for p in PRESETS if p.startswith("F")]


def play(spec: str, text: str) -> None:
    tts = TTSModule(**{**settings.models["tts"], "backend": "supertonic", "voice": spec})
    tts.load()
    r = tts.speak(text)
    print(f"      (첫 소리 {r.first_audio_s:.2f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?", default=None)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  [경고] %(message)s")
    from app.main import _setup_audio_device
    _setup_audio_device()

    cur = args.spec or str(settings.models["tts"]["voice"])
    prev = None
    print(f'문장: "{args.text}"')
    print("  아무 조합이나 쳐서 들어보세요. 도움말은 ?, 종료는 q\n")
    print(f"  ▶ {cur}")
    play(cur, args.text)

    while True:
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if cmd in ("q", "quit"):
            print(f"\n  마지막 값: {cur}")
            return
        if cmd == "?":
            print(__doc__)
            print(f"  프리셋: {' '.join(PRESETS)}")
            print("  명령: (빈줄)=다시 / n=주변 훑기 / r=리듬만 바꿔보기 / c=직전과 비교 / q=끝")
            continue
        if cmd == "":
            print(f"  ▶ {cur}")
            play(cur, args.text)
            continue
        if cmd == "c":
            if not prev:
                print("  비교할 직전 값이 없다.")
                continue
            for tag, s in (("직전", prev), ("지금", cur)):
                print(f"  ▶ {tag}  {s}")
                play(s, args.text)
            continue
        if cmd in ("n", "r"):
            cands = neighbors(cur) if cmd == "n" else rhythm_variants(cur)
            if not cands:
                print("  이 조합에서는 주변을 만들 수 없다(단일 프리셋).")
                continue
            print(f"  기준: {cur}")
            for i, s in enumerate(cands, 1):
                print(f"  ▶ {i}. {s}")
                play(s, args.text)
            print("  마음에 든 걸 그대로 쳐서 이어가면 된다.")
            continue
        try:
            TTSModule()._parse_blend(cmd.split("|")[0])
        except Exception as e:
            print(f"  못 읽었다({e}). 예: F3:-0.2+F1:1.2  또는  F1:0.7+F2:0.3|F4")
            continue
        prev, cur = cur, cmd
        print(f"  ▶ {cur}")
        try:
            play(cur, args.text)
        except Exception as e:
            print(f"  합성 실패({type(e).__name__}: {e}) — 직전 값으로 되돌린다")
            cur = prev


if __name__ == "__main__":
    main()
