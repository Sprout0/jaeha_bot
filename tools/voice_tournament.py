"""재하봇 목소리를 **귀로** 정하는 도구. 반드시 젯슨에서 돌릴 것.

🔴 왜 단계로 나누나 — 조합이 많아서다.
   F1~F5 만 놓고도 단일 5 + 균등쌍 10 + 가중쌍 20 + 음색/리듬 분리 20 = 55가지다.
   55개를 연달아 들으면 20번째쯤부터 귀가 무뎌지고, 40번째에는 3번이 어땠는지
   기억이 안 난다. 그렇게 고른 값은 못 믿는다.
   그래서 **단계마다 5~9개만 듣고 좁힌다.** 각 단계는 한 축만 바꾼다:

     1단계  음색 고르기   F1~F5 단일 5개              -> 마음에 든 것 2~3개
     2단계  섞기         고른 것들의 균등·가중 블렌드   -> 1개
     3단계  리듬 바꾸기   음색은 고정, 리듬만 F1~F5    -> 1개  (여기가 유아용의 핵심)
     4단계  시드         같은 스타일의 다른 캐릭터      -> 1개
     5단계  속도         0.95 ~ 1.15                 -> 최종

🔴 반드시 실기(ReSpeaker)로 들을 것. 파일로 고르면 2026-08-10 을 반복한다 —
   그때 노트북 wav 로 coral 을 골랐다가 실기에서 뒤집혔다(ReSpeaker 는 16kHz 전용이라
   고역이 통째로 잘린다. 좋게 들리게 하던 성분이 바로 거기 있었다).

⚠️ 기본 문장에 의성어를 넣어 뒀다. 동물 소리 놀이가 핵심이라 '멍멍'이 어색하면
   나머지가 아무리 좋아도 못 쓴다.

사용:
    python tools/voice_tournament.py                 # 1단계부터
    python tools/voice_tournament.py --stage 3       # 이어서
    python tools/voice_tournament.py --blind         # 이름 감추고 듣기(선입견 제거)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import settings  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402

STATE = BASE / "logs" / "voice_tournament.json"
PRESETS = ["F1", "F2", "F3", "F4", "F5"]
DEFAULT_TEXT = "멍멍! 이건 무슨 동물 소리일까? 재하야, 우리 같이 놀자!"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def build_explore(rnd: int, base: str, s: dict) -> tuple[str, list[dict]]:
    """2단계에서 고른 후보 **주변**을 더 넓게 듣는다. 라운드마다 한 축만 바꾼다.

    2단계는 고른 프리셋들 사이의 '중간 지점'만 훑는다. 그런데 정답이 그 사이에 없고
    **바깥**에 있을 수 있다 — 스타일이 선형 공간이라 프리셋 밖으로 나가는 게 가능하다.

      라운드 1  같은 두 목소리를 더 밀어붙인다(미세 비율 + 외삽)
      라운드 2  제3의 목소리를 소량 섞는다(F 나머지 + M 계열)

    🔴 외삽이란: 'F1:-0.2+F4:1.2' 는 F4 에서 F1 성분을 **빼는** 방향이라
       **어떤 프리셋보다도 더 F4 다운** 목소리가 나온다. 더 높고 더 아이 같은 쪽으로
       밀어붙일 때 쓴다. 다만 과하면 소리가 깨지므로 조금씩 늘려 본다.
    🔴 M 을 섞는 건 남성 목소리를 쓰자는 게 아니다. F 프리셋끼리는 다 얇아서
       안 나오는 '두께'가 소량(15~20%)에서 생긴다.
    """
    seed = s.get("seed", settings.models["tts"].get("seed", 777))
    speed = s.get("speed", settings.models["tts"].get("speed", 1.05))

    def mk(specs, desc):
        return (desc, [{"label": v, "voice": v, "seed": seed, "speed": speed}
                       for v in specs])

    parts = [p.split(":") for p in base.split("|")[0].split("+")]
    names = [p[0] for p in parts]

    if rnd == 1:
        # 🔴 고정 비율(0.15/0.2/0.4)로 만들면 안 된다. 기준이 'F3:-0.2+F1:1.2' 처럼
        #    외삽 지점일 때 0.3 근처를 훑게 되고, 그건 우승한 자리에서 **멀어지는** 쪽만
        #    들려주는 것이다. 주변 탐색은 언제나 '지금 값에서 ±' 여야 한다.
        from tools.voice_try import neighbors

        specs = [base] + neighbors(base, (-0.45, -0.15, 0.15, 0.45))  # 1번은 대조군
        # 폭을 0.45 까지 잡는 이유: 기준이 안쪽(예: 0.3)일 때도 한 칸은 **프리셋 밖으로**
        # 넘어가야 한다. 좁게만 잡으면 사이만 훑다 끝난다.
        return mk(specs, f"탐색1 · 더 밀어붙이기 — [{base}] 의 비율을 좌우로 넓힌다. "
                         "음수 가중치 = 그 성분을 빼는 방향이라 더 극단적인 목소리가 나온다")

    if rnd == 2:
        third = [p for p in PRESETS if p not in names] + ["M1", "M2", "M3", "M4", "M5"]
        specs = [base] + [f"{base}+{v}:0.2" for v in third]
        return mk(specs, f"탐색2 · 제3의 목소리 소량 섞기 — [{base}] 에 20%씩. "
                         "M 계열은 남성 목소리가 아니라 없던 '두께'를 얻으려는 것이다")

    raise SystemExit(f"탐색 라운드는 1~2 다: {rnd}")


def build_stage(stage: int, s: dict, base: str | None = None) -> tuple[str, list[dict]]:
    """(단계 설명, 후보 목록). 후보는 {label, voice, seed, speed}.

    `base` 를 주면 3~5단계가 저장된 결과 대신 그 목소리를 기준으로 삼는다.
    블라인드 토너먼트(voice_knockout.py)로 우승자를 뽑았을 때 그 값은 이 도구의
    상태 파일에 없기 때문이다 — 없으면 상태 파일을 손으로 고치는 수밖에 없었다.
    """
    base_seed = s.get("seed", settings.models["tts"].get("seed", 777))
    base_speed = s.get("speed", settings.models["tts"].get("speed", 1.05))

    if stage == 1:
        return ("1단계 · 음색 고르기 — 어느 목소리가 가장 아이 같은가",
                [{"label": v, "voice": v, "seed": base_seed, "speed": base_speed}
                 for v in PRESETS])

    if stage == 2:
        picks = s.get("stage1") or PRESETS[:3]
        cands = []
        for a, b in combinations(picks, 2):
            for spec in (f"{a}+{b}", f"{a}:0.7+{b}:0.3", f"{a}:0.3+{b}:0.7"):
                cands.append(spec)
        if len(picks) >= 3:
            cands.append("+".join(picks))
        cands = [c for c in picks] + cands          # 단일도 다시 비교군으로 남긴다
        return (f"2단계 · 섞기 — {', '.join(picks)} 의 비율을 바꿔 가며",
                [{"label": c, "voice": c, "seed": base_seed, "speed": base_speed}
                 for c in cands])

    if stage == 3:
        win = base or s.get("stage2") or s.get("stage1", ["F1"])[0]
        timbre = win.split("|")[0]
        return (f"3단계 · 리듬 바꾸기 — 음색은 [{timbre}] 그대로, 말의 리듬만 바꾼다\n"
                "        (밝은 음색은 좋은데 말이 촐랑거린다 싶으면 여기서 잡힌다)",
                [{"label": f"{timbre}|{r}", "voice": f"{timbre}|{r}",
                  "seed": base_seed, "speed": base_speed}
                 for r in [timbre] + [p for p in PRESETS if p != timbre]])

    if stage == 4:
        win = base or s.get("stage3") or s.get("stage2") or "F1"
        return (f"4단계 · 시드 — 같은 스타일인데 캐릭터가 달라진다 (voice={win})",
                [{"label": f"seed {sd}", "voice": win, "seed": sd, "speed": base_speed}
                 for sd in (777, 1234, 42, 7, 2024)])

    if stage == 5:
        win = base or s.get("stage3") or s.get("stage2") or "F1"
        sd = s.get("stage4_seed", base_seed)
        return (f"5단계 · 속도 — 2세는 빠르면 못 알아듣는다 (voice={win}, seed={sd})",
                [{"label": f"speed {sp}", "voice": win, "seed": sd, "speed": sp}
                 for sp in (0.95, 1.0, 1.05, 1.10, 1.15)])

    raise SystemExit(f"단계는 1~5 다: {stage}")


def play(cands: list[dict], text: str, idx: int, blind: bool) -> None:
    c = cands[idx]
    name = f"#{idx + 1}" if blind else f"#{idx + 1}  {c['label']}"
    print(f"  ▶ {name}", flush=True)
    tts = TTSModule(**{**settings.models["tts"], "backend": "supertonic",
                       "voice": c["voice"], "seed": c["seed"], "speed": c["speed"]})
    tts.load()
    r = tts.speak(text)
    print(f"      (첫 소리 {r.first_audio_s:.2f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--explore", type=int, metavar="N",
                    help="2단계 후보 '주변'을 더 넓게 듣는다(1=비율·외삽, 2=제3의 목소리)")
    ap.add_argument("--from", dest="base", metavar="SPEC",
                    help="기준이 될 목소리(예: 'F2:0.85+F5:0.15'). --explore 와 3~5단계에 쓴다. "
                         "없으면 저장된 결과를 따라간다")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--blind", action="store_true",
                    help="이름을 감추고 순서도 섞는다 — 'F1 이 좋다더라'는 선입견을 없앤다")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  [경고] %(message)s")
    from app.main import _setup_audio_device
    _setup_audio_device()

    s = load_state()
    if args.explore:
        base = args.base or s.get("stage2")
        if not base:
            raise SystemExit("탐색 기준이 없다. --from 'F1:0.3+F4:0.7' 처럼 줄 것")
        desc, cands = build_explore(args.explore, base, s)
    else:
        desc, cands = build_stage(args.stage, s, base=args.base)
    if args.blind:
        random.shuffle(cands)

    print(f"\n{desc}")
    print(f'문장: "{args.text}"')
    print(f"후보 {len(cands)}개\n")
    for i in range(len(cands)):
        play(cands, args.text, i, args.blind)

    print("\n  숫자        다시 듣기 (예: 3)")
    print("  a           전부 다시")
    print("  ok 2,4      마음에 든 번호 고르기(1단계는 2~3개, 나머지는 1개)")
    print("  q           그만두기")
    while True:
        try:
            cmd = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if cmd == "q":
            return
        if cmd == "a":
            for i in range(len(cands)):
                play(cands, args.text, i, args.blind)
            continue
        if cmd.isdigit() and 1 <= int(cmd) <= len(cands):
            play(cands, args.text, int(cmd) - 1, args.blind)
            continue
        if cmd.startswith("ok"):
            nums = [int(x) for x in cmd[2:].replace(",", " ").split() if x.isdigit()]
            chosen = [cands[n - 1] for n in nums if 1 <= n <= len(cands)]
            if not chosen:
                print("  번호를 못 읽었다. 예: ok 2,4")
                continue
            for c in chosen:
                print(f"    고름: {c['label']}  (voice={c['voice']} seed={c['seed']} speed={c['speed']})")
            if args.explore:
                s["stage2"] = chosen[0]["voice"]
                save_state(s)
                print(f"  저장됨 -> {STATE}")
                print("  더 넓게: --explore 2   /  이대로 진행: --stage 3")
                return
            if args.stage == 1:
                s["stage1"] = [c["voice"] for c in chosen]
            elif args.stage == 2:
                s["stage2"] = chosen[0]["voice"]
            elif args.stage == 3:
                s["stage3"] = chosen[0]["voice"]
            elif args.stage == 4:
                s["stage4_seed"] = chosen[0]["seed"]
                s["seed"] = chosen[0]["seed"]
            elif args.stage == 5:
                s["speed"] = chosen[0]["speed"]
            save_state(s)
            print(f"\n  저장됨 -> {STATE}")
            if args.stage < 5:
                print(f"  다음: python tools/voice_tournament.py --stage {args.stage + 1}")
            else:
                v = s.get("stage3") or s.get("stage2") or "F1"
                print("\n  ===== configs/model_paths.yaml 의 tts: 에 넣을 값 =====")
                print(f"    voice: {v}")
                print(f"    seed: {s.get('seed')}")
                print(f"    speed: {s.get('speed')}")
            return
        print("  숫자 / a / ok 2,4 / q")


if __name__ == "__main__":
    main()
