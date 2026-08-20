"""목소리를 **블라인드 토너먼트**로 고른다. 반드시 젯슨에서 돌릴 것.

🔴 왜 이걸 따로 만들었나 — "이게 좋은가?"는 사람이 잘 못 답한다.
   voice_tournament.py 는 후보를 늘어놓고 절대평가를 시킨다. 그런데 목소리처럼
   미묘한 것은 **혼자 들으면 좋은지 나쁜지 판단이 안 선다.** 반면 "A 와 B 중 어느 쪽?"은
   훨씬 정확하게 답할 수 있다(짝비교가 절대평가보다 신뢰도가 높다는 건 잘 알려진 사실이다).

   그래서 여기서는:
     - 후보를 **무작위로** 만들어 (사람이 못 떠올리는 조합까지 들어온다)
     - **이름을 감추고** A/B/C 로만 들려주고 ('F1 이 좋다더라'는 선입견 제거)
     - **한 판에 2~3개**만 비교해 이긴 것만 다음 판으로 올린다
   판마다 답할 것은 딱 하나 — "어느 쪽이 더 아이 목소리 같은가".

🔴 현행 목소리가 대조군으로 몰래 섞여 있다. 블라인드에서 현행이 이기면
   '바꿀 이유가 없다'가 결론이다 — 그것도 결과다.

⚠️ '구분이 안 된다'도 정직한 답이다. `t` 를 누르면 무작위로 하나를 올리고 그 판을
   '무승부'로 기록한다. 무승부가 많으면 **Supertonic 화자들이 애초에 서로 비슷하다**는
   뜻이고, 그건 다른 TTS 를 찾아야 한다는 근거가 된다.

🔴 반드시 실기(ReSpeaker)로 들을 것. 파일로 고르면 2026-08-10 을 반복한다.

사용:
    python tools/voice_knockout.py               # 15명 3파전 (약 22번 듣고 8번 선택)
    python tools/voice_knockout.py --pool 9 --k 2  # 더 짧게, 이지선다
    python tools/voice_knockout.py --seed 5       # 다른 후보 뽑기(재현 가능)
    python tools/voice_knockout.py --specs-file data/voice_candidates.txt --k 2
                                                 # 손으로 고른 목록을 짝비교로

후보 문법: 목소리는 app/tts_module.py 와 같고, 뒤에 '@시드'를 붙일 수 있다.
    F2                  설정된 시드 그대로
    F2@1234             같은 목소리, 시드만 다르게
    F2:0.85+F5:0.15|F1@42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import settings  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402

STATE = BASE / "logs" / "voice_knockout.json"
FEMALE = ["F1", "F2", "F3", "F4", "F5"]
MALE = ["M1", "M2", "M3", "M4", "M5"]
DEFAULT_TEXT = "멍멍! 이건 무슨 동물 소리일까? 재하야, 우리 같이 놀자!"


def random_spec(rng: random.Random) -> str:
    """무작위 목소리 하나. 아무렇게나 뽑으면 못 쓸 소리가 많이 나오므로 범위를 좁힌다.

    - 뼈대는 여성 프리셋 1~2개 (재하봇은 아이 친구 캐릭터다)
    - 가끔 제3의 목소리를 소량 (M 계열 포함 — 두께를 얻는 용도, 성별을 바꾸는 게 아니다)
    - 가끔 외삽 (프리셋 '사이'에만 답이 있다는 보장이 없다)
    - 가끔 음색/리듬 분리 (밝은 음색 + 차분한 리듬 같은 조합)
    """
    a, b = rng.sample(FEMALE, 2)
    if rng.random() < 0.2:                       # 단일 프리셋도 후보로 남긴다
        spec = a
    else:
        wa = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        if rng.random() < 0.2:                   # 외삽: 한쪽 성분을 빼는 방향
            wa = rng.choice([-0.1, -0.2, -0.3])
        spec = f"{a}:{round(wa, 2)}+{b}:{round(1 - wa, 2)}"
    if rng.random() < 0.3:
        third = rng.choice(FEMALE + MALE)
        if third not in spec:
            spec += f"+{third}:{rng.choice([0.1, 0.15, 0.2])}"
    if rng.random() < 0.3:
        spec += f"|{rng.choice(FEMALE)}"
    return spec


def make_pool(n: int, rng: random.Random, control: str,
              specs: list[str] | None = None) -> list[str]:
    """후보 n명. 현행 목소리를 대조군으로 반드시 넣고 위치는 섞는다.

    `specs` 를 주면 무작위 대신 그 목록을 쓴다(n 은 무시). 사람이 "F1 과 F4 가
    좋더라"까지 좁힌 뒤에는 무작위 후보가 오히려 방해가 되기 때문이다.
    🔴 손으로 고른 목록에도 **대조군은 그대로 넣는다.** 블라인드에서 현행이 이기면
       '바꿀 이유가 없다'가 결론이고 그것도 결과다 — 내 목록만 돌리면 그 결론이
       나올 길 자체가 사라진다.
    """
    pool = {control}
    if specs is not None:
        pool.update(s.strip() for s in specs if s.strip())
    else:
        guard = 0
        while len(pool) < n and guard < n * 50:
            pool.add(random_spec(rng))
            guard += 1
    pool = list(pool)
    rng.shuffle(pool)
    return pool


def make_heats(pool: list[str], k: int) -> list[list[str]]:
    """한 판에 k 명씩 나눈다. 1 명만 남으면 부전승(듣지 않고 올린다)."""
    return [pool[i:i + k] for i in range(0, len(pool), k)]


def parse_spec(entry: str) -> tuple[str, int | None]:
    """'F2@1234' -> ('F2', 1234). 시드가 없으면 (목소리, None).

    🔴 시드는 조합 문자열로 표현이 안 되는 축이라 여태 이 도구에 못 태웠고, 그래서
       시드 고르기는 voice_tournament 의 '5개 늘어놓고 고르기'(절대평가)로 갈 수밖에
       없었다. 그 방식은 2026-08-18 에 이미 실패했다 — 목소리처럼 미묘한 건 붙여
       들어야 판단이 선다. 뒤에 '@시드'를 붙여 같은 짝비교 절차에 태운다.
    ⚠️ 이 문법은 이 도구 안에서만 산다. app/tts_module.py 의 목소리 문법은 안 건드린다.
       '|'(리듬)와 섞여도 '@'가 항상 뒤이므로 오른쪽에서 한 번만 자르면 된다.
    """
    voice, sep, seed = entry.rpartition("@")
    if not sep:
        return entry.strip(), None
    try:
        return voice.strip(), int(seed.strip())
    except ValueError:
        raise ValueError(f"시드는 정수여야 한다: {entry!r}") from None


def play(spec: str, text: str, tag: str) -> None:
    print(f"  ▶ {tag}", flush=True)
    voice, seed = parse_spec(spec)
    cfg = {**settings.models["tts"], "backend": "supertonic", "voice": voice}
    if seed is not None:
        cfg["seed"] = seed
    tts = TTSModule(**cfg)
    tts.load()
    tts.speak(text)


def run_heat(heat: list[str], text: str, rng: random.Random) -> tuple[str, bool]:
    """한 판. (이긴 후보, 무승부였나) 를 돌려준다."""
    tags = "ABCDE"[:len(heat)]
    for t, spec in zip(tags, heat):
        play(spec, text, t)
    while True:
        try:
            cmd = input(f"  어느 쪽? [{'/'.join(tags)}]  (소문자=다시 듣기, t=구분안됨, q=중단) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if cmd == "q":
            raise SystemExit(0)
        if cmd == "t":
            return rng.choice(heat), True
        if cmd.upper() in tags and cmd.isupper():
            return heat[tags.index(cmd)], False
        if cmd.upper() in tags:                  # 소문자 = 다시 듣기
            play(heat[tags.index(cmd.upper())], text, cmd.upper())
            continue
        print("    대문자로 고르고, 소문자로 다시 듣는다. 예: A 로 고르기 / a 로 재생")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=15, help="후보 수(기본 15)")
    ap.add_argument("--k", type=int, default=3, help="한 판에 몇 개(2=이지선다, 기본 3)")
    ap.add_argument("--seed", type=int, default=None, help="후보 뽑기 시드(재현용)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--specs", metavar="A,B,C",
                    help="후보를 직접 지정(쉼표 구분). 주면 무작위 대신 이 목록을 쓴다")
    ap.add_argument("--specs-file", metavar="FILE",
                    help="후보 목록 파일(한 줄에 하나, # 은 주석)")
    args = ap.parse_args()

    specs = None
    if args.specs_file:
        specs = [ln.split("#")[0].strip()
                 for ln in Path(args.specs_file).read_text(encoding="utf-8").splitlines()]
        specs = [s for s in specs if s]
    elif args.specs:
        specs = [s.strip() for s in args.specs.split(",") if s.strip()]

    logging.basicConfig(level=logging.WARNING, format="  [경고] %(message)s")
    from app.main import _setup_audio_device
    _setup_audio_device()

    seed = args.seed if args.seed is not None else random.randrange(10_000)
    rng = random.Random(seed)
    control = str(settings.models["tts"]["voice"])
    pool = make_pool(args.pool, rng, control, specs=specs)

    print(f'\n블라인드 토너먼트 · 후보 {len(pool)}명 · 한 판 {args.k}개 · 시드 {seed}')
    print(f'문장: "{args.text}"')
    print("이름은 안 알려준다. 끝나면 공개한다.\n")

    ties = 0
    rnd = 1
    while len(pool) > 1:
        heats = make_heats(pool, args.k)
        print(f"── {rnd}회전 · {len(pool)}명 -> {len(heats)}명")
        winners = []
        for i, heat in enumerate(heats, 1):
            if len(heat) == 1:
                winners.append(heat[0])
                print(f"  [{i}/{len(heats)}] 부전승")
                continue
            print(f"  [{i}/{len(heats)}]")
            w, tie = run_heat(heat, args.text, rng)
            ties += tie
            winners.append(w)
        pool = winners
        rnd += 1
        print()

    win = pool[0]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"winner": win, "seed": seed, "ties": ties,
                                 "control": control}, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    print("=" * 60)
    print(f"  우승: {win}")
    if win == control:
        print("  🔴 현행 목소리가 이겼다 — 블라인드에서 이겼으니 바꿀 이유가 없다.")
    else:
        print(f"  (현행은 {control} 였다)")
    if ties:
        print(f"  ⚠️ 무승부 {ties}판 — 구분이 안 되는 판이 이만큼 있었다.")
        print("     무승부가 절반을 넘으면 Supertonic 화자들이 서로 비슷하다는 뜻이고,")
        print("     그건 다른 TTS 를 봐야 한다는 근거다.")
    print(f"\n  다시 들어보기: python tools/voice_tournament.py --explore 1 --from '{win}'")
    print(f"  확정하려면 configs/model_paths.yaml 의 tts.voice 를 위 값으로.")
    print("=" * 60)


if __name__ == "__main__":
    main()
