"""호출어 학습용 positive 음성을 Supertonic 으로 합성한다 (voxcpm 대체).

왜 바꿨나 (2026-08-07):
  voxcpm A/B 20개×4변형 결과 최고 통과율 40%, 폭주(3초 초과) 10~45%.
  품질 노브(inference_timesteps)를 올려도 그대로여서 손댈 데가 없었다.
  Supertonic 은 이미 봇 목소리로 매일 쓰는 검증된 한국어 TTS 이고,
  **속도를 파라미터로 직접 준다** — v2 를 시작한 이유가 "학습데이터에 빠른 발화가
  없다"였는데, voxcpm 은 그걸 프롬프트 문장으로만 바꿀 수 있었다(그래서 실패했다).

2단계로 나눈다 — 이게 이 도구의 핵심 설계다:

  1단계(합성)  안전한 속도(0.85~1.05)로만 만든다. 다양성은 화자·시드·문구로 낸다.
  2단계(--expand) 검수를 통과한 클립만 리샘플로 압축해 '빠른 발화'를 만든다.

🔴 왜 나눴나 — 실측 두 가지 (2026-08-07)

  ① Supertonic 은 짧은 문구를 빠르게 시키면 음절을 삼킨다.
     합성 speed 0.85~1.05 -> 통과 100% (13/13)
     합성 speed 1.15       -> 42~50%
     합성 speed 1.25~1.40  -> 7% (1/15)   '제발' '제압' 'Ciao!' 로 뭉개짐

  ② STT 검수는 빠르고 높은 소리에서 스스로 약해진다(whisper 는 성인 정속 음성으로
     학습됐다 — 배속 1.25~1.3 통과율이 50%대로 떨어졌다). 그 상태로 검수하면
     **우리가 가장 원하는 유아스러운 샘플을 골라서 버리게 된다.**
     그래서 검수는 정속 원본에만 걸고, 배속은 그 뒤에 얹는다.
     원본이 정확하면 배속본은 정의상 정확하다(신호처리는 음소를 새로 만들지 않는다).

리샘플 배속은 피치도 같이 올린다(rate 1.35 = 약 +5.2반음). 우리에겐 이득이다 —
유아 F0 가 성인보다 +4~7반음 높으므로 **빠른 발화와 유아 음역을 한 번에** 얻는다.

다양성 축: 화자 20종(F1~F5/M1~M5 + '+' 블렌딩) × 합성속도 4 × 문구 5 × 시드(클립마다)

사용 (이 순서를 지킬 것):
  python tools/gen_wake_supertonic.py --out output/positive_train -n 2000
  python tools/wake_qc.py output/positive_train --move-rejects
  python tools/gen_wake_supertonic.py --out output/positive_train --expand

실측 결과(50개 시험): 검수 통과 92%(46/50), 폭주 0건, 확장 후 184개,
길이 p10 0.61s / 중앙 0.77s / p90 0.97s (p90/p10 = 1.60배 = 속도 다양성 확보).
비교: voxcpm 은 최고 변형이 40%, 폭주 10~45%.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402

TARGET_SR = 16000  # livekit-wakeword 파이프라인 고정값

# 문구 — 호격/주격과 문장부호로 억양까지 흔든다.
DEFAULT_PHRASES = ["재하봇", "재하봇!", "재하봇아", "재하봇아!", "재하봇이"]

# 화자 20종: 단일 10 + 블렌딩 10. 블렌딩은 style 벡터 평균이라 '없던 화자'가 된다.
VOICES = [
    "F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5",
    "F1+F4", "F2+F5", "F3+F1", "F4+F2", "F5+F3",
    "M1+M3", "M2+M5", "M3+M1", "F1+M2", "F3+M4",
]

# 합성 속도 — 실측상 1.15 부터 통과율이 42% 로 꺾인다(1.05 이하는 78~100%).
# 안전 구간에서만 합성하고, 빠른 발화는 --expand 로 만든다.
SPEEDS = [0.85, 0.95, 1.0, 1.05]

# --expand 배속. 통과한 클립만 시간축 압축해 '빠른 발화 + 유아 음역'을 만든다.
EXPAND_RATES = [1.15, 1.25, 1.35]


def build_grid(phrases: list[str]) -> list[tuple[str, str, float]]:
    """(문구, 화자, 합성속도) 조합을 섞어 돌려준다. 시드 고정이라 재현된다."""
    grid = list(itertools.product(phrases, VOICES, SPEEDS))
    random.Random(20260806).shuffle(grid)
    return grid


def speed_up(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """리샘플로 rate 배 빠르게(= 길이 1/rate, 피치 ×rate). soxr 없으면 선형보간.

    음소를 새로 만들지 않고 시간축만 줄이므로, TTS 에 빠르게 말하라고 시킬 때처럼
    음절이 삼켜지지 않는다.
    """
    if rate == 1.0 or audio.size == 0:
        return audio
    try:
        import soxr
        return soxr.resample(audio, sr, sr / rate).astype(np.float32)
    except Exception:
        n = int(round(audio.size / rate))
        xp = np.linspace(0.0, 1.0, audio.size, dtype=np.float32)
        return np.interp(np.linspace(0.0, 1.0, n, dtype=np.float32), xp,
                         audio).astype(np.float32)


def expand(out_dir: str, sf) -> int:
    """검수를 통과한 클립을 배속 변형해 '빠른 발화' 구간을 채운다.

    왜 합성 때 같이 안 하고 따로 하나:
      STT 검수는 빠르고 높은 소리에서 스스로 약해진다(whisper 는 성인 정속 음성으로
      학습됐다 — 우리 실측에서도 배속 1.25~1.3 통과율이 50%대로 떨어졌다).
      그 상태로 검수하면 **우리가 가장 원하는 유아스러운 샘플을 골라서 버리게 된다.**
      그래서 '정속으로 만들어 검수까지 통과한 클립'에만 신호처리를 얹는다.
      원본이 정확하면 배속본도 정의상 정확하다.
    """
    import glob
    import json
    import re

    clips = sorted(glob.glob(os.path.join(out_dir, "clip_*.wav")))
    if not clips:
        print(f"clip_*.wav 없음: {out_dir}")
        return 1
    idxs = [int(m.group(1)) for c in clips
            if (m := re.search(r"clip_(\d+)\.wav$", os.path.basename(c)))]
    nxt = max(idxs) + 1

    man = open(os.path.join(out_dir, "manifest.jsonl"), "a", encoding="utf-8")
    made = 0
    for c in clips:
        y, sr = sf.read(c)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        for rate in EXPAND_RATES:
            z = speed_up(y, sr, rate)
            name = f"clip_{nxt:06d}.wav"
            sf.write(os.path.join(out_dir, name), z, sr)
            man.write(json.dumps({"clip": name, "src": os.path.basename(c),
                                  "rate": rate, "dur": round(z.size / sr, 3)},
                                 ensure_ascii=False) + "\n")
            nxt += 1
            made += 1
    man.close()
    print(f"배속 변형 {made}개 추가 (원본 {len(clips)} × {len(EXPAND_RATES)}배속) "
          f"-> 총 {len(clips) + made}개")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Supertonic 으로 호출어 positive 합성")
    ap.add_argument("--out", required=True, help="저장 폴더 (clip_%%06d.wav)")
    ap.add_argument("-n", "--count", type=int, default=3000)
    ap.add_argument("--phrases", default=",".join(DEFAULT_PHRASES))
    ap.add_argument("--start-index", type=int, default=0,
                    help="이어서 뽑을 때. 기존 파일을 덮어쓰지 않으려면 max+1")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="확산 스텝. 기본은 configs/model_paths.yaml 값")
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="시드 시작값. **평가셋은 반드시 학습셋과 다른 값으로 줄 것** — "
                         "같으면 확산 노이즈가 같아 똑같은 클립이 나와 시험지가 샌다")
    ap.add_argument("--expand", action="store_true",
                    help="합성 대신, 폴더의 기존 클립을 배속 변형해 이어붙인다"
                         " (검수 통과본에만 적용할 것)")
    args = ap.parse_args()

    import soundfile as sf

    if args.expand:
        return expand(args.out, sf)

    phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]
    cfg = load_models().get("tts", {})
    tts = TTSModule(
        model=cfg.get("model", "supertonic-3"),
        voice=VOICES[0],
        language=cfg.get("language", "ko"),
        total_steps=args.total_steps or int(cfg.get("total_steps", 12)),
        threads=int(cfg.get("threads", 4)),
        providers=cfg.get("providers"),
    )
    tts.load()
    src_sr = tts.sample_rate
    print(f"Supertonic 로드 완료 (합성 {src_sr}Hz -> 저장 {TARGET_SR}Hz)")

    os.makedirs(args.out, exist_ok=True)
    grid = build_grid(phrases)
    print(f"조합 {len(grid)}가지 × 시드 변주 -> {args.count}개 생성\n")

    # 어떤 클립이 어떤 조합에서 나왔는지 남긴다. voxcpm 때 이게 없어서
    # "설정은 바꿨는데 생성물이 왜 이런지"를 추적할 수 없었다.
    man = open(os.path.join(args.out, "manifest.jsonl"), "a", encoding="utf-8")

    t0, idx = time.time(), args.start_index
    style_cache: dict[str, object] = {}
    for i in range(args.count):
        text, voice, speed = grid[i % len(grid)]
        if voice not in style_cache:
            style_cache[voice] = tts._resolve_style(voice)
        tts._style = style_cache[voice]
        tts.speed = speed
        tts.seed = args.seed_base + i   # 클립마다 다른 초기 노이즈 = 억양·호흡 변주

        audio = tts._infer(text)
        audio = tts._trim(audio, keep_tail=0.15)   # 앞 무음 제거 + 말끝 여운 조금만
        audio = tts._resample(audio, src_sr, TARGET_SR)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            audio = (audio / peak * 0.95).astype(np.float32)

        name = f"clip_{idx:06d}.wav"
        sf.write(os.path.join(args.out, name), audio, TARGET_SR)
        man.write(json.dumps({"clip": name, "text": text, "voice": voice,
                              "speed": speed, "rate": 1.0,
                              "dur": round(audio.size / TARGET_SR, 3)},
                             ensure_ascii=False) + "\n")
        idx += 1

        if (i + 1) % 100 == 0 or i + 1 == args.count:
            el = time.time() - t0
            print(f"  {i+1}/{args.count}  {el:5.1f}s 경과  "
                  f"({el/(i+1):.3f}s/개, 남은 {el/(i+1)*(args.count-i-1)/60:.1f}분)")

    print(f"\n완료: {args.out} 에 {args.count}개 "
          f"(clip_{args.start_index:06d} ~ clip_{idx-1:06d}), {(time.time()-t0)/60:.1f}분")
    print("다음: python tools/wake_qc.py " + args.out + " --move-rejects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
