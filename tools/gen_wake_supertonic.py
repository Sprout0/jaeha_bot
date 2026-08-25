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

🔴🔴 2026-08-09 정정 — 위 ②의 '배속'을 리샘플로 만든 건 **설계 오류였다.**

  리샘플은 시간과 피치를 **같이** 바꾼다. 그래서 학습 데이터에서 '빠르다'와 '높다'가
  항상 붙어 다녔고(격자의 대각선만 채운 꼴), 모델은 둘을 구분할 수 없어 **피치를 골랐다.**
  실기 결과: 아이 목소리(높음)는 깨우는데 **어른이 빠르게 부르면 놓친다.**

  실측(배포된 v3 모델에 같은 문장을 두 방식으로 넣어 비교):
    리샘플로 0.74배 압축(피치↑)      -> 호출점수 중앙 0.795, 임계초과 87.5%
    WSOLA 로 0.79배 압축(피치 유지)  -> 호출점수 중앙 **0.453**, 임계초과 60.0%
    ...그런데 whisper 는 후자도 92.5% 를 '재하봇'으로 전사했다.
    **음성은 멀쩡하고 모델만 못 알아듣는다** = 학습 데이터 탓이다.

  → 이제 시간축(time_stretch, 피치 유지)과 피치축(pitch_shift, 길이 유지)을 **따로**
    주고 EXPAND_GRID 로 곱한다. '빠르지만 낮은' 칸과 '느리지만 높은' 칸이 둘 다 있어야
    모델이 두 축을 분리해 배운다.

  교훈: **증강 축이 서로 상관되면 모델은 그중 쉬운 축을 고른다.**
  v3 때 클래스 대칭(긍정·부정에 같은 증강)은 챙겼지만 **축 독립성**을 놓쳤다.

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
# 🔴 2026-08-25 '하이 티드'로 교체. 띄어쓴 형태와 붙인 형태를 **둘 다** 넣는다 —
#    붙여 쓰면 Supertonic 이 어절 사이 쉼 없이 발음해, 실제로 빨리 부르는 모양이 된다.
DEFAULT_PHRASES = ["하이 티드", "하이 티드!", "하이티드", "하이티드!", "하이 티드야"]

# 🔴🔴 2026-08-25 화자를 **실측 F0 대역**으로 재편했다 (tools/probe_wake_speed.py).
#
#   v4 실기 실패의 원인이 여기였다 — 학습 긍정 F0 중앙 165Hz 인데 사용자(아빠)는 104~116Hz.
#   v5 는 이걸 '피치를 낮추는 증강'으로 풀려다 실패했다(다른 칸의 몫을 빼앗음).
#   증강이 아니라 **화자를 골라서** 푸는 게 맞다. 그러려면 화자별 F0 를 알아야 했고, 쟀다:
#
#     저역 M2+M5 110 / M5 112 / M1+M3 126 / M3 128     ← 아빠(실측 104~116Hz)
#     중역 M2 143 / F5 180 / M1 206                    ← 엄마
#     고역 F3 232 / F1+M2 234 / F2 242 / F1 250 / F2+F5 314  ← 재하(2세, 미측정)
#
#   v4 의 20종은 여성 프리셋이 많아 저역 비중이 얇았다. 아래는 저역이 **1/3** 이다.
#   ⚠️ 대가: F0 를 안 잰 8종(F4·M4·F1+F4·F3+F1·F4+F2·F5+F3·M3+M1·F3+M4)을 뺐다.
#      화자 다양성이 20 -> 12 로 준다. v4 사후분석의 결론이 "다양성이 아니라 분포가
#      문제였다"였으므로 이 교환을 택한다. 다시 늘리려면 먼저 F0 를 재고 대역에 넣을 것.
VOICES = [
    "M2+M5", "M5", "M1+M3", "M3",          # 저역 110~128Hz
    "M2", "F5", "M1",                      # 중역 143~206Hz
    "F3", "F1+M2", "F2", "F1", "F2+F5",    # 고역 232~314Hz
]

# 합성 속도 상한 — **호출어마다 다르다.** 다시 재고 정할 것(tools/probe_wake_speed.py).
#   '재하봇'  : 1.15 에서 42~50%, 1.25~1.40 에서 7%   -> 상한 1.05
#   '하이 티드': 1.15 에서 **92%**, 1.25 에서 42%      -> 상한 **1.15** (2026-08-25 실측)
#
# 🟢 이 한 칸이 중요한 이유: 1.15 는 **진짜로 빨리 발음한 소리**다(조음 축약 포함).
#    WSOLA 로 늘린 가짜 빠르기와 다르다 — v4 실기에서 '빠르게' 통과가 10% 였던 게
#    바로 그 차이였다("합성 배속과 진짜 빠른 말은 다르다").
#    길이 실측: x0.85 1.04s -> x1.15 0.80s. 합성만으로 1.3배 폭이 나온다.
SPEEDS = [0.85, 0.95, 1.05, 1.15]

# --expand 격자: (시간배속, 피치배율). 통과한 클립에만 건다.
#
# 🔴 v3 실패의 원인이 여기 있었다. 예전엔 리샘플 하나로 처리해 **'빠르다'와 '높다'가
#    항상 같이** 움직였다(대각선만 채운 격자). 모델은 둘을 구분할 수 없었고 피치를 골랐다
#    — 실측: 피치 유지하고 빠르게만 하면 점수가 0.80 -> 0.45 로 반토막.
#    그래서 두 축을 **따로** 준다. 아래 격자에는 '빠르지만 낮은'(1.35, 1.0) 과
#    '느리지만 높은'(1.0, 1.35) 이 둘 다 있어야 한다. 없으면 같은 실패가 재현된다.
#
# 🔴 v4 실기가 남긴 것(2026-08-12, 실음성 44건). 축 분리는 됐는데 **성인 남성 목소리가
#    통째로 약했다.** 또박또박 천천히 불러도 통과 50%(점수 중앙 0.223)다.
#    녹음의 피치만 올려 다시 채점하면 이렇게 살아난다:
#        조건        원본    +15%    +35%
#        또박또박    0.223   0.351   0.814
#        보통        0.179   0.496   0.481
#        빠르게      0.080   0.320   0.125
#        흘려서      0.041   0.092   0.080   ← 유일하게 안 오른다(축약 발음, 별개 문제)
#    음량은 원인이 아니다(학습셋 음량으로 정규화해도 -0.010). 배경소음·SNR·F0 상관도 0 근처.
#    이유는 분포에 있다 — 학습 긍정 F0 중앙 **165Hz**(5~95% 88~229)인데 사용자는 **104~116Hz**.
#    범위 밖은 아니지만 얇은 꼬리이고, 게다가 v4 는 피치를 **올리는 쪽으로만** 증강해
#    무게중심을 더 위로 밀어 놨다.
# ➡️ v5 는 **내리는 쪽 칸**을 넣는다. 녹음을 +15%/+35% 올려 좋아졌다는 건 학습 쪽에서
#    1/1.15=0.87 · 1/1.35=0.74 로 낮춘 데이터를 넣는 것과 같은 말이다 → 0.85 · 0.75.
EXPAND_GRID = [
    (1.15, 1.0), (1.35, 1.0),    # 빠르기만 — 어른이 빨리 부르는 경우(v3 에 없던 칸)
    (1.0, 1.15), (1.0, 1.35),    # 높기만 — 아이 음역, 말 속도는 보통
    (1.15, 1.35), (1.35, 1.15),  # 섞임 — 두 축이 독립임을 보여 주는 칸
    (1.25, 1.25),                # 예전 리샘플과 같은 대각선(있어도 되지만 이것만 있으면 안 됨)
    (1.0, 0.85), (1.0, 0.75),    # v5: 낮기만 — 성인 남성 음역(실측 104~116Hz)
    (1.15, 0.85), (1.35, 0.85),  # v5: 낮고 빠름 — 어른이 낮은 목소리로 빨리 부르는 경우
]
# 클립 하나당 이만큼만 무작위로 골라 건다. 말뭉치 전체로는 위 칸들이 고루 덮이면서
# 총 개수는 v3·v4(원본×4)와 같게 유지된다 — 학습 시간이 그대로여야 비교가 된다.
# ⚠️ 칸이 7 -> 11 로 늘었으므로 칸당 표본은 그만큼 얇아진다(긍정 train 기준 780 -> 496).
#    그래도 개수를 안 늘리는 쪽을 택한다. 총량까지 같이 바꾸면 v4 와 비교가 깨진다.
EXPAND_PER_CLIP = 3


def build_grid(phrases: list[str]) -> list[tuple[str, str, float]]:
    """(문구, 화자, 합성속도) 조합을 섞어 돌려준다. 시드 고정이라 재현된다."""
    grid = list(itertools.product(phrases, VOICES, SPEEDS))
    random.Random(20260806).shuffle(grid)
    return grid


def speed_up(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """리샘플로 rate 배 빠르게(= 길이 1/rate, **피치도 ×rate**). soxr 없으면 선형보간.

    ⚠️ 이건 '빠른 발화'가 아니라 '빠르고 높은 소리'다. 단독으로 쓰지 말 것 —
       아래 pitch_shift 와 조합해 시간·피치를 **따로** 주는 데 쓴다.
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


def time_stretch(audio: np.ndarray, rate: float, sr: int = TARGET_SR,
                 N: int = 1024, Hs: int = 256) -> np.ndarray:
    """WSOLA — **피치를 유지한 채** rate 배 빠르게. 사람이 빨리 말하는 것과 같은 변형.

    numpy 만 쓴다(젯슨에 scipy·librosa 가 없고, 새 의존성을 늘리지 않는다).

    🔴 탐색 반경 delta 는 반드시 |Ha-Hs| 보다 작아야 한다. 크면 탐색기가 매번
       '시간을 안 줄이는 위치'(k+Hs)를 고른다 — 거기가 상관 1.0 이라서.
       그러면 **배속이 조용히 무효가 된다**(실제로 delta=160 으로 돌렸다가
       rate 1.35 에서 길이비 0.93 이 나왔다, 기대는 0.74). 길이비를 꼭 검산할 것.
    """
    if abs(rate - 1.0) < 1e-6 or audio.size < 4 * N:
        return audio.astype(np.float32)
    x = audio.astype(np.float32)
    Ha = int(round(Hs * rate))
    delta = max(8, int(0.9 * abs(Ha - Hs)))
    w = np.hanning(N).astype(np.float32)
    L = N - Hs
    out = np.zeros(int(x.size / rate) + 4 * N, dtype=np.float32)
    nrm = np.zeros_like(out)
    # ⚠️ 탐색 중심은 **직전에 고른 위치가 아니라 절대 이상위치**(ideal)여야 한다.
    #    직전 위치 기준으로 하면 '덜 압축하는 쪽'으로 치우친 선택이 매 프레임 누적돼
    #    배속이 요청보다 6~7% 덜 걸린다(실측). ideal 은 정확히 Ha 씩 나아간다.
    ideal = 0.0
    k = o = 0
    while ideal + N + Hs + delta < x.size and o + N < out.size:
        out[o:o + N] += x[k:k + N] * w
        nrm[o:o + N] += w
        tmpl = x[k + Hs:k + Hs + L]                  # 자연스러운 다음 이음매
        ideal += Ha
        ctr = int(round(ideal))
        lo, hi = max(0, ctr - delta), min(x.size - N - L, ctr + delta)
        if hi <= lo:
            k = min(max(ctr, 0), max(x.size - N - L, 0))
        else:
            c = np.arange(lo, hi + 1, 4)
            segs = np.stack([x[i:i + L] for i in c])
            sim = segs @ tmpl / (np.linalg.norm(segs, axis=1) + 1e-9)
            k = int(c[int(np.argmax(sim))])
        o += Hs
    nrm[nrm < 1e-6] = 1.0
    return (out[:o + N] / nrm[:o + N]).astype(np.float32)


def pitch_shift(audio: np.ndarray, factor: float, sr: int = TARGET_SR) -> np.ndarray:
    """**길이를 유지한 채** 피치만 ×factor (유아 음역 재현용).

    리샘플로 압축하면 피치가 올라가되 길이가 줄므로, 그만큼 WSOLA 로 늘려 되돌린다.
    """
    if abs(factor - 1.0) < 1e-6 or audio.size == 0:
        return audio.astype(np.float32)
    return time_stretch(speed_up(audio, sr, factor), 1.0 / factor, sr)


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
    rng = random.Random(20260809)          # 시드 고정 — 재현 가능해야 한다
    made, ratios, cover = 0, [], {}
    for c in clips:
        y, sr = sf.read(c)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        for rate, pitch in rng.sample(EXPAND_GRID, EXPAND_PER_CLIP):
            z = time_stretch(y, rate, sr)
            z = pitch_shift(z, pitch, sr)
            name = f"clip_{nxt:06d}.wav"
            sf.write(os.path.join(out_dir, name), z, sr)
            man.write(json.dumps({"clip": name, "src": os.path.basename(c),
                                  "rate": rate, "pitch": pitch,
                                  "dur": round(z.size / sr, 3)},
                                 ensure_ascii=False) + "\n")
            if y.size:
                ratios.append((rate, z.size / y.size))
            cover[(rate, pitch)] = cover.get((rate, pitch), 0) + 1
            nxt += 1
            made += 1
    man.close()
    print(f"변형 {made}개 추가 (원본 {len(clips)} × {EXPAND_PER_CLIP}칸) "
          f"-> 총 {len(clips) + made}개")

    # 🔴 배속이 실제로 걸렸는지 검산한다. WSOLA 는 탐색 반경을 잘못 잡으면 조용히
    #    아무 일도 안 한다 — 그러면 '빠른 발화'가 또 학습에서 빠지고, 우리는 그걸
    #    실기에서야 알게 된다(v3 가 정확히 그렇게 실패했다).
    print("\n  [검산] 시간배속별 실제 길이비 (기대 = 1/배속)")
    bad = False
    for rate in sorted({r for r, _ in EXPAND_GRID}):
        vals = [v for r, v in ratios if r == rate]
        if not vals:
            continue
        got, want = sum(vals) / len(vals), 1.0 / rate
        off = abs(got - want)
        bad = bad or off > 0.08
        print(f"    {rate:<5} 실제 {got:.3f} / 기대 {want:.3f}"
              f"{'   🔴 배속이 안 걸렸다' if off > 0.08 else ''}")
    print("\n  [격자] 칸별 개수")
    for key in sorted(cover):
        print(f"    시간 {key[0]:<5} 피치 {key[1]:<5} {cover[key]}")
    if bad:
        print("\n→ 길이가 기대와 다르다. 이대로 학습하면 v3 실패가 반복된다.")
        return 1
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
                    help="합성 대신, 폴더의 기존 클립에 시간×피치 격자를 걸어 이어붙인다"
                         " (검수 통과본에만 적용할 것). 끝에 길이비를 검산한다")
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
