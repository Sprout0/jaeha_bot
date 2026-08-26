"""실제 목소리로 호출어를 녹음하고, 배포된 감지기가 왜 못 잡는지 가른다.

왜 필요한가 (2026-08-10):
  v4 실기에서 "빠르게·흘려서 말하면 안 된다"가 남았다. 원인 후보가 둘인데
  둘의 처방이 정반대다:
    (가) 내 목소리가 학습 음역 밖이다      -> 피치 칸을 넓히면 된다(합성으로 해결)
    (나) 흘려 말하면 음소가 빠진다          -> 축약형 문구·실음성이 필요하다(합성으로 안 됨)
  추측으로 다음 데이터를 만들면 또 헛돈다. **같은 녹음 하나로 둘을 가른다.**

가르는 방법 — 실패한 녹음을 피치만 올려 다시 채점한다:
  점수가 뛰면  -> (가) 음역 문제. 낮은 피치 칸을 넣고 재학습하면 된다.
  그대로면     -> (나) 발음 문제. 피치로는 못 고친다.
whisper 전사는 대조군이다. whisper 가 '재하봇'으로 읽는데 감지기 점수가 낮으면
**음성은 멀쩡하고 모델만 못 듣는 것**이다(v3 때 이 방법으로 원인을 잡았다).

녹음물은 버리지 않고 data/wake_real/ 에 쌓는다 — 실음성 재학습의 씨앗이다.

🔴 2026-08-26 두 모드로 나눴다 — **2세는 시키는 대로 못 한다.**
  위의 조건 안내("빠르게, 그래도 다 발음해서")는 **읽고 따라 할 수 있는 사람**을 전제한다.
  2세에게는 셋 다 불가능하다: 카운트다운을 기다렸다가 신호에 맞춰 말하기, 네 가지
  발음 조건을 구분해 수행하기, 20회를 앉아서 반복하기. 그대로 쓰면 무음과 딴소리만
  쌓이고, 그걸 '못 알아들은 발화'로 세면 결론이 통째로 뒤집힌다(2026-08-12 에 이미
  같은 종류의 사고가 났다).

  - `--mode adult` (기본): 예전 방식. **부모 녹음에 쓴다**(20~30건 필요).
  - `--mode child`: 그냥 몇 분 **계속 녹음**하고, 부모가 아이를 놀이로 유도한다.
      끝나면 소리가 난 구간을 자동으로 잘라 하나씩 **부모가 y/n 으로 확인**한다.
      아이에게 시키는 게 아니라 **나온 것을 줍는다.**

사용법 (젯슨에서, 봇을 끄고):
    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/record_wake_real.py --mode child --minutes 3   # 아이
    python tools/record_wake_real.py --n 10                     # 부모(조건당 10회)
⚠️ ReSpeaker 는 장치가 하나뿐이라 봇이 돌고 있으면 마이크를 못 연다. 먼저 끌 것.
   (그 상태로 돌리면 젯슨 기본 입력 35번이 잡히는데, 이 장치는 **에러 없이 무음**을 준다.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models  # noqa: E402
from app.wake_onnx import OnnxWakeDetector  # noqa: E402
from tools.gen_wake_supertonic import pitch_shift  # noqa: E402

SR = 16000
FRAME = 1280          # 80ms — 런타임과 같은 단위
TAKE_S = 3.0          # 한 번 녹음 길이
# 감지기는 25프레임(2.0s)을 채워야 첫 점수를 낸다. 마이크는 끊기지 않고 흐르지만
# 잘라낸 파일은 그렇지 않으므로, 채점할 때 앞뒤에 무음을 덧대 같은 조건으로 만든다.
PAD_S = 2.5

ADULT_CONDITIONS = [
    ("또박또박", "또박또박 천천히  ―  '하 이  티 드'"),
    ("보통", "평소처럼  ―  '하이 티드'"),
    ("빠르게", "빠르게, 그래도 다 발음해서  ―  '하이 티드!'"),
    ("흘려서", "대충 흘려서, 실제로 부를 때처럼  ―  '하이티' 처럼 돼도 그대로"),
]

# 아이에게 주는 지시가 아니라 **부모에게 주는 유도 대본**이다. 2세는 조건을 못 고르므로
# 조건을 만들어 주는 쪽이 부모다. 아래를 섞어 쓰면 네 조건이 자연스럽게 다 나온다.
CHILD_ELICIT = [
    "티드한테 인사하자 —  '하이 티드!'  (부모가 먼저 하고 따라 하게)",
    "티드가 잠들었대. 깨워 볼까?  (아이가 크게 부르게 된다 = 빠르고 높은 소리)",
    "속닥속닥 작은 소리로 티드 불러 보자  (작은 소리 표본)",
    "저기 멀리 있는 티드 불러 보자  (외침 — 아이가 실제로 부를 때와 가장 비슷)",
    "티드야 뭐 해? 하고 물어보자  (호출어 뒤에 말이 붙는 실제 형태)",
]

# 아이 목소리는 400Hz 를 넘는다. 자기상관 탐색 상한을 성인 기준으로 두면 흥분한 아이의
# F0 가 천장에 눌려 조용히 낮게 찍힌다 — 격자의 아이 음역 칸을 정하는 근거가 그 숫자다.
F0_MAX_ADULT = 400
F0_MAX_CHILD = 600


# 무음 판정 기준. 젯슨 실측(2026-08-12): 조용한 방의 소음 바닥이 RMS 0.0035 정도고,
# 캡처가 아예 안 되면 **정확히 0.0** 이 나온다. 그 사이에 선을 긋는다.
SILENT_RMS = 2e-4


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))) if y.size else 0.0


def pick_device() -> int | None:
    """설정의 audio.device(이름 부분일치)에 해당하는 **입력 장치 번호**를 돌려준다.

    🔴 sd.default.device 에 넣고 끝내면 안 된다. 젯슨의 시스템 기본 입력은 35번인데
       이 장치는 **에러 없이 0.0 만 준다**(실측 2026-08-12). 전역 기본값은 다른 코드가
       바꿀 수도 있으므로, 번호를 직접 들고 다니며 sd.rec 에 매번 넘긴다.
    """
    name = (load_models().get("audio") or {}).get("device")
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            print(f"입력 장치: [{i}] {d['name']}")
            return i
    print(f"⚠️ '{name}' 을(를) 못 찾음 — 시스템 기본 마이크를 쓴다(USB 연결 확인)")
    return None


def check_mic(device: int | None) -> None:
    """녹음을 시작하기 전에 마이크가 실제로 소리를 잡는지 1초로 확인한다.

    🔴 이 관문이 없어서 무음 20개를 받아 놓고 '발음 문제'라는 엉뚱한 결론까지 냈다
       (2026-08-12). 못 잡으면 여기서 멈추는 게 20번 부르게 하는 것보다 낫다.
    """
    print("마이크 확인 중(1초, 아무 말 안 해도 됨)...")
    try:
        y = sd.rec(SR, samplerate=SR, channels=1, dtype="float32",
                   device=device, blocking=True).reshape(-1)
    except Exception as e:
        raise SystemExit(
            f"🔴 마이크를 열 수 없다: {type(e).__name__}: {e}\n"
            "   봇이 돌고 있으면 끄세요 — ReSpeaker 는 장치가 하나라 동시에 못 엽니다.")
    r = rms(y)
    print(f"  주변 소음 RMS {r:.5f}")
    if r < SILENT_RMS:
        raise SystemExit(
            "🔴 마이크가 소리를 전혀 잡지 못한다(무음). 녹음해도 의미가 없어 여기서 멈춘다.\n"
            "   1) 봇이 돌고 있으면 끄세요 (Ctrl+C) — 마이크는 하나뿐입니다.\n"
            "   2) ReSpeaker USB 연결 확인:  arecord -l\n"
            "   3) 음소거/게인 확인:  alsamixer -c 0  (F4 로 Capture 탭)")


def record_take(device: int | None) -> np.ndarray:
    for c in (3, 2, 1):
        print(f"\r  {c}...", end="", flush=True)
        time.sleep(0.7)
    print("\r  🔴 말하세요!   ", end="", flush=True)
    buf = sd.rec(int(TAKE_S * SR), samplerate=SR, channels=1, dtype="float32",
                 device=device)
    sd.wait()
    print("\r  (끝)          ")
    return buf.reshape(-1)


def score(det: OnnxWakeDetector, y: np.ndarray) -> float:
    pad = np.zeros(int(PAD_S * SR), dtype=np.float32)
    y = np.concatenate([pad, y.astype(np.float32), pad])
    det.reset()
    best = 0.0
    for i in range(0, len(y) - FRAME, FRAME):
        s = det.push(y[i:i + FRAME])
        if s is not None:
            best = max(best, s)
    return best


def f0_median(y: np.ndarray, sr: int = SR, f_max: int = F0_MAX_ADULT) -> float:
    """유성 구간의 기본주파수 중앙값(Hz). 자기상관 — numpy 만 쓴다.

    성인 남성 85~180 / 여성 165~255 / 유아 250~400, **흥분한 2세는 400 을 넘는다.**
    그래서 `f_max` 를 받는다. 상한을 낮게 두면 천장에 눌린 값이 조용히 나오고,
    그 값으로 격자의 아이 음역 칸을 정하면 데이터를 통째로 잘못 만든다.
    """
    n, hop = 1024, 256
    lo, hi = sr // f_max, sr // 70        # 70~f_max Hz 만 본다
    out = []
    for i in range(0, max(0, len(y) - n), hop):
        seg = y[i:i + n].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:   # 무음 건너뜀
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        peak = int(np.argmax(ac[lo:hi])) + lo
        if ac[peak] / ac[0] > 0.3:              # 충분히 주기적일 때만 유성음으로 인정
            out.append(sr / peak)
    return float(np.median(out)) if out else 0.0


# ══════════════════════════════════════════════════════════════════════
#  아이 모드 — 시키지 않고 줍는다
# ══════════════════════════════════════════════════════════════════════

def segment_utterances(y: np.ndarray, sr: int = SR, noise_floor: float = 0.0,
                       min_s: float = 0.35, max_s: float = 4.0,
                       gap_s: float = 0.45, pad_s: float = 0.30,
                       k: float = 3.0, floor: float = 0.006
                       ) -> list[tuple[int, int]]:
    """소리가 난 구간만 (시작, 끝) 표본 번호로 잘라낸다.

    왜 이렇게 자르나:
      - `gap_s` 0.45s: 아이 말은 중간에 쉰다. 우리 실측으로 아이 **내부 쉼 p90 이
        0.32s** 였다(2026-08-24, Smart Turn 조사 부산물). 그보다 넉넉해야 '하이…티드'가
        두 조각으로 갈리지 않는다.
      - `pad_s` 0.30s: 앞 자음이 잘리면 감지기가 못 잡는다. 판정이 아니라 녹음 탓이 된다.
      - `k`·`floor`: 문턱은 잰 소음 바닥에 비례하되 절대 하한을 둔다. 아주 조용한 방에서
        바닥이 0 에 가까우면 비례값만으로는 숨소리까지 구간이 된다.

    순수 함수다 — 마이크·모델 없이 테스트한다.
    """
    if y.size == 0:
        return []
    hop = FRAME
    nf = max(1, len(y) // hop)
    energy = np.array([rms(y[i * hop:(i + 1) * hop]) for i in range(nf)])
    thresh = max(noise_floor * k, floor)
    active = energy > thresh
    if not active.any():
        return []

    gap_f = max(1, int(round(gap_s * sr / hop)))
    segs: list[list[int]] = []
    for i, on in enumerate(active):
        if not on:
            continue
        if segs and i - segs[-1][1] <= gap_f:
            segs[-1][1] = i                      # 짧은 쉼은 이어 붙인다
        else:
            segs.append([i, i])

    pad = int(round(pad_s * sr))
    out: list[tuple[int, int]] = []
    for a, b in segs:
        # 🔴 길이 판정은 **덧대기 전** 실제 소리 구간으로 한다. 덧댄 뒤에 재면 0.1초짜리
        #    기침도 앞뒤 0.3초가 붙어 0.7초가 되어 통과한다.
        if (b - a + 1) * hop / sr < min_s:
            continue                             # 기침·문 닫는 소리 같은 것
        s0 = max(0, a * hop - pad)
        s1 = min(len(y), (b + 1) * hop + pad)
        if (s1 - s0) / sr > max_s:
            s1 = s0 + int(max_s * sr)            # 너무 길면 앞쪽만 — 호출어는 앞에 있다
        out.append((s0, s1))
    return out


def record_session(device: int | None, minutes: float, sr: int = SR) -> np.ndarray:
    """정해진 시간 동안 계속 녹음한다. Ctrl+C 로 일찍 끝낼 수 있다.

    🔴 아이를 기다리게 하지 않는다. 카운트다운도 신호도 없다 — 부모가 놀이를 하는
       동안 그냥 흐르고, 무엇을 주웠는지는 끝나고 나서 고른다.
    """
    total = int(minutes * 60 * sr)
    buf: list[np.ndarray] = []
    got = 0
    print(f"\n🔴 녹음 시작 — {minutes:g}분. 끝내려면 Ctrl+C.\n")
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=FRAME, device=device) as st:
            while got < total:
                block, overflowed = st.read(FRAME)
                if overflowed:
                    print("\r  ⚠️ 입력 넘침(overflow) — 소리가 일부 빠졌다", end="")
                b = block.reshape(-1).copy()
                buf.append(b)
                got += len(b)
                if len(buf) % 25 == 0:           # 2초마다
                    sec = got / sr
                    bar = "█" * int(sec / (minutes * 60) * 30)
                    print(f"\r  {sec:5.0f}s / {minutes*60:.0f}s  "
                          f"|{bar:<30}|  지금 소리 {rms(b):.4f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  (여기서 끝냄)")
    print()
    return np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)


def _ask(prompt: str) -> str:
    """y/n/q 를 받는다. 파이프로 돌리는 경우(EOF)엔 'n' 으로 본다."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "n"


def confirm_segments(segments: list[dict], play: bool = False,
                     out_device: int | None = None) -> list[dict]:
    """잘라낸 구간을 하나씩 보여 주고 **부모가** 호출어인지 고른다.

    🔴 자동 라벨링을 하지 않는 이유: whisper 는 유아 음성에서 못 미덥다
       (5세 WER 21%, [[research-child-stt-robots]]). 전사는 **힌트로만** 보여 주고
       판단은 사람이 한다. 잘못 라벨된 긍정 한 건이 학습을 망치는 쪽이 더 비싸다.
    """
    keep: list[dict] = []
    print(f"\n{'='*70}")
    print(f"소리 난 구간 {len(segments)}개를 찾았다. 호출어를 부른 것만 고르자.")
    print("  y = 호출어 맞음   n = 아님(기본)   a = 남은 것 전부 아님   q = 그만")
    print(f"{'='*70}")
    for i, seg in enumerate(segments):
        hint = f'  전사 "{seg["text"]}"' if seg.get("text") else ""
        print(f"\n[{i+1}/{len(segments)}] {seg['t0']:.1f}s  길이 {seg['dur']:.1f}s  "
              f"소리 {seg['rms']:.4f}  F0 {seg['f0']:.0f}Hz{hint}")
        if play:
            try:
                sd.play(seg["audio"], SR, device=out_device)
                sd.wait()
            except Exception as e:
                print(f"    ⚠️ 재생 실패(무시): {type(e).__name__}: {e}")
        a = _ask("    호출어였나? [y/N/a/q] ")
        if a == "q":
            break
        if a == "a":
            break
        if a == "y":
            keep.append(seg)
    print(f"\n➡️ {len(keep)}건을 호출어로 확인했다.")
    return keep


def _output_device() -> int | None:
    name = (load_models().get("audio") or {}).get("device")
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    return None


def run_child(det, thr: float, device: int | None, args, outdir: str,
              f_max: int) -> int:
    """자유 녹음 -> 자동 구간 분리 -> 부모 확인 -> 채점."""
    print(f"\n{'='*70}")
    print("아이 모드 — 아이에게 시키지 않는다. 부모가 놀이로 유도하고, 나온 것을 줍는다.")
    print(f"{'='*70}")
    for line in CHILD_ELICIT:
        print(f"  · {line}")
    print("\n아이가 딴 얘기를 해도 그냥 두세요. 끝나고 고릅니다.")
    _ask("준비되면 Enter — ")

    y = record_session(device, args.minutes)
    if y.size == 0:
        print("🔴 녹음된 게 없다.")
        return 1

    raw = os.path.join(outdir, "session.wav")
    sf.write(raw, y, SR)
    # 🔴 통짜 녹음을 버리지 않는다. 실제 거실에서 아이가 있는 소리 = 우리에게 **없는**
    #    배경음·부정 데이터다. 호출어 표본보다 이쪽이 더 귀할 수도 있다.
    print(f"통짜 녹음 저장: {raw}  ({len(y)/SR:.0f}초) — 배경음·부정 데이터로도 쓴다")

    hop = FRAME
    frames = np.array([rms(y[i*hop:(i+1)*hop]) for i in range(max(1, len(y)//hop))])
    floor_rms = float(np.percentile(frames, 20))   # 조용한 쪽 20% 를 바닥으로 본다
    print(f"소음 바닥(20퍼센타일) {floor_rms:.5f}")

    spans = segment_utterances(y, noise_floor=floor_rms)
    if not spans:
        print("🔴 소리 난 구간을 하나도 못 찾았다. 마이크 게인이나 거리 문제일 수 있다.")
        return 1

    segments = []
    for (s0, s1) in spans:
        clip = y[s0:s1].astype(np.float32)
        segments.append({"t0": s0 / SR, "dur": (s1 - s0) / SR,
                         "rms": rms(clip), "f0": f0_median(clip, f_max=f_max),
                         "audio": clip})

    if not args.no_stt:
        print(f"\nwhisper 로 {len(segments)}개 구간 전사 중(힌트로만 쓴다)...")
        try:
            from app.stt_module import STTModule
            stt = STTModule(**load_models().get("stt", {}))
            stt.load()
            for seg in segments:
                text, _ = stt.transcribe(seg["audio"])
                seg["text"] = (text or "").strip()
        except Exception as e:
            print(f"⚠️ 전사 실패(무시): {e}")

    keep = confirm_segments(segments, play=args.play,
                            out_device=_output_device() if args.play else None)
    if not keep:
        print("\n호출어로 확인된 게 없다. 통짜 녹음은 남겼으니 배경음으로는 쓸 수 있다.")
        return 1

    takes = []
    for i, seg in enumerate(keep):
        path = os.path.join(outdir, f"child_{i:02d}.wav")
        sf.write(path, seg["audio"], SR)
        s = score(det, seg["audio"])
        # 🔴 아이는 이미 음역이 높다. 성인 남성 때와 진단 방향이 **반대**다 —
        #    못 잡으면 '너무 낮아서'가 아니라 '너무 높아서'를 의심해야 하므로 내려서 잰다.
        takes.append({"cond": "아이", "path": path, "score": s,
                      "rms": seg["rms"], "f0": seg["f0"], "silent": False,
                      "dur": seg["dur"], "text": seg.get("text", ""),
                      "dn85": score(det, pitch_shift(seg["audio"], 0.85)),
                      "dn75": score(det, pitch_shift(seg["audio"], 0.75))})
        mark = "○" if s >= thr else "✗"
        print(f"  {mark} {os.path.basename(path)}  점수 {s:.3f}  F0 {seg['f0']:.0f}Hz")

    ss = sorted(t["score"] for t in takes)
    f0s = [t["f0"] for t in takes if t["f0"] > 0]
    passed = sum(1 for t in takes if t["score"] >= thr)
    print(f"\n{'='*70}")
    print(f"호출어 {len(takes)}건 중 **{passed}건 통과** (임계 {thr})  "
          f"점수 중앙 {ss[len(ss)//2]:.3f}")
    if f0s:
        print(f"재하 F0 중앙 **{np.median(f0s):.0f}Hz** "
              f"(최소 {min(f0s):.0f} / 최대 {max(f0s):.0f})")
        if np.median(f0s) > f_max * 0.95:
            print(f"   ⚠️ 탐색 상한 {f_max}Hz 에 눌렸을 수 있다 — --mode child 상한을 올려 다시 볼 것")
        print("   ➡️ 이 값이 EXPAND_GRID 의 아이 음역 칸을 정하는 근거다"
              " (그동안 재하 F0 는 **한 번도 측정된 적이 없었다**).")
    else:
        print("🔴 F0 가 하나도 안 잡혔다. 녹음을 직접 들어볼 것 — 판정하지 않는다.")

    fails = [t for t in takes if t["score"] < thr]
    if fails:
        gain = np.median([max(t["dn85"], t["dn75"]) - t["score"] for t in fails])
        print(f"\n실패 {len(fails)}건의 **피치 하강** 이득 중앙값: {gain:+.3f}")
        if gain > 0.15:
            print("➡️ **음역 문제 — 아이가 학습 음역보다 높다.** 격자 피치 상한을 올려"
                  " 재학습하면 된다(합성으로 해결 가능).")
        else:
            print("➡️ **음역이 아니다.** 피치를 내려도 안 살아난다 — 발음·길이·소음 쪽이다."
                  " 합성으로는 못 고친다. 이 녹음들을 학습에 넣는 게 답이다.")
    else:
        print("\n➡️ 전부 통과했다. 더 어려운 조건을 모아야 한다(멀리서·TV 켜고·뛰면서).")

    with open(os.path.join(outdir, "takes.jsonl"), "w", encoding="utf-8") as f:
        for t in takes:
            f.write(json.dumps({k: v for k, v in t.items() if k != "audio"},
                               ensure_ascii=False) + "\n")
    print(f"\n저장: {outdir}  (실음성 재학습의 씨앗 — 지금 학습 데이터엔 사람 목소리가 0건이다)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("adult", "child"), default="adult",
                    help="adult=조건별 지시 녹음(부모) / child=자유 녹음 후 골라내기")
    ap.add_argument("--n", type=int, default=5, help="[adult] 조건당 녹음 횟수")
    ap.add_argument("--minutes", type=float, default=3.0,
                    help="[child] 계속 녹음할 시간(분). Ctrl+C 로 일찍 끝낼 수 있다")
    ap.add_argument("--play", action="store_true",
                    help="[child] 확인할 때 구간을 스피커로 들려준다")
    ap.add_argument("--outdir", default="data/wake_real")
    ap.add_argument("--no-stt", action="store_true", help="whisper 전사 생략(빠름)")
    args = ap.parse_args()
    child = args.mode == "child"
    f_max = F0_MAX_CHILD if child else F0_MAX_ADULT

    cfg = load_models()["wake"]["onnx"]
    thr = float(cfg["threshold"])
    det = OnnxWakeDetector(model_dir=cfg["model_dir"], classifier=cfg["classifier"],
                           threshold=thr, trigger_frames=int(cfg.get("trigger_frames", 1)))
    print(f"감지기: {cfg['model_dir']}/{cfg['classifier']}  임계 {thr}")
    device = pick_device()
    check_mic(device)

    day = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = os.path.join(args.outdir, f"{'child' if child else 'adult'}_{day}")
    os.makedirs(outdir, exist_ok=True)

    if child:
        return run_child(det, thr, device, args, outdir, f_max)

    takes = []
    for label, guide in ADULT_CONDITIONS:
        print(f"\n{'='*60}\n[{label}]  {guide}\n{'='*60}")
        for k in range(args.n):
            print(f"({k+1}/{args.n})", end="")
            y = record_take(device)
            path = os.path.join(outdir, f"{label}_{k:02d}.wav")
            sf.write(path, y, SR)
            r = rms(y)
            if r < SILENT_RMS:
                # 중간에 마이크를 빼앗기는 경우가 있다. 조용히 넘어가면 그 표본이
                # '못 알아들은 발화'로 둔갑해 결론을 뒤집는다.
                print(f"      🔴 무음(RMS {r:.5f}) — 이 회차는 버린다")
                takes.append({"cond": label, "path": path, "score": 0.0,
                              "rms": r, "silent": True})
                continue
            s = score(det, y)
            mark = "○" if s >= thr else "✗"
            print(f"      {mark} 점수 {s:.3f}  (RMS {r:.4f})")
            takes.append({"cond": label, "path": path, "score": s,
                          "rms": r, "silent": False})

    live = [t for t in takes if not t["silent"]]
    if not live:
        print("\n🔴 전부 무음이라 판정할 수 없다. 봇을 끄고 마이크를 확인한 뒤 다시 하세요.")
        return 1
    if len(live) < len(takes):
        print(f"\n⚠️ {len(takes)-len(live)}회가 무음이라 표에서 뺐다.")
    takes = live

    # ── 전사(대조군): whisper 가 읽으면 음성은 멀쩡하다는 뜻 ──────────────
    if not args.no_stt:
        print("\nwhisper 로 전사 중(모델 로드에 몇 초)...")
        try:
            from app.stt_module import STTModule
            stt = STTModule(**load_models().get("stt", {}))
            stt.load()
            for t in takes:
                y, _ = sf.read(t["path"], dtype="float32")
                text, _sec = stt.transcribe(y)      # (텍스트, 처리시간) 을 돌려준다
                t["text"] = (text or "").strip()
        except Exception as e:                       # 전사는 보조 정보다 — 실패해도 계속
            print(f"⚠️ 전사 실패(무시): {e}")

    # ── 피치를 올려 재채점: 음역 문제인가 발음 문제인가 ────────────────
    print("피치를 올려 재채점 중...")
    for t in takes:
        y, _ = sf.read(t["path"], dtype="float32")
        t["f0"] = f0_median(y)
        t["up15"] = score(det, pitch_shift(y, 1.15))
        t["up35"] = score(det, pitch_shift(y, 1.35))

    # ── 결과 ────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"{'조건':<8}{'n':>3}{'통과':>6}{'점수중앙':>9}{'F0':>7}"
          f"{'피치+15%':>9}{'피치+35%':>9}   전사(다수)")
    print("-" * 74)
    for label, _ in ADULT_CONDITIONS:
        g = [t for t in takes if t["cond"] == label]
        if not g:
            continue
        ss = sorted(t["score"] for t in g)
        txts = [t.get("text", "") for t in g if t.get("text")]
        top = max(set(txts), key=txts.count)[:14] if txts else "-"
        print(f"{label:<8}{len(g):>3}{sum(1 for t in g if t['score'] >= thr):>5}회"
              f"{ss[len(ss)//2]:>9.3f}{np.median([t['f0'] for t in g]):>6.0f}Hz"
              f"{np.median([t['up15'] for t in g]):>9.3f}"
              f"{np.median([t['up35'] for t in g]):>9.3f}   {top}")

    fails = [t for t in takes if t["score"] < thr]
    print("-" * 74)
    # 🔴 판정 전에 표본이 판정할 만한 것인지 먼저 본다. 2026-08-12 에 무음 20개를 놓고
    #    '발음 문제'라는 결론을 내는 사고가 났다 — 결론은 늘 '왜 그렇게 볼 수 있는지'가
    #    성립할 때만 낸다.
    voiced = [t for t in takes if t["f0"] > 0]
    if len(voiced) < len(takes) * 0.5:
        print(f"🔴 {len(takes)-len(voiced)}/{len(takes)} 건에서 사람 목소리(F0)가 안 잡힌다.")
        print("   녹음이 제대로 안 됐다는 뜻이라 판정하지 않는다. 마이크부터 확인할 것.")
    elif not fails:
        print("➡️ 전부 통과했다. 실패 사례를 더 모아야 한다(더 빠르게·더 흘려서).")
    else:
        gain = np.median([max(t["up15"], t["up35"]) - t["score"] for t in fails])
        print(f"실패 {len(fails)}건의 피치 상승 이득 중앙값: {gain:+.3f}")
        if gain > 0.15:
            print("➡️ **음역 문제**. 낮은 피치 칸(0.85·0.75)을 격자에 넣고 재학습하면 된다.")
        else:
            print("➡️ **발음 문제**. 피치로는 안 고쳐진다. 축약형 문구 추가 + 실음성 재학습이 답이다.")
        if not any(t.get("text") for t in fails):
            print("   ⚠️ 단, 실패 건의 전사가 전부 비어 있다. whisper 도 못 읽는 소리라면"
                  " 모델 탓이 아니라 녹음 탓일 수 있으니 wav 를 직접 들어볼 것.")

    meta = os.path.join(outdir, "takes.jsonl")
    with open(meta, "w", encoding="utf-8") as f:
        for t in takes:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n녹음 {len(takes)}개 저장: {outdir}  (실음성 재학습에 그대로 쓴다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
