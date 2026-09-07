"""출력이 어디로 나가는지 층별로 가른다 — 소리가 안 날 때 제일 먼저 돌린다.

실행(젯슨):
    python -m tools.probe_audio_out            # 전체 진단(삑 소리 남)
    python -m tools.probe_audio_out --list     # 장치 목록만(소리 안 남)

🔴 왜 필요한가. `SoundDeviceSink.play()` 는 `sd.play(samples, target)` 를
   **출력 장치를 지정하지 않고** 부른다(app/audio_player.py). 그래서 시스템 기본
   출력이 엉뚱한 장치(HDMI·죽은 default)면 소리는 그리로 가고, 앱은 성공했다고
   믿는다 — `AudioPlayer.play()` 는 파일만 디코딩되면 True 를 돌려주기 때문이다.
   2026-08-24 에 마이크가 같은 병으로 0 을 줬다(장치 미지정 -> 죽은 default).

가르는 방법: 장치를 **명시해서** 하나씩 삑을 쏜다. 어느 장치에서 소리가 나는지
사람이 듣고 고르면, 기본 장치가 틀렸는지 / 출력 자체가 죽었는지가 갈린다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

BEEP_HZ = 440.0
BEEP_S = 1.0
BEEP_AMP = 0.15   # ⚠️ 이어폰으로 듣는다. 크게 하지 말 것.


def _beep(rate: int) -> np.ndarray:
    t = np.linspace(0.0, BEEP_S, int(rate * BEEP_S), endpoint=False, dtype=np.float32)
    # 앞뒤 페이드 — 딸깍 소리를 없앤다(잘림과 혼동하지 않으려고)
    env = np.minimum(1.0, np.minimum(t, BEEP_S - t) / 0.02).astype(np.float32)
    return (BEEP_AMP * env * np.sin(2 * np.pi * BEEP_HZ * t)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="장치 목록만 보고 끝낸다")
    args = ap.parse_args()

    import sounddevice as sd

    print(f"sounddevice {sd.__version__}\n")

    print("== 층 1: 장치 목록 ==")
    print(sd.query_devices())
    print(f"\nsd.default.device = {sd.default.device}")
    try:
        outdev = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        print(f"기본 출력 장치 -> {sd.query_devices(outdev, 'output')['name']}")
    except Exception as e:
        print(f"🔴 기본 출력 장치를 못 찾는다: {e}")

    outs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
    print(f"\n출력 가능 장치 {len(outs)}개: {[i for i, _ in outs]}")
    if args.list:
        return 0
    if not outs:
        print("🔴 출력 장치가 하나도 없다. 여기서 끝. ALSA/드라이버 문제다.")
        return 1

    print("\n== 층 2: 앱과 같은 경로(장치 미지정) ==")
    print("앱이 실제로 쓰는 길이다. 여기서 안 들리면 기본 장치가 범인일 가능성이 크다.")
    try:
        sd.play(_beep(48000), 48000)
        sd.wait()
        print("  재생 호출은 예외 없이 끝났다(소리 여부는 귀로 판단).")
    except Exception as e:
        print(f"  🔴 예외: {e}")

    print("\n== 층 3: 장치를 하나씩 명시해서 쏜다 ==")
    print("들리는 번호를 적어 두라. 그게 써야 할 출력 장치다.\n")
    for i, d in outs:
        rate = int(d["default_samplerate"]) or 48000
        print(f"  [{i}] {d['name']}  ({rate}Hz, {d['max_output_channels']}ch) ...", end="", flush=True)
        try:
            sd.play(_beep(rate), rate, device=i)
            sd.wait()
            print(" 재생 끝")
        except Exception as e:
            print(f" 🔴 {type(e).__name__}: {e}")

    print("\n== 정리 ==")
    print("  층 2에서 들렸다        -> 출력 경로 정상. 다른 원인을 봐야 한다.")
    print("  층 2 X, 층 3 어딘가 O  -> 기본 출력 장치가 틀렸다. 그 번호를 설정에 박으면 된다.")
    print("  층 3도 전부 X          -> ALSA 볼륨/뮤트 또는 배선. amixer 를 본다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
