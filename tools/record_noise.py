"""거실 소음을 길게 녹음한다 — 임베딩 컷의 **하한**을 정할 데이터.

## 왜 도구가 따로 필요한가 (2026-09-09)

`tools/enroll_wake.py --score-noise` 는 긴 wav 을 받아 컷을 정해 주는데, 정작
**그 wav 을 만드는 경로가 없었다.** `arecord` 로 찍으면 되지 않느냐 싶지만
안 된다 — 젯슨의 `/etc/asound.conf` 가 ALSA `default` 를 물리 장치가 없는
Tegra 크로스바로 보낸다. 장치를 안 정하고 녹음하면 **에러 없이 무음**이 쌓인다.

그래서 이 도구는 봇과 **같은 길**로 장치를 잡는다(`app.audio_device`).

## 이 도구가 막으려는 사고

  🔴 30~60분을 녹음했는데 전부 무음이었다.

젯슨의 죽은 입력(35번)은 예외를 안 던지고 0 을 준다(2026-08-24 실측). 끝나고
알면 저녁 하나가 통째로 날아간다. 그래서:

  - **첫 블록이 조용하면 그 자리서 멈춘다** (한 시간을 버리지 않는다)
  - 진행 중에 소리 크기를 계속 찍는다 (사람이 눈으로 본다)
  - **블록마다 파일에 쓴다** — 중간에 죽어도 그때까지는 남는다
    (09-02 에 Realtime 탐침이 끝에만 저장해서 27턴을 통째로 잃었다)

## 쓰는 법 (젯슨에서, 봇을 끄고)

    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/record_noise.py --minutes 45 --out logs/거실.wav

봇이 놓일 자리에 두고, 평소처럼 지낸다. TV·대화·주방 소리 다 좋다.
🔴 **호출어는 부르지 않는다** — 이 녹음은 '소음만' 이어야 한다.

⚠️ ReSpeaker 는 장치가 하나뿐이라 봇이 돌고 있으면 마이크를 못 연다. 먼저 끌 것.

다 되면:

    python tools/enroll_wake.py --score-noise logs/거실.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio_source import SAMPLE_RATE  # noqa: E402

SILENT_RMS = 2e-4        # record_wake_real.py 와 같은 값 — 판정 기준을 갈라 두지 않는다.


def block_rms(block) -> float:
    a = np.asarray(block, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def verdict(rms_values) -> tuple[bool, str]:
    """녹음이 쓸 만한가. **중앙값**으로 본다.

    평균이 아닌 이유: 장치가 중간에 죽으면 앞쪽 몇 블록이 평균을 들어 올려
    '소리가 있었다'로 통과시킨다. 중앙값은 안 속는다.
    """
    if not len(rms_values):
        return False, "녹음된 게 없다."
    med = float(np.median(np.asarray(rms_values, dtype=np.float32)))
    if med < SILENT_RMS:
        return False, (f"녹음이 사실상 무음이다(중앙 RMS {med:.6f} < {SILENT_RMS}). "
                       "마이크가 죽은 장치에 잡혔을 가능성이 크다 — "
                       "`./run.sh check` 로 장치를 확인하고 다시 할 것.")
    return True, ""


def should_abort(rms_values, block_s: float, after_s: float = 30.0) -> bool:
    """지금까지 들은 것만 보고 "이건 무음이다" 라고 접을 것인가.

    첫 블록 하나만 보면 안 된다 — 장치가 열리는 순간의 워밍업 값이 임계를 살짝
    넘겨 통과해 버린다(2026-09-09 노트북 실측: 1블록 0.00039, 그 뒤 0.00001).
    그래서 **처음 30초**를 모아 중앙값으로 본다.
    """
    if len(rms_values) * block_s < after_s:
        return False
    return not verdict(rms_values)[0]


def record(path: str, minutes: float, block_s: float = 10.0) -> dict:
    import sounddevice as sd
    import soundfile as sf

    from app.audio_device import setup_audio_device
    setup_audio_device()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    total_s = minutes * 60
    n = int(block_s * SAMPLE_RATE)
    rmss: list[float] = []
    t0 = time.monotonic()

    print(f"녹음 시작: {path}  ({minutes:g}분)")
    print("🔴 호출어는 부르지 마세요. 평소처럼 지내면 됩니다. (멈추려면 Ctrl+C)")
    print()
    try:
        with sf.SoundFile(path, "w", SAMPLE_RATE, 1, "PCM_16") as f:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32") as stream:
                while time.monotonic() - t0 < total_s:
                    block, _ = stream.read(n)
                    f.write(block)          # 블록마다 쓴다 — 죽어도 남는다
                    r = block_rms(block)
                    rmss.append(r)
                    done = time.monotonic() - t0
                    print(f"  {done / 60:5.1f}분 / {minutes:g}분   RMS {r:.5f}"
                          f"{'   🔴 조용하다' if r < SILENT_RMS else ''}")
                    if should_abort(rmss, block_s):
                        print()
                        print("🔴 처음 30초가 무음이다 — 여기서 멈춘다. "
                              "한 시간을 버리지 않기 위해서다.")
                        print("   봇이 돌고 있으면 끄고, `./run.sh check` 로 "
                              "ReSpeaker 가 잡히는지 보고 다시 할 것.")
                        break
    except KeyboardInterrupt:
        print()
        print("멈췄습니다 — 여기까지는 파일에 남아 있습니다.")

    ok, msg = verdict(rmss)
    secs = len(rmss) * block_s
    print()
    print(f"{'✅' if ok else '🔴'} {path}  ({secs / 60:.1f}분)")
    if not ok:
        print(f"   {msg}")
    else:
        a = np.asarray(rmss)
        print(f"   RMS  중앙 {np.median(a):.5f}  최소 {a.min():.5f}  최대 {a.max():.5f}")
        print()
        print("다음:")
        print(f"   python tools/enroll_wake.py --score-noise {path}")
    return {"path": path, "seconds": secs, "rms": rmss, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=45.0,
                    help="녹음 길이(분). 계획은 30~60분")
    ap.add_argument("--out", default=None, help="저장 경로(기본: logs/noise_<시각>.wav)")
    ap.add_argument("--block-s", type=float, default=10.0, help="진행 표시 간격(초)")
    args = ap.parse_args()

    out = args.out or os.path.join(
        "logs", f"noise_{datetime.now():%Y%m%d_%H%M}.wav")
    return 0 if record(out, args.minutes, args.block_s)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
