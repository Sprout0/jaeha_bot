"""1단계 게이트 — 음악이 나오는 중에 호출어를 알아듣는가.

🔴 왜 이 시험이 필요한가 (2026-09-09).

유튜브 재생을 붙이기로 했다. 그런데 스피커에서 노래가 나오면 **그 소리를 자기
마이크가 다시 듣는다.** 09-07 젯슨 실기의 자모거리 0.00 · 1패스는 **조용할 때**
숫자다. 음악이 깔린 숫자는 아직 아무도 재지 않았다.

여기서 못 잡으면 "노래 틀면 봇이 귀머거리가 되는" 물건이 된다. 그래서 브라우저를
깔기 전에, 설치 없이 지금 있는 mp3 만으로 먼저 잰다.

────────────────────────────────────────────────────────────
⚠️ 무엇을 재고 무엇을 안 재는가

  재는 것   : 봇이 **실제로 쓰는** 감지기(make_detector)가 음악 밑에서 몇 번 잡나
  안 재는 것: 판정 로직 자체 — 그건 여기서 다시 구현하지 않는다

08-20·08-24 에 측정 도구가 셋 틀려서 결론이 뒤집힌 적이 있다. 그래서 감지기는
config 그대로 만들고, 이 파일이 새로 짜는 건 **음악을 트는 지그**뿐이다.
(AudioPlayer.play() 는 볼륨 조절도 반복도 없어서 시험용으로는 못 쓴다.)

────────────────────────────────────────────────────────────
🔴 제일 중요한 안전장치 — 마이크가 음악을 정말 듣고 있는가

08-24 에 똑같은 함정에 빠졌다. 이어폰을 ReSpeaker 잭에 꽂고 쟀더니 마이크가
8회 중 0회 검출했다 — 이어폰은 음향 누설이 없어서다.

**이어폰을 꽂은 채로 이 시험을 돌리면 전부 통과한다. 그리고 그 결과는 거짓이다.**
스피커에서 실제로 소리가 나와 마이크로 돌아와야 의미가 있다. 그래서 본시험 전에
'마이크가 음악을 듣는가'를 먼저 확인하고, 안 들리면 **거기서 멈춘다.**

────────────────────────────────────────────────────────────
사용법 (젯슨에서):

    ./run.sh check                      # 먼저 장치 확인
    python -m tools.wake_under_music    # 기본: 볼륨 0/30/50/70%, 각 10회

    python -m tools.wake_under_music --trials 5 --volumes 0 50
    python -m tools.wake_under_music --song world_playground

결과는 reports/jetson/wake_under_music_<날짜>.jsonl 에 남는다.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from app.audio_device import setup_audio_device
from app.audio_player import default_library, _resample
from app.audio_source import AudioSource, SAMPLE_RATE, FRAME
from app.config import settings
from app.stt_module import STTModule
from app.wake import make_detector

log = logging.getLogger("wake_under_music")

# 한 번의 시도를 이만큼 기다린다. 80ms/프레임이므로 12초.
TRIAL_S = 12.0
TRIAL_FRAMES = int(TRIAL_S * SAMPLE_RATE / FRAME)

# 마이크가 음악을 '듣고 있다'고 인정할 최소 배수(무음 대비).
# 09-07 실측 소음바닥 0.00698 기준, 2배면 확실히 위로 올라온 것이다.
MIC_HEARS_RATIO = 2.0


class LoopingMusic:
    """음악을 볼륨을 조절해 무한 반복으로 튼다 — 시험용 지그.

    sd.play() 는 한 번 틀고 끝이라 12초 × N회를 못 버틴다. 볼륨도 없다.
    그래서 OutputStream 콜백으로 직접 돌린다.

    ⚠️ 출력은 반드시 ReSpeaker(USB)로 나가야 한다. XVF-3000 의 하드웨어 AEC 는
       'USB 로 나간 재생 신호'를 레퍼런스로 삼아 마이크에서 빼주기 때문이다.
       setup_audio_device() 가 이미 기본 장치를 ReSpeaker 로 잡아 두므로 그대로 쓴다.
    """

    def __init__(self, samples: np.ndarray, rate: int) -> None:
        import sounddevice as sd
        # 장치가 원본 레이트를 못 받으면(ReSpeaker 는 16000 전용) 미리 변환한다.
        target = rate
        try:
            sd.check_output_settings(samplerate=rate)
        except Exception:
            dev = sd.default.device
            outdev = dev[1] if isinstance(dev, (list, tuple)) else dev
            target = int(sd.query_devices(outdev, "output")["default_samplerate"])
            samples = _resample(samples, rate, target)
        self._samples = samples.astype("float32")
        self._rate = target
        self._pos = 0
        self._gain = 0.0
        self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        n = len(self._samples)
        out = np.empty(frames, dtype="float32")
        filled = 0
        while filled < frames:                     # 끝에 닿으면 앞으로 감는다
            take = min(frames - filled, n - self._pos)
            out[filled:filled + take] = self._samples[self._pos:self._pos + take]
            self._pos = (self._pos + take) % n
            filled += take
        outdata[:, 0] = out * self._gain

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=self._rate, channels=1,
                                       dtype="float32", callback=self._callback)
        self._stream.start()

    def set_volume(self, percent: float) -> None:
        self._gain = max(0.0, min(100.0, percent)) / 100.0

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def mic_rms(source: AudioSource, seconds: float = 2.0) -> float:
    """마이크로 들어오는 소리의 RMS 를 잰다."""
    source.drain()                                  # 묵은 프레임을 버리고 지금부터
    n = max(1, int(seconds * SAMPLE_RATE / FRAME))
    vals = []
    for _ in range(n):
        frame = source.read()
        if frame is not None and len(frame):
            vals.append(float(np.sqrt(np.mean(frame.astype("float64") ** 2))))
    return float(np.mean(vals)) if vals else 0.0


def check_mic_hears_music(source: AudioSource, music: LoopingMusic,
                          volume: float = 50.0) -> tuple[bool, float, float]:
    """🔴 본시험 전 관문 — 마이크가 음악을 정말 듣는가.

    이어폰만 꽂혀 있으면 여기서 걸린다(08-24 의 그 함정). 걸러내지 않으면
    '음악 밑에서도 100% 검출' 이라는 **거짓 합격**이 나온다.
    """
    music.set_volume(0.0)
    time.sleep(0.3)
    quiet = mic_rms(source, 2.0)

    music.set_volume(volume)
    time.sleep(0.5)                                 # 스피커가 소리를 낼 시간
    loud = mic_rms(source, 2.0)
    music.set_volume(0.0)

    ratio = (loud / quiet) if quiet > 0 else 0.0
    return (ratio >= MIC_HEARS_RATIO), quiet, loud


def run_trials(detector, source, music: LoopingMusic, volume: float,
               trials: int, records: list) -> dict:
    """한 볼륨에서 N 회 부르게 하고 몇 번 잡는지 센다."""
    music.set_volume(volume)
    time.sleep(0.5)
    noise = mic_rms(source, 1.5)

    print(f"\n{'='*56}")
    print(f"  볼륨 {volume:.0f}%   (마이크가 듣는 소음 RMS {noise:.5f})")
    print(f"  준비되면 Enter → '하이 티드' 라고 부르세요. {trials}회 반복합니다.")
    print(f"{'='*56}")
    input("  Enter: ")

    hits = 0
    for i in range(1, trials + 1):
        detector.reset()
        source.drain()
        print(f"   [{i}/{trials}] 지금 부르세요... ", end="", flush=True)

        t0 = time.time()
        result = detector.wait_for_wake(max_frames=TRIAL_FRAMES)
        elapsed = time.time() - t0

        if result is not None:
            hits += 1
            print(f"🟢 잡음 (점수 {result.score:.3f}, {elapsed:.1f}s)")
        else:
            print(f"🔴 놓침 ({elapsed:.1f}s)")

        records.append({
            "volume": volume, "trial": i, "hit": result is not None,
            "score": float(result.score) if result is not None else None,
            "elapsed_s": round(elapsed, 2), "noise_rms": noise,
        })
        time.sleep(0.4)

    music.set_volume(0.0)
    rate = hits / trials if trials else 0.0
    print(f"  → 볼륨 {volume:.0f}%: {hits}/{trials} ({rate:.0%})")
    return {"volume": volume, "hits": hits, "trials": trials,
            "rate": rate, "noise_rms": noise}


def main() -> int:
    ap = argparse.ArgumentParser(description="음악 중 호출어 검출률 측정")
    ap.add_argument("--song", default="world_playground", help="assets 에 있는 음원 id")
    ap.add_argument("--trials", type=int, default=10, help="볼륨당 시도 횟수")
    ap.add_argument("--volumes", type=float, nargs="+", default=[0, 30, 50, 70],
                    help="시험할 볼륨(%%). 0 은 기준선(조용할 때)")
    ap.add_argument("--force", action="store_true",
                    help="마이크가 음악을 못 들어도 강행(결과를 믿지 말 것)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # 1) 장치 — 봇과 같은 길로. 이걸 빼면 죽은 Tegra APE 로 나가 소리가 사라진다(09-07).
    setup_audio_device()

    # 2) 음원
    lib = default_library()
    asset = lib.get(args.song)
    if asset is None or not asset.exists:
        print(f"🔴 음원이 없다: {args.song}")
        print(f"   재생 가능: {[a.id for a in lib.playable()]}")
        return 2
    import soundfile as sf
    samples, rate = sf.read(asset.path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    print(f"음원: {asset.id} ({len(samples)/rate:.1f}초 @ {rate}Hz)")

    # 3) 봇과 똑같은 감지기. config 를 그대로 쓴다 — 여기서 임계값을 바꾸면
    #    시험한 물건과 실기에 도는 물건이 달라진다(08-26 v6 때 그렇게 틀렸다).
    wcfg = settings.models.get("wake", {}) or {}
    if not wcfg.get("enabled", True):
        print("🔴 config 에서 wake 가 꺼져 있다 — 켜고 다시 돌릴 것")
        return 2
    print("STT 적재 중(2단계 검증에 쓴다)...")
    stt = STTModule(**settings.models.get("stt", {}))
    source = AudioSource(
        preroll=float(wcfg.get("preroll", 0.5)),
        verify_window=float(((wcfg.get("onnx") or {}).get("verify") or {})
                            .get("window_s", 2.0)),
    ).open()
    detector = make_detector(wcfg, stt, source)
    if getattr(detector, "source", None) is None:
        print("🔴 ONNX 감지기가 아니라 STT 폴백이 잡혔다 — 모델을 확인할 것")
        source.close()
        return 2
    print(f"감지기 준비됨 (1단계 임계 {wcfg.get('onnx', {}).get('threshold')}, "
          f"2단계 컷 {wcfg.get('threshold')})")

    music = LoopingMusic(samples, rate)
    music.start()
    records: list = []
    summary: list = []
    try:
        # 4) 🔴 관문 — 마이크가 음악을 듣는가. 여기서 멈추는 게 거짓 합격보다 낫다.
        print("\n마이크가 음악을 듣는지 확인합니다(5초)...")
        heard, quiet, loud = check_mic_hears_music(source, music)
        print(f"  무음 RMS {quiet:.5f} → 음악 RMS {loud:.5f} "
              f"(배수 {loud/quiet if quiet else 0:.1f}, 기준 {MIC_HEARS_RATIO})")
        if not heard:
            print("\n🔴 마이크가 음악을 못 듣고 있다.")
            print("   이어폰만 꽂혀 있을 때 이렇게 된다(08-24 와 같은 상황).")
            print("   **스피커를 연결하고 다시 돌릴 것** — 이 상태의 결과는 거짓 합격이다.")
            print("   그래도 강행하려면 --force")
            if not args.force:
                return 3

        # 5) 본시험
        for vol in args.volumes:
            summary.append(run_trials(detector, source, music, vol,
                                      args.trials, records))
    except KeyboardInterrupt:
        print("\n중단됨 — 여기까지의 결과는 저장한다")
    finally:
        music.stop()
        source.close()

    # 6) 결과 저장 — 원본을 남긴다. 09-07 metrics 와 같은 자리·같은 형식.
    outdir = Path("reports/jetson")
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = outdir / f"wake_under_music_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 56)
    print("  결과")
    print("=" * 56)
    print(f"  {'볼륨':>6} {'검출':>8} {'비율':>7}  {'소음 RMS':>10}")
    for s in summary:
        print(f"  {s['volume']:>5.0f}% {s['hits']:>4}/{s['trials']:<3} "
              f"{s['rate']:>6.0%}  {s['noise_rms']:>10.5f}")

    # 통과 기준은 미리 정해 둔 것을 그대로 쓴다 — 결과를 보고 정하지 않는다.
    gate = next((s for s in summary if s["volume"] == 50), None)
    if gate:
        print()
        if gate["rate"] >= 0.8:
            print("  🟢 통과 — 볼륨 50%에서 8/10 이상. 2단계(크로미움)로 진행 가능")
        elif gate["rate"] >= 0.5:
            print("  🟡 애매 — 볼륨 상한을 걸면 쓸 만하다. 30% 결과를 같이 볼 것")
        else:
            print("  🔴 실패 — 설계를 바꿔야 한다(노래 중 ducking 또는 버튼 깨우기)")
    print(f"\n  원본: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
