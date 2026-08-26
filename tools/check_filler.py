"""필러(맞장구)가 젯슨에서 성립하나 — 귀로 한 번, 마이크로 한 번 확인한다.

실행(젯슨):
    python -m tools.check_filler --sweep        # 🔴 소리가 아예 안 나면 여기부터
    python -m tools.check_filler                # 1) 장치 2) 캐시 3) 하나씩 4) 실제 순서
    python -m tools.check_filler --measure      # 마이크로 '꼬리 잘림' 실측(숫자)
    python -m tools.check_filler --out-device 0 # 출력 장치를 바꿔 재볼 때

🔴 2026-08-26: 이 도구가 젯슨에서 **예외 없이 끝까지 돌면서 소리는 하나도 안 났다.**
   `/etc/asound.conf` 가 기본 출력을 `plug -> hw:APE,0` 으로 두는데 그건 Tegra 오디오
   패브릭의 DMA 입구(ADMAIF)라, XBAR 라우팅과 코덱이 없으면 아무 데도 안 간다.
   입력에서 겪은 '죽은 default' 와 **같은 함정이 출력에도 있었다.** --sweep 이 그걸 찾는다.

🔴 왜 이 도구가 필요한가. 2026-08-26 젯슨 청취에서 **필러도 답변도 말끝이 뚝 끊긴다**는
   보고가 나왔는데, 노트북에서는 잘릴 자리가 없었다:
     - 합성 파형: 말끝 감쇠 80~160ms, 파형 뒤 무음 288~648ms
     - 재생 버퍼: 끝 무음 485~798ms(PLAY_PAD_S 포함)
     - OutputStream 콜백 미소비 0프레임, sd.wait() 는 버퍼보다 500ms 더 기다림
   즉 원인은 **젯슨 출력 경로에만** 있다(ReSpeaker 16kHz 리샘플 / ALSA / 장치 선택).
   그래서 젯슨에서 도는 도구가 아니면 애초에 못 잡는다.

⚠️ 이 도구는 '고치는' 물건이 아니라 **기각 판정을 내리는** 물건이다. 필러는 실지연을
   1ms 도 안 줄인다(체감 설계다). 젯슨에서 깨끗하게 안 나오면 남길 이유가 없다.

관련: tools/measure_play_pad.py 는 같은 방식으로 **앞** 잘림을 잰다. 여기는 뒤를 잰다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from tools.measure_play_pad import (ANCHOR_HZ, REC_RATE, add_anchor,
                                    calibrate_noise_floor, make_probe,
                                    present_with_floor, tail_truncation_ms)

SILENCE_THR = 1e-3       # 무음 판정. tts_module._trim / filler._SILENCE_THR 과 같은 값

# 실제 턴에서 '아이 말 끝'부터 답이 재생되기까지. 젯슨 실측(2026-08-20, LLM 턴 18개)의
# 생각 시간에 TRT 합성(약 0.45s)을 더한 값이다. 최악(=가장 빨리 오는 답)이 중요하다 —
# 그때 필러가 아직 말하는 중이면 진짜 답이 필러를 끊고 들어온다.
ANSWER_AT_FAST_S = 1.72
ANSWER_AT_TYPICAL_S = 2.20


def audible_span(audio: np.ndarray, rate: int) -> tuple[float, float]:
    """버퍼 안에서 **들리는 소리**가 시작·끝나는 시각(초). 무음 패딩은 뺀다."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    nz = np.where(np.abs(a) > SILENCE_THR)[0]
    if nz.size == 0:
        return 0.0, 0.0
    return float(nz[0]) / rate, float(nz[-1] + 1) / rate


def overlap_verdict(buffer_s: float, audible_end_s: float,
                    answer_at_s: float) -> dict:
    """진짜 답이 필러를 끊나, 끊는다면 **들리는 부분**을 끊나.

    🔴 `sd.play()` 는 내부에서 먼저 `stop()` 을 부른다 — 앞 재생을 닫아버린다는 뜻이다.
       그래서 답이 필러보다 먼저 오면 필러는 그 자리에서 잘린다. 무음 꼬리만 잘리면
       아무도 모르지만, 말소리를 자르면 그게 '뚝' 이다.
    """
    cut = answer_at_s < buffer_s
    audible_cut = answer_at_s < audible_end_s
    return {
        "cut": cut,
        "audible_cut": audible_cut,
        "margin_s": answer_at_s - audible_end_s,
    }


def _fmt_env(sp: np.ndarray, rate: int, n: int = 8) -> str:
    """말끝 포락선(피크 대비 %). 뚝 끊긴 소리는 큰 값에서 곧장 0 으로 떨어진다."""
    peak = float(np.max(np.abs(sp))) or 1.0
    w = max(1, int(0.02 * rate))
    t = sp[-n * w:]
    env = [float(np.sqrt(np.mean(t[i * w:(i + 1) * w] ** 2))) / peak * 100
           for i in range(len(t) // w)]
    return " ".join(f"{v:4.1f}" for v in env)


# ── 0) 어느 장치가 진짜 소리를 내나 ──────────────────────────────────────────
# 🔴 2026-08-26 젯슨: `/etc/asound.conf` 가 기본 출력을 `plug -> hw:APE,0` 으로 두는데,
#    `hw:APE,0` 은 Tegra 오디오 패브릭의 DMA 입구(ADMAIF)다. XBAR 라우팅과 코덱이
#    없으면 **아무 데도 안 간다** — sd.play 는 성공하고 소리만 없다.
#    입력에서 겪은 '죽은 default' 와 같은 함정이 출력에도 있었다.

def output_candidates(devices) -> list[tuple[int, str]]:
    """소리를 내볼 만한 출력 장치. APE 내부 링크는 뺀다 — 24개가 전부 DMA 입구라
    하나씩 다 재보게 하면 사람이 못 듣고 지친다."""
    out = []
    for i, d in enumerate(devices):
        if d["max_output_channels"] <= 0:
            continue
        if "APE" in d["name"]:          # tegra-dlink / ADMAIF: 사람이 들을 물건이 아니다
            continue
        out.append((i, d["name"]))
    return out


def sweep(sd, tts, devices) -> int:
    """후보 장치마다 '몇 번'이라고 말해 본다. 들리는 번호가 곧 진짜 스피커다."""
    cands = output_candidates(devices)
    print(f"후보 {len(cands)}개. 각 장치로 번호를 말합니다 — **들리는 번호를 적어두세요.**")
    print()
    for i, name in cands:
        try:
            rate = int(sd.query_devices(i)["default_samplerate"])
            audio = tts.render(f"{i}번")
            play = audio if rate == tts.sample_rate else tts._resample(audio, tts.sample_rate, rate)
            print(f"  {i:3d} | {rate:6d}Hz | {name}", flush=True)
            sd.play(play, rate, device=i)
            sd.wait()
        except Exception as e:
            print(f"  {i:3d} | 열지 못함: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.5)
    print()
    print("들린 번호를 --out-device 에 넣어 다시 확인할 것:")
    print("  python -m tools.check_filler --out-device <번호>")
    print("  python -m tools.check_filler --measure --in-device 0 --out-device <번호>")
    return 0


# ── 1) 귀로 확인 ──────────────────────────────────────────────────────────────

def resolve_play_rate(sd, tts, out_dev, forced) -> int:
    """재생에 쓸 레이트. 강제값 > 지정장치 기본값 > 운영 경로 자동선택.

    🔴 젯슨 `/etc/asound.conf` 는 기본 출력을 `plug -> hw:APE,0 @48000` 으로 둔다.
       `plug` 는 **뭐든 받아주므로** `check_output_settings(44100)` 이 성공하고,
       운영 경로는 44100 을 고른다 — 그러면 ALSA 가 매 재생마다 44100->48000 을
       변환한다. `_resolve_play_rate` 는 '받아주나'를 물을 뿐 '네 원래 레이트냐'를
       묻지 않는다. 여기서 48000 을 강제해 그 변환을 빼고 A/B 하는 게 이 옵션이다.
    """
    if forced:
        return int(forced)
    if out_dev is not None:
        return int(sd.query_devices(out_dev)["default_samplerate"])
    return tts._resolve_play_rate(sd)


def listen(bank, tts, sd, out_dev, reply: str, play_rate: int) -> int:
    from app.tts_module import PLAY_PAD_S

    print(f"필러 {bank.size}개 / 앞뒤 무음 {bank.pad_s * 1000:.0f}ms "
          f"/ 목표 RMS {bank.target_rms}")
    if bank.size == 0:
        print("🔴 캐시가 비었다. ensure() 가 실패했다 — 위 경고를 볼 것.")
        return 1

    print(f"\n{'문구':10s} {'버퍼':>6s} {'말소리끝':>8s} {'꼬리무음':>8s}  말끝 포락선(%peak)")
    spans = []
    for (audio, rate), phrase in zip(bank._audio, bank.phrases):
        s, e = audible_span(audio, rate)
        spans.append((phrase, audio.size / rate, e))
        sp = audio[int(s * rate):int(e * rate)]
        print(f"{phrase:10s} {audio.size / rate:5.2f}s {e:7.2f}s "
              f"{(audio.size / rate - e) * 1000:7.0f}ms  {_fmt_env(sp, rate)}")

    # 답이 가장 빨리 오는 턴에 필러가 잘리나 — 숫자로 먼저 답한다.
    print(f"\n답이 재생되는 시각(젯슨 실측 기반): 최악 {ANSWER_AT_FAST_S:.2f}s / "
          f"보통 {ANSWER_AT_TYPICAL_S:.2f}s")
    worst = max(spans, key=lambda x: x[2])
    v = overlap_verdict(worst[1], worst[2], ANSWER_AT_FAST_S)
    if v["audible_cut"]:
        print(f"🔴 최악의 턴에서 '{worst[0]}' 의 **말소리**가 잘린다 "
              f"(말소리끝 {worst[2]:.2f}s > 답 {ANSWER_AT_FAST_S:.2f}s)")
    else:
        print(f"✅ 가장 늦게 끝나는 '{worst[0]}' 도 답보다 {v['margin_s']:.2f}s 먼저 끝난다"
              + (" (꼬리 무음만 잘림)" if v["cut"] else ""))

    input(f"\n[1/2] 필러를 하나씩 냅니다({play_rate}Hz, 장치 "
          f"{out_dev if out_dev is not None else '기본'}). 말끝이 맺히는지 들어보세요. Enter> ")
    for (audio, rate), phrase in zip(bank._audio, bank.phrases):
        out = audio if rate == play_rate else tts._resample(audio, rate, play_rate)
        print(f"   {phrase}")
        sd.play(out, play_rate, device=out_dev)
        sd.wait()
        time.sleep(0.4)

    # ⚠️ 여기는 **운영 경로 그대로** 간다(--play-rate / --out-device 를 안 탄다).
    #    실기에서 나는 소리를 재현하는 게 목적이라, 여기서 손대면 재현이 아니게 된다.
    input("\n[2/2] 실제 순서를 재현합니다(필러 -> 답, 운영 경로 그대로). Enter> ")
    for label, delay in (("최악(답이 빨리 옴)", ANSWER_AT_FAST_S),
                         ("보통", ANSWER_AT_TYPICAL_S)):
        print(f"   {label}: 필러 -> {delay:.2f}초 뒤 답")
        bank.play()
        time.sleep(delay)
        tts.speak(reply)
        time.sleep(0.6)

    print(f"\n현재 PLAY_PAD_S = {PLAY_PAD_S:.3f}s (앞뒤 공통). 뒤가 모자라면 --measure 로 잴 것.")
    return 0


# ── 2) 마이크로 꼬리 잘림 실측 ────────────────────────────────────────────────

def measure_tail(sd, tts, out_dev, in_dev, repeats: int, bursts: int,
                 burst_ms: float, margin_db: float, calib: int,
                 play_rate: int) -> int:
    """탐침을 **패딩 없이** 재생하고 되받아, 뒤에서 몇 ms 가 사라졌는지 센다.

    🔴 기준음을 **앞**에 붙인다. 뒤에 두면 재려는 그 잘림에 함께 쓸려나가 '녹음 실패'와
       '진짜 다 잘림'을 못 가른다(tools/measure_play_pad.add_anchor 참고).
    """
    from app.tts_module import PLAY_PAD_S

    sr = tts.sample_rate
    probe, freqs = make_probe(sr, bursts, burst_ms)
    limit = REC_RATE / 2 * 0.9
    if max(freqs) >= limit:
        print(f"🔴 탐침 최고 {max(freqs):.0f}Hz 가 녹음 한계 {limit:.0f}Hz 를 넘는다.")
        return 1

    signal = add_anchor(probe, sr, at_end=False)      # 기준음을 앞에
    rate = play_rate
    play = signal if rate == sr else tts._resample(signal, sr, rate)
    allf = freqs + [ANCHOR_HZ]

    print(f"탐침: 삑 {bursts}개 x {burst_ms:.0f}ms = {bursts * burst_ms:.0f}ms "
          f"/ 기준음 {ANCHOR_HZ:.0f}Hz(앞)")
    print(f"재생 {rate}Hz / 녹음 {REC_RATE}Hz / 현재 PLAY_PAD_S {PLAY_PAD_S * 1000:.0f}ms\n")

    def capture(seconds: float) -> np.ndarray:
        frames: list[np.ndarray] = []
        # ⚠️ sd.rec 는 sd.play 와 전역 스트림을 공유해 재생이 녹음을 끊는다. 따로 연다.
        with sd.InputStream(samplerate=REC_RATE, channels=1, dtype="float32",
                            device=in_dev,
                            callback=lambda d, n, t, st: frames.append(d.copy())):
            time.sleep(seconds)
        rec = np.concatenate(frames).ravel() if frames else np.zeros(1, np.float32)
        return np.nan_to_num(rec)

    if not np.any(capture(0.4)):
        print(f"🔴 녹음 장치({in_dev if in_dev is not None else '기본'})가 0 만 준다.")
        print("   --in-device 로 살아 있는 장치를 지정할 것(보통 0=ReSpeaker).")
        return 1

    cap_s = 0.3 + len(play) / rate + 0.35
    floor = [0.0] * len(allf)
    for _ in range(calib):
        floor = [max(x, y) for x, y in
                 zip(floor, calibrate_noise_floor(capture(cap_s), allf, REC_RATE))]
    print(f"잡음바닥 {calib}회 측정 완료 / 판정 여유 {margin_db:.0f}dB\n")

    cuts, failed = [], 0
    for i in range(1, repeats + 1):
        frames: list[np.ndarray] = []
        with sd.InputStream(samplerate=REC_RATE, channels=1, dtype="float32",
                            device=in_dev,
                            callback=lambda d, n, t, st: frames.append(d.copy())):
            time.sleep(0.3)
            sd.play(play, rate, device=out_dev)
            sd.wait()
            time.sleep(0.35)        # 꼬리와 방 울림까지 담는다
        rec = np.nan_to_num(np.concatenate(frames).ravel()
                            if frames else np.zeros(1, np.float32))
        got = present_with_floor(rec, allf, REC_RATE, floor, margin_db=margin_db)
        anchor, b = got[-1], got[:-1]
        marks = "".join("O" if g else "." for g in b)
        if not anchor:
            failed += 1
            print(f"  {i}회  [{marks}]  기준음 없음 -> **녹음 실패, 버림**")
        else:
            cut = tail_truncation_ms(b, burst_ms)
            cuts.append(cut)
            print(f"  {i}회  [{marks}]  뒤 잘림 {cut:5.0f}ms  (들린 삑 {sum(b)}/{len(b)})")
        time.sleep(0.3)

    print()
    if not cuts:
        print("🔴 쓸 만한 회차가 없다(전부 기준음 없음). 스피커·마이크·볼륨부터 볼 것.")
        return 1
    if failed:
        print(f"⚠️ {repeats}회 중 {failed}회는 녹음 실패로 버렸다 — 남은 {len(cuts)}회로 판단한다.")

    worst = max(cuts)
    print(f"뒤 잘림: 최대 {worst:.0f}ms / 중앙 {float(np.median(cuts)):.0f}ms  {cuts}")
    if worst == 0:
        print("✅ 뒤는 안 잘린다. 말끝 '뚝'의 원인은 재생 경로가 아니다.")
    elif worst <= PLAY_PAD_S * 1000:
        print(f"✅ 현재 PLAY_PAD_S({PLAY_PAD_S * 1000:.0f}ms)가 덮는다.")
    else:
        print(f"🔴 PLAY_PAD_S({PLAY_PAD_S * 1000:.0f}ms)로는 모자라다 — "
              f"뒤 무음을 최소 {worst:.0f}ms 로 올려야 한다.")
    if worst >= bursts * burst_ms:
        print("⚠️ 탐침 전체가 사라졌다 = 실제 잘림은 이보다 클 수 있다. --bursts 를 늘려 다시 잴 것.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="젯슨에서 필러가 성립하나 확인")
    p.add_argument("--measure", action="store_true", help="마이크로 꼬리 잘림을 잰다")
    p.add_argument("--sweep", action="store_true",
                   help="후보 출력 장치마다 번호를 말해 본다 — 소리가 안 날 때 먼저 이걸로 "
                        "진짜 스피커를 찾을 것(젯슨 기본 출력은 hw:APE,0 = 죽은 DMA 입구다)")
    p.add_argument("--out-device", default=None, help="재생 장치(운영 기본이 의심되면 지정)")
    p.add_argument("--in-device", default=None, help="녹음 장치(보통 0=ReSpeaker)")
    p.add_argument("--play-rate", type=int, default=None,
                   help="재생 레이트를 강제한다. 젯슨 기본 출력의 원래 레이트는 48000 인데 "
                        "운영 경로는 44100 을 고른다(ALSA plug 가 받아주므로) — 48000 을 "
                        "줘서 그 변환을 빼고 재보는 용도")
    p.add_argument("--reply", default="그건 말이야, 하늘이 파란 건 햇빛 때문이야.",
                   help="순서 재현에 쓸 답변 문장")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--bursts", type=int, default=12)
    p.add_argument("--burst-ms", type=float, default=15.0)
    p.add_argument("--margin-db", type=float, default=12.0)
    p.add_argument("--calibrate-repeats", type=int, default=3)
    a = p.parse_args(argv)

    import sounddevice as sd

    from app.audio_player import SoundDeviceSink
    from app.config import settings
    from app.filler import FillerBank
    from app.tts_module import TTSModule

    out_dev = int(a.out_device) if a.out_device not in (None, "") else None
    in_dev = int(a.in_device) if a.in_device not in (None, "") else None

    # 🔴 이걸 빼먹으면 **운영 경로가 아닌 걸 잰다.** 젯슨의 `default` 는 `plug -> hw:APE,0`
    #    (Tegra DMA 입구)라 sd.play 가 성공하고도 소리가 아무 데도 안 간다. app.main 은
    #    기동 때 이 함수로 audio.device(이름 부분일치, 'ReSpeaker')를 콕 집는다.
    #    2026-08-26: 이 도구가 그걸 안 불러서 "젯슨에서 소리가 하나도 안 난다"가 나왔다.
    from app.main import _setup_audio_device

    print(sd.query_devices())
    print(f"\n지정 전 기본 장치(입력, 출력) = {sd.default.device}")
    _setup_audio_device()
    print(f"지정 후 기본 장치(입력, 출력) = {sd.default.device}  <- 운영 경로")

    tts = TTSModule(**settings.models.get("tts", {}))
    tts.load()
    print(f"합성 {tts.sample_rate}Hz -> 재생 {tts._resolve_play_rate(sd)}Hz "
          f"(voice={tts.voice}, steps={tts.total_steps})\n")

    if a.sweep:
        return sweep(sd, tts, sd.query_devices())

    play_rate = resolve_play_rate(sd, tts, out_dev, a.play_rate)
    if play_rate != tts._resolve_play_rate(sd):
        print(f"⚠️ 재생 레이트를 {play_rate}Hz 로 바꿔 잰다 "
              f"(운영 경로는 {tts._resolve_play_rate(sd)}Hz — ALSA plug 가 그 차이를 변환한다)\n")

    if a.measure:
        return measure_tail(sd, tts, out_dev, in_dev, a.repeats, a.bursts,
                            a.burst_ms, a.margin_db, a.calibrate_repeats, play_rate)

    from pathlib import Path
    cfg = settings.models.get("filler", {}) or {}
    bank = FillerBank(cfg.get("phrases", []),
                      cache_dir=Path(cfg.get("cache_dir", "~/.cache/jaeha_filler")).expanduser(),
                      sink=SoundDeviceSink(),
                      delay_s=float(cfg.get("delay_s", 0.0)),
                      enabled=bool(cfg.get("enabled", True)))
    bank.ensure(tts)
    return listen(bank, tts, sd, out_dev, a.reply, play_rate)


if __name__ == "__main__":
    raise SystemExit(main())
