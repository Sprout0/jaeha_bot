"""필러(맞장구)가 젯슨에서 성립하나 — 귀로 한 번, 마이크로 한 번 확인한다.

실행(젯슨):
    python -m tools.check_filler                # 1) 장치 2) 캐시 3) 하나씩 4) 실제 순서
    python -m tools.check_filler --measure      # 마이크로 '꼬리 잘림' 실측(숫자)
    python -m tools.check_filler --out-device 0 # 출력 장치를 바꿔 재볼 때

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


# ── 1) 귀로 확인 ──────────────────────────────────────────────────────────────

def listen(bank, tts, sd, out_dev, reply: str) -> int:
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

    input("\n[1/2] 필러를 하나씩 냅니다. 각각 말끝이 맺히는지 들어보세요. Enter> ")
    for (audio, rate), phrase in zip(bank._audio, bank.phrases):
        print(f"   {phrase}")
        sd.play(audio, rate, device=out_dev)
        sd.wait()
        time.sleep(0.4)

    input(f"\n[2/2] 실제 순서를 재현합니다(필러 -> 답). Enter> ")
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
                 burst_ms: float, margin_db: float, calib: int) -> int:
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
    if out_dev is None:
        rate = tts._resolve_play_rate(sd)
    else:
        rate = int(sd.query_devices(out_dev)["default_samplerate"])
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
    p.add_argument("--out-device", default=None, help="재생 장치(운영 기본이 의심되면 지정)")
    p.add_argument("--in-device", default=None, help="녹음 장치(보통 0=ReSpeaker)")
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

    print(sd.query_devices())
    print(f"\n기본 장치(입력, 출력) = {sd.default.device}")

    tts = TTSModule(**settings.models.get("tts", {}))
    tts.load()
    print(f"합성 {tts.sample_rate}Hz -> 재생 {tts._resolve_play_rate(sd)}Hz "
          f"(voice={tts.voice}, steps={tts.total_steps})\n")

    if a.measure:
        return measure_tail(sd, tts, out_dev, in_dev, a.repeats, a.bursts,
                            a.burst_ms, a.margin_db, a.calibrate_repeats)

    from pathlib import Path
    cfg = settings.models.get("filler", {}) or {}
    bank = FillerBank(cfg.get("phrases", []),
                      cache_dir=Path(cfg.get("cache_dir", "~/.cache/jaeha_filler")).expanduser(),
                      sink=SoundDeviceSink(),
                      delay_s=float(cfg.get("delay_s", 0.0)),
                      enabled=bool(cfg.get("enabled", True)))
    bank.ensure(tts)
    return listen(bank, tts, sd, out_dev, a.reply)


if __name__ == "__main__":
    raise SystemExit(main())
