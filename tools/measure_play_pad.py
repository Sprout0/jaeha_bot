"""재생 앞 무음(PLAY_PAD_S)이 실제로 얼마나 필요한지 젯슨에서 잰다.

실행(젯슨, 조용한 환경에서):
    python -m tools.measure_play_pad --repeats 5

🔴 왜. `PLAY_PAD_S = 0.15` 는 **한 번도 측정된 적이 없다**. 도입 커밋(5c1892c)이
   '잘림 방지용'으로 넣은 뒤 그대로다. sounddevice 는 스트림이 열리기까지 앞쪽
   샘플을 흘리고, 그래서 첫 음절이 잘린다. 그 지연을 무음으로 흡수하는 값인데
   **그 앞 무음 동안은 아무 소리도 안 난다** — 지금 첫 소리 721ms 중 150ms(21%)다.

측정 방식: 주파수가 각각 다른 짧은 삑을 줄줄이 이어 붙여 **패딩 없이** 재생하고
마이크로 되받는다. 잘려나간 삑은 그 주파수가 녹음에 **아예 없다**.
⚠️ 시간 정렬로 재지 않는다 — 마이크 경로에도 지연이 있어 '녹음의 0초'와 '재생의
   0초'가 다르다. 주파수 유무로 세면 정렬이 필요 없고 방 울림·음량에도 안 흔들린다.
⚠️ 여러 번 재고 **최댓값**을 쓴다. PLAY_PAD_S 는 중앙값이 아니라 최악을 덮어야 한다.
   덜 덮으면 가끔 첫 음절이 잘리는데, 그건 아이가 바로 알아채는 품질 저하다.
⚠️ 재생 경로는 운영과 같아야 한다(젯슨 USB 는 16kHz 전용이라 리샘플이 낀다).
   그래서 TTSModule 의 _resolve_play_rate / _resample 을 그대로 탄다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

REC_RATE = 16000     # 젯슨 마이크 녹음 레이트. 탐침 주파수는 이 나이퀴스트 아래여야 한다
ANCHOR_HZ = 400.0    # 탐침 끝에 붙이는 기준음. 삑들과 충분히 떨어뜨린다


def make_probe(sr: int, n: int = 12, burst_ms: float = 15.0,
               f0: float = 600.0, df: float = 400.0) -> tuple[np.ndarray, list[float]]:
    """삑 n 개를 이어 붙인 탐침과 그 주파수 목록.

    삑마다 주파수를 달리 하는 게 요점이다 — 몇 번째가 사라졌는지 주파수만 보고 안다.
    """
    freqs = [f0 + df * i for i in range(n)]
    m = int(round(sr * burst_ms / 1000))
    t = np.arange(m) / sr
    # 앞뒤를 부드럽게 재워 경계에서 생기는 클릭(넓은 대역 잡음)을 없앤다.
    # 안 그러면 그 클릭이 다른 삑의 주파수에도 에너지를 뿌려 오판한다.
    win = np.hanning(m)
    out = [np.sin(2 * np.pi * f * t) * win for f in freqs]
    return np.concatenate(out).astype(np.float32), freqs


def add_anchor(sig: np.ndarray, sr: int, hz: float = ANCHOR_HZ,
               ms: float = 30.0, at_end: bool = True) -> np.ndarray:
    """탐침 끝(기본)에 기준음을 붙인다 — '녹음 실패'와 '진짜 다 잘림'을 가르는 장치.

    🔴 앞이 아무리 잘려도 끝은 남는다. 그러므로 기준음이 안 들리면 그건 잘림이 아니라
       녹음이 실패한 것이다. 첫 실기 측정이 이걸 구분 못 해서, sd.play 가 sd.rec 를
       끊어버린 회차를 '200ms 잘림'으로 집계하고 최댓값을 오염시켰다.
    🔴 **꼬리를 잴 때는 at_end=False 로 앞에 붙인다.** 뒤에 두면 재려는 그 잘림에
       기준음도 함께 쓸려나가, 안전장치가 하필 필요한 순간에 무용지물이 된다.
    """
    m = int(round(sr * ms / 1000))
    t = np.arange(m) / sr
    tone = (np.sin(2 * np.pi * hz * t) * np.hanning(m)).astype(sig.dtype)
    return np.concatenate([sig, tone] if at_end else [tone, sig])


def _magnitudes(x: np.ndarray, freqs: list[float], sr: int) -> list[float]:
    """각 목표 주파수에서의 상관 크기. 녹음 전체에 창을 씌우지 않는다.

    🔴 창을 씌우면 **안 된다.** 창은 앞뒤를 0 으로 재우는데, 하필 우리가 봐야 할
       자리가 맨 앞이다. 그러면 잘리지 않은 앞쪽 삑까지 사라진 것처럼 보여 잘림이
       부풀려진다(초기 구현이 이 함정에 빠졌고 테스트가 잡았다).
    """
    t = np.arange(x.size) / sr
    return [abs(complex(np.dot(x, np.exp(-2j * np.pi * f * t)))) for f in freqs]


def classify_magnitudes(mags: list[float], floor_mags: list[float] | None,
                        floor_db: float = -22.0, margin_db: float = 12.0) -> list[bool]:
    """상관 크기 목록을 '들렸다/안 들렸다'로 가른다.

    🔴 2026-08-24 젯슨 실측이 peak-relative 판정(floor_mags=None 이던 시절의 유일한
       방식)의 결함을 드러냈다. **재생을 전혀 안 했는데도** 600Hz·5000Hz·기준음(400Hz)
       이 8/8 회 '들렸다'로 오판됐다. '이 녹음 안의 최댓값' 을 기준으로 삼았는데,
       마이크 자체잡음은 주파수마다 균일하지 않아서(양 극단에서 우연히 크다) 잡음이
       제일 큰 주파수가 그냥 통과해버린다. 심지어 안전장치인 기준음까지 뚫려서, 녹음이
       통째로 실패해도 '성공'으로 오판할 수 있었다.
    ➡️ floor_mags 가 있으면 각 주파수를 **그 주파수 자신의 사전 측정 잡음바닥**과
       비교한다(margin_db 배만큼 넘어야 한다). 잡음바닥이 원래 높은 주파수는 문턱도
       그만큼 높아져 오판을 막는다. 없으면(합성 신호로 만든 기존 테스트들을 위해)
       예전의 peak-relative 로 되돌아간다.
    """
    if floor_mags is not None:
        mult = 10 ** (margin_db / 20)
        return [bool(m > max(f, 0.0) * mult) for m, f in zip(mags, floor_mags)]
    peak = max(mags) if mags else 0.0
    if peak <= 0:
        return [False] * len(mags)
    # bool() 로 감싼다 — numpy 불리언은 `is True` 로 비교되지 않아 쓰는 쪽이 다친다.
    return [bool(20 * np.log10(m / peak + 1e-12) > floor_db) for m in mags]


def present(rec: np.ndarray, freqs: list[float], sr: int,
            floor_db: float = -22.0) -> list[bool]:
    """각 주파수가 녹음 어딘가에 있나(peak-relative). 시간 정렬은 하지 않는다.

    ⚠️ 실기에서 이 방식은 불안정하다(위 classify_magnitudes 참고) — 실측에는
       present_with_floor 를 쓴다. 이 함수는 합성 신호로 만든 기존 단위테스트와
       하위 호환을 위해 남긴다.
    """
    x = np.asarray(rec, dtype=np.float64).ravel()
    if x.size < 16 or not np.any(x):
        return [False] * len(freqs)
    mags = _magnitudes(x, freqs, sr)
    return classify_magnitudes(mags, None, floor_db=floor_db)


def calibrate_noise_floor(rec: np.ndarray, freqs: list[float], sr: int) -> list[float]:
    """재생 없이 녹음한 표본에서 각 주파수의 '원래 이 정도는 시끄럽다' 크기를 잰다."""
    x = np.asarray(rec, dtype=np.float64).ravel()
    if x.size < 16:
        return [0.0] * len(freqs)
    return _magnitudes(x, freqs, sr)


def present_with_floor(rec: np.ndarray, freqs: list[float], sr: int,
                       floor_mags: list[float], margin_db: float = 12.0) -> list[bool]:
    """calibrate_noise_floor 로 잰 잡음바닥과 비교해 판정한다. 실기에서 쓸 것."""
    x = np.asarray(rec, dtype=np.float64).ravel()
    if x.size < 16 or not np.any(x):
        return [False] * len(freqs)
    mags = _magnitudes(x, freqs, sr)
    return classify_magnitudes(mags, floor_mags, margin_db=margin_db)


def truncation_ms(found: list[bool], burst_ms: float) -> float:
    """앞에서 **연속으로** 빠진 삑의 길이(ms). 중간 구멍은 잘림이 아니라 검출 실패다."""
    n = 0
    for ok in found:
        if ok:
            break
        n += 1
    return n * burst_ms


def tail_truncation_ms(found: list[bool], burst_ms: float) -> float:
    """뒤에서 **연속으로** 빠진 삑의 길이(ms). 앞쪽 짝과 같은 규칙, 방향만 반대다.

    🔴 스트림을 여는 비용이 앞을 흘리듯, 닫는 비용은 뒤를 흘린다. PLAY_PAD_S 는 앞뒤에
       같은 값을 대고 있는데 그 값이 앞 기준으로만 논의돼 왔다 — 뒤도 재야 한다.
    """
    return truncation_ms(list(reversed(found)), burst_ms)


# ── 실기 측정 ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PLAY_PAD_S 실측")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--bursts", type=int, default=12)
    p.add_argument("--burst-ms", type=float, default=15.0)
    p.add_argument("--f0", type=float, default=600.0)
    p.add_argument("--df", type=float, default=400.0)
    p.add_argument("--in-device", default=None,
                   help="녹음 장치. 젯슨 기본(default)은 0 만 준다 — 보통 0(ReSpeaker)")
    p.add_argument("--out-device", default=None,
                   help="재생 장치. 지정하면 TTSModule._resolve_play_rate 대신 이 장치로 재생한다"
                       "(운영 경로가 죽은 default 를 쓰는 게 의심되면 dmix 등으로 바꿔 재볼 것)")
    p.add_argument("--margin-db", type=float, default=12.0,
                   help="잡음바닥보다 이만큼(dB) 커야 '들렸다'. 12dB=4배")
    p.add_argument("--calibrate-repeats", type=int, default=3,
                   help="잡음바닥을 몇 번 재서 그 중 최댓값을 기준으로 삼을지")
    a = p.parse_args(argv)

    import sounddevice as sd

    from app.config import settings
    from app.tts_module import PLAY_PAD_S, TTSModule

    tts = TTSModule(**settings.models.get("tts", {}))
    sr = tts.sample_rate
    probe, freqs = make_probe(sr, a.bursts, a.burst_ms, a.f0, a.df)

    # 🔴 녹음 나이퀴스트를 넘으면 접혀서 엉뚱한 삑이 잡힌다. 실기에서 8,200Hz 를
    #    만들어 이 함정에 빠졌다 — 여기서 막는다.
    limit = REC_RATE / 2 * 0.9
    if max(freqs) >= limit:
        print(f"🔴 탐침 최고 주파수 {max(freqs):.0f}Hz 가 녹음 한계 {limit:.0f}Hz 를 넘는다."
              f" --bursts 를 줄이거나 --df 를 낮출 것.")
        return 1

    signal = add_anchor(probe, sr)
    out_dev = int(a.out_device) if a.out_device not in (None, "") else None
    if out_dev is None:
        rate = tts._resolve_play_rate(sd)
    else:
        # 지정된 출력 장치의 기본 레이트로 재생한다. TTSModule 의 자동 선택을
        # 건너뛴다 — 그게 죽은 장치를 고르고 있는지 자체가 지금 의심 대상이다.
        rate = int(sd.query_devices(out_dev)["default_samplerate"])
    play = signal if rate == sr else tts._resample(signal, sr, rate)
    probe_ms = a.bursts * a.burst_ms

    print(f"현재 PLAY_PAD_S = {PLAY_PAD_S:.3f}s ({PLAY_PAD_S * 1000:.0f}ms)")
    print(f"탐침: 삑 {a.bursts}개 x {a.burst_ms:.0f}ms = {probe_ms:.0f}ms "
          f"/ {freqs[0]:.0f}~{freqs[-1]:.0f}Hz / 기준음 {ANCHOR_HZ:.0f}Hz")
    print(f"재생 {rate}Hz (장치 {out_dev if out_dev is not None else '기본(운영 경로와 동일)'})"
          f" / 녹음 {REC_RATE}Hz")
    print()

    # 🔴 죽은 장치로 재면 '전부 잘림'처럼 보인다. 재기 전에 장치가 살아 있는지 본다.
    #    젯슨의 ALSA `default` 는 실제로 순수한 0 을 준다(2026-08-24 확인).
    dev = int(a.in_device) if a.in_device not in (None, "") else None
    allf = freqs + [ANCHOR_HZ]

    def capture(seconds: float) -> np.ndarray:
        frames: list[np.ndarray] = []
        with sd.InputStream(samplerate=REC_RATE, channels=1, dtype="float32",
                            device=dev,
                            callback=lambda d, n, t, st: frames.append(d.copy())):
            time.sleep(seconds)
        rec = np.concatenate(frames).ravel() if frames else np.zeros(1, np.float32)
        return np.nan_to_num(rec)

    probe_rec = capture(0.4)
    if not np.any(probe_rec):
        print(f"🔴 녹음 장치({dev if dev is not None else '기본'})가 순수한 0 을 준다 "
              f"— 마이크가 안 잡힌다.")
        print("   --in-device 로 살아 있는 장치를 지정할 것(sd.query_devices() 로 확인).")
        return 1
    print(f"녹음 장치 확인: rms {float(np.sqrt(np.mean(probe_rec.astype(np.float64) ** 2))):.2e}"
          f" — 살아 있다")

    # 🔴 잡음바닥을 **주파수별로** 잰다. 2026-08-24 젯슨 실측: 재생을 전혀 안 했는데도
    #    peak-relative 판정은 600Hz·5000Hz·기준음(400Hz)을 8/8 회 '들렸다'로 오판했다
    #    — 이 세 주파수가 마이크 자체잡음에서 원래 크기 때문이다. 회차마다 최댓값을
    #    잡아 보수적으로 잡는다(방 소음이 흔들려도 안전한 쪽으로).
    #    재생 구간과 같은 길이만큼 재야 잡음 통계가 비교 가능하다.
    cap_s = 0.3 + len(play) / rate + 0.25
    floor_mags = [0.0] * len(allf)
    for _ in range(a.calibrate_repeats):
        quiet = capture(cap_s)
        m = calibrate_noise_floor(quiet, allf, REC_RATE)
        floor_mags = [max(a_, b_) for a_, b_ in zip(floor_mags, m)]
    print(f"잡음바닥 측정 완료({a.calibrate_repeats}회, 각 {cap_s:.2f}s) "
          f"/ 판정 여유 {a.margin_db:.0f}dB")
    print()

    cuts, failed = [], 0
    for i in range(1, a.repeats + 1):
        frames: list[np.ndarray] = []
        # ⚠️ sd.rec 를 쓰면 안 된다 — sd.play 와 전역 스트림을 공유해서 재생이
        #    시작되는 순간 녹음이 끊긴다(실기에서 당했다). 입력 스트림을 따로 연다.
        with sd.InputStream(samplerate=REC_RATE, channels=1, dtype="float32",
                            device=dev,
                            callback=lambda d, n, t, st: frames.append(d.copy())):
            time.sleep(0.3)                 # 녹음이 확실히 흐르기 시작한 뒤에
            sd.play(play, rate, device=out_dev)
            sd.wait()
            time.sleep(0.25)                # 꼬리와 방 울림까지 담는다
        rec = np.concatenate(frames).ravel() if frames else np.zeros(1, np.float32)
        rec = np.nan_to_num(rec)

        got = present_with_floor(rec, allf, REC_RATE, floor_mags, margin_db=a.margin_db)
        anchor, bursts = got[-1], got[:-1]
        marks = "".join("O" if g else "." for g in bursts)
        if not anchor:
            failed += 1
            print(f"  {i}회  [{marks}]  기준음 없음 -> **녹음 실패, 버림**")
        else:
            cut = truncation_ms(bursts, a.burst_ms)
            cuts.append(cut)
            print(f"  {i}회  [{marks}]  앞 잘림 {cut:5.0f}ms  "
                  f"(들린 삑 {sum(bursts)}/{len(bursts)})")
        time.sleep(0.3)

    print()
    if not cuts:
        print("🔴 쓸 만한 회차가 하나도 없다(전부 기준음 없음).")
        print("   스피커·마이크·볼륨을 먼저 확인할 것. 이 결과로 값을 바꾸면 안 된다.")
        return 1
    if failed:
        print(f"⚠️ {a.repeats}회 중 {failed}회는 녹음 실패로 버렸다 "
              f"— 남은 {len(cuts)}회로 판단한다.")

    worst = max(cuts)
    if worst >= probe_ms:
        print(f"🔴 잘림이 탐침 길이({probe_ms:.0f}ms)를 넘는다 — 얼마인지 모른다.")
        print("   --bursts 를 늘려 다시 잴 것. 지금 값은 유지한다.")
        return 1

    print(f"앞 잘림: 최대 {worst:.0f}ms / 중앙 "
          f"{sorted(cuts)[len(cuts) // 2]:.0f}ms  (n={len(cuts)})")
    # 최악값에 여유를 더한다. 첫 음절이 잘리는 건 아이가 바로 알아채는 품질 저하라
    # 아끼는 쪽으로 기울면 안 된다.
    margin = max(20.0, worst * 0.5)
    need = (worst + margin) / 1000
    print(f"권장 PLAY_PAD_S = {need:.3f}s  (최악 {worst:.0f}ms + 여유 {margin:.0f}ms)")
    delta = PLAY_PAD_S - need
    if delta > 0.005:
        print(f"➡️ {PLAY_PAD_S:.3f}s -> {need:.3f}s 로 내리면 첫 소리 "
              f"**-{delta * 1000:.0f}ms**")
    elif delta < -0.005:
        print(f"⚠️ 지금 값이 오히려 부족하다 — {need:.3f}s 로 올려야 한다"
              f"(첫 음절이 가끔 잘리고 있을 수 있다)")
    else:
        print("➡️ 지금 값이 적정하다. 이 항목은 목록에서 지운다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
