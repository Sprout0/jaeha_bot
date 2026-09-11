"""호출어 2단계 A/B — whisper 검증과 임베딩 검증을 **같은 녹음·같은 감지기**로 잰다.

## 왜 (2026-09-11)

임베딩 단독 검증의 Task 0 은 임베딩만 쟀다(소음 최대 0.872 / 진짜 최저 0.889).
그걸로는 "지금 봇(whisper)보다 나은가" 를 말할 수 없다. whisper 쪽 헛깨움은
같은 조건에서 한 번도 잰 적이 없다. 이 도구가 둘을 같은 자로 잰다.

## 무엇을 똑같이 맞췄나

  - 감지기: 봇과 똑같이 `app.wake.make_detector` 로 만든다. 설정 복사본의 검증
    방식만 바꾼다. 판정 로직을 여기서 다시 짜지 않는다(08-20·08-24 에 측정
    도구가 셋 틀려 결론이 뒤집혔다).
  - 소리 입력: 실제 `AudioSource` 를 물려받아 마이크 대신 파일에서만 읽는다.
    프리롤·검증창 버퍼 규칙이 봇과 같다.
  - 🔴 시계: 쿨다운은 `time.monotonic` 으로 잰다. 파일을 실시간보다 빨리 흘리면
    느린 whisper 와 빠른 임베딩이 **서로 다른 쿨다운**을 받아 비교가 기운다.
    그래서 시계를 녹음 시간으로 돌리고, 검증에 실제로 걸린 시간을 쿨다운 기준
    시각에 한 번 더한다. 실제 봇에서 whisper 가 도는 동안 소리가 쌓여 쿨다운이
    그만큼 늦게 풀리는 것을 재현한다. 두 방식에 똑같이 적용한다.

## 🔴 봇의 임베딩 검증은 지금 창이 짧아서 절대 통과하지 못한다 (2026-09-11 발견)

검증창(window_s 2.0초 = 25프레임)으로는 임베딩이 10개밖에 안 나온다. 첫 임베딩에
멜 76줄 = 16프레임이 들고, 본보기 대조는 임베딩 16개(TEMPLATE_FRAMES)를 요구한다.
모자라면 best_similarity 가 0.0 을 준다 -> 모든 후보 기각. 16개를 얻으려면
31프레임 = 2.48초가 필요하다. 이 도구의 첫 점검에서 임베딩 판정 72번이 전부
유사도 0.000 이었다.

✅ 같은 날 봇 코드를 고쳤다: 임베딩 전용 창(verify.embed_window_s, 3초)과 임베딩 쪽만
따로 더 듣는 시간(verify.embed_settle_s, 0.48초)을 둔다. 이 도구는 이제 **봇 설정을
그대로** 쓴다 — 덮어쓰려면 --embed-window / --embed-settle. 옛 고장을 다시 보려면
`--embed-window 2.0 --embed-settle 0.3`.

🔴 창을 늘려도 봇의 **검증 시점**이 이르다. 후보는 1단계 점수가 처음 임계를 넘는
순간 뜨고, 봇은 거기서 settle_s(0.3초 -> 3프레임 = 0.24초)만 더 듣는다. 그런데
1단계 점수의 최고점은 보통 그보다 0.3~0.5초 뒤다(노트북 진단, 09-11 호출 11건).
본보기는 최고점 자리로 만들었으니 창이 호출어 끝을 자른다. 실측(3초 창, 11건):
  +0.24s 6/11 -> +0.48s 10/11 (컷 0.85) / 3/11 -> 10/11 (컷 0.88)
그래서 --embed-settle 로 임베딩 쪽만 더 기다리게 해서도 잰다. 임베딩은 0.1초 안에
끝나므로 0.24초를 더 기다려도 whisper(1.2초)보다 빨리 깨운다.

## 무엇을 안 맞췄나 (읽을 때 주의)

  - 깨어난 뒤 대화하는 동안(최대 30초)은 봇이 호출어를 안 듣는다. 여기서는
    깨우자마자 다시 듣는다 -> 헛깨움이 실제보다 조금 **많게** 나올 수 있다.
    두 방식에 똑같다.

## 쓰는 법 (젯슨에서, 봇을 끄고)

    python tools/wake_ab_verify.py                     # 기본: 09-11 녹음 전부
    python tools/wake_ab_verify.py --noise-minutes 3   # 빠른 점검
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio_source import FRAME, SAMPLE_RATE, AudioSource  # noqa: E402

PAD_S = 2.5     # 링버퍼를 채우려면 앞뒤 무음이 필요하다(enroll_wake 와 같은 값)

DEFAULT_CALLS = ["data/wake_real/adult_20260911_1144/held",
                 "data/wake_real/adult_20260826_1445"]
DEFAULT_NOISE = "logs/거실.wav"
DEFAULT_TEMPLATES = "models/wake/v6/templates_haitid.npy"


# ─────────────────────────────────────────────────────────── 시계와 입력
class AudioClock:
    """녹음 시간으로 가는 시계. `bump` 는 **바로 다음 한 번**에만 더해진다."""

    def __init__(self) -> None:
        self.t = 0.0
        self._bump = 0.0

    def advance(self, s: float) -> None:
        self.t += s

    def bump(self, s: float) -> None:
        self._bump += s

    def __call__(self) -> float:
        b, self._bump = self._bump, 0.0
        return self.t + b


class FileSource(AudioSource):
    """봇의 AudioSource 그대로, 마이크 대신 배열에서 읽는다. 다 읽으면 StopIteration."""

    def __init__(self, audio, clock: AudioClock, preroll: float = 0.5,
                 verify_window: float = 2.0, embed_window: float = 3.0) -> None:
        super().__init__(preroll=preroll, verify_window=verify_window,
                         embed_window=embed_window)
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._n = a.size // FRAME
        self._audio = a[:self._n * FRAME]
        self._i = 0
        self._clock = clock

    def _read_frame(self) -> np.ndarray:
        if self._i >= self._n:
            raise StopIteration
        f = self._audio[self._i * FRAME:(self._i + 1) * FRAME]
        self._i += 1
        self._clock.advance(FRAME / SAMPLE_RATE)
        return f

    def _available(self) -> int:
        return 0        # 밀린 버퍼가 없다 — 파일은 읽는 순간 생긴다

    @property
    def position_s(self) -> float:
        return self._i * FRAME / SAMPLE_RATE


def timed(fn, clock: AudioClock, timer=time.perf_counter):
    """fn 을 감싸 걸린 실제 시간을 재고, 그만큼을 시계의 다음 한 번에 올린다."""
    def wrapper(*a, **k):
        t0 = timer()
        try:
            return fn(*a, **k)
        finally:
            d = timer() - t0
            wrapper.calls += 1
            wrapper.seconds += d
            clock.bump(d)
    wrapper.calls = 0
    wrapper.seconds = 0.0
    return wrapper


class _Monotonic:
    """time.monotonic 을 지금 돌고 있는 AudioClock 으로 바꿔 끼운다."""

    def __init__(self) -> None:
        self.clock = AudioClock()
        self._orig = None

    def __enter__(self):
        self._orig = time.monotonic
        time.monotonic = lambda: self.clock()
        return self

    def __exit__(self, *exc) -> None:
        time.monotonic = self._orig


# ─────────────────────────────────────────────────────────── 감지기
def _read(path: str) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{path}: {sr}Hz 다 — 16000Hz 로 맞출 것")
    return y


def _padded(y: np.ndarray) -> np.ndarray:
    pad = np.zeros(int(PAD_S * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([pad, y, pad])


def build(wcfg: dict, stt, source, mode: str, cut, templates: str, clock: AudioClock,
          settle=None):
    """봇과 같은 make_detector 로 만들고, 검증 장치에만 시간 재기를 씌운다.

    settle 을 주면 verify.embed_settle_s 를 바꾼다(None = 봇 설정 그대로).
    whisper 쪽 settle_s 는 절대 안 건드린다.
    """
    from app.wake import make_detector
    w = copy.deepcopy(wcfg)
    v = w.setdefault("onnx", {}).setdefault("verify", {})
    v["enabled"] = True
    v["save_rejects_dir"] = None
    er = v.setdefault("embed_rescue", {})
    if mode == "whisper":
        v["mode"] = "whisper"
        er["enabled"] = False
    else:
        v["mode"] = "embed"
        er["enabled"] = True
        er["templates"] = templates
        er["min_similarity"] = float(cut)
    if settle is not None:
        v["embed_settle_s"] = float(settle)
    det = make_detector(w, stt if mode == "whisper" else None, source)

    # 🔴 조용히 엉뚱한 것을 재지 않도록 여기서 멈춘다.
    if getattr(det, "source", None) is not source:
        raise SystemExit("ONNX 감지기가 아니라 폴백(STT 감지기)이 만들어졌다 — 모델을 확인할 것")
    if mode == "whisper":
        if det.verifier is None:
            raise SystemExit("whisper 검증기가 없다 — verify 설정을 확인할 것")
        det.verifier = timed(det.verifier, clock)
        return det, det.verifier
    if det.embed_rescue is None:
        raise SystemExit(f"임베딩 본보기를 못 읽었다({templates}) — 이대로면 1단계 단독이다")
    det.embed_sequence = timed(det.embed_sequence, clock)
    return det, det.embed_sequence


def run_one(wcfg, stt, audio, mode, cut, templates, mono: _Monotonic, preroll, window,
            ewin, settle=None):
    mono.clock = AudioClock()
    src = FileSource(audio, mono.clock, preroll=preroll, verify_window=window,
                     embed_window=ewin)
    det, probe = build(wcfg, stt, src, mode, cut, templates, mono.clock, settle)
    wakes = []
    try:
        while True:
            r = det.wait_for_wake()
            if r is None:
                break
            wakes.append({"t_s": round(src.position_s, 2), "score": round(float(r.score), 3)})
    except StopIteration:
        pass
    return wakes, probe.calls, probe.seconds


def _label(mode: str, cut, win: float, settle=None) -> str:
    if mode == "whisper":
        return "whisper"
    tail = "" if settle is None else f"+{settle:g}"
    return f"임베딩{cut:.2f}/{win:g}초{tail}"


def _set_name(d: str) -> str:
    d = d.rstrip("/")
    base = os.path.basename(d)
    return os.path.basename(os.path.dirname(d)) + "/" + base if base in ("held", "enroll") else base


# ─────────────────────────────────────────────────────────── 본체
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", nargs="+", default=DEFAULT_CALLS, help="진짜 호출 wav 폴더들")
    ap.add_argument("--noise", default=DEFAULT_NOISE, help="호출어 없는 긴 녹음")
    ap.add_argument("--noise-minutes", type=float, default=None, help="앞에서 이만큼만(점검용)")
    ap.add_argument("--templates", default=DEFAULT_TEMPLATES)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.88, 0.85],
                    help="임베딩 컷들(0.88 = 09-11 판정값, 0.85 = 지금 설정값)")
    ap.add_argument("--embed-window", type=float, default=None,
                    help="임베딩 창(초). 없으면 설정의 embed_window_s. 2.48초 미만이면 유사도가 늘 0")
    ap.add_argument("--embed-settle", type=float, default=None,
                    help="임베딩 쪽만 후보 뒤 이만큼(초) 더 듣는다. 없으면 설정의 embed_settle_s")
    ap.add_argument("--no-whisper", action="store_true",
                    help="whisper 칸을 건너뛴다(이미 잰 뒤 임베딩만 다시 잴 때)")
    ap.add_argument("--out", default=None, help="결과 JSON(기본 reports/wake/ab_verify_<날짜>.json)")
    args = ap.parse_args()

    from app.config import settings
    from app.stt_module import STTModule

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"ab_verify_{stamp}.log")
    logging.basicConfig(level=logging.INFO, filename=log_path,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"검증 로그(전사·유사도 전부): {log_path}")

    wcfg = settings.models.get("wake", {}) or {}
    vcfg = (wcfg.get("onnx") or {}).get("verify") or {}
    preroll = float(wcfg.get("preroll", 0.5))
    window = float(vcfg.get("window_s", 2.0))

    print("whisper 올리는 중...")
    stt = STTModule(**settings.models.get("stt", {}))
    stt.load()
    stt.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))     # 첫 호출 워밍업

    # whisper 는 봇 그대로, 임베딩은 '지금 코드 그대로(고장)' 하나 + 창을 늘린 컷들.
    bot_ew = float(vcfg.get("embed_window_s", 3.0))
    bot_es = vcfg.get("embed_settle_s")
    ew = bot_ew if args.embed_window is None else float(args.embed_window)
    es = bot_es if args.embed_settle is None else float(args.embed_settle)
    modes = [] if args.no_whisper else [("whisper", None, bot_ew, None)]
    modes += [("embed", c, ew, es) for c in args.cuts]
    heads = [_label(*m) for m in modes]
    result = {"when": stamp, "templates": args.templates, "modes": heads,
              "verify_window_s": window, "embed_window_s": ew, "embed_settle_s": es,
              "bot_settle_s": float(vcfg.get("settle_s", 0.0)),
              "calls": {}, "noise": {}}

    with _Monotonic() as mono:
        # ── 진짜 호출 ──
        for d in args.calls:
            wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
            if not wavs:
                raise SystemExit(f"wav 가 없다: {d}")
            clips = [(os.path.basename(p), _padded(_read(p))) for p in wavs]
            per = {}
            for (m, c, w, s), h in zip(modes, heads):
                woke, missed = 0, []
                for name, audio in clips:
                    wakes, _, _ = run_one(wcfg, stt, audio, m, c, args.templates,
                                          mono, preroll, window, w, s)
                    if wakes:
                        woke += 1
                    else:
                        missed.append(name)
                per[h] = {"woke": woke, "total": len(clips), "missed": missed}
                print(f"  호출 {_set_name(d)}  {h}  {woke}/{len(clips)}", flush=True)
            result["calls"][d] = per

        # ── 호출어 없는 긴 녹음 ──
        noise = _read(args.noise)
        if args.noise_minutes:
            noise = noise[:int(args.noise_minutes * 60 * SAMPLE_RATE)]
        hours = noise.size / SAMPLE_RATE / 3600
        result["noise"]["path"] = args.noise
        result["noise"]["minutes"] = round(hours * 60, 1)
        for (m, c, w, s), h in zip(modes, heads):
            t0 = time.perf_counter()
            wakes, n_ver, sec = run_one(wcfg, stt, noise, m, c, args.templates,
                                        mono, preroll, window, w, s)
            result["noise"][h] = {
                "false_wakes": len(wakes), "per_hour": round(len(wakes) / hours, 2),
                "verifies": n_ver, "verify_s_mean": round(sec / n_ver, 4) if n_ver else None,
                "wall_s": round(time.perf_counter() - t0, 1), "wakes": wakes}
            print(f"  소음 {hours * 60:.1f}분  {h}  헛깨움 {len(wakes)}회  (검증 {n_ver}번)",
                  flush=True)

    out = args.out or os.path.join("reports", "wake", f"ab_verify_{stamp[:8]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # ── 요약 ──
    print()
    print("=" * 72)
    print(f"{'':<30}" + "".join(f"{h:<14}" for h in heads))
    for d, per in result["calls"].items():
        cells = "".join(f"{per[h]['woke']}/{per[h]['total']:<12}" for h in heads)
        print(f"{'호출 ' + _set_name(d):<30}" + cells)
    nz = result["noise"]
    cells = "".join(f"{nz[h]['false_wakes']}회({nz[h]['per_hour']}/h)".ljust(14) for h in heads)
    print(f"{'헛깨움 ' + str(nz['minutes']) + '분':<30}" + cells)
    cells = "".join(f"{nz[h]['verify_s_mean']!s:<14}" for h in heads)
    print(f"{'검증 1번 평균(초)':<30}" + cells)
    print("=" * 72)
    for d, per in result["calls"].items():
        for h in heads:
            if per[h]["missed"]:
                print(f"못 깨움  {h}  {_set_name(d)}: " + ", ".join(per[h]["missed"]))
    print(f"결과: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
