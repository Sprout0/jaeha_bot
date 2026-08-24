"""Smart Turn v3 오프라인 채점 — '우리 아이들 목소리에서도 맞히나'.

실행:
    python -m tools.eval_turn_detect --audio ~/_young --model models/smart-turn-v3.2-cpu.onnx

🔴 무엇을 재나. 체감 4.49s 의 내역에서 **VAD 꼬리 1.20s(27%)** 는 정의상 무음이다.
   말이 끝난 걸 알 방법이 '1.2초 조용한지 세기' 뿐이라 그렇다. Smart Turn 은 오디오만
   보고 '이 말이 끝났나'를 판정하므로, 맞힌다면 꼬리를 0.2s 로 줄일 수 있다(-1.0s).

   두 가지를 따로 재야 한다. 하나만 보면 속는다:
     재현율  = 진짜 끝났을 때 '끝났다'고 하는 비율   -> 높아야 시간을 번다
     오탐률  = 아직 말하는 중인데 '끝났다'고 하는 비율 -> 높으면 **아이 말을 끊는다**
   아이한테는 오탐이 훨씬 비싸다. 말 끊긴 아이는 다시 말 안 한다.

⚠️ 벤더 실측(v3.2 한국어 889샘플: 재현율 0.984 / 오탐 2.25%)은 **성인** 기준이다.
   아이는 문장 중간에 훨씬 오래 쉰다. 그 쉼을 '끝'으로 읽으면 오탐이 성인보다 나쁘다.
   이 도구는 정확히 그 격차를 재려고 있다 — 벤더 숫자를 우리 숫자로 대체하기 전엔
   아무것도 켜지 않는다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
WINDOW_S = 8.0          # 모델이 받는 최대 길이. 공식 스펙.
DEFAULT_MODEL = Path("models/smart-turn-v3.2-cpu.onnx")


# ── 오디오 준비 ──────────────────────────────────────────────────────────────

def fit_to_window(audio: np.ndarray, sr: int = SR,
                  seconds: float = WINDOW_S) -> np.ndarray:
    """8초 창에 맞춘다. 짧으면 **앞에** 0 을 채운다(공식 지침).

    뒤에 채우면 모델이 보는 마지막 순간이 무음이 돼 판정이 통째로 뒤틀린다.
    """
    want = int(round(sr * seconds))
    if len(audio) > want:
        return audio[-want:]
    if len(audio) < want:
        return np.concatenate([np.zeros(want - len(audio), dtype=audio.dtype), audio])
    return audio


def speech_span(audio: np.ndarray, sr: int = SR, floor_db: float = -40.0,
                frame_ms: float = 20.0) -> tuple[int, int]:
    """말소리가 있는 구간 [start, end) 를 샘플 단위로. 무음뿐이면 (0, 0)."""
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(audio) // hop
    if n == 0:
        return 0, 0
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return 0, 0
    loud = np.flatnonzero(20 * np.log10(rms / peak + 1e-12) > floor_db)
    if loud.size == 0:
        return 0, 0
    return int(loud[0] * hop), int(min(len(audio), (loud[-1] + 1) * hop))


def trim_tail(audio: np.ndarray, sr: int = SR,
              keep_ms: float = 200.0) -> np.ndarray:
    """말 끝난 뒤 무음을 keep_ms 만 남긴다.

    🔴 운영에서 모델이 불리는 **순간**을 재현하려는 것이다. 실제로는 VAD 가 짧은 무음
       (~0.2s)을 보자마자 부르지, 파일에 붙어 있는 3초짜리 꼬리를 다 듣고 부르지
       않는다. 원본 꼬리를 그대로 먹이면 모델에 '이만큼 조용했다'는 공짜 힌트를 줘
       재현율이 부풀려진다 — 그 부풀린 숫자로 꼬리를 줄이면 실기에서 무너진다.
    """
    start, end = speech_span(audio, sr)
    if start == end:
        return audio
    return audio[: min(len(audio), end + int(sr * keep_ms / 1000))]


def append_silence(audio: np.ndarray, sr: int = SR,
                   seconds: float = 0.0) -> np.ndarray:
    """뒤에 무음을 그만큼 붙인다 — 두 클래스의 조건을 맞추려고.

    🔴 2026-08-24 첫 실험이 이걸 안 해서 통째로 틀렸다. '완결+200ms' 대 '미완+0ms'
       를 비교했는데, 이 모델은 뒤 무음 길이를 크게 참고한다(아이 발화 30개 실측:
       뒤 무음 0.0s -> 중앙 0.187, 0.4s -> 0.366, 0.8s -> 0.765). 무음이 다르면
       모델은 무음만 보고도 두 클래스를 가를 수 있어, 정작 재려던 '이 말이 끝나게
       들리나'가 아니라 '무음이 기냐'를 재게 된다.
    ⚠️ 운영에서도 두 경우 모두 같은 시간만큼 조용하다. 맞춰야 공정하다.
    """
    if seconds <= 0:
        return audio
    return np.concatenate([audio, np.zeros(int(round(sr * seconds)), dtype=audio.dtype)])


def append_room_tone(audio: np.ndarray, sr: int = SR,
                     seconds: float = 0.0) -> np.ndarray:
    """뒤에 **그 녹음 자체의 가장 조용한 부분**을 이어 붙인다.

    🔴 디지털 0 은 실제 무음이 아니다. 진짜 조용한 순간에도 마이크는 룸톤을 담고,
       모델은 그런 오디오로 학습됐다. 순수 0 은 분포 밖 입력이라 0 으로 잰 숫자가
       실기와 다를 수 있다 — append_silence 와 나란히 돌려 대조하려고 있다.
    """
    if seconds <= 0:
        return audio
    need = int(round(sr * seconds))
    win = max(1, int(sr * 0.1))
    if len(audio) < win * 2:
        return append_silence(audio, sr, seconds)
    n = len(audio) // win
    frames = audio[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    quiet = frames[int(np.argmin(rms))]
    if not quiet.any():                             # 진짜 0 뿐이면 0 으로
        return append_silence(audio, sr, seconds)
    # 앞뒤를 번갈아 이어 붙여 100ms 주기의 인공적인 반복음이 생기지 않게 한다
    tiles = [quiet if i % 2 == 0 else quiet[::-1] for i in range(need // win + 1)]
    return np.concatenate([audio, np.concatenate(tiles)[:need].astype(audio.dtype)])


def make_incomplete(audio: np.ndarray, sr: int = SR,
                    fraction: float = 0.5) -> np.ndarray | None:
    """'아직 말하는 중'인 표본. 말소리 구간의 fraction 지점에서 자른다.

    ⚠️ 파일 길이 기준으로 자르면 안 된다 — 뒤 무음이 길수록 이미 끝난 지점을 잘라,
       '미완'이라 이름 붙인 표본이 실은 완결이 된다. 그러면 오탐률이 거짓이 된다.
    """
    start, end = speech_span(audio, sr)
    if start == end:
        return None
    cut = start + int((end - start) * fraction)
    return audio[:cut] if cut > start else None


def internal_gaps(audio: np.ndarray, sr: int = SR, floor_db: float = -40.0,
                  frame_ms: float = 20.0) -> list[float]:
    """발화 **안쪽**의 쉼 길이들(초). 앞뒤 녹음 여백은 세지 않는다.

    🔴 Smart Turn 이 안 먹혀도 고정 꼬리를 줄이는 길은 남는다. 지금 1.2s 를 쓰는
       이유는 '아이가 문장 중간에 쉬어도 안 끊기려고' 인데, 그 쉼이 실제로 얼마나
       긴지는 재본 적이 없다. 재보면 꼬리를 얼마까지 줄여도 되는지 데이터로 정해진다.
    ⚠️ 앞뒤 무음은 쉼이 아니라 녹음 여백이다 — 세면 값이 통째로 부풀려진다.
    """
    start, end = speech_span(audio, sr, floor_db, frame_ms)
    if start == end:
        return []
    core = audio[start:end]
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(core) // hop
    if n < 3:
        return []
    rms = np.sqrt(np.mean(core[: n * hop].reshape(n, hop).astype(np.float64) ** 2,
                          axis=1))
    peak = rms.max()
    if peak <= 0:
        return []
    quiet = 20 * np.log10(rms / peak + 1e-12) <= floor_db
    gaps, run = [], 0
    for q in quiet:
        if q:
            run += 1
        elif run:
            gaps.append(run * hop / sr)
            run = 0
    if run:                      # speech_span 안쪽이라 끝에 남은 것도 내부 쉼이다
        gaps.append(run * hop / sr)
    return gaps


# ── 집계 ─────────────────────────────────────────────────────────────────────
# 양성 = '말이 끝났다(complete)'. 벤더 표와 같은 정의라야 나란히 비교된다.

def summarize(rows: list[dict], threshold: float = 0.5) -> dict:
    comp = [r for r in rows if r["kind"] == "complete"]
    inc = [r for r in rows if r["kind"] == "incomplete"]
    hit = sum(1 for r in comp if r["prob"] > threshold)
    false = sum(1 for r in inc if r["prob"] > threshold)
    total = len(comp) + len(inc)
    return {
        "threshold": threshold,
        "n_complete": len(comp),
        "n_incomplete": len(inc),
        "recall": hit / len(comp) if comp else None,
        "fpr": false / len(inc) if inc else None,
        "accuracy": (hit + len(inc) - false) / total if total else None,
    }


# ── 모델 ─────────────────────────────────────────────────────────────────────

class SmartTurn:
    """ONNX 래퍼. 전처리는 공식 inference.py 와 한 글자도 다르지 않게 맞춘다."""

    def __init__(self, onnx_path: Path, intra_threads: int = 2) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        # 🔴 intra_op 를 **반드시 지정한다.** 노트북 CPU 실측(같은 입력 20회 중앙):
        #      기본값(전 코어)  199.9ms      <- 공식 예제엔 이 줄이 없다
        #      intra=1           71.1ms
        #      intra=2           43.5ms      <- 채택
        #      intra=4           54.2ms
        #    코어를 다 주면 스레드 경합으로 오히려 4배 느려진다. 게다가 운영에선
        #    STT·TTS 가 같은 CPU 를 쓰므로 여기서 코어를 독점하면 남을 굶긴다.
        # ⚠️ 이 값은 노트북 기준이다. 젯슨(ARM 6코어)에서 다시 재고 정할 것.
        so.intra_op_num_threads = intra_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(onnx_path), sess_options=so)
        self._fx = WhisperFeatureExtractor(chunk_length=int(WINDOW_S))

    def probability(self, audio: np.ndarray, sr: int = SR) -> float:
        feats = self._fx(
            fit_to_window(audio, sr),
            sampling_rate=sr,
            return_tensors="np",
            padding="max_length",
            max_length=int(WINDOW_S * sr),
            truncation=True,
            do_normalize=True,
        ).input_features
        feats = np.expand_dims(np.asarray(feats).squeeze(0).astype(np.float32), 0)
        return float(self._sess.run(None, {"input_features": feats})[0][0].item())


# ── 실행 ─────────────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio / peak).astype(np.float32) if peak > 1.0 else audio


def _pct(x: float | None) -> str:
    return "  n/a  " if x is None else f"{x:7.2%}"


def _one_in(x: float | None) -> str:
    return "?" if not x else f"{1 / x:.0f}"


def _recall_at_capped_fpr(rows: list[dict], cap: float) -> tuple[float, float, float]:
    """오탐을 cap 이하로 묶었을 때 쓸 수 있는 최고 재현율과 그 문턱.

    운영 질문이 정확히 이 모양이다 — "아이 말을 100번에 5번까지만 끊는다면,
    그 대신 몇 %의 턴에서 시간을 벌 수 있나?"
    """
    # 🔴 못 지키면 (0.0, None, None) — '불가'다. 예전엔 초기값 (0.0, 1.0, 1.0) 을
    #    그대로 돌려줘 표에 '오탐 100%' 로 찍혔는데, 그건 잰 값이 아니라 초기값이다.
    best: tuple[float, float | None, float | None] = (0.0, None, None)
    for th in [i / 100 for i in range(1, 100)]:
        m = summarize(rows, th)
        if m["fpr"] is None or m["recall"] is None:
            continue
        if m["fpr"] <= cap and m["recall"] > best[0]:
            best = (m["recall"], m["fpr"], th)
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smart Turn 오프라인 채점")
    p.add_argument("--audio", required=True, help="wav 가 있는 폴더")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fractions", default="0.4,0.6,0.8",
                   help="미완 표본을 만들 지점(말소리 구간 대비 비율)")
    p.add_argument("--sweep", default="0.2,0.4,0.6,0.8,1.0",
                   help="'이만큼 조용해진 순간'들. 완결·미완 **양쪽에 똑같이** 붙인다")
    p.add_argument("--pad", choices=("zero", "room"), default="zero",
                   help="붙일 무음의 종류. room = 그 녹음의 룸톤(실제에 가깝다)")
    p.add_argument("--baseline-tail", type=float, default=1.2,
                   help="지금 쓰는 고정 꼬리(초). 이득 계산의 기준")
    p.add_argument("--fpr-cap", type=float, default=0.05,
                   help="허용할 말끊김 비율")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--json", type=Path, help="원자료를 여기에 저장")
    a = p.parse_args(argv)

    root = Path(a.audio).expanduser()
    wavs = sorted(root.rglob("*.wav"))
    if a.limit:
        wavs = wavs[: a.limit]
    if not wavs:
        print(f"wav 가 없다: {root}", file=sys.stderr)
        return 1
    if not a.model.exists():
        print(f"모델이 없다: {a.model}", file=sys.stderr)
        return 1

    fractions = [float(x) for x in a.fractions.split(",") if x.strip()]
    sweep = [float(x) for x in a.sweep.split(",") if x.strip()]
    pad_fn = append_room_tone if a.pad == "room" else append_silence
    print(f"모델 {a.model.name} / wav {len(wavs)}개 / 절단점 {fractions}"
          f" / 무음종류 {a.pad}")
    print(f"무음을 맞춰 비교한다 — 완결·미완 둘 다 같은 길이로 조용해진 순간에 묻는다.")

    model = SmartTurn(a.model)
    rows: list[dict] = []
    took: list[float] = []
    max_gap: list[float] = []      # 발화별 '가장 긴 내부 쉼' — 꼬리를 줄일 여지
    skipped = 0

    for i, w in enumerate(wavs, 1):
        try:
            audio = load_wav(w)
        except Exception as e:                      # 손상 파일은 세고 넘어간다
            print(f"  !! {w.name}: {type(e).__name__}", file=sys.stderr)
            skipped += 1
            continue
        start, end = speech_span(audio)
        if start == end:
            skipped += 1
            continue

        # 말끝에 딱 맞춘 것(완결) 과 중간에서 끊은 것(미완). 무음은 아래서 똑같이 붙인다.
        max_gap.append(max(internal_gaps(audio, SR), default=0.0))
        cores = [("complete", 1.0, trim_tail(audio, SR, keep_ms=0.0))]
        for f in fractions:
            cut = make_incomplete(audio, SR, f)
            if cut is not None:
                cores.append(("incomplete", f, cut))

        for kind, frac, core in cores:
            for t in sweep:
                clip = pad_fn(core, SR, t)
                t0 = time.perf_counter()
                prob = model.probability(clip)
                took.append(time.perf_counter() - t0)
                rows.append({"file": w.name, "kind": kind, "fraction": frac,
                             "silence": t, "prob": prob})
        if i % 25 == 0:
            print(f"  ... {i}/{len(wavs)}", flush=True)

    if not rows:
        print("판정할 게 없다 — 전부 무음이거나 읽기 실패", file=sys.stderr)
        return 1

    n_c = len({r["file"] for r in rows if r["kind"] == "complete"})
    print()
    print(f"발화 {n_c}개" + (f" / 건너뜀 {skipped}" if skipped else "")
          + f" / 판정 {len(rows)}회 / 추론 중앙 "
          f"{statistics.median(took) * 1000:.0f}ms")

    if max_gap:
        g = sorted(max_gap)
        print()
        print("" + "=" * 72)
        print("[기준선] 모델 없이 고정 꼬리만 줄이면 — 몇 %의 발화가 중간에 끊기나")
        print("  꼬리T    끊기는 발화        모델 대비 이득")
        for t in sorted({*sweep, a.baseline_tail}):
            cut = sum(1 for x in g if x >= t)
            mark = "  <- 지금" if abs(t - a.baseline_tail) < 1e-9 else ""
            print(f"  {t:4.2f}s   {cut:3d}/{len(g)} = {cut / len(g):6.2%}"
                  f"      -{max(0.0, a.baseline_tail - t):.2f}s{mark}")

    print()
    print("" + "=" * 72)
    print(f"문턱 {a.threshold} 고정 — 그 순간 물었을 때")
    print("  무음T     재현율(시간 범)   오탐률(말 끊음)")
    for t in sweep:
        sub = [r for r in rows if r["silence"] == t]
        m = summarize(sub, a.threshold)
        print(f"  {t:4.2f}s   {_pct(m['recall'])}          {_pct(m['fpr'])}")

    print()
    print("" + "=" * 72)
    print(f"말끊김을 {a.fpr_cap:.0%} 이하로 묶었을 때 — 운영에서 물을 질문")
    print("  무음T     문턱    재현율    실제오탐   기대이득")
    best = None
    for t in sweep:
        sub = [r for r in rows if r["silence"] == t]
        rec, fpr, th = _recall_at_capped_fpr(sub, a.fpr_cap)
        if th is None:
            print(f"  {t:4.2f}s     —         —         —      상한을 지킬 문턱이 없다")
            continue
        gain = rec * max(0.0, a.baseline_tail - t)
        print(f"  {t:4.2f}s   {th:5.2f}   {rec:7.2%}   {fpr:7.2%}   -{gain:.2f}s")
        if best is None or gain > best[0]:
            best = (gain, t, th, rec, fpr)

    print()
    print("" + "-" * 72)
    if best and best[0] > 0.05:
        gain, t, th, rec, fpr = best
        print(f"최선: 무음 {t:.2f}s 에서 문턱 {th:.2f} 로 판정")
        print(f"  -> 턴의 {rec:.1%} 에서 꼬리를 {a.baseline_tail:.2f}s -> {t:.2f}s 로 줄여"
              f" **평균 -{gain:.2f}s**")
        print(f"  -> 대가: 아이 말이 {fpr:.2%} 확률로 끊긴다"
              f" (약 {_one_in(fpr)}턴에 한 번)")
    else:
        print("🔴 쓸 만한 지점이 없다 — 어느 무음에서도 이득이 유의미하지 않다.")
        print(f"   지금의 고정 꼬리 {a.baseline_tail:.2f}s 를 유지하는 게 낫다.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(
            {"model": a.model.name, "pad": a.pad, "sweep": sweep,
             "fractions": fractions,
             "baseline_tail": a.baseline_tail, "rows": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"원자료 -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
