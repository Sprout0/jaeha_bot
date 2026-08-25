"""호출어 후보가 **whisper 에게 읽히는 단어인가**를 잰다 — 2단계 검증 설계의 선행 실험.

왜 필요한가 (2026-08-25):
  '재하봇'은 whisper 에 없는 고유명사(OOV)라 힌트 없이는 41% 밖에 안 읽혔다.
  그래서 initial_prompt 를 줬는데, 그게 유튜브 소리에도 '재하봇'을 만들어 냈다(환각).
  환각을 막으려고 **2패스**(힌트없이 + 힌트주고)를 돌려야 했고, 검증 1회가 1.7초가 됐다.
  그 지연 때문에 1단계 임계값을 0.10 아래로 못 내렸고, 실기 재현율이 41% 에서 멈췄다.

  ➡️ 후보 단어를 **힌트 없이** whisper 가 읽어 준다면 이 사슬이 통째로 끊긴다.
     2패스 → 1패스, 검증 1.7s → 0.85s, 임계값을 더 열 수 있다.

무엇을 재나:
  Supertonic 으로 후보 문구를 화자·속도·시드를 흔들어 합성하고, 각 클립을
  ① 힌트 없이 ② 힌트를 주고 두 번 전사해 호출어와의 자모거리를 본다.
  '재하봇'을 **같은 조건으로 같이 돌려 대조군**으로 쓴다 — 절대값이 아니라 차이를 봐야 한다.

한계(정직하게):
  합성음이다. 실제 사람 목소리·실제 방 소음에서의 값이 아니다.
  여기서 지면 실음성에서도 진다(합성음이 더 또렷하다). 여기서 이겨도 실음성 확인은 필요하다.

사용:
  python tools/probe_wake_word.py --words "하이 티드,하이 테드,재하봇" -n 24
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_models          # noqa: E402
from app.stt_module import STTModule        # noqa: E402
from app.tts_module import TTSModule        # noqa: E402
from app.wake import best_wake_ratio        # noqa: E402

TARGET_SR = 16000
VOICES = ["F1", "F2", "F3", "F5", "M1", "M2", "M3", "M5", "F2+F5", "M1+M3", "F1+M2", "M2+M5"]
SPEEDS = [0.85, 0.95, 1.05]

# 운영과 같은 컷(configs/model_paths.yaml 의 wake.onnx.verify)
PLAIN_CUT = 0.65   # ① 힌트 없이 들었을 때
HINT_CUT = 0.45    # ② 힌트 주고 들었을 때


def hint_for(phrase: str) -> str:
    """검증용 문장형 힌트. '재하봇아, 재하봇이, 재하봇 불러.' 와 같은 모양을 만든다."""
    return f"{phrase}야, {phrase}아, {phrase} 불러."


def main() -> int:
    ap = argparse.ArgumentParser(description="호출어 후보의 whisper 가독성 측정")
    ap.add_argument("--words", default="하이 티드,하이 테드,재하봇",
                    help="쉼표로 구분한 후보 문구. 매칭어는 공백을 뺀 것을 쓴다.")
    ap.add_argument("-n", "--count", type=int, default=24, help="문구당 클립 수")
    ap.add_argument("--device", default=None, help="whisper 장치(기본: 설정값)")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--save-dir", default=None, help="합성 클립을 남길 디렉터리(선택)")
    ap.add_argument("--json-out", default=None, help="클립별 원자료 JSON 경로")
    args = ap.parse_args()

    phrases = [p.strip() for p in args.words.split(",") if p.strip()]
    models = load_models()
    tcfg, scfg = models.get("tts", {}), models.get("stt", {})

    tts = TTSModule(model=tcfg.get("model", "supertonic-3"), voice=VOICES[0],
                    language=tcfg.get("language", "ko"),
                    total_steps=int(tcfg.get("total_steps", 24)),
                    threads=int(tcfg.get("threads", 4)),
                    providers=["CPUExecutionProvider"])
    tts.load()
    src_sr = tts.sample_rate

    # 🔴 keywords/aliases 를 비워 둔다 — correct_stt 가 결과를 호출어로 스냅하면
    #    '읽혔는가'가 아니라 '스냅됐는가'를 재게 된다.
    stt = STTModule(model_size=scfg.get("model_size", "medium"),
                    device=args.device or scfg.get("device", "cpu"),
                    compute_type=args.compute_type or scfg.get("compute_type", "int8"),
                    language=scfg.get("language", "ko"),
                    keywords=[], aliases={})
    stt.load()
    print(f"TTS {src_sr}Hz / STT {stt.model_size} {stt.device}/{stt.compute_type}\n")

    rows, raw = [], []
    for phrase in phrases:
        word = phrase.replace(" ", "")
        prompt = hint_for(phrase)
        plain_r, hint_r, samples = [], [], []
        for i in range(args.count):
            voice = VOICES[i % len(VOICES)]
            speed = SPEEDS[(i // len(VOICES)) % len(SPEEDS)]
            tts._style = tts._resolve_style(voice)
            tts.speed = speed
            tts.seed = args.seed_base + i
            audio = tts._trim(tts._infer(phrase), keep_tail=0.15)
            audio = tts._resample(audio, src_sr, TARGET_SR)
            peak = float(np.abs(audio).max()) if audio.size else 0.0
            if peak > 0:
                audio = (audio / peak * 0.95).astype(np.float32)
            if args.save_dir:
                import soundfile as sf
                os.makedirs(args.save_dir, exist_ok=True)
                sf.write(os.path.join(args.save_dir, f"{word}_{i:03d}.wav"), audio, TARGET_SR)

            plain, _ = stt.transcribe(audio, initial_prompt=None)
            hinted, _ = stt.transcribe(audio, initial_prompt=prompt)
            rp = best_wake_ratio(plain or "", word)
            rh = best_wake_ratio(hinted or "", word)
            plain_r.append(rp)
            hint_r.append(rh)
            if len(samples) < 6:
                samples.append((plain or "∅", rp))
            raw.append({"word": word, "voice": voice, "speed": speed,
                        "plain": plain or "", "plain_ratio": rp,
                        "hinted": hinted or "", "hint_ratio": rh})
            print(f"  [{word}] {i + 1:2}/{args.count} {voice:6} x{speed:.2f}  "
                  f"힌트없이 {rp:.2f} '{(plain or '∅')[:20]}'  힌트주고 {rh:.2f}")

        pa, ha = np.array(plain_r), np.array(hint_r)
        rows.append((phrase, pa, ha, samples))

    print()
    print("=" * 92)
    print(f"{'문구':10} {'중앙':>7} {'p90':>7} {'최대':>7} "
          f"{'≤0.65':>7} {'≤0.45':>7} {'≤0.25':>7}  {'힌트주고중앙':>10}")
    print("-" * 92)
    for phrase, pa, ha, _ in rows:
        print(f"{phrase:10} {np.median(pa):>7.3f} {np.percentile(pa, 90):>7.3f} {pa.max():>7.3f} "
              f"{100 * (pa <= 0.65).mean():>6.0f}% {100 * (pa <= 0.45).mean():>6.0f}% "
              f"{100 * (pa <= 0.25).mean():>6.0f}%  {np.median(ha):>10.3f}")
    print("=" * 92)
    if args.json_out:
        import json
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=1)
        print(f"원자료 {len(raw)}건 -> {args.json_out}")
    print("\n[힌트 없이 whisper 가 실제로 뭐라고 적었나]")
    for phrase, _, _, samples in rows:
        print(f"  {phrase}:")
        for txt, r in samples:
            print(f"      {r:.2f}  {txt[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
