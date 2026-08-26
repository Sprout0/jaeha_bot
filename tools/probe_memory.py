"""무엇이 메모리를 먹는가 — build_pipeline 순서대로 하나씩 올리며 잰다.

🔴 왜 다시 재나 (2026-08-26): 08-19 에 폴백 지연적재로 RSS 를 5,535 -> **3,237MB**
   까지 내려놨는데, 오늘 실기 로그는 로드 직후 **5,337MB** 다. 2,100MB 가 어딘가로
   갔고 시스템이 7,245/7,607MB(95%)까지 찬다. 그 사이 바뀐 것은 TTS TensorRT(08-24)
   와 필러 캐시(08-25)다 — **추측하지 말고 구간별로 잰다.**

⚠️ 젯슨은 CPU/GPU 가 메모리를 공유한다(unified). CUDA·TensorRT 할당도 RSS 에 잡힌다.
⚠️ 봇을 끄고 돌릴 것. 같이 띄우면 두 벌이 올라가 OOM 난다.

사용법:
    conda activate jaeha_bot && cd ~/jaeha_bot
    python tools/probe_memory.py
    python tools/probe_memory.py --no-trt    # TRT 를 끄고 재서 TRT 몫을 가른다
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def sys_mb() -> tuple[float, float]:
    import psutil
    m = psutil.virtual_memory()
    return (m.total - m.available) / 1024 / 1024, m.total / 1024 / 1024


class Steps:
    def __init__(self) -> None:
        self.prev = rss_mb()
        self.rows: list[tuple[str, float, float]] = []
        print(f"{'단계':<34}{'증가':>10}{'누적 RSS':>11}{'시스템':>14}")
        print("-" * 70)
        self.mark("시작(파이썬 + import)")

    def mark(self, name: str) -> None:
        now = rss_mb()
        used, total = sys_mb()
        self.rows.append((name, now - self.prev, now))
        print(f"{name:<34}{now - self.prev:>+9.0f}MB{now:>9.0f}MB"
              f"{used:>8.0f}/{total:.0f}MB")
        self.prev = now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-trt", action="store_true", help="TTS TensorRT 를 끄고 잰다")
    ap.add_argument("--no-filler", action="store_true", help="필러 캐시를 건너뛴다")
    # TRT 엔진 메모리는 프로파일 상한을 따라간다. 운영값 200 은 13.9초 발화용인데
    # 실기 답변은 2~4초다. 줄이면 얼마나 주는지 재려고 둔다.
    # ⚠️ 값을 바꾸면 엔진을 다시 굽는다(확산 72s + 보코더 49s). 캐시 폴더도 나눠 쓴다.
    ap.add_argument("--trt-max-latent", type=int, default=None)
    ap.add_argument("--trt-workspace", type=int, default=None,
                    help="TRT 작업공간 상한(MB). 지금은 미설정 = ORT 기본값")
    ap.add_argument("--trt-targets", default=None,
                    help="쉼표 구분. 예: vector_estimator (보코더 TRT 를 뺀다)")
    ap.add_argument("--no-trt-backup", action="store_true",
                    help="⚠️ 원본 CUDA 세션을 안 들고 있는다 — 프로파일 밖 문장에서 소리가 안 난다")
    ap.add_argument("--stt-compute", default=None,
                    help="whisper compute_type override. 예: int8_float16")
    args = ap.parse_args()

    from app.config import settings  # noqa: E402

    s = Steps()

    from app.stt_module import STTModule  # noqa: E402
    from app.tts_module import TTSModule  # noqa: E402
    from app.vision_module import VisionDetector  # noqa: E402
    s.mark("모듈 import")

    stt_cfg = dict(settings.models.get("stt", {}))
    if args.stt_compute:
        stt_cfg["compute_type"] = args.stt_compute
        print(f"   (whisper compute_type={args.stt_compute})")
    stt = STTModule(**stt_cfg)
    tts_cfg = dict(settings.models.get("tts", {}))
    tag = []
    if args.trt_workspace:
        tts_cfg["trt_max_workspace_mb"] = args.trt_workspace
        tag.append(f"ws{args.trt_workspace}")
    if args.trt_targets:
        tts_cfg["trt_targets"] = tuple(x.strip() for x in args.trt_targets.split(","))
        tag.append("t-" + args.trt_targets.replace(",", "-"))
    if args.no_trt_backup:
        # 엔진 자체엔 영향이 없다 — 캐시를 새로 굽지 않게 tag 에 넣지 않는다.
        tts_cfg["trt_keep_backup"] = False
        print("   (원본 CUDA 세션 안 들고 있음)")
    if tag:
        # 엔진 캐시를 나눠 써야 운영 엔진을 안 덮어쓴다(빌드 조건이 다르면 다시 굽는다).
        tts_cfg["trt_cache"] = "~/.cache/trt_probe_" + "_".join(tag)
        print(f"   ({', '.join(tag)} — 엔진을 새로 구울 수 있다, 2분쯤)")
    if args.no_trt:
        tts_cfg["trt"] = False
        print("   (TRT 끔)")
    elif args.trt_max_latent:
        tts_cfg["trt_max_latent"] = args.trt_max_latent
        # 캐시를 나눠 써야 운영 엔진을 덮어쓰지 않는다.
        tts_cfg["trt_cache"] = f"~/.cache/trt_probe_L{args.trt_max_latent}"
        print(f"   (trt_max_latent={args.trt_max_latent}, 엔진 새로 굽는다 — 2분쯤 걸린다)")
    tts = TTSModule(**tts_cfg)
    VisionDetector(**settings.models.get("vision", {}))
    s.mark("객체 생성(모델 로드 전)")

    stt.load()
    s.mark(f"whisper {settings.models.get('stt', {}).get('model_size', '?')}")

    tts.load()
    s.mark("Supertonic TTS" + (" (TRT 끔)" if args.no_trt else " (+TensorRT)"))

    from app.agent import LLMAgent  # noqa: E402
    llm_cfg = dict(settings.models["llm"])
    mp = llm_cfg.pop("model_path")
    agent = LLMAgent(model_path=mp, system_prompt=settings.prompts["system"], **llm_cfg)
    agent.warm()
    s.mark("LLM warm(원격이면 GGUF 안 올림)")

    if not args.no_filler:
        try:
            from app.main import _build_filler   # 운영과 같은 방식으로 만든다
            f = _build_filler()
            if f is not None:
                f.ensure(tts)
                s.mark("필러 캐시")
        except Exception as e:
            print(f"   필러 건너뜀: {type(e).__name__}: {e}")

    from app.audio_source import AudioSource  # noqa: E402
    from app.wake import make_detector  # noqa: E402
    src = AudioSource()
    make_detector(settings.models.get("wake", {}), stt, src)
    s.mark("호출어 감지기(ONNX)")

    print("-" * 70)
    used, total = sys_mb()
    print(f"{'합계':<34}{'':>10}{rss_mb():>9.0f}MB{used:>8.0f}/{total:.0f}MB")

    print("\n큰 것부터:")
    for name, delta, _ in sorted(s.rows, key=lambda r: -r[1])[:5]:
        if delta > 1:
            print(f"  {delta:>7.0f}MB  {name}")

    print("""
비교 기준 (2026-08-19 실측, 폴백 지연적재 적용 후):
  whisper medium (CUDA)   1,881MB
  Supertonic (CUDA)       1,321MB   <- TensorRT 이전 값이다
  VisionDetector              0MB
  EXAONE GGUF 폴백        2,319MB   <- 원격이 살아 있으면 안 올라간다
  로드 직후 RSS           3,237MB   <- 오늘 실기는 5,337MB 였다""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
