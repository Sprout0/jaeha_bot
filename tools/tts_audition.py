"""TTS 목소리를 **젯슨 스피커로 직접** 들어 보는 도구. 반드시 젯슨에서 돌릴 것.

🔴🔴 이 도구가 있는 이유 — 2026-08-10 에 이걸 안 해서 틀렸다.
   OpenAI TTS 후보를 24kHz wav 로 뽑아 노트북으로 들려주고 목소리를 골랐는데,
   실제 재생 장치인 **ReSpeaker 는 16kHz 전용**이라 8kHz 위가 통째로 잘린다.
   coral 을 부드럽게 들리게 하던 고역이 바로 거기 있었다. 파일로는 "coral 이 좋다"였는데
   실기에서는 "이상해졌다, Supertonic 이 낫다"가 나왔다.
   **고른 목소리와 실제로 나오는 목소리가 애초에 달랐다.**

   ➡️ TTS 후보 비교는 파일 청취로 결론내지 말 것. 실제 장치로만 판단한다.

    python tools/tts_audition.py
    python tools/tts_audition.py --text "삐약삐약! 병아리 소리야!"

의성어("멍멍", "삐약삐약")를 꼭 포함해서 들을 것 — 동물 소리 놀이의 핵심이라
여기서 어색하면 놀이 자체가 깨진다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.config import settings  # noqa: E402
from app.tts_module import TTSModule  # noqa: E402

DEFAULT_TEXT = "멍멍! 이건 무슨 동물 소리일까? 재하야, 같이 놀자!"

# (라벨, 덮어쓸 설정). 후보를 늘리려면 여기만 고친다.
CANDIDATES = [
    ("Supertonic F1+F4 (현행)", {"backend": "supertonic"}),
    ("OpenAI coral 스트리밍", {"backend": "openai", "openai_voice": "coral"}),
    ("OpenAI shimmer 스트리밍", {"backend": "openai", "openai_voice": "shimmer"}),
    ("OpenAI nova 스트리밍", {"backend": "openai", "openai_voice": "nova"}),
    ("OpenAI coral 통짜수신", {"backend": "openai", "openai_voice": "coral",
                            "openai_stream": False}),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--auto", action="store_true", help="엔터 없이 연달아 재생")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  [경고] %(message)s")
    # 🔴 젯슨 기본 출력은 pulse('default')라 지정하지 않으면 엉뚱한 데로 나간다.
    from app.main import _setup_audio_device
    _setup_audio_device()

    print(f'문장: "{args.text}"\n')
    for label, over in CANDIDATES:
        tts = TTSModule(**{**settings.models["tts"], **over})
        tts.load()
        tts.warm()
        print(f">>> {label}")
        if not args.auto:
            input("    엔터를 누르면 재생...")
        r = tts.speak(args.text)
        print(f"    첫 소리 {r.first_audio_s:.2f}s / 발화 {r.play_s:.2f}s")
    print("\n어느 것이 가장 자연스러웠나? 속도와 의성어 발음을 특히 볼 것.")


if __name__ == "__main__":
    main()
