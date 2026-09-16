import subprocess
import sys

import pytest

from app.main_realtime import build_instructions, check_startup


def _models(mode="embed"):
    return {"wake": {"enabled": True, "detector": "onnx", "onnx": {"verify": {"mode": mode}}}}


def test_키가_없으면_기동하지_않는다():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        check_startup(_models(), {})


def test_호출어_검증이_임베딩이_아니면_기동하지_않는다():
    # 🔴 전면 API 에는 whisper 가 없다 — whisper 검증이면 2단계 없이 돈다.
    with pytest.raises(RuntimeError, match="embed"):
        check_startup(_models("whisper"), {"OPENAI_API_KEY": "k"})


def test_호출어를_안_쓰면_검증_모드는_안_본다():
    assert check_startup(_models("whisper"), {"OPENAI_API_KEY": "k"}, wake=False) == "k"


def test_정상이면_키를_돌려준다():
    assert check_startup(_models(), {"OPENAI_API_KEY": "k"}) == "k"


def test_노래가_켜지면_프롬프트를_바꾼다():
    from app.config import settings
    assert build_instructions(False) == settings.prompts["system"]
    assert build_instructions(True) != settings.prompts["system"]


def test_로컬_STT_TTS_를_싣지_않는다():
    code = ("import sys, app.main_realtime; "
            "bad=[m for m in ('app.stt_module','app.tts_module','faster_whisper') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
