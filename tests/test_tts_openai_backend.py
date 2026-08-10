"""TTS OpenAI 백엔드 + Supertonic 폴백. 네트워크·모델 로드 불필요.

여기서 지키려는 것:
- 아이 앞에서 **소리가 안 나는 일은 없어야 한다**. API 가 죽으면 조용히 로컬로 내려간다.
- 다만 '조용히'가 '아무도 모르게'는 아니다 — 폴백은 반드시 경고를 남긴다.
  (2026-08-10 실측: API TTS 는 28턴 중 2턴이 1.7s·2.2s 였다. 드물지만 0 이 아니다.)
- API 오디오(24kHz)를 모듈 샘플레이트로 맞춰야 downstream(트림·재생 리샘플)이 그대로 돈다.
"""
import io
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from app.tts_module import TTSModule

API_RATE = 24000        # OpenAI TTS 출력
MODULE_RATE = 44100     # Supertonic 합성 레이트


def _wav_bytes(seconds: float = 1.0, rate: int = API_RATE) -> bytes:
    """소리가 '있는' wav — 트림이 통째로 잘라내지 않도록 진폭을 준다."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, 0.5 * np.sin(2 * np.pi * 440 * t), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def fake_openai(monkeypatch):
    """openai 패키지를 흉내낸다. calls 에 호출 인자가 쌓이고, fail 을 켜면 예외를 던진다."""
    state = {"calls": [], "fail": None, "audio": _wav_bytes()}

    class FakeSpeech:
        def create(self, **kw):
            state["calls"].append(kw)
            if state["fail"]:
                raise state["fail"]
            return types.SimpleNamespace(content=state["audio"])

    class FakeClient:
        def __init__(self, **kw):
            self.audio = types.SimpleNamespace(speech=FakeSpeech())

    mod = types.ModuleType("openai")
    mod.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", mod)
    return state


def _tts(**kw):
    """Supertonic 을 로드하지 않고 폴백만 관찰할 수 있게 만든 인스턴스."""
    t = TTSModule(**kw)
    t.sample_rate = MODULE_RATE
    t._tts = object()          # load() 가 실제 모델을 불러오지 않게 한다
    t._local_calls = []

    def fake_local(text):
        t._local_calls.append(text)
        return np.full(MODULE_RATE, 0.25, dtype=np.float32)  # 1초짜리 '로컬 소리'

    t._infer_local = fake_local
    return t


# ------------------------------------------------------------------ 백엔드 선택
def test_openai_backend_calls_api_and_returns_audio(fake_openai):
    tts = _tts(backend="openai", openai_voice="shimmer")

    audio = tts._infer("칼은 위험해!")

    assert len(fake_openai["calls"]) == 1, "API 를 불러야 한다"
    assert fake_openai["calls"][0]["voice"] == "shimmer"
    assert fake_openai["calls"][0]["input"] == "칼은 위험해!"
    assert audio.size > 0, "오디오가 나와야 한다"
    assert tts._local_calls == [], "API 가 성공했으면 로컬을 부르지 않는다"


def test_default_backend_stays_supertonic(fake_openai):
    tts = _tts()

    tts._infer("안녕")

    assert fake_openai["calls"] == [], "기본값은 로컬이어야 한다(설정 없이 과금되면 안 됨)"
    assert tts._local_calls == ["안녕"]


def test_instructions_are_sent_when_configured(fake_openai):
    # 말 속도·톤 지시는 gpt-4o-mini-tts 의 핵심 이점이다. 안 실리면 의미가 없다.
    tts = _tts(backend="openai", openai_instructions="아기에게 말하듯 천천히")

    tts._infer("안녕")

    assert fake_openai["calls"][0]["instructions"] == "아기에게 말하듯 천천히"


# -------------------------------------------------------------------- 폴백
def test_falls_back_to_local_when_api_raises(fake_openai):
    fake_openai["fail"] = RuntimeError("connection reset")
    tts = _tts(backend="openai")

    audio = tts._infer("칼은 위험해!")

    assert tts._local_calls == ["칼은 위험해!"], "API 실패 시 로컬이 말해야 한다"
    assert audio.size > 0


def test_fallback_logs_warning(fake_openai, caplog):
    fake_openai["fail"] = RuntimeError("timeout")
    tts = _tts(backend="openai")

    with caplog.at_level("WARNING", logger="jaeha_bot.tts"):
        tts._infer("안녕")

    assert any("폴백" in r.getMessage() for r in caplog.records), \
        f"폴백은 조용해도 되지만 기록은 남아야 한다: {[r.getMessage() for r in caplog.records]}"


def test_missing_api_key_falls_back_without_crashing(monkeypatch):
    # openai 패키지는 있는데 키가 없어 OpenAI() 생성부터 터지는 경우.
    mod = types.ModuleType("openai")

    def boom(**kw):
        raise RuntimeError("api_key must be set")

    mod.OpenAI = boom
    monkeypatch.setitem(sys.modules, "openai", mod)
    tts = _tts(backend="openai")

    audio = tts._infer("안녕")

    assert tts._local_calls == ["안녕"]
    assert audio.size > 0


# ------------------------------------------------------------------- 샘플레이트
def test_api_audio_is_resampled_to_module_rate(fake_openai):
    # API 는 24kHz, 모듈은 44.1kHz. 안 맞추면 재생이 느려지고 음정이 내려간다.
    fake_openai["audio"] = _wav_bytes(seconds=1.0, rate=API_RATE)
    tts = _tts(backend="openai")

    audio = tts._infer("안녕")

    assert abs(audio.size - MODULE_RATE) < MODULE_RATE * 0.02, \
        f"1초 오디오는 모듈 레이트({MODULE_RATE})에 맞춰져야 함: {audio.size}"


def test_empty_text_skips_api_entirely(fake_openai):
    tts = _tts(backend="openai")

    audio = tts._infer("   ")

    assert fake_openai["calls"] == [], "빈 텍스트로 과금하면 안 된다"
    assert audio.size == 0
