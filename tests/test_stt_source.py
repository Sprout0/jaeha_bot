"""record_until_silence 의 공유 스트림·prefix 경로. 마이크·모델 불필요."""
import numpy as np
import pytest

from app.audio_source import FRAME
from app.stt_module import STTModule


class ScriptedSource:
    def __init__(self, frames, noise_floor=0.001):
        self._frames = list(frames)
        self._i = 0
        self.noise_floor = noise_floor

    def read(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        return f


def _loud(n):
    return [np.full(FRAME, 0.3, dtype=np.float32) for _ in range(n)]


def _quiet(n):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n)]


def test_prefix_is_included_in_output():
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    prefix = np.full(FRAME * 2, 0.4, dtype=np.float32)
    src = ScriptedSource(_loud(3) + _quiet(30))
    out = stt.record_until_silence(source=src, prefix=prefix)
    assert out.size >= prefix.size, "prefix 가 결과에 포함되어야 함"


def test_prefix_path_skips_start_wait():
    """prefix 가 있으면 '말 시작 대기' 없이 곧바로 무음 판정으로 간다."""
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    prefix = np.full(FRAME, 0.4, dtype=np.float32)
    # 처음부터 조용해도 prefix 덕에 녹음이 성립하고 곧 종료된다.
    src = ScriptedSource(_quiet(30))
    out = stt.record_until_silence(source=src, prefix=prefix)
    assert out.size > 0


def test_source_without_prefix_waits_for_speech():
    stt = STTModule(silence_duration=0.3, max_duration=5.0, start_timeout=1.0)
    src = ScriptedSource(_quiet(5) + _loud(3) + _quiet(30))
    out = stt.record_until_silence(source=src)
    assert out.size > 0, "말이 시작되면 녹음돼야 함"


def test_source_returns_empty_when_frames_exhausted():
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    src = ScriptedSource(_quiet(3))
    out = stt.record_until_silence(source=src)
    assert out.size == 0


def test_signature_still_accepts_no_arguments():
    """기존 호출부 호환 — 인자 없이 부를 수 있어야 한다(실행은 안 함)."""
    import inspect
    sig = inspect.signature(STTModule.record_until_silence)
    assert sig.parameters["source"].default is None
    assert sig.parameters["prefix"].default is None
