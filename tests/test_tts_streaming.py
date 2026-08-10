"""OpenAI TTS 스트리밍 재생. 네트워크·오디오 장치 불필요.

왜 스트리밍인가: 통짜로 받으면 첫 소리 = 다운로드 완료 시각이다(젯슨 실측 1.25s).
조각이 오는 대로 틀면 0.6초 부근까지 내려간다.

여기서 지키려는 것:
- PCM 은 16bit 라 **2바이트가 한 샘플**이다. 조각 경계가 샘플을 반으로 가르면
  다음 조각과 이어 붙여야 한다. 안 그러면 좌우로 밀린 잡음이 난다.
- ReSpeaker 는 16kHz 전용인데 API 는 24kHz 다. 조각마다 따로 리샘플하면 경계에서
  튀므로 상태를 이어가는 리샘플러를 써야 한다.
- 🔴 **소리가 한 번 나가면 되돌릴 수 없다.** 폴백 판단은 첫 소리가 나기 전에 끝나야 한다.
"""
import sys
import types

import numpy as np
import pytest

from app.tts_module import TTSModule

PCM_RATE = 24000


def _pcm(values: list[int]) -> bytes:
    return np.array(values, dtype="<i2").tobytes()


class FakeSink:
    """장치 대신 받아 적는다. rate 와 받은 프레임 순서를 검사한다."""

    def __init__(self, rate):
        self.rate = rate
        self.frames = []
        self.closed = False

    def write(self, frames):
        self.frames.append(np.asarray(frames))

    def close(self):
        self.closed = True

    @property
    def all(self):
        return np.concatenate(self.frames) if self.frames else np.zeros(0, dtype=np.float32)


def _tts(**kw):
    t = TTSModule(backend="openai", **kw)
    t.sample_rate = 44100
    t._tts = object()
    t._local_calls = []

    def fake_local(text):
        t._local_calls.append(text)
        return np.full(1000, 0.25, dtype=np.float32)

    t._infer_local = fake_local
    return t


# ------------------------------------------------------- PCM 바이트 -> float 프레임
def test_pcm_bytes_become_float_frames():
    tts = _tts()

    out = np.concatenate(list(tts._iter_pcm_floats([_pcm([0, 16384, -16384])])))

    assert out.shape == (3,)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5, abs=1e-4)
    assert out[2] == pytest.approx(-0.5, abs=1e-4)


def test_odd_byte_is_carried_to_the_next_chunk():
    # 한 샘플(2바이트)이 조각 경계에서 갈린 경우 — 이어 붙여야 값이 보존된다.
    raw = _pcm([16384, -16384])
    tts = _tts()

    out = np.concatenate(list(tts._iter_pcm_floats([raw[:1], raw[1:3], raw[3:]])))

    assert out.shape == (2,), f"샘플이 유실·중복되면 안 된다: {out}"
    assert out[0] == pytest.approx(0.5, abs=1e-4)
    assert out[1] == pytest.approx(-0.5, abs=1e-4)


def test_trailing_odd_byte_is_dropped_without_crashing():
    tts = _tts()

    out = list(tts._iter_pcm_floats([_pcm([100]) + b"\x01"]))

    assert np.concatenate(out).shape == (1,), "끝에 남은 1바이트는 버린다(터지면 안 됨)"


def test_empty_chunks_are_skipped():
    tts = _tts()

    out = list(tts._iter_pcm_floats([b"", _pcm([100]), b""]))

    assert sum(len(a) for a in out) == 1


# ------------------------------------------------------------------ 재생 경로
def _stream_bytes(n_samples, chunk=512):
    raw = _pcm(list(np.zeros(n_samples, dtype=int) + 8000))
    return [raw[i:i + chunk] for i in range(0, len(raw), chunk)]


def test_streaming_writes_every_frame_to_the_device():
    tts = _tts()
    sinks = []

    def make_sink(rate):
        s = FakeSink(rate)
        sinks.append(s)
        return s

    ok = tts._speak_streaming(_stream_bytes(2400), PCM_RATE, make_sink)

    assert ok is True
    assert sinks[0].all.size == 2400, "받은 샘플을 다 흘려보내야 한다"
    assert sinks[0].closed, "스트림을 닫아야 장치가 풀린다"


def test_streaming_reports_failure_when_no_audio_arrives():
    # 소리가 한 조각도 안 왔으면 아직 폴백할 수 있다 — 그게 이 반환값의 의미다.
    tts = _tts()

    ok = tts._speak_streaming(iter([]), PCM_RATE, lambda rate: FakeSink(rate))

    assert ok is False


def test_streaming_reports_failure_when_stream_raises_before_first_sound():
    def boom():
        raise RuntimeError("connection reset")
        yield b""      # pragma: no cover

    tts = _tts()

    ok = tts._speak_streaming(boom(), PCM_RATE, lambda rate: FakeSink(rate))

    assert ok is False, "첫 소리 전 실패는 폴백 가능해야 한다"


def test_resampler_flush_emits_the_tail():
    """마지막 조각을 넣은 뒤 리샘플러 안에 남은 샘플을 꺼내야 말끝이 안 잘린다.

    soxr.ResampleStream 은 내부 버퍼를 들고 있어서 last=True 로 비워 주지 않으면
    젯슨 실측 기준 12ms 가 사라진다(2초 발화에서 193샘플). 들릴락 말락 한 양이지만
    매 발화마다 말끝이 조금씩 깎이는 건 고쳐 두는 게 맞다.
    """
    from app.tts_module import _StreamResampler

    src, dst = 24000, 16000
    sig = np.sin(2 * np.pi * 440 * np.arange(src, dtype=np.float32) / src).astype(np.float32)
    r = _StreamResampler(src, dst)
    out = [r(sig[i:i + 2048]) for i in range(0, len(sig), 2048)]
    out.append(r.flush())

    total = sum(a.size for a in out)
    assert abs(total - dst) <= dst * 0.002, \
        f"1초를 넣었으면 {dst} 샘플이 나와야 한다(±0.2%): {total}"


def test_device_opens_at_playable_rate_not_api_rate(monkeypatch):
    # ReSpeaker 는 16kHz 전용이다. 24kHz 로 열면 재생이 실패한다.
    tts = _tts()
    tts._play_rate = 16000
    sinks = []

    ok = tts._speak_streaming(_stream_bytes(2400), PCM_RATE,
                              lambda rate: sinks.append(FakeSink(rate)) or sinks[-1])

    assert ok is True
    assert sinks[0].rate == 16000, f"장치가 받는 레이트로 열어야 한다: {sinks[0].rate}"
    # 24k -> 16k 면 길이가 2/3 로 줄어든다(±2% 는 리샘플러 지연 여유).
    assert abs(sinks[0].all.size - 1600) < 1600 * 0.02, sinks[0].all.size
