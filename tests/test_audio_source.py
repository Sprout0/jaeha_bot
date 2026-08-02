"""AudioSource 링버퍼·프리롤 검증. 마이크 불필요(_read_frame 오버라이드)."""
import numpy as np
import pytest

from app.audio_source import AudioSource, FRAME, SAMPLE_RATE


class FakeSource(AudioSource):
    """프레임을 미리 정해둔 시퀀스로 공급한다. 마이크를 열지 않는다."""

    def __init__(self, frames, **kw):
        super().__init__(**kw)
        self._frames = list(frames)
        self._i = 0
        self.noise_floor = 0.0

    def _read_frame(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        return f


def _ramp(n, start):
    """구분 가능한 프레임 n개. i번째 프레임은 전부 (start+i) 값."""
    return [np.full(FRAME, float(start + i), dtype=np.float32) for i in range(n)]


def test_read_returns_one_frame():
    src = FakeSource(_ramp(3, 0))
    f = src.read()
    assert f.shape == (FRAME,)
    assert f.dtype == np.float32


def test_preroll_holds_configured_duration():
    # preroll 0.5s = 0.5*16000/1280 = 6.25 -> 6 프레임
    src = FakeSource(_ramp(20, 0), preroll=0.5)
    for _ in range(20):
        src.read()
    pre = src.preroll()
    assert pre.size == 6 * FRAME


def test_preroll_keeps_most_recent_frames():
    src = FakeSource(_ramp(20, 0), preroll=0.5)
    for _ in range(20):
        src.read()
    pre = src.preroll().reshape(-1, FRAME)
    # 마지막 6개 프레임은 값 14,15,16,17,18,19
    assert [int(row[0]) for row in pre] == [14, 15, 16, 17, 18, 19]


def test_preroll_shorter_when_not_yet_full():
    src = FakeSource(_ramp(2, 0), preroll=0.5)
    src.read()
    src.read()
    assert src.preroll().size == 2 * FRAME


def test_clear_preroll_empties_buffer():
    src = FakeSource(_ramp(10, 0), preroll=0.5)
    for _ in range(10):
        src.read()
    src.clear_preroll()
    assert src.preroll().size == 0


def test_sample_rate_and_frame_constants():
    assert SAMPLE_RATE == 16000
    assert FRAME == 1280  # openWakeWord 규격: 80ms
