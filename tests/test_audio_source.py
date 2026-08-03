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


class DrainableSource(FakeSource):
    """입력 버퍼에 pending 프레임이 쌓여 있는 상황을 흉내낸다.

    실제 sd.InputStream 의 read_available 처럼, 읽을 때마다 남은 양이 줄어든다.
    """

    def __init__(self, frames, pending=0, **kw):
        super().__init__(frames, **kw)
        self.pending = pending

    def _available(self):
        return self.pending * self.frame

    def _read_frame(self):
        if self.pending:
            self.pending -= 1
        return super()._read_frame()


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


# ------------------------------------------------------------------ drain()
# drain 은 '봇이 자기 목소리를 듣는' 오작동을 막는 장치다. 조용히 안 돌면 증상이
# 호출어 오작동으로 나타나 원인을 찾기 어렵다 → 실제로 버리는지 값으로 확인한다.


def test_drain_discards_buffered_audio_so_next_read_is_fresh():
    # 프레임 0~3 = 봇이 말하는 동안 쌓인 자기 목소리, 4 = 그 뒤 들어온 진짜 입력.
    src = DrainableSource(_ramp(10, 0), pending=4)
    dropped = src.drain()
    assert dropped == 4 * FRAME, f"쌓인 4프레임을 버려야 함: {dropped}"
    assert int(src.read()[0]) == 4, "drain 뒤엔 버린 다음 프레임부터 읽어야 함"


def test_drain_empties_preroll():
    # 낡은 오디오가 호출어 프리롤로 쓰이면 안 된다.
    src = DrainableSource(_ramp(10, 0), pending=0, preroll=0.5)
    for _ in range(3):
        src.read()
    assert src.preroll().size == 3 * FRAME
    src.drain()
    assert src.preroll().size == 0


def test_drain_with_empty_buffer_reads_nothing():
    src = DrainableSource(_ramp(10, 0), pending=0)
    assert src.drain() == 0
    assert src._i == 0, "버릴 게 없으면 프레임을 읽어선 안 됨(다음 입력을 삼킨다)"


def test_drain_leaves_partial_frame_alone():
    # 한 프레임을 못 채우는 잔량(79ms 이하)은 남긴다 — 읽으면 블로킹되기 때문.
    src = DrainableSource(_ramp(10, 0), pending=0)
    src._available = lambda: FRAME - 1
    assert src.drain() == 0
    assert src._i == 0


def test_drain_without_open_stream_returns_zero():
    # 스트림을 연 적 없는 객체에서 불러도 예외 없이 0.
    assert AudioSource().drain() == 0


def test_sample_rate_and_frame_constants():
    assert SAMPLE_RATE == 16000
    assert FRAME == 1280  # openWakeWord 규격: 80ms
