"""tools/wake_ab_verify.py — whisper 검증과 임베딩 검증을 같은 자로 재는 지그의 시험.

재려는 것은 검증기 둘이지 이 지그가 아니다. 그래서 지그가 틀리면 안 되는 두 곳만 본다:
  1) 파일을 봇의 AudioSource 와 **같은 버퍼 규칙**으로 흘려 넣는가
  2) 쿨다운 시계가 **녹음 시간 + 실제 검증 시간**으로 가는가 — 여기가 틀리면
     빠른 쪽(임베딩)과 느린 쪽(whisper)이 서로 다른 쿨다운을 받아 비교가 기운다
"""
import numpy as np
import pytest

from app.audio_source import FRAME, SAMPLE_RATE
from tools.wake_ab_verify import AudioClock, FileSource, timed


def _src(n_frames: int, clock=None, verify_window=2.0):
    audio = np.arange(n_frames * FRAME, dtype=np.float32)
    return FileSource(audio, clock or AudioClock(), preroll=0.5,
                      verify_window=verify_window)


def test_파일을_프레임_순서대로_읽고_끝나면_StopIteration():
    s = _src(3)
    got = [s.read() for _ in range(3)]
    assert [int(g[0]) for g in got] == [0, FRAME, 2 * FRAME]
    with pytest.raises(StopIteration):
        s.read()


def test_검증창은_봇과_같은_길이다():
    """AudioSource 를 물려받았으니 검증창 규칙이 봇과 같아야 한다(2초 = 25프레임)."""
    s = _src(40)
    for _ in range(40):
        s.read()
    assert s.verify_window().size == int(2.0 * SAMPLE_RATE / FRAME) * FRAME


def test_시계는_읽은_만큼_녹음_시간으로_간다():
    c = AudioClock()
    s = _src(10, clock=c)
    for _ in range(10):
        s.read()
    assert c() == pytest.approx(10 * FRAME / SAMPLE_RATE)


def test_검증_시간은_바로_다음_한_번에만_더해진다():
    """감지기는 검증기가 돌아온 **바로 다음** time.monotonic() 으로 쿨다운 기준을 잡는다.

    실제 봇에서는 whisper 가 도는 동안 소리가 쌓이고, 그만큼 쿨다운이 늦게 풀린다.
    그 한 번에만 더하고, 그 뒤 '지금' 은 다시 녹음 시간이어야 한다.
    """
    c = AudioClock()
    c.advance(5.0)
    c.bump(1.2)
    assert c() == pytest.approx(6.2)     # 쿨다운 기준 시각
    assert c() == pytest.approx(5.0)     # 그 다음부터는 녹음 시간


def test_timed_는_걸린_시간을_시계에_올린다():
    c = AudioClock()
    ticks = iter([10.0, 11.5])
    f = timed(lambda x: x * 2, c, timer=lambda: next(ticks))
    assert f(3) == 6
    assert c() == pytest.approx(1.5)
    assert f.calls == 1
    assert f.seconds == pytest.approx(1.5)
