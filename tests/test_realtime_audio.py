import numpy as np

from app.realtime_audio import SR, MicGate, PcmAccumulator, to_pcm16


def test_16k_를_24k_pcm16_으로_바꾼다():
    pcm = to_pcm16(np.zeros(1280, dtype=np.float32), 16000)   # 80ms @16k
    assert len(pcm) == 1920 * 2                                 # 80ms @24k, 샘플당 2바이트


def test_24k_는_그대로_보낸다():
    back = np.frombuffer(to_pcm16(np.full(240, 0.5, dtype=np.float32), SR), dtype="<i2")
    assert back.size == 240 and back[0] == int(0.5 * 32767)


def test_범위를_넘는_값은_잘라서_보낸다():
    pcm = to_pcm16(np.array([2.0, -2.0], dtype=np.float32), SR)
    assert list(np.frombuffer(pcm, dtype="<i2")) == [32767, -32767]


def test_라이브_루프가_옮긴_것을_쓴다():
    import tools.realtime_live as live
    assert live.MicGate is MicGate and live.PcmAccumulator is PcmAccumulator


def _speaker(monkeypatch, rate=24000):
    import app.realtime_audio as ra

    class FakeStream:
        def __init__(self, **kw):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass

    import sounddevice as sd
    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    monkeypatch.setattr(ra, "resolve_rate", lambda kind, want: rate)
    return ra.StreamSpeaker(24000)


def _pull(sp, frames):
    out = np.zeros((frames, 1), dtype=np.float32)
    sp._cb(out, frames, None, None)
    return out[:, 0]


def test_표시_뒤로_재생한_양을_센다(monkeypatch):
    sp = _speaker(monkeypatch)
    sp.push(np.ones(100, np.float32))
    sp.mark()
    sp.push(np.ones(300, np.float32))
    _pull(sp, 250)
    assert sp.played() == 150


def test_표시_뒤_keep_넘는_소리는_버린다(monkeypatch):
    sp = _speaker(monkeypatch)
    sp.mark()
    sp.push(np.full(200, 0.5, np.float32))
    sp.push(np.full(200, 0.5, np.float32))
    _pull(sp, 50)
    sp.truncate(120)
    got = _pull(sp, 400)
    assert np.count_nonzero(got) == 70
    sp.push(np.full(10, 0.9, np.float32))            # 자른 뒤 이어 붙인 문장은 나간다
    assert np.count_nonzero(_pull(sp, 10)) == 10


def test_장치_레이트가_달라도_원본_샘플로_센다(monkeypatch):
    sp = _speaker(monkeypatch, rate=16000)
    sp.mark()
    sp.push(np.ones(2400, np.float32))               # 24k 0.1s → 16k 1600
    _pull(sp, 800)
    assert abs(sp.played() - 1200) <= 2
