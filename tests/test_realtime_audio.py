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
