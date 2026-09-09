"""tools/record_noise.py — 긴 소음 녹음의 '살아 있나' 판정 시험.

이 도구가 막으려는 사고는 하나다: **30~60분을 녹음했는데 전부 무음.**
젯슨의 죽은 `default` 장치는 에러 없이 0 을 준다(2026-08-24 실측). 그걸
끝나고 알면 저녁 하나가 통째로 날아간다.
"""
import numpy as np
import pytest

from tools.record_noise import SILENT_RMS, block_rms, verdict


def test_무음_블록은_죽은_마이크로_본다():
    assert not verdict([0.0, 0.0, 0.0])[0]


def test_소리가_있으면_통과한다():
    ok, msg = verdict([0.01, 0.02, 0.015])
    assert ok
    assert msg == ""


def test_한_블록만_살아도_통과시키지_않는다():
    """중간에 장치가 죽는 경우가 있다 — 전체가 조용하면 못 쓴다."""
    ok, _ = verdict([0.02] + [0.0] * 50)
    assert not ok


def test_경계값에서_살았다고_하지_않는다():
    assert not verdict([SILENT_RMS * 0.9])[0]


def test_블록이_없으면_통과가_아니다():
    assert not verdict([])[0]


def test_rms_는_진폭을_따라간다():
    quiet = block_rms(np.full(100, 0.001, dtype=np.float32))
    loud = block_rms(np.full(100, 0.1, dtype=np.float32))
    assert loud > quiet
    assert quiet == pytest.approx(0.001, rel=1e-3)


def test_워밍업_한_블록에_속아서_계속하지_않는다():
    """장치가 열리는 순간 값이 임계를 살짝 넘는 일이 있다(노트북 실측).

    그 한 블록 때문에 45분을 무음으로 채우면 안 된다.
    """
    from tools.record_noise import should_abort
    warm = [0.00039] + [0.00001] * 2      # 30초치
    assert should_abort(warm, block_s=10.0)


def test_30초가_되기_전에는_접지_않는다():
    from tools.record_noise import should_abort
    assert not should_abort([0.0], block_s=10.0)


def test_소리가_있으면_계속한다():
    from tools.record_noise import should_abort
    assert not should_abort([0.02] * 5, block_s=10.0)
