"""영상마다 다른 음량을 맞춘다 — 판단만(소리·크로미움 없이). 2026-09-19."""
import numpy as np

from app.loudness import LoudnessLeveler, block_db

SR = 16000


def _blk(db, n=SR // 2):
    """RMS 가 db dBFS 인 0.5초 사인파."""
    amp = 10 ** (db / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)


def _run(lv, db, seconds, t0=0.0):
    out = []
    for i in range(int(seconds * 2)):
        v = lv.feed(_blk(db), t0 + i * 0.5)
        if v is not None:
            out.append(v)
    return out


def test_블록_음량을_dBFS_로():
    rms, peak = block_db(_blk(-20))
    assert abs(rms + 20) < 0.1 and abs(peak + 17) < 0.2


def test_조용한_영상은_키운다():
    lv = LoudnessLeveler(target_db=-19, base=20)
    lv.reset(0.0)
    got = _run(lv, -29, 4)                 # 목표보다 10dB 작다
    assert got and got[0] > 20
    assert abs(got[0] - 20 * 10 ** (10 / 20)) < 2      # 첫 조정은 크게(+10dB 한 번에)


def test_큰_영상은_줄인다():
    lv = LoudnessLeveler(target_db=-19, base=20)
    lv.reset(0.0)
    got = _run(lv, -13, 4)
    assert got and got[0] < 20


def test_목표_근처면_건드리지_않는다():
    lv = LoudnessLeveler(target_db=-19, base=20)
    lv.reset(0.0)
    assert _run(lv, -19.5, 10) == []


def test_시작하고_잠깐은_기다린다():
    # 처음 몇 초는 광고·인트로·버퍼링일 수 있다
    lv = LoudnessLeveler(target_db=-19, base=20, settle_s=3)
    lv.reset(0.0)
    assert _run(lv, -35, 2.5) == []


def test_무음은_판단에_안_쓴다():
    # 일시정지(호출 중)·곡 사이 무음으로 볼륨을 끝까지 올리면 다음 곡이 터진다
    lv = LoudnessLeveler(target_db=-19, base=20)
    lv.reset(0.0)
    assert _run(lv, -70, 10) == []


def test_범위_안에서만():
    lv = LoudnessLeveler(target_db=-19, base=20, lo=5, hi=100)
    lv.reset(0.0)
    got = _run(lv, -60 + 15, 30)           # -45dBFS: 26dB 모자람
    assert max(got) <= 100
    lv.reset(100.0)
    got = _run(lv, 0 - 3, 30, t0=100.0)
    assert min(got) >= 5


def test_두_번째_조정부터는_한_번에_조금씩():
    lv = LoudnessLeveler(target_db=-19, base=20, max_step_db=6, first_step_db=12)
    lv.reset(0.0)
    got = _run(lv, -39, 3.5)               # 20dB 모자람 → 첫 조정 +12dB 까지만
    assert abs(got[0] - 20 * 10 ** (12 / 20)) < 2


def test_새_곡이면_기본값으로_돌아간다():
    lv = LoudnessLeveler(target_db=-19, base=20)
    lv.reset(0.0)
    _run(lv, -29, 4)
    assert lv.volume != 20
    lv.reset(10.0)
    assert lv.volume == 20


def test_봉우리가_깨질_만큼_크면_줄인다():
    lv = LoudnessLeveler(target_db=-19, base=20)
    lv.reset(0.0)
    loud = np.clip(_blk(-19) * 12, -1, 1)      # RMS 는 목표 근처가 아니고, 봉우리 0dBFS
    out = [lv.feed(loud, i * 0.5) for i in range(8)]
    assert any(v is not None and v < 20 for v in out)


def test_첫_조정_뒤에는_키울_땐_천천히_줄일_땐_빠르게():
    # 09-19 젯슨: 아기상어의 조용한 구간(-36.5)에 끌려 44→88 로 한 번에 올랐다.
    # 조용한 구간 뒤 본 노래가 오면 터진다 → 올리는 쪽만 작게.
    lv = LoudnessLeveler(target_db=-19, base=20, max_step_db=6, up_step_db=3)
    lv.reset(0.0)
    first = _run(lv, -25, 4)[0]                                  # 첫 조정
    up = _run(lv, -40, 6, t0=4.0)[0]
    assert abs(20 * np.log10(up / first) - 3) < 0.6
    lv2 = LoudnessLeveler(target_db=-19, base=20, max_step_db=6, up_step_db=3)
    lv2.reset(0.0)
    first = _run(lv2, -25, 4)[0]
    down = _run(lv2, -5, 6, t0=4.0)[0]
    assert abs(20 * np.log10(first / down) - 6) < 0.6
