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


# ── 마이크 버퍼 넘침 (2026-08-26) ────────────────────────────────────────
# 🔴 왜: 2단계 검증(whisper)이 1.2초쯤 걸리고 그동안 읽기 루프가 멈춘다. 유튜브를 켜면
#    후보가 1.2초마다 떠서 사실상 계속 검증 중이 된다(실기 로그 14:53:50~56, 6초에 6번).
#    그때 버퍼가 넘치면 진짜 호출이 통째로 사라지는데, `block, _ = read()` 로 플래그를
#    **버리고 있어** 넘쳤는지조차 알 수 없었다. 이 프로젝트는 조용한 실패에 반복해 당했다.

class _Overflowed:
    """PortAudio 가 '흘렸다'고 알려주는 status 객체."""
    input_overflow = True


def _src(buffer_s=1.0):
    """콜백 큐를 연 AudioSource. 마이크는 없다 — _on_audio 를 직접 부른다.

    ⚠️ 예전엔 여기서 가짜 **블로킹 스트림**을 심었다(read() 가 넘침 플래그를 함께
       돌려주는 모양). 2026-08-26 에 읽기를 콜백 큐로 바꾸면서 그 모양이 없어졌다 —
       테스트도 실제 접점(_on_audio)을 두드리도록 옮긴다.
    """
    from app.audio_source import AudioSource
    s = AudioSource()
    s._start_buffer(buffer_s=buffer_s)
    return s


def _feed(s, n=1, status=None):
    s._on_audio(np.full((s.frame, 1), float(n), dtype=np.float32), s.frame, None, status)


def test_overflow_flag_is_not_thrown_away():
    """🔴 넘침을 세지 않으면 소리가 버려진 걸 영영 모른다."""
    s = _src()
    _feed(s, status=_Overflowed())
    _feed(s, status=_Overflowed())
    assert s.overflows == 2


def test_no_overflow_leaves_the_counter_alone():
    """멀쩡할 때 카운터가 오르면 경고가 늑대소년이 된다."""
    s = _src()
    for _ in range(5):
        _feed(s)
    assert s.overflows == 0


def test_overflow_still_returns_usable_audio():
    """넘쳤다고 프레임을 버리면 안 된다 — 남은 소리는 그대로 써야 한다."""
    from app.audio_source import FRAME
    s = _src()
    _feed(s, status=_Overflowed())
    f = s._read_frame()
    assert f.shape == (FRAME,) and f.dtype == np.float32


def test_overflow_logging_is_rate_limited(caplog):
    """넘치는 상황에선 프레임마다 넘친다. 그대로 찍으면 초당 12줄이라 [호출] 이 묻힌다."""
    import logging
    s = _src()
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.audio"):
        for _ in range(50):
            s.note_overflow()
    assert s.overflows == 50
    lines = [r for r in caplog.records if "넘침" in r.message]
    assert len(lines) == 1, f"간격 제한이 안 걸렸다({len(lines)}줄)"


def test_overflow_total_is_reported_on_close(caplog):
    """중간 경고는 간격 제한에 걸려 안 나올 수 있다 — 총계는 반드시 남아야 한다."""
    import logging
    s = _src()
    s.overflows = 7
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.audio"):
        s.close()
    assert any("총 7회" in r.message for r in caplog.records)


def test_close_is_quiet_when_nothing_overflowed(caplog):
    """정상 세션에 경고를 남기면 로그를 안 믿게 된다."""
    import logging
    s = _src()
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.audio"):
        s.close()
    assert not [r for r in caplog.records if "넘침" in r.message]


def test_drain_overflow_is_not_a_warning():
    """🔴 drain() 은 **일부러 버리는** 자리다 — 거기서 난 넘침은 손실이 아니다.

    실기 첫 세션(2026-08-26)에서 넘침 17회 중 **15회가 이것**이었다. 턴이 끝날 때마다
    봇이 말하고 생각하는 동안 쌓인 자기 목소리를 버리는데, 그게 정상인데도
    '호출을 놓칠 수 있다'고 찍혔다. 이대로 두면 로그를 안 믿게 되고 진짜 2회가 묻힌다.
    """
    s = _src(buffer_s=0.16)          # 프레임 2개까지만
    s._draining = True
    for n in (1, 2, 3, 4, 5):        # 3개가 밀려난다
        _feed(s, n)
    s._draining = False

    assert s.overflows == 0, "버리는 자리의 넘침을 경고로 셌다"
    assert s.drained_overflows == 3, "그래도 세어는 둬야 진단이 된다"


def test_drain_flag_is_cleared_even_if_read_raises():
    """플래그가 남으면 그 뒤 진짜 넘침이 조용히 묻힌다."""
    s = _src()
    _feed(s)

    def _boom():
        raise RuntimeError("장치 사라짐")

    s._read_frame = _boom
    try:
        s.drain()
    except RuntimeError:
        pass

    assert s._draining is False
    s.note_overflow()
    assert s.overflows == 1, "drain 이 죽은 뒤 진짜 넘침을 못 셌다"


def test_real_overflow_still_warns_after_a_drain(caplog):
    """대기 중 넘침은 여전히 경고여야 한다 — 이게 '호출을 놓쳤다'의 유일한 단서다."""
    import logging
    s = _src()
    _feed(s)
    s.drain()                       # 먼저 '버리는 자리'를 지난다
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.audio"):
        _feed(s, status=_Overflowed())
    assert s.overflows == 1
    assert any("넘침" in r.message for r in caplog.records)


# ── 밀린 오디오를 버리지 말고 돌려준다 (2026-08-27) ──────────────────────
# 🔴 왜: 2단계 검증(whisper 1.2~1.5초) 동안 읽기 루프가 멈추는데, 아이가
#    '하이 티드 **이거 뭐야?**' 의 뒷말을 하는 게 정확히 그 구간이다. 그 오디오는
#    이미 버퍼에 있는데 깨어난 뒤 0.5초만 읽고 나머지를 버렸다 — 실기에서 STT 로
#    1.12초만 넘어갔고 전사가 비어 봇이 침묵했다.

def test_read_buffered_returns_what_piled_up():
    src = DrainableSource(_ramp(10, 0), pending=4)
    got = src.read_buffered(10)
    assert [int(f[0]) for f in got] == [0, 1, 2, 3], "쌓인 걸 순서대로 돌려줘야 함"


def test_read_buffered_does_not_block_when_nothing_is_waiting():
    """🔴 블로킹하면 인사말 경로가 그만큼 느려진다 — 공짜라는 전제가 깨진다."""
    src = DrainableSource(_ramp(10, 0), pending=0)
    assert src.read_buffered(10) == []
    assert src._i == 0, "버퍼가 비었는데 읽었다(다음 입력을 삼킨다)"


def test_read_buffered_respects_the_cap():
    """상한이 없으면 몇 초 전 TV 소리까지 딸려와 whisper 가 그걸 받아쓴다."""
    src = DrainableSource(_ramp(20, 0), pending=12)
    assert len(src.read_buffered(5)) == 5


def test_read_buffered_feeds_the_preroll_like_a_normal_read():
    """걷어온 프레임도 정상 읽기다 — 프리롤에 안 들어가면 뒤 계산이 어긋난다."""
    src = DrainableSource(_ramp(10, 0), pending=3, preroll=0.5)
    src.read_buffered(3)
    assert src.preroll().size == 3 * FRAME


# ── 처리가 멈춰도 소리가 사라지면 안 된다 ───────────────────────────────────
# 🔴 2026-08-26 젯슨 실기: 90초에 넘침 7회. 넘친 만큼 **소리가 통째로 버려진다.**
#    원인은 구조다 — 읽기가 블로킹이라 처리가 멈추면 아무도 마이크를 안 읽는다.
#    호출어 검증(whisper ~0.85s)·인식(~1.2s) 동안 매번 그렇게 되고, 그 사이에 아이가
#    한 말이 사라진다("호출을 놓칠 수 있음"의 실체).
#    ➡️ 콜백이 큐에 쌓게 하면 처리가 멈춰도 오디오는 계속 들어온다. 큐가 넘칠 때만
#       진짜 손실이고, 그때 가장 오래된 것부터 버린다(최신 소리가 더 중요하다).

def _framed(src, n):
    """콜백이 주는 모양(frames, 1) 으로 만든다."""
    return np.full((src.frame, 1), float(n), dtype=np.float32)


def test_the_callback_hands_frames_over_in_order():
    s = AudioSource()
    s._start_buffer(buffer_s=1.0)

    for n in (1, 2, 3):
        s._on_audio(_framed(s, n), s.frame, None, None)

    assert [float(s._read_frame()[0]) for _ in range(3)] == [1.0, 2.0, 3.0]


def test_audio_survives_a_stall_that_used_to_lose_it():
    """이 변경의 전부다 — 1초를 멈춰 있어도 그동안의 소리가 남아 있어야 한다."""
    s = AudioSource()
    s._start_buffer(buffer_s=3.0)

    for n in range(13):          # 13프레임 x 80ms = 1.04초
        s._on_audio(_framed(s, n), s.frame, None, None)

    assert s._available() == 13 * s.frame
    assert s.overflows == 0, "안 넘쳤는데 넘쳤다고 셌다"


def test_a_full_buffer_drops_the_oldest_and_counts_it():
    s = AudioSource()
    s._start_buffer(buffer_s=0.16)      # 프레임 2개까지만

    for n in (1, 2, 3):
        s._on_audio(_framed(s, n), s.frame, None, None)

    assert s.overflows == 1, "버렸는데 안 셌다 — 조용한 실패다"
    assert [float(s._read_frame()[0]) for _ in range(2)] == [2.0, 3.0], \
        "가장 오래된 것부터 버려야 한다(최신 소리가 더 중요하다)"


def test_portaudio_reporting_its_own_overflow_is_counted():
    """큐가 안 찼어도 장치 쪽에서 이미 흘렸을 수 있다."""

    class _Status:
        input_overflow = True

    s = AudioSource()
    s._start_buffer(buffer_s=1.0)

    s._on_audio(_framed(s, 1), s.frame, None, _Status())

    assert s.overflows == 1


def test_a_drop_while_draining_is_still_not_a_warning():
    """봇이 말하는 동안 쌓인 자기 목소리를 버리는 건 손실이 아니다(기존 규칙 유지)."""
    s = AudioSource()
    s._start_buffer(buffer_s=0.16)
    s._draining = True

    for n in (1, 2, 3):
        s._on_audio(_framed(s, n), s.frame, None, None)

    assert s.overflows == 0
    assert s.drained_overflows == 1


# ── 마이크가 0 만 주는 걸 조용히 넘기지 않는다 ──────────────────────────────
# 🔴 2026-08-27 젯슨: `_setup_audio_device` 가 "'ReSpeaker' 의 입력 장치를 못 찾아
#    기본 장치를 사용합니다" 를 찍고 넘어갔고, 그 세션은 **소음바닥 0.00000 / 모든
#    프레임 RMS 0.0** — 귀머거리였다. 직후 재실행하면 멀쩡히 찾는다(장치가 다른
#    프로세스에 잡혀 있던 순간이었다). 08-24 메모의 '젯슨 마이크가 0을 준다'가 이것이다.
#    경고 한 줄로는 부족하다 — 봇이 그대로 살아서 아무것도 못 듣는다.

def test_a_dead_mic_is_reported_loudly(caplog):
    import logging
    from app.audio_source import AudioSource

    class _Silent(AudioSource):
        def _read_frame(self):
            return np.zeros(self.frame, dtype=np.float32)

    s = _Silent()
    s._start_buffer()
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.audio"):
        s._measure_noise()

    assert s.mic_is_dead is True
    assert any("0" in r.message and "마이크" in r.message for r in caplog.records), \
        "귀머거리인데 조용히 지나갔다"


def test_a_live_mic_says_nothing(caplog):
    import logging
    from app.audio_source import AudioSource

    class _Live(AudioSource):
        def _read_frame(self):
            return np.full(self.frame, 0.01, dtype=np.float32)

    s = _Live()
    s._start_buffer()
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.audio"):
        s._measure_noise()

    assert s.mic_is_dead is False
    assert not [r for r in caplog.records if "마이크" in r.message]


# ── 장치 탐색은 한 번 만에 포기하지 않는다 ──────────────────────────────────
# 🔴 2026-08-27 젯슨: 같은 명령을 두 번 돌렸는데 1회차는 "'ReSpeaker' 의 입력 장치를
#    못 찾아" 였고 2회차는 "입력 0 / 출력 0" 으로 멀쩡했다. 장치가 직전 프로세스에
#    잡혀 있던 찰나였다. 한 번 보고 포기하면 **그 세션 전체가 귀머거리**가 된다.

def test_the_device_lookup_retries_before_giving_up(monkeypatch):
    import sys, types
    from app import main as m

    calls = {"n": 0}

    def query_devices(*a):
        calls["n"] += 1
        # 1회차엔 입력 0채널(잡혀 있는 모습), 2회차부터 정상
        ch_in = 0 if calls["n"] == 1 else 6
        return [{"name": "ReSpeaker 4 Mic Array", "max_input_channels": ch_in,
                 "max_output_channels": 2}]

    fake = types.SimpleNamespace(query_devices=query_devices, default=types.SimpleNamespace(device=[9, 9]))
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setitem(m.settings.models, "audio", {"device": "ReSpeaker"})

    m._setup_audio_device(retries=3, wait_s=0.0)

    assert calls["n"] >= 2, "한 번 보고 포기했다"
    assert fake.default.device == (0, 0), "재시도해서 찾은 걸 안 썼다"


def test_the_lookup_stops_as_soon_as_it_finds_both(monkeypatch):
    """멀쩡한 기계에서 기동이 느려지면 안 된다."""
    import sys, types
    from app import main as m

    calls = {"n": 0}

    def query_devices(*a):
        calls["n"] += 1
        return [{"name": "ReSpeaker", "max_input_channels": 6, "max_output_channels": 2}]

    fake = types.SimpleNamespace(query_devices=query_devices, default=types.SimpleNamespace(device=[9, 9]))
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setitem(m.settings.models, "audio", {"device": "ReSpeaker"})

    m._setup_audio_device(retries=5, wait_s=1.0)

    assert calls["n"] == 1
