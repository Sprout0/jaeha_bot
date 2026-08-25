"""필러(맞장구) 즉시 재생 — 아이가 처음 소리를 듣는 시각을 4.17s -> 1.70s 로.

🔴 이건 실지연을 줄이지 않는다. **진짜 답이 나오는 시각은 1ms 도 안 바뀐다.**
   바뀌는 건 침묵의 길이뿐이다. 그래서 여기 테스트는 '빨라졌나'가 아니라
   '엉뚱한 때 소리를 내지 않나'를 지킨다 — 잘못 나가면 지금보다 나쁘다.

설계: docs/superpowers/specs/2026-08-25-filler-design.md
"""
import numpy as np
import pytest

from app.filler import FillerBank

PHRASES = ["음~", "그래?", "아~", "오~"]


class _FakeTTS:
    """합성 호출을 세는 가짜. 목소리 설정은 캐시 키의 재료다."""

    def __init__(self, voice="F2@777", speed=1.05, total_steps=24):
        self.voice, self.speed, self.total_steps = voice, speed, total_steps
        self.sample_rate = 16000
        self.calls: list[str] = []

    def render(self, text):
        self.calls.append(text)
        # 0.3초짜리 '소리'. 무음이 아니어야 패딩 테스트가 의미를 갖는다.
        return np.full(int(0.3 * self.sample_rate), 0.5, dtype=np.float32)


class _FakeSink:
    def __init__(self):
        self.plays: list[tuple[int, int, bool]] = []   # (샘플수, 레이트, 블로킹)

    def play(self, samples, rate, block):
        self.plays.append((int(np.asarray(samples).size), int(rate), bool(block)))


class _BrokenSink:
    def play(self, samples, rate, block):
        raise RuntimeError("장치 없음")


def _bank(tmp_path, phrases=None, sink=None, **kw):
    return FillerBank(phrases or PHRASES, cache_dir=tmp_path,
                      sink=sink if sink is not None else _FakeSink(), **kw)


# ── 캐시 ─────────────────────────────────────────────────────────────────────

def test_every_phrase_is_synthesized_once(tmp_path):
    tts, bank = _FakeTTS(), _bank(tmp_path)

    bank.ensure(tts)

    assert tts.calls == PHRASES


def test_a_second_startup_reads_the_cache_instead_of_synthesizing(tmp_path):
    """기동마다 다시 합성하면 6개 × 0.45초를 매번 문다."""
    tts = _FakeTTS()
    _bank(tmp_path).ensure(tts)
    tts.calls.clear()

    _bank(tmp_path).ensure(tts)

    assert tts.calls == [], f"캐시가 있는데 다시 합성했다: {tts.calls}"


def test_changing_the_voice_rebuilds_the_cache(tmp_path):
    """🔴 F2 에서 다른 목소리로 갈아탔을 때 필러만 옛 목소리로 겉돌면 안 된다."""
    _bank(tmp_path).ensure(_FakeTTS(voice="F2@777"))
    other = _FakeTTS(voice="M1@100")

    _bank(tmp_path).ensure(other)

    assert other.calls == PHRASES, "목소리가 바뀌었는데 옛 캐시를 그대로 썼다"


def test_changing_the_speed_rebuilds_the_cache(tmp_path):
    _bank(tmp_path).ensure(_FakeTTS(speed=1.05))
    other = _FakeTTS(speed=1.20)

    _bank(tmp_path).ensure(other)

    assert other.calls == PHRASES


def test_changing_the_phrase_list_rebuilds_the_cache(tmp_path):
    tts = _FakeTTS()
    _bank(tmp_path).ensure(tts)
    tts.calls.clear()

    _bank(tmp_path, phrases=["음~", "어~"]).ensure(tts)

    assert tts.calls == ["음~", "어~"]


# ── 앞 무음 ──────────────────────────────────────────────────────────────────
# 🔴 SoundDeviceSink 는 TTSModule 과 달리 패딩을 안 붙인다. sounddevice 스트림
#    시작이 앞 샘플을 흘리므로, 구워 넣지 않으면 **첫 음절이 잘린다.**

def test_the_cached_audio_starts_with_silence_so_the_first_syllable_survives(tmp_path):
    tts = _FakeTTS()
    bank = _bank(tmp_path, pad_s=0.15)
    bank.ensure(tts)

    audio, rate = bank.next()

    pad = int(0.15 * rate)
    assert np.all(audio[:pad] == 0.0), "앞 무음이 없다 — 첫 음절이 잘린다"
    assert np.any(audio[pad:] != 0.0), "무음만 있고 소리가 없다"


# ── 고르기 ───────────────────────────────────────────────────────────────────

def test_the_same_filler_never_comes_out_twice_in_a_row(tmp_path):
    bank = _bank(tmp_path)
    bank.ensure(_FakeTTS())

    picks = [bank._pick_index() for _ in range(30)]

    assert all(a != b for a, b in zip(picks, picks[1:])), f"연속 반복: {picks}"


def test_a_single_phrase_still_works(tmp_path):
    """문구가 하나뿐이면 반복을 피할 방법이 없다 — 그래도 죽으면 안 된다."""
    bank = _bank(tmp_path, phrases=["음~"])
    bank.ensure(_FakeTTS())

    assert bank.next() is not None


def test_next_returns_nothing_before_ensure(tmp_path):
    assert _bank(tmp_path).next() is None


# ── 재생 ─────────────────────────────────────────────────────────────────────

def test_playing_is_non_blocking(tmp_path):
    """블로킹하면 필러가 끝날 때까지 LLM 을 못 친다 — 순 지연이 늘어난다."""
    sink = _FakeSink()
    bank = _bank(tmp_path, sink=sink)
    bank.ensure(_FakeTTS())

    bank.play()

    assert len(sink.plays) == 1
    assert sink.plays[0][2] is False, "블로킹으로 재생했다"


def test_disabled_bank_stays_silent(tmp_path):
    sink = _FakeSink()
    bank = _bank(tmp_path, sink=sink, enabled=False)
    bank.ensure(_FakeTTS())

    bank.play()

    assert sink.plays == []


def test_disabled_bank_does_not_even_synthesize(tmp_path):
    """꺼 놨는데 기동에서 2.7초를 쓰면 안 된다."""
    tts = _FakeTTS()

    _bank(tmp_path, enabled=False).ensure(tts)

    assert tts.calls == []


# ── 실패해도 대화는 산다 ─────────────────────────────────────────────────────
# 필러는 기능이 아니라 최적화다. 어떤 실패도 위로 안 던진다.

def test_a_broken_sink_does_not_break_the_turn(tmp_path):
    bank = _bank(tmp_path, sink=_BrokenSink())
    bank.ensure(_FakeTTS())

    assert bank.play() is False, "실패를 알리되 예외는 안 던진다"


def test_a_broken_tts_does_not_break_startup(tmp_path):
    class _BrokenTTS(_FakeTTS):
        def render(self, text):
            raise RuntimeError("TRT 프로파일 초과")

    bank = _bank(tmp_path)
    bank.ensure(_BrokenTTS())      # 예외가 새면 봇이 아예 안 뜬다

    assert bank.next() is None


def test_one_broken_phrase_does_not_lose_the_others(tmp_path):
    class _PickyTTS(_FakeTTS):
        def render(self, text):
            if text == "그래?":
                raise RuntimeError("이 문장만 실패")
            return super().render(text)

    bank = _bank(tmp_path)
    bank.ensure(_PickyTTS())

    assert bank.size == len(PHRASES) - 1


# ── 미루기 ───────────────────────────────────────────────────────────────────
# 기본은 즉시(0.0)다. 젯슨에서 들어보고 코드 수정 없이 밀 수 있어야 한다.

def test_delay_defaults_to_immediate(tmp_path):
    assert _bank(tmp_path).delay_s == 0.0


def test_a_delayed_filler_does_not_block_the_caller(tmp_path):
    """미루더라도 호출자를 붙잡으면 안 된다 — 그 시간만큼 답이 늦어진다."""
    import time

    sink = _FakeSink()
    bank = _bank(tmp_path, sink=sink, delay_s=0.3)
    bank.ensure(_FakeTTS())

    t0 = time.perf_counter()
    bank.play()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.1, f"호출자를 {elapsed:.2f}초 붙잡았다"
    assert sink.plays == [], "미루기로 했는데 즉시 냈다"
    time.sleep(0.5)
    assert len(sink.plays) == 1, "미뤄 둔 재생이 끝내 안 나왔다"


# ── 어느 턴에 나가나 ─────────────────────────────────────────────────────────
# 놀이 턴은 상태머신이 즉답(총 2.4초)이라 필러가 오히려 답을 늦춘다.
# 자유대화(LLM) 턴에만 낸다.

class _Games:
    def __init__(self, handle=None, start=None):
        self._handle, self._start = handle, start

    def handle(self, text):
        return self._handle

    def maybe_start(self, text):
        return self._start


class _Agent:
    def __init__(self, log=None):
        self.log = log

    def respond(self, text):
        if self.log is not None:
            self.log.append("llm")
        return {"text": "우주는 아주 커!"}


class _RecordingFiller:
    def __init__(self, log=None):
        self.log, self.plays = log, 0

    def play(self):
        self.plays += 1
        if self.log is not None:
            self.log.append("filler")
        return True


def test_a_game_turn_stays_silent(tmp_path):
    from app.main import _respond

    filler = _RecordingFiller()
    reply, kind = _respond(_Agent(), _Games(handle="잘했어!"), "빨간색", filler)

    assert (reply, kind) == ("잘했어!", "game")
    assert filler.plays == 0, "놀이는 즉답인데 필러가 답을 늦췄다"


def test_a_game_start_turn_stays_silent(tmp_path):
    from app.main import _respond

    filler = _RecordingFiller()
    _respond(_Agent(), _Games(start="동물 소리 놀이 하자!"), "놀자", filler)

    assert filler.plays == 0


def test_a_free_talk_turn_gets_a_filler(tmp_path):
    from app.main import _respond

    filler = _RecordingFiller()
    reply, kind = _respond(_Agent(), _Games(), "우주가 뭐야", filler)

    assert kind == "llm"
    assert filler.plays == 1


def test_the_filler_goes_out_before_the_llm_is_asked(tmp_path):
    """🔴 이 순서가 이 기능의 전부다. LLM 뒤에 내면 침묵이 그대로 남는다."""
    from app.main import _respond

    order: list[str] = []
    _respond(_Agent(order), _Games(), "우주가 뭐야", _RecordingFiller(order))

    assert order == ["filler", "llm"], f"순서가 뒤집혔다: {order}"


def test_no_filler_configured_still_answers(tmp_path):
    from app.main import _respond

    reply, kind = _respond(_Agent(), _Games(), "우주가 뭐야", None)

    assert (reply, kind) == ("우주는 아주 커!", "llm")


# ── TTS 쪽 이음매 ────────────────────────────────────────────────────────────

def test_tts_render_gives_audio_without_touching_the_speaker():
    """필러 합성은 소리를 내면 안 된다 — 기동 때 6번 떠들면 곤란하다."""
    from app.tts_module import TTSModule

    class _T(TTSModule):
        def __init__(self):
            self.sample_rate = 16000
            self.seen: list[str] = []

        def _infer(self, text):
            self.seen.append(text)
            return np.concatenate([np.zeros(1600, dtype=np.float32),
                                   np.full(1600, 0.5, dtype=np.float32)])

    t = _T()
    out = t.render("음~")

    assert t.seen == ["음~"]
    assert out.size < 3200, "앞 무음이 안 잘렸다(_trim 을 안 거쳤다)"
    assert np.any(out != 0.0)


# ── 기동 ─────────────────────────────────────────────────────────────────────

def test_preload_prepares_the_filler_cache():
    """캐시를 첫 턴에 만들면 그 턴만 2.7초 느려진다 — 기동에서 미리 만든다."""
    import types

    from app.main import preload

    class _Stage:
        def __init__(self): self.loaded = False
        def load(self): self.loaded = True

    class _Agent:
        def _ensure_loaded(self): return types.SimpleNamespace()

    class _Filler:
        def __init__(self): self.ensured_with = None
        def ensure(self, tts): self.ensured_with = tts

    stt, tts, filler = _Stage(), _Stage(), _Filler()
    preload(stt, tts, _Agent(), filler)

    assert filler.ensured_with is tts


def test_preload_without_a_filler_still_works():
    """필러는 선택 사항이다 — 없다고 기동이 죽으면 안 된다."""
    import types

    from app.main import preload

    class _Stage:
        def load(self): pass

    preload(_Stage(), _Stage(), types.SimpleNamespace())


def test_a_broken_filler_does_not_stop_startup():
    import types

    from app.main import preload

    class _Stage:
        def load(self): pass

    class _Boom:
        def ensure(self, tts): raise RuntimeError("디스크 꽉 참")

    preload(_Stage(), _Stage(), types.SimpleNamespace(), _Boom())


# ── 설정이 실제로 닿는가 ─────────────────────────────────────────────────────
# 🔴 설정을 읽어 놓고 안 넘기는 사고가 이 저장소에서 이미 한 번 있었다
#    (ca260c7 이 warm() 만 고치고 호출부를 놓쳐, 2.3GB 회수가 한 번도 일어난
#    적이 없었다). 값이 끝까지 닿는지를 따로 지킨다.

def test_config_values_reach_the_bank(monkeypatch):
    from app import main as m

    monkeypatch.setitem(m.settings.models, "filler", {
        "enabled": True,
        "phrases": ["어?", "그랬구나"],
        "delay_s": 0.4,
        "cache_dir": "~/.cache/test_filler",
    })

    bank = m._build_filler()

    assert bank.phrases == ["어?", "그랬구나"]
    assert bank.delay_s == 0.4
    assert bank.enabled is True


def test_the_off_switch_reaches_the_bank(monkeypatch):
    from app import main as m

    monkeypatch.setitem(m.settings.models, "filler", {"enabled": False,
                                                      "phrases": ["음~"]})

    assert m._build_filler().enabled is False


def test_a_missing_filler_section_does_not_crash(monkeypatch):
    from app import main as m

    monkeypatch.delitem(m.settings.models, "filler", raising=False)

    bank = m._build_filler()

    assert bank is not None and bank.phrases == []
