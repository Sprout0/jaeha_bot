"""노래 틀기 판단 — 무엇을 '틀어줘'로 보고 무엇을 안 보는가. (2026-09-11)

오탐이 미탐보다 나쁘다: 엉뚱하게 나간 노래는 되돌릴 수 없고, 못 알아들으면 아이가
한 번 더 말하면 된다(AudioLibrary.find 와 같은 원칙). 그래서 '아니어야 하는 것'을
'맞아야 하는 것'만큼 적는다.
"""
import pytest

from app.config import settings
from app.music import (SEARCH_FAILED, Command, MusicController, NEW_SONG_RULE,
                       adjust_prompt, parse)
from app.youtube import Video


# ── 판별 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,query", [
    ("티니핑 노래 틀어줘", "티니핑 노래"),
    ("티니핑송 틀어줘", "티니핑송"),
    ("아기상어 좀 틀어줘", "아기상어"),
    ("상어가족 들려줘", "상어가족"),
    ("뽀로로 노래 들을래", "뽀로로 노래"),
    ("나 티니핑 노래 듣고 싶어", "티니핑 노래"),      # 동사 사이 띄어쓰기
    ("하이 티드 티니핑 노래 틀어줘", "티니핑 노래"),   # 호출어가 앞에 붙어 들어온 경우
    ("다시 티니핑 노래 틀어줘", "티니핑 노래"),
])
def test_play_requests_extract_the_song(text, query):
    assert parse(text) == Command("play", query=query)


@pytest.mark.parametrize("text", ["노래 틀어줘", "노래 켜줘", "노래 불러줘", "음악 틀어줘"])
def test_play_without_a_title_uses_the_default(text):
    assert parse(text) == Command("play", query="")


@pytest.mark.parametrize("text", [
    "불 켜줘",               # 켜다 — 노래가 없으면 요청이 아니다
    "엄마 불러줘",           # 부르다 — 사람을 부르는 말
    "물 틀어줘",             # 틀 수 없는 물건
    "티비 틀어줘",
    "강아지 소리 들려줘",    # 효과음·놀이 쪽
    "이야기 들려줘",
    "오늘 뭐 했어",
    "노래 좋아해",           # 좋아한다고 틀어 달란 건 아니다
])
def test_everyday_speech_is_not_a_play_request(text):
    assert parse(text) is None


def test_stop_words_count_only_while_music_is_on():
    """평소의 "그만"은 놀이를 끝내는 말이다 — 노래 기능이 가로채면 놀이가 안 끝난다."""
    assert parse("그만", active=True) == Command("stop")
    assert parse("노래 꺼", active=True) == Command("stop")
    assert parse("그만") is None


def test_bare_continue_counts_only_while_music_is_on():
    """평소의 "계속"이 노래를 틀면 안 된다."""
    assert parse("계속", active=True) == Command("resume")
    assert parse("계속") is None
    assert parse("계속 놀자") is None


def test_explicit_replay_works_even_after_stop():
    assert parse("다시 틀어줘") == Command("resume")


def test_different_song():
    assert parse("다른 노래 틀어줘") == Command("play", different=True)
    assert parse("다른 거 틀어줘") == Command("play", different=True)


# ── 컨트롤러 ─────────────────────────────────────────────────────────────────
class FakeLocal:
    def __init__(self):
        self.played, self.stops, self.is_playing = [], 0, False

    def play(self, asset, block=False):
        self.played.append(asset)
        self.is_playing = True
        return True

    def stop(self):
        self.stops += 1
        self.is_playing = False


class FakeYouTube:
    def __init__(self, ok=True):
        self.ok, self.calls, self.is_playing = ok, [], False

    def play(self, vid):
        self.calls.append(("play", vid))
        self.is_playing = self.ok
        return self.ok

    def pause(self):
        self.calls.append(("pause",))
        self.is_playing = False

    def resume(self):
        self.calls.append(("resume",))
        self.is_playing = True

    def stop(self):
        self.calls.append(("stop",))
        self.is_playing = False

    def close(self):
        self.calls.append(("close",))


class FakeSearch:
    def __init__(self, videos=None, error=None):
        self.videos = videos if videos is not None else [
            Video("A", "a", 140, True), Video("B", "b", 150, True)]
        self.error, self.queries = error, []

    def find(self, query, exclude=()):
        self.queries.append((query, tuple(exclude)))
        if self.error:
            raise self.error
        return next((v for v in self.videos if v.id not in set(exclude)), None)


class FakeAsset:
    def __init__(self, title, exists=True):
        self.title, self.exists = title, exists


class FakeLib:
    def __init__(self, asset=None):
        self.asset = asset

    def find(self, text, kind=None):
        return self.asset


def make(asset=None, **kw):
    local, yt, search = FakeLocal(), kw.pop("yt", FakeYouTube()), kw.pop("search", FakeSearch())
    c = MusicController(library=FakeLib(asset), local=local, search=search, youtube=yt)
    return c, local, yt, search


def test_youtube_play_says_first_then_plays_then_standby():
    c, local, yt, search = make()
    r = c.handle("티니핑 노래 틀어줘")
    assert r.text == "티니핑 노래 틀어 줄게!"
    assert r.standby is True
    assert yt.calls == [], "말하기 전에 틀면 약속 위에 노래가 겹친다"
    assert r.action() is True
    assert yt.calls == [("play", "A")]
    assert search.queries == [("티니핑 노래", ())]


def test_no_title_searches_the_default_and_says_so():
    c, _, _, search = make()
    r = c.handle("노래 틀어줘")
    assert search.queries[0][0] == "동요"
    assert r.text == "신나는 동요 틀어 줄게!"


def test_local_song_wins_over_youtube():
    """공유마당 곡은 로컬로 — 시연·논문 트랙은 유튜브 없이 돌아야 한다."""
    c, local, yt, search = make(asset=FakeAsset("세상은 놀이터"))
    r = c.handle("세상은 놀이터 틀어줘")
    assert r.text == "세상은 놀이터 틀어 줄게!"
    assert r.action() is True
    assert len(local.played) == 1
    assert search.queries == [] and ("play", "A") not in yt.calls


def test_registered_song_without_file_goes_to_youtube():
    c, local, yt, search = make(asset=FakeAsset("곰 세 마리", exists=False))
    r = c.handle("곰 세 마리 틀어줘")
    r.action()
    assert local.played == [] and yt.calls[-1] == ("play", "A")


def test_search_failure_is_spoken_not_raised():
    c, _, _, _ = make(search=FakeSearch(error=RuntimeError("quota")))
    r = c.handle("티니핑 노래 틀어줘")
    assert r.text == SEARCH_FAILED and r.action is None and not r.standby


def test_nothing_found():
    c, _, _, _ = make(search=FakeSearch(videos=[]))
    r = c.handle("없는노래 틀어줘")
    assert "못 찾았어" in r.text and r.action is None


def test_play_failure_is_reported_to_main():
    c, _, _, _ = make(yt=FakeYouTube(ok=False))
    assert c.handle("티니핑 노래 틀어줘").action() is False


def test_stop_only_when_music_is_on():
    c, _, yt, _ = make()
    assert c.handle("그만") is None, "노래가 없는데 가로채면 놀이의 '그만'이 죽는다"
    c.handle("티니핑 노래 틀어줘").action()
    r = c.handle("그만")
    assert r.text == "노래 껐어!" and ("stop",) in yt.calls and not r.standby


def test_wake_pauses_and_resumes():
    c, _, yt, _ = make()
    assert c.pause_for_wake() is False, "노래가 없으면 멈출 것도 없다"
    c.handle("티니핑 노래 틀어줘").action()
    assert c.pause_for_wake() is True and yt.calls[-1] == ("pause",)
    assert c.paused_by_wake
    assert c.resume_after_wake() is True and yt.calls[-1] == ("resume",)
    assert c.resume_after_wake() is False, "두 번 이어 틀지 않는다"


def test_stop_works_while_paused_by_wake():
    """불러서 멈춘 상태는 재생 중이 아니지만 "그만"은 먹어야 한다 — 그러려고 부른 것이다."""
    c, _, yt, _ = make()
    c.handle("티니핑 노래 틀어줘").action()
    c.pause_for_wake()
    r = c.handle("그만")
    assert r is not None and r.text == "노래 껐어!"
    assert c.resume_after_wake() is False, "껐는데 다시 틀면 안 된다"


def test_different_song_skips_what_just_played():
    c, _, yt, search = make()
    c.handle("티니핑 노래 틀어줘").action()
    r = c.handle("다른 노래 틀어줘")
    assert r.text == "다른 노래 틀어 줄게!"
    r.action()
    assert search.queries[-1] == ("티니핑 노래", ("A",))
    assert yt.calls[-1] == ("play", "B")


def test_no_youtube_and_no_local_says_cannot():
    c = MusicController(library=FakeLib(None), local=FakeLocal(), search=None, youtube=None)
    assert c.handle("티니핑 노래 틀어줘").text == SEARCH_FAILED


# ── 프롬프트 ─────────────────────────────────────────────────────────────────
def test_prompt_no_longer_says_it_cannot_play():
    old = settings.prompts["system"]
    new = adjust_prompt(old)
    assert "못 부르고 못 튼다" in old, "기준 문구가 바뀌었다 — 이 테스트와 adjust_prompt 를 같이 볼 것"
    assert "못 부르고 못 튼다" not in new
    assert "틀어줘\"라고 하면 틀어 줄 수 있다" in new
    # 다른 규칙은 그대로 — 한 줄만 바꾼다
    assert new.count("\n- ") == old.count("\n- ")
    assert "동물 소리는 여덟뿐이다" in new


def test_prompt_without_the_rule_gets_it_appended():
    assert adjust_prompt("규칙:\n- 반말로 말한다.\n").endswith(NEW_SONG_RULE)
