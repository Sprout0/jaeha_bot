from app.education_modes import GameManager
from app.music import MusicReply
from app.realtime_turn import PHRASES, missing_required, route


class FakeMusic:
    def __init__(self, reply=None):
        self.reply, self.seen = reply, []

    def handle(self, text):
        self.seen.append(text)
        return self.reply


def _route(text, music=None, games=None):
    return route(text, music=music, games=games or GameManager(render=None), sleep_words=None)


def test_빈_글자는_아무것도_안_한다():
    assert _route("  ").kind == "empty"


def test_잠들기가_노래보다_먼저다():
    m = FakeMusic(MusicReply("노래 껐어!"))
    assert _route("바이바이", music=m).kind == "sleep"
    assert m.seen == []


def test_노래_명령은_안내_문장을_그대로_말한다():
    r = _route("상어가족 틀어줘", music=FakeMusic(MusicReply("상어가족 틀어 줄게!", standby=True)))
    assert r.kind == "music" and r.say == "상어가족 틀어 줄게!" and r.music.standby


def test_노래가_꺼져_있으면_노래로_안_간다():
    assert _route("상어가족 틀어줘", music=None).kind == "chat"


def test_놀이_시작은_지시를_붙인다():
    r = _route("동물 소리 놀이 하자")
    assert r.kind == "game_start" and "상황:" in r.instructions and r.say


def test_놀이_중에는_상태기계가_받는다():
    g = GameManager(render=None)
    route("동물 소리 놀이 하자", music=None, games=g, sleep_words=None)
    assert route("야옹", music=None, games=g, sleep_words=None).kind == "game"


def test_나머지는_자유대화():
    assert _route("오늘 뭐 했어").kind == "chat"


def test_빠진_필수_낱말():
    assert missing_required("좋아! 고양이는 야옹?", ["고양이", "강아지"]) == ["강아지"]
    assert missing_required("아무거나", []) == []


def test_고정_문구는_로컬_봇과_같다():
    import app.main as m
    from app.music import MUSIC_FAILED
    assert PHRASES["ready"] == m.READY_ASLEEP
    assert PHRASES["wake"] == m.WAKE_GREETING
    assert PHRASES["sleep"] == m.SLEEP_MSG
    assert PHRASES["recovery"] == m.SAFE_RECOVERY
    assert PHRASES["music_failed"] == MUSIC_FAILED
    assert PHRASES["lost"]


def test_기존_놀이_함수_동작은_그대로다():
    g = GameManager(render=None)
    assert isinstance(g.maybe_start("동물 소리 놀이 하자"), str)
    assert isinstance(g.handle("야옹"), str)
