import pytest
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


# ── 대화를 끝내고 싶다는 말 → 다시 대기 (2026-09-19) ─────────────────────
# 실기: "됐어 좀 쉬고 있어" 가 자유대화로 가서 봇이 계속 말을 걸었다.
# '그만할래' 는 놀이·노래를 끄는 말이기도 해서, 둘 다 없을 때만 대기로 보낸다.

class _Idle:
    def handle(self, text):
        return None


class _NoGame:
    def handle_beat(self, text):
        return None

    def maybe_start_beat(self, text):
        return None


class _GameOn(_NoGame):
    def handle_beat(self, text):
        return {"fallback": "놀이 끝!"}


@pytest.mark.parametrize("text", ["됐어 좀 쉬고 있어.", "대화 그만할래", "이제 얘기 그만하자",
                                  "나 이제 갈게", "잘 가", "그만할래.", "그만하자", "그만."])
def test_끝내고_싶다는_말이면_대기로(text):
    assert route(text, music=_Idle(), games=_NoGame(), sleep_words=None).kind == "sleep"


def test_놀이_중_그만할래는_놀이를_끈다():
    assert route("그만할래", music=_Idle(), games=_GameOn(), sleep_words=None).kind == "game"


def test_놀이_중이라도_대화_그만은_대기로():
    assert route("대화 그만하자", music=_Idle(), games=_GameOn(), sleep_words=None).kind == "sleep"


@pytest.mark.parametrize("text", ["그만 울어", "쉬는 시간이 뭐야?", "가위가 어딨지?", "그만큼 커?"])
def test_끝내자는_말이_아니면_대화(text):
    assert route(text, music=_Idle(), games=_NoGame(), sleep_words=None).kind == "chat"


def test_안전_문장과_정정_문장이_고정_문구에_있다():
    from app.reply_gate import GuardConfig
    assert PHRASES["safe"] and PHRASES["cant"]
    assert PHRASES["safe"] != GuardConfig().cant_line
