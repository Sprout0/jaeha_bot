import pytest

from app import reply_gate
from app.reply_gate import PROPOSE, GuardConfig, ReplyGate
from app.safety import question_risk


@pytest.mark.parametrize("text", ["칼 어딨어?", "이거 먹어도 돼?", "이거 무슨 맛이야?",
                                  "창문 열어줄까?", "불 켜 볼까"])
def test_위험_신호가_있는_아이_말(text):
    assert question_risk(text)


@pytest.mark.parametrize("text", ["공룡 좋아해", "노래 불러줘", "이거 뭐야?", "사과 먹었어", ""])
def test_위험_신호가_없는_아이_말(text):
    assert not question_risk(text)


def _gate(**kw):
    return ReplyGate(GuardConfig(**kw))


def test_위험_신호면_붙잡고_아니면_흘려보내고_그대로_읽기는_안_본다():
    g = _gate()
    assert g.begin("칼 어딨어?", verbatim=False, now=0.0).mode == "hold"
    assert g.begin("공룡 좋아해", verbatim=False, now=0.0).mode == "stream"
    assert g.begin("칼 어딨어?", verbatim=True, now=0.0).mode == "off"
    assert _gate(hold_on_risk=False).begin("칼 어딨어?", verbatim=False, now=0.0).mode == "stream"


def test_붙잡은_턴은_끝에서_문제없으면_내보낸다():
    t = _gate().begin("칼 어딨어?", verbatim=False, now=0.0)
    assert t.poll("칼은 위험해!", 1.0) is None
    v = t.finish("칼은 위험해! 엄마한테 말하자.")
    assert v.action == "release" and v.flags == []


def test_붙잡은_턴은_끝에서_걸리면_막는다():
    t = _gate().begin("칼 어딨어?", verbatim=False, now=0.0)
    v = t.finish("칼은 부엌에 있어! 같이 찾아볼까?")
    assert v.action == "block" and PROPOSE in v.flags


def test_붙잡은_턴은_어른유도없음만으로도_막는다():
    v = _gate().begin("칼 어딨어?", verbatim=False, now=0.0).finish("칼은 부엌에 있어.")
    assert v.action == "block" and v.flags == ["어른유도없음"]


def test_한도가_지나면_거기까지_보고_흘려보내기로_바뀐다():
    t = _gate(hold_cap_s=3.0).begin("칼 어딨어?", verbatim=False, now=0.0)
    assert t.poll("칼은", 2.9) is None
    assert t.poll("칼은", 3.0) == "release" and t.mode == "stream" and t.held


def test_한도가_지났는데_위험행동제안이면_막는다():
    t = _gate(hold_cap_s=3.0).begin("칼 어딨어?", verbatim=False, now=0.0)
    assert t.poll("칼 같이 찾아볼까", 3.1) == "block"


def test_흘려보내는_턴은_도중에_위험행동제안이면_막는다():
    t = _gate().begin("뭐 하고 놀까?", verbatim=False, now=0.0)
    assert t.poll("우리 칼 같이 찾아볼까", 0.5) == "block"
    assert t.poll("우리 공룡 놀이 할까", 0.5) is None


def test_흘려보내는_턴은_도중에_어른유도없음으로는_안_막는다():
    t = _gate().begin("뭐 하고 놀까?", verbatim=False, now=0.0)
    assert t.poll("칼은", 0.5) is None                 # 뒤에 '엄마한테' 가 올 수 있다


def test_흘려보내는_턴의_끝은_기록만_한다():
    v = _gate().begin("뭐 하고 놀까?", verbatim=False, now=0.0).finish("칼은 부엌에 있어.")
    assert v.action == "pass" and v.flags == ["어른유도없음"]


def test_못_하는_것은_정정_목록으로():
    v = _gate().begin("뭐 하고 놀까?", verbatim=False, now=0.0).finish("좋아! 색칠 놀이 하자!")
    assert v.action == "pass" and v.fabricated


def test_막힌_턴에는_정정을_안_붙인다():
    v = _gate().begin("칼 어딨어?", verbatim=False, now=0.0).finish("칼 같이 찾아볼까? 색칠 놀이도 하자")
    assert v.action == "block" and v.fabricated == []


def test_빈_답은_막지_않는다():
    v = _gate().begin("칼 어딨어?", verbatim=False, now=0.0).finish("")
    assert v.action == "release" and v.flags == []


def test_판정기가_고장나면_붙잡은_턴은_막고_흘려보내는_턴은_둔다(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(reply_gate.safety, "check_reply", boom)
    assert _gate().begin("칼 어딨어?", verbatim=False, now=0.0).finish("아무 말").action == "block"
    t = _gate().begin("안녕", verbatim=False, now=0.0)
    assert t.poll("아무 말", 0.1) is None and t.finish("아무 말").action == "pass"


def test_그대로_읽기_턴은_아무것도_안_본다():
    t = _gate().begin("칼 노래 틀어줘", verbatim=True, now=0.0)
    assert t.poll("칼 같이 찾아볼까", 9.0) is None
    assert t.finish("칼 같이 찾아볼까").action == "pass"


def test_설정은_모르는_키를_버린다():
    c = GuardConfig.from_dict({"hold_cap_s": 2, "없는키": 1})
    assert c.hold_cap_s == 2 and c.reconnect_tries == 3


def test_병아리_소리_삐약은_약이_아니다():
    # 09-22 실서버: 놀이 턴 "같이 해보자, 삐약" 이 '약'+'해보자' 로 위험행동제안에 막혔다.
    from app.safety import check_reply
    assert check_reply("같이 해보자, 삐약삐약!", child_text="몰라") == []
    assert question_risk("삐약삐약") is False
    assert "위험행동제안" in check_reply("약 같이 먹어보자", child_text="이거 뭐야")
