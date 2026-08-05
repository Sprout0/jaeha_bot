"""빈 인식일 때의 분기(_empty_text_action). 모델·마이크 불필요.

'인식 결과가 비었다'에는 성격이 다른 두 경우가 섞여 있다:
  1) 아이가 아무 말도 안 했다      -> 조용히 기다리다 시간이 되면 잠든다
  2) 말은 했는데 환각이라 버렸다   -> 침묵하면 안 된다. 되물어야 한다
빈 문자열만으로는 둘을 구분할 수 없어(둘 다 "") 조용히 잘못 동작한다.
2세 상대로 '말했는데 아무 반응 없음'은 최악의 응답이라 테스트로 못박는다.
"""
from app.main import _empty_text_action


def test_rejected_speech_triggers_reask():
    # 환각으로 버린 경우 — 아이는 분명 말을 했다.
    assert _empty_text_action(rejected=True, wake_enabled=True,
                              idle_s=0.0, sleep_timeout=30.0) == "reask"


def test_rejected_beats_sleep_timeout():
    """되묻기가 잠들기보다 우선.

    아이가 방금 말을 했는데 '무응답 30초'로 취급해 잠들어 버리면,
    말할수록 잠드는 이상한 로봇이 된다.
    """
    assert _empty_text_action(rejected=True, wake_enabled=True,
                              idle_s=999.0, sleep_timeout=30.0) == "reask"


def test_silence_past_timeout_sleeps():
    assert _empty_text_action(rejected=False, wake_enabled=True,
                              idle_s=31.0, sleep_timeout=30.0) == "sleep"


def test_silence_within_timeout_waits():
    assert _empty_text_action(rejected=False, wake_enabled=True,
                              idle_s=5.0, sleep_timeout=30.0) == "wait"


def test_never_sleeps_when_wake_disabled():
    # wake.enabled=false 는 '항상 대화' 모드라 잠들 개념이 없다(옛 동작).
    assert _empty_text_action(rejected=False, wake_enabled=False,
                              idle_s=999.0, sleep_timeout=30.0) == "wait"


def test_reask_works_even_without_wake_mode():
    assert _empty_text_action(rejected=True, wake_enabled=False,
                              idle_s=0.0, sleep_timeout=30.0) == "reask"
