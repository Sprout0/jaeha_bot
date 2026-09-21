import pytest

from app.song_names import correct, transcribe_prompt


@pytest.mark.parametrize("heard, want", [
    ("비니닝 노래", "티니핑 노래"),      # 09-19 실기
    ("티니 핑", "티니핑"),
    ("아기사어", "아기상어"),
    ("뽀르르 노래", "뽀로로 노래"),
    ("곰새마리", "곰세마리"),
    ("캐치 티니빙", "캐치티니핑"),
])
def test_잘못_적힌_이름을_바로잡는다(heard, want):
    assert correct(heard) == want


@pytest.mark.parametrize("q", [
    "스파이더맨 OST 로저", "한노도의 입춘", "파요", "핑크 노래", "토끼 노래", "코끼리",
    "자동차 노래", "", "노래",
])
def test_확실하지_않으면_그대로(q):
    assert correct(q) == q


def test_받아쓰기_힌트에_호출어와_이름이_들어간다():
    p = transcribe_prompt()
    assert p.startswith("하이 티드") and "티니핑" in p and "아기상어" in p


def test_받아쓰기_힌트에_명령_낱말도_있다():
    p = transcribe_prompt()
    for w in ("잘 자", "바이바이", "동물 소리 놀이", "따라 말하기 놀이", "노래 틀어줘"):
        assert w in p
