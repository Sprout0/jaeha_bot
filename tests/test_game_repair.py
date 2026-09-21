import numpy as np

from app.game_repair import check_done, check_stream, find_name, first_sentence_end, quiet_cut

BEAT = {"fix_react": "강아지는 멍멍 하고 울어!", "fix_next": "그럼 고양이는 어떻게 울어?",
        "next_need": ["고양이", "울어"], "allow": ["강아지", "고양이"],
        "names": ["강아지", "고양이", "소", "돼지", "병아리"]}


def test_이름은_낱말_경계로_찾는다():
    assert find_name("동물 소리 놀이 하자", "소") == -1
    assert find_name("그럼 소는 어떻게 울어?", "소") == 3
    assert find_name("소랑 놀자", "소") == 0


def test_첫_문장_끝():
    assert first_sentence_end("멍멍, 강아지는 멍멍 하고 울어! 그럼 고양이는?") == 18
    assert first_sentence_end("멍멍 하고 울어") == -1


def test_허용된_이름만이면_도중엔_아무것도_안_한다():
    assert check_stream("멍멍, 강아지는 멍멍 하고 울어! 그럼 고양이", BEAT) is None


def test_첫_문장에_틀린_이름이면_도중에_바로잡는다():
    r = check_stream("돼지는 꿀꿀", BEAT)
    assert r.reason == "wrong_name" and r.in_first and r.cut_char == 0
    assert r.lines == [BEAT["fix_react"], BEAT["fix_next"]]


def test_둘째_문장의_틀린_이름은_도중엔_안_보고_끝에서_본다():
    said = "멍멍, 강아지는 멍멍 하고 울어! 그럼 사자는 어흥?"
    beat = dict(BEAT, names=BEAT["names"] + ["사자"])
    assert check_stream(said, beat) is None
    r = check_done(said, beat)
    assert r.reason == "wrong_name" and not r.in_first
    assert r.cut_char == first_sentence_end(said) and r.lines == [BEAT["fix_next"]]


def test_질문이_빠지면_끝에서_첫_문장_뒤를_바꾼다():
    said = "멍멍, 강아지는 멍멍 하고 울어! 이제 따라 해볼까?"
    r = check_done(said, BEAT)
    assert r.reason == "missing_next" and r.lines == [BEAT["fix_next"]]


def test_한_문장뿐이고_질문이_빠지면_끝에_붙인다():
    said = "멍멍, 강아지는 멍멍 하고 울어"
    r = check_done(said, BEAT)
    assert r.cut_char == len(said) and r.lines == [BEAT["fix_next"]]


def test_제대로면_끝에서도_없다():
    assert check_done("멍멍, 강아지는 멍멍 하고 울어! 그럼 고양이는 어떻게 울어?", BEAT) is None


def test_조각이_빈_비트는_아무것도_안_한다():
    assert check_done("돼지 꿀꿀", dict(BEAT, fix_react="", fix_next="")) is None


def test_쉼_찾기는_가까운_무음_안을_고른다():
    sr = 24000
    talk = np.sin(np.arange(sr) * 0.3).astype(np.float32) * 0.5
    audio = np.concatenate([talk, np.zeros(int(0.2 * sr), np.float32), talk])
    cut = quiet_cut(audio, near=int(1.3 * sr), sr=sr)
    assert sr <= cut <= int(1.2 * sr)
