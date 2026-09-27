import numpy as np

from app.game_repair import (Repair, check_done, check_stream, final_cut, find_name,
                             first_sentence_end, quiet_cut)

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


def test_글자가_덜_와서_끝에_걸린_이름은_다음_글자까지_기다린다():
    # 09-22 실서버: "좋아, 동물 소" 까지 온 순간 '소'(=소리의 앞 글자)를 틀린 이름으로 봤다.
    beat = dict(BEAT, allow=["양"], names=["양", "소"])
    assert check_stream("좋아, 동물 소", beat) is None
    assert check_stream("좋아, 동물 소리", beat) is None
    assert check_stream("소는", beat) is not None


def test_끝에서_틀린_이름을_자를_땐_이름_앞에서만_쉼을_찾는다():
    # 이름 바로 뒤에 더 조용한 틈이 있어도 거기서 자르면 이름이 들린다(2026-09-27).
    sr = 24000
    talk = (np.sin(np.arange(2 * sr) * 0.3) * 0.5).astype(np.float32)
    sec = lambda x: int(x * sr)                                    # noqa: E731
    audio = np.concatenate([talk[:sec(1.2)], talk[:sec(0.1)] * 0.2,   # 1.2~1.3s 작은 쉼
                            talk[:sec(0.3)], np.zeros(sec(0.2), np.float32),  # 1.6~1.8s 무음
                            talk[:sec(1.2)]])                      # 모두 3.0s
    text = "가" * 15 + "소" + "나" * 14                           # 이름 '소' = 1.5s
    rep = Repair(cut_char=15, lines=["x"], reason="wrong_name", in_first=True)
    cut = final_cut(audio, rep, text, sr)
    assert sec(1.2) <= cut <= sec(1.3)
    # 문장 사이를 자를 때는 앞뒤를 본다 — 뒤쪽 무음을 고른다
    rep2 = Repair(cut_char=15, lines=["x"], reason="missing_next")
    assert sec(1.6) <= final_cut(audio, rep2, text, sr) <= sec(1.8)
