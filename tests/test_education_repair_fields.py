import random

from app.education_modes import (ANIMAL_ITEMS, REPEAT_ITEMS, GameManager, all_fix_phrases)


def _play(start, answers):
    random.seed(3)
    g = GameManager(render=None)
    beats = [g.maybe_start_beat(start)]
    for a in answers:
        b = g.handle_beat(a)
        if b is None:
            break
        beats.append(b)
    return beats


def test_모든_비트에_조각과_허용_이름이_있다():
    for beats in (_play("동물 소리 놀이 하자", ["멍멍", "몰라", "몰라", "야옹", "음메", "응", "그만"]),
                  _play("따라 말하기 놀이 하자", ["사과", "몰라", "몰라", "우유", "응", "그만"])):
        for b in beats:
            assert set(b) >= {"fix_react", "fix_next", "next_need", "allow", "names"}
            assert b["fix_react"] or b["fix_next"]
            assert set(b["allow"]) <= set(b["names"])


def test_다음_질문_비트는_다음_이름을_허용하고_요구한다():
    b = _play("동물 소리 놀이 하자", ["멍멍"])[1]
    nxt = b["allow"][-1]
    assert nxt in b["fix_next"] and nxt in b["next_need"]


def test_조각_문장은_카드에서_전부_나온다():
    ps = all_fix_phrases()
    assert "강아지는 멍멍 하고 울어!" in ps
    assert "고양이는 어떻게 울어?" in ps or "고양이는 야옹?" in ps
    assert "따라 해봐, 사과!" in ps and "더 할래?" in ps
    assert all("{" not in p for p in ps) and len(ps) == len(set(ps))
    assert len(ps) < 80


def test_동물_놀이는_카드_밖_흔한_동물도_이름_목록에_있다():
    # 09-22 실서버: 모델이 "호랑이는 어흥!" 을 지어냈는데 카드에 없어 못 잡았다.
    from app.game_repair import check_stream
    b = _play("동물 소리 놀이 하자", [])[0]
    assert "호랑이" in b["names"] and "사자" in b["names"]
    assert check_stream("호랑이는 어흥! ", b) is not None


def test_되묻기는_정답_소리도_확인한다():
    beats = _play("동물 소리 놀이 하자", ["몰라"])
    retry = beats[1]
    assert retry["fix_next"].startswith("같이 해보자") and len(retry["next_need"]) == 2
