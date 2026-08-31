"""봇이 '아는 동물 소리'는 놀이 카드와 **같아야** 한다.

🔴 2026-08-31 실기: [아이] 코끼리 -> [티드] "코끼리는 퉁퉁 해!"
   되물어도("코끼리가 그렇게 울어?") "응, 코끼리는 퉁퉁 울지!" 하고 우겼다.
   재현 실측(카드에 없는 동물 10문항): 기린 "멤멈", 토끼 "끼앙", 원숭이 "응애",
   공룡 "음~~" — 그리고 **되물음에 10개 중 8개가 물러서지 않았다.**
   2세에게 없는 낱말을 확신을 갖고 가르치는 셈이다.

🔴 1차 수정("모르면 지어내지 마라")은 날조를 0 으로 만들었지만 **아는 것까지 흔들었다**:
   같은 8마리에서 이전 7/8 -> 6/8 (돼지가 '음메', 오리가 '꼬끼오'), 게다가 '멍멍'조차
   되물으면 "잘 모르겠어"가 됐다. 놀이의 뼈대가 흔들린 것이다.
   ➡️ **아는 것과 모르는 것을 갈라 줘야 한다** — 여덟은 프롬프트에 적고 확신을 주고,
      나머지만 되묻게 한다.

⚠️ 그러면 프롬프트에 적힌 여덟과 놀이 카드가 **어긋날 수 있다.** 카드를 고친 사람이
   프롬프트를 같이 못 고치면, 봇이 놀이에서 묻는 소리와 대화에서 말하는 소리가 달라진다.
   이 시험이 그 어긋남을 잡는다.
"""
from app.config import settings
from app.education_modes import ANIMAL_ITEMS


def test_every_card_animal_and_its_sound_is_named_in_the_prompt():
    system = settings.prompts["system"]

    missing = [f"{a} {s}" for a, s in ANIMAL_ITEMS
               if a not in system or s not in system]
    assert not missing, (
        f"놀이 카드에 있는데 프롬프트가 모르는 소리: {missing}. "
        f"scenario_cards.json 을 고쳤으면 prompt_templates.yaml 도 같이 고칠 것."
    )


def test_the_prompt_does_not_claim_sounds_for_animals_it_has_no_card_for():
    """카드에 없는 동물을 프롬프트가 '아는 소리'로 적으면 놀이와 대화가 갈린다.

    ⚠️ 기린·토끼·물고기는 **'소리가 없는 예'** 로 적혀 있으므로 소리와 함께 적히면 안 된다.
    """
    system = settings.prompts["system"]
    known = {a for a, _ in ANIMAL_ITEMS}

    for animal, sound in [("코끼리", "뿌우"), ("사자", "어흥"), ("기린", "멤멈"),
                          ("토끼", "끼앙"), ("원숭이", "우끼끼")]:
        if animal in known:
            continue
        assert sound not in system, \
            f"카드에 없는 {animal} 의 소리 '{sound}' 가 프롬프트에 적혀 있다"


def test_the_prompt_tells_it_not_to_invent_the_rest():
    """여덟을 적어 주는 것만으로는 부족하다 — 나머지를 지어내지 말라고 해야 한다."""
    system = settings.prompts["system"]

    assert "지어내지" in system and "같이 만들어볼까" in system, \
        "모르는 동물 소리를 지어내지 말라는 규칙이 사라졌다"
