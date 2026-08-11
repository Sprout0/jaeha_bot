"""놀이 모드(STEP 8) — 결정론 상태머신 + LLM 문구 렌더링.

설계 원칙(모델이 바뀌어도 재활용되게):
  - 흐름/정답판정/트리거는 '코드'가 결정한다(상태머신). LLM 은 흐름을 못 건드린다.
  - 칭찬·질문 '문장'만 LLM 이 렌더링한다. 렌더 실패/드리프트 시엔 템플릿으로 폴백해
    놀이가 절대 멈추지 않는다(안전망).
  - STT 오인식에 강하게: '흉내/따라' 방향이라 정답 단어 인식이 필요 없다.
    아이가 뭐라도 말하면 성공(칭찬), 소리가 얼추 맞으면(jamo 유사도) 보너스 리액션.

놀이 '내용'(동물·단어)은 코드가 아니라 scenarios/scenario_cards.json 에서 읽는다(편집 쉽게).
두 놀이(동물 소리 / 따라 말하기)는 같은 뼈대(Game)를 공유하고 내용(items)만 다르다.
배치(3~5개)마다 "더 할래?"로 끊을 기회를 주고, 언제든 "그만"이면 종료한다.

마이크 없이 검증: GameManager(render=None) 이면 템플릿만 쓰므로 LLM 없이 로직 확인 가능.
실행: python -m app.education_modes
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from .text_norm import jamo_ratio

log = logging.getLogger("jaeha_bot.education")

_SCENARIO_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "scenario_cards.json"

# JSON 로드 실패 시에도 놀이가 되도록 하는 최소 기본값.
_DEFAULT_ANIMALS = [("강아지", "멍멍"), ("고양이", "야옹"), ("소", "음메"), ("돼지", "꿀꿀")]
_DEFAULT_WORDS = ["엄마", "사과", "바나나"]


def _load_items() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """scenario_cards.json 에서 (동물 소리, 따라말 단어) 항목을 읽는다.

    동물: (answer=동물이름, sound=울음소리) 쌍. 따라말: (word, word) 쌍.
    파일이 없거나 깨지면 기본값으로 대체(놀이는 계속 동작).
    """
    try:
        data = json.loads(_SCENARIO_PATH.read_text(encoding="utf-8"))
        animals = [(c["answer"], c["sound"])
                   for c in data.get("sound_match", {}).get("cards", [])
                   if c.get("answer") and c.get("sound")]
        words = [(c["word"], c["word"])
                 for c in data.get("repeat_word", {}).get("cards", [])
                 if c.get("word")]
        if not animals:
            animals = list(_DEFAULT_ANIMALS)
        if not words:
            words = [(w, w) for w in _DEFAULT_WORDS]
        return animals, words
    except (OSError, ValueError, KeyError) as e:
        log.warning("놀이 카드 로드 실패(기본값 사용): %s", e)
        return list(_DEFAULT_ANIMALS), [(w, w) for w in _DEFAULT_WORDS]


ANIMAL_ITEMS, REPEAT_ITEMS = _load_items()

# ── 트리거/응답 판정용 어휘(1단: 어휘 + 자모 유사도) ────────────────────────────
# 정확한 문장이 아니어도, 오인식·변형·유사표현을 관대하게 잡는다.
_ANIMAL_TRIGGERS = ["동물소리놀이", "동물놀이", "동물소리", "울음소리", "동물흉내", "동물소리게임"]
_REPEAT_TRIGGERS = ["따라말하기", "따라하기", "따라말해", "따라쟁이", "따라해놀이"]

# 구성 단서(아래 match_trigger 2번)에 쓸 동물 이름의 최소 길이.
# 🔴 '소'·'양' 같은 한 글자 이름은 흔한 낱말 속에 그대로 들어 있다 — '목소리', '소파',
#    '양말', '모양'. 여기에 '소리'나 '놀이'만 겹치면 아무 말이나 놀이로 바뀐다.
#    실기(2026-08-10)에서 아이가 "와 목소리가 왜그래?" 한 마디에 동물 놀이가 시작돼
#    4턴을 끌려갔다. claims.py 의 _MIN_NAME 과 같은 취지의 가드다.
#    대가: "소 놀이 하자"로는 못 켠다. '동물 소리'라고 하거나 두 글자 이름을 쓰면 된다.
_MIN_ANIMAL_NAME = 2

# 놀이에서 빠져나오는 말. '그만' 계열만 있으면 아이가 흔히 쓰는 "다른 놀이 하자"에
# 갇힌다(실기에서 실제로 갇혔다). ⚠️ '다른 동물'은 계속하겠다는 뜻이라 넣지 않는다.
_STOP_WORDS = ["그만", "안할래", "싫어", "그만할래", "안해", "하기싫어", "됐어", "끝났어",
               "다른놀이", "딴놀이", "다른거", "딴거"]
_NO_WORDS = ["아니", "안할래", "그만", "싫어", "됐어", "안해"]


def _norm(s: str) -> str:
    """공백·문장부호 제거(한글/영숫자만 남김). 트리거 매칭 전처리."""
    return re.sub(r"\W+", "", s or "")


def _j(word: str) -> str:
    """받침 유무로 주격 조사 은/는을 붙인다(양→양은, 강아지→강아지는). TTS 어색함 방지."""
    if not word:
        return word
    last = word[-1]
    if 0xAC00 <= ord(last) <= 0xD7A3:
        has_jong = (ord(last) - 0xAC00) % 28 != 0
        return word + ("은" if has_jong else "는")
    return word + "는"


def _any_in(norm: str, words: list[str]) -> bool:
    return any(w in norm for w in words)


def _is_stop(text: str) -> bool:
    return _any_in(_norm(text), _STOP_WORDS)


def _is_no(text: str) -> bool:
    return _any_in(_norm(text), _NO_WORDS)


def match_trigger(text: str) -> str | None:
    """발화가 어떤 놀이 시작 요청인지 판정한다. animal/repeat/None.

    1단(어휘+자모): 동의어 부분일치, 가까운 자모 유사도, 구성 단서(동물명+놀이/소리 등).
    """
    n = _norm(text)
    if not n:
        return None
    # 1) 동의어 부분일치 또는 근접(오인식 보정)
    for t in _ANIMAL_TRIGGERS:
        if t in n or jamo_ratio(n, t) <= 0.2:
            return "animal"
    for t in _REPEAT_TRIGGERS:
        if t in n or jamo_ratio(n, t) <= 0.2:
            return "repeat"
    # 2) 구성 단서: (동물 or 동물이름) + 놀이/소리/흉내 → 동물놀이
    animal_names = [a for a, _ in ANIMAL_ITEMS if len(a) >= _MIN_ANIMAL_NAME]
    if ("동물" in n or any(a in n for a in animal_names)) and (
        "놀이" in n or "게임" in n or "소리" in n or "흉내" in n or "울음" in n
    ):
        return "animal"
    # 3) '따라' + 말/해/하기/놀이 → 따라말하기
    if "따라" in n and ("말" in n or "해" in n or "하기" in n or "놀이" in n or "쟁이" in n):
        return "repeat"
    return None


# ── Beat: 상태머신이 만드는 '무엇을 말할지'(구조화된 의도) ──────────────────────
def _beat(fallback: str, instruction: str | None = None, require=()) -> dict:
    """fallback=템플릿(안전망), instruction=LLM 렌더 지시, require=LLM 출력에 꼭 있어야 할 토큰."""
    return {"fallback": fallback, "instruction": instruction, "require": list(require)}


# ── 공통 게임 뼈대(상태머신) ───────────────────────────────────────────────────
class Game:
    """배치 진행 + 체크포인트('더 할래?') + 언제든 종료를 처리하는 상태머신.

    내용(items)과 문구 빌더(_intro_beat 등)는 하위 클래스가 채운다.
    상태: await_answer(아이 대답 기다림) / await_continue(더 할래? 대답 기다림).
    """

    ITEMS: list[tuple[str, str]] = []   # (제시대상, 기대 소리/단어)
    BATCH_MIN = 4
    BATCH_MAX = 4

    def __init__(self) -> None:
        self.seq = list(self.ITEMS)
        random.shuffle(self.seq)
        self.pos = 0
        self.batch: list[tuple[str, str]] = []
        self.bi = 0
        self.state = "await_answer"
        self.done = False

    def _next_batch(self) -> None:
        n = random.randint(self.BATCH_MIN, self.BATCH_MAX)
        self.batch = []
        for _ in range(n):
            if self.pos >= len(self.seq):
                random.shuffle(self.seq)
                self.pos = 0
            self.batch.append(self.seq[self.pos])
            self.pos += 1
        self.bi = 0
        self.state = "await_answer"

    def start(self) -> dict:
        self._next_batch()
        subj, tgt = self.batch[0]
        return self._intro_beat(subj, tgt)

    def step(self, text: str) -> dict:
        # 언제든 '그만' 이면 즉시 종료
        if _is_stop(text):
            self.done = True
            return self._bye_beat()

        # 체크포인트: '더 할래?' 에 대한 대답 해석
        if self.state == "await_continue":
            if _is_no(text):
                self.done = True
                return self._bye_beat()
            # 응/모호 → 다음 배치 시작(그만은 위에서 이미 처리됨)
            self._next_batch()
            subj, tgt = self.batch[0]
            return self._ask_beat(subj, tgt)

        # await_answer: 현재 항목에 리액션하고 다음으로
        subj, tgt = self.batch[self.bi]
        said = self._heard(tgt, text)   # 인정한 낱말 / 못 알아들었으면 None
        self.bi += 1
        if self.bi >= len(self.batch):
            # 배치 끝 → 칭찬 + '더 할래?' (체크포인트 진입)
            self.state = "await_continue"
            return self._react_checkpoint_beat(subj, tgt, said)
        nsubj, ntgt = self.batch[self.bi]
        return self._react_next_beat(subj, tgt, said, nsubj, ntgt)

    # 소리/단어가 얼추 맞는지(STT 관대). 하위 클래스 공통 구현.
    def _is_close(self, target: str, text: str) -> bool:
        return jamo_ratio(_norm(text), _norm(target)) <= 0.5

    def _heard(self, target: str, text: str) -> str | None:
        """얼추 맞으면 **카드의 정답 낱말**을 돌려준다. 아이 발화 원문은 내보내지 않는다.

        🔴 원문을 되받으면 STT 오인식이 증폭된다 — 아이가 '멍멍' 했는데 '명명'으로
           받아쓰면 봇이 "명명! 강아지가 명명 해!" 라며 잘못된 발음을 가르친다.
           기대 어휘로 정규화해 되받으면 그 사고가 구조적으로 불가능하다.
           (기존 '제한 어휘 priming' 방침과 같은 결)
        """
        return target if self._is_close(target, text) else None

    def _bye_beat(self) -> dict:
        return _beat("재밌었어! 또 놀자!",
                     "아이랑 놀이를 즐겁게 마무리하는 인사를 밝은 반말 한 문장으로 해.", [])

    # 하위 클래스가 구현
    def _intro_beat(self, subj, tgt) -> dict: raise NotImplementedError
    def _ask_beat(self, subj, tgt) -> dict: raise NotImplementedError
    def _react_next_beat(self, subj, tgt, said, nsubj, ntgt) -> dict: raise NotImplementedError
    def _react_checkpoint_beat(self, subj, tgt, said) -> dict: raise NotImplementedError


# ── 동물 소리 놀이(흉내 방향) ──────────────────────────────────────────────────
class AnimalSoundGame(Game):
    ITEMS = ANIMAL_ITEMS
    BATCH_MIN = 4
    BATCH_MAX = 4

    # require 에 다음 동물 이름 + '울어'(질문 형태)를 함께 넣어, LLM 이 설명으로
    # 새고 질문을 빠뜨리면(고양이는 눈빛으로 표현해…) 검증 실패→템플릿으로 폴백된다.
    def _intro_beat(self, animal, sound):
        return _beat(
            f"좋아, 동물 소리 놀이 하자! {_j(animal)} 어떻게 울어?",
            f"아이랑 동물 소리 놀이를 시작해. 밝게 인사하고 '{_j(animal)} 어떻게 울어?' 하고 물어봐. 반말로 짧게.",
            [animal, "울어"])

    def _ask_beat(self, animal, sound):
        return _beat(
            f"좋아! {_j(animal)} 어떻게 울어?",
            f"놀이를 이어서 '{_j(animal)} 어떻게 울어?' 하고 밝게 물어봐. 반말 한 문장.",
            [animal, "울어"])

    def _react_next_beat(self, animal, sound, said, nanimal, nsound):
        if said:
            # 모방(먼저 그대로 따라 하기) → 확장(한두 낱말만 붙이기).
            # said 는 _heard 가 정규화한 카드의 정답 낱말이라 오인식이 섞이지 않는다.
            fb = f"{said}! {_j(animal)} {said} 하고 울어! 그럼 {_j(nanimal)}?"
            ins = (f"아이가 '{said}' 라고 말했어. 먼저 '{said}!' 하고 그대로 따라 하고, "
                   f"'{_j(animal)} {said} 하고 울어' 처럼 한두 낱말만 붙여 늘려줘. "
                   f"그다음 '{_j(nanimal)}?' 하고 물어봐. 반말 두 문장.")
            req = [said, nanimal]
        else:
            # 지적하지 않고 정답 소리를 들려준다(모델링).
            fb = f"{_j(animal)} {sound} 하고 울어! 그럼 {_j(nanimal)}?"
            ins = (f"{_j(animal)} '{sound}' 하고 운다고 밝게 알려주고, "
                   f"이어서 '{_j(nanimal)}?' 하고 물어봐. 반말 두 문장.")
            req = [sound, nanimal]
        return _beat(fb, ins, req)

    def _react_checkpoint_beat(self, animal, sound, said):
        praise = (f"{said}! {_j(animal)} {said} 하고 울어!" if said
                  else f"{_j(animal)} {sound} 하고 울어!")
        return _beat(f"{praise} 더 할래?",
                     f"'{praise}' 를 밝게 말하고 '더 할래?' 하고 물어봐. 반말.",
                     [sound, "더"])


# ── 따라 말하기 놀이(배치 3~5개) ───────────────────────────────────────────────
class RepeatWordGame(Game):
    ITEMS = REPEAT_ITEMS
    BATCH_MIN = 3
    BATCH_MAX = 5

    def _intro_beat(self, word, _tgt):
        return _beat(
            f"따라 말하기 놀이 하자! 따라 해봐, {word}!",
            f"아이랑 따라 말하기 놀이를 시작해. 밝게 '따라 해봐, {word}!' 하고 말해. 반말 짧게.",
            [word])

    def _ask_beat(self, word, _tgt):
        return _beat(f"좋아! 따라 해봐, {word}!",
                     f"'따라 해봐, {word}!' 하고 밝게 말해. 반말 한 문장.", [word])

    def _react_next_beat(self, word, _tgt, close, nword, _ntgt):
        fb = (f"우와 잘 따라했어! 이번엔 {nword}!" if close
              else f"잘했어! 이번엔 {nword}!")
        ins = f"아이가 따라 말한 걸 밝게 칭찬하고 이어서 '이번엔 {nword}!' 하고 말해. 반말."
        return _beat(fb, ins, [nword])

    def _react_checkpoint_beat(self, word, _tgt, close):
        return _beat("우와 잘했어! 더 할래?",
                     "아이를 밝게 칭찬하고 '더 할래?' 하고 물어봐. 반말 한 문장.", ["더"])


# ── 라우터: 대화 루프에 끼워 넣는 진입점 ───────────────────────────────────────
class GameManager:
    """대화 루프와 놀이 상태머신을 잇는다.

    render: callable(instruction:str)->str  (LLM 문구 렌더러). None 이면 템플릿만 사용.
    사용(main): handle() 로 진행 중 놀이를 처리, None 이면 maybe_start() 로 시작 트리거 확인.
    """

    def __init__(self, render=None) -> None:
        self.render = render
        self.active: Game | None = None

    def _voice(self, beat: dict | None) -> str | None:
        """Beat 를 실제 문장으로. LLM 렌더 시도→검증 실패 시 템플릿 폴백."""
        if beat is None:
            return None
        if self.render and beat.get("instruction"):
            try:
                out = (self.render(beat["instruction"]) or "").strip()
            except Exception:
                out = ""
            # 비었거나 꼭 필요한 토큰(다음 동물/단어)이 빠졌으면 폴백
            if out and all(tok in out for tok in beat.get("require", [])):
                return out
        return beat["fallback"]

    def maybe_start(self, text: str) -> str | None:
        """시작 트리거면 놀이를 시작하고 첫 멘트를 돌려준다. 아니면 None."""
        if self.active is not None:
            return None
        kind = match_trigger(text)
        if not kind:
            return None
        self.active = AnimalSoundGame() if kind == "animal" else RepeatWordGame()
        return self._voice(self.active.start())

    def handle(self, text: str) -> str | None:
        """진행 중 놀이가 있으면 한 턴 처리. 없으면 None(→ 평소 대화로)."""
        if self.active is None:
            return None
        beat = self.active.step(text)
        if self.active.done:
            self.active = None
        return self._voice(beat)


def _repl() -> None:
    """마이크·LLM 없이 상태머신 로직 확인(템플릿 모드).

    실행: python -m app.education_modes   (종료: 빈 줄/exit)
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mgr = GameManager(render=None)
    print("놀이 상태머신 테스트(템플릿 모드). 예: '동물 소리 놀이 하자' / '따라 말하기 하자'")
    while True:
        try:
            user = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        reply = mgr.handle(user)
        if reply is None:
            reply = mgr.maybe_start(user)
        if reply is None:
            reply = "(평소 대화 → agent.respond 로 감)"
        print(f"재하봇: {reply}")


if __name__ == "__main__":
    _repl()
