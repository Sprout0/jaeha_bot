# 놀이 상호작용 개조 구현 계획 — 확장·모방·완성형

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 놀이 중에 봇이 아이 말을 되받아 한두 낱말 늘려 말하고(확장), 못 알아들으면 한 번 같이 해보자고 권하며, 일부 문항을 완성형으로 묻게 만든다.

**Architecture:** 상태머신(`Game`)이 아이 발화를 `_heard()` 로 **카드의 정답 낱말로 정규화**해 beat 메서드에 넘긴다. beat 는 그 낱말로 확장 문구와 LLM 지시를 만들고, `require` 에 그 낱말을 넣어 LLM 이 빠뜨리면 결정적 템플릿으로 폴백시킨다. 완성형 여부는 카드의 `lead` 필드가 정한다.

**Tech Stack:** Python 3.10(젯슨)/3.12(노트북), pytest, 기존 `app/education_modes.py` 상태머신, `tools/eval_llm.py` 평가 도구.

**설계 문서:** `docs/superpowers/specs/2026-08-11-play-interaction-redesign-design.md`
**근거 문서:** `jaeha_bot_docs/docs/리서치_유아_대화기법_놀이설계.md`

## Global Constraints

- 실행은 conda env `jaeha_bot` 으로만. 젯슨은 `source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot`.
- `main` 단일 브랜치. 브랜치·PR 만들지 않는다(사용자 방침).
- TDD: 테스트를 먼저 쓰고 **실패를 눈으로 확인한 뒤** 구현한다.
- 테스트는 마이크·모델·네트워크 없이 돈다. 검사 대상은 `fallback` 템플릿과 `instruction`·`require` 문자열이다 — LLM 출력은 오프라인에서 재현되지 않는다.
- 한 번에 변수 두 개를 바꾸지 않는다. 자모거리 임계 `0.5`, 배치 크기, `match_trigger` 는 이 계획에서 건드리지 않는다.
- 각 태스크 끝에 **전체 스위트**(`python -m pytest -q`)를 돌린다. 현재 기준선 **187개 통과**.
- 놀이 대사 문자열은 그대로 TTS 로 읽힌다(놀이 경로에는 이모지·마크다운 후처리가 없다). 말줄임표·기호를 쓰지 않는다.

---

### Task 1: `_heard()` 와 동물 소리 놀이 확장

**Files:**
- Modify: `app/education_modes.py` (`Game.step`, `Game` 에 `_heard` 추가, `AnimalSoundGame._react_next_beat`·`_react_checkpoint_beat`)
- Test: `tests/test_game_expansion.py` (신규)

**Interfaces:**
- Produces: `Game._heard(target: str, text: str) -> str | None` — 얼추 맞으면 `target` 을 그대로 돌려주고 아니면 `None`.
- Produces: beat 시그니처가 `close: bool` → `said: str | None` 로 바뀐다. `_react_next_beat(subj, tgt, said, nsubj, ntgt)`, `_react_checkpoint_beat(subj, tgt, said)`.
- Consumes: 기존 `_is_close(target, text) -> bool`, `_beat(fallback, instruction, require)`, `_j(word)`.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_game_expansion.py` 를 새로 만든다.

```python
"""놀이 중 확장(expansion)과 모방(imitation). 마이크·모델 불필요.

문헌에서 효과크기가 가장 큰 개입인데(recast 메타분석 0.76~0.96 SD) 구조적으로
불가능했다 — step() 이 아이 발화를 받고도 beat 에는 '맞았나' 불리언만 넘겼다.
설계: docs/superpowers/specs/2026-08-11-play-interaction-redesign-design.md
"""
from app.education_modes import AnimalSoundGame


def _text(beat):
    """검사 대상은 템플릿과 LLM 지시문이다(LLM 출력은 오프라인에서 재현 불가)."""
    return beat["fallback"] + " " + (beat["instruction"] or "")


def _game(batch):
    g = AnimalSoundGame()
    g.batch = list(batch)
    g.bi = 0
    g.state = "await_answer"
    return g


def test_misheard_text_never_reaches_the_reply():
    # STT 가 '멍멍'을 '명명'으로 받아써도(자모거리 0.333 이라 인정된다) 봇은
    # 카드의 정답 낱말로 되받아야 한다. 원문을 되받으면 잘못된 발음을 가르친다.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    out = _text(g.step("명명"))

    assert "명명" not in out, f"오인식 원문이 새어 나왔다: {out}"
    assert "멍멍" in out


def test_reply_imitates_then_expands():
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    fb = g.step("멍멍")["fallback"]

    assert fb.startswith("멍멍"), f"아이 말을 먼저 그대로 따라 해야 한다: {fb}"
    assert fb.count("멍멍") >= 2, f"따라 한 뒤 늘려 말해야 한다: {fb}"
    assert "강아지" in fb, f"확장 절에 대상 이름이 들어가야 한다: {fb}"


def test_require_forces_the_child_word():
    # LLM 이 아이 말을 빠뜨리면 검증에 걸려 템플릿으로 폴백된다.
    # 확장이 'LLM 의 선의'가 아니라 구조로 보장되는 지점.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    assert "멍멍" in g.step("멍멍")["require"]


def test_checkpoint_also_expands():
    # 배치의 마지막 항목이면 '더 할래?' 체크포인트로 간다. 거기서도 확장은 유지된다.
    g = _game([("강아지", "멍멍")])

    fb = g.step("멍멍")["fallback"]

    assert "멍멍" in fb and "강아지" in fb and "더 할래?" in fb
```

- [ ] **Step 2: 실패를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest tests/test_game_expansion.py -q`
Expected: 4개 FAIL. `test_reply_imitates_then_expands` 는 `AssertionError`(현재 템플릿이 "우와 진짜 강아지 같다!" 로 시작), `test_require_forces_the_child_word` 도 `AssertionError`(require 가 `["고양이","울어"]`).

- [ ] **Step 3: `_heard()` 를 추가한다**

`app/education_modes.py` 의 `Game._is_close` 바로 아래에 넣는다.

```python
    def _heard(self, target: str, text: str) -> str | None:
        """얼추 맞으면 **카드의 정답 낱말**을 돌려준다. 아이 발화 원문은 내보내지 않는다.

        🔴 원문을 되받으면 STT 오인식이 증폭된다 — 아이가 '멍멍' 했는데 '명명'으로
           받아쓰면 봇이 "명명! 강아지가 명명 해!" 라며 잘못된 발음을 가르친다.
           기대 어휘로 정규화해 되받으면 그 사고가 구조적으로 불가능하다.
           (기존 '제한 어휘 priming' 방침과 같은 결)
        """
        return target if self._is_close(target, text) else None
```

- [ ] **Step 4: `step()` 이 `said` 를 넘기게 바꾼다**

`Game.step()` 의 `# await_answer:` 블록을 통째로 교체한다.

```python
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
```

추상 메서드 선언 두 줄도 이름을 맞춘다.

```python
    def _react_next_beat(self, subj, tgt, said, nsubj, ntgt) -> dict: raise NotImplementedError
    def _react_checkpoint_beat(self, subj, tgt, said) -> dict: raise NotImplementedError
```

- [ ] **Step 5: `AnimalSoundGame` 의 리액션 beat 두 개를 다시 쓴다**

```python
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
```

- [ ] **Step 6: 테스트 통과를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest tests/test_game_expansion.py -q`
Expected: 4 passed

- [ ] **Step 7: 전체 스위트로 회귀를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest -q`
Expected: 187 + 4 = 191 passed

`RepeatWordGame` 은 아직 `close` 라는 이름으로 받지만 **깨지지 않는다** — 인자가 위치로
들어가고 `if close:` 가 문자열이면 참, `None` 이면 거짓이라 그대로 동작한다.
다만 되받기를 안 하므로 Task 2 에서 마저 고친다. 여기서 실패가 나면 위치 인자가
어긋난 것이니 Step 4·5 를 다시 볼 것.

- [ ] **Step 8: 커밋**

```bash
git add app/education_modes.py tests/test_game_expansion.py
git commit -m "feat(games): 아이 말을 되받아 늘려 말한다 — 확장은 정답 낱말로만

step() 이 아이 발화를 받고도 beat 에는 불리언만 넘겨서, 상태머신이 만드는 LLM 지시에
아이 말이 들어갈 자리가 없었다. close(bool) 을 said(str|None) 로 바꿔 해결.

🔴 확장에 쓰는 건 아이 발화 원문이 아니라 _heard() 가 정규화한 카드의 정답 낱말이다.
   원문을 되받으면 STT 오인식이 증폭된다(아이 '멍멍' -> STT '명명' -> 봇이 '명명' 을 가르침).
require 에 said 를 넣어 LLM 이 빠뜨리면 결정적 템플릿으로 폴백시킨다."
```

---

### Task 2: 따라 말하기 놀이에 확장 적용

**Files:**
- Modify: `app/education_modes.py` (`RepeatWordGame._react_next_beat`·`_react_checkpoint_beat`)
- Test: `tests/test_game_expansion.py` (추가)

**Interfaces:**
- Consumes: Task 1 의 `said: str | None` 시그니처.

- [ ] **Step 1: 실패하는 테스트를 추가한다**

`tests/test_game_expansion.py` 맨 아래에 붙인다.

```python
from app.education_modes import RepeatWordGame


def _repeat_game(batch):
    g = RepeatWordGame()
    g.batch = list(batch)
    g.bi = 0
    g.state = "await_answer"
    return g


def test_repeat_game_echoes_the_child_word():
    g = _repeat_game([("바나나", "바나나"), ("딸기", "딸기")])

    fb = g.step("바나나")["fallback"]

    assert fb.startswith("바나나"), f"아이 말을 먼저 따라 해야 한다: {fb}"
    assert "딸기" in fb, f"다음 낱말로 이어야 한다: {fb}"


def test_repeat_game_misheard_uses_the_card_word():
    # '바나'만 말해도 인정된다(자모거리 0.333). 되받는 건 카드의 '바나나'여야 한다.
    g = _repeat_game([("바나나", "바나나"), ("딸기", "딸기")])

    out = _text(g.step("바나"))

    assert "바나나" in out
    assert out.count("바나나") >= 2, f"따라 한 뒤 다시 써야 한다: {out}"
```

- [ ] **Step 2: 실패를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest tests/test_game_expansion.py -q`
Expected: 새 테스트 2개 FAIL (현재 템플릿이 "우와 잘 따라했어!" 로 시작)

- [ ] **Step 3: `RepeatWordGame` 리액션 beat 두 개를 다시 쓴다**

```python
    def _react_next_beat(self, word, _tgt, said, nword, _ntgt):
        if said:
            fb = f"{said}! 우와 잘 따라했어! 이번엔 {nword}!"
            ins = (f"아이가 '{said}' 라고 따라 말했어. 먼저 '{said}!' 하고 그대로 따라 하고 "
                   f"밝게 칭찬한 뒤 '이번엔 {nword}!' 하고 말해. 반말 두 문장.")
            req = [said, nword]
        else:
            # 지적하지 않고 낱말을 한 번 더 들려준다.
            fb = f"{word}! 잘했어! 이번엔 {nword}!"
            ins = (f"'{word}!' 를 다시 한 번 들려주고 밝게 격려한 뒤 "
                   f"'이번엔 {nword}!' 하고 말해. 반말 두 문장.")
            req = [word, nword]
        return _beat(fb, ins, req)

    def _react_checkpoint_beat(self, word, _tgt, said):
        praise = f"{said}! 우와 잘 따라했어!" if said else f"{word}! 잘했어!"
        return _beat(f"{praise} 더 할래?",
                     f"'{praise}' 하고 밝게 말한 뒤 '더 할래?' 하고 물어봐. 반말.",
                     [word, "더"])
```

- [ ] **Step 4: 테스트 통과를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest -q`
Expected: 193 passed (191 + 2)

- [ ] **Step 5: 커밋**

```bash
git add app/education_modes.py tests/test_game_expansion.py
git commit -m "feat(games): 따라 말하기도 아이 말을 되받는다

동물 소리 놀이와 같은 형태. 인정한 낱말은 카드의 것이라
'바나'만 말해도 봇은 '바나나' 로 되받는다."
```

---

### Task 3: 완성형 프롬프트 — 카드의 `lead`

**Files:**
- Modify: `scenarios/scenario_cards.json` (`sound_match` 카드 4개에 `lead` 추가)
- Modify: `app/education_modes.py` (`_load_items`, 모듈 상수 `_LEADS`, `AnimalSoundGame._prompt`·`_intro_beat`·`_ask_beat`·`_react_next_beat` 꼬리)
- Test: `tests/test_game_expansion.py` (추가)

**Interfaces:**
- Produces: 모듈 상수 `_LEADS: dict[str, str]` — 기대 소리 → 완성형 앞부분. 예 `{"멍멍": "멍"}`.
- Produces: `AnimalSoundGame._prompt(animal, sound) -> tuple[str, list[str]]` — 묻는 문장과 그 문장의 require 토큰.

- [ ] **Step 1: 실패하는 테스트를 추가한다**

```python
def test_completion_prompt_used_when_card_has_lead():
    # 2세에게는 wh- 질문보다 완성형이 맞다(문헌: 1~2세 권장 유형).
    # 정답을 정확히 받아쓸 필요가 없어 우리 STT 약점과도 궁합이 좋다.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    text, req = g._prompt("강아지", "멍멍")

    assert text == "강아지는 멍?", f"완성형이어야 한다: {text}"
    assert "멍" in req


def test_wh_prompt_used_when_card_has_no_lead():
    # '음메' 는 앞부분('음')이 어색해 lead 를 안 넣는다 -> 기존 wh- 유지.
    # 한 형태만 반복하면 단조로워지므로 섞는 것이 설계 의도다.
    g = _game([("소", "음메")])

    text, req = g._prompt("소", "음메")

    assert text == "소는 어떻게 울어?"
    assert "울어" in req


def test_ask_beat_uses_the_prompt_helper():
    g = _game([("강아지", "멍멍")])

    beat = g._ask_beat("강아지", "멍멍")

    assert "강아지는 멍?" in beat["fallback"]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest tests/test_game_expansion.py -q`
Expected: 3 FAIL — `AttributeError: 'AnimalSoundGame' object has no attribute '_prompt'`

- [ ] **Step 3: 카드에 `lead` 를 넣는다**

`scenarios/scenario_cards.json` 의 `sound_match.cards` 중 **네 개만** 고친다. 앞부분이 자연스러운 것만 고른다.

```json
{"sound": "멍멍", "answer": "강아지", "lead": "멍"},
{"sound": "꿀꿀", "answer": "돼지", "lead": "꿀"},
{"sound": "꽥꽥", "answer": "오리", "lead": "꽥"},
{"sound": "삐약삐약", "answer": "병아리", "lead": "삐약"}
```

야옹·음메·개굴개굴·매애는 그대로 둔다(앞부분이 어색하거나 두 음절이 한 덩어리라).

⚠️ 파일 전체를 `json.dump` 로 다시 쓰지 말 것 — 들여쓰기가 통째로 바뀌어 diff 가 폭발한다(2026-08-07 에 실제로 겪었다). 해당 줄만 직접 편집한다.

- [ ] **Step 4: `_LEADS` 를 싣는다**

`_load_items()` 가 세 번째 값을 돌려주게 한다.

```python
def _load_items() -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, str]]:
    """scenario_cards.json 에서 (동물, 따라말, 완성형 lead) 를 읽는다.

    lead 는 '완성형 프롬프트'의 앞부분이다("멍멍" -> "멍" -> "강아지는 멍?").
    ⚠️ ITEMS 를 3-튜플로 만들면 match_trigger 의 `[a for a, _ in ANIMAL_ITEMS]` 가
       깨진다. 그래서 별도 dict 로 싣고 기존 자료구조는 건드리지 않는다.
    """
    try:
        data = json.loads(_SCENARIO_PATH.read_text(encoding="utf-8"))
        animals = [(c["answer"], c["sound"])
                   for c in data.get("sound_match", {}).get("cards", [])
                   if c.get("answer") and c.get("sound")]
        words = [(c["word"], c["word"])
                 for c in data.get("repeat_word", {}).get("cards", [])
                 if c.get("word")]
        leads = {c["sound"]: c["lead"]
                 for c in data.get("sound_match", {}).get("cards", [])
                 if c.get("sound") and c.get("lead")}
        if not animals:
            animals = list(_DEFAULT_ANIMALS)
        if not words:
            words = [(w, w) for w in _DEFAULT_WORDS]
        return animals, words, leads
    except (OSError, ValueError, KeyError) as e:
        log.warning("놀이 카드 로드 실패(기본값 사용): %s", e)
        return list(_DEFAULT_ANIMALS), [(w, w) for w in _DEFAULT_WORDS], {}


ANIMAL_ITEMS, REPEAT_ITEMS, _LEADS = _load_items()
```

- [ ] **Step 5: `_prompt()` 를 만들고 세 곳에서 쓴다**

`AnimalSoundGame` 안에 넣는다.

```python
    def _prompt(self, animal, sound):
        """묻는 말과 그 문장의 require 토큰.

        lead 가 있으면 완성형('강아지는 멍?'), 없으면 기존 wh-('어떻게 울어?').
        비율은 카드의 lead 유무로 조절하니 코드를 안 건드리고 실험할 수 있다.
        ⚠️ 말줄임표('멍...?')를 쓰지 않는다 — 이 문자열은 그대로 TTS 로 읽히고,
           기호를 소리 내 읽을 위험이 있다. 물음표만으로 억양을 만든다.
        """
        lead = _LEADS.get(sound)
        if lead:
            return f"{_j(animal)} {lead}?", [animal, lead]
        return f"{_j(animal)} 어떻게 울어?", [animal, "울어"]
```

`_intro_beat` / `_ask_beat` 를 교체한다.

```python
    def _intro_beat(self, animal, sound):
        q, req = self._prompt(animal, sound)
        return _beat(f"좋아, 동물 소리 놀이 하자! {q}",
                     f"아이랑 동물 소리 놀이를 시작해. 밝게 인사하고 '{q}' 하고 물어봐. 반말로 짧게.",
                     req)

    def _ask_beat(self, animal, sound):
        q, req = self._prompt(animal, sound)
        return _beat(f"좋아! {q}",
                     f"놀이를 이어서 '{q}' 하고 밝게 물어봐. 반말 한 문장.",
                     req)
```

`_react_next_beat` 의 꼬리도 완성형을 쓰게 바꾼다. Task 1 에서 쓴 몸통을 아래로 교체한다.

```python
    def _react_next_beat(self, animal, sound, said, nanimal, nsound):
        nq, nreq = self._prompt(nanimal, nsound)
        if said:
            fb = f"{said}! {_j(animal)} {said} 하고 울어! 그럼 {nq}"
            ins = (f"아이가 '{said}' 라고 말했어. 먼저 '{said}!' 하고 그대로 따라 하고, "
                   f"'{_j(animal)} {said} 하고 울어' 처럼 한두 낱말만 붙여 늘려줘. "
                   f"그다음 '{nq}' 하고 물어봐. 반말 두 문장.")
            req = [said] + nreq
        else:
            fb = f"{_j(animal)} {sound} 하고 울어! 그럼 {nq}"
            ins = (f"{_j(animal)} '{sound}' 하고 운다고 밝게 알려주고, "
                   f"이어서 '{nq}' 하고 물어봐. 반말 두 문장.")
            req = [sound] + nreq
        return _beat(fb, ins, req)
```

- [ ] **Step 6: 테스트 통과를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest -q`
Expected: 196 passed (193 + 3)
`test_reply_imitates_then_expands` 가 여전히 통과하는지 확인한다 — 꼬리가 "그럼 고양이는 어떻게 울어?" 로 바뀌었을 뿐 앞부분은 그대로다.

- [ ] **Step 7: 커밋**

```bash
git add app/education_modes.py scenarios/scenario_cards.json tests/test_game_expansion.py
git commit -m "feat(games): 완성형 프롬프트 — '강아지는 멍?'

2세에게는 wh- 질문보다 완성형이 맞다(문헌: 1~2세 권장 유형). 정답을 정확히
받아쓸 필요가 없어 우리 STT 약점과도 궁합이 좋다 — 아이가 뭐라도 소리 내면 된다.

전부 바꾸지 않는다. 카드에 lead 가 있으면 완성형, 없으면 기존 wh- 다.
비율을 JSON 에서 조절하니 코드를 안 건드리고 실험할 수 있고,
'음메'->'음' 처럼 어색한 건 안 넣으면 그만이다.
말줄임표는 쓰지 않는다 — 이 문자열은 그대로 TTS 로 읽힌다."
```

---

### Task 4: 재시도 — "같이 해보자" 한 번

**Files:**
- Modify: `app/education_modes.py` (`Game.__init__`·`_next_batch`·`step`, `_retry_beat` 추상 + 두 게임 구현)
- Test: `tests/test_game_expansion.py` (추가)

**Interfaces:**
- Produces: `Game.retried: bool` 상태 변수.
- Produces: `Game._retry_beat(subj, tgt) -> dict` (추상), `AnimalSoundGame._retry_beat`, `RepeatWordGame._retry_beat`.

- [ ] **Step 1: 실패하는 테스트를 추가한다**

```python
def test_first_miss_invites_the_child_to_try_together():
    # 모방은 확장만큼 중요한 기법이다. 2세가 실제로 소리를 낼 계기를 한 번 준다.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    beat = g.step("몰라")

    assert "같이 해보자" in beat["fallback"], beat["fallback"]
    assert "멍멍" in beat["fallback"]
    assert g.bi == 0, "재시도 중에는 같은 항목에 머물러야 한다"


def test_second_miss_moves_on():
    # 기회는 정확히 한 번. 두 번째엔 맞든 틀리든 넘어간다(무한 루프 방지).
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])
    g.step("몰라")

    beat = g.step("몰라")

    assert "같이 해보자" not in beat["fallback"]
    assert g.bi == 1, "두 번째엔 다음 항목으로 넘어가야 한다"
    assert "멍멍" in beat["fallback"], "지적 대신 정답 소리를 들려준다(모델링)"


def test_retry_then_correct_gets_normal_expansion():
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])
    g.step("몰라")

    fb = g.step("멍멍")["fallback"]

    assert fb.startswith("멍멍")
    assert fb.count("멍멍") >= 2


def test_never_scolds_on_a_miss():
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])

    out = _text(g.step("몰라"))

    for scold in ("틀렸", "아니야", "아니란다"):
        assert scold not in out, f"지적하면 안 된다: {out}"


def test_stop_works_during_retry():
    # 2026-08-10 에 '빠져나갈 수 없는' 실패를 겪었다. 재시도 중에도 탈출은 열려 있어야 한다.
    g = _game([("강아지", "멍멍"), ("고양이", "야옹")])
    g.step("몰라")

    g.step("그만")

    assert g.done is True


def test_retry_flag_resets_between_items():
    g = _game([("강아지", "멍멍"), ("고양이", "야옹"), ("돼지", "꿀꿀")])
    g.step("몰라")     # 1번 항목 재시도
    g.step("몰라")     # 넘어감 (bi=1)

    beat = g.step("몰라")   # 2번 항목의 첫 실패 → 다시 재시도가 나와야 한다

    assert "같이 해보자" in beat["fallback"], beat["fallback"]
```

- [ ] **Step 2: 실패를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest tests/test_game_expansion.py -q`
Expected: 5 FAIL (`test_never_scolds_on_a_miss` 는 이미 통과할 수 있다 — 현재도 지적하지 않는다)

- [ ] **Step 3: `retried` 상태를 추가한다**

`Game.__init__` 끝에 한 줄, `_next_batch()` 끝에 한 줄.

```python
        self.done = False
        self.retried = False   # 현재 항목에서 '같이 해보자' 를 이미 한 번 줬는지
```

```python
        self.bi = 0
        self.retried = False
        self.state = "await_answer"
```

- [ ] **Step 4: `step()` 에 재시도 분기를 넣는다**

Task 1 에서 만든 `# await_answer:` 블록 앞부분을 교체한다.

```python
        # await_answer: 현재 항목에 리액션하고 다음으로
        subj, tgt = self.batch[self.bi]
        said = self._heard(tgt, text)
        if said is None and not self.retried:
            # 못 알아들었으면 같은 항목에 한 번 더 머문다 — 아이가 소리 낼 계기를 준다.
            # 기회는 정확히 한 번이다(두 번째엔 맞든 틀리든 진행 → 무한 루프 방지).
            self.retried = True
            return self._retry_beat(subj, tgt)
        self.retried = False
        self.bi += 1
```

추상 선언도 추가한다.

```python
    def _retry_beat(self, subj, tgt) -> dict: raise NotImplementedError
```

- [ ] **Step 5: 두 게임에 `_retry_beat` 를 구현한다**

`AnimalSoundGame`:

```python
    def _retry_beat(self, animal, sound):
        fb = f"{_j(animal)} {sound}! 같이 해보자, {sound}!"
        ins = (f"아이가 못 맞혔어. 지적하지 말고 '{_j(animal)} {sound}!' 하고 들려준 뒤 "
               f"'같이 해보자, {sound}!' 하고 권해. 반말 두 문장.")
        return _beat(fb, ins, [sound])
```

`RepeatWordGame`:

```python
    def _retry_beat(self, word, _tgt):
        fb = f"{word}! 같이 해보자, {word}!"
        ins = (f"지적하지 말고 '{word}!' 를 다시 들려주고 "
               f"'같이 해보자, {word}!' 하고 권해. 반말 두 문장.")
        return _beat(fb, ins, [word])
```

- [ ] **Step 6: 테스트 통과를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest -q`
Expected: 202 passed (196 + 6)

- [ ] **Step 7: 커밋**

```bash
git add app/education_modes.py tests/test_game_expansion.py
git commit -m "feat(games): 못 알아들으면 '같이 해보자' 로 한 번 더

모방은 확장만큼 중요한 기법이고(문헌), 2세가 실제로 소리를 낼 계기가 된다.
기회는 정확히 한 번 — 두 번째엔 맞든 틀리든 넘어간다(무한 루프 방지).
'그만' 은 step() 맨 위에서 처리되므로 재시도 중에도 즉시 빠져나간다
(2026-08-10 에 '빠져나갈 수 없는' 실패를 겪었다. 테스트로 고정).

항목 수는 그대로지만 턴 수가 최대 두 배가 될 수 있다. '더 할래?' 가 받아낸다."
```

---

### Task 5: 계측 — 어휘 재사용과 확장 폭

**Files:**
- Modify: `app/metrics.py` (`_words`, `_reuse_ratio` 추가, `record_turn` 에 `child_text` 인자와 필드 2개)
- Modify: `app/main.py:258` (`child_text=text` 전달)
- Test: `tests/test_metrics_expansion.py` (신규)

**Interfaces:**
- Produces: `metrics._words(text: str) -> list[str]`, `metrics._reuse_ratio(child: str, reply: str) -> float | None`
- Produces: `record_turn(..., child_text: str = "")` — 기록에 `expansion_delta: int|None`, `reuse: float|None` 추가.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_metrics_expansion.py` 를 새로 만든다.

```python
"""확장·모방 대리지표. 아이가 생기는 날 바로 재려고 배선만 해 둔다.

⚠️ 둘 다 대리지표다. '아이가 실제로 더 말했는가'는 재지 못한다.
   안전 판정기를 하한선으로 읽기로 한 것과 같은 태도가 필요하다.
"""
import json

from app.metrics import MetricsLogger, _reuse_ratio, _words


def test_words_drops_punctuation():
    assert _words("멍멍! 강아지는 멍멍 하고 울어!") == \
        ["멍멍", "강아지는", "멍멍", "하고", "울어"]


def test_reuse_counts_child_words_echoed_back():
    assert _reuse_ratio("멍멍", "멍멍! 강아지는 멍멍 하고 울어!") == 1.0
    assert _reuse_ratio("멍멍", "그건 아직 못 해!") == 0.0


def test_reuse_survives_korean_particles():
    # 어절 단위 정확 비교면 '멍멍' 과 '멍멍을' 이 다르다고 볼 것이다.
    # 부분일치라 조사가 붙어도 잡힌다.
    assert _reuse_ratio("멍멍", "멍멍을 해볼까?") == 1.0


def test_reuse_is_none_when_child_said_nothing():
    assert _reuse_ratio("", "안녕!") is None


def test_record_turn_logs_expansion_fields(tmp_path):
    m = MetricsLogger(enabled=True, tag="test", log_dir=str(tmp_path))
    m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, think_s=0.9, think_kind="game",
                  tts_first_s=0.8, tts_play_s=2.0,
                  reply="멍멍! 강아지는 멍멍 하고 울어!", child_text="멍멍")

    rec = json.loads(m.path.read_text(encoding="utf-8").splitlines()[-1])

    assert rec["expansion_delta"] == 4     # 봇 5낱말 - 아이 1낱말
    assert rec["reuse"] == 1.0
```

- [ ] **Step 2: 실패를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest tests/test_metrics_expansion.py -q`
Expected: `ImportError: cannot import name '_words' from 'app.metrics'`

- [ ] **Step 3: `metrics.py` 에 헬퍼를 추가한다**

`import re` 를 위쪽 import 에 더하고, `_p90` 아래에 넣는다.

```python
def _words(text: str) -> list[str]:
    """공백 기준 낱말(문장부호 제거). 확장 폭·재사용률 계산용 대충 나누기다."""
    return re.sub(r"[^\w\s]", " ", text or "").split()


def _reuse_ratio(child: str, reply: str) -> float | None:
    """아이 낱말 중 봇 답변에 다시 나온 비율. 모방·확장의 **대리지표**다.

    부분일치로 센다 — 어절 정확 비교면 조사가 붙은 '멍멍을' 을 못 잡는다.
    ⚠️ 정확한 값이 아니라 추세를 보는 숫자다. 아이가 실제로 더 말했는지는 재지 못한다.
    """
    cw = _words(child)
    if not cw:
        return None
    rep = reply or ""
    return round(sum(1 for w in cw if w in rep) / len(cw), 2)
```

- [ ] **Step 4: `record_turn` 에 인자와 필드를 더한다**

시그니처에 `child_text: str = ""` 를 마지막 키워드로 추가하고, docstring 에 한 줄, `rec` 딕셔너리에 두 줄을 넣는다.

```python
    def record_turn(self, *, stt_wait_s: float, stt_rec_s: float,
                    think_s: float, think_kind: str, tts_first_s: float,
                    tts_play_s: float = 0.0, reply: str = "",
                    child_text: str = "") -> None:
```

docstring 끝에:

```
        child_text : 아이가 한 말(STT 출력). 확장·재사용 대리지표 계산에만 쓴다.
```

`rec` 안 `"reply_len"` 다음에:

```python
            "expansion_delta": (len(_words(reply)) - len(_words(child_text))
                                if child_text else None),
            "reuse": _reuse_ratio(child_text, reply),
```

- [ ] **Step 5: `main.py` 에서 넘긴다**

`app/main.py:258` 의 호출에 인자 하나를 더한다.

```python
            metrics.record_turn(stt_wait_s=stt_wait, stt_rec_s=tr_dt,
                                think_s=think_s, think_kind=kind,
                                tts_first_s=tm.first_audio_s,
                                tts_play_s=tm.play_s, reply=reply,
                                child_text=text)
```

- [ ] **Step 6: 테스트 통과를 확인한다**

Run: `conda run -n jaeha_bot python -m pytest -q`
Expected: 207 passed (202 + 5)

- [ ] **Step 7: 커밋**

```bash
git add app/metrics.py app/main.py tests/test_metrics_expansion.py
git commit -m "feat(metrics): 확장 폭·어휘 재사용 계측 — 아이가 생기는 날 바로 재려고

child_text 하나만 넘기면 metrics 가 계산한다. GameManager 와 결합이 없어
놀이·자유대화 양쪽에 다 붙는다.

⚠️ 둘 다 대리지표다. '아이가 실제로 더 말했는가'는 재지 못한다.
   대상 아이가 아직 없으므로 지금은 배선만 하고 숫자로 판단하지 않는다."
```

---

### Task 6: 자유 대화 — 프롬프트·few-shot·recovery

**Files:**
- Modify: `configs/prompt_templates.yaml` (대화기법 블록 신설, `recovery` 교체)
- Modify: `data/finetune_seed.jsonl` (확장 few-shot 5쌍 추가)

**Interfaces:**
- Consumes: 없음(설정·데이터만).
- Produces: 없음. 단 프롬프트 길이가 늘어난다 — OpenAI 자동 캐시 최소치 1024 토큰을 이미 넘겨 둔 상태라 캐시는 유지된다.

- [ ] **Step 1: 대화기법 블록을 프롬프트 앞쪽에 넣는다**

`configs/prompt_templates.yaml` 의 `규칙:` 첫 줄(반말 규칙) **앞**에 넣는다. 금지 목록과 섞지 않는다 — 지금 규칙 20개 중 15개가 금지라, 기법이 그 사이에 묻힌다.

```yaml
  아이와 말하는 법(가장 중요):
  - 아이가 한 말을 먼저 그대로 따라 하고, 한두 낱말만 더 붙여 되돌려준다.
    (아이 "멍멍" -> "멍멍! 강아지가 멍멍 해!")
  - 아이가 쓴 낱말을 반드시 네 대답에 다시 쓴다. 네 말로 갈아치우지 않는다.
  - 아이가 틀리게 말해도 고쳐주지 말고, 바른 말로 되받아 말해준다.
    (아이 "강아지 갔어" -> "응, 강아지가 갔구나!")
  - 물어볼 때는 문장을 비워 두고 아이가 채우게 한다. ("강아지는 멍?")
  - 한 번에 한 가지만 묻는다.
  규칙:
```

- [ ] **Step 2: `recovery` 를 모델링 문구로 바꾼다**

```yaml
recovery: "그렇구나! 그럼 강아지는 뭐라고 해?"
```

지금 값 `"음, 다시 한 번 말해줄래?"` 는 **재시도를 요구**한다. 못 알아들은 건 봇 사정인데 아이에게 부담을 넘긴다. 문헌은 반대로, 못 알아들어도 모델링으로 이어가라고 한다.

- [ ] **Step 3: few-shot 5쌍을 추가한다**

`data/finetune_seed.jsonl` 끝에 다섯 줄을 붙인다. 한 줄에 하나씩(JSONL).

```
{"few_shot": true, "messages": [{"role": "user", "content": "멍멍"}, {"role": "assistant", "content": "멍멍! 강아지가 멍멍 해!"}]}
{"few_shot": true, "messages": [{"role": "user", "content": "빵빵"}, {"role": "assistant", "content": "빵빵! 자동차가 빵빵 해!"}]}
{"few_shot": true, "messages": [{"role": "user", "content": "엄마 갔어"}, {"role": "assistant", "content": "응, 엄마가 갔구나! 곧 올 거야."}]}
{"few_shot": true, "messages": [{"role": "user", "content": "이거 뭐야"}, {"role": "assistant", "content": "그거 궁금해? 같이 볼까?"}]}
{"few_shot": true, "messages": [{"role": "user", "content": "딸기 조아"}, {"role": "assistant", "content": "딸기 좋아하는구나! 딸기는 빨개!"}]}
```

⚠️ 예시 길이가 답변 길이를 끌고 간다(2026-08-03 실측: 예시를 26.2→19.5자로 줄이니 답변이 22% 짧아졌다). 위 다섯 개는 전부 20자 안팎으로 맞췄다. **길게 쓰지 말 것.**

- [ ] **Step 4: 프롬프트가 조립되는지 확인한다**

Run:
```bash
conda run -n jaeha_bot python -c "import sys;sys.path.insert(0,'.');from tools.eval_llm import build_system_prompt as b;s=b();print(len(s),'자');print('아이와 말하는 법' in s, '멍멍! 강아지가 멍멍 해!' in s)"
```
Expected: 글자 수가 2165보다 커지고 `True True`

- [ ] **Step 5: 전체 스위트를 돌린다**

Run: `conda run -n jaeha_bot python -m pytest -q`
Expected: 207 passed (프롬프트는 테스트가 내용을 검사하지 않으므로 변화 없음)

- [ ] **Step 6: 커밋**

```bash
git add configs/prompt_templates.yaml data/finetune_seed.jsonl
git commit -m "feat(prompt): 대화기법 블록 신설 — 금지 20개 중 기법이 0개였다

규칙 약 20개 중 15개가 금지였고 상호작용 기법은 하나도 없었다. 문헌에서
효과크기가 가장 큰 개입(확장·모방, recast 메타분석 0.76~0.96 SD)이 빠져 있었다.

금지 목록 사이에 끼워 넣으면 묻히므로 별도 블록으로 앞쪽에 둔다.
few-shot 5쌍도 추가 — 2.4B 시절 배운 대로 말투와 길이는 예시가 잡는다.
예시는 전부 20자 안팎으로 맞췄다(길면 답변이 따라 길어진다).

recovery 를 '음, 다시 한 번 말해줄래?' 에서 모델링 문구로 교체.
못 알아들은 건 봇 사정인데 아이에게 재시도를 요구하고 있었다."
```

---

### Task 7: 안전 회귀 확인과 젯슨 실기

**Files:**
- 코드 변경 없음. 검증만 한다.

**Interfaces:**
- Consumes: Task 6 이 바꾼 프롬프트, `tools/eval_llm.py`, `data/eval_set_safety.jsonl`.

**왜 하나:** 2026-08-07 에 안전 실패를 14/28 → **0/28** 로 만든 것이 바로 이 프롬프트다. Task 6 이 거기에 블록을 더하고 few-shot 을 바꿨다. 메모리 교훈 — *"few-shot 을 바꾸면 프롬프트가 통째로 달라져 궤적도 달라진다"*. **빼먹으면 '놀이는 좋아졌는데 안전이 조용히 나빠진' 상태를 못 알아챈다.**

- [ ] **Step 1: 안전 평가를 돌린다**

Run:
```bash
conda run -n jaeha_bot python tools/eval_llm.py --models gpt-4o-mini --eval-set data/eval_set_safety.jsonl --repeat 2
```
Expected: `unsafe 0`, `empty 0`. 하나라도 걸리면 **Task 6 의 문구를 고치고 다시 돌린다.**

- [ ] **Step 2: 답변 전문을 사람이 읽는다**

자동 판정기는 하한선이다(답변만 보므로 아이가 "이거"라고만 한 세제 케이스를 못 잡는다). `reports/eval/` 에 저장된 최신 json 을 열어 **14개 답변을 눈으로 훑는다.** 특히 위험물 문항에서 "같이 해보자"·"찾아볼까" 류가 없는지 본다.

- [ ] **Step 3: 젯슨에 배포한다**

Run:
```bash
bash push_code.sh
```

- [ ] **Step 4: 젯슨에서 놀이를 손으로 돌려 본다**

Run:
```bash
ssh -t jaeha_bot@100.65.22.17 "cd ~/jaeha_bot && source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && python -m app.education_modes"
```

`_repl()` 이 마이크·LLM 없이 상태머신만 돌린다(템플릿 모드). 확인할 것:
- `동물 소리 놀이 하자` → 완성형이 섞여 나오는가("강아지는 멍?")
- `멍멍` → `멍멍! 강아지는 멍멍 하고 울어! 그럼 …`
- `몰라` → `… 같이 해보자, …!` 가 한 번만
- `다른 놀이 하자` → 빠져나오는가

- [ ] **Step 5: 실제 대화로 확인한다**

Run:
```bash
ssh -t jaeha_bot@100.65.22.17 "cd ~/jaeha_bot && source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && python -m app.main"
```

⚠️ 마이크·스피커가 붙은 젯슨에서 직접 돌리는 편이 낫다. 확인할 것:
- 완성형("강아지는 멍?")이 TTS 로 **자연스럽게 읽히는가** — 물음표만 썼지만 억양이 어색하면 문구를 조정한다
- 로그에 `폴백` 경고가 없는가
- `[계측]` 줄과 `metrics_*.jsonl` 에 `expansion_delta`·`reuse` 가 찍히는가

- [ ] **Step 6: 결과를 기록하고 커밋한다**

`jaeha-bot-progress.md` 메모리에 다음을 남긴다: 안전 재평가 결과(0/28 유지 여부), 완성형 TTS 청취 인상, 실기에서 드러난 문제.

코드 변경이 있었으면 커밋한다. 없으면 이 태스크는 커밋 없이 끝난다.

---

## Self-Review

**스펙 커버리지**

| 스펙 절 | 태스크 |
|---|---|
| 3-1 `close` → `said` | Task 1 |
| 3-2 `_heard()` 오인식 차단 | Task 1 |
| 3-3 확장 + `require` 강제 | Task 1(동물), Task 2(따라말) |
| 3-4 완성형 `lead` | Task 3 |
| 3-5 재시도 | Task 4 |
| 3-6 계측 | Task 5 |
| 3-7 자유 대화 프롬프트·few-shot·recovery | Task 6 |
| 4 테스트 7종 | Task 1(1·2·4), Task 3(3), Task 4(5·6), 각 태스크 마지막 스텝(7) |
| 5 안 하는 것 | 어느 태스크에도 없음 ✓ |
| 6 열린 항목(효과 검증 불가, silence_duration) | 범위 밖으로 명시됨 ✓ |

**타입 일관성**
- `_heard(target, text) -> str | None` — Task 1 정의, Task 4 에서 같은 이름·같은 반환으로 사용 ✓
- `said` 인자 위치 — `_react_next_beat(subj, tgt, said, nsubj, ntgt)` 로 Task 1·2·3 전부 동일 ✓
- `_prompt(animal, sound) -> tuple[str, list[str]]` — Task 3 정의, 같은 태스크 안에서만 사용 ✓
- `_retry_beat(subj, tgt) -> dict` — Task 4 에서 추상·구현 모두 정의 ✓
- `child_text` 키워드 이름 — Task 5 의 metrics·main 양쪽 동일 ✓

**빠뜨린 것 없음 확인**
- Task 1 Step 7 에 `RepeatWordGame` 시그니처 불일치 가능성을 미리 적어 뒀다(옛 `close` 를 받는 상태라 깨질 수 있다).
- 테스트 누적 개수: 187 → 191 → 193 → 196 → 202 → 207.
