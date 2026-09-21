# 전면 API 봇 대화 어긋남 고치기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 놀이 턴의 대본 이탈을 지연 없이 바로잡고, 짧은 명령을 모델 도구로 알아듣게 하고, 말 끝 기다리기(1.2s) 대안을 아이 녹음으로 잰다.

**Architecture:** 놀이 비트에 '대신 말할 조각 문장'과 '나와도 되는 이름'을 싣고, 순수 판정기(`app/game_repair.py`)가 글자를 보고 자를 곳을 정한다. 대화 구간은 스피커의 새 기능(`mark/played/truncate`)으로 그 자리에서 자르고 캐시 문장을 잇는다. 명령은 `chat` 턴에만 도구를 열어 모델이 소리를 듣고 부르게 한다. 말 끝은 별도 측정 도구로 재고 결정은 사용자가 한다.

**Tech Stack:** Python 3.11(conda `jaeha_bot`), asyncio, websockets, numpy, pytest.

## Global Constraints

- spec: `docs/superpowers/specs/2026-09-22-realtime-turn-fixes-design.md`
- 로컬 봇(`app/main.py`)은 건드리지 않는다. `education_modes` 변경은 **필드 추가만**(로컬 봇의 `_voice` 경로 불변).
- 코드 실행: `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe`. 긴 편집은 스크래치 파일에 파이썬 스크립트로 쓰고 실행한다(bash heredoc 이 긴 스크립트를 깬다 — 09-21).
- `git add` 는 파일을 하나씩. 커밋 끝 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- 놀이 턴은 **소리를 붙잡지 않는다**(지연 0).
- 첫 문장 안의 틀린 이름 → **글자가 온 즉시** 멈춤. 둘째 문장의 틀린 이름·질문 빠짐 → **done 에서** 문장 사이 쉼에서 자름.
- 이름 찾기는 낱말 경계(조사 허용): '소'가 '소리'에 걸리면 안 된다.
- 도구는 `chat` 턴만 `tool_choice: "auto"`, 나머지 요청은 `"none"`.
- 말 끝 측정: 3~6세 153발화, 방식 6개(`server_vad` 1200/900/600, `semantic_vad` low/medium/high), 먼저 `--n 10` 으로 비용 확인.

---

### Task 1: 말 끝 측정 도구 (`tools/turn_end_probe.py`) — 먼저 돌려 둔다

**Files:**
- Create: `tools/turn_end_probe.py`
- Test: `tests/test_turn_end_probe.py`

**Interfaces:**
- Produces: `VARIANTS: dict[str, dict]`, `analyze(stops: list[float], t_end: float) -> dict`(`latency_s`, `premature`, `splits`), `summarize(rows) -> list[str]`

- [ ] **Step 1: 실패하는 시험**

```python
from tools.turn_end_probe import VARIANTS, analyze, summarize


def test_말끝_뒤에_온_첫_판정이_빨라짐이다():
    r = analyze([11.3], t_end=10.0)
    assert abs(r["latency_s"] - 1.3) < 1e-9 and r["premature"] is False and r["splits"] == 0


def test_말끝보다_먼저_온_판정은_잘림이다():
    r = analyze([9.5, 11.2], t_end=10.0)
    assert r["premature"] is True and r["splits"] == 1 and abs(r["latency_s"] - 1.2) < 1e-9


def test_0_1초_안쪽은_잘림으로_안_센다():
    assert analyze([9.95], t_end=10.0)["premature"] is False


def test_판정이_없으면_빈칸():
    r = analyze([], t_end=10.0)
    assert r["latency_s"] is None and r["premature"] is False


def test_방식은_여섯이고_답을_안_만든다():
    assert set(VARIANTS) == {"server1200", "server900", "server600",
                             "semantic_low", "semantic_medium", "semantic_high"}
    assert all(v["create_response"] is False for v in VARIANTS.values())


def test_요약은_방식마다_한_줄():
    rows = [{"variant": "server1200", "latency_s": 1.3, "premature": False},
            {"variant": "server1200", "latency_s": 1.1, "premature": True},
            {"variant": "server900", "latency_s": None, "premature": False}]
    out = "\n".join(summarize(rows))
    assert "server1200" in out and "50%" in out and "server900" in out
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_turn_end_probe.py -q` / Expected: FAIL (`No module named 'tools.turn_end_probe'`)

- [ ] **Step 3: 구현** — `tools/turn_end_probe.py`

```python
"""말 끝 판정 방식 비교 — 아이 녹음(3~6세 153발화)을 실서버에 실시간으로 흘린다. (2026-09-22)

spec: docs/superpowers/specs/2026-09-22-realtime-turn-fixes-design.md §5
재는 것 (발화마다):
  빨라짐 latency_s : 실제 말 끝을 보낸 시각 → 그 뒤 첫 speech_stopped 도착
  잘림   premature : 실제 말 끝보다 0.1s 넘게 **먼저** speech_stopped 가 옴 = 말 중간에 끝났다고 봄
답은 만들지 않는다(create_response false, 받아 적기 끔) — 입력 소리만 보낸다.

실행:
  python -m tools.turn_end_probe --n 10            # 먼저 비용·동작 확인
  python -m tools.turn_end_probe --json reports/turn/realtime_turn_end.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
SR = 24000
CHUNK = 960                         # 40ms
LEAD_S, TAIL_S = 0.5, 6.0           # 앞 무음 / 뒤 무음(semantic low 는 오래 기다린다)
EARLY_S = 0.1


def _vad(kind: str, **kw) -> dict:
    return {"type": kind, "create_response": False, "interrupt_response": False, **kw}


VARIANTS = {
    "server1200": _vad("server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=1200),
    "server900": _vad("server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=900),
    "server600": _vad("server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=600),
    "semantic_low": _vad("semantic_vad", eagerness="low"),
    "semantic_medium": _vad("semantic_vad", eagerness="medium"),
    "semantic_high": _vad("semantic_vad", eagerness="high"),
}


def analyze(stops: list[float], t_end: float) -> dict:
    early = [s for s in stops if s < t_end - EARLY_S]
    after = [s for s in stops if s >= t_end - EARLY_S]
    return {"latency_s": (after[0] - t_end) if after else None,
            "premature": bool(early), "splits": len(early)}


def summarize(rows: list[dict]) -> list[str]:
    out = ["| 방식 | n | 빨라짐 중앙 | p90 | 잘림 |", "|---|---|---|---|---|"]
    for name in VARIANTS:
        rs = [r for r in rows if r.get("variant") == name]
        if not rs:
            continue
        lat = sorted(r["latency_s"] for r in rs if r.get("latency_s") is not None)
        pre = sum(1 for r in rs if r.get("premature"))
        med = f"{statistics.median(lat):.2f}s" if lat else "-"
        p90 = f"{lat[int(0.9 * (len(lat) - 1))]:.2f}s" if lat else "-"
        out.append(f"| {name} | {len(rs)} | {med} | {p90} | {pre}/{len(rs)} = {100 * pre / len(rs):.0f}% |")
    return out


def _items(n: int) -> list[dict]:
    idx = json.loads((BASE / "reports/stt/young/audio_index.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in
            (BASE / "reports/stt/young/manifest.jsonl").read_text(encoding="utf-8").splitlines() if x]
    items = [dict(r, path=idx[r["file"]]) for r in rows if r["file"] in idx and os.path.exists(idx[r["file"]])]
    return items[:n] if n else items


async def _run_variant(name: str, items: list[dict], model: str, sink: Path | None) -> list[dict]:
    from tools.realtime_probe import connect, load_pcm
    cfg = {"type": "session.update", "session": {
        "type": "realtime", "output_modalities": ["text"],
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": SR},
                            "turn_detection": VARIANTS[name]}}}}
    ws = await connect(model, cfg)
    stops: list[float] = []

    async def reader():
        async for raw in ws:
            ev = json.loads(raw)
            if ev.get("type") == "input_audio_buffer.speech_stopped":
                stops.append(time.monotonic())
            elif ev.get("type") == "error":
                print(f"   [{name}] 오류 {ev.get('error')}", flush=True)

    rt = asyncio.create_task(reader())
    rows = []
    silence = np.zeros(CHUNK, dtype="<i2").tobytes()

    async def send(pcm: bytes) -> None:
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(pcm).decode()}))
        await asyncio.sleep(len(pcm) / 2 / SR)

    try:
        for i, it in enumerate(items, 1):
            for _ in range(int(LEAD_S * SR / CHUNK)):
                await send(silence)
            stops.clear()
            pcm = load_pcm(Path(it["path"]), "speech")
            for k in range(0, len(pcm), CHUNK * 2):
                await send(pcm[k:k + CHUNK * 2])
            t_end = time.monotonic()
            for _ in range(int(TAIL_S * SR / CHUNK)):
                await send(silence)
            r = {"variant": name, "file": it["file"], "age": it.get("age"),
                 "sec": round(len(pcm) / 2 / SR, 2), **analyze(list(stops), t_end)}
            rows.append(r)
            if sink:
                with sink.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"   [{name}] {i}/{len(items)}", flush=True)
    finally:
        rt.cancel()
        await ws.close()
    return rows


async def _main(a) -> list[dict]:
    items = _items(a.n)
    names = a.only.split(",") if a.only else list(VARIANTS)
    sink = Path(a.json).with_suffix(".jsonl") if a.json else None
    print(f"발화 {len(items)}개 × 방식 {len(names)}개 — 동시에 흘린다", flush=True)
    res = await asyncio.gather(*(_run_variant(n, items, a.model, sink) for n in names))
    return [r for rs in res for r in rs]


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    p = argparse.ArgumentParser(description="말 끝 판정 방식 비교(아이 녹음, 실서버)")
    p.add_argument("--n", type=int, default=0, help="0 이면 전부(153)")
    p.add_argument("--only", help="쉼표로 방식 고르기")
    p.add_argument("--model", default="gpt-realtime-mini")
    p.add_argument("--json")
    a = p.parse_args()
    rows = asyncio.run(_main(a))
    print("\n".join(summarize(rows)))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_turn_end_probe.py -q` / Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add tools/turn_end_probe.py tests/test_turn_end_probe.py
git commit -m "feat(tools): 말 끝 판정 방식 비교 도구 — 아이 녹음을 실서버에 흘려 빨라짐·잘림을 잰다"
```

- [ ] **Step 6: 10개로 확인 → 전체를 뒤에서 돌린다**

Run: `PYTHONPATH=. .../python.exe -m tools.turn_end_probe --n 10`
Expected: 방식 6줄 표. 오류 없음. 사용량 페이지에서 비용 확인(입력 소리만이라 작아야 한다).
그다음 백그라운드로: `PYTHONPATH=. .../python.exe -m tools.turn_end_probe --json reports/turn/realtime_turn_end.json` (약 25분)

---

### Task 2: 놀이 비트에 조각 문장·허용 이름

**Files:**
- Modify: `app/education_modes.py`
- Test: `tests/test_education_repair_fields.py`

**Interfaces:**
- Produces: 비트 dict 새 키 — `fix_react: str`, `fix_next: str`, `next_need: list[str]`, `allow: list[str]`, `names: list[str]`; 함수 `education_modes.all_fix_phrases() -> list[str]`

- [ ] **Step 1: 실패하는 시험** — `tests/test_education_repair_fields.py`

```python
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
    assert any(p.startswith("그럼 고양이는") for p in ps)
    assert "이번엔 사과!" in ps and "더 할래?" in ps
    assert all("{" not in p for p in ps) and len(ps) == len(set(ps))
    assert len(ps) < 80
```

⚠️ `"강아지는 멍멍 하고 울어!"` 는 `_j("강아지")` 가 "강아지는" 을 만든다는 가정이다. 파일의 `_j` 를 확인하고 다르면 시험 문자열을 맞춘다. 카드 순서는 `ANIMAL_ITEMS`·`REPEAT_ITEMS` 에서 확인한다.

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_education_repair_fields.py -q` / Expected: FAIL (ImportError `all_fix_phrases`)

- [ ] **Step 3: 구현** — `app/education_modes.py`

`_beat` 를 바꾼다(기존 호출은 그대로 동작):
```python
def _beat(fallback: str, instruction: str | None = None, require=(), *, fix_react: str = "",
          fix_next: str = "", next_need=(), allow=()) -> dict:
    """fallback=템플릿(안전망), instruction=LLM 렌더 지시, require=LLM 출력에 꼭 있어야 할 토큰.

    fix_react/fix_next/next_need/allow (2026-09-22, 전면 API): 모델이 대본을 벗어나면
    봇이 그 자리에서 잘라 이어 붙일 **정해진 문장**(미리 녹음)과, 이 턴에 나와도 되는
    놀이 이름. 로컬 경로(_voice)는 쓰지 않는다.
    """
    return {"fallback": fallback, "instruction": instruction, "require": list(require),
            "fix_react": fix_react, "fix_next": fix_next, "next_need": list(next_need),
            "allow": list(allow)}
```
`Game.start` 와 `Game.step` 이 돌려주는 비트에 `names` 를 붙인다 — `start` 끝과 `step` 의 모든 `return` 대신, 두 메서드를 감싸는 공개 메서드를 쓴다. 가장 작은 변경: `GameManager.maybe_start_beat` / `handle_beat` 에서 비트를 돌려주기 직전에
```python
        if beat is not None:
            beat["names"] = [s for s, _ in type(game).ITEMS]
```
(`handle_beat` 는 `self.active` 가 `None` 이 되기 전에 `game = self.active` 로 잡아 둔다.)

`Game._bye_beat`:
```python
        return _beat("재밌었어! 또 놀자!",
                     "아이랑 놀이를 즐겁게 마무리하는 인사를 밝은 반말 한 문장으로 해.", [],
                     fix_react="재밌었어!", fix_next="또 놀자!")
```
`AnimalSoundGame`:
```python
    def _intro_beat(self, animal, sound):
        q, req = self._prompt(animal, sound)
        return _beat(f"좋아, 동물 소리 놀이 하자! {q}", <기존 지시문 그대로>, req,
                     fix_react="좋아, 동물 소리 놀이 하자!", fix_next=q, next_need=req,
                     allow=[animal])

    def _ask_beat(self, animal, sound):
        q, req = self._prompt(animal, sound)
        return _beat(f"좋아! {q}", <기존 지시문>, req,
                     fix_react="좋아!", fix_next=q, next_need=req, allow=[animal])
```
`_react_next_beat` 끝의 `return _beat(fb, ins, req)` 를:
```python
        return _beat(fb, ins, req, fix_react=f"{_j(animal)} {sound} 하고 울어!",
                     fix_next=f"그럼 {nq}", next_need=nreq, allow=[animal, nanimal])
```
`_react_checkpoint_beat` 의 return 에 `fix_react=f"{_j(animal)} {sound} 하고 울어!", fix_next="더 할래?", next_need=["더"], allow=[animal]`.
`_retry_beat` 의 return 에 `fix_react=f"{_j(animal)} {sound}!", fix_next=f"같이 해보자, {sound}!", next_need=["같이"], allow=[animal]`.

`RepeatWordGame`:
- `_intro_beat`: `fix_react="따라 말하기 놀이 하자!", fix_next=f"따라 해봐, {word}!", next_need=[word], allow=[word]`
- `_ask_beat`: `fix_react="좋아!", fix_next=f"따라 해봐, {word}!", next_need=[word], allow=[word]`
- `_react_next_beat`: `fix_react=f"{word}, 잘했어!", fix_next=f"이번엔 {nword}!", next_need=[nword], allow=[word, nword]`
- `_react_checkpoint_beat`: `fix_react=f"{word}, 잘했어!", fix_next="더 할래?", next_need=["더"], allow=[word]`
- `_retry_beat`: `fix_react=f"{word}!", fix_next=f"같이 해보자, {word}!", next_need=["같이"], allow=[word]`

파일 끝(`_repl` 앞)에:
```python
def all_fix_phrases() -> list[str]:
    """놀이 조각 문장 전부 — 전면 API 봇이 기동 때 미리 녹음한다(2026-09-22)."""
    out: list[str] = ["재밌었어!", "또 놀자!", "좋아, 동물 소리 놀이 하자!", "좋아!",
                      "따라 말하기 놀이 하자!", "더 할래?"]
    a = AnimalSoundGame.__new__(AnimalSoundGame)
    for animal, sound in AnimalSoundGame.ITEMS:
        q, _ = a._prompt(animal, sound)
        out += [q, f"그럼 {q}", f"{_j(animal)} {sound} 하고 울어!", f"{_j(animal)} {sound}!",
                f"같이 해보자, {sound}!"]
    for word, _ in RepeatWordGame.ITEMS:
        out += [f"따라 해봐, {word}!", f"{word}, 잘했어!", f"이번엔 {word}!", f"{word}!",
                f"같이 해보자, {word}!"]
    return list(dict.fromkeys(out))
```
⚠️ `len(ps) < 80` 이 깨지면(카드가 늘었으면) 시험 상한을 카드 수에 맞게 고친다 — 캐시 생성 시간(문장당 약 4초) 때문에 상한을 둔 것이다.

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_education_repair_fields.py tests/test_education_modes.py -q` / Expected: PASS(기존 시험 포함)

- [ ] **Step 5: 커밋**

```bash
git add app/education_modes.py tests/test_education_repair_fields.py
git commit -m "feat(game): 놀이 비트에 조각 문장·허용 이름 — 전면 API 가 대본 이탈을 그 자리에서 바로잡을 재료"
```

---

### Task 3: 이탈 판정 · 쉼 찾기 (`app/game_repair.py`)

**Files:**
- Create: `app/game_repair.py`
- Test: `tests/test_game_repair.py`

**Interfaces:**
- Consumes: 비트 dict(`fix_react`, `fix_next`, `next_need`, `allow`, `names`)
- Produces: `Repair(cut_char: int, lines: list[str], reason: str, in_first: bool)`; `find_name(text, name) -> int`; `first_sentence_end(text) -> int`; `check_stream(said, beat) -> Repair | None`(첫 문장 안의 틀린 이름만); `check_done(said, beat) -> Repair | None`; `quiet_cut(audio: np.ndarray, near: int, sr: int, *, before_s=0.5, after_s=0.5) -> int`

- [ ] **Step 1: 실패하는 시험** — `tests/test_game_repair.py`

```python
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
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_game_repair.py -q` / Expected: FAIL (모듈 없음)

- [ ] **Step 3: 구현** — `app/game_repair.py`

```python
"""놀이 턴 대본 이탈 — 어디서 자르고 무엇을 이어 붙일지(2026-09-22). 소리·소켓을 모른다.

spec: docs/superpowers/specs/2026-09-22-realtime-turn-fixes-design.md §3
실측(09-21 실서버 놀이 8턴): 동물 이름 글자가 스피커보다 중앙 0.98s·최소 0.14s 앞서 온다.
  첫 문장 안의 틀린 이름 → 글자가 온 즉시(check_stream)
  둘째 문장의 틀린 이름·질문 빠짐 → done 에서(check_done) — 둘째 문장은 1~2s 뒤에 들린다
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_PARTICLE = "은는이가을를도랑야와과의에한"
_END = re.compile(r"[.!?](\s|$)")


@dataclass
class Repair:
    cut_char: int
    lines: list[str] = field(default_factory=list)
    reason: str = ""
    in_first: bool = False


def find_name(text: str, name: str) -> int:
    """낱말 경계로 찾는다(조사 허용). '소'가 '소리'에 걸리지 않게."""
    m = re.search(rf"(?<![가-힣]){re.escape(name)}(?=[{_PARTICLE}]?(?![가-힣]))", text or "")
    return m.start() if m else -1


def first_sentence_end(text: str) -> int:
    m = _END.search(text or "")
    return m.start() + 1 if m else -1


def _wrong_name(said: str, beat: dict) -> int:
    allow = set(beat.get("allow") or [])
    hits = [p for n in beat.get("names") or [] if n not in allow and (p := find_name(said, n)) >= 0]
    return min(hits) if hits else -1


def _lines(beat: dict, in_first: bool) -> list[str]:
    lines = [beat.get("fix_react", ""), beat.get("fix_next", "")] if in_first else [beat.get("fix_next", "")]
    return [x for x in lines if x]


def check_stream(said: str, beat: dict) -> Repair | None:
    """글자가 들어오는 도중 — 첫 문장 안의 틀린 이름만 본다(여유가 0.14s 뿐이라 기다릴 수 없다)."""
    pos = _wrong_name(said, beat)
    if pos < 0:
        return None
    end = first_sentence_end(said)
    if end != -1 and pos >= end:
        return None                       # 둘째 문장 — done 에서 정확히 자른다
    lines = _lines(beat, True)
    return Repair(pos, lines, "wrong_name", True) if lines else None


def check_done(said: str, beat: dict) -> Repair | None:
    """글자가 다 왔다 — 틀린 이름(어디든) 또는 다음 질문 빠짐."""
    end = first_sentence_end(said)
    pos = _wrong_name(said, beat)
    if pos >= 0:
        in_first = end == -1 or pos < end
        lines = _lines(beat, in_first)
        return Repair(pos if in_first else end, lines, "wrong_name", in_first) if lines else None
    need = beat.get("next_need") or []
    if need and not all(tok in said for tok in need):
        lines = _lines(beat, False)
        return Repair(end if end != -1 else len(said), lines, "missing_next") if lines else None
    return None


def quiet_cut(audio: np.ndarray, near: int, sr: int, *, before_s: float = 0.5,
              after_s: float = 0.5) -> int:
    """near 근처(앞 before_s·뒤 after_s)에서 가장 조용한 30ms 의 가운데 샘플."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    win = max(1, int(0.03 * sr))
    lo = max(0, near - int(before_s * sr))
    hi = min(a.size, near + int(after_s * sr))
    if hi - lo < win:
        return max(0, min(near, a.size))
    seg = a[lo:hi]
    n = (seg.size - win) // win + 1
    energy = [float(np.mean(seg[i * win:(i + 1) * win] ** 2)) for i in range(n)]
    best = int(np.argmin(energy))
    return lo + best * win + win // 2
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_game_repair.py -q` / Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/game_repair.py tests/test_game_repair.py
git commit -m "feat(game): 대본 이탈 판정 — 첫 문장 틀린 이름은 도중에, 둘째 문장·질문 빠짐은 끝에서, 쉼에서 자른다"
```

---

### Task 4: 스피커 — 이 턴 소리 세기·뒤를 버리기

**Files:**
- Modify: `app/realtime_audio.py` (`StreamSpeaker`)
- Test: `tests/test_realtime_audio.py`

**Interfaces:**
- Produces: `StreamSpeaker.mark() -> None`, `.played() -> int`(원본 24k 샘플 수, mark 이후 재생), `.truncate(keep: int) -> None`(mark 이후 밀어 넣은 원본 샘플 중 keep 뒤를 버린다)

- [ ] **Step 1: 실패하는 시험** — 기존 파일에 가짜 스트림으로 `StreamSpeaker` 를 만드는 방식이 있으면 그걸 따른다. 없으면:

```python
def _speaker(monkeypatch, rate=24000):
    import app.realtime_audio as ra

    class FakeStream:
        def __init__(self, **kw):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass

    import sounddevice as sd
    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    monkeypatch.setattr(ra, "resolve_rate", lambda kind, want: rate)
    return ra.StreamSpeaker(24000)


def _pull(sp, frames):
    out = np.zeros((frames, 1), dtype=np.float32)
    sp._cb(out, frames, None, None)
    return out[:, 0]


def test_표시_뒤로_재생한_양을_센다(monkeypatch):
    sp = _speaker(monkeypatch)
    sp.push(np.ones(100, np.float32))
    sp.mark()
    sp.push(np.ones(300, np.float32))
    _pull(sp, 250)
    assert sp.played() == 150


def test_표시_뒤_keep_넘는_소리는_버린다(monkeypatch):
    sp = _speaker(monkeypatch)
    sp.mark()
    sp.push(np.full(200, 0.5, np.float32))
    sp.push(np.full(200, 0.5, np.float32))
    _pull(sp, 50)
    sp.truncate(120)
    got = _pull(sp, 400)
    assert np.count_nonzero(got) == 70
    sp.push(np.full(10, 0.9, np.float32))            # 자른 뒤 이어 붙인 문장은 나간다
    assert np.count_nonzero(_pull(sp, 10)) == 10


def test_장치_레이트가_달라도_원본_샘플로_센다(monkeypatch):
    sp = _speaker(monkeypatch, rate=16000)
    sp.mark()
    sp.push(np.ones(2400, np.float32))               # 24k 0.1s → 16k 1600
    _pull(sp, 800)
    assert abs(sp.played() - 1200) <= 2
```
(파일 위에 `import numpy as np` 가 없으면 추가.)

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_realtime_audio.py -q` / Expected: FAIL (`mark` 없음)

- [ ] **Step 3: 구현** — `StreamSpeaker`

`__init__` 에 `import threading` 을 쓰고:
```python
        self._lock = threading.Lock()
        self._marked = 0          # mark 이후 밀어 넣은 **장치** 샘플
        self._played = 0          # mark 이후 재생한 장치 샘플
```
`_cb` 전체를 `with self._lock:` 로 감싸고, `block` 을 만든 직후 `self._played += min(frames, sum(g.size for g in got))`.
`push` 의 대기열 추가를 `with self._lock: self._q.append(samples); self._marked += samples.size`.
`clear` 를 `with self._lock:` 로 감싼다.
새 메서드:
```python
    def _to_dev(self, n: int) -> int:
        return int(round(n * self.rate / self.src_rate))

    def mark(self) -> None:
        """이 턴 소리의 시작점 — 이후 played()/truncate() 의 기준(2026-09-22 놀이 바로잡기)."""
        with self._lock:
            self._marked = self._played = 0

    def played(self) -> int:
        """mark 이후 실제로 재생한 양(원본 레이트 샘플)."""
        with self._lock:
            return int(round(self._played * self.src_rate / self.rate))

    def truncate(self, keep: int) -> None:
        """mark 이후 밀어 넣은 소리 중 keep(원본 샘플) 뒤를 버린다. 이미 나간 건 못 되돌린다."""
        with self._lock:
            drop = self._marked - max(self._to_dev(keep), self._played)
            while drop > 0 and self._q:
                last = self._q[-1]
                if last.size <= drop:
                    drop -= last.size
                    self._q.pop()
                else:
                    self._q[-1] = last[:last.size - drop]
                    drop = 0
            if drop > 0 and self._left.size:
                self._left = self._left[:max(0, self._left.size - drop)]
            self._marked = max(self._to_dev(keep), self._played)
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_realtime_audio.py tests/test_realtime_live.py -q` / Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/realtime_audio.py tests/test_realtime_audio.py
git commit -m "feat(audio): 스피커가 이 턴 재생량을 세고 뒤를 버린다 — 놀이 바로잡기용"
```

---

### Task 5: 대화 구간에 놀이 바로잡기 연결

**Files:**
- Modify: `app/realtime_conversation.py`, `app/main_realtime.py`, `app/metrics.py`
- Test: `tests/test_realtime_conversation.py`, `tests/test_metrics_realtime.py`

**Interfaces:**
- Consumes: `game_repair.check_stream/check_done/quiet_cut/Repair`, `StreamSpeaker.mark/played/truncate`, `education_modes.all_fix_phrases`
- Produces: `Route.beat`(놀이 턴의 비트 dict), `record_realtime_turn(..., game_fixed: str | None = None, game_leaked: bool = False)`

- [ ] **Step 1: 실패하는 시험**

`tests/test_realtime_conversation.py` 의 `FakeSpeaker` 에 추가:
```python
    def mark(self):
        self.marks = getattr(self, "marks", 0) + 1

    def played(self):
        return getattr(self, "played_n", 0)

    def truncate(self, keep):
        self.truncated = keep
```
파일 끝에:
```python
def _game_conv(s, sp, history=None):
    return _guard_conv(s, sp, FakeGuardian(), history=history)


def _start_game(c):
    return c._on_transcript("동물 소리 놀이 하자", 0.0)


def test_놀이_첫_문장에_틀린_이름이면_바로_자르고_조각을_잇는다():
    async def go():
        s, sp, history = FakeSession(), FakeSpeaker(), []
        c = _game_conv(s, sp, history)
        await _start_game(c)
        beat = c._pending.route.beat
        wrong = next(n for n in beat["names"] if n not in beat["allow"])
        await c._on_event(_audio_delta())
        await c._on_event(_text(f"{wrong}는 "))
        await c._on_event(_done())
        return s, sp, c, beat, history

    s, sp, c, beat, history = asyncio.run(go())
    assert {"type": "response.cancel"} in s.sent and hasattr(sp, "truncated")
    assert beat["fix_react"] in c.cache.asked and beat["fix_next"] in c.cache.asked
    assert history[-1][1].endswith(beat["fix_next"])


def test_놀이_질문이_빠지면_끝에서_질문만_잇는다():
    async def go():
        s, sp = FakeSession(), FakeSpeaker()
        c = _game_conv(s, sp)
        await _start_game(c)
        beat = c._pending.route.beat
        await c._on_event(_audio_delta())
        await c._on_event(_text("좋아, 동물 소리 놀이 하자! 이제 따라 해볼까?"))
        await c._on_event(_done())
        return s, sp, c, beat

    s, sp, c, beat = asyncio.run(go())
    assert {"type": "response.cancel"} not in s.sent
    assert beat["fix_next"] in c.cache.asked and beat["fix_react"] not in c.cache.asked
    assert hasattr(sp, "truncated")


def test_놀이가_제대로면_아무것도_안_한다():
    async def go():
        s, sp = FakeSession(), FakeSpeaker()
        c = _game_conv(s, sp)
        await _start_game(c)
        beat = c._pending.route.beat
        await c._on_event(_audio_delta())
        await c._on_event(_text(f"좋아, 동물 소리 놀이 하자! {beat['fix_next']}"))
        await c._on_event(_done())
        return sp, c, beat

    sp, c, beat = asyncio.run(go())
    assert not hasattr(sp, "truncated") and beat["fix_next"] not in c.cache.asked
```
`tests/test_metrics_realtime.py` 끝에:
```python
def test_놀이_바로잡기를_남긴다(tmp_path):
    m = MetricsLogger(enabled=True, tag="t", log_dir=str(tmp_path))
    try:
        m.record_realtime_turn(kind="game", perceived_s=1.0, transcribe_s=0.5, respond_first_s=0.5,
                               filler=False, reply="r", child_text="c", cost_usd=0.001,
                               cached_tokens=0, safety=[], game_missing=[],
                               game_fixed="wrong_name", game_leaked=True)
    finally:
        m._sampler.stop()
    rec = json.loads(m.path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["game_fixed"] == "wrong_name" and rec["game_leaked"] is True
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_realtime_conversation.py tests/test_metrics_realtime.py -q` / Expected: FAIL (`Route` 에 `beat` 없음 등)

- [ ] **Step 3: 구현**

`app/realtime_turn.py` — `Route` 에 `beat: dict | None = None` 필드, `_game()` 두 return 에 `beat=beat` 를 넘긴다.

`app/metrics.py` — `record_realtime_turn` 시그니처 끝에 `game_fixed: str | None = None, game_leaked: bool = False`, rec 에 `"game_fixed": game_fixed, "game_leaked": bool(game_leaked),`.

`app/realtime_conversation.py`:
- import `from .game_repair import check_done, check_stream, quiet_cut` 와 `from .realtime_audio import SR as _SR`(이미 `realtime_audio` 에서 가져오는 줄에 `SR` 추가).
- `_Pending` 에 `audio: list = field(default_factory=list)`(이 턴 받은 소리 복사), `fixed: str | None = None`, `fix_leaked: bool = False`, `fix_said: str = ""`(나간 말 기록용).
- 요청을 보낼 때(`_on_transcript` 끝, `await self._send(msg)` 앞) 놀이 턴이면: `if r.beat is not None: self.speaker.mark()`.
- audio 분기의 `self.speaker.push(samples)` 앞에 `p.audio.append(samples)` (붙잡는 턴이면 held 로 가므로 거기서도 `p.audio.append(samples)`).
- `p.fixed` 가 있으면 audio·text 분기에서 **버린다**(blocked 와 같은 자리: `if p.blocked or p.fixed: return`).
- text 분기 끝(`await self._poll_guard(p, now)` 다음):
```python
            if p.route.beat is not None and not p.blocked and not p.fixed:
                rep = check_stream(p.said, p.route.beat)
                if rep is not None:
                    await self._repair(p, rep, final=False)
```
- `_on_done` 에서 가드 처리 뒤·정정 처리 앞:
```python
        if p.route.beat is not None and not p.blocked and not p.fixed:
            rep = check_done(reply, p.route.beat)
            if rep is not None:
                await self._repair(p, rep, final=True)
```
- 새 메서드:
```python
    async def _repair(self, p: _Pending, rep, *, final: bool) -> None:
        """놀이 대본 이탈 — rep.cut_char 앞에서 자르고 미리 녹음한 조각 문장을 잇는다."""
        audio = np.concatenate(p.audio) if p.audio else np.zeros(0, np.float32)
        text = p.said
        if final and len(text):
            near = int(audio.size * rep.cut_char / len(text))
            cut = quiet_cut(audio, near, _SR, before_s=0.5, after_s=0.5)
        else:
            cut = self.speaker.played()   # 도중: 아직 안 나간 건 전부 버린다(틀린 이름은 아직 안 들렸다)
        played = self.speaker.played()
        p.fix_leaked = cut < played
        self.speaker.truncate(max(cut, played))
        if not final:
            await self._send(cancel())
        for line in rep.lines:
            a = self.cache.get(line)
            if a is None:
                log.warning("[놀이] 조각 문장 캐시가 없다: %s", line)
                continue
            self.speaker.push(a)
        p.fixed = rep.reason
        p.fix_said = (text[:rep.cut_char].rstrip() + " " + " ".join(rep.lines)).strip()
        log.warning("[놀이] 대본 이탈(%s) — %d자에서 자르고 잇는다: %s ← %s",
                    rep.reason, rep.cut_char, " ".join(rep.lines), text)
```
- `_on_done` 의 history 부분에서 `said` 를 정할 때 `p.fixed` 면 `p.fix_said` 를 쓴다:
```python
            said = PHRASES["safe"] if p.blocked else (
                p.fix_said if p.fixed else (f"{reply} {PHRASES['cant']}" if corrected else reply))
```
- `record_realtime_turn(...)` 에 `game_fixed=p.fixed, game_leaked=p.fix_leaked`.
- 도중 바로잡기(`final=False`)는 취소한 뒤 done 이 안 올 수 있다 — `_tick` 의 막기 대비 줄을 `if p is not None and (p.blocked or p.fixed) and now - (p.blocked_at or p.requested_at) >= 2.0:` 로 넓히고, `_repair(final=False)` 에서 `p.blocked_at = self.clock()` 도 둔다.
- 정정(`cant`)은 `p.fixed` 면 붙이지 않는다(`if v is not None and v.fabricated and not p.blocked and not p.fixed:`).

`app/main_realtime.py` — 캐시 준비 줄:
```python
    from .education_modes import all_fix_phrases
    made = cache.ensure(list(PHRASES.values()) + filler_phrases + all_fix_phrases(),
                        lambda p: asyncio.run(synthesize(cfg, api_key, p)))
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/ -q` / Expected: 전부 PASS

- [ ] **Step 5: 커밋**

```bash
git add app/realtime_turn.py app/realtime_conversation.py app/metrics.py app/main_realtime.py tests/test_realtime_conversation.py tests/test_metrics_realtime.py
git commit -m "feat(game): 놀이 대본 이탈을 지연 없이 바로잡는다 — 틀린 이름 앞에서 자르고 미리 녹음한 문장으로 잇기"
```

---

### Task 6: 짧은 명령 — 받아 적기 힌트 + 모델 도구

**Files:**
- Modify: `app/song_names.py`, `app/realtime_protocol.py`, `app/realtime_session.py`, `app/realtime_conversation.py`, `app/main_realtime.py`, `app/metrics.py`
- Test: `tests/test_song_names.py`, `tests/test_realtime_protocol.py`, `tests/test_realtime_conversation.py`

**Interfaces:**
- Produces: `realtime_protocol.command_tools(music_on: bool) -> list[dict]`; `session_update(cfg, instructions, tools=None)`; `respond(instructions=None, *, tools: bool = False)` — `tools=False` 면 `tool_choice: "none"`; `say_exactly` 는 늘 `tool_choice: "none"`; `tool_output(call_id: str) -> dict`; `realtime_protocol.function_calls(response: dict) -> list[tuple[str, dict, str]]`; `RealtimeSession(cfg, key, connect=None, tools=None)`; `record_realtime_turn(..., tool: str | None = None)`

- [ ] **Step 1: 실패하는 시험**

`tests/test_song_names.py`:
```python
def test_받아쓰기_힌트에_명령_낱말도_있다():
    p = transcribe_prompt()
    for w in ("잘 자", "바이바이", "동물 소리 놀이", "따라 말하기 놀이", "노래 틀어줘"):
        assert w in p
```
`tests/test_realtime_protocol.py`:
```python
def test_명령_도구는_노래가_꺼지면_play_song_을_뺀다():
    from app.realtime_protocol import command_tools
    names = lambda ts: {t["name"] for t in ts}
    assert names(command_tools(True)) == {"go_to_sleep", "play_song", "start_game"}
    assert names(command_tools(False)) == {"go_to_sleep", "start_game"}


def test_도구는_세션에_싣고_요청마다_쓸지_정한다():
    from app.realtime_protocol import command_tools
    s = session_update(RealtimeConfig(), "지시", tools=command_tools(False))["session"]
    assert s["tools"] and s["tool_choice"] == "auto"
    assert respond(tools=True) == {"type": "response.create", "response": {"tool_choice": "auto"}}
    assert respond()["response"]["tool_choice"] == "none"
    assert respond("지시")["response"]["tool_choice"] == "none"
    assert say_exactly("안녕")["response"]["tool_choice"] == "none"


def test_응답에서_도구_호출을_꺼낸다():
    from app.realtime_protocol import function_calls, tool_output
    resp = {"output": [{"type": "function_call", "name": "play_song",
                        "arguments": "{\"title\": \"상어가족\"}", "call_id": "c1"},
                       {"type": "message", "content": []}]}
    assert function_calls(resp) == [("play_song", {"title": "상어가족"}, "c1")]
    assert function_calls({"output": [{"type": "function_call", "name": "x", "arguments": "{깨짐",
                                       "call_id": "c2"}]}) == [("x", {}, "c2")]
    o = tool_output("c1")
    assert o["item"] == {"type": "function_call_output", "call_id": "c1", "output": "ok"}
```
⚠️ 기존 시험 `test_자유대화는_답을_요청하고_이력에_남긴다` 는 `s.requests() == [{"type": "response.create"}]` 를 본다 — 이제 chat 턴 요청은 `{"type": "response.create", "response": {"tool_choice": "auto"}}` 다. 그 기대값을 이렇게 고친다. `test_끊기면_말없이_다시_붙고_답하던_말을_다시_묻는다` 의 `s2.requests()` 기대값도 같이 고친다.

`tests/test_realtime_conversation.py` 끝에:
```python
def _fc(name, args="{}"):
    return {"type": "response.done", "response": {"output": [
        {"type": "function_call", "name": name, "arguments": args, "call_id": "c9"}]}}


def test_모델이_잠들기를_부르면_잔다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache)
        asyncio.create_task(_feed(s, [_transcript("탈자"), _fc("go_to_sleep")]))
        return await c.run(greet=False), s, cache

    reason, s, cache = asyncio.run(go())
    assert reason == "sleep" and PHRASES["sleep"] in cache.asked
    assert any(m.get("item", {}).get("type") == "function_call_output" for m in s.sent)


def test_모델이_놀이를_부르면_놀이를_시작한다():
    async def go():
        s = FakeSession()
        c = _conv(s, sleep_timeout=0.3)
        asyncio.create_task(_feed(s, [_transcript("움월 소리놀이 하자"),
                                      _fc("start_game", "{\"kind\": \"animal\"}")]))
        await c.run(greet=False)
        return s

    reqs = asyncio.run(go()).requests()
    assert len(reqs) == 2 and "상황:" in reqs[1]["response"]["instructions"]


def test_모델이_노래를_부르면_노래_경로():
    played = []

    class Music:
        def handle(self, text):
            if "틀어줘" in text:
                return MusicReply(f"{text.split(' 틀어줘')[0]} 틀어 줄게!",
                                  action=lambda: played.append(text) or True, standby=True)
            return None

    async def go():
        s = FakeSession()
        c = _conv(s, music=Music())
        asyncio.create_task(_feed(s, [_transcript("비니닝"),
                                      _fc("play_song", "{\"title\": \"티니핑\"}"),
                                      _audio_delta(), {"type": "response.done", "response": {}}]))
        return await c.run(greet=False)

    assert asyncio.run(go()) == "music" and played == ["티니핑 틀어줘"]
```
⚠️ 이 시험의 `Music.handle` 은 `route()` 가 먼저 부를 때 `"비니닝"` 에는 None 을 돌려야 chat 으로 간다 — 위 구현이 그렇다.

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_song_names.py tests/test_realtime_protocol.py tests/test_realtime_conversation.py -q` / Expected: FAIL

- [ ] **Step 3: 구현**

`app/song_names.py`:
```python
COMMANDS = ("잘 자", "바이바이", "그만할래", "쉬고 있어", "동물 소리 놀이", "따라 말하기 놀이",
            "노래 틀어줘")


def transcribe_prompt(names=NAMES) -> str:
    """받아쓰기 힌트. 목록만 준다 — 문장을 주면 그 문장을 받아 적는 일이 있다.
    2026-09-22: 짧은 명령 낱말도 넣는다('잘 자'→'탈자' 합성음 사례)."""
    return "하이 티드, " + ", ".join(COMMANDS) + ", " + ", ".join(names)
```
`app/realtime_protocol.py`:
```python
def command_tools(music_on: bool) -> list[dict]:
    """글자로 못 알아본 명령을 모델이 소리로 듣고 부르는 도구(2026-09-22)."""
    tools = [
        {"type": "function", "name": "go_to_sleep",
         "description": "아이가 대화를 끝내고 싶어 할 때만 부른다(잘 자, 바이바이, 그만할래, 쉬고 있어). "
                        "인형·동물에게 '잘 자'라고 하는 놀이 속 말이면 부르지 않는다.",
         "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "start_game",
         "description": "아이가 놀이를 하자고 할 때만 부른다. animal=동물 소리 놀이, repeat=따라 말하기 놀이.",
         "parameters": {"type": "object", "properties": {
             "kind": {"type": "string", "enum": ["animal", "repeat"]}}, "required": ["kind"]}},
    ]
    if music_on:
        tools.insert(1, {"type": "function", "name": "play_song",
                         "description": "아이가 노래를 **틀어 달라고** 할 때만 부른다. 노래 이야기만 하면 부르지 않는다.",
                         "parameters": {"type": "object", "properties": {
                             "title": {"type": "string", "description": "노래·캐릭터 이름. 모르면 빈 문자열"}}}})
    return tools
```
`session_update(cfg, instructions, tools=None)`: 세션 dict 에 `if tools: s["tools"] = tools; s["tool_choice"] = "auto"`.
`respond`:
```python
def respond(instructions: str | None = None, *, tools: bool = False) -> dict:
    r: dict = {"tool_choice": "auto" if tools else "none"}
    if instructions is not None:
        r["instructions"] = instructions
    return {"type": "response.create", "response": r}
```
`say_exactly` 의 `response` 에 `"tool_choice": "none"`.
```python
def tool_output(call_id: str) -> dict:
    return {"type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": "ok"}}


def function_calls(response: dict) -> list[tuple[str, dict, str]]:
    out = []
    for item in (response or {}).get("output") or []:
        if item.get("type") != "function_call":
            continue
        try:
            args = json.loads(item.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        out.append((item.get("name", ""), args if isinstance(args, dict) else {}, item.get("call_id", "")))
    return out
```
(`import json` 추가.)

`app/realtime_session.py` — `__init__(self, cfg, api_key, connect=None, tools=None)` → `self.tools = tools`; `open()` 의 `session_update(self.cfg, instructions)` → `session_update(self.cfg, instructions, self.tools)`.

`app/realtime_conversation.py`:
- `_on_transcript` 에서 요청 만들기 뒤쪽을 `_request(r, text, now)` 메서드로 뽑는다(내용 그대로: verbatim 판정 → msg → `_Pending` → `mark` → `_send`). chat 은 `msg = respond(tools=True)`.
- `_on_done` 맨 앞(`self._pending = None` 다음):
```python
        calls = function_calls(response) if p.route.kind == "chat" else []
        if calls:
            await self._on_tool(p, *calls[0])
            return
```
- 새 메서드:
```python
    async def _on_tool(self, p: _Pending, name: str, args: dict, call_id: str) -> None:
        """모델이 소리를 듣고 명령이라고 판단했다 — 기존 경로를 탄다(2026-09-22)."""
        await self._send(tool_output(call_id))
        log.info("[도구] %s %s ← %s", name, args, p.child)
        if self.metrics is not None:
            self.metrics.record_realtime_turn(
                kind="tool", perceived_s=None, transcribe_s=None, respond_first_s=None,
                filler=p.filler, reply="", child_text=p.child, cost_usd=None, cached_tokens=0,
                safety=[], game_missing=[], tool=name)
        if name == "go_to_sleep":
            self._say_cached("sleep")
            self._finish("sleep")
            return
        title = (args.get("title") or "").strip()
        text = {"play_song": f"{title} 틀어줘" if title else "노래 틀어줘",
                "start_game": ("동물 소리 놀이 하자" if args.get("kind") != "repeat"
                               else "따라 말하기 놀이 하자")}.get(name)
        r = route(text, music=self.music, games=self.games, sleep_words=self.sleep_words) if text else None
        if r is None or r.kind == "chat" or r.kind == "empty":
            log.warning("[도구] %s 를 처리할 경로가 없다 → 되묻는다", name)
            self._say_cached("recovery")
            return
        await self._request(r, p.child, self.clock())
```
⚠️ `play_song` 의 `title` 이 비면 `"노래 틀어줘"` → `music.parse` 가 기본 동요로 간다. 있으면 `"티니핑 틀어줘"` — `music.parse` 는 '틀어' 만으로 요청으로 본다.
- import 에 `function_calls, tool_output` 추가.
- `record_realtime_turn` 에 `tool: str | None = None` 인자와 `"tool": tool` 필드(`app/metrics.py`).

`app/main_realtime.py`:
```python
    from .realtime_protocol import command_tools
    tools = command_tools(music_on)

    def make_session():
        return RealtimeSession(cfg, api_key, tools=tools)
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/ -q` / Expected: 전부 PASS

- [ ] **Step 5: 커밋**

```bash
git add app/song_names.py app/realtime_protocol.py app/realtime_session.py app/realtime_conversation.py app/main_realtime.py app/metrics.py tests/test_song_names.py tests/test_realtime_protocol.py tests/test_realtime_conversation.py
git commit -m "feat(realtime): 짧은 명령 — 받아 적기 힌트 + chat 턴에만 잠들기·노래·놀이 도구"
```

---

### Task 7: 실서버 확인 · 젯슨 반영 · 기록

- [ ] **Step 1: 놀이 바로잡기 실서버** — scratchpad `drift_lead.py` 를 `Conversation` 경로로 바꾼 확인 스크립트(가짜 스피커가 `mark/played/truncate` 를 기록)로 놀이 10턴 이상. 이탈 몇 번, 자른 위치(글자), `game_leaked` 몇 번.
- [ ] **Step 2: 도구 실서버** — marin 합성 "탈자"(글자 그대로 합성), "잘 자", "움월 소리놀이 하자", "비니닝 노래", "고양이한테 잘 자 해줘"(부르면 안 됨)를 흘려 어떤 도구가 불리는지.
- [ ] **Step 3: 말 끝 결과** — Task 1 의 전체 실행 표를 `reports/turn/realtime_turn_end.md` 에 쓰고(원자료 jsonl 은 저장소 제외 — `.gitignore` 확인) 커밋. 사용자에게 표를 보이고 값을 고르게 한다.
- [ ] **Step 4: 젯슨 반영** — 봇이 꺼져 있을 때(`ps aux | grep "[a]pp.main"`) 바뀐 파일 scp(`tr -d '\r'`), `./run.sh check`. 첫 기동에 조각 문장 캐시 약 40개(2~3분) 안내.
- [ ] **Step 5: spec §8 보류 기록에 결과 적고 커밋·푸시.**
