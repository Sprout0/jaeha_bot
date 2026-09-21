# 전면 API 봇 안전 막기 · 정정 · 끊김 복구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 전면 API 봇이 위험한 답을 소리 전에 막고, 못 하는 걸 하겠다는 답을 정정하고, 끊기면 말없이 다시 붙는다.

**Architecture:** 판단은 소리·소켓을 모르는 `app/reply_gate.py`(턴마다 `TurnGuard`)에 둔다. `Conversation` 은 가드의 결정(`release`/`block`)대로 소리 조각을 스피커나 보류 버퍼로 보내고, 막거나 정정한 일은 `app/guardian_log.py` 에 남긴다. 끊김은 `Conversation` 이 `make_session()` 으로 새 연결을 만들어 답하던 턴을 다시 요청한다.

**Tech Stack:** Python 3.11(conda `jaeha_bot`), asyncio, pytest, 기존 `app/safety.py`·`app/claims.py`.

## Global Constraints

- spec: `docs/superpowers/specs/2026-09-21-realtime-guard-design.md`
- 로컬 봇(`app/main.py`)은 건드리지 않는다.
- 코드 실행은 conda: `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe`
- `git add` 는 파일을 하나씩 지정한다(다른 세션이 동시에 작업한다). 커밋 끝에 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- 어휘는 `configs/safety_rules.yaml` 에서 읽는다(코드에 박지 않는다).
- 안전 문장 기본값 `그건 위험해! 엄마 아빠한테 같이 가자.` / 정정 문장 기본값 `아, 그건 아직 못 해. 대신 동물 소리 놀이 할까?`
- 붙잡기 한도 `hold_cap_s: 3.0`, 다시 붙기 `reconnect_tries: 3`(간격 0.5 → 1 → 2초)
- `빈응답` 판정은 막는 이유로 쓰지 않는다.
- 글자 도중 판정은 `위험행동제안` 만 쓴다(`어른유도없음` 은 문장 끝에서만).
- 판정기 예외: 붙잡은 턴은 막는 쪽으로(안전 쪽 실패), 흘려보내는 턴은 그대로.

---

### Task 1: 위험 신호와 턴 가드 (`app/reply_gate.py`)

**Files:**
- Modify: `app/safety.py` (끝에 `question_risk` 추가)
- Modify: `configs/safety_rules.yaml` (`reply_check.child_ingest` 추가)
- Create: `app/reply_gate.py`
- Test: `tests/test_reply_gate.py`

**Interfaces:**
- Produces:
  - `safety.question_risk(child_text: str) -> bool`
  - `reply_gate.GuardConfig(hold_on_risk=True, hold_cap_s=3.0, reconnect_tries=3, safe_line=..., cant_line=...)`, `GuardConfig.from_dict(d)`
  - `reply_gate.ReplyGate(cfg: GuardConfig)` · `.begin(child: str, *, verbatim: bool, now: float) -> TurnGuard`
  - `TurnGuard.mode: str` (`hold` | `stream` | `off`), `TurnGuard.held: bool`(처음에 붙잡았나), `.poll(said: str, now: float) -> str | None` (`"release"` | `"block"` | None), `.finish(said: str) -> Verdict`
  - `reply_gate.Verdict(action: str, flags: list[str], fabricated: list[str])` — action `release` | `block` | `pass`
  - 상수 `reply_gate.PROPOSE = "위험행동제안"`

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_reply_gate.py`

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_reply_gate.py -q`
Expected: FAIL (`ImportError: cannot import name 'question_risk'` / `No module named 'app.reply_gate'`)

- [ ] **Step 3: 구현**

`configs/safety_rules.yaml` 의 `reply_check:` 아래 `ingest:` 목록 다음에 추가:

```yaml
  child_ingest:      # 아이 말 쪽 먹기 단서 — '이거'와 같이 오면 답을 소리 전에 붙잡는다(2026-09-21)
    - 먹
    - 맛
    - 마셔
    - 마실
    - 삼켜
    - 입에
```

`app/safety.py` 끝에 추가:

```python
def question_risk(child_text: str) -> bool:
    """아이 말에 위험 신호가 있나 — 전면 API 봇이 답을 소리 전에 붙잡을지 정한다(2026-09-21).

    위험 낱말이 있거나, 무엇인지 모를 '이거' 류가 먹기 단서와 같이 오면 True.
    08-12 "이거 무슨 맛이야?" 사례가 뒤쪽이다.
    """
    t = child_text or ""
    rules = _rules()
    if any(_contains(w, t) for w in rules.get("danger") or []):
        return True
    unknown = any(_contains(w, t) for w in rules.get("unknown_ref") or [])
    return unknown and any(w in t for w in rules.get("child_ingest") or [])
```

`app/reply_gate.py`:

```python
"""답이 소리로 나가기 전에 — 붙잡을지, 막을지, 정정할지 판단한다. (2026-09-21)

spec: docs/superpowers/specs/2026-09-21-realtime-guard-design.md
소리·소켓·시계를 모른다. 시각은 인자로 받는다 — 그래서 소리 없이 시험한다.

턴마다 TurnGuard 하나:
  hold   : 아이 말에 위험 신호 → 소리를 모아 두고 글자가 끝나면 검사한다
  stream : 평소 → 소리는 바로 나가고, 글자 도중 '위험행동제안'이면 끊는다
  off    : 우리가 쓴 문장을 그대로 읽는 턴(노래 안내·놀이 템플릿) → 안 본다
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields

from . import claims, safety

log = logging.getLogger("jaeha_bot.reply_gate")

EMPTY = "빈응답"
PROPOSE = "위험행동제안"      # 글자 도중에도 판정할 수 있는 유일한 것(위험물 × 행동 유도 조합)
JUDGE_ERROR = "판정오류"


@dataclass
class GuardConfig:
    hold_on_risk: bool = True
    hold_cap_s: float = 3.0
    reconnect_tries: int = 3
    safe_line: str = "그건 위험해! 엄마 아빠한테 같이 가자."
    cant_line: str = "아, 그건 아직 못 해. 대신 동물 소리 놀이 할까?"

    @classmethod
    def from_dict(cls, d: dict | None) -> "GuardConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class Verdict:
    action: str                                   # release | block | pass
    flags: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)


def _flags(said: str, child: str) -> list[str]:
    return [f for f in safety.check_reply(said, child_text=child) if f != EMPTY]


class TurnGuard:
    def __init__(self, child: str, *, mode: str, cap_s: float, started: float) -> None:
        self.child, self.mode, self.cap_s, self.started = child, mode, float(cap_s), started
        self.held = mode == "hold"

    def poll(self, said: str, now: float) -> str | None:
        """글자가 더 왔거나 시간이 흘렀다. 'release'(모은 소리 내보내기) / 'block' / None."""
        if self.mode == "off" or (self.mode == "hold" and now - self.started < self.cap_s):
            return None
        if not said:
            if self.mode == "hold":             # 한도가 지났는데 글자가 없다 — 소리만이라도
                self.mode = "stream"
                return "release"
            return None
        try:
            proposes = PROPOSE in _flags(said, self.child)
        except Exception as e:                  # noqa: BLE001 — 판정기가 봇을 죽이면 안 된다
            log.warning("[안전] 도중 판정 실패: %s", e)
            proposes = self.mode == "hold"      # 붙잡은 턴은 안전 쪽으로
        if proposes:
            return "block"
        if self.mode == "hold":
            self.mode = "stream"
            return "release"
        return None

    def finish(self, said: str) -> Verdict:
        """답의 글자가 끝났다."""
        if self.mode == "off":
            return Verdict("pass")
        try:
            flags = _flags(said, self.child) if said else []
        except Exception as e:                  # noqa: BLE001
            log.warning("[안전] 끝 판정 실패: %s", e)
            flags = [JUDGE_ERROR] if self.mode == "hold" else []
        if self.mode == "hold" and flags:
            return Verdict("block", flags)
        try:
            fabricated = claims.find_fabrications(said) if said else []
        except Exception as e:                  # noqa: BLE001
            log.warning("[정정] 판정 실패: %s", e)
            fabricated = []
        return Verdict("release" if self.mode == "hold" else "pass", flags, fabricated)


class ReplyGate:
    def __init__(self, cfg: GuardConfig | None = None) -> None:
        self.cfg = cfg or GuardConfig()

    def begin(self, child: str, *, verbatim: bool, now: float) -> TurnGuard:
        if verbatim:
            mode = "off"
        elif self.cfg.hold_on_risk and safety.question_risk(child):
            mode = "hold"
        else:
            mode = "stream"
        return TurnGuard(child, mode=mode, cap_s=self.cfg.hold_cap_s, started=now)
```

- [ ] **Step 4: 통과 확인**

Run: `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_reply_gate.py tests/test_safety.py -q`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/reply_gate.py app/safety.py configs/safety_rules.yaml tests/test_reply_gate.py
git commit -m "feat(guard): 턴 가드 — 위험 신호 턴은 소리를 붙잡고, 도중엔 위험행동제안만 끊는다"
```

---

### Task 2: 부모 기록 (`app/guardian_log.py`)

**Files:**
- Create: `app/guardian_log.py`
- Test: `tests/test_guardian_log.py`

**Interfaces:**
- Produces: `GuardianLog(log_dir, notifier=None, now=datetime.now)` · `.record(kind: str, **fields) -> dict`; `NullNotifier().notify(event: dict) -> None`

- [ ] **Step 1: 실패하는 시험**

```python
import json
from datetime import datetime

from app.guardian_log import GuardianLog


def _now():
    return datetime(2026, 9, 21, 10, 12, 3)


def test_한_줄씩_날짜_파일에_남긴다(tmp_path):
    g = GuardianLog(tmp_path, now=_now)
    g.record("safety_block", child="칼 어딨어?", reply="칼 같이 찾아볼까?", flags=["위험행동제안"])
    g.record("fabrication", child="놀자", reply="색칠 하자", items=["색칠"])
    lines = (tmp_path / "guardian_20260921.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert len(lines) == 2 and first["kind"] == "safety_block"
    assert first["ts"] == "2026-09-21T10:12:03" and first["child"] == "칼 어딨어?"


def test_보내는_자리를_부른다(tmp_path):
    got = []

    class N:
        def notify(self, ev):
            got.append(ev)

    GuardianLog(tmp_path, notifier=N(), now=_now).record("connection_lost", tries=3)
    assert got[0]["kind"] == "connection_lost" and got[0]["tries"] == 3


def test_쓰기나_보내기가_실패해도_예외를_안_던진다(tmp_path):
    class Bad:
        def notify(self, ev):
            raise RuntimeError("x")

    blocked = tmp_path / "f"
    blocked.write_text("파일이라 디렉터리로 못 쓴다")
    GuardianLog(blocked, notifier=Bad(), now=_now).record("safety_block")
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_guardian_log.py -q` / Expected: FAIL (`No module named 'app.guardian_log'`)

- [ ] **Step 3: 구현** — `app/guardian_log.py`

```python
"""부모 기록 — 막은 일·정정한 일·끊김 복구 실패를 남긴다. (2026-09-21)

logs/guardian_YYYYMMDD.jsonl 에 한 줄씩. **소리는 남기지 않는다** — 글자만.
notify() 는 나중에 텔레그램을 끼우는 자리다. 기본은 아무것도 안 보낸다.
무엇을 기기 밖으로 내보낼지(원문/요약)는 그걸 붙일 때 정한다.
기록·전송이 실패해도 대화는 멈추지 않는다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("jaeha_bot.guardian")


class NullNotifier:
    def notify(self, event: dict) -> None:
        return None


class GuardianLog:
    def __init__(self, log_dir, notifier=None, now=datetime.now) -> None:
        self.dir = Path(log_dir)
        self.notifier = notifier or NullNotifier()
        self._now = now

    def record(self, kind: str, **fields) -> dict:
        t = self._now()
        ev = {"ts": t.isoformat(timespec="seconds"), "kind": kind, **fields}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f"guardian_{t:%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception as e:                  # noqa: BLE001
            log.warning("부모 기록을 못 썼다: %s", e)
        try:
            self.notifier.notify(ev)
        except Exception as e:                  # noqa: BLE001
            log.warning("부모 알림을 못 보냈다: %s", e)
        log.info("[부모 기록] %s %s", kind, {k: v for k, v in fields.items() if k != "reply"})
        return ev
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_guardian_log.py -q` / Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/guardian_log.py tests/test_guardian_log.py
git commit -m "feat(guard): 부모 기록 — logs/guardian_*.jsonl 과 notify 자리(텔레그램은 나중)"
```

---

### Task 3: 고정 문장 · 글자 메시지 · 계측 필드

**Files:**
- Modify: `app/realtime_turn.py` (`PHRASES` 에 `safe`, `cant`)
- Modify: `app/realtime_protocol.py` (`user_text`)
- Modify: `app/metrics.py` (`record_realtime_turn` 필드)
- Modify: `configs/model_paths.yaml` (`realtime.guard`)
- Test: `tests/test_realtime_turn.py`, `tests/test_realtime_protocol.py`, `tests/test_metrics_realtime.py`

**Interfaces:**
- Consumes: `reply_gate.GuardConfig`
- Produces: `PHRASES["safe"]`, `PHRASES["cant"]`; `realtime_protocol.user_text(text: str) -> dict`; `record_realtime_turn(..., held=False, hold_s=None, blocked=False, corrected=False, reconnects=0)`

- [ ] **Step 1: 실패하는 시험** — 각 파일 끝에 추가

`tests/test_realtime_turn.py`:
```python
def test_안전_문장과_정정_문장이_고정_문구에_있다():
    from app.reply_gate import GuardConfig
    assert PHRASES["safe"] and PHRASES["cant"]
    assert PHRASES["safe"] != GuardConfig().cant_line
```

`tests/test_realtime_protocol.py`:
```python
def test_아이_말을_글자_메시지로_넣는다():
    from app.realtime_protocol import user_text
    m = user_text("칼 어딨어?")
    assert m["type"] == "conversation.item.create" and m["item"]["role"] == "user"
    assert m["item"]["content"][0] == {"type": "input_text", "text": "칼 어딨어?"}
```

`tests/test_metrics_realtime.py` (기존 파일의 로거 만드는 방식을 따른다 — 파일 안 첫 시험을 보고 같은 픽스처를 쓴다):
```python
def test_막기_붙잡기_다시붙기를_남긴다(tmp_path, monkeypatch):
    import json
    from app import metrics as m
    monkeypatch.setattr(m, "LOG_DIR", tmp_path, raising=False)
    lg = m.MetricsLogger(enabled=True, tag="t")
    lg.record_realtime_turn(kind="chat", perceived_s=1.0, transcribe_s=0.5, respond_first_s=0.5,
                            filler=False, reply="r", child_text="c", cost_usd=0.001,
                            cached_tokens=0, safety=[], game_missing=[],
                            held=True, hold_s=0.8, blocked=True, corrected=False, reconnects=1)
    rec = [json.loads(x) for x in lg.path.read_text(encoding="utf-8").splitlines()][-1]
    assert rec["held"] is True and rec["hold_s"] == 0.8 and rec["blocked"] is True
    assert rec["corrected"] is False and rec["reconnects"] == 1
```
⚠️ 위 시험의 로그 경로 잡는 법(`LOG_DIR`, `lg.path`)은 `tests/test_metrics_realtime.py` 의 기존 시험과 똑같이 맞춘다. 다르면 기존 방식으로 고쳐 쓴다.

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_realtime_turn.py tests/test_realtime_protocol.py tests/test_metrics_realtime.py -q` / Expected: FAIL (KeyError 'safe' / ImportError user_text / unexpected keyword 'held')

- [ ] **Step 3: 구현**

`app/realtime_turn.py` — `PHRASES` 정의 바로 아래:
```python
# 🔴 2026-09-21 안전 막기·정정(spec 2026-09-21-realtime-guard). 문장은 설정에서 바꾼다.
from .reply_gate import GuardConfig as _GuardConfig

_GUARD = _GuardConfig.from_dict((settings.models.get("realtime") or {}).get("guard"))
PHRASES["safe"] = _GUARD.safe_line
PHRASES["cant"] = _GUARD.cant_line
```

`app/realtime_protocol.py` — `history_items` 아래:
```python
def user_text(text: str) -> dict:
    """아이 말을 글자로 대화에 넣는다 — 끊겼다 다시 붙었을 때 답하던 말을 다시 묻는다."""
    return _message("user", text)
```

`app/metrics.py` — `record_realtime_turn` 시그니처 끝에 `held: bool = False, hold_s: float | None = None, blocked: bool = False, corrected: bool = False, reconnects: int = 0` 를 더하고, `rec` 의 `"game_missing": ...` 뒤에:
```python
               "held": bool(held), "hold_s": r(hold_s), "blocked": bool(blocked),
               "corrected": bool(corrected), "reconnects": int(reconnects),
```
로그 줄 끝에 `%s` 하나를 더해 `(" 막음" if blocked else "") + (" 붙잡음" if held else "")` 를 붙인다.

`configs/model_paths.yaml` — `realtime:` 블록 끝(`voice_cache_dir` 다음):
```yaml
  # 🔴 2026-09-21 안전 막기·정정·끊김 복구(spec 2026-09-21-realtime-guard-design.md)
  guard:
    hold_on_risk: true      # 아이 말에 위험 신호가 있으면 답 소리를 붙잡고 글자를 다 본 뒤 낸다
    hold_cap_s: 3.0         # 붙잡기 한도 — 넘으면 거기까지 보고 흘려보낸다
    reconnect_tries: 3      # 끊기면 말없이 다시 붙기(0.5 → 1 → 2초)
    safe_line: "그건 위험해! 엄마 아빠한테 같이 가자."
    cant_line: "아, 그건 아직 못 해. 대신 동물 소리 놀이 할까?"
```

- [ ] **Step 4: 통과 확인** — 같은 명령 / Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/realtime_turn.py app/realtime_protocol.py app/metrics.py configs/model_paths.yaml tests/test_realtime_turn.py tests/test_realtime_protocol.py tests/test_metrics_realtime.py
git commit -m "feat(guard): 안전·정정 고정 문장, 글자 메시지, 막기·붙잡기·다시붙기 계측"
```

---

### Task 4: 대화 구간에 가드 연결 (붙잡기 · 막기 · 정정)

**Files:**
- Modify: `app/realtime_conversation.py`
- Test: `tests/test_realtime_conversation.py`

**Interfaces:**
- Consumes: `ReplyGate.begin`, `TurnGuard.poll/finish/mode/held`, `Verdict`, `PROPOSE`, `PHRASES["safe"|"cant"]`, `GuardianLog.record`, `record_realtime_turn(held=, hold_s=, blocked=, corrected=, reconnects=)`
- Produces: `Conversation(..., gate=None, guardian=None)` — gate 가 None 이면 `ReplyGate()`(기본 설정)

- [ ] **Step 1: 실패하는 시험** — `tests/test_realtime_conversation.py`

`FakeSpeaker` 에 추가:
```python
    def clear(self):
        self.cleared = getattr(self, "cleared", 0) + 1
```
파일 끝에 추가:
```python
class FakeGuardian:
    def __init__(self):
        self.records = []

    def record(self, kind, **fields):
        self.records.append((kind, fields))


def _text(t):
    return {"type": "response.output_audio_transcript.delta", "delta": t}


def _done():
    return {"type": "response.done", "response": {"usage": {}}}


def _guard_conv(s, speaker, guardian, history=None, **kw):
    return Conversation(session=s, mic=asyncio.Queue(), speaker=speaker, cache=FakeCache(),
                        cfg=_cfg(), music=None, games=GameManager(render=None), sleep_words=None,
                        instructions="지시", history=history if history is not None else [],
                        sleep_timeout=0.4, filler_phrases=[], tick_s=0.01, guardian=guardian, **kw)


def test_위험_신호_턴은_글자가_끝나기_전엔_소리를_안_낸다():
    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        c = _guard_conv(s, sp, g)
        await c._on_transcript("칼 어딨어?", 0.0)
        await c._on_event(_audio_delta())
        before = len(sp.pushed)
        await c._on_event(_text("칼은 위험해! 엄마한테 말하자."))
        await c._on_event(_done())
        return before, sp, g

    before, sp, g = asyncio.run(go())
    assert before == 0 and len(sp.pushed) == 1 and g.records == []


def test_위험_신호_턴이_걸리면_소리는_버리고_안전_문장과_부모_기록():
    async def go():
        s, sp, g, history = FakeSession(), FakeSpeaker(), FakeGuardian(), []
        c = _guard_conv(s, sp, g, history=history)
        await c._on_transcript("칼 어딨어?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("칼은 부엌에 있어! 같이 찾아볼까?"))
        await c._on_event(_done())
        return c, sp, g, history

    c, sp, g, history = asyncio.run(go())
    assert PHRASES["safe"] in c.cache.asked and len(sp.pushed) == 1   # 안전 문장 하나만
    assert g.records[0][0] == "safety_block" and g.records[0][1]["leaked"] is False
    assert history == [("칼 어딨어?", PHRASES["safe"])]


def test_흘려보내는_턴이_도중에_걸리면_취소하고_스피커를_비운다():
    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        c = _guard_conv(s, sp, g)
        await c._on_transcript("뭐 하고 놀까?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("우리 칼 같이 찾아볼까"))
        await c._on_event(_audio_delta())                  # 막은 뒤 오는 조각은 버린다
        await c._on_event(_done())
        return s, c, sp, g

    s, c, sp, g = asyncio.run(go())
    assert {"type": "response.cancel"} in s.sent and sp.cleared == 1
    assert PHRASES["safe"] in c.cache.asked and len(sp.pushed) == 2      # 새어 나간 1 + 안전 문장
    assert g.records[0][1]["leaked"] is True


def test_못_하는_것이면_답_뒤에_정정한다():
    async def go():
        s, sp, g, history = FakeSession(), FakeSpeaker(), FakeGuardian(), []
        c = _guard_conv(s, sp, g, history=history)
        await c._on_transcript("뭐 하고 놀까?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("좋아! 색칠 놀이 하자!"))
        await c._on_event(_done())
        return c, g, history

    c, g, history = asyncio.run(go())
    assert PHRASES["cant"] in c.cache.asked and g.records[0][0] == "fabrication"
    assert history[0][1].endswith(PHRASES["cant"])


def test_붙잡기_한도가_지나면_시계가_내보낸다():
    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        from app.reply_gate import GuardConfig, ReplyGate
        c = _guard_conv(s, sp, g, gate=ReplyGate(GuardConfig(hold_cap_s=0.05)))
        asyncio.create_task(_feed(s, [_transcript("칼 어딨어?"), _audio_delta(),
                                      _text("칼은")], gap=0.01))
        asyncio.create_task(_feed(s, [_done()], gap=0.25))
        await c.run(greet=False)
        return sp

    assert len(asyncio.run(go()).pushed) >= 1


def test_계측에_붙잡기와_막기가_남는다():
    class Rec:
        kw = None

        def record_realtime_turn(self, **kw):
            Rec.kw = kw

    async def go():
        s, sp, g = FakeSession(), FakeSpeaker(), FakeGuardian()
        c = _guard_conv(s, sp, g)
        c.metrics = Rec()
        await c._on_transcript("칼 어딨어?", 0.0)
        await c._on_event(_audio_delta())
        await c._on_event(_text("칼은 부엌에 있어."))
        await c._on_event(_done())

    asyncio.run(go())
    assert Rec.kw["held"] is True and Rec.kw["blocked"] is True and Rec.kw["reconnects"] == 0
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_realtime_conversation.py -q` / Expected: FAIL (`unexpected keyword argument 'guardian'`)

- [ ] **Step 3: 구현** — `app/realtime_conversation.py`

import 에 추가:
```python
from .realtime_protocol import (append_audio, cached_tokens, cancel, cost_usd, event_kind,
                                respond, say_exactly, user_text)
from .reply_gate import PROPOSE, ReplyGate, TurnGuard
```
`_Pending` 에 필드 추가(기존 필드 뒤):
```python
    msg: dict | None = None                         # 다시 붙었을 때 같은 방식으로 다시 요청
    guard: TurnGuard | None = None
    held: list = field(default_factory=list)        # 붙잡은 소리 조각
    first_delta_at: float | None = None
    hold_s: float | None = None
    blocked: bool = False
    blocked_at: float | None = None
    block_flags: list = field(default_factory=list)
    reconnects: int = 0
```
`__init__` 인자에 `gate=None, guardian=None` 추가, 본문에:
```python
        self.gate = gate or ReplyGate()
        self.guardian = guardian
```
⚠️ 기존 `self.gate = MicGate(...)` 와 이름이 겹친다 → 마이크 게이트를 `self.mic_gate` 로 바꾸고 `_pump_mic` 의 `self.gate.sync/should_send` 도 `self.mic_gate` 로 바꾼다.

`_record` 도우미(클래스 안):
```python
    def _guardian_record(self, kind: str, **fields) -> None:
        if self.guardian is not None:
            self.guardian.record(kind, **fields)
```
`_on_transcript` 의 요청 만들기를 바꾼다:
```python
        verbatim = r.kind == "music" or (r.kind.startswith("game") and not r.instructions)
        if verbatim:
            msg = say_exactly(r.say)
        elif r.kind.startswith("game"):
            msg = respond(r.instructions)
        else:
            msg = respond()
        t = self.clock()
        self._pending = _Pending(r, text, self._stopped_at, self._speech_end_at, now, t, msg=msg,
                                 guard=self.gate.begin(text, verbatim=verbatim, now=t))
        if self._pending.guard.held:
            log.info("[안전] 위험 신호 — 답 소리를 붙잡는다: %s", text)
        await self._send(msg)
```
`_on_event` 의 audio / text 분기를 바꾼다:
```python
        elif k == "audio" and p is not None:
            if p.blocked:
                return
            samples = p.acc.feed(base64.b64decode(ev.get("delta", "")))
            if p.first_delta_at is None:
                p.first_delta_at = now
            if p.guard is not None and p.guard.mode == "hold":
                p.held.append(samples)
            else:
                if p.first_audio_at is None:
                    p.first_audio_at = now
                self.speaker.push(samples)
        elif k == "text" and p is not None:
            if p.blocked:
                return
            p.said += ev.get("delta", "")
            await self._poll_guard(p, now)
```
새 메서드:
```python
    async def _poll_guard(self, p: _Pending, now: float) -> None:
        act = p.guard.poll(p.said, now) if p.guard is not None else None
        if act == "release":
            self._release(p, now)
        elif act == "block":
            await self._block(p, now, [PROPOSE], cancel_response=True)

    def _release(self, p: _Pending, now: float) -> None:
        for s in p.held:
            self.speaker.push(s)
        if p.held and p.first_audio_at is None:
            p.first_audio_at = now
        p.held.clear()
        if p.first_delta_at is not None and p.hold_s is None:
            p.hold_s = now - p.first_delta_at

    async def _block(self, p: _Pending, now: float, flags: list, *, cancel_response: bool) -> None:
        leaked = p.first_audio_at is not None
        p.blocked, p.blocked_at, p.block_flags = True, now, list(flags)
        p.held.clear()
        if leaked:
            self.speaker.clear()
        if cancel_response:
            await self._send(cancel())
        self._say_cached("safe")
        if p.first_audio_at is None:
            p.first_audio_at = now        # 안전 문장이 첫 소리 — 응답 제한에 안 걸린다
        log.warning("[안전] 막음 %s ← %s", flags, p.said.strip())
        self._guardian_record("safety_block", child=p.child, reply=p.said.strip(), flags=list(flags),
                              held=bool(p.guard and p.guard.held), leaked=leaked)
```
`_on_done` 앞부분(`reply, kind = ...` 부터 안전 로그까지)을 바꾼다:
```python
        reply, kind, now = p.said.strip(), p.route.kind, self.clock()
        corrected = False
        if p.blocked:
            flags = p.block_flags
        else:
            v = p.guard.finish(reply) if p.guard is not None else None
            flags = v.flags if v else []
            if v is not None and v.action == "release":
                self._release(p, now)
            elif v is not None and v.action == "block":
                await self._block(p, now, v.flags, cancel_response=False)
            elif flags:
                log.warning("[안전] %s ← %s (다 나간 뒤라 기록만)", flags, reply)
                self._guardian_record("safety_late", child=p.child, reply=reply, flags=flags)
            if v is not None and v.fabricated and not p.blocked:
                self._say_cached("cant")
                corrected = True
                self._guardian_record("fabrication", child=p.child, reply=reply,
                                      items=list(v.fabricated))
```
`record_realtime_turn(...)` 호출 끝에 `held=bool(p.guard and p.guard.held), hold_s=p.hold_s, blocked=p.blocked, corrected=corrected, reconnects=p.reconnects` 추가.
대화 기록 부분을 바꾼다:
```python
        if kind in ("chat", "game", "game_start") and (reply or p.blocked):
            said = PHRASES["safe"] if p.blocked else (
                f"{reply} {PHRASES['cant']}" if corrected else reply)
            self.history.append((p.child, said))
            if len(self.history) > self.cfg.history_turns:
                del self.history[:len(self.history) - self.cfg.history_turns]
```
`_tick` 의 `now, p = ...` 바로 다음에:
```python
            if p is not None and p.blocked and now - (p.blocked_at or now) >= 2.0:
                await self._on_done(p, {})      # 취소 뒤 done 이 안 와도 턴을 닫는다
                continue
            if p is not None and not p.blocked and p.guard is not None and p.guard.mode == "hold":
                await self._poll_guard(p, now)
```

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_realtime_conversation.py tests/test_reply_gate.py -q` / Expected: PASS (기존 시험 포함)

- [ ] **Step 5: 커밋**

```bash
git add app/realtime_conversation.py tests/test_realtime_conversation.py
git commit -m "feat(guard): 대화 구간에 가드 — 위험 턴은 소리를 붙잡고, 걸리면 안전 문장·부모 기록, 못 하는 건 정정"
```

---

### Task 5: 끊기면 말없이 다시 붙기

**Files:**
- Modify: `app/realtime_conversation.py`
- Test: `tests/test_realtime_conversation.py`

**Interfaces:**
- Consumes: `user_text`, `Task 4` 의 `_Pending.msg/guard/reconnects`, `ReplyGate.begin`
- Produces: `Conversation(..., make_session=None, reconnect_delays=(0.5, 1.0, 2.0))`, 속성 `reconnects: int`

- [ ] **Step 1: 실패하는 시험** — 파일 끝에 추가

```python
class DeadSession(FakeSession):
    async def open(self, instructions, history):
        raise ConnectionLost("안 붙는다")


def test_끊기면_말없이_다시_붙고_답하던_말을_다시_묻는다():
    s1, s2 = FakeSession(), FakeSession()
    made = iter([s2])

    async def go():
        history = [("전", "후")]
        c = _conv(s1, history=history, sleep_timeout=0.4)
        c.make_session, c.reconnect_delays = (lambda: next(made)), (0.01,)
        asyncio.create_task(_feed(s1, [_transcript("공룡 좋아해?"), None]))
        asyncio.create_task(_feed(s2, [_audio_delta(), _text("응 좋아!"), _done()], gap=0.1))
        reason = await c.run(greet=False)
        return c, history, reason

    c, history, reason = asyncio.run(go())
    assert c.reconnects == 1 and reason == "idle"
    assert s2.opened[1][0] == ("전", "후")                      # 기록을 다시 넣었다
    items = [m for m in s2.sent if m["type"] == "conversation.item.create"]
    assert items[0]["item"]["content"][0]["text"] == "공룡 좋아해?"
    assert s2.requests() == [{"type": "response.create"}]
    assert history[-1] == ("공룡 좋아해?", "응 좋아!")
    assert PHRASES["lost"] not in c.cache.asked


def test_다시_붙기가_다_실패하면_lost_와_부모_기록():
    g = FakeGuardian()

    async def go():
        c = _conv(FakeSession())
        c.guardian, c.make_session, c.reconnect_delays = g, DeadSession, (0.01, 0.01, 0.01)
        asyncio.create_task(_feed(c.session, [None]))
        return await c.run(greet=False), c

    reason, c = asyncio.run(go())
    assert reason == "lost" and PHRASES["lost"] in c.cache.asked
    assert g.records[-1][0] == "connection_lost" and g.records[-1][1]["tries"] == 3


def test_첫_연결이_실패해도_다시_시도한다():
    s2 = FakeSession()

    async def go():
        c = _conv(DeadSession(), sleep_timeout=0.1)
        c.make_session, c.reconnect_delays = (lambda: s2), (0.01,)
        return await c.run(greet=False), c

    reason, c = asyncio.run(go())
    assert reason == "idle" and s2.opened is not None


def test_연결_함수가_없으면_옛_동작():
    async def go():
        c = _conv(FakeSession())
        asyncio.create_task(_feed(c.session, [None]))
        return await c.run(greet=False)

    assert asyncio.run(go()) == "lost"
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_realtime_conversation.py -q` / Expected: FAIL (`reconnects` 없음 / 옛 동작으로 lost)

- [ ] **Step 3: 구현** — `app/realtime_conversation.py`

`__init__` 인자에 `make_session=None, reconnect_delays=(0.5, 1.0, 2.0)` 추가, 본문에:
```python
        self.make_session = make_session
        self.reconnect_delays = tuple(reconnect_delays)
        self.reconnects = 0
        self._reconnecting = False
        self._tasks: list[asyncio.Task] = []
```
`run()` 의 연결 부분을 바꾼다:
```python
        if not await self._open():
            self._say_cached("lost")
            self._guardian_record("connection_lost", child="", tries=len(self._delays()))
            await self._wait_quiet()
            return "lost"
```
그리고 task 목록을 `self._tasks = [...]` 로 두고, `finally` 에서 `for t in self._tasks:` 로 취소·`gather(*self._tasks, ...)` 한다.

새 메서드:
```python
    def _delays(self) -> tuple:
        return self.reconnect_delays if self.make_session is not None else ()

    async def _open(self) -> bool:
        try:
            await self.session.open(self.instructions, self.history)
            return True
        except ConnectionLost as e:
            log.warning("Realtime 연결 실패: %s", e)
        return await self._reopen()

    async def _reopen(self) -> bool:
        delays = self._delays()
        for i, delay in enumerate(delays):
            await asyncio.sleep(delay)
            s = self.make_session()
            try:
                await s.open(self.instructions, self.history)
            except ConnectionLost as e:
                log.warning("다시 붙기 %d/%d 실패: %s", i + 1, len(delays), e)
                continue
            self.session = s
            self.reconnects += 1
            log.info("다시 붙었다 (%d번째 시도)", i + 1)
            return True
        return False

    async def _reconnect(self) -> None:
        try:
            await self.session.close()
        except Exception:                        # noqa: BLE001
            pass
        p = self._pending
        if await self._reopen():
            self._reconnecting = False
            if p is not None and p is self._pending and not p.blocked:
                await self._retry_turn(p)
            self._tasks.append(asyncio.create_task(self._read_events()))
            return
        self._reconnecting = False
        self._pending = None
        self._say_cached("lost")
        self._guardian_record("connection_lost", child=p.child if p else "",
                              tries=len(self._delays()))
        self._finish("lost")

    async def _retry_turn(self, p: _Pending) -> None:
        """답하던 중에 끊겼다 — 반쯤 나간 소리를 비우고 같은 말을 글자로 다시 묻는다."""
        self.speaker.clear()
        now = self.clock()
        verbatim = p.guard is not None and p.guard.mode == "off"
        p.said, p.acc, p.held = "", PcmAccumulator(), []
        p.first_audio_at = p.first_delta_at = p.hold_s = None
        p.requested_at, p.reconnects = now, p.reconnects + 1
        p.guard = self.gate.begin(p.child, verbatim=verbatim, now=now)
        log.info("[다시 붙기] 답하던 말을 다시 묻는다: %s", p.child)
        await self._send(user_text(p.child))
        await self._send(p.msg)
```
`_lost` 를 바꾼다:
```python
    def _lost(self, e) -> None:
        if (self._done is not None and self._done.done()) or self._reconnecting:
            return
        if self.make_session is None:
            log.warning("Realtime 끊김: %s", e)
            self._pending = None
            self._say_cached("lost")
            self._finish("lost")
            return
        log.warning("Realtime 끊김: %s → 말없이 다시 붙는다", e)
        self._reconnecting = True
        self._tasks.append(asyncio.create_task(self._reconnect()))
```
`_send` 맨 앞에 `if self._reconnecting: return`, `_pump_mic` 의 `if self.mic_gate.should_send(now):` 를 `if not self._reconnecting and self.mic_gate.should_send(now):` 로.

⚠️ `_retry_turn` 은 `self._reconnecting = False` 뒤에 불러야 `_send` 가 막히지 않는다(위 순서 그대로). 새 읽기 task 는 다시 묻기를 **보낸 뒤** 만든다 — 먼저 만들면 새 연결의 이벤트가 초기화 전의 턴에 섞인다.

- [ ] **Step 4: 통과 확인** — Run: `... -m pytest tests/test_realtime_conversation.py -q` / Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/realtime_conversation.py tests/test_realtime_conversation.py
git commit -m "feat(guard): 끊기면 말없이 다시 붙고 답하던 말을 다시 묻는다 — 3번 실패하면 lost·부모 기록"
```

---

### Task 6: 진입점 연결 · 젯슨 배포

**Files:**
- Modify: `app/main_realtime.py`
- Test: `tests/test_main_realtime.py`

**Interfaces:**
- Consumes: `GuardConfig.from_dict`, `ReplyGate`, `GuardianLog`, `Conversation(make_session=, reconnect_delays=, gate=, guardian=)`
- Produces: `reconnect_delays(tries: int) -> tuple[float, ...]` (0.5·2^i)

- [ ] **Step 1: 실패하는 시험** — `tests/test_main_realtime.py` 끝에

```python
def test_다시_붙기_간격은_두_배씩():
    from app.main_realtime import reconnect_delays
    assert reconnect_delays(3) == (0.5, 1.0, 2.0) and reconnect_delays(0) == ()
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/test_main_realtime.py -q` / Expected: FAIL (ImportError)

- [ ] **Step 3: 구현** — `app/main_realtime.py`

모듈 수준 함수:
```python
def reconnect_delays(tries: int) -> tuple[float, ...]:
    """끊겼을 때 다시 붙기 간격 — 0.5 → 1 → 2초(spec 2026-09-21 §6)."""
    return tuple(0.5 * 2 ** i for i in range(max(0, int(tries))))
```
`main()` 안, `common = dict(...)` 앞에:
```python
    from .config import BASE_DIR
    from .guardian_log import GuardianLog
    from .reply_gate import GuardConfig, ReplyGate
    gcfg = GuardConfig.from_dict((models.get("realtime") or {}).get("guard"))
    log.info("안전 가드 — 위험 신호 턴 붙잡기 %s(한도 %.1fs) / 다시 붙기 %d번",
             "켬" if gcfg.hold_on_risk else "끔", gcfg.hold_cap_s, gcfg.reconnect_tries)
```
`common` 에 `gate=ReplyGate(gcfg), guardian=GuardianLog(BASE_DIR / "logs"), reconnect_delays=reconnect_delays(gcfg.reconnect_tries)` 추가.
대화 구간 부르는 곳을 바꾼다:
```python
            def make_session():
                return RealtimeSession(cfg, api_key)
            ...
                reason = asyncio.run(_awake_period(
                    mic=mic, rate=source.samplerate, session=make_session(),
                    make_session=make_session,
                    preroll=preroll, greet=greet, woke_during_music=woke_during_music,
                    **common))
```
(`make_session` 은 while 루프 밖, `common` 정의 뒤에 한 번만 둔다.)

- [ ] **Step 4: 전체 스위트 통과**

Run: `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/ -q`
Expected: 전부 PASS

- [ ] **Step 5: 커밋 · 푸시 · 젯슨 반영**

```bash
git add app/main_realtime.py tests/test_main_realtime.py
git commit -m "feat(guard): 진입점에 가드·부모 기록·다시 붙기 연결"
git push
for f in app/reply_gate.py app/guardian_log.py app/safety.py app/realtime_turn.py app/realtime_protocol.py app/metrics.py app/realtime_conversation.py app/main_realtime.py configs/model_paths.yaml configs/safety_rules.yaml; do tr -d '\r' < $f | ssh jaeha_bot@100.65.22.17 "cat > ~/jaeha_bot/$f"; done
ssh jaeha_bot@100.65.22.17 'cd ~/jaeha_bot && ./run.sh check 2>&1 | grep -E "기동 검사|pipeline"'
```
Expected: `기동 검사: 통과`. 봇이 켜져 있지 않을 때만 반영한다(`ps aux | grep "[a]pp.main"` 로 먼저 확인).

---

### Task 7: 젯슨 실기 (사람 필요) · 기록

- [ ] **Step 1:** 사용자가 `cd ~/jaeha_bot && ./run.sh` — 첫 기동에 안전·정정 문장 캐시가 새로 만들어진다(`새로 만듦 2`).
- [ ] **Step 2:** 시험 말 — "칼 어딨어?", "이거 먹어도 돼?"(붙잡기), 평소 말(붙잡지 않음), "색칠 놀이 하자"(정정), 대화 중 공유기 선 잠깐 뽑기(다시 붙기).
- [ ] **Step 3:** 로그 확인: `grep -E "\[안전\]|\[다시 붙기\]|다시 붙었다|부모 기록" logs/jaeha_$(date +%Y%m%d).log`, `logs/guardian_*.jsonl`, `metrics` 의 `held/hold_s/blocked/reconnects`.
- [ ] **Step 4:** 결과를 spec §11 보류 기록에 적고 커밋.
