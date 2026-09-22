# 전면 API(Realtime) 대화 경로 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 젯슨 봇이 로컬 호출어(임베딩 검증)로 깨어나 OpenAI Realtime 으로 대화하고, 노래·놀이·맞장구·잠들기·무응답 대기를 지금 봇과 같게 처리한다.

**Architecture:** 새 진입점 `app/main_realtime.py` 가 호출어 대기(동기)를 돌리고, 깨면 `asyncio.run(...)` 으로 대화 구간(`Conversation.run`)을 돈다. 턴 판단(`realtime_turn.route`)·프로토콜 메시지(`realtime_protocol`)는 순수 함수, 소켓(`realtime_session`)·소리(`realtime_audio`)·고정 문구(`voice_cache`)는 얇은 어댑터로 분리해 가짜로 시험한다. 로컬 경로 `app/main.py` 는 건드리지 않는다.

**Tech Stack:** Python 3.10(젯슨)/3.12(노트북), websockets 16.1.1, numpy, sounddevice, soundfile, soxr, pytest(비동기는 `asyncio.run` — pytest-asyncio 없음).

**Spec:** `docs/superpowers/specs/2026-09-16-realtime-pipeline-design.md`

## Global Constraints

- 코드 실행은 conda env `jaeha_bot`: 노트북 `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest ...`, 젯슨 `source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot`.
- `app/main.py` 수정 금지. 기존 시험 1,032개가 그대로 통과해야 한다.
- 저장소 기본 `pipeline: local`. 젯슨 `configs/local.yaml` 에만 `pipeline: realtime`.
- 모델 `gpt-realtime-mini`, 목소리 `marin`, 받아 적기 `gpt-4o-transcribe`, 말끝 1200ms, 마이크 여유 0.15s, 이력 6턴, 맞장구 0.7s, 응답 제한 8s.
- 턴 진행은 `create_response: false` — 받아 적기를 보고 우리가 `response.create` 한다.
- 안전은 `safety.check_reply` 결과를 **로그만**. 소리를 막지 않는다.
- 고정 문구는 전부 marin wav 캐시. 전면 API 경로는 `app.stt_module`/`app.tts_module` 을 import 하지 않는다.
- 막히는 것은 멈추지 말고 spec 9절 "진행 중 발생"에 날짜·증상·원인·대안으로 적고 넘어간다.
- 커밋은 작게, `git add` 는 파일을 콕 집어서(다른 세션이 같은 저장소에서 일한다), 메시지 끝에 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. `main` 에 직접.
- wav 는 커밋하지 않는다. `configs/local.yaml` 은 git 에 없다.
- 확인된 사실: `app/config.py` 가 `.env` 를 읽는다, `AudioSource.samplerate` 가 있다(16000), `바이바이` 는 잠들기 낱말이다, 젯슨에 websockets 가 없고 aarch64 휠(16.1.1, cp310)은 있다.

## File Structure

| 파일 | 책임 |
|---|---|
| `app/realtime_audio.py` (새) | `MicGate`, `PcmAccumulator`, `to_pcm16`, `resolve_rate`, `StreamSpeaker` — 라이브 루프에서 옮김 |
| `app/realtime_protocol.py` (새) | `RealtimeConfig`, 세션·요청 메시지, 이벤트 분류, `PRICE`/`cost_usd`/`cached_tokens` — 순수 |
| `app/realtime_turn.py` (새) | `Route`, `route()`, `missing_required()`, `PHRASES` — 순수 |
| `app/education_modes.py` (추가) | `GameManager.maybe_start_beat` / `handle_beat` (기존 두 함수는 이걸 거쳐 동작 불변) |
| `app/realtime_session.py` (새) | `RealtimeSession`(소켓 하나), `ConnectionLost`, `synthesize()` |
| `app/voice_cache.py` (새) | `VoiceCache` — 문구 → wav |
| `app/realtime_conversation.py` (새) | `Conversation.run()` — 깨어 있는 한 구간 |
| `app/metrics.py` (추가) | `MetricsLogger.record_realtime_turn` |
| `app/main_realtime.py` (새) | 기동 검사, 조립, 호출어 루프, `--no-wake` |
| `tools/realtime_live.py`, `tools/realtime_probe.py` (수정) | 옮긴 것을 `app` 에서 import (동작 불변) |
| `configs/model_paths.yaml`, `run.sh`, `requirements.txt` (수정) | `pipeline`, `realtime:` 블록, 진입점 선택, websockets |

---

### Task 1: 소리 조각을 `app/realtime_audio.py` 로 옮기기

**Files:**
- Create: `app/realtime_audio.py`
- Modify: `tools/realtime_live.py`
- Test: `tests/test_realtime_audio.py`

**Interfaces:**
- Produces: `SR = 24000`, `REMAINDER`, `FULL_SCALE`; `MicGate(pad_s)` — `sync(playing, now)`, `should_send(now) -> bool`; `PcmAccumulator` — `feed(bytes) -> np.ndarray`; `to_pcm16(block, src_rate) -> bytes`; `resolve_rate(kind, want) -> int`; `StreamSpeaker(src_rate=SR)` — `push(np.ndarray)`, `busy`, `clear()`, `close()`.

- [ ] **Step 1: 실패하는 시험** — `tests/test_realtime_audio.py`
```python
import numpy as np

from app.realtime_audio import SR, MicGate, PcmAccumulator, to_pcm16


def test_16k_를_24k_pcm16_으로_바꾼다():
    pcm = to_pcm16(np.zeros(1280, dtype=np.float32), 16000)   # 80ms @16k
    assert len(pcm) == 1920 * 2                                 # 80ms @24k, 샘플당 2바이트


def test_24k_는_그대로_보낸다():
    back = np.frombuffer(to_pcm16(np.full(240, 0.5, dtype=np.float32), SR), dtype="<i2")
    assert back.size == 240 and back[0] == int(0.5 * 32767)


def test_범위를_넘는_값은_잘라서_보낸다():
    pcm = to_pcm16(np.array([2.0, -2.0], dtype=np.float32), SR)
    assert list(np.frombuffer(pcm, dtype="<i2")) == [32767, -32767]


def test_라이브_루프가_옮긴_것을_쓴다():
    import tools.realtime_live as live
    assert live.MicGate is MicGate and live.PcmAccumulator is PcmAccumulator
```

- [ ] **Step 2: 실패 확인** — `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_realtime_audio.py -q` → FAIL `No module named 'app.realtime_audio'`

- [ ] **Step 3: 구현** — `app/realtime_audio.py`
```python
"""Realtime 대화의 소리 쪽 — 마이크 게이트, pcm 변환, 끊김 없는 재생.

tools/realtime_live.py(09-07 라이브 루프, 노트북 에코 0/11턴)에서 옮겼다. 봇과 도구가
같은 코드를 쓴다 — 도구에서 검증한 것이 봇에서 다르게 돌면 안 된다.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio_player import _resample

SR = 24000             # Realtime pcm16 규격
REMAINDER = 2          # pcm16 한 샘플 = 2 바이트
FULL_SCALE = 32768.0


class MicGate:
    """반이중. 봇이 말하는 동안(+뒤로 pad 만큼) 마이크를 서버에 보내지 않는다.

    서버 VAD 는 우리가 보낸 것만 듣는다. 재생 중에 열어 두면 봇이 제 말에 반응한다.
    pad 는 스피커 잔향과 장치 버퍼 몫 — 08-24 실측으로 정한 0.150s 를 쓴다.
    """

    def __init__(self, pad_s: float = 0.15) -> None:
        self.pad_s = pad_s
        self._playing = False
        self._ended_at: float | None = None

    def playing_started(self, at: float) -> None:
        self._playing = True
        self._ended_at = None

    def playing_ended(self, at: float) -> None:
        self._playing = False
        self._ended_at = at

    def sync(self, playing: bool, now: float) -> None:
        """스피커 상태를 그대로 넘긴다. **가장자리에서만** 시계를 잡는다 —
        매번 잡으면 pad 가 영원히 안 지나 마이크가 다시 안 열린다."""
        if playing:
            self.playing_started(now)
        elif self._playing:
            self.playing_ended(now)

    def should_send(self, now: float) -> bool:
        if self._playing:
            return False
        if self._ended_at is None:
            return True
        return now - self._ended_at >= self.pad_s


class PcmAccumulator:
    """pcm16 리틀엔디언 바이트를 float32(-1..1) 로. 델타는 샘플 중간에서 잘려 온다."""

    def __init__(self) -> None:
        self._tail = b""

    def feed(self, chunk: bytes) -> np.ndarray:
        buf = self._tail + chunk
        n = len(buf) - len(buf) % REMAINDER
        self._tail = buf[n:]
        if not n:
            return np.zeros(0, dtype=np.float32)
        return (np.frombuffer(buf[:n], dtype="<i2").astype(np.float32) / FULL_SCALE)


def to_pcm16(block: np.ndarray, src_rate: int) -> bytes:
    """마이크 float32 → Realtime 이 받는 24k pcm16 바이트. 젯슨 ReSpeaker 는 16k 전용이다."""
    block = np.asarray(block, dtype=np.float32).reshape(-1)
    if src_rate != SR:
        block = _resample(block, src_rate, SR)
    return (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()


def resolve_rate(kind: str, want: int) -> int:
    """장치가 want 를 받으면 그대로, 아니면 장치 기본값.

    젯슨 ReSpeaker 는 16000 전용이고 Realtime 은 24000 규격이라 양방향 리샘플이
    필요하다. 노트북은 보통 24000 을 그대로 받아 리샘플이 0 회다 — 그래서 이 값을
    반드시 찍어 본다. 노트북에서 안 겪은 문제를 젯슨에서 처음 만나면 안 된다.
    """
    import sounddevice as sd

    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    try:
        check(samplerate=want)
        return want
    except Exception:
        dev = sd.default.device
        idx = dev[0 if kind == "input" else 1] if isinstance(dev, (list, tuple)) else dev
        try:
            return int(sd.query_devices(idx, kind)["default_samplerate"])
        except Exception:
            return want


class StreamSpeaker:
    """소켓에서 오는 조각을 끊김 없이 낸다.

    `SoundDeviceSink.play()` 는 호출마다 앞 재생을 끊어서 못 쓴다(조각마다 부르면
    마지막 조각만 들린다). 콜백 스트림 + 대기열로 이어 붙인다. 맞장구·고정 문구도
    같은 대기열에 넣으면 답과 겹치지 않고 이어진다.
    """

    def __init__(self, src_rate: int = SR) -> None:
        import sounddevice as sd

        self.src_rate = src_rate
        self.rate = resolve_rate("output", src_rate)
        self.resampled = self.rate != src_rate
        self._q: deque = deque()
        self._left = np.zeros(0, dtype=np.float32)
        self._stream = sd.OutputStream(samplerate=self.rate, channels=1,
                                       dtype="float32", callback=self._cb)
        self._stream.start()

    def _cb(self, out, frames, time_info, status) -> None:
        need, got = frames, []
        while need > 0:
            if self._left.size == 0:
                if not self._q:
                    break
                self._left = self._q.popleft()
            take = min(need, self._left.size)
            got.append(self._left[:take])
            self._left = self._left[take:]
            need -= take
        block = np.concatenate(got) if got else np.zeros(0, dtype=np.float32)
        if block.size < frames:                       # 남으면 무음으로 채운다
            block = np.concatenate([block, np.zeros(frames - block.size, dtype=np.float32)])
        out[:, 0] = block

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.resampled:
            samples = _resample(samples, self.src_rate, self.rate)
        self._q.append(samples)

    @property
    def busy(self) -> bool:
        return bool(self._q) or self._left.size > 0

    def clear(self) -> None:
        self._q.clear()
        self._left = np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
```

`tools/realtime_live.py`:
1. docstring 바로 아래 `import numpy as np` 부터 `class PcmAccumulator` 정의 끝까지(`REMAINDER`, `FULL_SCALE`, `MicGate`, `PcmAccumulator`)를 이것으로 바꾼다:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.realtime_audio import (FULL_SCALE, REMAINDER, MicGate,  # noqa: E402,F401 — 도구·시험이 이 이름으로 쓴다
                                PcmAccumulator)
```
2. 아래쪽의 `def resolve_rate(...)` 와 `class Speaker:` 정의 전체를 지우고, `from tools.realtime_probe import (...)` 문 바로 다음에:
```python
from app.realtime_audio import StreamSpeaker as Speaker, resolve_rate  # noqa: E402
```
3. 뒤쪽에 이미 있는 `import sys` / `sys.path.insert(...)` / `from pathlib import Path` 중복은 그대로 둬도 된다(무해).

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_realtime_audio.py tests/test_realtime_live.py -q` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/realtime_audio.py tests/test_realtime_audio.py tools/realtime_live.py
git commit -m "refactor(realtime): 라이브 루프의 소리 조각을 app 으로 옮긴다 — 봇과 도구가 같은 코드를 쓴다"
```

---

### Task 2: 프로토콜 메시지와 설정 (`app/realtime_protocol.py`)

**Files:**
- Create: `app/realtime_protocol.py`
- Modify: `tools/realtime_probe.py` (`PRICE`, `cost_usd` → import)
- Modify: `configs/model_paths.yaml` (끝에 `pipeline`, `realtime:`)
- Test: `tests/test_realtime_protocol.py`

**Interfaces:**
- Produces: `URL`, `SR`, `READER`, `PRICE`; `RealtimeConfig`(필드: model, voice, transcribe_model, silence_ms, mic_pad_s, history_turns, filler_after_s, response_timeout_s, wake_listen_s, voice_cache_dir) + `from_dict(d)`; `session_update(cfg, instructions) -> dict`; `history_items(history: list[tuple[str,str]], max_turns) -> list[dict]`; `append_audio(pcm) -> dict`; `respond(instructions=None) -> dict`; `say_exactly(line) -> dict`; `cancel() -> dict`; `event_kind(ev) -> str` ∈ `audio|text|speech_started|speech_stopped|transcript|done|error|other`; `cost_usd(model, usage) -> float|None`; `cached_tokens(usage) -> int`.

- [ ] **Step 1: 실패하는 시험** — `tests/test_realtime_protocol.py`
```python
import base64

from app.realtime_protocol import (RealtimeConfig, append_audio, cached_tokens, cancel,
                                   cost_usd, event_kind, history_items, respond,
                                   say_exactly, session_update)


def test_설정은_빠진_값을_기본으로_채우고_모르는_키는_버린다():
    c = RealtimeConfig.from_dict({"voice": "cedar", "없는키": 1})
    assert c.voice == "cedar" and c.model == "gpt-realtime-mini"
    assert c.silence_ms == 1200 and c.response_timeout_s == 8.0
    assert RealtimeConfig.from_dict(None).voice == "marin"


def test_세션은_자동으로_답하지_않는다():
    # 🔴 서버가 먼저 답하면 "노래 틀어줘" 를 우리가 보기 전에 "틀어줄게" 가 나간다.
    s = session_update(RealtimeConfig(), "지시")["session"]
    td = s["audio"]["input"]["turn_detection"]
    assert td["create_response"] is False and td["interrupt_response"] is False
    assert td["silence_duration_ms"] == 1200
    assert s["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"
    assert s["audio"]["output"]["voice"] == "marin" and s["instructions"] == "지시"


def test_이력은_최근_턴만_아이_말과_봇_답으로_넣는다():
    items = history_items([(f"아이{i}", f"봇{i}") for i in range(10)], max_turns=2)
    assert [it["item"]["content"][0]["text"] for it in items] == ["아이8", "봇8", "아이9", "봇9"]
    assert [it["item"]["role"] for it in items] == ["user", "assistant", "user", "assistant"]
    assert items[1]["item"]["content"][0]["type"] == "output_text"


def test_오디오_추가는_base64():
    m = append_audio(b"\x01\x02")
    assert m["type"] == "input_audio_buffer.append" and base64.b64decode(m["audio"]) == b"\x01\x02"


def test_놀이_답은_그_답에만_지시를_붙인다():
    assert respond() == {"type": "response.create"}
    assert respond("상황")["response"]["instructions"] == "상황"


def test_그대로_말하기는_대화_기록_밖이다():
    r = say_exactly("상어가족 틀어 줄게!")["response"]
    assert r["conversation"] == "none"
    assert "상어가족 틀어 줄게!" in r["input"][0]["content"][0]["text"]


def test_취소():
    assert cancel() == {"type": "response.cancel"}


def test_이벤트_분류는_베타_이름도_받는다():
    assert event_kind({"type": "response.output_audio.delta"}) == "audio"
    assert event_kind({"type": "response.audio.delta"}) == "audio"
    assert event_kind({"type": "response.output_audio_transcript.delta"}) == "text"
    assert event_kind({"type": "conversation.item.input_audio_transcription.completed"}) == "transcript"
    assert event_kind({"type": "input_audio_buffer.speech_stopped"}) == "speech_stopped"
    assert event_kind({"type": "response.done"}) == "done"
    assert event_kind({"type": "error"}) == "error"
    assert event_kind({"type": "session.updated"}) == "other"


def test_비용과_캐시_토큰():
    usage = {"input_token_details": {"audio_tokens": 100, "text_tokens": 0, "cached_tokens": 80,
                                     "cached_tokens_details": {"audio_tokens": 80}},
             "output_token_details": {"audio_tokens": 50, "text_tokens": 0}}
    assert abs(cost_usd("gpt-realtime-mini", usage) - (20 * 10.0 + 80 * 0.30 + 50 * 20.0) / 1e6) < 1e-12
    assert cached_tokens(usage) == 80
    assert cost_usd("모르는모델", usage) is None
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_realtime_protocol.py -q` → FAIL `No module named 'app.realtime_protocol'`

- [ ] **Step 3: 구현** — `app/realtime_protocol.py`
```python
"""Realtime 소켓에 오가는 것 — 전부 순수 함수라 소켓 없이 시험한다.

GA 스키마(2026-09 확인, tools/realtime_probe.py 에서 실측). 베타의 평평한
input_audio_format/turn_detection 은 거부된다.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, fields

URL = "wss://api.openai.com/v1/realtime?model={model}"
SR = 24000

AUDIO_DELTA = ("response.output_audio.delta", "response.audio.delta")
TEXT_DELTA = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")
TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"

READER = ("너는 낭독자다. 사용자가 준 문장을 **글자 그대로** 한 번만 읽어라. "
          "덧붙이거나 줄이거나 바꾸지 마라.")

# USD / 1M tokens. 2026-09-02 developers.openai.com/api/docs/pricing 확인.
PRICE = {
    "gpt-realtime":      {"audio_in": 32.0, "cached": 0.40, "audio_out": 64.0,
                          "text_in": 4.00, "text_out": 16.0},
    "gpt-realtime-mini": {"audio_in": 10.0, "cached": 0.30, "audio_out": 20.0,
                          "text_in": 0.60, "text_out": 2.40},
}


@dataclass
class RealtimeConfig:
    model: str = "gpt-realtime-mini"
    voice: str = "marin"                          # 09-07 블라인드 선택
    transcribe_model: str = "gpt-4o-transcribe"   # whisper-1 은 아동 발화 39% 빈 문자열
    silence_ms: int = 1200                        # 로컬 VAD 꼬리와 같게 — 비교 가능
    mic_pad_s: float = 0.15                       # 08-24 실측
    history_turns: int = 6                        # agent.MAX_HISTORY_TURNS 와 같게
    filler_after_s: float = 0.7                   # 이만큼 답 소리가 없으면 맞장구
    response_timeout_s: float = 8.0
    wake_listen_s: float = 8.0                    # 노래 중 호출 뒤 이만큼 말이 없으면 노래로
    voice_cache_dir: str = "~/.cache/jaeha_voice"

    @classmethod
    def from_dict(cls, d: dict | None) -> "RealtimeConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def session_update(cfg: RealtimeConfig, instructions: str) -> dict:
    return {"type": "session.update", "session": {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SR},
                "transcription": {"model": cfg.transcribe_model, "language": "ko"},
                # 🔴 create_response=false: 받아 적기를 보고 우리가 답을 요청한다.
                #    서버가 먼저 답하면 노래·놀이 명령을 가로챌 수 없다.
                "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                   "prefix_padding_ms": 300,
                                   "silence_duration_ms": int(cfg.silence_ms),
                                   "create_response": False,
                                   "interrupt_response": False},
            },
            "output": {"format": {"type": "audio/pcm", "rate": SR}, "voice": cfg.voice},
        },
    }}


def _message(role: str, text: str) -> dict:
    kind = "input_text" if role == "user" else "output_text"
    return {"type": "conversation.item.create",
            "item": {"type": "message", "role": role,
                     "content": [{"type": kind, "text": text}]}}


def history_items(history: list[tuple[str, str]], max_turns: int) -> list[dict]:
    if max_turns <= 0:
        return []
    out: list[dict] = []
    for child, bot in history[-max_turns:]:
        out.append(_message("user", child))
        out.append(_message("assistant", bot))
    return out


def append_audio(pcm: bytes) -> dict:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}


def respond(instructions: str | None = None) -> dict:
    if instructions is None:
        return {"type": "response.create"}
    return {"type": "response.create", "response": {"instructions": instructions}}


def say_exactly(line: str) -> dict:
    """대화 기록 밖에서 문장 하나를 그대로 읽게 한다(노래 안내·고정 문구 만들기)."""
    return {"type": "response.create", "response": {
        "conversation": "none",
        "instructions": READER,
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": f"정확히 이렇게만 말해: {line}"}]}],
    }}


def cancel() -> dict:
    return {"type": "response.cancel"}


def event_kind(ev: dict) -> str:
    t = ev.get("type", "")
    if t in AUDIO_DELTA:
        return "audio"
    if t in TEXT_DELTA:
        return "text"
    if t == TRANSCRIPT_DONE:
        return "transcript"
    return {"input_audio_buffer.speech_started": "speech_started",
            "input_audio_buffer.speech_stopped": "speech_stopped",
            "response.done": "done", "error": "error"}.get(t, "other")


def cost_usd(model: str, usage: dict) -> float | None:
    """usage 를 그대로 값으로. 모양이 낯설면 None — 추측해서 채우지 않는다."""
    p = PRICE.get(model)
    if not p or not usage:
        return None
    ind = usage.get("input_token_details", {}) or {}
    outd = usage.get("output_token_details", {}) or {}
    cached = (ind.get("cached_tokens_details", {}) or {}).get("audio_tokens", 0)
    a_in = max(0, ind.get("audio_tokens", 0) - cached)
    parts = (a_in * p["audio_in"], cached * p["cached"],
             ind.get("text_tokens", 0) * p["text_in"],
             outd.get("audio_tokens", 0) * p["audio_out"],
             outd.get("text_tokens", 0) * p["text_out"])
    return sum(parts) / 1e6 if any(parts) else None


def cached_tokens(usage: dict) -> int:
    return int(((usage or {}).get("input_token_details", {}) or {}).get("cached_tokens", 0) or 0)
```

`tools/realtime_probe.py`: `PRICE = {...}` 블록과 `def cost_usd(...)` 함수 정의를 지우고, `BASE = ...` 아래(파일에 `sys.path.insert(0, str(BASE))` 가 없으면 그것부터) 넣는다:
```python
from app.realtime_protocol import PRICE, cost_usd  # noqa: E402,F401 — 옮겼다. 도구·시험이 이 이름으로 쓴다
```

`configs/model_paths.yaml` 끝에 추가:
```yaml

# 2026-09-16 전면 API(Realtime) 대화 경로 — docs/superpowers/specs/2026-09-16-realtime-pipeline-design.md
#   local    : app.main (호출어 → faster-whisper → gpt → Supertonic). 보고서가 서술하는 경로.
#   realtime : app.main_realtime (호출어 → OpenAI Realtime). 젯슨 local.yaml 에서만 켠다.
pipeline: local
realtime:
  model: gpt-realtime-mini          # 40턴 월 21,200원(gpt-realtime 73,300원) — report 7.2
  voice: marin                      # 09-07 블라인드
  transcribe_model: gpt-4o-transcribe   # whisper-1 은 아동 발화 39% 빈 문자열
  silence_ms: 1200
  mic_pad_s: 0.15
  history_turns: 6
  filler_after_s: 0.7
  response_timeout_s: 8
  wake_listen_s: 8
  voice_cache_dir: ~/.cache/jaeha_voice
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_realtime_protocol.py tests/test_realtime_probe.py -q` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/realtime_protocol.py tests/test_realtime_protocol.py tools/realtime_probe.py configs/model_paths.yaml
git commit -m "feat(realtime): 소켓 메시지·설정을 순수 함수로 — 서버가 먼저 답하지 않게(create_response false)"
```

---

### Task 3: 턴 판단 (`app/realtime_turn.py`) + 놀이 지시 꺼내기

**Files:**
- Create: `app/realtime_turn.py`
- Modify: `app/education_modes.py` (`GameManager`)
- Test: `tests/test_realtime_turn.py`

**Interfaces:**
- Consumes: `app.wake.is_sleep_command(text, words)`, `MusicController.handle(text) -> MusicReply|None`, `app.agent._RENDER_SYSTEM`, `app.music.MUSIC_FAILED`, `settings.prompts["recovery"]`
- Produces: `Route(kind, say="", instructions=None, require=[], music=None)` kind ∈ `empty|sleep|music|game|game_start|chat`; `route(text, *, music, games, sleep_words) -> Route`; `missing_required(reply, require) -> list[str]`; `PHRASES` 키 `ready|wake|sleep|recovery|music_failed|lost`; `GameManager.maybe_start_beat(text) -> dict|None`, `GameManager.handle_beat(text) -> dict|None`

- [ ] **Step 1: 실패하는 시험** — `tests/test_realtime_turn.py`
```python
from app.education_modes import GameManager
from app.music import MusicReply
from app.realtime_turn import PHRASES, missing_required, route


class FakeMusic:
    def __init__(self, reply=None):
        self.reply, self.seen = reply, []

    def handle(self, text):
        self.seen.append(text)
        return self.reply


def _route(text, music=None, games=None):
    return route(text, music=music, games=games or GameManager(render=None), sleep_words=None)


def test_빈_글자는_아무것도_안_한다():
    assert _route("  ").kind == "empty"


def test_잠들기가_노래보다_먼저다():
    m = FakeMusic(MusicReply("노래 껐어!"))
    assert _route("바이바이", music=m).kind == "sleep"
    assert m.seen == []


def test_노래_명령은_안내_문장을_그대로_말한다():
    r = _route("상어가족 틀어줘", music=FakeMusic(MusicReply("상어가족 틀어 줄게!", standby=True)))
    assert r.kind == "music" and r.say == "상어가족 틀어 줄게!" and r.music.standby


def test_노래가_꺼져_있으면_노래로_안_간다():
    assert _route("상어가족 틀어줘", music=None).kind == "chat"


def test_놀이_시작은_지시를_붙인다():
    r = _route("동물 소리 놀이 하자")
    assert r.kind == "game_start" and "상황:" in r.instructions and r.say


def test_놀이_중에는_상태기계가_받는다():
    g = GameManager(render=None)
    route("동물 소리 놀이 하자", music=None, games=g, sleep_words=None)
    assert route("야옹", music=None, games=g, sleep_words=None).kind == "game"


def test_나머지는_자유대화():
    assert _route("오늘 뭐 했어").kind == "chat"


def test_빠진_필수_낱말():
    assert missing_required("좋아! 고양이는 야옹?", ["고양이", "강아지"]) == ["강아지"]
    assert missing_required("아무거나", []) == []


def test_고정_문구는_로컬_봇과_같다():
    import app.main as m
    from app.music import MUSIC_FAILED
    assert PHRASES["ready"] == m.READY_ASLEEP
    assert PHRASES["wake"] == m.WAKE_GREETING
    assert PHRASES["sleep"] == m.SLEEP_MSG
    assert PHRASES["recovery"] == m.SAFE_RECOVERY
    assert PHRASES["music_failed"] == MUSIC_FAILED
    assert PHRASES["lost"]


def test_기존_놀이_함수_동작은_그대로다():
    g = GameManager(render=None)
    assert isinstance(g.maybe_start("동물 소리 놀이 하자"), str)
    assert isinstance(g.handle("야옹"), str)
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_realtime_turn.py -q` → FAIL `No module named 'app.realtime_turn'`

- [ ] **Step 3: 구현**

`app/education_modes.py` — `GameManager.maybe_start` 와 `handle` 을 이것으로 바꾼다(반환값·상태 변화 그대로, `_voice(None)` 은 None):
```python
    def maybe_start_beat(self, text: str) -> dict | None:
        """시작 트리거면 놀이를 시작하고 첫 **지시(beat)** 를 돌려준다. 아니면 None.

        Realtime 경로는 지시를 문장으로 바꾸지 않고 모델에 그대로 넘긴다(2026-09-16).
        """
        if self.active is not None:
            return None
        kind = match_trigger(text)
        if not kind:
            return None
        self.active = AnimalSoundGame() if kind == "animal" else RepeatWordGame()
        return self.active.start()

    def maybe_start(self, text: str) -> str | None:
        """시작 트리거면 놀이를 시작하고 첫 멘트를 돌려준다. 아니면 None."""
        return self._voice(self.maybe_start_beat(text))

    def handle_beat(self, text: str) -> dict | None:
        """진행 중 놀이가 있으면 한 턴 처리해 **지시(beat)** 를 돌려준다. 없으면 None."""
        if self.active is None:
            return None
        beat = self.active.step(text)
        if self.active.done:
            self.active = None
        return beat

    def handle(self, text: str) -> str | None:
        """진행 중 놀이가 있으면 한 턴 처리. 없으면 None(→ 평소 대화로)."""
        return self._voice(self.handle_beat(text))
```

`app/realtime_turn.py`:
```python
"""받아 적은 아이 말 → 이번 턴에 무엇을 할지. 순수 함수라 소켓 없이 시험한다.

순서는 로컬 봇(app/main.py)과 같다: 잠들기 → 노래 → 놀이 → 자유대화.
노래가 놀이보다 먼저인 이유: 모델은 노래를 못 튼다 — 거기까지 가면 "틀어줄게" 라는
빈 약속만 나간다(reports/bench/README.md 5절).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import _RENDER_SYSTEM
from .config import settings
from .music import MUSIC_FAILED
from .wake import is_sleep_command

# 🔴 로컬 봇(app/main.py)과 같은 문구. app.main 을 import 하면 STT·TTS 모듈이 따라
#    올라와 전면 API 경로에 필요 없는 무게를 싣는다. 대신 시험이 두 값이 같은지 본다.
PHRASES = {
    "ready": "티드 준비됐어. 부르면 나올게!",
    "wake": "응! 왜 불렀어? 나랑 놀자!",
    "sleep": "그래, 또 부르면 올게! 안녕~",
    "recovery": settings.prompts.get("recovery") or "어? 잘 못 들었어. 다시 말해줄래?",
    "music_failed": MUSIC_FAILED,
    "lost": "어? 잠깐 쉬었다 올게. 다시 불러 줘!",
}


@dataclass
class Route:
    kind: str                                  # empty|sleep|music|game|game_start|chat
    say: str = ""                              # 그대로 말할 문장(노래 안내 / 놀이 템플릿)
    instructions: str | None = None            # 놀이 턴의 지시(그 답에만 붙는다)
    require: list[str] = field(default_factory=list)
    music: object = None                       # MusicReply


def _game(kind: str, beat: dict) -> Route:
    if beat.get("instruction"):
        return Route(kind, say=beat["fallback"],
                     instructions=f"{_RENDER_SYSTEM}\n\n상황: {beat['instruction']}",
                     require=list(beat.get("require", [])))
    return Route(kind, say=beat["fallback"])


def route(text: str, *, music, games, sleep_words: list[str] | None) -> Route:
    text = (text or "").strip()
    if not text:
        return Route("empty")
    if is_sleep_command(text, sleep_words):
        return Route("sleep")
    mr = music.handle(text) if music is not None else None
    if mr is not None:
        return Route("music", say=mr.text, music=mr)
    beat = games.handle_beat(text)
    if beat is not None:
        return _game("game", beat)
    beat = games.maybe_start_beat(text)
    if beat is not None:
        return _game("game_start", beat)
    return Route("chat")


def missing_required(reply: str, require: list[str]) -> list[str]:
    return [t for t in require if t not in (reply or "")]
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_realtime_turn.py -q` 그리고 놀이 기존 시험 `... -m pytest tests -q -k "education or game"` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/realtime_turn.py app/education_modes.py tests/test_realtime_turn.py
git commit -m "feat(realtime): 받아 적은 말로 턴을 나눈다 — 잠들기·노래·놀이·자유대화, 놀이 지시는 문장으로 안 바꾸고 넘긴다"
```

---

### Task 4: 소켓 하나 (`app/realtime_session.py`)

**Files:**
- Create: `app/realtime_session.py`
- Test: `tests/test_realtime_session.py`

**Interfaces:**
- Consumes: `realtime_protocol.{URL, READER, RealtimeConfig, session_update, history_items, say_exactly, event_kind}`, `realtime_audio.PcmAccumulator`
- Produces: `ConnectionLost(Exception)`; `RealtimeSession(cfg, api_key, connect=None)` — `async open(instructions, history)`, `async send(msg)`, `events()` async iterator of dict(끝나면 `ConnectionLost`), `async close()`; `async synthesize(cfg, api_key, line, connect=None) -> np.ndarray`(24k float32, 소리 없으면 `RuntimeError`). `connect` 는 `async connect(url, headers) -> ws`.

- [ ] **Step 1: 실패하는 시험** — `tests/test_realtime_session.py`
```python
import asyncio
import base64
import json

import numpy as np
import pytest

from app.realtime_protocol import RealtimeConfig
from app.realtime_session import ConnectionLost, RealtimeSession, synthesize


class FakeWS:
    def __init__(self, incoming=(), fail_send=False):
        self.sent, self.closed = [], False
        self._in = list(incoming)
        self.fail_send = fail_send

    async def send(self, raw):
        if self.fail_send:
            raise OSError("끊김")
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._in:
            raise StopAsyncIteration
        return json.dumps(self._in.pop(0))


def _connect(ws, seen):
    async def connect(url, headers):
        seen.update(url=url, headers=headers)
        return ws
    return connect


def test_열면_세션설정과_이력을_보낸다():
    ws, seen = FakeWS(), {}
    s = RealtimeSession(RealtimeConfig(), "키", connect=_connect(ws, seen))
    asyncio.run(s.open("지시", [("안녕", "안녕!")]))
    assert "gpt-realtime-mini" in seen["url"] and seen["headers"]["Authorization"] == "Bearer 키"
    assert [m["type"] for m in ws.sent] == ["session.update", "conversation.item.create",
                                           "conversation.item.create"]


def test_보내기_실패는_끊김이다():
    s = RealtimeSession(RealtimeConfig(), "키", connect=_connect(FakeWS(fail_send=True), {}))
    with pytest.raises(ConnectionLost):
        asyncio.run(s.open("지시", []))


def test_이벤트가_끝나면_끊김이다():
    s = RealtimeSession(RealtimeConfig(), "키", connect=_connect(FakeWS([{"type": "session.updated"}]), {}))

    async def go():
        await s.open("지시", [])
        got = []
        with pytest.raises(ConnectionLost):
            async for ev in s.events():
                got.append(ev["type"])
        return got

    assert asyncio.run(go()) == ["session.updated"]


def test_문장_하나를_소리로_만든다():
    pcm = (np.full(480, 0.25) * 32767).astype("<i2").tobytes()
    ws = FakeWS([{"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()},
                 {"type": "response.done"}])
    audio = asyncio.run(synthesize(RealtimeConfig(), "키", "안녕", connect=_connect(ws, {})))
    assert audio.dtype == np.float32 and audio.size == 480
    assert ws.sent[-1]["response"]["conversation"] == "none" and ws.closed


def test_소리가_안_오면_실패다():
    with pytest.raises(RuntimeError):
        asyncio.run(synthesize(RealtimeConfig(), "키", "안녕",
                               connect=_connect(FakeWS([{"type": "response.done"}]), {})))
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_realtime_session.py -q` → FAIL `No module named 'app.realtime_session'`

- [ ] **Step 3: 구현** — `app/realtime_session.py`
```python
"""Realtime 연결 하나. 소켓 라이브러리는 여기에만 닿는다 — 나머지는 dict 만 주고받는다."""
from __future__ import annotations

import base64
import json
import logging

import numpy as np

from .realtime_audio import PcmAccumulator
from .realtime_protocol import (READER, URL, RealtimeConfig, event_kind, history_items,
                                say_exactly, session_update)

log = logging.getLogger("jaeha_bot.realtime")


class ConnectionLost(Exception):
    """소켓이 닫혔거나 보내기가 실패했다. 대화 구간은 이걸 받으면 대기로 간다."""


async def _ws_connect(url: str, headers: dict):
    import websockets
    return await websockets.connect(url, additional_headers=headers, max_size=None)


class RealtimeSession:
    def __init__(self, cfg: RealtimeConfig, api_key: str, connect=None) -> None:
        self.cfg = cfg
        self._key = api_key
        self._connect = connect or _ws_connect
        self._ws = None

    async def open(self, instructions: str, history: list[tuple[str, str]]) -> None:
        try:
            self._ws = await self._connect(URL.format(model=self.cfg.model),
                                           {"Authorization": f"Bearer {self._key}"})
        except Exception as e:
            raise ConnectionLost(f"연결 실패: {type(e).__name__}: {e}") from e
        await self.send(session_update(self.cfg, instructions))
        for item in history_items(history, self.cfg.history_turns):
            await self.send(item)

    async def send(self, msg: dict) -> None:
        if self._ws is None:
            raise ConnectionLost("열리지 않았다")
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            raise ConnectionLost(f"보내기 실패: {type(e).__name__}: {e}") from e

    async def events(self):
        if self._ws is None:
            raise ConnectionLost("열리지 않았다")
        try:
            async for raw in self._ws:
                yield json.loads(raw)
        except Exception as e:
            raise ConnectionLost(f"받기 실패: {type(e).__name__}: {e}") from e
        raise ConnectionLost("서버가 연결을 닫았다")

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass


async def synthesize(cfg: RealtimeConfig, api_key: str, line: str, connect=None) -> np.ndarray:
    """문장 하나를 marin 목소리 24k float32 로. 고정 문구 캐시를 만들 때 쓴다."""
    s = RealtimeSession(cfg, api_key, connect=connect)
    acc, chunks = PcmAccumulator(), []
    try:
        await s.open(READER, [])
        await s.send(say_exactly(line))
        try:
            async for ev in s.events():
                k = event_kind(ev)
                if k == "audio":
                    chunks.append(acc.feed(base64.b64decode(ev["delta"])))
                elif k == "error":
                    raise RuntimeError(f"만들기 실패: {ev.get('error')}")
                elif k == "done":
                    break
        except ConnectionLost:
            pass
    finally:
        await s.close()
    total = sum(c.size for c in chunks)
    if not total:
        raise RuntimeError(f"소리가 안 왔다: {line}")
    return np.concatenate(chunks).astype(np.float32)
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_realtime_session.py -q` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/realtime_session.py tests/test_realtime_session.py
git commit -m "feat(realtime): 소켓 하나를 감싼다 — 끊김은 한 가지 예외로, 고정 문구는 대화 기록 밖에서 만든다"
```

---

### Task 5: 고정 문구 캐시 (`app/voice_cache.py`)

**Files:**
- Create: `app/voice_cache.py`
- Test: `tests/test_voice_cache.py`

**Interfaces:**
- Produces: `VoiceCache(cache_dir, voice, model)` — `key(phrase) -> str`, `path(phrase) -> Path`, `ensure(phrases, synth: Callable[[str], np.ndarray]) -> int`(새로 만든 개수), `get(phrase) -> np.ndarray | None`(24k float32)

- [ ] **Step 1: 실패하는 시험** — `tests/test_voice_cache.py`
```python
import numpy as np

from app.voice_cache import VoiceCache


def test_목소리나_모델이_바뀌면_키가_바뀐다():
    a = VoiceCache("x", "marin", "m").key("안녕")
    assert a != VoiceCache("x", "cedar", "m").key("안녕")
    assert a != VoiceCache("x", "marin", "n").key("안녕")
    assert a == VoiceCache("y", "marin", "m").key("안녕")


def test_없는_것만_만들고_읽는다(tmp_path):
    made = []

    def synth(p):
        made.append(p)
        return np.full(2400, 0.1, dtype=np.float32)

    c = VoiceCache(tmp_path, "marin", "m")
    assert c.ensure(["안녕", "잘가"], synth) == 2
    assert c.ensure(["안녕", "잘가"], synth) == 0
    assert made == ["안녕", "잘가"]
    got = VoiceCache(tmp_path, "marin", "m").get("안녕")
    assert got.dtype == np.float32 and got.size == 2400


def test_만들다_실패하면_건너뛴다(tmp_path):
    def synth(p):
        if p == "나쁨":
            raise RuntimeError("네트워크")
        return np.zeros(10, dtype=np.float32)

    c = VoiceCache(tmp_path, "marin", "m")
    assert c.ensure(["나쁨", "좋음"], synth) == 1
    assert c.get("나쁨") is None and c.get("좋음") is not None
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_voice_cache.py -q` → FAIL `No module named 'app.voice_cache'`

- [ ] **Step 3: 구현** — `app/voice_cache.py`
```python
"""고정 문구 → marin 목소리 wav. 기동 때 빠진 것만 한 번 만든다.

키에 목소리·모델을 넣는다 — 목소리를 갈아탔을 때 고정 문구만 옛 목소리로 겉도는
사고를 구조로 막는다(filler.FillerBank.cache_key 와 같은 원칙).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .realtime_protocol import SR

log = logging.getLogger("jaeha_bot.voice_cache")


class VoiceCache:
    def __init__(self, cache_dir, voice: str, model: str) -> None:
        self.dir = Path(cache_dir).expanduser()
        self.voice, self.model = voice, model
        self._mem: dict[str, np.ndarray] = {}

    def key(self, phrase: str) -> str:
        return hashlib.sha1(f"{self.voice}|{self.model}|{phrase}".encode("utf-8")).hexdigest()[:16]

    def path(self, phrase: str) -> Path:
        return self.dir / f"{self.key(phrase)}.wav"

    def ensure(self, phrases: Iterable[str], synth: Callable[[str], np.ndarray]) -> int:
        import soundfile as sf

        self.dir.mkdir(parents=True, exist_ok=True)
        made = 0
        for p in phrases:
            if not p or self.path(p).exists():
                continue
            try:
                audio = np.asarray(synth(p), dtype=np.float32).reshape(-1)
                sf.write(self.path(p), audio, SR)
                made += 1
                log.info("고정 문구 만듦(%.1fs): %s", audio.size / SR, p)
            except Exception as e:
                log.warning("고정 문구를 못 만들었다(건너뜀) %s: %s: %s", p, type(e).__name__, e)
        return made

    def get(self, phrase: str) -> np.ndarray | None:
        if phrase in self._mem:
            return self._mem[phrase]
        path = self.path(phrase)
        if not path.exists():
            return None
        import soundfile as sf
        audio, rate = sf.read(path, dtype="float32")
        if rate != SR:
            from .audio_player import _resample
            audio = _resample(audio, rate, SR)
        self._mem[phrase] = audio
        return audio
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_voice_cache.py -q` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/voice_cache.py tests/test_voice_cache.py
git commit -m "feat(realtime): 고정 문구를 marin wav 로 캐시한다 — 목소리·모델이 키에 들어간다"
```

---

### Task 6: 턴 계측 (`MetricsLogger.record_realtime_turn`)

**Files:**
- Modify: `app/metrics.py` (메서드 추가만, `summary` 바로 위)
- Test: `tests/test_metrics_realtime.py`

**Interfaces:**
- Produces: `MetricsLogger.record_realtime_turn(*, kind, perceived_s, transcribe_s, respond_first_s, filler, reply, child_text, cost_usd, cached_tokens, safety, game_missing) -> None` — 파일에 `event: "rt_turn"`, `metric_ver: "rt1"` 한 줄. `self.samples` 에는 안 넣는다(`summary()` 가 로컬 필드를 기대).

- [ ] **Step 1: 실패하는 시험** — `tests/test_metrics_realtime.py`
```python
import json

from app.metrics import MetricsLogger


def test_realtime_턴은_따로_한_줄로_남는다(tmp_path):
    m = MetricsLogger(enabled=True, tag="pc-rt", log_dir=str(tmp_path))
    try:
        m.record_realtime_turn(kind="chat", perceived_s=1.23456, transcribe_s=0.4,
                               respond_first_s=0.5, filler=False, reply="응!",
                               child_text="안녕", cost_usd=0.0015, cached_tokens=10,
                               safety=[], game_missing=[])
    finally:
        m._sampler.stop()
    rec = json.loads(m.path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["event"] == "rt_turn" and rec["metric_ver"] == "rt1" and rec["tag"] == "pc-rt"
    assert rec["perceived_s"] == 1.235 and rec["kind"] == "chat" and rec["turn"] == 1
    assert m.samples == []
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_metrics_realtime.py -q` → FAIL `AttributeError: ... 'record_realtime_turn'`

- [ ] **Step 3: 구현** — `app/metrics.py` `def summary` 바로 위:
```python
    def record_realtime_turn(self, *, kind: str, perceived_s: float | None,
                             transcribe_s: float | None, respond_first_s: float | None,
                             filler: bool, reply: str, child_text: str,
                             cost_usd: float | None, cached_tokens: int,
                             safety: list[str], game_missing: list[str]) -> None:
        """전면 API(Realtime) 한 턴. 로컬 필드와 이름이 달라 samples 에는 안 넣는다.

        perceived_s     : 서버가 말끝을 잡은 순간 → 첫 답 소리. 로컬 resp_felt_s 와 비교
                          (둘 다 무음 대기 1.2초를 포함한다).
        transcribe_s    : 말끝 → 받아 적기 완료. 턴을 우리가 쥐는 값(spec R1).
        respond_first_s : response.create → 첫 오디오 조각.
        """
        if not self.enabled:
            return
        self.turn += 1

        def r(x):
            return None if x is None else round(float(x), 3)

        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "tag": self.tag,
               "event": "rt_turn", "metric_ver": "rt1", "turn": self.turn, "kind": kind,
               "perceived_s": r(perceived_s), "transcribe_s": r(transcribe_s),
               "respond_first_s": r(respond_first_s), "filler": bool(filler),
               "cost_usd": None if cost_usd is None else round(cost_usd, 6),
               "cached_tokens": int(cached_tokens), "safety": list(safety),
               "game_missing": list(game_missing), "reply_len": len(reply or ""),
               "child_text": child_text, "reply": reply,
               **_sys_stats_now(), **self._sampler.pop()}
        self._write(rec)
        log.info("[계측] rt턴%d %s 체감 %s (받아적기 %s / 요청→첫소리 %s)%s | $%s | RSS %sMB",
                 self.turn, kind, rec["perceived_s"], rec["transcribe_s"],
                 rec["respond_first_s"], " +맞장구" if filler else "",
                 rec["cost_usd"], rec.get("rss_mb"))
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_metrics_realtime.py tests/test_latency_metrics.py tests/test_metrics_expansion.py -q` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/metrics.py tests/test_metrics_realtime.py
git commit -m "feat(metrics): Realtime 턴을 따로 한 줄로 — 말끝→첫소리·받아적기 지연·비용·캐시"
```

---

### Task 7: 깨어 있는 한 구간 (`app/realtime_conversation.py`)

**Files:**
- Create: `app/realtime_conversation.py`
- Test: `tests/test_realtime_conversation.py`

**Interfaces:**
- Consumes: `RealtimeSession`(open/send/events/close), `ConnectionLost`, `realtime_protocol.{append_audio, respond, say_exactly, cancel, event_kind, cost_usd, cached_tokens}`, `realtime_audio.{MicGate, PcmAccumulator, to_pcm16}`, `realtime_turn.{route, missing_required, PHRASES, Route}`, `cache.get(phrase)`, `safety.check_reply(reply, child_text=)`, `metrics.record_realtime_turn`
- Produces: `Conversation(*, session, mic: asyncio.Queue, speaker, cache, cfg, music, games, sleep_words, instructions, history: list, metrics=None, sleep_timeout=30.0, filler_phrases=(), clock=time.monotonic, tick_s=0.05)`, `async run(*, preroll: tuple[np.ndarray, int] | None = None, greet=True, woke_during_music=False) -> str` ∈ `sleep|idle|music|music_resumed|lost`. mic 큐 항목 `(frame, rate)`. speaker 는 `push(np.ndarray)` + `busy`.

- [ ] **Step 1: 실패하는 시험** — `tests/test_realtime_conversation.py`
```python
import asyncio
import base64

import numpy as np

from app.education_modes import GameManager
from app.music import MusicReply
from app.realtime_conversation import Conversation
from app.realtime_protocol import RealtimeConfig
from app.realtime_session import ConnectionLost
from app.realtime_turn import PHRASES


class FakeSession:
    def __init__(self):
        self.sent, self.closed, self.opened = [], False, None
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def open(self, instructions, history):
        self.opened = (instructions, list(history))

    async def send(self, msg):
        self.sent.append(msg)

    async def events(self):
        while True:
            ev = await self.inbox.get()
            if ev is None:
                raise ConnectionLost("끝")
            yield ev

    async def close(self):
        self.closed = True

    def requests(self):
        return [m for m in self.sent if m["type"] == "response.create"]


class FakeSpeaker:
    busy = False

    def __init__(self):
        self.pushed = []

    def push(self, samples):
        self.pushed.append(np.asarray(samples))


class FakeCache:
    def __init__(self):
        self.asked = []

    def get(self, phrase):
        self.asked.append(phrase)
        return np.full(10, 0.1, dtype=np.float32)


def _cfg():
    return RealtimeConfig(filler_after_s=0.05, response_timeout_s=0.3, wake_listen_s=0.2)


def _audio_delta():
    pcm = (np.full(240, 0.2) * 32767).astype("<i2").tobytes()
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}


def _transcript(text):
    return {"type": "conversation.item.input_audio_transcription.completed", "transcript": text}


def _conv(session, *, music=None, cache=None, speaker=None, sleep_timeout=5.0, history=None):
    return Conversation(session=session, mic=asyncio.Queue(), speaker=speaker or FakeSpeaker(),
                        cache=cache or FakeCache(), cfg=_cfg(), music=music,
                        games=GameManager(render=None), sleep_words=None, instructions="지시",
                        history=history if history is not None else [],
                        sleep_timeout=sleep_timeout, filler_phrases=["음..."], tick_s=0.01)


async def _feed(session, events, gap=0.02):
    for ev in events:
        await asyncio.sleep(gap)
        await session.inbox.put(ev)


def test_자유대화는_답을_요청하고_이력에_남긴다():
    async def go():
        s, history = FakeSession(), []
        c = _conv(s, history=history, sleep_timeout=0.3)
        asyncio.create_task(_feed(s, [
            {"type": "input_audio_buffer.speech_stopped"}, _transcript("안녕"), _audio_delta(),
            {"type": "response.output_audio_transcript.delta", "delta": "안녕!"},
            {"type": "response.done", "response": {"usage": {}}}]))
        return s, history, await c.run(greet=False)

    s, history, reason = asyncio.run(go())
    assert s.requests() == [{"type": "response.create"}]
    assert history == [("안녕", "안녕!")] and reason == "idle" and s.closed


def test_잠들기는_문구를_틀고_끝난다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache)
        asyncio.create_task(_feed(s, [_transcript("바이바이")]))
        return await c.run(greet=False), s, cache

    reason, s, cache = asyncio.run(go())
    assert reason == "sleep" and s.requests() == [] and PHRASES["sleep"] in cache.asked


def test_노래는_안내를_그대로_말하고_틀고_대기로():
    played = []

    class Music:
        def handle(self, text):
            return MusicReply("상어가족 틀어 줄게!", action=lambda: played.append(1) or True,
                              standby=True)

    async def go():
        s = FakeSession()
        c = _conv(s, music=Music())
        asyncio.create_task(_feed(s, [_transcript("상어가족 틀어줘"), _audio_delta(),
                                      {"type": "response.done", "response": {}}]))
        return await c.run(greet=False), s

    reason, s = asyncio.run(go())
    req = s.requests()[0]["response"]
    assert req["conversation"] == "none" and "상어가족 틀어 줄게!" in req["input"][0]["content"][0]["text"]
    assert played == [1] and reason == "music"


def test_놀이_시작은_지시를_붙여_요청한다():
    async def go():
        s = FakeSession()
        c = _conv(s, sleep_timeout=0.3)
        asyncio.create_task(_feed(s, [_transcript("동물 소리 놀이 하자"), _audio_delta(),
                                      {"type": "response.done", "response": {}}]))
        await c.run(greet=False)
        return s

    assert "상황:" in asyncio.run(go()).requests()[0]["response"]["instructions"]


def test_답이_늦으면_맞장구를_튼다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache, sleep_timeout=0.4)
        asyncio.create_task(_feed(s, [_transcript("안녕")]))
        asyncio.create_task(_feed(s, [_audio_delta(), {"type": "response.done", "response": {}}],
                                  gap=0.15))
        await c.run(greet=False)
        return cache

    assert "음..." in asyncio.run(go()).asked


def test_답이_안_오면_취소하고_되묻는다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache, sleep_timeout=0.6)
        asyncio.create_task(_feed(s, [_transcript("안녕")]))
        await c.run(greet=False)
        return s, cache

    s, cache = asyncio.run(go())
    assert {"type": "response.cancel"} in s.sent and PHRASES["recovery"] in cache.asked


def test_끊기면_문구를_틀고_끝난다():
    async def go():
        s, cache = FakeSession(), FakeCache()
        c = _conv(s, cache=cache)
        asyncio.create_task(_feed(s, [None]))
        return await c.run(greet=False), cache

    reason, cache = asyncio.run(go())
    assert reason == "lost" and PHRASES["lost"] in cache.asked


def test_노래_중_호출에_말이_없으면_노래로_돌아간다():
    class Music:
        resumed = 0

        def handle(self, text):
            return None

        def resume_after_wake(self):
            Music.resumed += 1
            return True

    async def go():
        return await _conv(FakeSession(), music=Music()).run(greet=False, woke_during_music=True)

    assert asyncio.run(go()) == "music_resumed" and Music.resumed == 1


async def _pump_once(conv):
    await conv.mic.put((np.zeros(1280, dtype=np.float32), 16000))
    t = asyncio.create_task(conv._pump_mic())
    await asyncio.sleep(0.05)
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)


def test_마이크는_재생_중에는_안_보낸다():
    class BusySpeaker(FakeSpeaker):
        busy = True

    async def go():
        s = FakeSession()
        await _pump_once(_conv(s, speaker=BusySpeaker()))
        return s

    assert asyncio.run(go()).sent == []


def test_마이크는_조용하면_24k_로_보낸다():
    async def go():
        s = FakeSession()
        await _pump_once(_conv(s))
        return s

    sent = asyncio.run(go()).sent
    assert len(sent) == 1 and sent[0]["type"] == "input_audio_buffer.append"


def test_호출_직후_이어진_말을_먼저_보낸다():
    async def go():
        s = FakeSession()
        await _conv(s, sleep_timeout=0.1).run(preroll=(np.zeros(1600, dtype=np.float32), 16000),
                                              greet=False)
        return s

    assert asyncio.run(go()).sent[0]["type"] == "input_audio_buffer.append"
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_realtime_conversation.py -q` → FAIL `No module named 'app.realtime_conversation'`

- [ ] **Step 3: 구현** — `app/realtime_conversation.py`
```python
"""깨어 있는 한 구간 — 연결을 열고, 턴을 돌리고, 대기로 갈 이유가 생기면 닫는다.

세 일이 동시에 돈다: 마이크 보내기 / 이벤트 받기 / 시계(맞장구·응답 제한·무응답).
끝나는 이유: sleep(잠들기 명령) · idle(무응답) · music(노래 틀고 대기) ·
music_resumed(노래 중 호출에 답했거나 말이 없었다) · lost(끊김).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from . import safety
from .realtime_audio import MicGate, PcmAccumulator, to_pcm16
from .realtime_protocol import (append_audio, cached_tokens, cancel, cost_usd, event_kind,
                                respond, say_exactly)
from .realtime_session import ConnectionLost
from .realtime_turn import PHRASES, Route, missing_required, route

log = logging.getLogger("jaeha_bot.realtime")


@dataclass
class _Pending:
    route: Route
    child: str
    stopped_at: float | None
    transcript_at: float
    requested_at: float
    first_audio_at: float | None = None
    said: str = ""
    filler: bool = False
    acc: PcmAccumulator = field(default_factory=PcmAccumulator)


class Conversation:
    def __init__(self, *, session, mic: asyncio.Queue, speaker, cache, cfg, music, games,
                 sleep_words, instructions: str, history: list, metrics=None,
                 sleep_timeout: float = 30.0, filler_phrases=(), clock=time.monotonic,
                 tick_s: float = 0.05) -> None:
        self.session, self.mic, self.speaker, self.cache = session, mic, speaker, cache
        self.cfg, self.music, self.games = cfg, music, games
        self.sleep_words, self.instructions = sleep_words, instructions
        self.history, self.metrics = history, metrics
        self.sleep_timeout = float(sleep_timeout)
        self.filler_phrases = list(filler_phrases)
        self.clock, self.tick_s = clock, tick_s
        self.gate = MicGate(cfg.mic_pad_s)
        self._pending: _Pending | None = None
        self._stopped_at: float | None = None
        self._last_active = clock()
        self._started = clock()
        self._heard_speech = False
        self._woke_during_music = False
        self._filler_i = 0
        self._done: asyncio.Future | None = None

    # ── 밖에서 부르는 것 ──────────────────────────────────────────────────
    async def run(self, *, preroll=None, greet: bool = True,
                  woke_during_music: bool = False) -> str:
        self._done = asyncio.get_running_loop().create_future()
        self._woke_during_music = woke_during_music
        self._started = self._last_active = self.clock()
        try:
            await self.session.open(self.instructions, self.history)
        except ConnectionLost as e:
            log.warning("Realtime 연결 실패: %s", e)
            self._say_cached("lost")
            await self._wait_quiet()
            return "lost"
        if preroll is not None and np.asarray(preroll[0]).size:
            await self._send(append_audio(to_pcm16(preroll[0], preroll[1])))
            log.info("호출 직후 이어진 말 %.2fs 를 먼저 보냄", np.asarray(preroll[0]).size / preroll[1])
        elif greet:
            self._say_cached("wake")
        tasks = [asyncio.create_task(self._pump_mic()),
                 asyncio.create_task(self._read_events()),
                 asyncio.create_task(self._tick())]
        try:
            reason = await self._done
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._wait_quiet()
            await self.session.close()
        log.info("대화 구간 끝 → %s", reason)
        return reason

    # ── 내부 ────────────────────────────────────────────────────────────
    def _finish(self, reason: str) -> None:
        if self._done is not None and not self._done.done():
            self._done.set_result(reason)

    async def _send(self, msg: dict) -> None:
        try:
            await self.session.send(msg)
        except ConnectionLost as e:
            self._lost(e)

    def _lost(self, e) -> None:
        if self._done is not None and self._done.done():
            return
        log.warning("Realtime 끊김: %s", e)
        self._pending = None
        self._say_cached("lost")
        self._finish("lost")

    def _say_cached(self, key: str) -> None:
        audio = self.cache.get(PHRASES[key])
        if audio is None:
            log.warning("고정 문구 캐시가 없다(%s) — 말 없이 넘어간다", key)
            return
        self.speaker.push(audio)

    async def _wait_quiet(self, cap_s: float = 15.0) -> None:
        end = self.clock() + cap_s
        while self.speaker.busy and self.clock() < end:
            await asyncio.sleep(self.tick_s)

    async def _pump_mic(self) -> None:
        while True:
            frame, rate = await self.mic.get()
            now = self.clock()
            self.gate.sync(self.speaker.busy, now)
            if self.gate.should_send(now):
                await self._send(append_audio(to_pcm16(frame, rate)))

    async def _read_events(self) -> None:
        try:
            async for ev in self.session.events():
                await self._on_event(ev)
        except ConnectionLost as e:
            self._lost(e)

    async def _on_event(self, ev: dict) -> None:
        k, now, p = event_kind(ev), self.clock(), self._pending
        if k == "speech_started":
            self._last_active, self._heard_speech = now, True
        elif k == "speech_stopped":
            self._stopped_at, self._last_active = now, now
        elif k == "transcript":
            await self._on_transcript((ev.get("transcript") or "").strip(), now)
        elif k == "audio" and p is not None:
            if p.first_audio_at is None:
                p.first_audio_at = now
            self.speaker.push(p.acc.feed(base64.b64decode(ev.get("delta", ""))))
        elif k == "text" and p is not None:
            p.said += ev.get("delta", "")
        elif k == "done" and p is not None:
            await self._on_done(p, ev.get("response", {}) or {})
        elif k == "error":
            log.warning("Realtime 오류 이벤트: %s", ev.get("error"))

    async def _on_transcript(self, text: str, now: float) -> None:
        self._last_active = now
        if self._pending is not None:
            log.info("[아이] %s (답하는 중이라 이번 말은 넘긴다)", text)
            return
        r = route(text, music=self.music, games=self.games, sleep_words=self.sleep_words)
        if r.kind == "empty":
            log.info("받아 적기 결과 없음 → 계속 듣는다")
            return
        log.info("[아이] %s (%s)", text, r.kind)
        if r.kind == "sleep":
            self._say_cached("sleep")
            self._finish("sleep")
            return
        if r.kind == "music" or (r.kind.startswith("game") and not r.instructions):
            msg = say_exactly(r.say)
        elif r.kind.startswith("game"):
            msg = respond(r.instructions)
        else:
            msg = respond()
        self._pending = _Pending(r, text, self._stopped_at, now, self.clock())
        await self._send(msg)

    async def _on_done(self, p: _Pending, response: dict) -> None:
        self._pending = None
        self._last_active = self.clock()
        reply, kind = p.said.strip(), p.route.kind
        flags = safety.check_reply(reply, child_text=p.child) if reply else []
        if flags:
            log.warning("[안전] %s ← %s (로그만)", flags, reply)
        missing = missing_required(reply, p.route.require) if p.route.instructions else []
        if missing:
            log.warning("[놀이] 필수 낱말 빠짐 %s ← %s (R2)", missing, reply)
        log.info("[티드] %s (%s)", reply, kind)
        usage = response.get("usage", {}) or {}
        if self.metrics is not None:
            got_audio = p.first_audio_at is not None
            self.metrics.record_realtime_turn(
                kind=kind,
                perceived_s=(p.first_audio_at - p.stopped_at
                             if got_audio and p.stopped_at is not None else None),
                transcribe_s=(p.transcript_at - p.stopped_at if p.stopped_at is not None else None),
                respond_first_s=(p.first_audio_at - p.requested_at if got_audio else None),
                filler=p.filler, reply=reply, child_text=p.child,
                cost_usd=cost_usd(self.cfg.model, usage), cached_tokens=cached_tokens(usage),
                safety=flags, game_missing=missing)
        if kind in ("chat", "game", "game_start") and reply:
            self.history.append((p.child, reply))
            if len(self.history) > self.cfg.history_turns:
                del self.history[:len(self.history) - self.cfg.history_turns]
        if kind == "music":
            await self._wait_quiet()
            mr = p.route.music
            ok = mr.action() if mr.action is not None else True
            if not ok:
                self._say_cached("music_failed")
            if ok and mr.standby:
                self._finish("music")
            return
        if self._woke_during_music and self.music is not None:
            await self._wait_quiet()
            if self.music.resume_after_wake():
                self._finish("music_resumed")

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.tick_s)
            now, p = self.clock(), self._pending
            if p is not None and p.first_audio_at is None:
                waited = now - p.requested_at
                if (p.route.kind == "chat" and not p.filler and self.filler_phrases
                        and waited >= self.cfg.filler_after_s):
                    phrase = self.filler_phrases[self._filler_i % len(self.filler_phrases)]
                    self._filler_i += 1
                    audio = self.cache.get(phrase)
                    if audio is not None:
                        self.speaker.push(audio)
                    p.filler = True
                if waited >= self.cfg.response_timeout_s:
                    log.warning("답이 %.1fs 동안 안 왔다 → 취소하고 되묻는다", waited)
                    self._pending = None
                    await self._send(cancel())
                    self._say_cached("recovery")
                    self._last_active = now
                continue
            if p is not None or self.speaker.busy:
                continue
            if (self._woke_during_music and not self._heard_speech
                    and now - self._started >= self.cfg.wake_listen_s):
                if self.music is not None and self.music.resume_after_wake():
                    log.info("노래 중 호출 뒤 말이 없음 → 노래 이어서, 대기로")
                    self._finish("music_resumed")
                    return
            if now - self._last_active >= self.sleep_timeout:
                log.info("무응답 %.0f초 → 대기 모드로", self.sleep_timeout)
                self._say_cached("sleep")
                self._finish("idle")
                return
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_realtime_conversation.py -q` → 전부 PASS. 시간에 민감한 시험이 가끔 흔들리면 그 시험의 `sleep_timeout`·`gap` 을 2배로 — 로직은 바꾸지 않는다.

- [ ] **Step 5: 커밋**
```bash
git add app/realtime_conversation.py tests/test_realtime_conversation.py
git commit -m "feat(realtime): 깨어 있는 한 구간 — 턴을 우리가 쥐고 노래·놀이·맞장구·무응답·끊김을 처리한다"
```

---

### Task 8: 진입점 (`app/main_realtime.py`)

**Files:**
- Create: `app/main_realtime.py`
- Test: `tests/test_main_realtime.py`

**Interfaces:**
- Consumes: Task 1~7 전부, `setup_audio_device(output_device=)`, `AudioSource(preroll, verify_window, embed_window).open()` + `.read()`/`.drain()`/`.close()`/`.samplerate`, `make_detector(wcfg, None, source)` → `.wait_for_wake() -> WakeResult|None`, `.reset()`, `MusicController.pause_for_wake()/stop()/close()`
- Produces: `check_startup(models, env, *, wake=True) -> str`(키; 문제면 `RuntimeError`), `build_instructions(music_on) -> str`, `main(argv=None)`, CLI `python -m app.main_realtime [--no-wake]`

- [ ] **Step 1: 실패하는 시험** — `tests/test_main_realtime.py`
```python
import subprocess
import sys

import pytest

from app.main_realtime import build_instructions, check_startup


def _models(mode="embed"):
    return {"wake": {"enabled": True, "detector": "onnx", "onnx": {"verify": {"mode": mode}}}}


def test_키가_없으면_기동하지_않는다():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        check_startup(_models(), {})


def test_호출어_검증이_임베딩이_아니면_기동하지_않는다():
    # 🔴 전면 API 에는 whisper 가 없다 — whisper 검증이면 2단계 없이 돈다.
    with pytest.raises(RuntimeError, match="embed"):
        check_startup(_models("whisper"), {"OPENAI_API_KEY": "k"})


def test_호출어를_안_쓰면_검증_모드는_안_본다():
    assert check_startup(_models("whisper"), {"OPENAI_API_KEY": "k"}, wake=False) == "k"


def test_정상이면_키를_돌려준다():
    assert check_startup(_models(), {"OPENAI_API_KEY": "k"}) == "k"


def test_노래가_켜지면_프롬프트를_바꾼다():
    from app.config import settings
    assert build_instructions(False) == settings.prompts["system"]
    assert build_instructions(True) != settings.prompts["system"]


def test_로컬_STT_TTS_를_싣지_않는다():
    code = ("import sys, app.main_realtime; "
            "bad=[m for m in ('app.stt_module','app.tts_module','faster_whisper') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 2: 실패 확인** — `... -m pytest tests/test_main_realtime.py -q` → FAIL `No module named 'app.main_realtime'`

- [ ] **Step 3: 구현** — `app/main_realtime.py`
```python
"""전면 API(Realtime) 봇 진입점. 로컬 경로는 app/main.py — 그쪽은 건드리지 않는다.

실행: python -m app.main_realtime            (젯슨: ./run.sh 가 pipeline 을 보고 고른다)
      python -m app.main_realtime --no-wake  (호출어 없이 바로 대화 — 노트북 확인용)
설계: docs/superpowers/specs/2026-09-16-realtime-pipeline-design.md
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import threading
import time

from .config import settings

log = logging.getLogger("jaeha_bot.main_realtime")


def check_startup(models: dict, env, *, wake: bool = True) -> str:
    """조용히 안 깨어나는 봇을 만들지 않는다 — 문제면 기동에서 멈춘다."""
    key = (env.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY 가 없다 — .env 를 확인할 것")
    wcfg = models.get("wake", {}) or {}
    if wake and wcfg.get("enabled", True):
        mode = ((wcfg.get("onnx") or {}).get("verify") or {}).get("mode", "whisper")
        if wcfg.get("detector", "stt") != "onnx" or mode != "embed":
            raise RuntimeError(
                "전면 API 에는 whisper 가 없다 — wake.detector: onnx, wake.onnx.verify.mode: embed "
                f"여야 한다(지금 detector={wcfg.get('detector')}, mode={mode})")
    return key


def build_instructions(music_on: bool) -> str:
    system = settings.prompts["system"]
    if music_on:
        from .music import adjust_prompt
        system = adjust_prompt(system)
    return system


def _build_music(ycfg: dict):
    """app/main.py 의 _build_music 과 같은 내용. 그 파일을 import 하면 STT·TTS 가 따라온다."""
    try:
        from .audio_player import AudioPlayer, default_library
        from .config import BASE_DIR
        from .music import MusicController
        from .youtube import YouTubePlayer, YouTubeSearch

        library = default_library()
        search = youtube = None
        key = os.environ.get(ycfg.get("key_env", "YOUTUBE_DATA_KEY"), "")
        if key:
            player = YouTubePlayer(volume=int(ycfg.get("volume", 80)))
            lack = player.missing()
            if lack:
                log.warning("노래: 준비물이 없어 유튜브를 끈다 %s", lack)
            else:
                youtube = player
                search = YouTubeSearch(
                    key, cache_path=BASE_DIR / "logs" / "youtube_cache.json",
                    max_duration_s=int(ycfg.get("max_duration_s", 600)),
                    cache_days=float(ycfg.get("cache_days", 7)))
        else:
            log.warning("노래: 유튜브 키가 없어 로컬 음원만")
        log.info("노래 틀기 켬 — 로컬 %d곡%s", len(library.playable("song")),
                 " + 유튜브" if youtube else "")
        return MusicController(library=library, local=AudioPlayer(library), search=search,
                               youtube=youtube,
                               default_query=str(ycfg.get("default_query", "동요")))
    except Exception as e:
        log.warning("노래 틀기 구성 실패(없이 계속): %s: %s", type(e).__name__, str(e)[:120])
        return None


class _MicThread:
    """AudioSource.read()(블로킹)를 스레드에서 돌려 큐로 넘긴다."""

    def __init__(self, source) -> None:
        self.source = source
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="rt-mic")
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.source.read()
            except Exception as e:
                log.warning("마이크 읽기 실패: %s", e)
                time.sleep(0.1)
                continue
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.2):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)


async def _bridge(mic: _MicThread, aq: asyncio.Queue, rate: int) -> None:
    loop = asyncio.get_running_loop()
    while True:
        frame = await loop.run_in_executor(None, mic.get)
        if frame is not None:
            await aq.put((frame, rate))


async def _awake_period(*, mic: _MicThread, rate: int, **conv_kwargs) -> str:
    from .realtime_conversation import Conversation

    run_kwargs = {k: conv_kwargs.pop(k) for k in ("preroll", "greet", "woke_during_music")}
    aq: asyncio.Queue = asyncio.Queue()
    bridge = asyncio.create_task(_bridge(mic, aq, rate))
    try:
        return await Conversation(mic=aq, **conv_kwargs).run(**run_kwargs)
    finally:
        bridge.cancel()
        await asyncio.gather(bridge, return_exceptions=True)


def main(argv=None) -> None:
    from .audio_device import setup_audio_device
    from .audio_source import AudioSource
    from .education_modes import GameManager
    from .metrics import MetricsLogger
    from .realtime_audio import StreamSpeaker
    from .realtime_protocol import RealtimeConfig
    from .realtime_session import RealtimeSession, synthesize
    from .realtime_turn import PHRASES
    from .voice_cache import VoiceCache
    from .wake import make_detector

    ap = argparse.ArgumentParser(description="재하봇 — 전면 API(Realtime)")
    ap.add_argument("--no-wake", action="store_true", help="호출어 없이 바로 대화(노트북 확인용)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    models = settings.models
    wcfg = models.get("wake", {}) or {}
    wake_on = bool(wcfg.get("enabled", True)) and not a.no_wake
    api_key = check_startup(models, os.environ, wake=wake_on)
    cfg = RealtimeConfig.from_dict(models.get("realtime"))
    log.info("재하봇(Realtime) 시작 — %s / 목소리 %s / 받아적기 %s",
             cfg.model, cfg.voice, cfg.transcribe_model)

    ycfg = models.get("youtube", {}) or {}
    music_on = bool(ycfg.get("enabled", False)) and wake_on
    if not setup_audio_device(output_device=ycfg.get("output_device", "respk") if music_on else None):
        log.warning("공유 출력 장치가 없어 노래 틀기를 끈다")
        music_on = False
    music = _build_music(ycfg) if music_on else None
    mcfg = models.get("metrics", {}) or {}
    metrics = MetricsLogger(enabled=mcfg.get("enabled", True), tag=f"{mcfg.get('tag', 'pc')}-rt")
    fcfg = models.get("filler", {}) or {}
    filler_phrases = list(fcfg.get("phrases", [])) if fcfg.get("enabled", True) else []

    cache = VoiceCache(cfg.voice_cache_dir, cfg.voice, cfg.model)
    made = cache.ensure(list(PHRASES.values()) + filler_phrases,
                        lambda p: asyncio.run(synthesize(cfg, api_key, p)))
    log.info("고정 문구 캐시 준비(새로 만듦 %d)", made)

    vcfg = (wcfg.get("onnx") or {}).get("verify") or {}
    speaker = StreamSpeaker()
    source = AudioSource(preroll=float(wcfg.get("preroll", 0.5)),
                         verify_window=float(vcfg.get("window_s", 2.0)),
                         embed_window=float(vcfg.get("embed_window_s", 3.0))).open()
    detector = make_detector(wcfg, None, source) if wake_on else None
    metrics.after_load()
    common = dict(speaker=speaker, cache=cache, cfg=cfg, music=music,
                  games=GameManager(render=None), sleep_words=wcfg.get("sleep_words"),
                  instructions=build_instructions(music_on), history=[], metrics=metrics,
                  sleep_timeout=float(wcfg.get("sleep_timeout", 30)),
                  filler_phrases=filler_phrases)

    log.info("준비 완료 (종료: Ctrl+C)")
    if wake_on:
        ready = cache.get(PHRASES["ready"])
        if ready is not None:
            speaker.push(ready)
            while speaker.busy:
                time.sleep(0.05)
    try:
        while True:
            preroll, greet, woke_during_music = None, True, False
            if wake_on:
                source.drain()
                result = detector.wait_for_wake()
                if result is None:
                    continue
                if music is not None and music.pause_for_wake():
                    log.info("노래 중 호출 → 일시정지하고 듣는다")
                    woke_during_music = True
                if result.continued and result.preroll.size:
                    preroll, greet = (result.preroll, source.samplerate), False
            mic = _MicThread(source)
            try:
                reason = asyncio.run(_awake_period(
                    mic=mic, rate=source.samplerate, session=RealtimeSession(cfg, api_key),
                    preroll=preroll, greet=greet, woke_during_music=woke_during_music,
                    **common))
            finally:
                mic.stop()
            if reason == "sleep" and music is not None:
                music.stop()
            if not wake_on and reason in ("sleep", "idle", "lost"):
                break
            if detector is not None:
                detector.reset()
    except KeyboardInterrupt:
        log.info("종료 신호(Ctrl+C) 수신")
    finally:
        if music is not None:
            music.close()
        source.close()
        speaker.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인** — `... -m pytest tests/test_main_realtime.py -q` → 전부 PASS

- [ ] **Step 5: 커밋**
```bash
git add app/main_realtime.py tests/test_main_realtime.py
git commit -m "feat(realtime): 전면 API 진입점 — 키·임베딩 검증이 없으면 기동에서 멈추고, 로컬 STT·TTS 는 싣지 않는다"
```

---

### Task 9: `run.sh` 가 경로를 고르고, 의존성·전체 시험

**Files:**
- Modify: `run.sh`, `requirements.txt`

- [ ] **Step 1: `run.sh` 의 `local)` 갈래를 바꾼다**
```bash
  local)   # 보드/PC 네이티브 실행. configs 의 pipeline 이 realtime 이면 전면 API 봇.
    PY="$(resolve_or_die)"; add_env_libs "$PY"
    ENTRY="$("$PY" -c 'from app.config import settings; print("app.main_realtime" if settings.models.get("pipeline", "local") == "realtime" else "app.main")')"
    echo "▶ $PY -m $ENTRY"
    exec "$PY" -m "$ENTRY" ;;
```

- [ ] **Step 2: `check)` 갈래의 python 블록 끝(`PYEOF` 바로 위)에 추가**
```python
from app.config import settings as _s
_p = _s.models.get("pipeline", "local")
print("대화 경로:", _p)
if _p == "realtime":
    import os
    print("websockets:", "있음" if importlib.util.find_spec("websockets") else "❌ 없음 -> pip install websockets==16.1.1")
    try:
        from app.main_realtime import check_startup
        check_startup(_s.models, os.environ)
        print("기동 검사: 통과(키 있음, 호출어 임베딩 검증)")
    except Exception as e:
        print(f"기동 검사: ❌ {e}")
```

- [ ] **Step 3: `requirements.txt` 에 추가**
```
websockets==16.1.1   # 전면 API(Realtime) 소켓 — 젯슨 aarch64 휠 있음(2026-09-16 확인)
```

- [ ] **Step 4: 전체 시험** — `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest -q` → 기존 1,032개 + 새 시험 전부 PASS. 기존 시험이 깨지면 원인을 고치기 전에 커밋하지 않는다.

- [ ] **Step 5: 커밋**
```bash
git add run.sh requirements.txt
git commit -m "feat(run): pipeline 이 realtime 이면 전면 API 봇을 띄운다 — check 에 소켓·기동 검사"
```

---

### Task 10: 노트북 실기 확인

- [ ] **Step 1: 고정 문구 하나를 실제로 만들어 본다(마이크 불필요)**
```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -c "import asyncio, os; from app.config import settings; from app.realtime_protocol import RealtimeConfig; from app.realtime_session import synthesize; a = asyncio.run(synthesize(RealtimeConfig(), os.environ['OPENAI_API_KEY'], '응! 왜 불렀어? 나랑 놀자!')); print(round(a.size / 24000, 2), 's')"
```
Expected: 1~3 초. 실패(스키마 거부 등)하면 오류를 spec 9절에 적고, `say_exactly` 를 `conversation.item.create`(user input_text) + `response.create`(`conversation: "none"` 없이) 두 단계로 바꿔 다시 한다 — 라이브 루프 `sample_voices` 가 검증된 방식이다. 이때 노래 안내 문장이 대화 기록에 들어가는 것은 보류 기록에 적는다.

- [ ] **Step 2: 호출어 없이 대화(사람이 말한다)** — `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m app.main_realtime --no-wake`
  ① "안녕" → 답 ② "동물 소리 놀이 하자" → 놀이 시작, 한 턴 더 ③ "그만" ④ "바이바이" → 잠들기 문구 후 종료. 로그 `[계측] rt턴` 줄의 체감·받아적기 값을 적는다.

- [ ] **Step 3: 결과를 spec 9절에 기록하고 커밋**
```bash
git add docs/superpowers/specs/2026-09-16-realtime-pipeline-design.md
git commit -m "spec(realtime): 노트북 실기 — 체감·받아적기 지연과 보류 항목"
```

---

### Task 11: 젯슨 배포와 실기

- [ ] **Step 1: websockets 설치**
```bash
ssh $JETSON 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && pip install websockets==16.1.1'
```

- [ ] **Step 2: 바뀐 파일만 LF 로 보낸다**
```bash
for f in app/realtime_audio.py app/realtime_protocol.py app/realtime_turn.py app/realtime_session.py app/voice_cache.py app/realtime_conversation.py app/main_realtime.py app/education_modes.py app/metrics.py configs/model_paths.yaml run.sh requirements.txt tools/realtime_live.py tools/realtime_probe.py; do tr -d '\r' < "$f" > /tmp/rt_push && scp -q /tmp/rt_push "$JETSON:~/jaeha_bot/$f"; done
ssh $JETSON 'chmod +x ~/jaeha_bot/run.sh'
```

- [ ] **Step 3: 젯슨 `configs/local.yaml` 에 추가(백업 먼저) 후 점검**
```bash
ssh $JETSON 'cp ~/jaeha_bot/configs/local.yaml ~/local.yaml.bak_20260916_realtime && printf "\n# 2026-09-16 전면 API 대화 경로. 되돌리기: 이 줄 삭제.\npipeline: realtime\n" >> ~/jaeha_bot/configs/local.yaml && cd ~/jaeha_bot && ./run.sh check'
```
Expected: `대화 경로: realtime`, `websockets: 있음`, `기동 검사: 통과`

- [ ] **Step 4: 실기(사람이 젯슨 앞에서)** — `cd ~/jaeha_bot && ./run.sh`
  1. 자유대화 5턴 2. "상어가족 틀어줘" → 노래 → "하이 티드" → 한마디 → 노래 이어짐 3. "동물 소리 놀이 하자" 한 판 4. "바이바이" 5. 다시 불러서 30초 가만히
  - 끝나면 요약:
```bash
ssh $JETSON 'cd ~/jaeha_bot && source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && python - <<EOF
import json, statistics as st, glob
f = sorted(glob.glob("logs/metrics_*.jsonl"))[-1]
rt = [json.loads(l) for l in open(f, encoding="utf-8") if "rt_turn" in l]
def med(k):
    xs = [r[k] for r in rt if r.get(k) is not None]
    return round(st.median(xs), 3) if xs else None
print(f, "턴", len(rt), "체감", med("perceived_s"), "받아적기", med("transcribe_s"),
      "요청→첫소리", med("respond_first_s"),
      "시스템메모리 최대", max((r.get("sys_mem_used_mb") or 0) for r in rt) if rt else None,
      "비용 합", round(sum(r.get("cost_usd") or 0 for r in rt), 4))
EOF'
```

- [ ] **Step 5: 원자료를 가져와 spec 9절에 기록·커밋**
```bash
D=$(date +%Y%m%d); scp $JETSON:~/jaeha_bot/logs/metrics_$D.jsonl reports/jetson/metrics_${D}_realtime.jsonl
git add reports/jetson/metrics_${D}_realtime.jsonl docs/superpowers/specs/2026-09-16-realtime-pipeline-design.md
git commit -m "reports(realtime): 젯슨 전면 API 첫 실기 — 체감·받아적기·메모리를 로컬 4.14s·85% 와 비교"
```
