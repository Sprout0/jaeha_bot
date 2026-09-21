"""Realtime 소켓에 오가는 것 — 전부 순수 함수라 소켓 없이 시험한다.

GA 스키마(2026-09 확인, tools/realtime_probe.py 에서 실측). 베타의 평평한
input_audio_format/turn_detection 은 거부된다.
"""
from __future__ import annotations

import base64
import json
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
    speed: float = 1.0                            # API 허용 0.25~1.5(09-19 실측)
    transcribe_model: str = "gpt-4o-transcribe"   # whisper-1 은 아동 발화 39% 빈 문자열
    transcribe_prompt: str = ""                   # 받아쓰기 힌트(노래·캐릭터 이름). 봇이 채운다
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


def session_update(cfg: RealtimeConfig, instructions: str, tools: list | None = None) -> dict:
    transcription = {"model": cfg.transcribe_model, "language": "ko"}
    if cfg.transcribe_prompt:
        transcription["prompt"] = cfg.transcribe_prompt
    msg = _session_update(cfg, instructions, transcription)
    if tools:
        # 2026-09-22 짧은 명령 — 요청마다 tool_choice 로 쓸지 정한다(chat 턴만 auto).
        msg["session"]["tools"] = list(tools)
        msg["session"]["tool_choice"] = "auto"
    return msg


def _session_update(cfg: RealtimeConfig, instructions: str, transcription: dict) -> dict:
    return {"type": "session.update", "session": {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SR},
                "transcription": transcription,
                # 🔴 create_response=false: 받아 적기를 보고 우리가 답을 요청한다.
                #    서버가 먼저 답하면 노래·놀이 명령을 가로챌 수 없다.
                "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                   "prefix_padding_ms": 300,
                                   "silence_duration_ms": int(cfg.silence_ms),
                                   "create_response": False,
                                   "interrupt_response": False},
            },
            "output": {"format": {"type": "audio/pcm", "rate": SR}, "voice": cfg.voice,
                       "speed": float(cfg.speed)},
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


def user_text(text: str) -> dict:
    """아이 말을 글자로 대화에 넣는다 — 끊겼다 다시 붙었을 때 답하던 말을 다시 묻는다."""
    return _message("user", text)


def append_audio(pcm: bytes) -> dict:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}


def respond(instructions: str | None = None, *, tools: bool = False) -> dict:
    """답 요청. tools=True 는 chat 턴 — 모델이 명령 도구를 부를 수 있다(2026-09-22)."""
    r: dict = {"tool_choice": "auto" if tools else "none"}
    if instructions is not None:
        r["instructions"] = instructions
    return {"type": "response.create", "response": r}


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


def tool_output(call_id: str) -> dict:
    return {"type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": "ok"}}


def function_calls(response: dict) -> list[tuple[str, dict, str]]:
    """response.done 의 출력에서 도구 호출(이름, 인자, call_id)을 꺼낸다."""
    out = []
    for item in (response or {}).get("output") or []:
        if item.get("type") != "function_call":
            continue
        try:
            args = json.loads(item.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        out.append((item.get("name", ""), args if isinstance(args, dict) else {},
                    item.get("call_id", "")))
    return out


def say_exactly(line: str) -> dict:
    """대화 기록 밖에서 문장 하나를 그대로 읽게 한다(노래 안내·고정 문구 만들기)."""
    return {"type": "response.create", "response": {
        "conversation": "none",
        "tool_choice": "none",
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
