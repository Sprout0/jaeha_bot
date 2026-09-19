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
    speed: float = 1.0                            # API 허용 0.25~1.5(09-19 실측)
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
