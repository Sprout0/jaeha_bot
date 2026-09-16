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
