"""받아 적은 아이 말 → 이번 턴에 무엇을 할지. 순수 함수라 소켓 없이 시험한다.

순서는 로컬 봇(app/main.py)과 같다: 잠들기 → 노래 → 놀이 → 자유대화.
노래가 놀이보다 먼저인 이유: 모델은 노래를 못 튼다 — 거기까지 가면 "틀어줄게" 라는
빈 약속만 나간다(reports/bench/README.md 5절).
"""
from __future__ import annotations

import re
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

# 🔴 2026-09-21 안전 막기·정정(spec 2026-09-21-realtime-guard). 문장은 설정에서 바꾼다.
from .reply_gate import GuardConfig as _GuardConfig  # noqa: E402

_GUARD = _GuardConfig.from_dict((settings.models.get("realtime") or {}).get("guard"))
PHRASES["safe"] = _GUARD.safe_line
PHRASES["cant"] = _GUARD.cant_line


@dataclass
class Route:
    kind: str                                  # empty|sleep|music|game|game_start|chat
    say: str = ""                              # 그대로 말할 문장(노래 안내 / 놀이 템플릿)
    instructions: str | None = None            # 놀이 턴의 지시(그 답에만 붙는다)
    require: list[str] = field(default_factory=list)
    music: object = None                       # MusicReply
    beat: dict | None = None                   # 놀이 턴의 비트(대본 이탈 바로잡기, 2026-09-22)


# 🔴 2026-09-19 실기: "됐어 좀 쉬고 있어" 가 자유대화로 가서 봇이 계속 말을 걸었다.
# 대화를 끝내자는 말 → 다시 대기. 둘로 나눈다.
#   _END_HARD : 대화 자체를 끝내는 말. 놀이·노래 중이어도 대기로.
#   _END_SOFT : '그만할래' 류. 놀이·노래를 끄는 말이기도 해서(education_modes._STOP_WORDS,
#               music._STOP) 둘 다 안 받았을 때만 대기로.
# ⚠️ 한 낱말로 느슨하게 잡지 않는다 — '잘 가' 가 '잘 가르쳐', '그만' 이 '그만큼' 에 걸린다.
_END_HARD = re.compile(
    r"쉬고\s?있어|(대화|얘기|이야기|말)\s?(좀\s?)?그만|그만\s?(얘기|말)"
    r"|(이제|나)\s?갈게|잘\s?가(?![가-힣])|이따\s?봐")
_END_SOFT = re.compile(r"^(이제\s?|나\s?)?(그만|그만해|그만할래|그만하자|그만할게)$")
_PUNCT = re.compile(r"[.,!?~…·\"'\s]+")


def wants_to_end(text: str, *, idle: bool) -> bool:
    """대화를 끝내고 대기로 가자는 말인지. idle=놀이·노래가 이 말을 안 받았다."""
    t = (text or "").strip()
    if _END_HARD.search(t):
        return True
    return idle and bool(_END_SOFT.match(_PUNCT.sub(" ", t).strip()))


def _game(kind: str, beat: dict) -> Route:
    if beat.get("instruction"):
        return Route(kind, say=beat["fallback"],
                     instructions=f"{_RENDER_SYSTEM}\n\n상황: {beat['instruction']}",
                     require=list(beat.get("require", [])), beat=beat)
    return Route(kind, say=beat["fallback"], beat=beat)


def route(text: str, *, music, games, sleep_words: list[str] | None) -> Route:
    text = (text or "").strip()
    if not text:
        return Route("empty")
    if is_sleep_command(text, sleep_words) or wants_to_end(text, idle=False):
        return Route("sleep")
    mr = music.handle(text) if music is not None else None
    if mr is not None:
        return Route("music", say=mr.text, music=mr)
    beat = games.handle_beat(text)
    if beat is not None:
        return _game("game", beat)
    if wants_to_end(text, idle=True):
        return Route("sleep")
    beat = games.maybe_start_beat(text)
    if beat is not None:
        return _game("game_start", beat)
    return Route("chat")


def missing_required(reply: str, require: list[str]) -> list[str]:
    return [t for t in require if t not in (reply or "")]
