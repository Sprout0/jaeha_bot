"""노래 틀기 — 아이 말에서 '틀어줘 / 그만 / 다시'를 가려 로컬 음원·유튜브로 잇는다.

🔴 판단은 전부 여기서 한다(main 은 몇 줄만 붙인다). 테스트가 이 파일을 지킨다.

흐름 (스마트 스피커와 같다):
  "티니핑 노래 틀어줘"  → "티니핑 노래 틀어 줄게!" → 재생 → **대기 모드**(호출어만 듣는다)
  노래 중 "하이 티드"   → 일시정지 → 한마디 듣는다
      "그만"            → 끈다, 대화 계속
      "다른 노래 틀어줘" → 바꾼다, 대기
      그 밖의 말 / 침묵  → 답하고 **이어서 튼다**, 대기

  노래가 나오는 동안 대화 모드로 두면 STT 가 가사를 받아 적고 대답한다. 그래서 대기다.

오탐이 미탐보다 나쁘다 — AudioLibrary.find() 와 같은 원칙. 엉뚱하게 나간 노래는
되돌릴 수 없고, 못 알아들으면 아이가 한 번 더 말하면 된다. 그래서:
  - "켜줘 / 불러줘" 는 노래·음악이 같이 있을 때만("불 켜줘", "엄마 불러줘")
  - "들려줘" 는 소리·이야기면 제외("강아지 소리 들려줘", "이야기 들려줘")
  - 틀 수 없는 물건은 제외("물 틀어줘", "티비 틀어줘")
  - 동사 없는 "계속 / 다시" 는 노래가 나오고 있을 때만 — 평소의 "계속"이 노래를 틀면 안 된다
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from .song_names import correct

log = logging.getLogger("jaeha_bot.music")

MUSIC_FAILED = "어? 노래가 안 나오네. 다음에 다시 해 보자!"
SEARCH_FAILED = "지금은 노래를 못 찾겠어. 같이 불러 볼까?"

_MUSIC_NOUNS = ("노래", "음악", "동요", "뮤직", "송")
# '틀다' 계열은 노래 없이도 요청이다("아기상어 틀어줘"). 켜다·부르다는 아니다.
_PLAY_VERBS = ("틀어", "틀자", "틀래", "들려", "들을래", "듣고싶", "듣자", "들어볼래")
_NOUN_ONLY_VERBS = ("켜", "불러", "부르")
_NOT_MUSIC = ("물", "불", "티비", "tv", "텔레비전", "에어컨", "선풍기", "히터", "보일러",
              "전등", "소리", "이야기", "얘기", "동화", "책", "라디오")
_STOP = ("그만", "멈춰", "멈춤", "꺼줘", "꺼", "끄", "조용", "스톱", "정지")
_RESUME_VERBAL = ("다시틀어", "계속틀어", "다시들려", "계속들려", "다시켜")
_RESUME_BARE = ("계속", "계속해", "다시", "다시해")
_DIFFERENT = ("다른노래", "다른거", "다른음악", "다른동요", "딴노래", "다음노래")

_WAKE_PREFIX = re.compile(r"^\s*(하이\s*티드야?|티드야)[\s,!]*")
_LEAD = re.compile(r"^(나|저|우리|다시|또|그럼)\s+")
_TRAIL = re.compile(r"(\s*(좀|한\s*번|을|를|이랑|하고))+\s*$")
# "듣고 싶어"처럼 동사 사이에 띄어쓰기가 들어와도 잡는다
_VERB_RE = re.compile("|".join(r"\s*".join(map(re.escape, v))
                               for v in _PLAY_VERBS + _NOUN_ONLY_VERBS))


@dataclass(frozen=True)
class Command:
    kind: str                    # play | stop | resume
    query: str = ""              # 비었으면 기본 곡(동요)
    different: bool = False      # "다른 노래"


def _n(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def _cut_at_verb(text: str) -> str:
    """동사 앞까지를 곡 이름으로 본다. "티니핑 노래 틀어줘" → "티니핑 노래"."""
    m = _VERB_RE.search(text)
    head = text[:m.start()] if m else text
    q = _TRAIL.sub("", head.strip())
    return _LEAD.sub("", q).strip(" ,.!?")


def parse(text: str, *, active: bool = False) -> Command | None:
    """노래 명령이 아니면 None — 그러면 평소처럼 놀이·LLM 으로 간다.

    active: 노래가 나오는 중(또는 불러서 잠깐 멈춘 중)인가. 끄기·'계속'은 이때만 본다.
    """
    raw = _WAKE_PREFIX.sub("", text or "").strip()
    n = _n(raw)
    if not n:
        return None
    # 끄기는 노래가 있을 때만 — 평소의 "그만"은 놀이 쪽 말이다.
    if active and any(w in n for w in _STOP):
        return Command("stop")
    if any(w in n for w in _DIFFERENT):
        return Command("play", different=True)
    if any(n.startswith(w) for w in _RESUME_VERBAL):
        rest = _n(_cut_at_verb(raw)).replace("다시", "").replace("계속", "")
        if not rest:
            return Command("resume")
    if active and n in _RESUME_BARE:
        return Command("resume")
    has_noun = any(w in n for w in _MUSIC_NOUNS)
    has_play = any(v in n for v in _PLAY_VERBS)
    has_noun_verb = any(v in n for v in _NOUN_ONLY_VERBS)
    if not (has_play or (has_noun and has_noun_verb)):
        return None
    query = _cut_at_verb(raw)
    qn = _n(query)
    if not has_noun and any(qn == w or qn.endswith(w) for w in _NOT_MUSIC):
        return None
    if has_noun and any(w in qn for w in ("소리", "이야기", "얘기", "동화")):
        return None
    if qn in {_n(w) for w in _MUSIC_NOUNS} or qn in ("노래를", "음악을"):
        query = ""
    return Command("play", query=query)


@dataclass
class MusicReply:
    text: str                                   # 먼저 말할 것
    action: Callable[[], bool] | None = None    # 말한 뒤 할 것(재생). False = 실패
    standby: bool = False                       # 성공하면 대기 모드로


class MusicController:
    """로컬 음원이 있으면 로컬(시연·논문 트랙), 없으면 유튜브(집 트랙)."""

    def __init__(self, *, library=None, local=None, search=None, youtube=None,
                 default_query: str = "동요") -> None:
        self.library, self.local = library, local
        self.search, self.youtube = search, youtube
        self.default_query = default_query
        self._current: tuple[str, object, str] | None = None   # (local|youtube, 대상, 질의)
        self._played: list[str] = []
        self._paused_by_wake = False

    @property
    def active(self) -> bool:
        return bool((self.local is not None and self.local.is_playing)
                    or (self.youtube is not None and self.youtube.is_playing))

    @property
    def paused_by_wake(self) -> bool:
        return self._paused_by_wake

    # ── 명령 ────────────────────────────────────────────────────────────────
    def handle(self, text: str) -> MusicReply | None:
        cmd = parse(text, active=self.active or self._paused_by_wake)
        if cmd is None:
            return None
        if cmd.kind == "stop":
            self.stop()
            return MusicReply("노래 껐어!")
        if cmd.kind == "resume":
            if self._current is not None:
                self._paused_by_wake = False
                return MusicReply("다시 틀어 줄게!", action=self._resume, standby=True)
            cmd = Command("play")
        return self._plan_play(cmd, text)

    def _plan_play(self, cmd: Command, text: str) -> MusicReply:
        self._paused_by_wake = False
        query = cmd.query
        if cmd.different and not query:
            query = self._current[2] if self._current else ""
        query = query or self.default_query
        # 받아쓰기가 이름을 틀리게 적었으면 사전 이름으로(확실할 때만) — '비니닝' → '티니핑'
        query = correct(query)
        # 1) 로컬 — 원문을 넘긴다(find 는 '틀어줘' 같은 요청 단서를 본다)
        if self.library is not None and self.local is not None and not cmd.different:
            asset = self.library.find(text, kind="song")
            if asset is not None and asset.exists:
                return MusicReply(f"{asset.title} 틀어 줄게!",
                                  action=lambda: self._play_local(asset, query), standby=True)
        # 2) 유튜브
        if self.search is None or self.youtube is None:
            return MusicReply(SEARCH_FAILED)
        try:
            video = self.search.find(query, exclude=self._played[-5:] if cmd.different else ())
        except Exception as e:
            log.warning("노래 검색 실패: %s", str(e)[:160])
            return MusicReply(SEARCH_FAILED)
        if video is None:
            return MusicReply("그 노래는 못 찾았어. 다른 노래 말해 줄래?")
        if cmd.different:
            spoken = "다른 노래"
        elif not cmd.query:
            spoken = "신나는 동요"
        else:
            spoken = query
        return MusicReply(f"{spoken} 틀어 줄게!",
                          action=lambda: self._play_youtube(video, query), standby=True)

    # ── 재생 ────────────────────────────────────────────────────────────────
    def _play_local(self, asset, query: str) -> bool:
        if self.youtube is not None:
            self.youtube.stop()
        ok = bool(self.local.play(asset, block=False))
        if ok:
            self._current = ("local", asset, query)
        return ok

    def _play_youtube(self, video, query: str) -> bool:
        if self.local is not None:
            self.local.stop()
        ok = bool(self.youtube.play(video.id))
        if ok:
            self._current = ("youtube", video, query)
            self._played.append(video.id)
            log.info("[노래] 유튜브 %s (%ds) %s", video.id, video.duration_s, video.title[:40])
        return ok

    def _resume(self) -> bool:
        if self._current is None:
            return False
        kind, target, _ = self._current
        if kind == "youtube":
            self.youtube.resume()
            return True
        return bool(self.local.play(target, block=False))    # 로컬은 처음부터

    # ── 호출어와의 관계 ──────────────────────────────────────────────────────
    def pause_for_wake(self) -> bool:
        """노래 중에 불렸다 — 멈추고 한마디 듣는다. 노래가 없었으면 False."""
        if not self.active:
            return False
        if self._current and self._current[0] == "youtube":
            self.youtube.pause()
        elif self.local is not None:
            self.local.stop()
        self._paused_by_wake = True
        return True

    def resume_after_wake(self) -> bool:
        """불러서 멈췄던 노래를 이어 튼다. 그런 적 없으면 False."""
        if not self._paused_by_wake:
            return False
        self._paused_by_wake = False
        return self._resume()

    def stop(self) -> None:
        self._paused_by_wake = False
        if self.local is not None:
            self.local.stop()
        if self.youtube is not None:
            self.youtube.stop()

    def close(self) -> None:
        self.stop()
        if self.youtube is not None:
            self.youtube.close()


# ── 프롬프트 ────────────────────────────────────────────────────────────────
NEW_SONG_RULE = (
    '- 노래는 네가 직접 부르지는 못한다. 대신 아이가 "○○ 틀어줘"라고 하면 틀어 줄 수 있다.\n'
    '  아이가 노래를 듣고 싶어 하면 "곰 세 마리 틀어줘, 처럼 말해 줘!"라고 알려준다.\n'
    '  네가 먼저 "틀어줄게"라고 약속하지는 않는다.\n'
)


# 🔴 2026-09-22 전면 API 에 명령 도구가 생겼다. 위 규칙("○○ 틀어줘, 처럼 말해 줘!")을 그대로 두면
#    모델이 도구를 부르지 않고 아이에게 명령어를 가르친다(실서버: "'아기 상어' 틀어줘, 그렇게 말해 봐!").
TOOL_SONG_RULE = (
    '- 노래는 네가 직접 부르지는 못한다. 아이가 노래를 틀어 달라거나 듣고 싶다고 하면\n'
    '  말로 답하지 말고 play_song 도구를 부른다(곡을 모르면 title 은 빈 문자열).\n'
    '  노래 이야기만 할 때는 부르지 않는다. 네가 먼저 "틀어줄게"라고 약속하지는 않는다.\n'
)


def adjust_prompt(system: str, rule: str = NEW_SONG_RULE) -> str:
    """'노래는 못 튼다' 규칙을 바꾼다. 켜 놓고 그대로 두면 봇이 "못 틀어"라고 거짓말한다."""
    lines = system.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("- 노래는"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and not lines[j].lstrip().startswith("- "):
                j += 1
            indent = line[:len(line) - len(line.lstrip())]
            rule = "".join(indent + ln for ln in rule.splitlines(keepends=True))
            return "".join(lines[:i]) + rule + "".join(lines[j:])
    log.warning("프롬프트에 '- 노래는' 규칙이 없다 — 새 규칙을 끝에 붙인다")
    return system.rstrip("\n") + "\n" + rule
