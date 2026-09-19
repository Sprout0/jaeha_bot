"""유튜브 노래 — 공식 IFrame Player API(재생) + Data API v3(검색). (2026-09-11)

🔴 스트림을 뽑아내지 않는다. 진짜 유튜브 플레이어를 띄워 재생하므로 광고·조회수·
   권리자 정산이 사람이 누른 것과 같다. 09-01 에 막혔던 건 '오디오 분리'였고 이건
   그 반대다. 약관이 요구하는 것 두 가지를 지킨다:
     - 임베드 플레이어 최소 200×200 (PLAYER_W/H — 테스트가 지킨다)
     - 광고·플레이어 UI 를 가리거나 막지 않는다(아무 조작도 안 한다)

구조 (젯슨 실측 2026-09-11):

    봇 ──HTTP(127.0.0.1)──> 크로미움(Xvfb) ──> IFrame Player ──> YouTube
                                  │
                                  └─ PulseAudio ─> respk(dmix) ─> ReSpeaker ─> 스피커
    봇 TTS ─────────────────────────────────────> respk(dmix) ─┘  (같은 스피커를 섞어 쓴다)

  - snap 크로미움은 소리를 **PulseAudio 로만** 낸다(ALSA 직접 연결 없음). 그런데 젯슨엔
    jaeha_bot 몫의 PulseAudio 가 없다(gdm 것만 있다). 그래서 여기서 띄운다.
  - ReSpeaker 출력은 **독점**이라 PulseAudio 가 쥐면 봇이 말을 못 한다(실측
    'Device unavailable'). 그래서 둘 다 dmix(respk)를 거친다 — ~/.asoundrc,
    `./run.sh setup-youtube` 가 넣는다.
  - 헤드리스 크로미움은 소리가 안 난다. Xvfb 로 가상 화면을 띄운다.
  - 요즘 유튜브 임베드는 출처(Referer)가 없으면 거부한다(error 153). data: 주소가
    아니라 로컬 HTTP 서버로 페이지를 준다.
  - 음소거 상태의 플레이어는 크로미움이 오디오 스트림을 아예 안 연다. 무음 시험으로는
    경로를 증명할 수 없다(그래서 09-11 에 실제 소리로 쟀다).

비용(젯슨, 봇 끈 채): 크로미움 PSS 510~590MB 고정·안 늘어남, CPU 켤 때 122% →
안정 52~60%. **처음 틀어 달라고 할 때 띄운다** — 평소엔 0 이다.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("jaeha_bot.youtube")

API = "https://www.googleapis.com/youtube/v3/"

# 약관: 임베드 플레이어 뷰포트는 최소 200×200. 09-11 시험의 320×180 은 어겼다.
PLAYER_W, PLAYER_H = 356, 200


class YouTubeError(Exception):
    """검색·재생 실패. 호출부는 잡아서 '지금은 못 찾겠어'로 이어가면 된다."""


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    duration_s: int
    made_for_kids: bool


# ── 검색 ────────────────────────────────────────────────────────────────────
def parse_duration(iso: str) -> int:
    """ISO-8601 기간(PT2M26S) → 초. 모르는 형식이면 0."""
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", iso or "")
    if not m:
        return 0
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def rank(videos: list[Video], max_duration_s: int) -> list[Video]:
    """아동용 먼저, 그다음 '한 곡 길이'(30초~상한) 먼저. 나머지는 검색 순서를 지킨다.

    '티니핑 노래' 실검색(09-11) 1위가 20분짜리 모음집이었다. 2세에게 "틀어줘" 한 번에
    20분이 나가면 끄는 게 일이 된다. 버리지는 않는다 — 한 곡짜리가 없으면 그거라도 튼다.
    """
    def key(item: tuple[int, Video]):
        i, v = item
        fits = 30 <= v.duration_s <= max_duration_s
        return (not v.made_for_kids, not fits, i)
    return [v for _, v in sorted(enumerate(videos), key=key)]


def _norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _default_fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.load(r)


class YouTubeSearch:
    """Data API v3 검색. safeSearch=strict + 임베드 가능한 것만.

    🔴 키는 로그·예외 어디에도 안 남긴다. 요청 URL 에 들어가므로 URL 을 찍는 순간 샌다.
    쿼터: 검색 100유닛 + 상세 1유닛, 하루 10,000 → 약 99회/일. 그래서 캐시한다.
    """

    def __init__(self, key: str | None, *, fetch=None, cache_path: Path | None = None,
                 max_duration_s: int = 600, cache_days: float = 7, results: int = 10,
                 region: str = "KR", lang: str = "ko", now=time.time,
                 kids_only: bool = False) -> None:
        self.key = key or ""
        # 🔴 2026-09-19 '아동용(madeForKids)' 지정 영상만. 유튜브 키즈 목록은 API 가 없고,
        #    키즈 앱은 이 지정 영상 풀에서 가져간다 — 가장 가까운 공식 기준이다.
        self.kids_only = bool(kids_only)
        self._fetch = fetch or _default_fetch
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_duration_s = int(max_duration_s)
        self.cache_s = float(cache_days) * 86400
        self.results = int(results)
        self.region, self.lang = region, lang
        self._now = now
        self._cache = self._load()

    # 캐시 — 망가져 있으면 버리고 새로 시작한다(노래 못 트는 것보다 낫다)
    def _load(self) -> dict:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("유튜브 검색 캐시를 못 읽어 버린다: %s", e)
            return {}

    def _save(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False),
                                       encoding="utf-8")
        except Exception as e:
            log.warning("유튜브 검색 캐시 저장 실패(무시): %s", e)

    def _call(self, endpoint: str, **params) -> dict:
        if not self.key:
            raise YouTubeError("YOUTUBE_DATA_KEY 가 없다(.env 확인)")
        params["key"] = self.key
        url = API + endpoint + "?" + urllib.parse.urlencode(params)
        try:
            return self._fetch(url)
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8", "replace") or "{}")
                err = body.get("error") or {}
                reasons = [x.get("reason") for x in err.get("errors", [])]
            except Exception:
                reasons = []
            raise YouTubeError(f"{endpoint}: HTTP {e.code} {reasons}") from None
        except Exception as e:
            msg = str(e).replace(self.key, "<키>")
            raise YouTubeError(f"{endpoint}: {type(e).__name__}: {msg[:120]}") from None

    def candidates(self, query: str) -> list[Video]:
        """순위대로 정렬된 후보. 캐시에 있으면 네트워크를 안 탄다."""
        k = _norm_query(query)
        hit = self._cache.get(k)
        if hit and self._now() - float(hit.get("t", 0)) < self.cache_s:
            return [Video(**v) for v in hit.get("videos", [])]
        s = self._call("search", part="snippet", q=query, type="video",
                       videoEmbeddable="true", safeSearch="strict",
                       maxResults=self.results, regionCode=self.region,
                       relevanceLanguage=self.lang)
        ids = [it["id"]["videoId"] for it in s.get("items", [])
               if (it.get("id") or {}).get("videoId")]
        videos: list[Video] = []
        if ids:
            v = self._call("videos", part="snippet,status,contentDetails", id=",".join(ids))
            by_id = {it["id"]: it for it in v.get("items", [])}
            for vid in ids:                     # 검색 순서를 지킨다
                it = by_id.get(vid)
                if not it:
                    continue
                st = it.get("status") or {}
                # 검색에서 임베드 가능만 달라고 했어도 상세에서 한 번 더 본다 — 막힌 걸
                # 틀면 error 150/101 로 조용히 안 나온다.
                if not st.get("embeddable", False):
                    continue
                videos.append(Video(
                    id=vid,
                    title=(it.get("snippet") or {}).get("title", ""),
                    duration_s=parse_duration((it.get("contentDetails") or {}).get("duration", "")),
                    made_for_kids=bool(st.get("madeForKids", False)),
                ))
        ranked = rank(videos, self.max_duration_s)
        self._cache[k] = {"t": self._now(), "videos": [asdict(x) for x in ranked]}
        self._save()
        return ranked

    def find(self, query: str, exclude=()) -> Video | None:
        """제일 나은 후보 하나. exclude 는 '다른 노래'일 때 방금 튼 것들."""
        skip = set(exclude)
        return next((v for v in self.candidates(query)
                     if v.id not in skip and (v.made_for_kids or not self.kids_only)), None)


# ── 플레이어 ────────────────────────────────────────────────────────────────
PLAYER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>jaeha-youtube</title></head>
<body style="margin:0;background:#000"><div id="p"></div>
<script>
let P = null, seq = 0;
function send(o) { fetch('/state', {method: 'POST', body: JSON.stringify(o)}).catch(() => {}); }
function vid() { return (P && P.getVideoData && (P.getVideoData() || {}).video_id) || ''; }
function run(c) {
  if (c.op === 'load') { P.loadVideoById(c.id); P.unMute(); P.setVolume(c.volume); }
  else if (c.op === 'pause') P.pauseVideo();
  else if (c.op === 'resume') P.playVideo();
  else if (c.op === 'stop') P.stopVideo();
  else if (c.op === 'volume') P.setVolume(c.volume);
}
async function poll() {
  try {
    const r = await fetch('/cmd?after=' + seq);
    for (const c of await r.json()) { seq = c.seq; try { run(c); } catch (e) { send({error: 'js:' + e}); } }
  } catch (e) {}
  setTimeout(poll, 250);
}
function onYouTubeIframeAPIReady() {
  P = new YT.Player('p', {width: '__W__', height: '__H__',
    playerVars: {autoplay: 1, playsinline: 1, rel: 0},
    events: {
      onReady: () => { send({ready: true}); poll(); },
      onStateChange: e => send({state: e.data, vid: vid()}),
      onError: e => send({error: e.data}) }});
  setInterval(() => { if (P && P.getPlayerState) send({state: P.getPlayerState(), vid: vid()}); }, 1000);
}
</script>
<script src="https://www.youtube.com/iframe_api"></script>
</body></html>""".replace("__W__", str(PLAYER_W)).replace("__H__", str(PLAYER_H))

# IFrame 상태 코드
ENDED, PLAYING, PAUSED, BUFFERING = 0, 1, 2, 3


class _Bridge:
    """봇 ↔ 페이지. 페이지가 명령을 물어가고(poll) 상태를 보낸다. 추가 라이브러리 불필요."""

    def __init__(self, on_state=None) -> None:
        self._cond = threading.Condition()
        self._cmds: list[dict] = []
        self._seq = 0
        self.state: dict = {}
        self._on_state = on_state

    def push(self, **cmd) -> None:
        with self._cond:
            self._seq += 1
            self._cmds = (self._cmds + [{"seq": self._seq, **cmd}])[-50:]

    def since(self, after: int) -> list[dict]:
        with self._cond:
            return [c for c in self._cmds if c["seq"] > after]

    def update(self, st: dict) -> None:
        with self._cond:
            self.state.update(st)
            self._cond.notify_all()
        # 잠금 밖에서 부른다 — 여기서 pactl(최대 5초)을 돌리므로 wait_for 를 막으면 안 된다.
        if self._on_state is not None:
            try:
                self._on_state(dict(self.state))
            except Exception as e:
                log.warning("상태 알림 실패(무시): %s: %s", type(e).__name__, str(e)[:80])

    def wait_for(self, pred, timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: pred(self.state), timeout)


def _handler(bridge: _Bridge):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # 초당 4번 물어간다 — 로그를 덮으면 안 된다
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/":
                self._send(200, PLAYER_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/cmd":
                after = int((urllib.parse.parse_qs(u.query).get("after") or ["0"])[0] or 0)
                self._send(200, json.dumps(bridge.since(after)).encode(), "application/json")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self):
            if self.path == "/state":
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    bridge.update(json.loads(self.rfile.read(n) or b"{}"))
                except Exception:
                    pass
                self._send(204, b"", "text/plain")
            else:
                self._send(404, b"", "text/plain")
    return H


def _free_display(start: int = 99) -> str:
    for n in range(start, start + 20):
        if not Path(f"/tmp/.X11-unix/X{n}").exists() and not Path(f"/tmp/.X{n}-lock").exists():
            return f":{n}"
    raise YouTubeError("비어 있는 X 디스플레이가 없다")


def _kill(p: subprocess.Popen | None) -> None:
    """프로세스 그룹째 끈다. snap 크로미움은 자식을 여럿 띄워서 본체만 죽이면 남는다
    (09-11 시험에서 웹서버가 껍데기 셸만 죽고 본체가 남은 것과 같은 병)."""
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=4)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


class YouTubePlayer:
    """크로미움 한 대를 띄워 IFrame Player 로 튼다. 실패해도 예외 대신 False.

    처음 play() 때 띄우고(start), 봇이 끝날 때 close() 로 전부 내린다.
    stop() 은 소리를 끄고 **PulseAudio 싱크를 재운다** — 재우면 ReSpeaker 를 놓아서,
    스피커를 직접(hw) 여는 다른 도구(호출어 녹음 등)가 다시 쓸 수 있다.
    """

    def __init__(self, *, volume: int = 80, sink_device: str = "respk",
                 sink_name: str = "jaeha_respk", start_timeout_s: float = 25.0,
                 play_timeout_s: float = 12.0, profile_dir: str | None = None,
                 chromium: tuple[str, ...] = ("snap", "run", "chromium"),
                 leveler=None, sink_volume_pct: int = 100) -> None:
        self.volume = max(0, min(100, int(volume)))
        # 영상마다 다른 음량 맞추기(app/loudness.py). 켜면 싱크를 키우고 기본 볼륨을 낮춘다.
        self.leveler = leveler
        self.sink_volume_pct = int(sink_volume_pct)
        self.sink_device, self.sink_name = sink_device, sink_name
        self.start_timeout_s, self.play_timeout_s = start_timeout_s, play_timeout_s
        self.profile_dir = profile_dir or str(Path.home() / "snap/chromium/common/jaeha-youtube")
        self.chromium = chromium
        self._bridge = _Bridge(on_state=self._on_state)
        self._last_state = None
        self._server: ThreadingHTTPServer | None = None
        self._procs: list[subprocess.Popen] = []
        self._tmp: tempfile.TemporaryDirectory | None = None
        uid = os.getuid() if hasattr(os, "getuid") else 0
        self._runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
        self._pulse_sock = f"{self._runtime}/pulse/native"
        self._started = False
        self._lock = threading.Lock()

    # 준비물 점검 — 없으면 기동 때 한 줄로 알린다
    def missing(self) -> list[str]:
        need = ["Xvfb", "pulseaudio", "pactl", self.chromium[0]]
        return [b for b in need if not shutil.which(b)]

    def _on_state(self, st: dict) -> None:
        """곡이 끝까지 갔다 — 아무도 안 부르므로 여기서 ReSpeaker 를 놓는다.

        페이지가 1초마다 같은 상태를 보내니 **들어간 순간 한 번만** 재운다.
        (stop() 은 자기가 재우고, resume()·play() 는 깨운다.)
        """
        state, prev = st.get("state"), self._last_state
        self._last_state = state
        if state == ENDED and prev != ENDED and self._started:
            log.info("[노래] 곡이 끝났다 — 출력 장치를 놓는다")
            self._pactl("suspend-sink", self.sink_name, "1")

    def _pactl(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, XDG_RUNTIME_DIR=self._runtime)
        return subprocess.run(["pactl", "-s", f"unix:{self._pulse_sock}", *args],
                              env=env, capture_output=True, text=True, timeout=5)

    def _start_pulse(self) -> None:
        # 🔴 snap 크로미움은 AppArmor 로 $XDG_RUNTIME_DIR/pulse/native 만 열 수 있다.
        #    소켓 경로를 바꾸면 크로미움이 못 붙는다. 그래서 기본 경로를 쓴다.
        r = self._pactl("list", "short", "sinks")
        if r.returncode == 0:
            if self.sink_name in r.stdout:
                log.info("PulseAudio 가 이미 떠 있다 — 그대로 쓴다")
                self._pactl("set-sink-volume", self.sink_name, f"{self.sink_volume_pct}%")
                return
            raise YouTubeError("다른 PulseAudio 가 떠 있는데 우리 싱크가 없다")
        assert self._tmp is not None
        script = Path(self._tmp.name) / "jaeha.pa"
        script.write_text(
            "load-module module-native-protocol-unix\n"
            f"load-module module-alsa-sink device={self.sink_device} sink_name={self.sink_name}"
            " rate=16000 channels=2\n"
            f"set-default-sink {self.sink_name}\n"
            f"set-sink-volume {self.sink_name} {int(0x10000 * self.sink_volume_pct / 100)}\n",
            encoding="utf-8")
        env = dict(os.environ, XDG_RUNTIME_DIR=self._runtime)
        p = subprocess.Popen(["pulseaudio", "-n", "-F", str(script), "--exit-idle-time=-1",
                              "--daemonize=no", "--log-target=stderr"],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             start_new_session=True)
        self._procs.append(p)
        for _ in range(40):
            time.sleep(0.1)
            if p.poll() is not None:
                err = (p.stderr.read() or b"").decode("utf-8", "replace")[-300:]
                raise YouTubeError(f"PulseAudio 기동 실패: {err}")
            if self._pactl("list", "short", "sinks").returncode == 0:
                return
        raise YouTubeError("PulseAudio 가 응답하지 않는다")

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            lack = self.missing()
            if lack:
                raise YouTubeError(f"준비물이 없다: {lack} (./run.sh setup-youtube)")
            t0 = time.perf_counter()
            self._tmp = tempfile.TemporaryDirectory(prefix="jaeha-yt-")
            try:
                self._start_pulse()
                disp = _free_display()
                self._procs.append(subprocess.Popen(
                    ["Xvfb", disp, "-screen", "0", "640x400x24", "-nolisten", "tcp"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
                self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self._bridge))
                threading.Thread(target=self._server.serve_forever, daemon=True).start()
                url = f"http://127.0.0.1:{self._server.server_address[1]}/"
                env = dict(os.environ, DISPLAY=disp, XDG_RUNTIME_DIR=self._runtime,
                           PULSE_SERVER=f"unix:{self._pulse_sock}")
                self._procs.append(subprocess.Popen(
                    [*self.chromium, f"--user-data-dir={self.profile_dir}",
                     "--no-first-run", "--no-default-browser-check", "--password-store=basic",
                     "--disable-gpu", "--autoplay-policy=no-user-gesture-required",
                     f"--window-size={PLAYER_W + 40},{PLAYER_H + 60}", url],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True))
                if not self._bridge.wait_for(lambda s: s.get("ready"), self.start_timeout_s):
                    raise YouTubeError(f"플레이어가 {self.start_timeout_s:.0f}초 안에 안 떴다")
            except Exception:
                self._teardown()
                raise
            self._started = True
            if self.leveler is not None:
                self._start_leveler()
            log.info("유튜브 플레이어 준비 (%.1fs)", time.perf_counter() - t0)

    # 16kHz 모노 s16 0.5초. 싱크를 16kHz 로 연다(_start_pulse) — 그대로 잰다.
    _LEVEL_BLOCK = 16000

    def _start_leveler(self) -> None:
        """나가는 소리(싱크 모니터)를 재서 플레이어 볼륨을 고친다. 실패해도 노래는 나온다."""
        env = dict(os.environ, XDG_RUNTIME_DIR=self._runtime)
        try:
            p = subprocess.Popen(
                ["parec", "-s", f"unix:{self._pulse_sock}", "-d", f"{self.sink_name}.monitor",
                 "--format=s16le", "--rate=16000", "--channels=1", "--raw"],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as e:
            log.warning("음량 맞춤을 못 켰다(노래는 그대로): %s", e)
            return
        self._procs.append(p)
        threading.Thread(target=self._level_loop, args=(p,), daemon=True).start()

    def _level_loop(self, p) -> None:
        import numpy as np
        while True:
            data = p.stdout.read(self._LEVEL_BLOCK) if p.stdout else b""
            if not data:
                return
            if not self.is_playing:
                continue
            block = np.frombuffer(data[:len(data) // 2 * 2], "<i2").astype(np.float32) / 32768
            v = self.leveler.feed(block, time.monotonic())
            if v is not None:
                self._bridge.push(op="volume", volume=v)
                log.info("[노래] 음량 맞춤 — 크게 나오는 구간 %.1f dBFS → 볼륨 %d (목표 %.0f)",
                         self.leveler.last_level, v, self.leveler.target_db)

    def play(self, video_id: str) -> bool:
        try:
            self.start()
            self._pactl("suspend-sink", self.sink_name, "0")
            self._bridge.update({"state": None, "vid": "", "error": None})
            volume = self.volume
            if self.leveler is not None:
                self.leveler.reset(time.monotonic())
                volume = self.leveler.volume
            self._bridge.push(op="load", id=video_id, volume=volume)
            ok = self._bridge.wait_for(
                lambda s: s.get("error") is not None
                or (s.get("state") == PLAYING and s.get("vid") == video_id),
                self.play_timeout_s)
            err = self._bridge.state.get("error")
            if not ok or err is not None:
                log.warning("유튜브 재생 실패: %s (영상 %s)", err or "시간 초과", video_id)
                return False
            return True
        except Exception as e:
            log.warning("유튜브 재생 실패: %s: %s", type(e).__name__, str(e)[:160])
            return False

    def pause(self) -> None:
        if self._started:
            self._bridge.push(op="pause")

    def resume(self) -> None:
        if self._started:
            self._pactl("suspend-sink", self.sink_name, "0")
            self._bridge.push(op="resume")

    def stop(self) -> None:
        if not self._started:
            return
        self._bridge.push(op="stop")
        time.sleep(0.3)
        self._pactl("suspend-sink", self.sink_name, "1")   # ReSpeaker 를 놓아 준다

    @property
    def is_playing(self) -> bool:
        return self._started and self._bridge.state.get("state") in (PLAYING, BUFFERING)

    def _teardown(self) -> None:
        for p in reversed(self._procs):
            _kill(p)
        self._procs.clear()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def close(self) -> None:
        with self._lock:
            self._teardown()
            self._started = False


def make_player(ycfg: dict) -> YouTubePlayer:
    """설정 → 플레이어. level.enabled 면 음량 맞춤(app/loudness.py)을 붙인다."""
    lcfg = ycfg.get("level") or {}
    if not lcfg.get("enabled", False):
        return YouTubePlayer(volume=int(ycfg.get("volume", 80)))
    from .loudness import LoudnessLeveler
    lv = LoudnessLeveler(target_db=float(lcfg.get("target_db", -19)),
                         base=int(lcfg.get("base_volume", 20)),
                         lo=int(lcfg.get("min_volume", 5)), hi=int(lcfg.get("max_volume", 100)))
    return YouTubePlayer(volume=lv.base, leveler=lv,
                         sink_volume_pct=int(lcfg.get("sink_volume_pct", 160)))
