"""유튜브 검색·플레이어 — 네트워크·크로미움 없이 판단만 본다. (2026-09-11)"""
import io
import json
import urllib.error
import urllib.parse

import pytest

from app import youtube as yt
from app.youtube import (PLAYER_H, PLAYER_HTML, PLAYER_W, Video, YouTubeError,
                         YouTubeSearch, parse_duration, rank)

KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE12"


@pytest.mark.parametrize("iso,sec", [
    ("PT2M26S", 146), ("PT20M15S", 1215), ("PT1H1M41S", 3701), ("PT45S", 45),
    ("P0D", 0), ("", 0), ("garbage", 0),
])
def test_parse_duration(iso, sec):
    assert parse_duration(iso) == sec


def test_rank_prefers_kids_then_single_song_length():
    """'티니핑 노래' 실검색 1위는 20분 모음집이었다 — 한 곡짜리가 먼저 나가야 한다."""
    vids = [Video("long", "", 1215, True), Video("adult", "", 150, False),
            Video("song", "", 146, True), Video("song2", "", 145, True)]
    assert [v.id for v in rank(vids, 600)] == ["song", "song2", "long", "adult"]


def _fake_api(search_ids, details, calls):
    def fetch(url):
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        calls.append((u.path.rsplit("/", 1)[-1], q))
        if u.path.endswith("/search"):
            return {"items": [{"id": {"videoId": i}} for i in search_ids]}
        ids = q["id"].split(",")
        return {"items": [details[i] for i in ids if i in details]}
    return fetch


def _detail(vid, dur="PT2M20S", embeddable=True, kids=True):
    return {"id": vid, "snippet": {"title": f"t-{vid}"},
            "status": {"embeddable": embeddable, "madeForKids": kids},
            "contentDetails": {"duration": dur}}


def test_search_is_kid_safe_and_embeddable_only():
    calls = []
    s = YouTubeSearch(KEY, fetch=_fake_api(["a", "b"], {"a": _detail("a"), "b": _detail("b")}, calls))
    s.find("티니핑 노래")
    params = calls[0][1]
    assert params["safeSearch"] == "strict", "2세가 쓰는 물건이다"
    assert params["videoEmbeddable"] == "true" and params["type"] == "video"


def test_details_recheck_embeddable():
    """검색 필터를 믿지 않는다 — 막힌 영상은 error 150 으로 조용히 안 나온다."""
    details = {"a": _detail("a", embeddable=False), "b": _detail("b")}
    s = YouTubeSearch(KEY, fetch=_fake_api(["a", "b"], details, []))
    assert [v.id for v in s.candidates("x")] == ["b"]


def test_cache_saves_quota(tmp_path):
    calls = []
    fetch = _fake_api(["a"], {"a": _detail("a")}, calls)
    path = tmp_path / "c.json"
    YouTubeSearch(KEY, fetch=fetch, cache_path=path).find("동요")
    n = len(calls)
    again = YouTubeSearch(KEY, fetch=fetch, cache_path=path)      # 파일에서 다시 읽는다
    assert again.find("동요").id == "a"
    assert len(calls) == n, "캐시가 있는데 또 검색했다(검색 1회 = 100유닛)"


def test_cache_expires(tmp_path):
    calls, t = [], [1000.0]
    s = YouTubeSearch(KEY, fetch=_fake_api(["a"], {"a": _detail("a")}, calls),
                      cache_days=1, now=lambda: t[0])
    s.find("동요")
    t[0] += 2 * 86400
    s.find("동요")
    assert sum(1 for c in calls if c[0] == "search") == 2


def test_exclude_gives_the_next_one():
    s = YouTubeSearch(KEY, fetch=_fake_api(["a", "b"], {"a": _detail("a"), "b": _detail("b")}, []))
    assert s.find("x", exclude=["a"]).id == "b"


def test_missing_key_fails_before_network():
    called = []
    s = YouTubeSearch("", fetch=lambda u: called.append(u))
    with pytest.raises(YouTubeError):
        s.find("x")
    assert called == []


def test_http_error_never_leaks_the_key():
    def fetch(url):
        body = json.dumps({"error": {"message": "x", "errors": [{"reason": "quotaExceeded"}]}})
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(body.encode()))
    with pytest.raises(YouTubeError) as e:
        YouTubeSearch(KEY, fetch=fetch).find("x")
    assert "quotaExceeded" in str(e.value) and "403" in str(e.value)
    assert KEY not in str(e.value)


def test_other_errors_redact_the_key():
    def fetch(url):
        raise OSError(f"failed to open {url}")          # URL 에 키가 들어 있다
    with pytest.raises(YouTubeError) as e:
        YouTubeSearch(KEY, fetch=fetch).find("x")
    assert KEY not in str(e.value)


# ── 플레이어 ─────────────────────────────────────────────────────────────────
def test_player_meets_the_minimum_embed_size():
    """YouTube API 약관: 임베드 플레이어 최소 200×200. 09-11 시험의 320×180 은 어겼다."""
    assert PLAYER_W >= 200 and PLAYER_H >= 200
    assert f"width: '{PLAYER_W}'" in PLAYER_HTML and f"height: '{PLAYER_H}'" in PLAYER_HTML


def test_player_uses_the_official_iframe_api():
    assert '<script src="https://www.youtube.com/iframe_api">' in PLAYER_HTML
    assert "__W__" not in PLAYER_HTML and "__H__" not in PLAYER_HTML


def test_bridge_hands_out_commands_in_order():
    b = yt._Bridge()
    b.push(op="load", id="a")
    b.push(op="pause")
    assert [c["op"] for c in b.since(0)] == ["load", "pause"]
    assert [c["op"] for c in b.since(1)] == ["pause"]
    b.update({"state": 1})
    assert b.wait_for(lambda s: s.get("state") == 1, 0.1)


# ── 곡이 자연히 끝났을 때 ────────────────────────────────────────────────────
# "그만" 은 sink 를 재워 ReSpeaker 를 놓는다. 그런데 곡이 끝까지 가면 아무도 안 놓아
# PulseAudio 가 장치를 계속 쥔다 — 호출어 녹음처럼 hw 를 직접 여는 도구가 막힌다.
# 페이지는 1초마다 상태를 보내므로, 놓는 일은 **들어간 순간 한 번만** 해야 한다.
def _player_with_fake_pactl():
    p = yt.YouTubePlayer()
    calls = []
    p._pactl = lambda *a: calls.append(a)
    p._started = True
    return p, calls


def test_sink_is_released_when_the_song_ends_by_itself():
    p, calls = _player_with_fake_pactl()
    p._bridge.update({"state": yt.PLAYING, "vid": "a"})
    p._bridge.update({"state": yt.ENDED, "vid": "a"})
    assert ("suspend-sink", p.sink_name, "1") in calls


def test_sink_is_released_only_once_though_the_page_repeats_ended():
    p, calls = _player_with_fake_pactl()
    p._bridge.update({"state": yt.PLAYING, "vid": "a"})
    for _ in range(5):                       # 1초마다 같은 상태가 온다
        p._bridge.update({"state": yt.ENDED, "vid": "a"})
    assert calls.count(("suspend-sink", p.sink_name, "1")) == 1


def test_playing_never_releases_the_sink():
    p, calls = _player_with_fake_pactl()
    p._bridge.update({"state": yt.PLAYING, "vid": "a"})
    p._bridge.update({"state": yt.BUFFERING, "vid": "a"})
    assert calls == []


# ── 아동용 영상만 (2026-09-19) ─────────────────────────────────────────────
# 유튜브 키즈 목록 자체는 API 가 없다. 키즈 앱이 가져가는 풀인 '아동용(madeForKids)'
# 지정 영상만 튼다. 실검색: '티니핑 노래' 8/10·'동요' 10/10 vs '스파이더맨 OST' 0/10.

def test_아동용만_켜면_아동용이_아닌_영상은_안_튼다():
    details = {"a": _detail("a", kids=False), "b": _detail("b", kids=True)}
    s = YouTubeSearch(KEY, fetch=_fake_api(["a", "b"], details, []), kids_only=True)
    assert s.find("x").id == "b"


def test_아동용이_하나도_없으면_못_찾음():
    s = YouTubeSearch(KEY, fetch=_fake_api(["a"], {"a": _detail("a", kids=False)}, []),
                      kids_only=True)
    assert s.find("스파이더맨 OST") is None


def test_아동용만_끄면_옛_동작():
    s = YouTubeSearch(KEY, fetch=_fake_api(["a"], {"a": _detail("a", kids=False)}, []))
    assert s.find("x").id == "a"
