"""노래 틀기가 켜지면 '노래 약속'은 날조가 아니다. (2026-09-11)

claims 는 '할 수 있는 것'을 코드·파일에서 읽는다. 유튜브를 켜면 등록된 노래는 파일이
없어도 틀 수 있으므로, 그때 "곰 세 마리"를 날조로 세면 정직한 답에 벌점을 준다.
"""
from app import claims


def _with_music(monkeypatch, on: bool):
    models = dict(claims.settings.models)
    models["youtube"] = {"enabled": on}
    monkeypatch.setattr(claims.settings, "models", models)


def test_songs_without_files_are_available_when_music_is_on(monkeypatch):
    _with_music(monkeypatch, True)
    assert "곰 세 마리" in claims.available_names()
    assert "곰 세 마리" not in claims.unavailable_names()
    assert claims.find_fabrications("곰 세 마리 틀어줘, 라고 말해 줘!") == []


def test_music_off_keeps_the_old_judgement(monkeypatch):
    _with_music(monkeypatch, False)
    assert "곰 세 마리" in claims.unavailable_names()


def test_music_does_not_make_unimplemented_games_available(monkeypatch):
    """노래만 바뀐다 — 색칠 놀이는 여전히 못 한다."""
    _with_music(monkeypatch, True)
    assert "색칠 놀이 스무고개" in claims.unavailable_names()
