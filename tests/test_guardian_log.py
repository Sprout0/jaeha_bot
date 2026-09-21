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
