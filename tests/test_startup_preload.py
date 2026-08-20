"""기동 예열이 `warm()` 의 결정을 덮어쓰지 않는지.

🔴 2026-08-20 실기 로그에서 잡혔다. 같은 기동에서 5초 사이에 두 줄이 연달아 찍혔다:
     10:23:25 [agent] 로컬 폴백은 올리지 않는다(원격 정상 — 메모리 약 2.3GB 절약)
     10:23:30 [agent] LLM 로딩 중: models/exaone-3.5-2.4b-q4.gguf
   `warm()` 은 올바로 '안 올린다'고 정했는데, 바로 뒤 `app/main.py` 의 미리-로딩
   단계가 `agent._ensure_loaded()` 를 무조건 불러 **그 결정을 덮어썼다.**
   커밋 ca260c7 이 warm() 만 고치고 이 호출부를 놓쳤다 — 즉 2,332MB 회수는
   **한 번도 실제로 일어난 적이 없다.** 로그에 '절약했다'고 찍히기까지 해서
   더 안 보였다.
⚠️ 젯슨 시스템 메모리가 6,801/7,607MB(89%)까지 찬 상태라 이 2.3GB 가 크다.
"""
import types

from app.main import preload


class _FakeStage:
    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True


class _FakeAgent:
    """_ensure_loaded 가 불리면 기록한다 — 그게 바로 그 버그다."""

    def __init__(self):
        self.ensure_calls = 0

    def _ensure_loaded(self):
        self.ensure_calls += 1
        return types.SimpleNamespace()


def test_stt_and_tts_are_loaded():
    stt, tts, agent = _FakeStage(), _FakeStage(), _FakeAgent()

    preload(stt, tts, agent)

    assert stt.loaded and tts.loaded


def test_preload_never_forces_the_local_llm_up():
    """🔴 여기서 GGUF 를 올리면 warm() 의 '안 올린다' 결정이 무의미해진다."""
    stt, tts, agent = _FakeStage(), _FakeStage(), _FakeAgent()

    preload(stt, tts, agent)

    assert agent.ensure_calls == 0, (
        "미리-로딩이 로컬 LLM 을 강제로 올렸다 — warm() 이 아낀 2.3GB 를 도로 문다")
