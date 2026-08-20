"""VAD 꼬리 동안 **답까지 미리 만들어 두는지**.

🔴 왜: 실기 15턴에서 체감 4.53s 중 '생각'이 1.71s(38%)로 가장 큰 덩어리다. 그런데
   그 앞의 VAD 꼬리 1.20s 는 정의상 무음이라 **아무것도 안 하고 기다리는 시간**이다.
   선행 인식(b3db02c)이 그 안에 STT 를 밀어넣어 -0.85s 를 벌었다. 같은 수를 한 단계
   더 쓴다 — 인식이 꼬리 안에서 끝나면 그 텍스트로 LLM 을 곧바로 미리 친다.

🔴 안전장치는 **키가 '아이가 한 말 그 자체'** 라는 것이다. 최종 인식이 다르면 키가
   안 맞아 추측이 그냥 버려진다 — 틀린 추측이 답으로 나가는 일이 구조적으로 불가능하다.
   (STT 선행 인식이 '마지막 말소리까지 모인 샘플 수'를 키로 쓰는 것과 같은 장치다.)
⚠️ 버린 추측의 손해는 API 호출 한 번이고 지연은 0 이다 — 어차피 기다리는 중이었다.
"""
import threading
import time

import pytest

from app.agent import LLMAgent


class _Agent(LLMAgent):
    """_complete 만 가짜로. 나머지(후처리·이력·추측 배선)는 진짜 코드를 탄다."""

    def __init__(self, reply="응 좋아!", delay=0.0, **kw):
        super().__init__(model_path="없어도된다.gguf", system_prompt="시스템", **kw)
        self._reply = reply
        self._delay = delay
        self.calls = []
        self._lock = threading.Lock()

    def _complete(self, messages):
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.calls.append(messages[-1]["content"])
        return self._reply


def test_speculated_answer_is_reused_without_calling_again():
    a = _Agent()
    a.speculate("강아지 좋아")

    out = a.respond("강아지 좋아")

    assert out["text"] == "응 좋아!"
    assert a.calls == ["강아지 좋아"], "미리 만들어 뒀는데 또 불렀다 — 겹치기가 무의미해진다"


def test_a_different_final_transcript_discards_the_guess():
    """🔴 아이가 말을 이어가 인식이 바뀌면, 옛 추측은 절대 쓰이면 안 된다."""
    a = _Agent()
    a.speculate("강아지")

    out = a.respond("강아지 좋아해")

    assert out["text"] == "응 좋아!"
    assert a.calls[-1] == "강아지 좋아해", "옛 추측의 답이 새 발화의 답으로 나갔다"


def test_works_normally_without_any_speculation():
    a = _Agent()

    assert a.respond("안녕")["text"] == "응 좋아!"
    assert a.calls == ["안녕"]


def test_speculation_does_not_touch_history():
    """추측은 '아직 일어나지 않은 턴'이다. 이력에 남으면 맥락이 오염된다."""
    a = _Agent()
    a.speculate("강아지 좋아")
    time.sleep(0.05)

    assert a.history == []

    a.respond("강아지 좋아")
    assert len(a.history) == 2, "정작 진짜 턴은 이력에 남아야 한다"


def test_the_guess_uses_the_history_that_exists_at_guess_time():
    a = _Agent()
    a.respond("첫 턴")
    a.speculate("둘째 턴")
    time.sleep(0.05)

    out = a.respond("둘째 턴")
    assert out["text"] == "응 좋아!"
    assert a.calls == ["첫 턴", "둘째 턴"]


def test_speculation_failure_falls_back_to_the_normal_path():
    """최적화가 본 기능을 죽이면 안 된다."""

    class Boom(_Agent):
        def _complete(self, messages):
            if threading.current_thread().name.startswith("llm-spec"):
                raise RuntimeError("추측 실패")
            return super()._complete(messages)

    a = Boom()
    a.speculate("안녕")
    time.sleep(0.05)

    assert a.respond("안녕")["text"] == "응 좋아!"


def test_respond_waits_for_a_guess_still_in_flight():
    """꼬리보다 생각이 길면 추측이 아직 돌고 있다. 그때 두 번 치면 안 된다.

    ⚠️ 로컬 llama.cpp 는 스레드 안전하지 않다 — 동시에 두 번 부르면 위험하다.
    """
    a = _Agent(delay=0.3)
    a.speculate("안녕")
    time.sleep(0.2)          # 꼬리가 흐르는 동안 추측이 먼저 달린다

    t = time.perf_counter()
    out = a.respond("안녕")
    waited = time.perf_counter() - t

    assert out["text"] == "응 좋아!"
    assert len(a.calls) == 1, "돌고 있는 추측을 안 기다리고 또 쳤다"
    assert waited < 0.25, f"추측이 꼬리 동안 벌어놓은 시간을 못 쓰고 있다({waited:.2f}s)"


def test_vision_context_skips_the_guess():
    """추측은 카메라 상태 없이 만들어졌다. 비전이 붙는 턴에 재사용하면 안 된다."""
    a = _Agent()
    a.speculate("이게 뭐야")

    a.respond("이게 뭐야", vision_context="빨간 공")

    assert a.calls[-1].startswith("[카메라 상태: 빨간 공]")


def test_empty_text_is_not_speculated():
    a = _Agent()
    a.speculate("")
    a.speculate("   ")
    time.sleep(0.05)

    assert a.calls == []
