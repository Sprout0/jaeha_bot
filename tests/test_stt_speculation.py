"""VAD 꼬리 동안 인식을 미리 돌린다(선행 인식).

🔴 왜 되는가 (2026-08-19):
   말이 끝났다고 판단하려면 silence_duration(1.2s) 을 더 들어야 하는데, **그 1.2초는
   정의상 무음이다.** 즉 마지막 말소리 프레임에서 이미 발화 전체를 손에 쥐고 있다.
   그때부터 인식을 돌리면 꼬리가 끝날 무렵 결과가 나와 있다.
   게다가 transcribe() 는 _trim_edges 로 앞뒤 무음을 어차피 잘라내므로, 꼬리를
   포함해 넣든 안 넣든 **whisper 가 보는 입력이 같다** — 결과가 달라지지 않는다.

   젯슨 실측(08-19 세션 14턴): 꼬리 1.20s · 인식 1.41s. 겹치면 인식이 0.21s 가 된다.

⚠️ 아이가 다시 말하면 그 추측은 버린다. 손해는 GPU 한 번이고 **지연은 0** 이다 —
   어차피 기다리는 중이었다. 버린 뒤 새 키로 다시 추측하므로 재개해도 손해가 없다.
"""
import threading
import time

import numpy as np

from app.audio_source import FRAME
from app.stt_module import SAMPLE_RATE, STTModule, _Speculation


def _loud(n):
    return [np.full(FRAME, 0.3, dtype=np.float32) for _ in range(n)]


def _quiet(n):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n)]


class ScriptedSource:
    def __init__(self, frames, noise_floor=0.001):
        self._frames = list(frames)
        self._i = 0
        self.noise_floor = noise_floor

    def read(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        return f


# ── _Speculation 단위 ────────────────────────────────────────────────────────

def test_result_is_returned_for_the_key_it_was_submitted_with():
    s = _Speculation(lambda a: ("안녕", 1.4))
    s.submit(np.zeros(10, dtype=np.float32), key=100)

    assert s.take(100) == ("안녕", 1.4)


def test_a_stale_key_is_not_reused():
    """아이가 다시 말했으면 그때까지의 추측은 답이 아니다."""
    s = _Speculation(lambda a: ("멍멍", 1.4))
    s.submit(np.zeros(10, dtype=np.float32), key=100)
    s.take(100)

    assert s.take(250) is None, "지난 발화의 인식 결과를 새 발화의 답으로 썼다"


def test_take_waits_for_work_still_in_flight():
    """꼬리보다 인식이 길면 기다려야 한다 — 안 기다리면 두 번 돌린다."""
    started = threading.Event()

    def slow(_a):
        started.set()
        time.sleep(0.3)
        return ("느린 답", 0.3)

    s = _Speculation(slow)
    s.submit(np.zeros(10, dtype=np.float32), key=7)
    started.wait(2.0)

    assert s.take(7) == ("느린 답", 0.3)


def test_the_same_key_is_never_recognised_twice():
    calls = []

    def work(_a):
        calls.append(1)
        return ("응", 0.1)

    s = _Speculation(work)
    for _ in range(3):
        s.submit(np.zeros(10, dtype=np.float32), key=42)
    s.take(42)

    assert len(calls) == 1, f"같은 오디오를 {len(calls)}번 인식했다 — GPU 낭비"


def test_a_crash_in_the_background_never_escapes():
    """계측·최적화가 본 기능을 죽이면 안 된다. 실패하면 그냥 평소 경로로 간다."""
    s = _Speculation(lambda a: 1 / 0)
    s.submit(np.zeros(10, dtype=np.float32), key=1)

    assert s.take(1) is None


# ── listen() 통합 ────────────────────────────────────────────────────────────

def test_listen_recognises_once_not_twice(monkeypatch):
    """선행 인식을 해 놓고 또 인식하면 아무것도 못 번다."""
    stt = STTModule(silence_duration=0.8, max_duration=5.0)
    calls = []

    def fake(audio, initial_prompt=None):
        calls.append(audio.size)
        return "동물 소리 놀이 하자", 1.4

    monkeypatch.setattr(stt, "transcribe", fake)
    src = ScriptedSource(_loud(4) + _quiet(20))

    text, _ = stt.listen(source=src)

    assert text == "동물 소리 놀이 하자"
    assert len(calls) == 1, f"인식을 {len(calls)}번 했다 — 꼬리에 숨긴 값이 안 쓰였다"


def test_listen_reports_the_wait_not_the_compute_time(monkeypatch):
    """🔴 계측이 연산시간(1.4s)을 그대로 찍으면 개선이 지표에 안 보인다.

    아이가 겪는 건 '녹음이 끝난 뒤 얼마나 더 기다렸나'다. 꼬리에 숨겼으면 그 값은
    0 에 가까워야 하고, stt_rec_s 는 그 값이어야 한다.
    """
    stt = STTModule(silence_duration=0.8, max_duration=5.0)
    monkeypatch.setattr(stt, "transcribe",
                        lambda audio, initial_prompt=None: ("응", 1.4))
    src = ScriptedSource(_loud(4) + _quiet(20))

    _text, wait_s = stt.listen(source=src)

    assert wait_s < 0.5, f"기다린 시간이 아니라 연산시간({wait_s}s)을 보고했다"


def test_resumed_speech_gets_the_full_utterance(monkeypatch):
    """꼬리 중간에 아이가 다시 말하면 **이어진 발화 전체**를 인식해야 한다."""
    stt = STTModule(silence_duration=0.8, max_duration=6.0)
    seen = []

    def fake(audio, initial_prompt=None):
        seen.append(audio.size)
        return "전체", 0.01

    monkeypatch.setattr(stt, "transcribe", fake)
    # 말 -> 잠깐 쉼 -> 다시 말 -> 진짜 끝
    src = ScriptedSource(_loud(3) + _quiet(4) + _loud(3) + _quiet(20))

    text, _ = stt.listen(source=src)

    assert text == "전체"
    # 마지막으로 쓰인 인식은 뒤 발화까지 포함한 길이여야 한다
    assert max(seen) >= 9 * FRAME, "앞부분만 인식한 결과를 답으로 썼다"


# ── 꼬리 도중 추측 텍스트 흘려주기 (2026-08-20) ───────────────────────────────
# 🔴 왜: 선행 인식이 꼬리 안에서 끝나면, 그 시점에 **아이가 한 말을 이미 알고 있다.**
#    그런데 지금은 그걸 listen() 이 끝날 때까지 쥐고만 있다. 남은 꼬리 동안 LLM 을
#    미리 칠 수 있는데 놀리는 것이다(실기 15턴: 생각 1.71s 가 체감의 38%).
#    on_partial 로 흘려주면 main 이 agent.speculate() 를 걸 수 있다.
# ⚠️ 흘려준 텍스트가 최종과 다를 수 있다(아이가 이어 말하면). 그건 받는 쪽이
#    '말 자체'를 키로 써서 거른다 — app/agent.py _Speculation 참조.

def test_partial_text_is_handed_over_during_the_tail():
    seen = []
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("칼 어딨어", 0.0))

    stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                             on_partial=seen.append)

    assert seen == ["칼 어딨어"], "꼬리 동안 인식 결과를 안 흘려줬다"


def test_partial_is_handed_over_only_once():
    """매 프레임마다 부르면 LLM 을 꼬리 동안 수십 번 친다."""
    seen = []
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("안녕", 0.0))

    stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                             on_partial=seen.append)

    assert len(seen) == 1


def test_empty_recognition_is_not_handed_over():
    seen = []
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("", 0.0))

    stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                             on_partial=seen.append)

    assert seen == []


def test_a_broken_callback_does_not_kill_the_recording():
    """최적화가 본 기능을 죽이면 안 된다."""
    def boom(_):
        raise RuntimeError("콜백 터짐")

    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("안녕", 0.0))

    audio = stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                                     on_partial=boom)

    assert audio.size > 0


def test_recording_still_works_without_a_callback():
    stt = STTModule(silence_duration=1.2, max_duration=10.0)

    audio = stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)))

    assert audio.size > 0


def test_peek_does_not_block_when_nothing_is_ready():
    s = _Speculation(lambda a: ("느림", 1.0))

    t = time.perf_counter()
    assert s.peek(999) is None
    assert time.perf_counter() - t < 0.05, "peek 가 기다리고 있다 — 꼬리 루프가 멈춘다"


# ── 왜 못 넘겼는지 남긴다 ────────────────────────────────────────────────────
# 🔴 2026-08-27 젯슨 실기 8턴: 선행생각 발동 **0/8**, 이유는 전부 `no_guess` —
#    즉 추측이 시작조차 안 됐다. 그런데 그게 (a)선행 인식이 꼬리 안에 못 끝나서인지
#    (b)끝났는데 빈 문자열이라 안 넘긴 건지 **로그로 갈 수가 없었다.**
#    고칠 곳이 다르다: (a)는 STT 를 더 줄여야 하고 (b)는 VAD·발화가 문제다.
#    ⚠️ 그리고 (a)면 **얼마나 늦었는지**가 다음 결정의 전부다 — 0.02초 차이면
#       spec_after 를 당기면 되고, 0.5초면 다른 수를 찾아야 한다.

def test_a_handover_says_so(caplog):
    import logging
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("칼 어딨어", 0.0))

    with caplog.at_level(logging.INFO, logger="jaeha_bot.stt"):
        stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                                 on_partial=lambda t: None)

    assert any("선행인식" in r.message and "끝남" in r.message for r in caplog.records)


def test_a_late_guess_says_how_late(caplog):
    """예산을 얼마나 넘겼는지가 다음 결정의 전부다."""
    import logging
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: (_ for _ in ()).throw(RuntimeError("안 끝남")))

    with caplog.at_level(logging.INFO, logger="jaeha_bot.stt"):
        stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                                 on_partial=lambda t: None)

    msgs = [r.message for r in caplog.records if "선행인식" in r.message]
    assert msgs, "못 넘겼는데 아무 말도 안 남겼다"
    assert "못 끝냈다" in msgs[0] or "빈 문자열" in msgs[0]
    # ⚠️ 여기서 '몇 초 늦었다'를 요구하면 안 된다. 이 시점엔 인식이 안 끝나서 모른다 —
    #    처음엔 찍었다가 매번 예산과 똑같은 수(1.04)만 나왔다(동어반복). listen() 이 찍는다.
    assert not any("경과" in m for m in msgs)


def test_an_empty_guess_is_told_apart_from_a_late_one(caplog):
    import logging
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("", 0.0))

    with caplog.at_level(logging.INFO, logger="jaeha_bot.stt"):
        stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                                 on_partial=lambda t: None)

    assert any("빈 문자열" in r.message for r in caplog.records), \
        "빈 결과를 '늦었다'로 뭉뚱그리면 엉뚱한 걸 고치게 된다"


def test_nothing_is_logged_when_nobody_is_listening(caplog):
    """on_partial 이 없으면 선행 인식을 흘릴 데가 없다 — 그 turn 을 실패로 세면 안 된다."""
    import logging
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("안녕", 0.0))

    with caplog.at_level(logging.INFO, logger="jaeha_bot.stt"):
        stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)))

    assert not [r for r in caplog.records if "선행인식" in r.message]


# ── 얼마나 아깝게 놓쳤나 ────────────────────────────────────────────────────
# 🔴 2026-08-27: 처음 붙인 로그가 **동어반복**이었다. "제출 후 1.04s 경과"는 제출부터
#    녹음 종료까지의 시간인데, 그건 정의상 `silence_duration - 0.16` = 예산 그 자체다.
#    매번 똑같은 1.04 가 찍혀 '얼마나 늦었나'를 하나도 못 알려줬다.
#    ➡️ 진짜 답은 `_spec.take()` 가 돌려주는 **실제 인식 시간**과, 녹음이 끝난 뒤
#       아이가 **더 기다린 시간**이다. 후자가 0.1초면 spec_after 를 당기면 되고,
#       0.5초면 다른 수를 찾아야 한다. 실기에서 그 값이 0.08~0.15s 였다.

def test_the_loop_records_whether_it_handed_over():
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("칼 어딨어", 0.0))

    stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                             on_partial=lambda t: None)

    assert stt.last_spec_handed is True


def test_no_handover_is_recorded_too():
    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    stt._spec = _Speculation(lambda a: ("", 0.0))

    stt.record_until_silence(source=ScriptedSource(_loud(4) + _quiet(40)),
                             on_partial=lambda t: None)

    assert stt.last_spec_handed is False


def test_a_near_miss_says_how_near(caplog, monkeypatch):
    """아깝게 놓친 폭을 숫자로 남긴다 — 이게 spec_after 를 당길지 정하는 유일한 근거다."""
    import logging

    stt = STTModule(silence_duration=1.2, max_duration=10.0)
    # 인식은 성공하지만 꼬리 안에는 못 끝난 척: 넘겨줄 텍스트가 없다가 뒤늦게 생긴다.
    def _work(a):
        time.sleep(0.05)            # 꼬리(즉시 끝나는 스크립트 소스) 안에는 못 끝난다
        return ("늦게 나온 답", 1.12)

    stt._spec = _Speculation(_work)
    with caplog.at_level(logging.INFO, logger="jaeha_bot.stt"):
        stt.listen(source=ScriptedSource(_loud(4) + _quiet(40)),
                   on_partial=lambda t: None)

    msgs = [r.message for r in caplog.records if "선행인식" in r.message]
    assert any("1.12" in m for m in msgs), f"실제 인식 시간을 안 남겼다: {msgs}"
