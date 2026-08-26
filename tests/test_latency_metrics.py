"""응답 지연 계측이 '봇이 말하는 시간'을 섞지 않는지 검증. 모델·스피커 불필요.

왜 필요한가 (2026-08-09):
  speak() 은 sd.wait() 로 재생이 끝날 때까지 막힌다. 예전엔 바깥에서 speak() 전체를
  재서 tts_s 에 넣고 그걸 응답지연에 더했다 → **봇이 말하는 시간이 지연으로 집계**됐다.
  젯슨 실측 tts 중앙 7.74s / 응답 12.87s 가 그렇게 나온 값이다.
  숫자만 틀리고 예외는 안 나므로 테스트로만 잡힌다.
"""
import json

import pytest

from app.metrics import MetricsLogger
from app.tts_module import PLAY_PAD_S, SpeakTiming


def test_speak_timing_total_is_first_plus_play():
    t = SpeakTiming(first_audio_s=1.2, synth_s=1.05, play_s=3.4)
    assert abs(t.total_s - 4.6) < 1e-9


def _log(tmp_path):
    return MetricsLogger(enabled=True, tag="test", log_dir=str(tmp_path))


def test_resp_excludes_playback(tmp_path):
    """응답지연 = STT + 생각 + 첫소리까지. 발화 시간(4.0s)은 들어가면 안 된다."""
    m = _log(tmp_path)
    m.record_turn(stt_wait_s=2.0, stt_rec_s=0.6, think_s=1.0, think_kind="llm",
                  tts_first_s=0.9, tts_play_s=4.0, reply="안녕 재하야")
    rec = m.samples[0]
    assert rec["resp_compute_s"] == 2.5, "0.6+1.0+0.9 이어야 한다"
    assert rec["tts_play_s"] == 4.0, "발화 시간은 버리지 말고 따로 남긴다"
    assert rec["metric_ver"] == 3, "옛 기록과 구분되는 표식이 있어야 한다"


def test_playback_length_does_not_change_resp(tmp_path):
    """답이 길어져 발화가 3배가 돼도 응답지연은 그대로여야 한다."""
    m = _log(tmp_path)
    for play in (1.0, 3.0, 9.0):
        m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, think_s=1.0, think_kind="llm",
                      tts_first_s=0.8, tts_play_s=play, reply="답")
    assert {r["resp_compute_s"] for r in m.samples} == {2.3}


def test_summary_reports_first_sound_and_keeps_play_separate(tmp_path, capsys):
    m = _log(tmp_path)
    for _ in range(3):
        m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, think_s=1.0, think_kind="llm",
                      tts_first_s=0.8, tts_play_s=5.0, reply="답")
    m.summary()
    out = capsys.readouterr().out
    assert "첫 소리" in out
    written = [json.loads(x) for x in
               m.path.read_text(encoding="utf-8").splitlines() if x.strip()]
    summ = [r for r in written if r.get("event") == "summary"][0]
    assert summ["resp_median_s"] == 2.3
    assert summ["tts_play_median_s"] == 5.0
    assert summ["under_3s_ratio"] == 1.0, "2.3s 는 3초 목표 안이다"


def test_disabled_logger_records_nothing(tmp_path):
    m = MetricsLogger(enabled=False, tag="test", log_dir=str(tmp_path))
    m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, think_s=1.0, think_kind="llm",
                  tts_first_s=0.8, tts_play_s=2.0)
    assert m.samples == []


def test_empty_audio_reports_no_playback():
    """합성 결과가 비면 재생이 없으니 발화 시간은 0 이어야 한다."""
    t = SpeakTiming(first_audio_s=0.4, synth_s=0.4, play_s=0.0)
    assert t.play_s == 0.0 and t.total_s == 0.4


def test_speak_does_not_count_playback_as_latency(monkeypatch):
    """🔴 이 세션에서 고친 바로 그 버그를 고정한다.

    speak() 은 재생이 끝날 때까지 막힌다. 그 대기가 first_audio_s 에 새어 들어가면
    답이 길어질수록 '응답 지연'이 늘어나는 것처럼 보인다. 가짜 sounddevice 로
    재생을 0.3초 흉내내고, 첫 소리 시각이 그 영향을 안 받는지 본다.
    """
    import sys
    import time
    import types

    import numpy as np

    from app.tts_module import TTSModule

    waited = []

    fake = types.ModuleType("sounddevice")
    fake.play = lambda *a, **k: None
    fake.wait = lambda: (time.sleep(0.3), waited.append(1))
    fake.check_output_settings = lambda **k: None
    fake.default = types.SimpleNamespace(device=(0, 0))
    fake.query_devices = lambda *a, **k: {"default_samplerate": 16000}
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    tts = TTSModule()
    tts.sample_rate = 16000
    monkeypatch.setattr(tts, "load", lambda: None)
    monkeypatch.setattr(tts, "_infer", lambda text: np.zeros(1600, dtype=np.float32))
    monkeypatch.setattr(tts, "_trim", lambda a, **k: a)
    monkeypatch.setattr(tts, "_resolve_play_rate", lambda sd: 16000)

    t = tts.speak("안녕")

    assert waited == [1], "재생 완료까지 기다리긴 해야 한다(에코 쿨다운 전제)"
    assert t.play_s >= 0.25, f"발화 시간은 재생 대기를 담아야 한다: {t.play_s}"
    assert t.first_audio_s < 0.25, (
        f"첫 소리까지에 재생 대기가 새어 들어갔다: {t.first_audio_s}")
    assert t.first_audio_s >= PLAY_PAD_S, "앞에 덧댄 무음 동안은 아직 소리가 없다"


def test_speak_with_empty_audio_skips_playback(monkeypatch):
    """합성이 비면 재생을 시도하지 않는다(장치 없는 환경에서 죽지 않게)."""
    import sys
    import types

    import numpy as np

    from app.tts_module import TTSModule

    fake = types.ModuleType("sounddevice")
    fake.play = lambda *a, **k: (_ for _ in ()).throw(AssertionError("재생하면 안 됨"))
    fake.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    tts = TTSModule()
    monkeypatch.setattr(tts, "load", lambda: None)
    monkeypatch.setattr(tts, "_infer", lambda text: np.zeros(0, dtype=np.float32))
    monkeypatch.setattr(tts, "_trim", lambda a, **k: a)

    t = tts.speak("")
    assert t.play_s == 0.0


# ── 말끝 무음은 앞과 따로다 ──────────────────────────────────────────────────
# 🔴 2026-08-26 젯슨 청취: 답변도 말끝이 끊긴다. 필러 6개를 비교해 보니 꼬리 무음이
#    472~719ms 인데도 `응, 응` 은 두 번째 음절을 통째로 잃었다 — 장치가 끝에서 그만큼
#    흘린다는 뜻이다. 앞(0.15s)과 같은 값으로는 못 덮는다.
# ⚠️ 앞과 달리 뒤는 **공짜가 아니다.** speak() 은 sd.wait() 로 버퍼 끝까지 기다리므로
#    늘린 만큼 봇이 늦게 듣기 시작한다(체감 응답속도가 아니라 다음 턴 진입이 늦어짐).
#    그래서 필러(0.6s)보다 짜게 잡고, --measure 로 실측되면 그 값으로 내린다.

def test_the_tail_silence_is_longer_than_the_head():
    from app.tts_module import PLAY_PAD_S, PLAY_TAIL_PAD_S

    assert PLAY_TAIL_PAD_S > PLAY_PAD_S


def test_the_tail_silence_can_be_tuned_per_machine():
    """젯슨에서 실측한 값으로 코드 수정 없이 내려야 한다."""
    from app.tts_module import TTSModule

    assert TTSModule(play_tail_pad_s=0.35).play_tail_pad_s == 0.35


def test_the_head_silence_still_decides_when_the_first_sound_lands():
    """뒤를 늘렸다고 첫 소리 계산까지 흔들리면 지연 측정이 통째로 망가진다."""
    from app.tts_module import PLAY_PAD_S, PLAY_TAIL_PAD_S, TTSModule
    import inspect

    src = inspect.getsource(TTSModule.speak)

    assert "+ PLAY_PAD_S" in src, "첫 소리에는 **앞** 무음만 더해야 한다"
    assert "+ PLAY_TAIL_PAD_S" not in src


# ── 필러 대기도 아이가 겪는 시간이다 ────────────────────────────────────────
# 🔴 await_quiet() 은 think 측정이 끝난 뒤, speak 이 시작되기 **전**에 잔다. 그래서
#    꼬리+인식+생각+첫소리 어디에도 안 잡힌다 — 아이는 기다렸는데 지표는 모른다.
#    metrics.py 가 08-09·08-13 에 두 번 당한 것과 **같은 종류의 실수**다
#    (이름과 내용이 어긋나 아무도 못 보는 구간이 생긴다).

def test_the_filler_wait_lands_in_the_felt_time(tmp_path):
    m = MetricsLogger(log_dir=str(tmp_path))
    m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, vad_tail_s=1.2,
                  think_s=1.0, think_kind="llm", tts_first_s=0.8,
                  filler_wait_s=0.3)

    rec = m.samples[-1]
    assert rec["filler_wait_s"] == 0.3
    assert rec["resp_felt_s"] == pytest.approx(1.2 + 0.5 + 1.0 + 0.8 + 0.3)


def test_no_filler_wait_leaves_the_number_where_it_was(tmp_path):
    """안 기다린 턴은 예전과 같은 값이어야 한다 — 지표가 통째로 어긋나면 안 된다."""
    m = MetricsLogger(log_dir=str(tmp_path))
    m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, vad_tail_s=1.2,
                  think_s=1.0, think_kind="llm", tts_first_s=0.8)

    assert m.samples[-1]["resp_felt_s"] == pytest.approx(3.5)
