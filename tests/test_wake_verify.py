"""2단계(감지 → 검증) 구조 검증. 마이크·모델·whisper 불필요.

왜 필요한가 (2026-08-12 실측, 실음성 44건):
  1단계 임계를 0.25 -> 0.03 으로 낮추면 실음성 재현율이 **30% -> 82%** 로 오른다.
  모델이 못 듣는 게 아니라 확신이 없을 뿐이라 신호는 살아 있다. 대신 헛깨움이
  시간당 13.6회로 폭발하므로, 후보를 whisper 로 한 번 더 검증해 걸러낸다.
  캐스케이드 실측 = 재현율 80% / 헛깨움 추정 0.27회·시간.

여기서 고정하는 것 — 전부 '조용히 망가지는' 자리다:
  · verifier=None 이면 **기존 동작과 완전히 동일**해야 한다(롤백 보장)
  · 기각 후 재호출 폭주 방지(히스테리시스 + 쿨다운). 없으면 초당 12번 whisper 를 부른다
  · 검증 버퍼(2.0s)와 프리롤(0.5s)이 서로 오염되지 않아야 한다
"""
import numpy as np

from app.audio_source import FRAME, SAMPLE_RATE, AudioSource
from app.wake_onnx import OnnxWakeDetector


class FakeSource(AudioSource):
    """프레임을 미리 정해 두고 흘려보내는 소스. 마이크를 안 쓴다."""

    def __init__(self, frames, **kw):
        super().__init__(**kw)
        self._frames = list(frames)
        self._i = 0
        self.noise_floor = 0.0

    def _read_frame(self):
        if self._i >= len(self._frames):
            raise StopIteration
        f = self._frames[self._i]
        self._i += 1
        return f

    def _available(self):
        return FRAME if self._i < len(self._frames) else 0


def _det(scores, verifier=None, **kw):
    """점수 수열을 그대로 뱉는 감지기. ONNX 를 로드하지 않는다."""
    d = OnnxWakeDetector.__new__(OnnxWakeDetector)
    seq = list(scores)
    frames = [np.full(FRAME, 0.2, dtype=np.float32) for _ in range(len(seq) + 20)]
    src = FakeSource(frames, preroll=0.5, verify_window=2.0)
    d._init_state(threshold=kw.pop("threshold", 0.03), trigger_frames=1,
                  continuation_window=0.0, source=src, verifier=verifier,
                  verify_cooldown_s=kw.pop("verify_cooldown_s", 1.0))
    for k, v in kw.items():
        setattr(d, k, v)
    it = iter(seq)
    d.push = lambda frame: next(it, 0.0)
    return d, src


# ── 롤백 보장 ────────────────────────────────────────────────────────────
def test_no_verifier_behaves_exactly_like_before():
    """verifier 가 없으면 점수만으로 깨어난다 — 설정 한 줄로 되돌아갈 수 있어야 한다."""
    d, _ = _det([0.0, 0.9], verifier=None)
    r = d.wait_for_wake(max_frames=5)
    assert r is not None and r.score == 0.9


def test_verifier_none_does_not_touch_the_audio_source():
    """롤백 상태에서는 검증 버퍼를 읽지도 말아야 한다(부작용 0)."""
    d, src = _det([0.9], verifier=None)
    src.verify_window = lambda: (_ for _ in ()).throw(AssertionError("읽으면 안 된다"))
    assert d.wait_for_wake(max_frames=3) is not None


# ── 2단계 판정 ───────────────────────────────────────────────────────────
def test_verifier_pass_wakes():
    d, _ = _det([0.9], verifier=lambda a: True)
    assert d.wait_for_wake(max_frames=3) is not None


def test_verifier_reject_does_not_wake():
    """🔴 여기가 헛깨움을 막는 자리다."""
    d, _ = _det([0.9] + [0.0] * 5, verifier=lambda a: False)
    assert d.wait_for_wake(max_frames=6) is None


def test_verifier_sees_the_long_window_not_the_preroll():
    """호출어 전체(약 1초)를 보려면 2.0초 창이어야 한다. 0.5초 프리롤로는 잘린다.

    앞에 조용한 프레임을 흘려 링버퍼를 채운다 — 실기에서도 감지기는 계속 듣고 있어
    후보가 뜨는 시점엔 두 버퍼가 이미 차 있다.
    """
    seen = {}
    d, _ = _det([0.0] * 30 + [0.9],
                verifier=lambda a: seen.setdefault("n", a.size) is not None)
    d.wait_for_wake(max_frames=40)
    assert seen["n"] > 0.5 * SAMPLE_RATE, f"검증에 넘긴 오디오가 너무 짧다: {seen['n']}"
    assert abs(seen["n"] - 2.0 * SAMPLE_RATE) <= FRAME, "2.0초 창이 아니다"


# ── 폭주 방지 ────────────────────────────────────────────────────────────
def test_rejected_candidate_does_not_refire_every_frame():
    """🔴 이게 없으면 TV 소리 한 문장에 whisper 를 초당 12번 부른다.

    임계(0.03)를 계속 넘는 점수가 이어져도, 한 번 기각했으면 점수가 임계 아래로
    내려갔다 오기 전에는 다시 부르지 않아야 한다.
    """
    calls = []
    d, _ = _det([0.9] * 12, verifier=lambda a: calls.append(1) or False)
    d.wait_for_wake(max_frames=12)
    assert len(calls) == 1, f"검증기를 {len(calls)}번 불렀다 — 히스테리시스가 없다"


def test_score_dropping_below_threshold_rearms_the_verifier():
    """점수가 한 번 내려갔다 오면 그건 새로운 발화다 — 다시 검증해야 한다."""
    calls = []
    d, _ = _det([0.9, 0.9, 0.0, 0.0, 0.9], verifier=lambda a: calls.append(1) or False,
                verify_cooldown_s=0.0)
    d.wait_for_wake(max_frames=8)
    assert len(calls) == 2, f"재무장이 안 됐다(호출 {len(calls)}회)"


def test_cooldown_blocks_refire_even_after_score_drops():
    """쿨다운 안에서는 점수가 내려갔다 와도 참는다(연속 오탐 방어)."""
    calls = []
    d, _ = _det([0.9, 0.0, 0.9, 0.0, 0.9], verifier=lambda a: calls.append(1) or False,
                verify_cooldown_s=99.0)
    d.wait_for_wake(max_frames=8)
    assert len(calls) == 1, f"쿨다운이 안 먹었다(호출 {len(calls)}회)"


def test_verifier_error_does_not_crash_the_bot():
    """whisper 가 터져도 봇은 계속 들어야 한다. 안전하게 '기각'으로 본다."""
    def boom(_a):
        raise RuntimeError("whisper 죽음")
    d, _ = _det([0.9] + [0.0] * 4, verifier=boom)
    assert d.wait_for_wake(max_frames=6) is None


# ── 버퍼 분리 ────────────────────────────────────────────────────────────
def test_two_rings_have_independent_lengths():
    """프리롤 0.5초를 늘리면 호출 직전 TV·부모 말소리가 섞여 whisper 가 환각한다.
    그래서 검증용 창은 **따로** 둔다."""
    src = FakeSource([np.full(FRAME, 0.1, dtype=np.float32)] * 60,
                     preroll=0.5, verify_window=2.0)
    for _ in range(50):
        src.read()
    assert abs(src.preroll().size - 0.5 * SAMPLE_RATE) <= FRAME
    assert abs(src.verify_window().size - 2.0 * SAMPLE_RATE) <= FRAME


def test_clear_preroll_also_clears_verify_window():
    """낡은 오디오를 검증에 쓰면 안 된다 — 비울 땐 같이 비운다."""
    src = FakeSource([np.full(FRAME, 0.1, dtype=np.float32)] * 60,
                     preroll=0.5, verify_window=2.0)
    for _ in range(40):
        src.read()
    src.clear_preroll()
    assert src.verify_window().size == 0


# ── 에너지 게이트 ────────────────────────────────────────────────────────
# 🔴 2026-08-12 실기: 2단계를 켰더니 헛깨움이 늘었다. 원인 하나가 확인됐다 —
#    initial_prompt 는 whisper 를 '재하봇' 쪽으로 편향시키는데, 조용한 구간에서는
#    그 편향이 그대로 환각이 된다(방 소음 41건 중 1건이 자모거리 0.00 '재하봇').
#    말소리 세기에 못 미치면 whisper 를 부르지도 않는다.

def _quiet_det(level, verifier, floor=0.0):
    d = OnnxWakeDetector.__new__(OnnxWakeDetector)
    frames = [np.full(FRAME, level, dtype=np.float32) for _ in range(40)]
    src = FakeSource(frames, preroll=0.5, verify_window=2.0)
    src.noise_floor = floor
    d._init_state(threshold=0.03, trigger_frames=1, continuation_window=0.0,
                  source=src, verifier=verifier, verify_cooldown_s=0.0)
    it = iter([0.0] * 30 + [0.9])
    d.push = lambda frame: next(it, 0.0)
    return d


def test_silent_window_never_reaches_whisper():
    """디지털 무음에는 whisper 를 쓰지 않는다(절약 장치). 환각 방어는 두 번 대조가 한다."""
    calls = []
    d = _quiet_det(0.001, lambda a: calls.append(1) or True)
    assert d.wait_for_wake(max_frames=40) is None
    assert calls == [], "무음인데 whisper 를 불렀다"


def test_speech_level_window_does_reach_whisper():
    """말소리 세기면 정상적으로 검증한다 — 게이트가 진짜 호출을 막으면 안 된다."""
    calls = []
    d = _quiet_det(0.05, lambda a: calls.append(1) or True)
    assert d.wait_for_wake(max_frames=40) is not None
    assert calls == [1]


def test_gate_does_not_scale_with_noise_floor():
    """🔴 2026-08-12 실기 사고를 고정한다.

    게이트를 '소음 바닥 × 2' 로 두면, 유튜브를 틀어 바닥이 0.005 -> 0.024 로 오를 때
    기준이 0.0485 가 되어 **진짜 호출**(최대프레임 RMS 최소 0.0515)을 잘라 버린다.
    실제로 점수 0.103 짜리 후보가 RMS 0.0344 로 검증도 못 받고 버려졌다.
    시끄러울수록 귀를 닫는 설계였다 — 바닥에 비례시키지 않는다.
    """
    calls = []
    d = _quiet_det(0.02, lambda a: calls.append(1) or True, floor=0.03)
    assert d.wait_for_wake(max_frames=40) is not None, "시끄럽다고 진짜 호출을 막았다"
    assert calls == [1]


# ── 설정 정합성 ──────────────────────────────────────────────────────────
def test_shipped_config_pairs_threshold_with_verify():
    """🔴 임계값과 verify.enabled 는 **항상 같이** 움직여야 한다.

    2단계를 끄면서 임계를 0.03 에 두면 1단계 혼자 시간당 13.6회 깨어난다.
    반대로 2단계를 켜면서 임계를 0.25 에 두면 재현율이 30% 에 머물러(실음성 실측)
    2단계를 붙인 의미가 없다. 한 줄만 되돌리는 사고를 여기서 막는다.
    """
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    o = cfg["wake"]["onnx"]
    thr = float(o["threshold"])
    on = bool((o.get("verify") or {}).get("enabled", False))
    if on:
        assert thr <= 0.10, f"2단계가 켜졌는데 1단계 임계가 높다({thr}) — 2단계가 무의미하다"
    else:
        assert thr >= 0.15, f"2단계가 꺼졌는데 1단계 임계가 낮다({thr}) — 헛깨움이 폭발한다"


def test_two_pass_rejects_prompt_hallucination():
    """🔴 유튜브를 틀어 놓기만 해도 깨어나던 사고를 고정한다(2026-08-12).

    initial_prompt 는 whisper 에게 '재하봇이 있다'고 알려 주는 장치라 없는 오디오에도
    만들어 낸다. 힌트 없이 들으면 실제 내용이 나온다 — 그 차이로 가른다.
    실측: 유튜브 후보 15곳 중 힌트만 쓰면 47~100% 통과, 두 번 대조하면 **0건**.
    """
    from app.wake import make_wake_verifier

    class Stt:
        """유튜브 소리를 흉내낸다 — 힌트를 주면 호출어를 만들어 내는 그 상황."""

        def transcribe(self, audio, initial_prompt=None):
            return ("재하봇, 재하봇" if initial_prompt else "우리의 러시아 형님이오"), 0.1

    v = {"enabled": True, "initial_prompt": "재하봇아, 재하봇이, 재하봇 불러.",
         "max_ratio": 0.45, "plain_max_ratio": 0.65}
    assert make_wake_verifier(v, Stt(), "재하봇")(None) is False


def test_two_pass_still_accepts_a_real_call():
    """진짜 호출은 힌트 없이도 호출어에 가깝게 읽힌다 — 통과해야 한다."""
    from app.wake import make_wake_verifier

    class Stt:
        def transcribe(self, audio, initial_prompt=None):
            # 힌트 없으면 OOV 라 살짝 뭉개지고, 힌트를 주면 정확해진다
            return ("재하봇" if initial_prompt else "재하봇."), 0.1

    v = {"enabled": True, "initial_prompt": "힌트", "max_ratio": 0.45,
         "plain_max_ratio": 0.65}
    assert make_wake_verifier(v, Stt(), "재하봇")(None) is True


def test_two_pass_skips_second_call_when_plain_pass_fails():
    """가짜는 첫 호출에서 끝나야 한다 — whisper 를 두 번 쓰면 비용이 두 배가 된다."""
    from app.wake import make_wake_verifier

    calls = []

    class Stt:
        def transcribe(self, audio, initial_prompt=None):
            calls.append(initial_prompt)
            return "전혀 다른 말이라 자모거리가 멀다", 0.1

    v = {"enabled": True, "initial_prompt": "힌트", "max_ratio": 0.45,
         "plain_max_ratio": 0.65}
    make_wake_verifier(v, Stt(), "재하봇")(None)
    assert calls == [None], f"힌트 없는 판정에서 끝냈어야 한다: {calls}"


def test_conversation_stt_has_no_wake_prompt():
    """검증용 힌트가 대화 STT 로 새면 아이 말이 전부 '재하봇'으로 편향된다."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    assert not (cfg.get("stt") or {}).get("initial_prompt")
