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


def test_louder_score_rearms_even_while_noise_stays_above_threshold():
    """🔴 2026-08-12 실기 사고 — 유튜브를 틀면 첫 기각 이후 영영 귀를 닫았다.

    로그: 14:11:14 후보 기각(점수 0.060) → 14:11:29 최고 점수 **0.226 인데 검증 없음**.
    소음이 계속돼 점수가 임계(0.05) 아래로 안 떨어지면 재무장 조건이 성립하지 않아,
    그 뒤에 온 진짜 호출을 통째로 흘려보냈다.
    → 직전 검증보다 뚜렷이 높은 점수는 '새 사건'으로 보고 재무장해야 한다.
    """
    calls = []
    # 소음이 임계 위에 머무는 상태(0.06)에서 진짜 호출(0.23)이 온다
    d, _ = _det([0.06] + [0.06] * 5 + [0.23],
                verifier=lambda a: calls.append(1) or False,
                verify_cooldown_s=0.0)
    d.wait_for_wake(max_frames=10)
    assert len(calls) == 2, (
        f"소음이 안 걷혀도 큰 점수는 검증해야 한다(호출 {len(calls)}회)")


def test_small_fluctuation_above_threshold_does_not_rearm():
    """잔물결까지 새 사건으로 보면 whisper 폭주로 되돌아간다. 뚜렷한 상승만 인정한다."""
    calls = []
    d, _ = _det([0.06] + [0.07, 0.08, 0.09] * 3,
                verifier=lambda a: calls.append(1) or False,
                verify_cooldown_s=0.0)
    d.wait_for_wake(max_frames=12)
    assert len(calls) == 1, f"잔물결에 재무장했다(호출 {len(calls)}회)"


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


# ── 우회컷 (2026-08-26) ──────────────────────────────────────────────────
# 🔴 왜: 2단계가 진짜 호출을 죽이고 있었다. 실음성 20건에서 1단계 19/20 -> 캐스케이드
#    11/20. 죽은 것 중 1단계 점수 **0.873** 짜리가 있다(whisper 가 '하이치 루' 로 읽음).
#    실제 거실 소음 3분의 최고 점수는 0.173 이라, 그 위는 소음이 만들 수 있는 값이 아니다.

def test_high_score_skips_the_verifier_entirely():
    """모델이 확신하면 whisper 를 안 부른다 — 0.873 이 '하이치 루' 로 기각된 실례가 있다."""
    calls = []

    def verifier(audio):
        calls.append(1)
        return False                      # 불렸다면 기각시켜 실패를 드러낸다

    d, _ = _det([0.0, 0.9], verifier=verifier, verify_bypass=0.20)
    r = d.wait_for_wake(max_frames=5)
    assert r is not None, "우회컷 위인데 검증에 걸렸다"
    assert calls == [], "우회컷 위에서 whisper 를 불렀다"


def test_low_score_still_goes_through_the_verifier():
    """우회는 확신할 때만이다. 낮은 점수까지 통과시키면 헛깨움이 폭발한다."""
    calls = []

    def verifier(audio):
        calls.append(1)
        return False

    d, _ = _det([0.0, 0.10], verifier=verifier, verify_bypass=0.20)
    assert d.wait_for_wake(max_frames=5) is None
    assert calls == [1], "우회컷 아래인데 검증을 건너뛰었다"


def test_bypass_is_off_by_default_so_rollback_is_a_config_line():
    """기본값은 꺼짐이어야 한다 — 설정을 지우면 옛 동작으로 돌아가야 한다."""
    calls = []
    d, _ = _det([0.0, 0.99], verifier=lambda a: (calls.append(1), True)[1])
    d.wait_for_wake(max_frames=5)
    assert calls == [1], "설정 없이도 우회가 켜져 있다 — 점수는 1.0 을 못 넘어야 한다"


def test_bypass_ignores_cooldown_and_hysteresis():
    """🔴 우회는 쿨다운·히스테리시스 **위**에 있어야 한다.

    아래에 두면 '방금 소음을 기각했다'는 이유로 바로 뒤따르는 진짜 호출이 막힌다.
    실제로 그 순서 때문에 점수 0.226 짜리 진짜 호출을 통째로 흘려보낸 로그가 있다
    (2026-08-12 14:11). 우회는 그 함정을 지나가야 한다.
    """
    d, _ = _det([0.0, 0.10, 0.9], verifier=lambda a: False,
                verify_bypass=0.20, verify_cooldown_s=999.0)
    r = d.wait_for_wake(max_frames=6)
    assert r is not None and r.score == 0.9, "쿨다운이 우회를 막았다"


def test_shipped_bypass_sits_above_the_measured_room_noise():
    """🔴 설정값이 실측 방 소음(3분 최고 0.173) 위에 있어야 한다.

    우회컷을 넘는 소음은 whisper 를 건너뛰므로 **곧바로 헛깨움**이 된다.
    실측 사건율(환산 회·시간): 0.20 -> 0 / 0.15 -> 20 / 0.10 -> 40 / 0.05 -> 160.
    ⚠️ 0.173 은 3분 표본의 최고치다. 더 길게 재서 이 값을 확정할 것.
    """
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    v = (cfg["wake"]["onnx"].get("verify") or {})
    byp = float(v.get("bypass_score", 1.01))
    if byp >= 1.0:
        return                                    # 우회 꺼짐 — 검사할 게 없다
    assert byp > 0.173, f"우회컷 {byp} 이 실측 방 소음 최고치 0.173 아래다"
    assert byp > float(cfg["wake"]["onnx"]["threshold"]), \
        "우회컷이 1단계 임계보다 낮으면 2단계가 통째로 무력화된다"


# ── 1패스 (2026-08-26) ───────────────────────────────────────────────────
# 🔴 2패스는 '재하봇'이 OOV 라 힌트 없이는 41% 밖에 못 읽어서 넣은 것이다.
#    '하이티드'로 바꾼 뒤 실기 로그 전수에서 **2패스가 판정을 뒤집은 건 0건**이고,
#    성공한 호출마다 whisper 를 한 번 더(1.22초) 돌릴 뿐이었다.

class _CountingStt:
    """힌트 유무에 따라 다른 답을 주고, 몇 번 불렸는지 센다."""

    def __init__(self, plain, hinted):
        self.plain, self.hinted = plain, hinted
        self.calls = []

    def transcribe(self, audio, initial_prompt=None):
        self.calls.append(initial_prompt)
        return (self.hinted if initial_prompt else self.plain), 0.0


def _verifier(stt, **over):
    from app.wake import make_wake_verifier
    cfg = {"enabled": True, "initial_prompt": "하이 티드.",
           "max_ratio": 0.45, "plain_max_ratio": 0.35}
    cfg.update(over)
    return make_wake_verifier(cfg, stt, "하이티드")


def test_single_pass_calls_whisper_once():
    """성공한 호출마다 1.22초를 아낀다 — whisper 를 두 번 부르지 않아야 한다."""
    stt = _CountingStt("하이치드", "하이티드")
    assert _verifier(stt, two_pass=False)(np.zeros(32000, dtype=np.float32)) is True
    assert stt.calls == [None], f"whisper 를 {len(stt.calls)}번 불렀다"


def test_single_pass_still_rejects_on_the_plain_gate():
    """🔴 정밀도는 전부 1패스가 만든다. 여기가 뚫리면 유튜브에 깨어난다."""
    stt = _CountingStt("MBC 뉴스 이재경입니다", "하이티드")
    assert _verifier(stt, two_pass=False)(np.zeros(32000, dtype=np.float32)) is False
    assert stt.calls == [None], "기각인데 힌트 패스를 돌았다"


def test_single_pass_does_not_use_the_hint_at_all():
    """힌트는 환각을 만든다(유튜브 후보의 47~100%가 호출어로 전사됐다).
    1패스에서는 힌트가 **아예 쓰이면 안 된다** — 쓰이면 그 위험이 돌아온다."""
    stt = _CountingStt("하이티드", "하이티드")
    _verifier(stt, two_pass=False)(np.zeros(32000, dtype=np.float32))
    assert None in stt.calls and not any(c for c in stt.calls), stt.calls


def test_two_pass_is_the_default_so_old_configs_keep_working():
    """설정에 two_pass 가 없으면 옛 동작이어야 한다(되돌릴 길)."""
    stt = _CountingStt("하이치드", "하이티드")
    assert _verifier(stt)(np.zeros(32000, dtype=np.float32)) is True
    assert stt.calls == [None, "하이 티드."], stt.calls


def test_shipped_config_turned_two_pass_off():
    """설정이 실제로 1패스인지 — 이게 켜져 있으면 깨움이 1.2초 느리다."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    v = (cfg["wake"]["onnx"].get("verify") or {})
    assert v.get("two_pass") is False, "2패스가 켜져 있다 — 근거는 wake.py 주석 참고"
    assert v.get("initial_prompt"), "되돌릴 때 필요하니 힌트는 남겨 둔다"


# ── 2패스 컷을 실기 전사로 고정 (2026-08-27) ──────────────────────────────
# 🔴 왜: 컷 0.35 로 젯슨 실기 80초에서 **통과 0건**이었다. 재하님이 계속 부르는데
#    한 번도 안 깨어났다. whisper 는 알아듣고 있었고(전부 '하이치X') 거리가 0.38~0.44 라
#    **0.03 차이로** 전부 떨어진 것이다. 합성음(중앙 0.000)만 보고 정한 값이 실음성에서
#    뒤집혔다 — 이 프로젝트가 같은 실수를 세 번째 하는 자리다.
#    아래 세 테스트는 **실기 로그의 진짜 전사**로 컷을 양쪽에서 조인다.

def _shipped_plain_cut() -> float:
    from pathlib import Path

    import yaml
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    return float(cfg["wake"]["onnx"]["verify"]["plain_max_ratio"])


# 2026-08-27 11:12~11:13 젯슨 실기 로그 전수. 사람이 부른 것 / 아닌 것.
REAL_CALLS = ["하이치카", "하이치테", "하이치티", "하이킥", "하이픽", "하이치켓", "하이키밤"]
ROOM_NOISE = ["아이차티", "아이쿠", "아이스크림", "맛있게", "안녕히계세요", ""]


def test_real_call_transcripts_pass_the_shipped_cut():
    """🔴 실기에서 실제로 나온 전사가 안 통과하면 봇은 영영 안 깨어난다."""
    from app.wake import best_wake_ratio
    cut = _shipped_plain_cut()
    bad = [(t, best_wake_ratio(t, "하이티드")) for t in REAL_CALLS
           if best_wake_ratio(t, "하이티드") > cut]
    assert not bad, f"컷 {cut} 이 진짜 호출을 막는다: {bad}"


def test_room_noise_transcripts_stay_rejected():
    """컷을 여는 건 좋지만 소음까지 받으면 2단계가 있을 이유가 없다."""
    from app.wake import best_wake_ratio
    cut = _shipped_plain_cut()
    leaked = [(t or "(빈)", best_wake_ratio(t, "하이티드")) for t in ROOM_NOISE
              if best_wake_ratio(t, "하이티드") <= cut]
    assert not leaked, f"컷 {cut} 으로 소음이 샜다: {leaked}"


def test_cut_never_admits_a_bare_hai():
    """🔴 컷을 0.50 으로 열면 `하이`·`티드` 한 마디(둘 다 0.50)가 봇을 깨운다.

    사람이 하루에 수십 번 하는 말이다. 0.44(관측된 진짜 호출 최대)와 0.50 사이가
    **반드시 컷이 있어야 할 자리**다. 넉넉해 보인다고 올리지 말 것.
    """
    from app.wake import best_wake_ratio
    cut = _shipped_plain_cut()
    for word in ("하이", "티드", "아이디", "하이볼"):
        r = best_wake_ratio(word, "하이티드")
        assert r > cut, f"'{word}' 자모거리 {r:.2f} <= 컷 {cut} — 이 한 마디로 깨어난다"


# ── 쿨다운은 검증이 '끝난' 시각부터 (2026-08-27) ────────────────────────
# 🔴 왜: `_last_verify` 를 whisper 를 부르기 **전에** 찍어서, whisper 가 도는
#    1.2~2.0초가 쿨다운 1.0초를 통째로 먹었다. 실기 로그:
#      11:13:26.841 검증 끝 → 11:13:26.939 다음 검증 (0.10초 뒤)
#    같은 발화 하나를 3~5번 검증하고 그동안 읽기 루프가 멈춰 버퍼가 넘쳤다(29회).

def test_cooldown_is_measured_from_when_verification_finished():
    """🔴 느린 검증기가 자기 쿨다운을 다 먹어버리면 안 된다."""
    import time as _t
    calls = []

    def slow(_audio):
        calls.append(1)
        _t.sleep(0.25)          # whisper 가 도는 동안
        return False

    # 쿨다운 0.2s < 검증 0.25s. 시작 기준이면 다음 후보가 곧바로 통과해 버린다.
    d, _ = _det([0.9, 0.0, 0.9, 0.0, 0.9], verifier=slow, verify_cooldown_s=0.2)
    d.wait_for_wake(max_frames=8)
    assert len(calls) == 1, (
        f"검증기를 {len(calls)}번 불렀다 — 쿨다운을 검증 시간이 먹었다")


def test_cooldown_still_expires_so_a_later_call_is_heard():
    """폭주를 막자고 영영 안 듣게 되면 그건 더 나쁜 고장이다."""
    import time as _t
    calls = []

    def slow(_audio):
        calls.append(1)
        _t.sleep(0.05)
        return False

    d, _ = _det([0.9, 0.0, 0.9, 0.0, 0.9], verifier=slow, verify_cooldown_s=0.0)
    d.wait_for_wake(max_frames=8)
    assert len(calls) >= 2, f"쿨다운 0 인데도 한 번만 불렀다({len(calls)}회)"


def test_cooldown_is_refreshed_even_when_the_verifier_raises():
    """검증기가 고장난 동안이 폭주하기 가장 쉬운 때다 — 거기서도 쿨다운이 살아야 한다."""
    calls = []

    def boom(_audio):
        calls.append(1)
        raise RuntimeError("whisper 죽음")

    d, _ = _det([0.9, 0.0, 0.9, 0.0, 0.9], verifier=boom, verify_cooldown_s=99.0)
    d.wait_for_wake(max_frames=8)
    assert len(calls) == 1, f"예외 뒤 쿨다운이 안 걸렸다({len(calls)}회)"


# ── 검증창을 뜨기 전에 조금 더 듣는다 (2026-08-27) ──────────────────────
# 🔴 왜: 1단계가 '하이' 까지만 듣고도 임계(0.05)를 넘긴다. 그 순간 창을 뜨면 끝이
#    '티드' 앞이라 whisper 가 '하이' 라고 받아쓰고 자모거리 0.50 으로 기각된다.
#    실기 11:51:53 — **점수 0.272 로 높은데 전사가 '하이'**.

def test_settle_reads_exactly_the_configured_frames():
    """호출어 끝이 창 안에 들어오도록 정확히 그만큼 더 읽어야 한다."""
    at_verify = []

    d, src = _det([0.9] * 12, verify_settle_s=0.24)   # 0.24s = 80ms 프레임 3개
    d.verifier = lambda _a: at_verify.append(src._i) or False
    d.wait_for_wake(max_frames=12)

    assert at_verify, "검증기가 안 불렸다"
    d2, src2 = _det([0.9] * 12, verify_settle_s=0.0)
    plain = []
    d2.verifier = lambda _a: plain.append(src2._i) or False
    d2.wait_for_wake(max_frames=12)
    assert at_verify[0] - plain[0] == 3, (
        f"프레임 3개를 더 읽어야 한다(실제 {at_verify[0] - plain[0]}개)")


def test_settle_zero_keeps_the_old_behaviour():
    """되돌리기가 설정 한 줄이어야 한다 — 0 이면 프레임을 더 읽지 않는다."""
    reads = []

    def spy(_audio):
        reads.append(src._i)
        return False

    d, src = _det([0.9] * 12, verify_settle_s=0.0)
    d.verifier = spy
    d.wait_for_wake(max_frames=12)
    at_zero = reads[0]

    reads2 = []

    def spy2(_audio):
        reads2.append(src2._i)
        return False

    d2, src2 = _det([0.9] * 12, verify_settle_s=0.24)
    d2.verifier = spy2
    d2.wait_for_wake(max_frames=12)
    assert reads2[0] > at_zero, (
        f"settle 을 줬는데 더 안 읽었다({reads2[0]} vs {at_zero})")


def test_settle_survives_a_source_that_runs_out():
    """프레임이 떨어져도 예외로 깨움 경로를 무너뜨리면 안 된다."""
    d, _ = _det([0.9], verifier=lambda a: True, verify_settle_s=2.0)
    d.wait_for_wake(max_frames=30)          # 예외 없이 끝나면 통과


def test_shipped_settle_is_small_enough_to_not_hurt_wake_latency():
    """깨움이 이미 1.3초다 — settle 이 크면 그만큼 아이가 더 기다린다."""
    from pathlib import Path

    import yaml
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    s = float((cfg["wake"]["onnx"]["verify"] or {}).get("settle_s", 0.0))
    assert 0.0 <= s <= 0.5, f"settle_s {s} 는 깨움 지연에 그대로 더해진다"


# ── 기각된 후보를 소리로 남긴다 (2026-08-27) ────────────────────────────
# 🔴 왜: 실기 11:51:43 — 점수 0.459 인데 전사가 '안녕히계세요'(자모거리 0.79).
#    이게 놓친 호출인지 정상 기각인지 **로그로는 구분할 방법이 없다.** 추측으로
#    임계를 정했다가 이미 두 번 틀렸다(우회컷 0.2, 컷 0.35).

def _saving_verifier(tmp_path, text="안녕히계세요", **kw):
    from app.wake import make_wake_verifier

    class _Stt:
        def transcribe(self, audio, initial_prompt=None):
            return text, 0.0

    cfg = {"enabled": True, "plain_max_ratio": 0.45, "two_pass": False,
           "save_rejects_dir": str(tmp_path), **kw}
    return make_wake_verifier(cfg, _Stt(), "하이티드")


def test_rejected_audio_is_saved_for_listening(tmp_path):
    v = _saving_verifier(tmp_path, "안녕히계세요")
    assert v(np.zeros(32000, dtype=np.float32)) is False
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1, "기각인데 소리를 안 남겼다"
    assert "0.79" in wavs[0].name, f"파일명에 자모거리가 없다: {wavs[0].name}"
    assert "안녕히계세요" in wavs[0].name, f"파일명에 전사가 없다: {wavs[0].name}"


def test_passing_audio_is_not_saved(tmp_path):
    """통과까지 남기면 디스크가 금방 찬다 — 진단 대상은 기각뿐이다."""
    v = _saving_verifier(tmp_path, "하이티드")
    assert v(np.zeros(32000, dtype=np.float32)) is True
    assert list(tmp_path.glob("*.wav")) == []


def test_saving_is_off_by_default(tmp_path):
    v = _saving_verifier(tmp_path, "안녕히계세요", save_rejects_dir=None)
    assert v(np.zeros(32000, dtype=np.float32)) is False
    assert list(tmp_path.glob("*.wav")) == []


def test_a_save_failure_never_breaks_the_wake_path(tmp_path):
    """🔴 진단 기능이 봇을 못 깨우게 만들면 본말전도다."""
    blocked = tmp_path / "파일이라_폴더가_안된다"
    blocked.write_text("x", encoding="utf-8")
    v = _saving_verifier(blocked)
    assert v(np.zeros(32000, dtype=np.float32)) is False   # 예외 없이 정상 기각


def test_empty_transcript_still_gets_a_filename(tmp_path):
    """빈 전사가 가장 궁금한 경우다 — 이름이 비면 파일이 안 만들어진다."""
    v = _saving_verifier(tmp_path, "")
    assert v(np.zeros(32000, dtype=np.float32)) is False
    assert len(list(tmp_path.glob("*무음*.wav"))) == 1
