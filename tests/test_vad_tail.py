"""VAD 꼬리 대기(말끝 → 녹음 종료) 계측. 마이크·모델 불필요.

왜 필요한가: 이 구간은 **아이가 순전히 기다리는 시간**인데 지금까지 `stt_wait_s`
안에서 아이가 말한 시간과 섞여 있어 따로 볼 수 없었다. 그래서 '말끝부터 첫
소리까지'라고 적힌 `resp_compute_s` 가 실제로는 이 구간을 빼고 세고 있었다.

🔴 `silence_duration` 을 그대로 꼬리로 쓰면 안 된다. quiet 카운터가 말소리
   프레임에서 감쇠(`quiet - 1`)하므로 실제 꼬리는 설정값보다 **짧을 수도 있다.**
   가정하지 말고 마지막 말소리 프레임을 기준으로 재야 한다.
"""
import numpy as np

from app.audio_source import FRAME
from app.stt_module import SAMPLE_RATE, STTModule

FRAME_S = FRAME / SAMPLE_RATE  # 0.08s


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


def _loud(n):
    return [np.full(FRAME, 0.3, dtype=np.float32) for _ in range(n)]


def _quiet(n):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n)]


def test_tail_is_measured_from_last_speech():
    """말이 뚝 끊기면 꼬리 = quiet_needed 프레임."""
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    src = ScriptedSource(_loud(3) + _quiet(30))
    stt.record_until_silence(source=src)
    # quiet_needed = int(0.3*16000/1280) = 3
    assert abs(stt.last_vad_tail_s - 3 * FRAME_S) < 1e-6


def test_tail_can_be_shorter_than_silence_duration():
    """중간 블립이 있으면 quiet 이 누적돼 꼬리가 설정값보다 짧아진다.

    L L L Q Q L Q Q  -> 마지막 말소리 뒤 quiet 2개에서 종료(카운터가 1로만 감쇠).
    설정값 0.3s(=3프레임)보다 짧은 0.16s 다. 이걸 1.2초로 가정하면 틀린다.
    """
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    src = ScriptedSource(_loud(3) + _quiet(2) + _loud(1) + _quiet(30))
    stt.record_until_silence(source=src)
    assert abs(stt.last_vad_tail_s - 2 * FRAME_S) < 1e-6, (
        f"블립 뒤 꼬리는 2프레임이어야 한다: {stt.last_vad_tail_s}")


def test_tail_resets_between_turns():
    """이전 턴 값이 남아 다음 턴 지연으로 잘못 집계되면 안 된다."""
    stt = STTModule(silence_duration=0.3, max_duration=5.0, start_timeout=0.5)
    stt.record_until_silence(source=ScriptedSource(_loud(3) + _quiet(30)))
    first = stt.last_vad_tail_s
    assert first > 0
    # 아무 말도 없이 끝난 턴은 꼬리가 없다.
    stt.record_until_silence(source=ScriptedSource(_quiet(3)))
    assert stt.last_vad_tail_s == 0.0, "말이 없던 턴은 꼬리 0 이어야 한다"


def test_max_duration_cap_still_reports_tail():
    """캡에 걸려 끊겨도 꼬리는 마지막 말소리 기준으로 계산된다."""
    stt = STTModule(silence_duration=5.0, max_duration=0.5)
    src = ScriptedSource(_loud(2) + _quiet(30))
    stt.record_until_silence(source=src)
    # max_samples = 0.5*16000 = 8000 -> 프레임 1280 기준 7프레임에서 캡.
    # loud 2개 뒤 quiet 5개가 담기고 종료 -> 꼬리 5프레임.
    assert abs(stt.last_vad_tail_s - 5 * FRAME_S) < 1e-6
