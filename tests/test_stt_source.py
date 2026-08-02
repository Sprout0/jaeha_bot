"""record_until_silence 의 공유 스트림·prefix 경로. 마이크·모델 불필요."""
import numpy as np

from app.audio_source import FRAME
from app.stt_module import STTModule


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


def test_prefix_is_included_in_output():
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    prefix = np.full(FRAME * 2, 0.4, dtype=np.float32)
    src = ScriptedSource(_loud(3) + _quiet(30))
    out = stt.record_until_silence(source=src, prefix=prefix)
    # quiet_needed = int(0.3*16000/1280) = 3. 3개 loud(변화 없음) + 무음판정까지의
    # 3개 quiet 프레임 = 6프레임이 prefix 뒤에 더 붙는다. prefix 를 통째로 버려도
    # 우연히 참이 되지 않도록 정확한 값으로 고정한다(느슨한 >= 는 대기루프가
    # prefix 없이도 같은 크기를 모아 버려 통과해버림).
    assert out.size == prefix.size + 6 * FRAME


def test_prefix_path_skips_start_wait():
    """prefix 가 있으면 '말 시작 대기' 없이 곧바로 무음 판정으로 간다."""
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    prefix = np.full(FRAME, 0.4, dtype=np.float32)
    # 처음부터 조용해도 prefix 덕에 녹음이 성립하고 곧 종료된다.
    src = ScriptedSource(_quiet(30))
    out = stt.record_until_silence(source=src, prefix=prefix)
    assert out.size > 0


def test_source_without_prefix_waits_for_speech():
    stt = STTModule(silence_duration=0.3, max_duration=5.0, start_timeout=1.0)
    src = ScriptedSource(_quiet(5) + _loud(3) + _quiet(30))
    out = stt.record_until_silence(source=src)
    assert out.size > 0, "말이 시작되면 녹음돼야 함"


def test_source_returns_empty_when_frames_exhausted():
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    src = ScriptedSource(_quiet(3))
    out = stt.record_until_silence(source=src)
    assert out.size == 0


def test_counters_use_80ms_source_frames_not_30ms():
    """카운터 단위 고정 회귀: quiet_needed 가 source 의 실제 프레임(80ms=1280)
    이 아니라 own-stream 시절 30ms 단위로 새면 이 값이 크게 어긋나 실패한다.

    silence_duration=0.8 -> quiet_needed = int(0.8*16000/1280) = 10 (정확히).
    30ms 단위로 잘못 새면 int(0.8/0.03) = 26 이 되어 전혀 다른 프레임 수가 나온다.
    """
    stt = STTModule(silence_duration=0.8, max_duration=5.0)
    src = ScriptedSource(_loud(1) + _quiet(20))
    out = stt.record_until_silence(source=src)
    # 시작을 알린 loud 1프레임 + quiet_needed(10)개의 조용한 프레임 = 11프레임.
    assert out.size == 11 * FRAME


def test_forgiving_grace_is_one_frame_not_three():
    """Important-1 회귀(변이 검증용): quiet 감쇠가 own-stream 의 -3(30ms 단위,
    =90ms) 이 아니라 -1(80ms 프레임 1개) 이어야 한다. 2 quiet + 1 loud 블립을
    반복하는 '말끝을 흘리는 아이' 패턴에서:
      - 감쇠 -1  : quiet 카운터가 사이클마다 순증가해 15프레임째 조기 종료.
      - 감쇠 -3(버그): 사이클마다 quiet 가 0 으로 씻겨 내려가 절대 안 끊기고
                      max_duration 캡(25프레임=32000샘플)까지 끌려간다.
    이 테스트는 감쇠를 -3 으로 되돌리면(수동 변이) out.size 가 32000 이 되어
    실패한다 — task-6-report.md 의 변이 검증 기록 참고.
    """
    stt = STTModule(silence_duration=0.5, max_duration=2.0)
    frames = _loud(1)
    for _ in range(10):
        frames += _quiet(2) + _loud(1)
    src = ScriptedSource(frames)
    out = stt.record_until_silence(source=src)
    assert out.size == 15 * FRAME


def test_no_prefix_path_keeps_pre_roll_before_speech_onset():
    """Important-2 회귀: prefix 없이 시작을 기다릴 때도 own-stream 처럼 최근
    pre_roll 프레임을 링버퍼에 담아 두다가, 말이 시작되면 그 프레임들부터
    결과에 포함해야 한다(첫 음절 잘림 방지). pre_roll=0.3s -> 1280 프레임
    기준 3프레임.
    """
    stt = STTModule(silence_duration=0.3, max_duration=5.0, start_timeout=1.0)
    src = ScriptedSource(_quiet(5) + _loud(1) + _quiet(30))
    out = stt.record_until_silence(source=src)
    # pre_roll 3프레임 + 시작을 알린 loud 1프레임 + 무음판정까지 quiet_needed(3)
    # 개 프레임 = 7프레임. pre_roll 이 빠지면 4프레임(=5120)이 되어 이 값보다 작다.
    assert out.size == 7 * FRAME


def test_signature_still_accepts_no_arguments():
    """기존 호출부 호환 — 인자 없이 부를 수 있어야 한다(실행은 안 함)."""
    import inspect
    sig = inspect.signature(STTModule.record_until_silence)
    assert sig.parameters["source"].default is None
    assert sig.parameters["prefix"].default is None
