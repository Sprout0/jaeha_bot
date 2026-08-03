"""대화 → 대기 복귀 뒷정리(_to_standby) 검증. 모델·마이크 불필요.

이 4줄이 조용히 안 돌면 증상이 엉뚱하게 나타난다:
  - detector.reset() 누락 → 대화 중 쌓인 임베딩이 남아 대기 첫 순간 엉뚱한 점수
  - source.drain() 누락 → 방금 한 작별인사(SLEEP_MSG)를 호출어로 오인
둘 다 예외가 안 나므로 테스트로만 잡힌다.
"""
from app.main import _to_standby


class FakeDetector:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeSource:
    def __init__(self):
        self.drain_calls = 0

    def drain(self):
        self.drain_calls += 1
        return 0


def test_standby_resets_detector_and_drains_source():
    d, s = FakeDetector(), FakeSource()
    _to_standby(d, s)
    assert d.reset_calls == 1, "감지기 링버퍼를 비워야 함"
    assert s.drain_calls == 1, "작별인사가 남은 입력 버퍼를 비워야 함"


def test_standby_without_source_still_resets_detector():
    # STT 폴백 경로 — 공유 스트림이 없다(main 에서 close 후 None).
    d = FakeDetector()
    _to_standby(d, None)
    assert d.reset_calls == 1


def test_standby_without_detector_is_noop():
    # wake.enabled=false — 감지기 자체가 없다.
    s = FakeSource()
    _to_standby(None, s)
    assert s.drain_calls == 1


def test_standby_with_nothing_does_not_raise():
    _to_standby(None, None)
