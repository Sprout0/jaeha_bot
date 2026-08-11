"""확장·모방 대리지표. 아이가 생기는 날 바로 재려고 배선만 해 둔다.

⚠️ 둘 다 대리지표다. '아이가 실제로 더 말했는가'는 재지 못한다.
   안전 판정기를 하한선으로 읽기로 한 것과 같은 태도가 필요하다.
"""
import json

from app.metrics import MetricsLogger, _reuse_ratio, _words


def test_words_drops_punctuation():
    assert _words("멍멍! 강아지는 멍멍 하고 울어!") == \
        ["멍멍", "강아지는", "멍멍", "하고", "울어"]


def test_reuse_counts_child_words_echoed_back():
    assert _reuse_ratio("멍멍", "멍멍! 강아지는 멍멍 하고 울어!") == 1.0
    assert _reuse_ratio("멍멍", "그건 아직 못 해!") == 0.0


def test_reuse_survives_korean_particles():
    # 어절 단위 정확 비교면 '멍멍' 과 '멍멍을' 이 다르다고 볼 것이다.
    # 부분일치라 조사가 붙어도 잡힌다.
    assert _reuse_ratio("멍멍", "멍멍을 해볼까?") == 1.0


def test_reuse_is_none_when_child_said_nothing():
    assert _reuse_ratio("", "안녕!") is None


def test_record_turn_logs_expansion_fields(tmp_path):
    m = MetricsLogger(enabled=True, tag="test", log_dir=str(tmp_path))
    m.record_turn(stt_wait_s=1.0, stt_rec_s=0.5, think_s=0.9, think_kind="game",
                  tts_first_s=0.8, tts_play_s=2.0,
                  reply="멍멍! 강아지는 멍멍 하고 울어!", child_text="멍멍")

    rec = json.loads(m.path.read_text(encoding="utf-8").splitlines()[-1])

    assert rec["expansion_delta"] == 4     # 봇 5낱말 - 아이 1낱말
    assert rec["reuse"] == 1.0
