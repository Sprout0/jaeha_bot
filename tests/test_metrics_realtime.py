import json

from app.metrics import MetricsLogger


def test_realtime_턴은_따로_한_줄로_남는다(tmp_path):
    m = MetricsLogger(enabled=True, tag="pc-rt", log_dir=str(tmp_path))
    try:
        m.record_realtime_turn(kind="chat", perceived_s=1.23456, transcribe_s=0.4,
                               respond_first_s=0.5, filler=False, reply="응!",
                               child_text="안녕", cost_usd=0.0015, cached_tokens=10,
                               safety=[], game_missing=[])
    finally:
        m._sampler.stop()
    rec = json.loads(m.path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["event"] == "rt_turn" and rec["metric_ver"] == "rt1" and rec["tag"] == "pc-rt"
    assert rec["perceived_s"] == 1.235 and rec["kind"] == "chat" and rec["turn"] == 1
    assert m.samples == []


def test_막기_붙잡기_다시붙기를_남긴다(tmp_path):
    m = MetricsLogger(enabled=True, tag="t", log_dir=str(tmp_path))
    try:
        m.record_realtime_turn(kind="chat", perceived_s=1.0, transcribe_s=0.5, respond_first_s=0.5,
                               filler=False, reply="r", child_text="c", cost_usd=0.001,
                               cached_tokens=0, safety=[], game_missing=[],
                               held=True, hold_s=0.8, blocked=True, corrected=False, reconnects=1)
    finally:
        m._sampler.stop()
    rec = json.loads(m.path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["held"] is True and rec["hold_s"] == 0.8 and rec["blocked"] is True
    assert rec["corrected"] is False and rec["reconnects"] == 1


def test_놀이_바로잡기를_남긴다(tmp_path):
    m = MetricsLogger(enabled=True, tag="t", log_dir=str(tmp_path))
    try:
        m.record_realtime_turn(kind="game", perceived_s=1.0, transcribe_s=0.5, respond_first_s=0.5,
                               filler=False, reply="r", child_text="c", cost_usd=0.001,
                               cached_tokens=0, safety=[], game_missing=[],
                               game_fixed="wrong_name", game_leaked=True)
    finally:
        m._sampler.stop()
    rec = json.loads(m.path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["game_fixed"] == "wrong_name" and rec["game_leaked"] is True
