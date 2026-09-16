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
