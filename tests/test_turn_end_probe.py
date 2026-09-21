from tools.turn_end_probe import VARIANTS, analyze, summarize


def test_말끝_뒤에_온_첫_판정이_빨라짐이다():
    r = analyze([11.3], t_end=10.0)
    assert abs(r["latency_s"] - 1.3) < 1e-9 and r["premature"] is False and r["splits"] == 0


def test_말끝보다_먼저_온_판정은_잘림이다():
    r = analyze([9.5, 11.2], t_end=10.0)
    assert r["premature"] is True and r["splits"] == 1 and abs(r["latency_s"] - 1.2) < 1e-9


def test_0_1초_안쪽은_잘림으로_안_센다():
    assert analyze([9.95], t_end=10.0)["premature"] is False


def test_판정이_없으면_빈칸():
    r = analyze([], t_end=10.0)
    assert r["latency_s"] is None and r["premature"] is False


def test_방식은_여섯이고_답을_안_만든다():
    assert set(VARIANTS) == {"server1200", "server900", "server600",
                             "semantic_low", "semantic_medium", "semantic_high"}
    assert all(v["create_response"] is False for v in VARIANTS.values())


def test_요약은_방식마다_한_줄():
    rows = [{"variant": "server1200", "latency_s": 1.3, "premature": False},
            {"variant": "server1200", "latency_s": 1.1, "premature": True},
            {"variant": "server900", "latency_s": None, "premature": False}]
    out = "\n".join(summarize(rows))
    assert "server1200" in out and "50%" in out and "server900" in out
