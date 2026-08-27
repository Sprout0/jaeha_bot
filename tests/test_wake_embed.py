"""임베딩 대조 — whisper 환각으로 죽는 호출을 소리로 건지는 장치.

여기서 지키는 것은 세 가지다.
  1) 대조 자체가 맞게 계산되는가(본보기 고르기·창 미끄러뜨리기·코사인)
  2) **꺼진 게 조용히 켜지거나, 켠 게 조용히 꺼지지 않는가** — 이 프로젝트가
     지연적재에서 정확히 그렇게 당한 적이 있다
  3) 검증 흐름에 OR 로만 붙는가(통과를 막지 않고, 기각만 뒤집는다)
"""
import logging

import numpy as np
import pytest

from app.wake_embed import (TEMPLATE_FRAMES, EmbedRescue, best_similarity,
                            load_rescue, make_template, save_templates)

K = TEMPLATE_FRAMES
DIM = 96


def _seq(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float32)


# ── 대조 계산 ────────────────────────────────────────────────────────────
def test_a_recording_matches_its_own_template_exactly():
    """자기 자신과는 유사도가 1이어야 한다. 아니면 정규화나 창 정렬이 틀린 것이다."""
    embs = _seq(40)
    t = make_template(embs)
    assert t is not None
    assert best_similarity(embs, t[None, :]) == pytest.approx(1.0, abs=1e-4)


def test_unrelated_audio_does_not_match():
    t = make_template(_seq(40, seed=1))
    assert best_similarity(_seq(40, seed=2), t[None, :]) < 0.5


def test_template_is_taken_from_where_stage_one_reacted():
    """🔴 본보기 창은 **1단계 점수가 가장 높은 자리**여야 한다.

    감지기가 실제로 반응하는 지점과 본보기가 어긋나면 대조가 뜻을 잃는다.
    """
    embs = _seq(40, seed=3)
    scores = np.zeros(len(embs), dtype=np.float32)
    peak_end = 25                      # 창의 '끝'이 여기일 때 점수가 최고
    scores[peak_end] = 0.9
    t = make_template(embs, scores)
    want = embs[peak_end - K + 1:peak_end + 1].reshape(-1)
    want = want / np.linalg.norm(want)
    assert np.dot(t, want) == pytest.approx(1.0, abs=1e-4)


def test_short_recording_makes_no_template():
    """창보다 짧은 녹음으로 본보기를 만들면 안 된다(있는 척하면 더 나쁘다)."""
    assert make_template(_seq(K - 1)) is None


def test_similarity_is_zero_when_there_is_nothing_to_compare():
    t = make_template(_seq(40))
    assert best_similarity(_seq(K - 1), t[None, :]) == 0.0
    assert best_similarity(_seq(40), np.zeros((0, K * DIM), dtype=np.float32)) == 0.0


def test_template_of_the_wrong_shape_warns_instead_of_scoring_zero(caplog):
    """🔴 모델을 바꾸면 본보기 규격이 안 맞는다. 조용히 0 을 주면 '못 건졌다'와
    구별이 안 돼 원인을 영영 못 찾는다."""
    bad = np.zeros((1, K * DIM + 7), dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.wake_embed"):
        assert best_similarity(_seq(40), bad) == 0.0
    assert any("규격" in r.message for r in caplog.records), "규격 불일치를 안 알렸다"


def test_rescue_compares_against_the_configured_cut():
    embs = _seq(40, seed=4)
    t = make_template(embs)
    assert EmbedRescue(t, 0.99).passes(embs)[0] is True
    assert EmbedRescue(t, 1.01).passes(embs)[0] is False


# ── 켜짐/꺼짐이 조용히 뒤집히지 않는다 ────────────────────────────────────
def test_missing_template_file_warns_and_disables(tmp_path, caplog):
    """🔴 켰는데 파일이 없으면 **말해야 한다.** 조용히 None 이면 '켰는데 왜 그대로지'
    를 영영 못 알아낸다."""
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.wake_embed"):
        assert load_rescue(tmp_path / "없다.npy", 0.85) is None
    assert any("본보기가 없다" in r.message for r in caplog.records)


def test_empty_template_file_warns_and_disables(tmp_path, caplog):
    p = tmp_path / "빈.npy"
    np.save(str(p), np.zeros((0, K * DIM), dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.wake_embed"):
        assert load_rescue(p, 0.85) is None
    assert any("비어 있다" in r.message for r in caplog.records)


def test_corrupt_template_file_does_not_kill_the_bot(tmp_path, caplog):
    p = tmp_path / "깨짐.npy"
    p.write_bytes("이건 npy 가 아니다".encode("utf-8"))
    with caplog.at_level(logging.WARNING, logger="jaeha_bot.wake_embed"):
        assert load_rescue(p, 0.85) is None


def test_saved_templates_round_trip(tmp_path):
    t = make_template(_seq(40, seed=5))
    p = tmp_path / "sub" / "templates.npy"      # 폴더가 없어도 만들어야 한다
    save_templates(p, np.stack([t]))
    r = load_rescue(p, 0.85)
    assert r is not None and len(r) == 1


def test_no_path_means_off():
    assert load_rescue(None, 0.85) is None
    assert load_rescue("", 0.85) is None


# ── 설정 배선 ────────────────────────────────────────────────────────────
def test_config_default_is_off():
    """🔴 컷이 아직 3분짜리 소음 표본으로만 정해져 있다. 켠 채로 배포되면 안 된다."""
    from app.wake import _make_embed_rescue
    assert _make_embed_rescue({}) is None
    assert _make_embed_rescue({"embed_rescue": {"enabled": False}}) is None


def test_shipped_config_keeps_embed_rescue_off():
    """설정 파일이 실수로 켠 채 커밋되는 걸 막는다."""
    from pathlib import Path

    import yaml
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "model_paths.yaml")
        .read_text(encoding="utf-8"))
    e = cfg["wake"]["onnx"]["verify"]["embed_rescue"]
    assert e["enabled"] is False, (
        "임베딩 대조가 켜진 채다 — 거실 30~60분 녹음으로 컷을 다시 잡기 전엔 끌 것")
