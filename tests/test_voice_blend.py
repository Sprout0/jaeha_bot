"""목소리 스타일 블렌딩 — 가중치와 '음색/리듬 분리'.

Supertonic 의 Style 은 벡터 둘이다: ttl(음색, 50x256)과 dp(리듬·발화 속도감, 8x16).
기존 `_resolve_style` 은 이 둘을 **묶어서 균등 평균**만 냈다. 그래서 못 하던 게 둘 있다:

  - 비율 조절      "F1 을 70%, F4 를 30%" 같은 중간 지점
  - 음색/리듬 분리  "높고 밝은 음색 + 차분한 리듬"

유아용 목소리를 고를 때 이 둘이 핵심이다 — 밝은 음색은 원하지만 말이 촐랑거리면
2세가 못 알아듣는다. 균등 블렌드만으로는 그 사이를 못 찾는다.

문법 (기존 표기는 그대로 동작해야 한다 — 운영 config 가 `F1+F4` 다):
    F1                  단일 프리셋
    F1+F4               균등 블렌드(기존)
    F1:0.7+F4:0.3       가중 블렌드
    F1|F4               음색 F1, 리듬 F4
    F1:0.7+F2:0.3|F4    가중 음색 + 리듬 F4
"""
import sys
import types

import numpy as np
import pytest

from app.tts_module import TTSModule


class FakeStyle:
    def __init__(self, ttl, dp):
        self.ttl = ttl
        self.dp = dp


@pytest.fixture
def tts(monkeypatch):
    """프리셋마다 '알아볼 수 있는' 값을 준다 — F1 은 1, F2 는 2 …
    그래야 평균이 맞게 났는지 눈으로 검산할 수 있다."""
    core = types.ModuleType("supertonic.core")
    core.Style = FakeStyle
    monkeypatch.setitem(sys.modules, "supertonic.core", core)

    class FakeTTS:
        def get_voice_style(self, name):
            n = float(name[1:])                     # F1 -> 1.0
            return FakeStyle(np.full((1, 2, 2), n, dtype=np.float32),
                             np.full((1, 2, 2), n * 10, dtype=np.float32))

    m = TTSModule()
    m._tts = FakeTTS()
    return m


def test_single_preset_unchanged(tts):
    s = tts._resolve_style("F3")

    assert s.ttl.mean() == pytest.approx(3.0)
    assert s.dp.mean() == pytest.approx(30.0)


def test_equal_blend_unchanged(tts):
    """운영 config 가 이 표기다. 깨지면 목소리가 조용히 바뀐다."""
    s = tts._resolve_style("F1+F4")

    assert s.ttl.mean() == pytest.approx(2.5)      # (1+4)/2
    assert s.dp.mean() == pytest.approx(25.0)


def test_weighted_blend(tts):
    s = tts._resolve_style("F1:0.7+F4:0.3")

    assert s.ttl.mean() == pytest.approx(0.7 * 1 + 0.3 * 4)
    assert s.dp.mean() == pytest.approx(0.7 * 10 + 0.3 * 40)


def test_weights_are_normalised(tts):
    """비율을 3:1 처럼 쓰고 싶을 때 합이 1 이 아니어도 뜻대로 동작해야 한다."""
    s = tts._resolve_style("F1:3+F4:1")

    assert s.ttl.mean() == pytest.approx(0.75 * 1 + 0.25 * 4)


def test_timbre_and_rhythm_split(tts):
    """이게 이번 확장의 핵심 — 음색과 리듬을 따로 고른다."""
    s = tts._resolve_style("F1|F4")

    assert s.ttl.mean() == pytest.approx(1.0), "음색이 F1 이 아니다"
    assert s.dp.mean() == pytest.approx(40.0), "리듬이 F4 가 아니다"


def test_split_with_weighted_timbre(tts):
    s = tts._resolve_style("F1:0.7+F2:0.3|F4")

    assert s.ttl.mean() == pytest.approx(0.7 * 1 + 0.3 * 2)
    assert s.dp.mean() == pytest.approx(40.0)


def test_rhythm_side_can_blend_too(tts):
    s = tts._resolve_style("F1|F2+F4")

    assert s.ttl.mean() == pytest.approx(1.0)
    assert s.dp.mean() == pytest.approx(30.0)      # (20+40)/2


def test_blend_result_is_float32(tts):
    """onnxruntime 입력이라 dtype 이 어긋나면 런타임에서 터진다."""
    s = tts._resolve_style("F1:0.7+F4:0.3")

    assert s.ttl.dtype == np.float32
    assert s.dp.dtype == np.float32
