"""배치 추론(BatchedInferencePipeline) 스위치 — 모델·GPU 불필요.

🔴 왜 스위치인가 (2026-08-27 젯슨 실측, 실음성 85개):
     현재   중앙 1.155s  p90 1.552s   선행생각 예산(1.04s) 통과 20/85
     배치8  중앙 1.012s  p90 1.048s   통과 67/85            -12%
   진짜 이득은 중앙이 아니라 **p90 -32%** 다. 편차가 무너져서, 선행 생각을 처음으로
   예산 안에 넣는다(beam 축소는 08-05 에 죽었고 남은 후보가 이거뿐이었다).

⚠️ 그런데 **전사가 달라진다.** 같은 85개에서 36개(42%)가 달랐고, 배치가 말끝을
   흘린 사례가 있었다("… 아니야 아니야" -> "…"). 그래서 기본값은 **끔**이고,
   정답 있는 실아동 음성으로 CER 을 재서 이긴 뒤에만 켠다.
"""
import sys
import types

import numpy as np
import pytest

from app.stt_module import STTModule


@pytest.fixture
def spy(monkeypatch):
    """WhisperModel/BatchedInferencePipeline 을 가로채 무엇이 어떻게 불렸는지 본다."""
    seen = {"wrapped": None, "kw": None}

    class _Seg:
        text = "안녕"

    class _Model:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, audio, **kw):
            seen["kw"] = kw
            return [_Seg()], None

    class _Batched:
        def __init__(self, model=None):
            seen["wrapped"] = model

        def transcribe(self, audio, **kw):
            seen["kw"] = kw
            return [_Seg()], None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(
        WhisperModel=_Model, BatchedInferencePipeline=_Batched))
    return seen


def _run(stt):
    stt.transcribe(np.full(16000, 0.1, dtype=np.float32))


def test_the_switch_is_off_by_default():
    """전사가 42% 달라지는 물건이다. 재서 이기기 전엔 켜지지 않는다."""
    assert STTModule().batched is False


def test_off_means_the_plain_model_and_no_batch_size(spy):
    stt = STTModule(batched=False)
    _run(stt)

    assert spy["wrapped"] is None, "안 켰는데 배치로 감쌌다"
    assert "batch_size" not in spy["kw"], "순차 모델은 batch_size 를 못 받는다 — 터진다"


def test_on_wraps_the_model_and_passes_the_batch_size(spy):
    stt = STTModule(batched=True, batch_size=8)
    _run(stt)

    assert spy["wrapped"] is not None, "배치로 안 감쌌다"
    assert spy["kw"]["batch_size"] == 8


def test_the_decoding_options_stay_identical(spy):
    """배치를 켠다고 beam·temperature 가 달라지면 A/B 가 아니라 딴 실험이 된다."""
    STTModule(batched=False).transcribe(np.full(16000, 0.1, dtype=np.float32))
    plain = dict(spy["kw"])
    STTModule(batched=True, batch_size=8).transcribe(np.full(16000, 0.1, dtype=np.float32))
    batched = dict(spy["kw"])
    batched.pop("batch_size")

    assert plain == batched, f"디코딩 옵션이 달라졌다: {plain} vs {batched}"
