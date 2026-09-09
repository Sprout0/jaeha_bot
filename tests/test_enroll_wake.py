"""tools/enroll_wake.py 의 호출 채점(--score-calls) 시험.

이 단계가 **컷의 상한**을 정한다. 여기 숫자가 틀리면 임계값이 통째로 틀리고,
호출어가 조용히 죽는다(이 프로젝트는 컷을 두 번 잘못 잡았다).

감지기는 가짜를 쓴다 — ONNX 모델 파일은 저장소에 없고, 여기서 재려는 것은
모델 성능이 아니라 **채점 방식**이다.
"""
import numpy as np
import pytest

from app.wake_embed import TEMPLATE_FRAMES, make_template
from tools.enroll_wake import score_calls

DIM = 96


class _Det:
    """1단계 점수와 임베딩 열을 미리 정해 주는 가짜 감지기."""

    def __init__(self, score: float, embs) -> None:
        self._score = float(score)
        self._embs = np.asarray(embs, dtype=np.float32)

    def reset(self) -> None:
        pass

    def push(self, chunk):
        return self._score

    def embed_sequence(self, audio):
        return self._embs


def _embs(seed: int, n: int = TEMPLATE_FRAMES + 4):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def _wav(d, name: str, seconds: float = 1.5):
    import soundfile as sf
    p = d / name
    sf.write(str(p), np.zeros(int(seconds * 16000), dtype="float32"), 16000)
    return p


def test_본보기와_같은_소리는_유사도가_1에_가깝다(tmp_path):
    e = _embs(1)
    det = _Det(0.9, e)
    _wav(tmp_path, "call.wav")

    out = score_calls(det, [make_template(e)], str(tmp_path), thr=0.05)

    assert out["low"] == pytest.approx(1.0, abs=1e-3)


def test_1단계를_못_넘은_녹음은_2단계_실패로_세지_않는다(tmp_path):
    """1단계에서 안 걸린 호출은 검증기가 **불리지도 않는다**.

    그걸 낮은 유사도로 세면 컷을 실제보다 낮게 잡게 되고, 소음이 뚫린다.
    """
    e = _embs(2)
    det = _Det(0.01, _embs(99))          # 점수 0.01 < 임계 0.05
    _wav(tmp_path, "call.wav")

    out = score_calls(det, [make_template(e)], str(tmp_path), thr=0.05)

    assert out["stage1_miss"] == 1
    assert out["sims"] == []
    assert out["low"] is None


def test_최저값이_컷의_상한이다(tmp_path):
    e = _embs(3)
    det = _Det(0.9, e)
    for i in range(3):
        _wav(tmp_path, f"call_{i}.wav")

    out = score_calls(det, [make_template(e)], str(tmp_path), thr=0.05)

    assert len(out["sims"]) == 3
    assert out["low"] == min(out["sims"])


def test_wav_이_없으면_조용히_넘어가지_않는다(tmp_path):
    with pytest.raises(SystemExit):
        score_calls(_Det(0.9, _embs(4)), [make_template(_embs(4))],
                    str(tmp_path), thr=0.05)
