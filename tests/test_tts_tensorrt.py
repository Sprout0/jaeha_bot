"""확산·보코더를 TensorRT 로 갈아끼우는 배선. GPU·TRT 없이 검증한다.

🔴 왜 TRT 인가 (2026-08-24 젯슨 실측). CUDA EP 는 **직전 호출과 입력 모양이 다르면**
   확산 1스텝에 ~180ms 를 문다. 두 모양을 각각 6번씩 예열한 뒤에도 그렇다:
     A 181.2ms(직전과 다름) → 50.2 → 27.1 → 27.0   |   B 184.5ms   |   A 180.6ms
   문장마다 길이가 다르니 실기에선 매번 낸다. '모양 가짓수를 줄이는' 버킷팅은 그래서
   무의미하고(연속 호출이 다르면 몇 종이든 문다), 유일한 방법인 패딩은 소리를 바꾼다
   (text_enc 최대차 0.444 / vector_est 0.206 / 최종 파형 상관 0.007~0.40).
   ORT 옵션(cudnn_conv_algo_search·mem_pattern·arena)도 전부 무효였다.
   TRT 는 min/opt/max 프로파일로 엔진을 미리 구워 이 비용을 없앤다: +271.5ms → +10.7ms.

   문장 합성(길이 섞인 실기 조건): CUDA@12 817ms → **TRT@24 453.8ms**.
   확산 스텝을 12→24 로 **두 배 올리고도** -363ms 다.

🔴 이 파일이 지키는 것은 **안전한 폴백**이다. configs/model_paths.yaml 은 젯슨과
   노트북이 함께 쓴다. 노트북엔 TensorRT 가 없으므로, TRT 를 못 붙이는 상황에서
   **말을 못 하게 되면 안 된다.** 못 붙으면 원래 CUDA/CPU 세션 그대로 계속 가야 한다.
⚠️ 다만 조용히 넘어가면 안 된다 — 젯슨에서 TRT 가 조용히 실패하면 '왜 느려졌는지'를
   아무도 못 찾는다. 그래서 실패는 경고로 남긴다(없는 게 정상인 노트북은 제외).
"""
import sys
import types

import numpy as np
import pytest

from app.tts_module import TTSModule


class _Pipeline:
    """supertonic 의 TTS 파이프라인 흉내. model 을 들고 synthesize 를 제공한다.

    TRT 세션이 붙어 있으면 실패하도록 만들어 '프로파일을 벗어난 문장' 을 재현한다.
    """

    def __init__(self, model, fail_on_trt=False):
        self.model = model
        self.fail_on_trt = fail_on_trt
        self.seen = []

    def synthesize(self, text, **kw):
        self.seen.append(self.model.vector_est_ort)
        if self.fail_on_trt and str(self.model.vector_est_ort).startswith("trt"):
            raise RuntimeError(
                "Set dimension [1,144,222] for tensor noisy_latent does not "
                "satisfy any optimization profiles")
        return np.ones(8, dtype=np.float32), 0.5


class _Model:
    """supertonic 의 Supertonic 코어 흉내 — 우리가 바꿀 두 세션만 들고 있다."""

    def __init__(self):
        self.vector_est_ort = "cuda-확산"
        self.vocoder_ort = "cuda-보코더"
        self.dp_ort = "cuda-dp"
        self.text_enc_ort = "cuda-텍스트"


def _tts(*, fail_on_trt=False, **kw) -> TTSModule:
    t = TTSModule(**kw)
    t._tts = _Pipeline(_Model(), fail_on_trt=fail_on_trt)
    return t


@pytest.fixture
def trt_available(monkeypatch):
    """onnxruntime 이 TensorRT 를 제공한다고 알려준다(젯슨 상황)."""
    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: [
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


@pytest.fixture
def trt_missing(monkeypatch):
    """TensorRT 가 없는 기계(노트북)."""
    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


# ── 켜고 끄기 ────────────────────────────────────────────────────────────────

def test_trt_off_leaves_the_sessions_alone(trt_available):
    t = _tts(trt=False)

    t._apply_trt()

    assert t._tts.model.vector_est_ort == "cuda-확산"
    assert t._tts.model.vocoder_ort == "cuda-보코더"


def test_trt_on_replaces_diffusion_and_vocoder(trt_available, monkeypatch):
    t = _tts(trt=True)
    monkeypatch.setattr(t, "_make_trt_session", lambda name: f"trt-{name}")

    t._apply_trt()

    assert t._tts.model.vector_est_ort == "trt-vector_estimator"
    assert t._tts.model.vocoder_ort == "trt-vocoder"


def test_trt_never_touches_dp_or_text_encoder(trt_available, monkeypatch):
    """🔴 이 둘은 TRT 빌드가 실패한다(Pad 출력에 shape 없음 — ORT-TRT 알려진 제약).

    게다가 모양 고정 상태에서 합쳐 20ms 뿐이라 붙일 값어치도 없다.
    """
    t = _tts(trt=True)
    monkeypatch.setattr(t, "_make_trt_session", lambda name: f"trt-{name}")

    t._apply_trt()

    assert t._tts.model.dp_ort == "cuda-dp"
    assert t._tts.model.text_enc_ort == "cuda-텍스트"


# ── 폴백 — 여기가 이 파일의 본론 ─────────────────────────────────────────────

def test_missing_tensorrt_keeps_the_working_sessions(trt_missing, monkeypatch):
    """🔴 노트북에는 TRT 가 없다. 같은 config 를 쓰므로 여기서 죽으면 안 된다."""
    t = _tts(trt=True)
    monkeypatch.setattr(t, "_make_trt_session",
                        lambda name: pytest.fail("TRT 없는데 세션을 만들려 했다"))

    t._apply_trt()

    assert t._tts.model.vector_est_ort == "cuda-확산"
    assert t._tts.model.vocoder_ort == "cuda-보코더"


def test_missing_tensorrt_does_not_warn(trt_missing, caplog):
    """없는 게 정상인 기계다. 경고를 띄우면 진짜 경고가 묻힌다."""
    t = _tts(trt=True)

    with caplog.at_level("WARNING"):
        t._apply_trt()

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_failed_engine_leaves_that_session_untouched(trt_available, monkeypatch):
    """엔진 하나가 안 구워져도 봇은 말을 해야 한다."""
    def boom(name):
        if name == "vocoder":
            raise RuntimeError("엔진 빌드 실패")
        return f"trt-{name}"

    t = _tts(trt=True)
    monkeypatch.setattr(t, "_make_trt_session", boom)

    t._apply_trt()

    assert t._tts.model.vector_est_ort == "trt-vector_estimator", "성공한 쪽은 바뀌어야"
    assert t._tts.model.vocoder_ort == "cuda-보코더", "실패한 쪽은 원래 것을 지켜야"


def test_a_failed_engine_is_logged_loudly(trt_available, monkeypatch, caplog):
    """⚠️ 조용한 실패는 '왜 느려졌는지'를 못 찾게 만든다."""
    t = _tts(trt=True)
    monkeypatch.setattr(t, "_make_trt_session",
                        lambda name: (_ for _ in ()).throw(RuntimeError("빌드 실패")))

    with caplog.at_level("WARNING"):
        t._apply_trt()

    assert any("vector_estimator" in r.message or "vector_estimator" in str(r.args)
               for r in caplog.records if r.levelname == "WARNING")


# ── 프로파일 ─────────────────────────────────────────────────────────────────
# min/opt/max 를 미리 줘야 TRT 가 한 엔진으로 범위를 커버한다. 범위를 벗어나면
# 엔진을 다시 굽느라 오히려 느려지므로 실기 최대치보다 넉넉해야 한다.

def test_vector_estimator_profile_has_both_dynamic_axes():
    t = _tts(trt=True)

    p = t._trt_profile("vector_estimator", lat=40, txt=50)

    assert "noisy_latent:1x144x40" in p
    assert "text_emb:1x256x50" in p
    assert "latent_mask:1x1x40" in p and "text_mask:1x1x50" in p


def test_vocoder_profile_only_has_the_latent_axis():
    t = _tts(trt=True)

    p = t._trt_profile("vocoder", lat=40, txt=50)

    assert p == "latent:1x144x40"


def test_max_profile_follows_the_configured_limits():
    """실기 문장이 이 범위를 넘으면 엔진을 다시 굽는다 — 설정으로 올릴 수 있어야."""
    t = _tts(trt=True, trt_max_latent=300, trt_max_text=400)

    p = t._trt_profile("vector_estimator", lat=t.trt_max_latent, txt=t.trt_max_text)

    assert "noisy_latent:1x144x300" in p and "text_emb:1x256x400" in p


# ── 프로파일을 벗어난 문장 ───────────────────────────────────────────────────
# 🔴 2026-08-24 젯슨에서 실제로 터졌다. 프로파일 상한(latent 200)을 넘는 문장을 주면:
#      Set dimension [1,144,222] ... does not satisfy any optimization profiles
#      -> RuntimeException -> **합성이 통째로 실패**
#    이 모듈이 스스로 정한 원칙이 "아이 앞에서 소리가 안 나는 것이 최악" 이다.
#    상한을 아무리 높여도 넘는 문장은 언제든 나올 수 있으므로, 값이 아니라 **경로**로
#    막는다 — 그 문장만 원래 CUDA 세션으로 다시 합성한다.

def test_out_of_profile_sentence_still_produces_audio(trt_available, monkeypatch,
                                                      caplog):
    t = _tts(trt=True, fail_on_trt=True)
    monkeypatch.setattr(t, "_make_trt_session", lambda name: f"trt-{name}")
    t._apply_trt()

    with caplog.at_level("WARNING"):
        out = t._infer_local("아주 긴 문장")

    assert len(out) == 8, "TRT 가 실패했는데 소리가 안 났다 — 최악의 결과다"
    assert t._tts.seen[0].startswith("trt"), "처음엔 TRT 로 시도해야 한다"
    assert not str(t._tts.seen[1]).startswith("trt"), "폴백은 원래 세션이어야 한다"


def test_the_fallback_is_logged(trt_available, monkeypatch, caplog):
    t = _tts(trt=True, fail_on_trt=True)
    monkeypatch.setattr(t, "_make_trt_session", lambda name: f"trt-{name}")
    t._apply_trt()

    with caplog.at_level("WARNING"):
        t._infer_local("아주 긴 문장")

    assert any(r.levelname == "WARNING" for r in caplog.records),         "조용히 느려지면 원인을 못 찾는다"


def test_trt_is_restored_after_the_fallback(trt_available, monkeypatch):
    """한 문장이 길다고 그 뒤 전부를 느리게 만들면 안 된다."""
    t = _tts(trt=True, fail_on_trt=True)
    monkeypatch.setattr(t, "_make_trt_session", lambda name: f"trt-{name}")
    t._apply_trt()

    t._infer_local("아주 긴 문장")

    assert t._tts.model.vector_est_ort == "trt-vector_estimator"
    assert t._tts.model.vocoder_ort == "trt-vocoder"


def test_the_retry_reseeds_so_the_voice_stays_the_same(trt_available, monkeypatch):
    """🔴 실패한 시도가 이미 난수를 뽑아 썼다. 다시 시드를 박지 않으면 목소리가 달라진다."""
    seeds = []
    monkeypatch.setattr(np.random, "seed", lambda v: seeds.append(v))
    t = _tts(trt=True, fail_on_trt=True, seed=777)
    monkeypatch.setattr(t, "_make_trt_session", lambda name: f"trt-{name}")
    t._apply_trt()

    t._infer_local("아주 긴 문장")

    assert seeds == [777, 777], f"시드를 두 번 박아야 한다: {seeds}"


def test_without_trt_a_failure_is_not_swallowed(monkeypatch):
    """TRT 를 안 쓰는데 합성이 실패하면 그건 진짜 문제다 — 숨기면 안 된다."""
    t = _tts(trt=False)

    def boom(text, **kw):
        raise RuntimeError("진짜 고장")

    t._tts.synthesize = boom

    with pytest.raises(RuntimeError, match="진짜 고장"):
        t._infer_local("안녕")
