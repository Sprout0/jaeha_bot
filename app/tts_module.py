"""TTS: Supertonic(한국어) 기반 텍스트 -> 친근한 음성.

Piper 공식엔 한국어가 없고, 커뮤니티 KSS(goruut) 모델은 발음 누락/오류가 있었으며,
MeloTTS 는 이 환경(Windows/py3.12)에서 네이티브 빌드가 막혀 설치 불가였다.
그래서 Supertonic-3 로 확정. [[jaeha-bot-progress]] 참고.

Supertonic 장점(우리 목표에 부합):
- ONNX Runtime 로 구동 → 경량, torch 불필요 → Jetson 적합
- 한국어 품질 양호(음절 누락 없음), 합성 ~2초/문장
- OpenRAIL-M 라이선스(상업 친화적)
모델 가중치는 첫 실행 때 HuggingFace(Supertone/supertonic-3)에서 자동 다운로드.

목표(가이드 4-2): 짧은 문장 위주, 끊김 0건.
REPL 테스트: python -m app.tts_module
"""
from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("jaeha_bot.tts")

# 문장 경계(마침표/물음표/느낌표 등). 문장 스트리밍용 분리에 사용.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？…])\s+|\n+")

# 재생 앞뒤에 덧대는 무음(초). 장치 스트림 시작·종료 지연을 흡수한다.
# 앞쪽 무음 동안은 **아직 아무 소리도 안 난다** — 체감 지연에 포함해야 한다.
PLAY_PAD_S = 0.15

# OpenAI TTS 스트리밍 재생 관련.
OPENAI_PCM_RATE = 24000   # response_format="pcm" 은 24kHz 16bit mono LE 고정(컨테이너 없음)
STREAM_CHUNK = 4096       # HTTP 조각 크기
# 재생 시작 전에 모아 둘 분량(초). 네트워크가 잠깐 멈춰도 말이 끊기지 않게 하는 완충이다.
# 🔴 싸지 않다 — 젯슨 실측(2026-08-10) 첫 소리 중앙:
#      0.05s -> 0.650s | 0.15s -> 0.601s | 0.25s -> 1.030s
#    API 가 첫 조각을 보낸 뒤 잠깐 쉬어서, 0.25초어치를 기다리면 그 공백을 그대로 문다.
#    0.15 가 최적점이다. 더 줄여도 이득이 없고 굶을 위험만 커진다.
STREAM_PREBUFFER_S = 0.15


class _StreamResampler:
    """조각 경계를 이어 붙이는 리샘플러(스트리밍 전용).

    조각마다 따로 리샘플하면 경계에서 파형이 튀어 '탁' 소리가 난다. 상태를 들고 있어야 한다.
    soxr 가 있으면 ResampleStream(고품질), 없으면 직전 조각 꼬리를 물고 가는 선형보간.
    (젯슨엔 soxr 가 있고 노트북엔 없다 — 버퍼 경로의 _resample 과 같은 사정이다.)
    """

    def __init__(self, src: int, dst: int) -> None:
        self.src, self.dst = src, dst
        self._soxr = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0.0
        if src == dst:
            return
        try:
            import soxr

            self._soxr = soxr.ResampleStream(src, dst, 1, dtype="float32")
        except Exception:      # 미설치·미지원이면 선형 폴백
            log.info("soxr 없음 — 스트리밍 리샘플을 선형보간으로 처리한다(%d->%d)", src, dst)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.src == self.dst:
            return frames
        if self._soxr is not None:
            return np.asarray(self._soxr.resample_chunk(frames), dtype=np.float32)
        # 선형 폴백: 남은 입력에 이어 붙이고, 뽑을 수 있는 출력 좌표까지만 소비한다.
        self._buf = np.concatenate([self._buf, frames])
        step = self.src / self.dst
        last = len(self._buf) - 1
        if last < 0 or self._pos > last:
            return np.zeros(0, dtype=np.float32)
        n = int((last - self._pos) / step) + 1
        idx = self._pos + step * np.arange(n)
        out = np.interp(idx, np.arange(len(self._buf)), self._buf).astype(np.float32)
        keep = int(idx[-1])                    # 다음 보간에 필요한 왼쪽 샘플부터 남긴다
        self._buf = self._buf[keep:]
        self._pos = idx[-1] + step - keep
        return out

    def flush(self) -> np.ndarray:
        """리샘플러 안에 남은 꼬리를 꺼낸다. 안 부르면 발화 끝이 조금씩 깎인다.

        soxr 는 내부 버퍼를 들고 있어 last=True 로 비워 줘야 한다(젯슨 실측 12ms/발화).
        """
        if self.src == self.dst:
            return np.zeros(0, dtype=np.float32)
        if self._soxr is not None:
            return np.asarray(
                self._soxr.resample_chunk(np.zeros(0, dtype=np.float32), last=True),
                dtype=np.float32)
        # 선형 폴백: 마지막 샘플까지 한 번 더 훑고 버퍼를 비운다.
        tail, self._buf = self._buf, np.zeros(0, dtype=np.float32)
        if tail.size < 2:
            return np.zeros(0, dtype=np.float32)
        step = self.src / self.dst
        n = int((len(tail) - 1 - self._pos) / step) + 1
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = self._pos + step * np.arange(n)
        self._pos = 0.0
        return np.interp(idx, np.arange(len(tail)), tail).astype(np.float32)


class _SoundDeviceSink:
    """실제 스피커. write() 가 블로킹이라 재생 속도에 맞춰 자연히 조절된다."""

    def __init__(self, rate: int) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
        self._stream.start()

    def write(self, frames: np.ndarray) -> None:
        self._stream.write(np.ascontiguousarray(frames, dtype=np.float32))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


@dataclass(frozen=True)
class SpeakTiming:
    """speak() 한 번의 시간 분해. '지연'과 '봇이 말하는 시간'을 섞지 않으려고 나눈다.

    first_audio_s: 부르고 나서 **첫 소리가 나기까지**. 아이가 체감하는 대기시간이 이것.
    synth_s      : 합성 연산 시간(모델 로드가 필요했다면 그것도 포함).
    play_s       : 소리가 나기 시작해서 끝날 때까지 = 봇이 말하는 시간. 지연이 아니다.
    total_s      : first_audio_s + play_s. 예전 tts_s 와 같은 값.
    """

    first_audio_s: float
    synth_s: float
    play_s: float

    @property
    def total_s(self) -> float:
        return self.first_audio_s + self.play_s


@contextlib.contextmanager
def _suppress_c_stderr():
    """ALSA/PortAudio 가 C 레벨(fd 2)로 찍는 경고를 잠깐 가린다.

    장치 지원 레이트를 확인(check_output_settings)할 때 미지원이면 libasound 가
    무해한 경고를 stderr 로 쏟는다. 파이썬 로깅이 아니라 C 레벨이라 이렇게 fd 를
    잠시 /dev/null 로 돌려야 가려진다. 확인이 끝나면 원복.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class TTSModule:
    def __init__(
        self,
        model: str = "supertonic-3",
        voice: str = "F1",          # 유아용 밝은 여성. 대안: F2~F5, M1~M5
        language: str = "ko",
        *,
        speed: float = 1.0,         # 0.7~2.0. 유아용은 1.0 또는 약간 느리게(0.9)
        total_steps: int = 8,       # 높을수록 품질↑/느려짐
        threads: int = 4,           # onnxruntime intra-op 스레드. 실측 최적=4(6/8은 경합으로↓)
        providers: list[str] | None = None,  # onnxruntime 실행 프로바이더(GPU 가속).
                                    # None 이면 supertonic 기본값(CPU). 젯슨은 config 에서 지정.
        seed: int | None = 1234,    # diffusion 초기 노이즈 시드 고정 → 말투·속도 일정.
                                    # None 이면 매번 랜덤(말투 들쭉날쭉). 값 바꾸면 목소리 캐릭터가 달라짐.
        gap: float = 0.2,           # 문장 사이 삽입 무음(초). 앞뒤 pad 합쳐 총 간격≈0.3s
                                    # (TTS 표준 문장 간격). 더 붙이려면 0.15, 더 띄우려면 0.3.
        # ── 원격 백엔드(선택) ────────────────────────────────────────────────
        # "supertonic"(기본, 로컬) | "openai". openai 여도 Supertonic 은 계속 로드한다 —
        # 폴백이 차가우면(로드 3.5초) 네트워크가 끊긴 순간 아이가 그만큼 기다린다.
        backend: str = "supertonic",
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "shimmer",   # 청취 비교로 채택(2026-08-10). 대안: coral
        openai_instructions: str = "",   # 톤·속도 지시. tts-1 계열은 이 인자를 받지 않으니 비워둘 것
        openai_timeout: float = 3.0,     # 실측 p95 1.72s / 최대 2.24s → 3초면 이상 상황만 걸린다
        # 조각이 오는 대로 재생한다(첫 소리를 앞당긴다). 통짜 수신은 1.25s, 스트리밍은 0.6s 대.
        # ⚠️ 소리가 한 번 나가면 폴백이 불가능하다 — 중간에 끊기면 말이 잘린 채로 끝난다.
        openai_stream: bool = True,
        # ── TensorRT(젯슨 전용 가속) ─────────────────────────────────────────
        # 확산·보코더 세션을 TRT 로 갈아끼운다. 자세한 근거는 _apply_trt 참고.
        # ⚠️ 노트북엔 TRT 가 없다. 같은 config 를 공유하므로 없으면 조용히 지나간다.
        trt: bool = False,
        trt_cache: str = "~/.cache/trt_supertonic",
        trt_max_latent: int = 200,   # latent 200 ≈ 13.9초 발화. 답변은 2~4초라 넉넉하다
        trt_max_text: int = 256,
    ) -> None:
        self.model = model
        self.voice = voice
        self.language = language
        self.speed = speed
        self.total_steps = total_steps
        self.threads = threads
        self.providers = list(providers) if providers else None
        self.seed = seed
        self.gap = gap
        self.backend = backend
        self.openai_model = openai_model
        self.openai_voice = openai_voice
        self.openai_instructions = openai_instructions
        self.openai_timeout = openai_timeout
        self.openai_stream = openai_stream
        self.trt = trt
        self.trt_cache = trt_cache
        self.trt_max_latent = trt_max_latent
        self.trt_max_text = trt_max_text
        self._trt_backup = {}     # TRT 로 갈아끼우기 **전**의 세션. 폴백용으로 들고 있는다
        self._oa = None           # OpenAI 클라이언트(첫 사용 때 1회 생성)
        self._tts = None
        self._style = None
        self.sample_rate = 44100  # load() 에서 실제 값으로 갱신
        self._play_rate = None    # 재생 장치가 지원하는 레이트(첫 재생 때 1회 결정·캐시)

    # ------------------------------------------------------------------ 로딩
    def load(self):
        if self._tts is not None:
            return
        from supertonic import TTS

        self._apply_providers()
        self._tts = TTS(
            model=self.model,
            auto_download=True,
            intra_op_num_threads=self.threads,
        )
        self._report_providers()
        self._apply_trt()
        self._style = self._resolve_style(self.voice)
        self.sample_rate = int(self._tts.sample_rate)

    # ------------------------------------------------- 실행 프로바이더(GPU 가속)
    def _apply_providers(self) -> None:
        """onnxruntime 실행 프로바이더를 갈아끼운다. TTS 생성 '전에' 불러야 한다.

        supertonic 은 providers 인자를 받지 않고 config.DEFAULT_ONNX_PROVIDERS
        (=["CPUExecutionProvider"] 하나)를 쓴다.

        🔴 함정: loader.py 가 `from .config import DEFAULT_ONNX_PROVIDERS` 로 이름을
        **바인딩**해두기 때문에 supertonic.config 쪽을 고치면 조용히 무시된다.
        반드시 supertonic.loader 의 이름을 덮어써야 한다.

        loader 는 요청 목록을 ort.get_available_providers() 와 교집합하고 비면 CPU 로
        조용히 폴백한다 — 예외가 없으므로 _report_providers() 로 결과를 확인한다.
        """
        if not self.providers:
            return
        from supertonic import loader

        loader.DEFAULT_ONNX_PROVIDERS = list(self.providers)

    def _report_providers(self) -> None:
        """실제로 무엇이 붙었는지 확인해 남긴다(요청과 결과가 다를 수 있다)."""
        if not self.providers:
            return
        try:
            import onnxruntime as ort

            got = self._tts.model.vocoder_ort.get_providers()
        except Exception as e:  # 버전이 올라 내부 구조가 바뀌어도 합성은 계속돼야 한다
            log.warning("프로바이더 확인 실패(%s: %s) — 합성은 계속", type(e).__name__, e)
            return
        log.info("onnxruntime 프로바이더: %s", got)
        # 요청했고 이 기기에 있는데도 안 붙었다면 진짜 문제다(위 함정에 빠진 경우).
        available = set(ort.get_available_providers())
        missed = [p for p in self.providers if p in available and p not in got]
        if missed:
            log.warning("요청한 프로바이더가 붙지 않음: %s — CPU 로 동작 중", missed)

    # ------------------------------------------------- TensorRT(젯슨 전용 가속)
    # 어떤 세션을 바꿀지. dp·text_enc 는 뺀다 — TRT 빌드가 실패하고(Pad 출력에 shape
    # 없음, ORT-TRT 알려진 제약) 모양 고정 상태에서 합쳐 20ms 뿐이라 값어치도 없다.
    _TRT_TARGETS = (("vector_estimator", "vector_est_ort"), ("vocoder", "vocoder_ort"))
    # (latent, text) 최솟값. 1 로 잡는다 — TRT 는 opt 를 기준으로 최적화하고 min/max 는
    # 유효 범위일 뿐이라 낮춰도 손해가 없다. 반대로 짧은 발화("응")가 min 아래로 내려가면
    # 프로파일을 벗어나 위험하다. 참고: '고양이는 야옹 하고 울어.'(12자)가 text_ids 37 이다.
    _TRT_MIN = (1, 1)
    _TRT_OPT = (40, 40)    # 최적화 기준점. 실기 답변 18~33자가 이 근처다

    def _trt_profile(self, name: str, lat: int, txt: int) -> str:
        """TRT 최적화 프로파일의 shape 문자열."""
        if name == "vocoder":
            return f"latent:1x144x{lat}"
        return (f"noisy_latent:1x144x{lat},text_emb:1x256x{txt},style_ttl:1x50x256,"
                f"latent_mask:1x1x{lat},text_mask:1x1x{txt},"
                f"current_step:1,total_step:1")

    def _make_trt_session(self, name: str):
        """같은 onnx 파일을 TensorRT 프로바이더로 다시 연다."""
        import onnxruntime as ort

        attr = dict(self._TRT_TARGETS)[name]
        # 경로는 현재 세션에서 가져온다 — 캐시 위치가 바뀌어도 따라간다.
        # ⚠️ _model_path 는 onnxruntime 내부 속성이다. 버전이 올라 사라지면 여기서
        #    AttributeError 가 나고, _apply_trt 가 잡아 원래 세션을 그대로 둔다.
        path = getattr(self._tts.model, attr)._model_path
        cache = str(Path(self.trt_cache).expanduser())
        Path(cache).mkdir(parents=True, exist_ok=True)

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts = {
            "trt_engine_cache_enable": True,     # 없으면 기동마다 72초씩 굽는다
            "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True,
            "trt_profile_min_shapes": self._trt_profile(name, *self._TRT_MIN),
            "trt_profile_opt_shapes": self._trt_profile(name, *self._TRT_OPT),
            "trt_profile_max_shapes": self._trt_profile(
                name, self.trt_max_latent, self.trt_max_text),
        }
        return ort.InferenceSession(
            str(path), sess_options=so,
            providers=[("TensorrtExecutionProvider", opts),
                       ("CUDAExecutionProvider", {}), "CPUExecutionProvider"])

    def _apply_trt(self) -> None:
        """확산·보코더를 TensorRT 세션으로 갈아끼운다. 못 하면 원래 것을 지킨다.

        🔴 왜 (2026-08-24 젯슨 실측). CUDA EP 는 **직전 호출과 입력 모양이 다르면**
           확산 1스텝에 ~180ms 를 문다. 두 모양을 각각 6번 예열한 뒤에도 그렇다:
             A 181.2ms(직전과 다름) → 50.2 → 27.1 → 27.0  |  B 184.5ms  |  A 180.6ms
           문장마다 길이가 달라 실기에선 매번 낸다. '처음 보는 모양'이 아니라 '직전과
           다른 모양'이 비용이라, 모양 가짓수를 줄이는 버킷팅은 **원리적으로 무의미**하다.
           유일한 방법인 패딩은 소리를 바꾼다(text_enc 최대차 0.444 / vector_est 0.206).
           ORT 옵션(cudnn_conv_algo_search·mem_pattern·arena)도 전부 무효였다.
           TRT 는 min/opt/max 프로파일로 엔진을 미리 구워 이 비용을 없앤다:
             모양 바뀜 +271.5ms → +10.7ms, 고정 상태에서도 2.4배 빠름
           문장 합성(길이 섞인 실기 조건) CUDA@12 817ms → **TRT@24 453.8ms**.

        ⚠️ 기동이 6.4초 늘어난다(엔진 캐시가 더울 때. 첫 빌드는 확산 72s + 보코더 49s).
           캐시를 지우면 그 값을 다시 문다.
        ⚠️ 노트북엔 TRT 가 없다. 같은 configs/model_paths.yaml 을 쓰므로 여기서
           죽으면 안 된다 — 없으면 경고 없이 지나가고 CUDA/CPU 로 계속 간다.
        🔴 반대로 젯슨에서 조용히 실패하면 '왜 느려졌는지'를 못 찾는다. 실패는 경고로.
        """
        if not self.trt or self._tts is None:
            return
        try:
            import onnxruntime as ort

            if "TensorrtExecutionProvider" not in ort.get_available_providers():
                log.info("TensorRT 없음 — 기존 프로바이더로 계속(노트북에선 정상)")
                return
        except Exception as e:
            log.info("TensorRT 확인 불가(%s) — 기존 프로바이더로 계속", type(e).__name__)
            return

        for name, attr in self._TRT_TARGETS:
            t0 = time.perf_counter()
            try:
                sess = self._make_trt_session(name)
            except Exception as e:   # 하나 실패해도 봇은 말을 해야 한다
                log.warning("TensorRT 적용 실패 [%s] (%s: %s) — 기존 세션 유지",
                            name, type(e).__name__, str(e)[:160])
                continue
            # 원래 세션은 버리지 않는다 — 프로파일을 벗어난 문장이 오면 되돌아간다.
            self._trt_backup[attr] = getattr(self._tts.model, attr)
            setattr(self._tts.model, attr, sess)
            log.info("TensorRT 적용 [%s] %.1fs", name, time.perf_counter() - t0)

    def _without_trt(self, fn):
        """이번 한 번만 원래 CUDA 세션으로 돌려 실행하고, 끝나면 TRT 를 되돌린다.

        ⚠️ 영구히 되돌리지 않는 이유: 긴 문장 하나 때문에 그 뒤 모든 발화를 두 배 느리게
           만들 이유가 없다. 긴 문장은 드물다.
        """
        swapped = {}
        for attr, original in self._trt_backup.items():
            swapped[attr] = getattr(self._tts.model, attr)
            setattr(self._tts.model, attr, original)
        try:
            return fn()
        finally:
            for attr, sess in swapped.items():
                setattr(self._tts.model, attr, sess)

    # ---------------------------------------------------- 보이스 스타일 해석(+블렌딩)
    def _parse_blend(self, spec: str) -> list[tuple[str, float]]:
        """'F1:0.7+F4:0.3' -> [('F1', 0.7), ('F4', 0.3)]. 가중치는 합이 1 이 되게 정규화.

        정규화하는 이유: 'F1:3+F4:1'(3:1 비율)처럼 쓰는 게 사람에게 자연스럽다.
        가중치를 안 적으면 1.0 으로 보므로 'F1+F4' 는 그대로 균등이 된다.
        """
        parts: list[tuple[str, float]] = []
        for term in spec.split("+"):
            term = term.strip()
            if not term:
                continue
            name, _, w = term.partition(":")
            parts.append((name.strip(), float(w) if w.strip() else 1.0))
        if not parts:
            raise ValueError(f"목소리 지정이 비었다: {spec!r}")
        total = sum(w for _, w in parts)
        if total <= 0:
            raise ValueError(f"가중치 합이 0 이다: {spec!r}")
        return [(n, w / total) for n, w in parts]

    def _resolve_style(self, voice: str):
        """voice 문자열 -> Style.

        Supertonic 의 voice style 은 numpy 벡터 2개일 뿐이라 선형 평균으로 새 화자를
        만들 수 있다(style latent 는 연속·보간 가능 공간):
            ttl = 음색(50x256)   dp = 리듬·발화 속도감(8x16)
        이 둘이 **따로** 있다는 게 중요하다. 묶어서 평균내면 '밝은 음색'을 가져올 때
        '촐랑거리는 리듬'까지 딸려온다 — 2세에겐 그게 알아듣기 어렵다.

        문법:
            F1                  단일 프리셋 (M1~M5 / F1~F5)
            F1+F4               균등 블렌드
            F1:0.7+F4:0.3       가중 블렌드 (비율이라 'F1:3+F4:1' 도 같은 뜻)
            F1|F4               음색 F1, 리듬 F4
            F1:0.7+F2:0.3|F4    가중 음색 + 리듬 F4

        후보를 귀로 고르는 도구는 tools/voice_tournament.py.
        🔴 반드시 젯슨 실기로 들을 것 — 파일 청취로 고르면 08-10 을 반복한다.
        """
        from supertonic.core import Style

        spec = str(voice)
        timbre_spec, sep, rhythm_spec = spec.partition("|")
        if not sep:
            rhythm_spec = timbre_spec

        timbre = self._parse_blend(timbre_spec)
        rhythm = self._parse_blend(rhythm_spec)
        # 단일 프리셋이고 분리도 없으면 프리셋을 그대로 쓴다(부동소수 오차 0).
        if not sep and len(timbre) == 1:
            return self._tts.get_voice_style(timbre[0][0])

        ttl = sum(w * self._tts.get_voice_style(n).ttl for n, w in timbre)
        dp = sum(w * self._tts.get_voice_style(n).dp for n, w in rhythm)
        return Style(np.asarray(ttl, dtype=np.float32), np.asarray(dp, dtype=np.float32))

    # ---------------------------------------------------------- 합성(-> numpy)
    def _infer(self, text: str) -> np.ndarray:
        """백엔드를 골라 합성한다. 원격이 실패하면 **말없이 멈추지 않고** 로컬로 내려간다.

        아이 앞에서 소리가 안 나는 것이 최악이므로 폴백은 예외를 밖으로 던지지 않는다.
        대신 반드시 경고를 남긴다 — 조용한 폴백은 '느려진 이유'를 못 찾게 만든다.
        """
        self.load()   # 원격을 쓰더라도 폴백을 미리 데워 둔다(로드 3.5초를 실패 시점에 치르지 않게)
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if self.backend == "openai":
            try:
                return self._infer_openai(text)
            except Exception as e:  # 네트워크·인증·타임아웃 무엇이든 로컬로 간다
                log.warning("OpenAI TTS 실패(%s: %s) — Supertonic 으로 폴백",
                            type(e).__name__, str(e)[:120])
        return self._infer_local(text)

    def warm(self) -> None:
        """원격 백엔드면 API 커넥션(DNS·TCP·TLS)을 미리 맺어 둔다.

        LLM 과 같은 사정이다 — 첫 호출만 3초대이고 그 다음부터 0.5~0.6초다.
        한 글자만 합성해 연결을 열고 오디오는 버린다(과금은 무시할 수준).
        로컬 백엔드면 아무것도 하지 않는다. 실패해도 기동을 막지 않는다.
        """
        if self.backend != "openai":
            return
        try:
            self._openai_client().audio.speech.create(
                model=self.openai_model, voice=self.openai_voice,
                input="응", response_format="wav",
            )
        except Exception as e:
            log.warning("TTS API 커넥션 예열 실패(%s: %s) — 첫 발화가 느려질 뿐이다",
                        type(e).__name__, str(e)[:120])

    # ------------------------------------------------------- 원격 합성(OpenAI)
    def _infer_openai(self, text: str) -> np.ndarray:
        """gpt-4o-mini-tts -> wav -> 모듈 샘플레이트 float32 mono.

        wav 를 쓰는 이유: soundfile 로 어디서나 디코드된다(mp3 는 libsndfile 버전을 탄다).
        3초 발화에서 wav-mp3 전송량 차이는 체감 지연에 묻힌다.
        ⚠️ 지금은 통짜로 받아서 재생한다 — 첫 소리 = 다운로드 완료 시각이다.
           스트리밍 재생(response_format="pcm")으로 바꾸면 여기서 더 줄일 여지가 있다.
        """
        import io

        import soundfile as sf

        kw = dict(model=self.openai_model, voice=self.openai_voice,
                  input=text, response_format="wav")
        if self.openai_instructions:
            kw["instructions"] = self.openai_instructions
        data = self._openai_client().audio.speech.create(**kw).content

        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if audio.ndim > 1:                      # 혹시 스테레오로 오면 모노로
            audio = audio.mean(axis=1)
        return self._resample(np.asarray(audio, dtype=np.float32), rate, self.sample_rate)

    def _openai_client(self):
        if self._oa is None:
            from openai import OpenAI

            self._oa = OpenAI(timeout=self.openai_timeout)
        return self._oa

    # ------------------------------------------------------- 스트리밍 재생(OpenAI)
    @staticmethod
    def _iter_pcm_floats(byte_iter) -> "Iterator[np.ndarray]":
        """PCM 바이트 조각 -> float32 프레임.

        16bit 라 **2바이트가 한 샘플**이다. HTTP 조각 경계는 샘플을 반으로 가를 수 있으므로
        남은 1바이트를 다음 조각 앞에 이어 붙인다. 안 그러면 이후 전체가 한 바이트씩 밀려
        잡음이 된다. 맨 끝에 남는 홀수 바이트는 버린다(반쪽 샘플은 복원할 수 없다).
        """
        leftover = b""
        for raw in byte_iter:
            if not raw:
                continue
            buf = leftover + raw
            usable = len(buf) - (len(buf) % 2)
            leftover = buf[usable:]
            if usable:
                yield np.frombuffer(buf[:usable], dtype="<i2").astype(np.float32) / 32768.0

    def _speak_streaming(self, byte_iter, src_rate: int, make_sink) -> bool:
        """조각이 오는 대로 흘려보낸다. 첫 소리가 나기 전에 실패하면 False(호출부가 폴백).

        🔴 폴백 판단은 여기서 끝난다 — 스피커에 한 조각이라도 나간 뒤에는 되돌릴 수 없다.
           그래서 소리를 내기 전까지의 실패와 그 뒤의 실패를 구분해 다루고, 후자는
           '말이 잘렸다'고 남긴다(조용히 끝나면 원인을 못 찾는다).
        """
        rate = self._play_rate or src_rate
        resample = _StreamResampler(src_rate, rate)
        prebuffer = int(STREAM_PREBUFFER_S * rate)
        sink, pending, buffered = None, [], 0
        try:
            for frames in self._iter_pcm_floats(byte_iter):
                out = resample(frames)
                if out.size == 0:
                    continue
                if sink is not None:
                    sink.write(out)
                    continue
                pending.append(out)
                buffered += out.size
                if buffered >= prebuffer:      # 완충이 찼으니 이제 소리를 낸다
                    sink = make_sink(rate)
                    for part in pending:
                        sink.write(part)
                    pending.clear()
            tail = resample.flush()            # 리샘플러 안에 남은 꼬리(말끝) 회수
            if tail.size:
                pending.append(tail)
            if sink is None and pending:       # 완충도 못 채우고 끝난 짧은 발화
                sink = make_sink(rate)
                for part in pending:
                    sink.write(part)
            elif sink is not None and tail.size:
                sink.write(tail)
        except Exception as e:
            if sink is None:
                log.warning("OpenAI TTS 스트리밍 실패(%s: %s) — 버퍼 경로로 폴백",
                            type(e).__name__, str(e)[:120])
                return False
            log.warning("OpenAI TTS 재생 도중 끊김(%s: %s) — 말이 잘렸다",
                        type(e).__name__, str(e)[:120])
        finally:
            if sink is not None:
                sink.close()
        return sink is not None

    def _speak_openai_stream(self, text: str, t0: float) -> "SpeakTiming | None":
        """스트리밍 경로 전체. 첫 소리 전에 실패하면 None 을 돌려 버퍼 경로로 넘긴다."""
        import sounddevice as sd

        self._resolve_play_rate(sd)   # 장치가 받는 레이트를 먼저 확정한다
        started: dict[str, float] = {}

        def make_sink(rate: int):
            started["t"] = time.perf_counter()
            return _SoundDeviceSink(rate)

        kw = dict(model=self.openai_model, voice=self.openai_voice,
                  input=text, response_format="pcm")
        if self.openai_instructions:
            kw["instructions"] = self.openai_instructions
        try:
            with self._openai_client().audio.speech.with_streaming_response.create(**kw) as resp:
                ok = self._speak_streaming(resp.iter_bytes(STREAM_CHUNK),
                                           OPENAI_PCM_RATE, make_sink)
        except Exception as e:   # 연결 자체가 안 된 경우 — 아직 아무 소리도 안 났다
            log.warning("OpenAI TTS 연결 실패(%s: %s) — 버퍼 경로로 폴백",
                        type(e).__name__, str(e)[:120])
            return None
        if not ok:
            return None
        # 첫 소리 = 스트림을 연 시각. 장치 자체 버퍼 지연은 여기 안 잡히므로 약간 낙관적이다.
        first_audio_s = started["t"] - t0
        return SpeakTiming(first_audio_s=first_audio_s, synth_s=first_audio_s,
                           play_s=max(0.0, time.perf_counter() - started["t"]))

    # ------------------------------------------------------ 로컬 합성(Supertonic)
    def _infer_local(self, text: str) -> np.ndarray:
        """Supertonic 합성. TRT 가 걸려 있고 실패하면 그 문장만 CUDA 로 다시 만든다.

        🔴 2026-08-24 젯슨에서 실제로 터졌다. TRT 프로파일 상한(latent 200)을 넘는
           문장을 주면 `does not satisfy any optimization profiles` 로 예외가 나고
           **합성이 통째로 실패**한다 — 이 모듈이 스스로 정한 최악("아이 앞에서 소리가
           안 나는 것")이다. 상한을 아무리 올려도 넘는 문장은 언제든 나올 수 있으므로
           값이 아니라 경로로 막는다.
        ⚠️ TRT 를 안 쓸 땐 예외를 그대로 올린다. 그건 진짜 고장이고 숨기면 안 된다.
        """
        try:
            return self._synthesize_local(text)
        except Exception as e:
            if not self._trt_backup:
                raise
            log.warning("TensorRT 합성 실패(%s: %s) — 이 문장만 기존 세션으로 다시",
                        type(e).__name__, str(e)[:160])
            return self._without_trt(lambda: self._synthesize_local(text))

    def _synthesize_local(self, text: str) -> np.ndarray:
        # Supertonic 은 초기 노이즈를 np.random.randn(전역 RNG)으로 뽑는다. 합성 직전
        # 시드를 고정하면 문장마다 같은 노이즈에서 출발해 말투·속도가 일정해진다.
        # 🔴 재시도에서도 반드시 다시 박아야 한다 — 실패한 시도가 이미 난수를 써버려서,
        #    안 박으면 폴백된 문장만 목소리가 달라진다.
        if self.seed is not None:
            np.random.seed(self.seed)
        wav, _dur = self._tts.synthesize(
            text,
            voice_style=self._style,
            lang=self.language,
            speed=self.speed,
            total_steps=self.total_steps,
        )
        return np.asarray(wav, dtype=np.float32).squeeze()

    # ------------------------------------------------------------ 파일로 저장
    def synthesize(self, text: str, out_path: str = "logs/last_tts.wav") -> str:
        """텍스트 -> wav 경로."""
        import soundfile as sf

        audio = self._infer(text)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, self.sample_rate)
        return str(out)

    # ------------------------------------------------------- 문장 분리(스트리밍용)
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
        return parts or [text]

    # ------------------------------------------------ 앞뒤 flat 무음 제거(공백↓·뚝끊김 없음)
    def _trim(self, audio: np.ndarray, keep_lead: float = 0.04, keep_tail: float | None = None,
              thr: float = 1e-3, fade_in: float = 0.01) -> np.ndarray:
        """앞 무음을 잘라 시작을 빠르게 한다. 끝(말끝 여운)은 기본적으로 건드리지 않는다.

        - 임계값 thr=1e-3(≈-50dB) 이하는 사실상 무음이라 앞부분을 잘라도 안 들린다.
        - keep_tail=None(기본): **끝을 자르지 않는다** → Supertonic 의 자연스러운 말끝 여운을
          그대로 살려 '뚝 끊김'이 없다. (통짜 합성은 문장 이어붙이기가 없어 끝을 자를 이유가 없음.)
        - keep_tail 에 값을 주면 그만큼만 남기고 뒤 flat 무음도 자른다(문장 이어붙일 때만 사용).
          이때도 **페이드아웃은 절대 안 한다**(예전 뚝 끊김의 원인).
        - 잘린 '앞'만 10ms 페이드인(자를 때 생길 수 있는 click 방지). 앞은 어차피 무음이라 무해.
        """
        if audio.size == 0:
            return audio
        nz = np.where(np.abs(audio) > thr)[0]
        if nz.size == 0:
            return audio
        sr = self.sample_rate
        start = max(0, int(nz[0]) - int(keep_lead * sr))
        end = audio.size if keep_tail is None else min(audio.size, int(nz[-1]) + int(keep_tail * sr) + 1)
        out = audio[start:end]
        fi = int(fade_in * sr)
        if fi > 0 and out.size > fi:
            out = out.copy()
            out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)  # 잘린 시작만 매끄럽게(뒤는 손대지 않음)
        return out

    def _silence(self, dur: float) -> np.ndarray:
        return np.zeros(int(dur * self.sample_rate), dtype=np.float32)

    def _resample(self, audio: np.ndarray, src: int, dst: int) -> np.ndarray:
        """src->dst 샘플레이트 변환. 장치가 합성레이트(44100)를 못 받을 때만 사용
        (예: Jetson USB ReSpeaker=16000 전용). soxr(고품질) 우선, 없으면 선형보간 폴백."""
        if src == dst or audio.size == 0:
            return audio
        try:
            import soxr
            return soxr.resample(audio, src, dst).astype(np.float32)
        except Exception:
            n = int(round(audio.shape[0] * dst / src))
            xp = np.linspace(0.0, 1.0, audio.shape[0], dtype=np.float32)
            xn = np.linspace(0.0, 1.0, n, dtype=np.float32)
            return np.interp(xn, xp, audio).astype(np.float32)

    def _resolve_play_rate(self, sd) -> int:
        """재생에 쓸 샘플레이트를 1회 결정·캐시한다.

        장치가 합성 레이트(self.sample_rate)를 지원하면 그대로(리샘플 없음, PC 기본).
        미지원이면(예: Jetson USB ReSpeaker=16000 전용) 장치 기본 레이트에 맞춰
        리샘플한다. 미리 확인하므로 실패 로그(paInvalidSampleRate)가 뜨지 않는다.
        """
        if self._play_rate is not None:
            return self._play_rate
        rate = self.sample_rate
        try:
            with _suppress_c_stderr():
                sd.check_output_settings(samplerate=self.sample_rate)
        except Exception:
            try:
                dev = sd.default.device
                outdev = dev[1] if isinstance(dev, (list, tuple)) else dev
                dr = int(sd.query_devices(outdev, "output")["default_samplerate"])
                if dr:
                    rate = dr
            except Exception:
                pass
        self._play_rate = rate
        return rate

    # ---------------------------------------------------------------- 재생
    def speak(self, text: str) -> SpeakTiming:
        """통짜 합성 재생: 답변 전체를 한 번에 합성한다. 시간 분해를 돌려준다.

        문장을 쪼개 따로 합성(스트리밍)하면 문장마다 억양이 독립적이라 전환이 뚝뚝 끊겨
        '매끄럽지 못'하게 들린다. 전체를 한 번에 넘기면 Supertonic 이 문장 간 억양을
        자연스럽게 이어 준다(실측: 통짜가 쪼개기보다 합성도 더 빠름 2.34s vs 4.23s).
        앞뒤 flat 무음만 잘라(말끝 여운은 보존) 시작 지연과 끝 공백을 줄인다.
        (참고: 예전 문장 쪼개기+gap 스트리밍 방식은 git/메모리 기록 참조. 되돌리려면 그 버전으로.)

        ⚠️ 통짜 합성이라 **첫 소리 = 전체 합성이 끝난 뒤**다. 즉 답이 길수록 첫 소리도
           늦어진다. 여기서 재는 first_audio_s 가 그 값이며, 줄이려면 문장 스트리밍으로
           돌아가야 한다(대신 억양이 끊긴다) — 측정이 먼저다.
        """
        import sounddevice as sd

        t0 = time.perf_counter()
        self.load()
        # 스트리밍 경로: 조각이 오는 대로 재생해 첫 소리를 앞당긴다(실측 1.25s -> 0.6s대).
        # 실패해도 아직 소리가 안 났다면 None 이 와서 아래 버퍼 경로가 그대로 이어받는다.
        if self.backend == "openai" and self.openai_stream and (text or "").strip():
            timing = self._speak_openai_stream(text, t0)
            if timing is not None:
                return timing
        audio = self._trim(self._infer(text))
        synth_s = time.perf_counter() - t0
        if not audio.size:
            return SpeakTiming(first_audio_s=synth_s, synth_s=synth_s, play_s=0.0)

        # 실시간 재생(sounddevice, 기본 MME)은 스트림 시작·종료 지연이 커서
        # 버퍼 앞뒤 샘플을 흘린다 → 앞 발음 잘림 + 끝 '뚝' 끊김.
        # 앞뒤에 짧은 무음을 덧대 그 지연을 무음으로 흡수한다(합성 오디오는 손대지 않음).
        # 오디션 wav(파일 재생)에는 이 현상이 없어 앱만 빠르고 잘려 들렸던 원인.
        audio = np.concatenate([self._silence(PLAY_PAD_S), audio,
                                self._silence(PLAY_PAD_S)])
        # 장치가 합성 레이트를 지원하는지 미리 확인해 처음부터 맞는 레이트로 재생한다
        # (실패-후-재시도가 아님 → paInvalidSampleRate 로그 안 뜸).
        # PC 는 대개 그대로, Jetson USB(16000 전용)는 리샘플해서 재생.
        rate = self._resolve_play_rate(sd)
        play_audio = audio if rate == self.sample_rate else self._resample(audio, self.sample_rate, rate)
        sd.play(play_audio, rate)
        # sd.play 는 바로 돌아온다(재생은 뒤에서 계속). 여기까지가 '소리 나기 직전'이고,
        # 덧댄 앞 무음 동안은 아직 안 들리므로 그만큼 더한 게 진짜 첫 소리 시각이다.
        play_started = time.perf_counter()
        first_audio_s = play_started - t0 + PLAY_PAD_S
        sd.wait()
        return SpeakTiming(first_audio_s=first_audio_s, synth_s=synth_s,
                           play_s=max(0.0, time.perf_counter() - play_started))


def _repl() -> None:
    """터미널 TTS 테스트. 문장을 입력하면 소리로 말한다. 빈 줄이면 종료."""
    from .config import settings

    tts = TTSModule(**settings.models.get("tts", {}))
    print("TTS 로딩 중... (첫 실행은 모델 다운로드로 느릴 수 있음)")
    tts.load()
    print(f"준비 완료! (voice={tts.voice}, sr={tts.sample_rate}) 문장을 입력하세요. (빈 줄=종료)\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        tts.speak(text)
    print("종료합니다.")


if __name__ == "__main__":
    _repl()
