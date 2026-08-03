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
from pathlib import Path

import numpy as np

log = logging.getLogger("jaeha_bot.tts")

# 문장 경계(마침표/물음표/느낌표 등). 문장 스트리밍용 분리에 사용.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？…])\s+|\n+")


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

    # ---------------------------------------------------- 보이스 스타일 해석(+블렌딩)
    def _resolve_style(self, voice: str):
        """voice 문자열 -> Style. 'F1+F4' 처럼 '+' 로 이으면 프리셋들을 균등 블렌딩.

        Supertonic 의 voice style 은 numpy 벡터 2개(ttl=음색, dp=리듬)일 뿐이라
        선형 평균으로 새 화자를 만들 수 있다(style latent 는 연속·보간 가능 공간).
        F1+F4 블렌드가 유아용으로 가장 밝고 자연스러워 채택(청취 비교, speed 1.05).
        단일 이름이면 그대로 프리셋 로드. [[jaeha-bot-progress]] 참고.
        """
        from supertonic.core import Style

        names = [n.strip() for n in str(voice).split("+") if n.strip()]
        if len(names) <= 1:
            return self._tts.get_voice_style(names[0] if names else voice)
        styles = [self._tts.get_voice_style(n) for n in names]
        w = 1.0 / len(styles)
        ttl = sum(w * s.ttl for s in styles).astype(np.float32)
        dp = sum(w * s.dp for s in styles).astype(np.float32)
        return Style(ttl, dp)

    # ---------------------------------------------------------- 합성(-> numpy)
    def _infer(self, text: str) -> np.ndarray:
        self.load()
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        # Supertonic 은 초기 노이즈를 np.random.randn(전역 RNG)으로 뽑는다. 합성 직전
        # 시드를 고정하면 문장마다 같은 노이즈에서 출발해 말투·속도가 일정해진다.
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
    def speak(self, text: str) -> None:
        """통짜 합성 재생: 답변 전체를 한 번에 합성한다.

        문장을 쪼개 따로 합성(스트리밍)하면 문장마다 억양이 독립적이라 전환이 뚝뚝 끊겨
        '매끄럽지 못'하게 들린다. 전체를 한 번에 넘기면 Supertonic 이 문장 간 억양을
        자연스럽게 이어 준다(실측: 통짜가 쪼개기보다 합성도 더 빠름 2.34s vs 4.23s).
        앞뒤 flat 무음만 잘라(말끝 여운은 보존) 시작 지연과 끝 공백을 줄인다.
        (참고: 예전 문장 쪼개기+gap 스트리밍 방식은 git/메모리 기록 참조. 되돌리려면 그 버전으로.)
        """
        import sounddevice as sd

        self.load()
        audio = self._trim(self._infer(text))
        if audio.size:
            # 실시간 재생(sounddevice, 기본 MME)은 스트림 시작·종료 지연이 커서
            # 버퍼 앞뒤 샘플을 흘린다 → 앞 발음 잘림 + 끝 '뚝' 끊김.
            # 앞뒤에 짧은 무음을 덧대 그 지연을 무음으로 흡수한다(합성 오디오는 손대지 않음).
            # 오디션 wav(파일 재생)에는 이 현상이 없어 앱만 빠르고 잘려 들렸던 원인.
            audio = np.concatenate([self._silence(0.15), audio, self._silence(0.15)])
            # 장치가 합성 레이트를 지원하는지 미리 확인해 처음부터 맞는 레이트로 재생한다
            # (실패-후-재시도가 아님 → paInvalidSampleRate 로그 안 뜸).
            # PC 는 대개 그대로, Jetson USB(16000 전용)는 리샘플해서 재생.
            rate = self._resolve_play_rate(sd)
            play_audio = audio if rate == self.sample_rate else self._resample(audio, self.sample_rate, rate)
            sd.play(play_audio, rate)
            sd.wait()


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
