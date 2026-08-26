"""필러(맞장구): 아이 말이 끝나자마자 낼 짧은 소리를 미리 만들어 둔다.

🔴 이건 실지연을 줄이지 않는다. **진짜 답이 나오는 시각은 1ms 도 안 바뀐다.**

    지금        아이 말 끝 ├──────── 4.17초 침묵 ────────┤ "그건 말이야…"
    필러 넣으면  아이 말 끝 ├─1.70s─┤"음~"├── 2.47초 ──┤ "그건 말이야…"

   바뀌는 건 아이가 **처음 소리를 듣는 시각**뿐이다(4.17 -> 1.70초).
   즉 파이프라인 최적화가 아니라 체감 설계다 — 실지연 레버와 안 겹치므로 같이 쓴다.

🔴 가장 큰 값어치는 **재발화 차단**이다. 침묵이 길면 아이는 "티드?" 하고 다시 부르는데,
   그 시점엔 `listen()` 이 이미 끝나 봇이 안 듣고 있고 `source.drain()` 이 쓸어낸다 —
   **아이가 다시 말한 것이 통째로 사라진다.** 빠른 맞장구가 그걸 막는다.

⚠️ `configs/audio_assets.yaml` 에 넣으면 안 된다. 그 파일의 playable()/titles() 는
   LLM 툴 스키마의 enum 으로 그대로 쓰여서, 필러를 등록하면 봇이 "음~ 틀어줄게" 라고
   말할 수 있게 된다 — 현재활동 날조를 구조로 막아 둔 장치가 깨진다.

설계: docs/superpowers/specs/2026-08-25-filler-design.md
"""
from __future__ import annotations

import hashlib
import logging
import random
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("jaeha_bot.filler")

# 앞에 구워 넣는 무음(초). tts_module.PLAY_PAD_S 와 같은 이유·같은 값이다.
# 🔴 SoundDeviceSink 는 TTSModule 과 달리 재생 때 패딩을 안 붙인다. sounddevice
#    스트림 시작이 앞 샘플을 흘리므로, 파일에 구워 넣지 않으면 첫 음절이 잘린다.
DEFAULT_PAD_S = 0.15

# 필러의 목표 음량(말소리 구간 RMS). 답변 TTS 실측 중앙값이다
# (2026-08-26, F2/speed 1.05, 문장 5개: 0.0538~0.0681, 중앙 0.0625).
# 🔴 문구마다 그냥 두면 RMS 가 1.9배(피크로는 2.7배)까지 벌어진다 — 어떤 맞장구는
#    안 들리고 어떤 건 놀랜다. 답변과 같은 크기로 들려야 한 사람 목소리로 들린다.
TARGET_RMS = 0.0625
# 조용한 문구를 끌어올리다 찢어지지 않게 하는 천장.
PEAK_CEILING = 0.95
# 무음 판정(≈-60dB). tts_module._trim 의 thr 과 같은 발상.
_SILENCE_THR = 1e-3


class FillerBank:
    """미리 합성해 캐시해 둔 맞장구 모음. 고르고, 튼다.

    실패는 전부 삼킨다 — 필러는 기능이 아니라 최적화라서, 여기서 난 예외가
    대화를 죽이면 손해가 이득보다 크다.
    """

    def __init__(self, phrases, cache_dir, *, sink=None,
                 pad_s: float = DEFAULT_PAD_S, delay_s: float = 0.0,
                 target_rms: float = TARGET_RMS, enabled: bool = True) -> None:
        self.phrases = [p for p in (phrases or []) if p and p.strip()]
        self.cache_dir = Path(cache_dir)
        self.sink = sink
        self.pad_s = float(pad_s)
        self.delay_s = float(delay_s)
        self.target_rms = float(target_rms)
        self.enabled = bool(enabled)
        self._audio: list[tuple[np.ndarray, int]] = []
        self._last = -1
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 캐시
    def cache_key(self, tts) -> str:
        """목소리·speed·확산스텝·문구·음량기준이 하나라도 다르면 다른 키가 된다.

        🔴 목소리를 갈아탔을 때 필러만 옛 목소리로 겉도는 사고를 구조로 막는다.
        """
        raw = "|".join([
            str(getattr(tts, "voice", "")),
            str(getattr(tts, "speed", "")),
            str(getattr(tts, "total_steps", "")),
            str(self.pad_s),
            str(self.target_rms),
            *self.phrases,
        ])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def ensure(self, tts) -> int:
        """캐시를 채운다. 새로 합성한 개수를 돌려준다. 기동에서 한 번 부른다."""
        if not self.enabled or not self.phrases:
            return 0
        import soundfile as sf

        key = self.cache_key(tts)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("필러 캐시 디렉터리를 못 만든다(필러 없이 계속): %s", e)
            return 0

        made = 0
        self._audio = []
        for i, phrase in enumerate(self.phrases):
            path = self.cache_dir / f"{key}_{i}.wav"
            try:
                if path.is_file():
                    audio, rate = sf.read(str(path), dtype="float32")
                else:
                    audio, rate = self._render(tts, phrase)
                    sf.write(str(path), audio, rate)
                    made += 1
                self._audio.append((np.asarray(audio, dtype=np.float32).reshape(-1), int(rate)))
            except Exception as e:   # 한 문장이 실패해도 나머지는 살린다
                log.warning("필러 준비 실패(건너뜀) [%s]: %s: %s",
                            phrase, type(e).__name__, str(e)[:120])
        if made:
            log.info("필러 %d개 새로 합성(캐시 %s)", made, self.cache_dir)
        return made

    def _render(self, tts, phrase) -> tuple[np.ndarray, int]:
        audio = self._normalize(np.asarray(tts.render(phrase), dtype=np.float32).reshape(-1))
        rate = int(getattr(tts, "sample_rate", 44100))
        # 🔴 앞뒤 **둘 다** 덧댄다. 뒤가 없으면 장치가 끝 샘플을 흘려 말끝이 '뚝' 끊긴다
        #    (TTSModule.speak 가 PLAY_PAD_S 를 앞뒤로 붙이는 것과 같은 이유).
        pad = np.zeros(int(self.pad_s * rate), dtype=np.float32)
        return np.concatenate([pad, audio, pad]), rate

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """말소리 구간 RMS 를 target_rms 에 맞춘다. 피크는 PEAK_CEILING 을 안 넘는다.

        ⚠️ 페이드는 하지 않는다 — tts_module._trim 이 '페이드아웃 금지'로 못 박아 둔
           그 이유(예전 뚝 끊김의 원인)가 여기도 그대로 적용된다. 크기만 건드린다.
        """
        if audio.size == 0:
            return audio
        nz = np.where(np.abs(audio) > _SILENCE_THR)[0]
        speech = audio[nz[0]:nz[-1] + 1] if nz.size else audio
        rms = float(np.sqrt(np.mean(speech ** 2)))
        if rms <= 0.0:                       # 통무음 — 0 으로 나누면 NaN 이 스피커로 간다
            return audio
        gain = self.target_rms / rms
        peak = float(np.max(np.abs(audio)))
        if peak * gain > PEAK_CEILING:       # 찢어지느니 조금 작은 편이 낫다
            gain = PEAK_CEILING / peak
        return (audio * gain).astype(np.float32)

    @property
    def size(self) -> int:
        return len(self._audio)

    # ----------------------------------------------------------------- 고르기
    def _pick_index(self) -> int:
        """직전과 다른 것을 고른다. 같은 소리가 연달아 나오면 금방 질린다."""
        n = len(self._audio)
        if n == 0:
            return -1
        if n == 1:
            self._last = 0
            return 0
        choices = [i for i in range(n) if i != self._last]
        self._last = random.choice(choices)
        return self._last

    def next(self):
        """다음에 낼 (오디오, 레이트). 준비된 게 없으면 None."""
        i = self._pick_index()
        return self._audio[i] if i >= 0 else None

    # ------------------------------------------------------------------ 재생
    def play(self) -> bool:
        """하나 내보내라고 **맡긴다**. 맡겼으면 True(=소리가 났다는 뜻은 아니다).

        🔴 반드시 딴 스레드로 넘긴다. `sd.play()` 는 block=False 여도 반환까지
           318~480ms 를 먹는다(2026-08-26 실측, MME 스트림 여는 비용) — 여기서
           기다리면 필러가 줄이려던 침묵을 **진짜 지연으로 바꿔** 되돌려준다.
        """
        if not self.enabled or self.sink is None or not self._audio:
            return False
        picked = self.next()
        if picked is None:
            return False
        if self.delay_s > 0:
            t = threading.Timer(self.delay_s, self._emit, args=picked)
        else:
            t = threading.Thread(target=self._emit, args=picked)
        t.daemon = True
        self._thread = t
        t.start()
        return True

    def wait(self, timeout: float | None = None) -> None:
        """맡긴 재생이 장치로 넘어갈 때까지 기다린다. 테스트·종료용."""
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _emit(self, audio, rate) -> bool:
        try:
            self.sink.play(audio, rate, False)
            return True
        except Exception as e:      # 장치가 없거나 물려 있어도 대화는 계속돼야 한다
            log.warning("필러 재생 실패(무시): %s: %s", type(e).__name__, str(e)[:120])
            return False
