"""고정 문구 → marin 목소리 wav. 기동 때 빠진 것만 한 번 만든다.

키에 목소리·모델을 넣는다 — 목소리를 갈아탔을 때 고정 문구만 옛 목소리로 겉도는
사고를 구조로 막는다(filler.FillerBank.cache_key 와 같은 원칙).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .realtime_protocol import SR

log = logging.getLogger("jaeha_bot.voice_cache")


class VoiceCache:
    def __init__(self, cache_dir, voice: str, model: str) -> None:
        self.dir = Path(cache_dir).expanduser()
        self.voice, self.model = voice, model
        self._mem: dict[str, np.ndarray] = {}

    def key(self, phrase: str) -> str:
        return hashlib.sha1(f"{self.voice}|{self.model}|{phrase}".encode("utf-8")).hexdigest()[:16]

    def path(self, phrase: str) -> Path:
        return self.dir / f"{self.key(phrase)}.wav"

    def ensure(self, phrases: Iterable[str], synth: Callable[[str], np.ndarray]) -> int:
        import soundfile as sf

        self.dir.mkdir(parents=True, exist_ok=True)
        made = 0
        for p in phrases:
            if not p or self.path(p).exists():
                continue
            try:
                audio = np.asarray(synth(p), dtype=np.float32).reshape(-1)
                sf.write(self.path(p), audio, SR)
                made += 1
                log.info("고정 문구 만듦(%.1fs): %s", audio.size / SR, p)
            except Exception as e:
                log.warning("고정 문구를 못 만들었다(건너뜀) %s: %s: %s", p, type(e).__name__, e)
        return made

    def get(self, phrase: str) -> np.ndarray | None:
        if phrase in self._mem:
            return self._mem[phrase]
        path = self.path(phrase)
        if not path.exists():
            return None
        import soundfile as sf
        audio, rate = sf.read(path, dtype="float32")
        if rate != SR:
            from .audio_player import _resample
            audio = _resample(audio, rate, SR)
        self._mem[phrase] = audio
        return audio
