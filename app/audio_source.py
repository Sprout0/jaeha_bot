"""단일 마이크 스트림 소유자 — 호출어 감지기와 STT가 같은 스트림을 나눠 쓴다.

왜 하나로 묶는가:
  1) 젯슨 ReSpeaker 는 하드웨어 장치가 하나뿐이라 InputStream 을 둘 동시에 못 연다.
  2) 감지 후 스트림을 닫고 STT 가 새로 열면 ALSA 재오픈 + 소음바닥 재측정(0.3s) 로
     '귀 먹은 구간' 이 생겨 아이 말 첫머리가 잘린다. 상용 스피커(Alexa)도 스트림을
     닫지 않고 프리롤 버퍼를 둔다. [[jaeha-bot-progress]]

마이크 접점은 _read_frame() 하나뿐이라, 테스트는 이것만 오버라이드하면 된다.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

log = logging.getLogger("jaeha_bot.audio")

SAMPLE_RATE = 16000  # faster-whisper·openWakeWord 공통
FRAME = 1280         # 80ms @16kHz — openWakeWord 규격(멜 8프레임 = 보폭 1칸)


class AudioSource:
    """마이크 스트림 하나를 열어두고 80ms 프레임을 공급한다.

    preroll: 최근 이 시간(초)만큼의 프레임을 링버퍼에 보관한다. 호출어가 걸린 순간
             직전 음성을 STT 로 넘겨 '재하봇 이거 뭐야?' 의 뒷말이 안 잘리게 한다.
    """

    def __init__(self, samplerate: int = SAMPLE_RATE, frame: int = FRAME,
                 preroll: float = 0.5) -> None:
        self.samplerate = samplerate
        self.frame = frame
        self.preroll_frames = max(1, int(preroll * samplerate / frame))
        self._ring: deque[np.ndarray] = deque(maxlen=self.preroll_frames)
        self._stream = None
        self.noise_floor = 0.0

    # ------------------------------------------------------------- 스트림 수명
    def open(self, measure_noise: bool = True) -> "AudioSource":
        """마이크를 열고 주변 소음 바닥을 한 번만 잰다(STT 가 재측정하지 않게)."""
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1,
            dtype="float32", blocksize=self.frame,
        )
        self._stream.start()
        if measure_noise:
            vals = []
            for _ in range(10):  # 10프레임 = 0.8초
                vals.append(_rms(self._read_frame()))
            self.noise_floor = float(np.mean(vals)) if vals else 0.0
            log.info("소음 바닥 측정: %.5f", self.noise_floor)
        return self

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- 프레임 공급
    def _read_frame(self) -> np.ndarray:
        """마이크와 닿는 유일한 지점. 테스트는 여기만 오버라이드한다."""
        block, _ = self._stream.read(self.frame)
        return np.asarray(block, dtype=np.float32).reshape(-1)

    def read(self) -> np.ndarray:
        """프레임 하나를 읽고 프리롤 링버퍼에도 넣는다."""
        f = self._read_frame()
        self._ring.append(f)
        return f

    # --------------------------------------------------------------- 프리롤
    def preroll(self) -> np.ndarray:
        """링버퍼에 남은 최근 오디오를 이어붙여 돌려준다(비었으면 빈 배열)."""
        if not self._ring:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._ring)).astype(np.float32)

    def clear_preroll(self) -> None:
        self._ring.clear()


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))
