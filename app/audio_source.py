"""단일 마이크 스트림 소유자 — 호출어 감지기와 STT가 같은 스트림을 나눠 쓴다.

왜 하나로 묶는가:
  1) 젯슨 ReSpeaker 는 하드웨어 장치가 하나뿐이라 InputStream 을 둘 동시에 못 연다.
  2) 감지 후 스트림을 닫고 STT 가 새로 열면 ALSA 재오픈 + 소음바닥 재측정(0.3s) 로
     '귀 먹은 구간' 이 생겨 아이 말 첫머리가 잘린다. 상용 스피커(Alexa)도 스트림을
     닫지 않고 프리롤 버퍼를 둔다. [[jaeha-bot-progress]]

마이크 접점은 _read_frame()·_available() 둘뿐이라, 테스트는 이것만 오버라이드하면 된다.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

log = logging.getLogger("jaeha_bot.audio")

SAMPLE_RATE = 16000  # faster-whisper·openWakeWord 공통
FRAME = 1280         # 80ms @16kHz — openWakeWord 규격(멜 한 묶음 = 임베딩 보폭 1칸)


class AudioSource:
    """마이크 스트림 하나를 열어두고 80ms 프레임을 공급한다.

    preroll: 최근 이 시간(초)만큼의 프레임을 링버퍼에 보관한다. 호출어가 걸린 순간
             직전 음성을 STT 로 넘겨 '하이 티드 이거 뭐야?' 의 뒷말이 안 잘리게 한다.
    verify_window: 2단계 검증(whisper 재확인)에 넘길 최근 오디오 길이(초).

    ⚠️ **왜 링버퍼를 두 개 두는가** — 용도가 다르기 때문이다.
      preroll 0.5초는 Alexa 와 같은 값이고, 늘리면 호출 직전 TV·부모 말소리가 섞여
      whisper 가 환각한다(그래서 0.5 로 정했다). 반면 2단계 검증은 호출어 **전체**
      (약 1초)를 봐야 하므로 2.0초가 필요하다. 하나를 늘려 둘 다 쓰면 한쪽이 반드시
      망가진다. [[jaeha-bot-progress]]
    """

    def __init__(self, samplerate: int = SAMPLE_RATE, frame: int = FRAME,
                 preroll: float = 0.5, verify_window: float = 2.0) -> None:
        self.samplerate = samplerate
        self.frame = frame
        self.preroll_frames = max(1, int(preroll * samplerate / frame))
        self.verify_frames = max(1, int(verify_window * samplerate / frame))
        self._ring: deque[np.ndarray] = deque(maxlen=self.preroll_frames)
        self._verify_ring: deque[np.ndarray] = deque(maxlen=self.verify_frames)
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
        """마이크와 닿는 지점 1 — 프레임 하나를 읽는다. 테스트는 여기를 오버라이드한다."""
        block, _ = self._stream.read(self.frame)
        return np.asarray(block, dtype=np.float32).reshape(-1)

    def _available(self) -> int:
        """마이크와 닿는 지점 2 — 지금 당장 읽을 수 있는 샘플 수(스트림 없으면 0).

        drain() 이 '버퍼가 빌 때까지' 를 판단하는 유일한 근거다. 이걸 심으로 빼둬야
        drain() 도 _read_frame() 만 쓰게 되어(=마이크 직접 호출 없음) 테스트가 된다.
        """
        if self._stream is None:
            return 0
        return int(getattr(self._stream, "read_available", 0))

    def read(self) -> np.ndarray:
        """프레임 하나를 읽고 두 링버퍼(프리롤·검증창)에도 넣는다."""
        f = self._read_frame()
        self._ring.append(f)
        self._verify_ring.append(f)
        return f

    # --------------------------------------------------------------- 프리롤
    def preroll(self) -> np.ndarray:
        """링버퍼에 남은 최근 오디오를 이어붙여 돌려준다(비었으면 빈 배열)."""
        if not self._ring:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._ring)).astype(np.float32)

    def verify_window(self) -> np.ndarray:
        """2단계 검증용 — 최근 verify_window 초를 이어붙여 돌려준다.

        호출어가 걸린 '순간'을 끝점으로 하는 구간이라 호출어 전체가 들어온다.
        """
        if not self._verify_ring:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._verify_ring)).astype(np.float32)

    def clear_preroll(self) -> None:
        """두 버퍼를 함께 비운다 — 낡은 오디오를 검증에 쓰면 안 된다."""
        self._ring.clear()
        self._verify_ring.clear()

    # ----------------------------------------------------------------- 비우기
    def drain(self) -> int:
        """지금까지 쌓인 입력 버퍼를 버린다. 버린 샘플 수를 돌려준다.

        공유 스트림이라 봇이 말하거나(TTS) 생각하는(LLM) 동안에도 마이크는 계속
        녹음돼 버퍼에 쌓인다. 그대로 두면 다음 읽기에서 **봇 자기 목소리**나 몇 초 전
        잡음을 먼저 읽어 오인식한다(자체 스트림 방식일 땐 매번 새로 열어 없던 문제).
        그래서 말한 직후 에코 쿨다운이 끝나면 이걸 불러 버퍼를 비운다.
        프리롤도 같이 비운다 — 낡은 오디오를 호출어 프리롤로 쓰면 안 되기 때문.
        """
        dropped = 0
        # 한 프레임도 못 채울 만큼 남을 때까지 읽어서 버린다(최대 79ms 는 남을 수 있다).
        while self._available() >= self.frame:
            self._read_frame()
            dropped += self.frame
        self._ring.clear()
        self._verify_ring.clear()
        if dropped:
            log.debug("입력 버퍼 %d샘플(%.2fs) 버림", dropped, dropped / self.samplerate)
        return dropped


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))
