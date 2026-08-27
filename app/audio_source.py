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
        # 마이크 버퍼 넘침 — 넘친 만큼 **소리가 버려졌다**. 아래 _note_overflow 참고.
        self.overflows = 0
        self._overflow_since_log = 0
        self._overflow_logged_at = 0.0
        # drain() 안에서 난 넘침은 따로 센다 — 거기선 **일부러 버리는** 중이라 손실이 아니다.
        self.drained_overflows = 0
        self._draining = False

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
        # 세션 총계를 한 줄 남긴다 — 간격 제한 때문에 중간 경고가 몇 줄 안 나올 수 있어,
        # 나중에 로그만 보고 '이 세션에서 소리가 얼마나 버려졌나'를 알 수 있어야 한다.
        if self.overflows:
            log.warning("🔴 이 세션 마이크 버퍼 넘침 총 %d회 — 놓친 호출이 있을 수 있다",
                        self.overflows)

    def __enter__(self) -> "AudioSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- 프레임 공급
    def _read_frame(self) -> np.ndarray:
        """마이크와 닿는 지점 1 — 프레임 하나를 읽는다. 테스트는 여기를 오버라이드한다."""
        block, overflowed = self._stream.read(self.frame)
        if overflowed:
            self.note_overflow()
        return np.asarray(block, dtype=np.float32).reshape(-1)

    # 넘침 로그 최소 간격(초). 넘치는 상황에선 프레임마다 넘치므로 그대로 찍으면
    # 초당 12줄이 쌓여 정작 봐야 할 [검증]·[호출] 줄이 묻힌다.
    OVERFLOW_LOG_S = 10.0

    def note_overflow(self) -> None:
        """마이크 버퍼가 넘쳤다 = **그 사이 들어온 소리가 버려졌다.**

        🔴 왜 이걸 세는가 (2026-08-26 추가):
          2단계 검증(whisper)은 한 번에 1.2초쯤 걸리고, 그동안 이 읽기 루프는 멈춰
          있다. 유튜브를 켜면 후보가 1.2초마다 떠서 사실상 **계속 검증 중**이 된다
          (logs/jaeha_20260826.log 14:53:50~56 이 그 모습이다 — 6초 동안 6번).
          그때 PortAudio 버퍼가 넘치면 진짜 호출이 통째로 사라지는데, 지금까지
          `block, _ = read()` 로 **플래그를 버리고 있어 넘쳤는지조차 알 수 없었다.**

        이 프로젝트는 조용한 실패에 반복해서 당했다 — 젯슨 기본 마이크가 에러 없이
        0.0 만 주던 것, 증강 검사가 빈 목록을 받고 '통과'를 찍던 것. 넘침도 같은 종류라
        **소리 없이 지나가게 두지 않는다.**

        호출부(main·진단)는 `self.overflows` 로 누적 횟수를 볼 수 있다.
        """
        import time

        # 🔴 drain() 중의 넘침은 경고하지 않는다 (2026-08-26 실기에서 바로 드러났다).
        #   drain() 은 봇이 말하고 생각하는 동안 쌓인 **자기 목소리를 일부러 버리는** 자리다.
        #   턴이 끝날 때마다 버퍼가 넘쳐 있는 게 정상이고, 그때 '호출을 놓칠 수 있다'고
        #   찍으면 거짓말이다. 실기 첫 세션에서 17회 중 **15회가 이것**이었다 —
        #   이대로 두면 로그를 안 믿게 되고, 진짜 2회가 묻힌다.
        if self._draining:
            self.drained_overflows += 1
            return

        self.overflows += 1
        self._overflow_since_log += 1
        now = time.monotonic()
        if now - self._overflow_logged_at < self.OVERFLOW_LOG_S:
            return
        log.warning("🔴 마이크 버퍼 넘침 %d회(누적 %d) — 그만큼 소리가 버려졌다. "
                    "검증이 길어 읽기가 밀렸을 수 있다(호출을 놓칠 수 있음)",
                    self._overflow_since_log, self.overflows)
        self._overflow_since_log = 0
        self._overflow_logged_at = now

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
    def read_buffered(self, max_frames: int) -> list:
        """이미 버퍼에 들어와 있는 프레임만 걷어온다. **블로킹하지 않는다.**

        🔴 왜 필요한가 (2026-08-27): 2단계 검증(whisper)이 도는 1.2~1.5초 동안
           읽기 루프가 멈춰 있고, 아이가 '하이 티드 **이거 뭐야?**' 의 뒷말을 하는 게
           정확히 그 구간이다. 그 오디오는 **이미 마이크 버퍼에 들어와 있는데**
           깨어난 뒤 0.5초만 읽고 나머지를 버리고 있었다. 실기(11:41) 에서 STT 에
           넘어간 건 1.12초뿐이었고 전사 결과가 비어 봇이 아무 말도 안 했다.

        `drain()` 과 같은 것을 읽지만 **버리는 대신 돌려준다**. 이미 들어와 있는
        것이라 읽는 데 드는 시간이 0 이다 — 인사말 경로가 느려지지 않는다.
        """
        out = []
        while len(out) < max_frames and self._available() >= self.frame:
            out.append(self.read())
        return out

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
        # _draining 동안의 버퍼 넘침은 경고하지 않는다 — 여기가 버리는 자리다(note_overflow 참고).
        self._draining = True
        try:
            while self._available() >= self.frame:
                self._read_frame()
                dropped += self.frame
        finally:
            self._draining = False
        self._ring.clear()
        self._verify_ring.clear()
        if dropped:
            log.debug("입력 버퍼 %d샘플(%.2fs) 버림", dropped, dropped / self.samplerate)
        return dropped


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))
