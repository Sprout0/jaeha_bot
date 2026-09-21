"""Realtime 대화의 소리 쪽 — 마이크 게이트, pcm 변환, 끊김 없는 재생.

tools/realtime_live.py(09-07 라이브 루프, 노트북 에코 0/11턴)에서 옮겼다. 봇과 도구가
같은 코드를 쓴다 — 도구에서 검증한 것이 봇에서 다르게 돌면 안 된다.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio_player import _resample

SR = 24000             # Realtime pcm16 규격
REMAINDER = 2          # pcm16 한 샘플 = 2 바이트
FULL_SCALE = 32768.0


class MicGate:
    """반이중. 봇이 말하는 동안(+뒤로 pad 만큼) 마이크를 서버에 보내지 않는다.

    서버 VAD 는 우리가 보낸 것만 듣는다. 재생 중에 열어 두면 봇이 제 말에 반응한다.
    pad 는 스피커 잔향과 장치 버퍼 몫 — 08-24 실측으로 정한 0.150s 를 쓴다.
    """

    def __init__(self, pad_s: float = 0.15) -> None:
        self.pad_s = pad_s
        self._playing = False
        self._ended_at: float | None = None

    def playing_started(self, at: float) -> None:
        self._playing = True
        self._ended_at = None

    def playing_ended(self, at: float) -> None:
        self._playing = False
        self._ended_at = at

    def sync(self, playing: bool, now: float) -> None:
        """스피커 상태를 그대로 넘긴다. **가장자리에서만** 시계를 잡는다 —
        매번 잡으면 pad 가 영원히 안 지나 마이크가 다시 안 열린다."""
        if playing:
            self.playing_started(now)
        elif self._playing:
            self.playing_ended(now)

    def should_send(self, now: float) -> bool:
        if self._playing:
            return False
        if self._ended_at is None:
            return True
        return now - self._ended_at >= self.pad_s


class PcmAccumulator:
    """pcm16 리틀엔디언 바이트를 float32(-1..1) 로. 델타는 샘플 중간에서 잘려 온다."""

    def __init__(self) -> None:
        self._tail = b""

    def feed(self, chunk: bytes) -> np.ndarray:
        buf = self._tail + chunk
        n = len(buf) - len(buf) % REMAINDER
        self._tail = buf[n:]
        if not n:
            return np.zeros(0, dtype=np.float32)
        return (np.frombuffer(buf[:n], dtype="<i2").astype(np.float32) / FULL_SCALE)


def to_pcm16(block: np.ndarray, src_rate: int) -> bytes:
    """마이크 float32 → Realtime 이 받는 24k pcm16 바이트. 젯슨 ReSpeaker 는 16k 전용이다."""
    block = np.asarray(block, dtype=np.float32).reshape(-1)
    if src_rate != SR:
        block = _resample(block, src_rate, SR)
    return (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()


def resolve_rate(kind: str, want: int) -> int:
    """장치가 want 를 받으면 그대로, 아니면 장치 기본값.

    젯슨 ReSpeaker 는 16000 전용이고 Realtime 은 24000 규격이라 양방향 리샘플이
    필요하다. 노트북은 보통 24000 을 그대로 받아 리샘플이 0 회다 — 그래서 이 값을
    반드시 찍어 본다. 노트북에서 안 겪은 문제를 젯슨에서 처음 만나면 안 된다.
    """
    import sounddevice as sd

    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    try:
        check(samplerate=want)
        return want
    except Exception:
        dev = sd.default.device
        idx = dev[0 if kind == "input" else 1] if isinstance(dev, (list, tuple)) else dev
        try:
            return int(sd.query_devices(idx, kind)["default_samplerate"])
        except Exception:
            return want


class StreamSpeaker:
    """소켓에서 오는 조각을 끊김 없이 낸다.

    `SoundDeviceSink.play()` 는 호출마다 앞 재생을 끊어서 못 쓴다(조각마다 부르면
    마지막 조각만 들린다). 콜백 스트림 + 대기열로 이어 붙인다. 맞장구·고정 문구도
    같은 대기열에 넣으면 답과 겹치지 않고 이어진다.
    """

    def __init__(self, src_rate: int = SR) -> None:
        import sounddevice as sd

        self.src_rate = src_rate
        self.rate = resolve_rate("output", src_rate)
        self.resampled = self.rate != src_rate
        self._q: deque = deque()
        self._left = np.zeros(0, dtype=np.float32)
        # 2026-09-22 놀이 바로잡기: 이 턴 소리를 세고 뒤를 버린다(재생 콜백은 다른 스레드).
        import threading
        self._lock = threading.Lock()
        self._marked = 0          # mark 이후 밀어 넣은 **장치** 샘플
        self._played = 0          # mark 이후 재생한 장치 샘플(앞 턴 잔여 포함)
        self._pre = 0             # mark 순간 대기열에 남아 있던 앞 소리(맞장구 등)
        self._stream = sd.OutputStream(samplerate=self.rate, channels=1,
                                       dtype="float32", callback=self._cb)
        self._stream.start()

    def _cb(self, out, frames, time_info, status) -> None:
        need, got = frames, []
        with self._lock:
            while need > 0:
                if self._left.size == 0:
                    if not self._q:
                        break
                    self._left = self._q.popleft()
                take = min(need, self._left.size)
                got.append(self._left[:take])
                self._left = self._left[take:]
                need -= take
            self._played += frames - need
        block = np.concatenate(got) if got else np.zeros(0, dtype=np.float32)
        if block.size < frames:                       # 남으면 무음으로 채운다
            block = np.concatenate([block, np.zeros(frames - block.size, dtype=np.float32)])
        out[:, 0] = block

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.resampled:
            samples = _resample(samples, self.src_rate, self.rate)
        with self._lock:
            self._q.append(samples)
            self._marked += samples.size

    @property
    def busy(self) -> bool:
        return bool(self._q) or self._left.size > 0

    def clear(self) -> None:
        with self._lock:
            self._q.clear()
            self._left = np.zeros(0, dtype=np.float32)

    def _to_dev(self, n: int) -> int:
        return int(round(n * self.rate / self.src_rate))

    def mark(self) -> None:
        """이 턴 소리의 시작점 — 이후 played()/truncate() 의 기준(2026-09-22 놀이 바로잡기)."""
        with self._lock:
            self._pre = self._left.size + sum(q.size for q in self._q)
            self._marked = self._played = 0

    def _turn_played(self) -> int:
        return max(0, self._played - self._pre)

    def played(self) -> int:
        """mark 이후 **이 턴 소리**를 실제로 재생한 양(원본 레이트 샘플). 앞 잔여는 빼고 센다."""
        with self._lock:
            return int(round(self._turn_played() * self.src_rate / self.rate))

    def truncate(self, keep: int) -> None:
        """mark 이후 밀어 넣은 소리 중 keep(원본 샘플) 뒤를 버린다. 이미 나간 건 못 되돌린다."""
        with self._lock:
            floor = max(self._to_dev(keep), self._turn_played())
            drop = self._marked - floor
            while drop > 0 and self._q:
                last = self._q[-1]
                if last.size <= drop:
                    drop -= last.size
                    self._q.pop()
                else:
                    self._q[-1] = last[:last.size - drop]
                    drop = 0
            if drop > 0 and self._left.size:
                self._left = self._left[:max(0, self._left.size - drop)]
            self._marked = min(self._marked, floor)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
