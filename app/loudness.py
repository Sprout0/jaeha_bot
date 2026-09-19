"""유튜브 영상마다 다른 음량을 맞춘다 — 나가는 소리를 재서 플레이어 볼륨을 고친다. (2026-09-19)

🔴 왜: 영상마다 올린 사람의 녹음 크기가 다르다. 젯슨에서 아이 노래 8곡을 같은 볼륨(80)으로
   무음 측정(null sink)했더니 크게 나오는 구간(p90)이 **-17 ~ -31 dBFS, 14dB 차이**였다
   (콩순이·곰세마리 -17, 티니핑 -27, 아기상어 -31).

구조(젯슨 실측):
  - 재는 곳: PulseAudio 싱크의 모니터(parec). 모니터는 **볼륨이 적용된 뒤** 소리다
    (음소거하면 0) → 재고 고치는 되먹임으로 돈다.
  - 고치는 곳: 플레이어 setVolume. 소리 크기에 거의 정비례한다(80→40 = -6.2dB).
  - 키울 여유: 플레이어가 100 을 못 넘으니 싱크를 160%(+11.7dB)로 두고 기본을 20 으로
    낮춘다. '20 × 160%' ≈ 옛 '80 × 100%'(실측 -19.0 vs -18.8). 이제 5~100 이 -12~+14dB.

판단은 크게 나오는 구간(p90)으로 한다. 조용한 인트로·말하는 부분에 끌려 볼륨을 올리면
본 노래에서 터진다.
"""
from __future__ import annotations

from collections import deque

import numpy as np

SILENCE_DB = -50.0      # 이보다 작은 블록은 무음(일시정지·곡 사이) — 판단에서 뺀다
PEAK_DB = -1.0          # 봉우리가 이걸 넘으면 깨진다 — RMS 와 상관없이 줄인다


def block_db(block: np.ndarray) -> tuple[float, float]:
    """(RMS dBFS, 봉우리 dBFS)."""
    a = np.asarray(block, dtype=np.float32)
    if a.size == 0:
        return -120.0, -120.0
    rms = float(np.sqrt(np.mean(a * a)))
    peak = float(np.max(np.abs(a)))
    return 20 * np.log10(rms + 1e-9), 20 * np.log10(peak + 1e-9)


class LoudnessLeveler:
    def __init__(self, *, target_db: float = -19.0, base: int = 20, lo: int = 5, hi: int = 100,
                 tol_db: float = 1.5, max_step_db: float = 6.0, first_step_db: float = 12.0,
                 up_step_db: float = 3.0, window_s: float = 6.0, settle_s: float = 3.0, every_s: float = 2.0,
                 min_blocks: int = 4) -> None:
        self.target_db, self.base, self.lo, self.hi = float(target_db), int(base), int(lo), int(hi)
        self.tol_db, self.max_step_db, self.first_step_db = tol_db, max_step_db, first_step_db
        # 🔴 09-19 젯슨: 아기상어 조용한 구간(-36.5)에 끌려 44→88 로 한 번에 올랐다. 첫 조정
        #    뒤에는 **올릴 땐 천천히, 내릴 땐 빠르게** — 조용한 구간 뒤 본 노래가 터지지 않게.
        self.up_step_db = up_step_db
        self.window_s, self.settle_s, self.every_s = window_s, settle_s, every_s
        self.min_blocks = min_blocks
        self.volume = self.base
        self.last_level: float | None = None     # 마지막 판단의 p90 (로그용)
        self._blocks: deque = deque()
        self._since = 0.0
        self._last = None
        self._adjusted = False

    def reset(self, now: float) -> None:
        """새 곡. 기본 볼륨으로 돌아가 처음부터 잰다."""
        self.volume = self.base
        self._blocks.clear()
        self._since, self._last, self._adjusted = now, None, False

    def feed(self, block: np.ndarray, now: float) -> int | None:
        """재생 중인 소리 한 조각. 볼륨을 바꿔야 하면 새 값, 아니면 None."""
        rms, peak = block_db(block)
        if rms >= SILENCE_DB:
            self._blocks.append((now, rms, peak))
        while self._blocks and now - self._blocks[0][0] > self.window_s:
            self._blocks.popleft()
        if now - self._since < self.settle_s or len(self._blocks) < self.min_blocks:
            return None
        if self._last is not None and now - self._last < self.every_s:
            return None

        level = float(np.percentile([b[1] for b in self._blocks], 90))
        self.last_level = level
        err = self.target_db - level
        if max(b[2] for b in self._blocks) > PEAK_DB:
            err = min(err, -3.0)
        elif abs(err) < self.tol_db:
            return None
        if not self._adjusted:
            err = max(-self.first_step_db, min(self.first_step_db, err))
        else:
            err = max(-self.max_step_db, min(self.up_step_db, err))
        new = int(round(self.volume * 10 ** (err / 20)))
        new = max(self.lo, min(self.hi, new))
        self._last = now
        if new == self.volume:
            return None
        self._adjusted = True
        self.volume = new
        # 이미 잰 블록은 옛 볼륨 소리다 — 버리고 새 볼륨으로 다시 잰다
        self._blocks.clear()
        return new
