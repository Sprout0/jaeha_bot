# ONNX 호출어 감지기 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 젯슨에서 ONNX 호출어 모델로 "재하봇"을 감지해 대기 상태의 봇을 깨운다.

**Architecture:** 단일 `sd.InputStream`을 `AudioSource`가 소유하고 닫지 않는다. 대기 중에는 `OnnxWakeDetector`가 80ms 프레임을 받아 melspectrogram→embedding→classifier 3단 ONNX 체인으로 점수를 내고, 깨어나면 같은 스트림을 `record_until_silence`가 이어받는다. 감지기는 `wake.detector` 설정 한 줄로 기존 STT 방식과 교체된다.

**Tech Stack:** onnxruntime 1.24 (젯슨 기설치), numpy, sounddevice, pytest(노트북 개발용)

## Global Constraints

- **실행 환경:** 모든 코드는 conda 환경 `jaeha_bot`로 실행한다. 노트북 파이썬 경로는 `C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe`, 젯슨은 `source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot`.
- **젯슨 Python 3.10.20** — `livekit-wakeword` 패키지(≥3.11 요구)를 젯슨에 설치하지 않는다. ONNX 파일만 가져온다.
- **젯슨에 새 런타임 의존성 추가 금지** — onnxruntime 1.24.0과 numpy 2.2.6은 이미 있다. 그 외 추가하지 않는다.
- **오디오 규격 고정:** 샘플레이트 16000, 프레임 1280샘플(80ms), 모노.
- **멜 입력 스케일:** melspectrogram ONNX는 **int16 범위 값(-32768~32767)을 float32 dtype으로** 받는다. 우리 오디오는 `[-1,1]` float32이므로 **반드시 32767을 곱한다.**
- **멜 정규화:** melspectrogram 출력에 **`x/10 + 2`** 를 적용한다.
- **링버퍼 크기:** 멜 76프레임, 임베딩 16개.
- **프리롤:** 0.5초.
- **기존 동작 보존:** `record_until_silence()`를 인자 없이 호출하면 지금과 100% 같게 동작해야 한다(`_repl`, 스크래치패드 검증 스크립트 호환).
- **git 미초기화:** 이 프로젝트는 git 저장소가 아니다. 각 Task의 커밋 스텝은 `git init` 후에만 유효하다. 초기화하지 않을 경우 커밋 스텝은 건너뛴다.
- **젯슨 배포:** `bash push_code.sh` (app·configs·scenarios·data 전송). `tests/`와 `docs/`는 전송되지 않는다.

---

## File Structure

| 파일 | 상태 | 책임 |
|---|---|---|
| `app/audio_source.py` | 신규 | 단일 InputStream 소유, 80ms 프레임 공급, 프리롤 링버퍼, 소음 바닥 1회 측정 |
| `app/wake_onnx.py` | 신규 | `WakeResult`, `OnnxWakeDetector` — 3단 ONNX 체인, 링버퍼 2개, 점수 판정, 인사말 분기 |
| `app/wake.py` | 수정 | 기존 함수 유지 + `SttWakeDetector` 래퍼 + `make_detector()` 팩토리(폴백 포함) |
| `app/stt_module.py` | 수정 | `record_until_silence(source=, prefix=)` / `listen(source=, prefix=)` 인자 추가 |
| `app/main.py` | 수정 | 대기 분기를 `detector.wait_for_wake()`로 교체 |
| `configs/model_paths.yaml` | 수정 | `wake.detector`, `wake.onnx.*` 추가 |
| `models/wake/` | 신규 | `melspectrogram.onnx`, `embedding_model.onnx`, `jaehabot.onnx` |
| `tests/` | 신규 | pytest 테스트 |

---

## Task 1: 테스트 환경과 AudioSource 프리롤 링버퍼

**Files:**
- Create: `app/audio_source.py`
- Create: `tests/__init__.py`
- Create: `tests/test_audio_source.py`

**Interfaces:**
- Consumes: 없음 (첫 Task)
- Produces:
  - `app.audio_source.SAMPLE_RATE = 16000`, `FRAME = 1280`
  - `AudioSource(samplerate=16000, frame=1280, preroll=0.5)`
  - `AudioSource._read_frame() -> np.ndarray` — 하위 클래스가 오버라이드할 수 있는 유일한 마이크 접점
  - `AudioSource.read() -> np.ndarray` shape `(1280,)` float32
  - `AudioSource.preroll() -> np.ndarray` — 최근 0.5초, shape `(N*1280,)`
  - `AudioSource.clear_preroll() -> None`
  - `AudioSource.noise_floor: float`
  - `AudioSource.open() / close()`, 컨텍스트 매니저 지원

- [ ] **Step 1: pytest 설치 (노트북 개발용, 젯슨엔 불필요)**

```bash
"C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pip install pytest
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/__init__.py` 는 빈 파일로 생성한다.

`tests/test_audio_source.py`:

```python
"""AudioSource 링버퍼·프리롤 검증. 마이크 불필요(_read_frame 오버라이드)."""
import numpy as np
import pytest

from app.audio_source import AudioSource, FRAME, SAMPLE_RATE


class FakeSource(AudioSource):
    """프레임을 미리 정해둔 시퀀스로 공급한다. 마이크를 열지 않는다."""

    def __init__(self, frames, **kw):
        super().__init__(**kw)
        self._frames = list(frames)
        self._i = 0
        self.noise_floor = 0.0

    def _read_frame(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        return f


def _ramp(n, start):
    """구분 가능한 프레임 n개. i번째 프레임은 전부 (start+i) 값."""
    return [np.full(FRAME, float(start + i), dtype=np.float32) for i in range(n)]


def test_read_returns_one_frame():
    src = FakeSource(_ramp(3, 0))
    f = src.read()
    assert f.shape == (FRAME,)
    assert f.dtype == np.float32


def test_preroll_holds_configured_duration():
    # preroll 0.5s = 0.5*16000/1280 = 6.25 -> 6 프레임
    src = FakeSource(_ramp(20, 0), preroll=0.5)
    for _ in range(20):
        src.read()
    pre = src.preroll()
    assert pre.size == 6 * FRAME


def test_preroll_keeps_most_recent_frames():
    src = FakeSource(_ramp(20, 0), preroll=0.5)
    for _ in range(20):
        src.read()
    pre = src.preroll().reshape(-1, FRAME)
    # 마지막 6개 프레임은 값 14,15,16,17,18,19
    assert [int(row[0]) for row in pre] == [14, 15, 16, 17, 18, 19]


def test_preroll_shorter_when_not_yet_full():
    src = FakeSource(_ramp(2, 0), preroll=0.5)
    src.read()
    src.read()
    assert src.preroll().size == 2 * FRAME


def test_clear_preroll_empties_buffer():
    src = FakeSource(_ramp(10, 0), preroll=0.5)
    for _ in range(10):
        src.read()
    src.clear_preroll()
    assert src.preroll().size == 0


def test_sample_rate_and_frame_constants():
    assert SAMPLE_RATE == 16000
    assert FRAME == 1280  # openWakeWord 규격: 80ms
```

- [ ] **Step 3: 테스트 실패 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_audio_source.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.audio_source'`

- [ ] **Step 4: 최소 구현 작성**

`app/audio_source.py`:

```python
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
```

- [ ] **Step 5: 테스트 통과 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_audio_source.py -v
```
Expected: PASS (6 passed)

- [ ] **Step 6: 커밋** *(git init 한 경우에만)*

```bash
git add app/audio_source.py tests/__init__.py tests/test_audio_source.py && git commit -m "feat(wake): AudioSource 단일 스트림 + 프리롤 링버퍼"
```

---

## Task 2: 특징추출 ONNX 확보와 체인 형상 검증

**Files:**
- Create: `models/wake/melspectrogram.onnx` (다운로드)
- Create: `models/wake/embedding_model.onnx` (다운로드)
- Create: `tests/test_wake_chain_shapes.py`

**Interfaces:**
- Consumes: 없음
- Produces: `models/wake/` 아래 특징추출 ONNX 2개. Task 3이 이 파일들을 로드한다.

> **왜 지금 하나:** 이 두 파일은 **범용 사전학습 모델**이라 우리 '재하봇' 학습 결과와 무관하다. Colab 학습이 끝나기를 기다릴 필요가 없고, 먼저 받아두면 Task 3~4의 체인 코드를 실제 텐서로 검증할 수 있다.

- [ ] **Step 1: 파일 확보**

Colab(학습 중인 노트북)에서 패키지에 번들된 파일을 꺼내는 것이 가장 확실하다:

```python
# Colab 셀에서 실행
import livekit.wakeword, pathlib, shutil
pkg = pathlib.Path(livekit.wakeword.__file__).parent
for p in pkg.rglob("*.onnx"):
    print(p)          # melspectrogram.onnx / embedding_model.onnx 경로 확인
    shutil.copy(p, "/content/")   # 다운로드 받을 위치로 복사
```

받은 두 파일을 노트북의 `models/wake/` 에 넣는다:

```bash
mkdir -p "C:/Users/Moon/Desktop/jaeha_bot/models/wake"
```

- [ ] **Step 2: 파일 존재와 텐서 계약 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -c "
import onnxruntime as ort
for name in ['melspectrogram', 'embedding_model']:
    s = ort.InferenceSession(f'models/wake/{name}.onnx', providers=['CPUExecutionProvider'])
    print(name)
    for i in s.get_inputs():  print('  in :', i.name, i.shape, i.type)
    for o in s.get_outputs(): print('  out:', o.name, o.shape, o.type)
"
```
Expected: 두 모델의 입출력 이름·shape이 출력된다. **여기서 나온 실제 shape을 Step 3 테스트에 반영한다** — 아래 테스트는 openWakeWord 규격 기준이며, 다르면 실제 값에 맞춘다.

- [ ] **Step 3: 형상 검증 테스트 작성**

`tests/test_wake_chain_shapes.py`:

```python
"""특징추출 ONNX 2개의 실제 텐서 계약을 고정한다.

이 테스트가 깨지면 모델 파일이 바뀐 것이므로 app/wake_onnx.py 의 체인도 함께 봐야 한다.
"""
import pathlib

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models" / "wake"
MEL = MODEL_DIR / "melspectrogram.onnx"
EMB = MODEL_DIR / "embedding_model.onnx"

pytestmark = pytest.mark.skipif(
    not (MEL.exists() and EMB.exists()),
    reason="특징추출 ONNX 미배치 — Task 2 Step 1 참고",
)


def _sess(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_melspectrogram_accepts_1280_samples():
    s = _sess(MEL)
    # 중요: int16 '범위' 값을 float32 dtype 으로 넣는다([-1,1] 아님)
    audio = (np.random.uniform(-0.5, 0.5, 1280) * 32767).astype(np.float32)
    out = s.run(None, {s.get_inputs()[0].name: audio[None, :]})[0]
    mel = np.squeeze(out)
    assert mel.ndim == 2
    assert mel.shape[1] == 32, f"멜 밴드가 32가 아님: {mel.shape}"
    assert mel.shape[0] == 8, f"1280샘플은 멜 8프레임이어야 함(보폭 1칸): {mel.shape}"


def test_embedding_accepts_76_mel_frames_and_returns_96():
    s = _sess(EMB)
    mel_window = np.zeros((1, 76, 32, 1), dtype=np.float32)
    out = s.run(None, {s.get_inputs()[0].name: mel_window})[0]
    assert np.squeeze(out).shape == (96,), f"임베딩 차원이 96이 아님: {out.shape}"


def test_normalization_constant_changes_values():
    """x/10+2 정규화가 실제로 값을 바꾸는지 = 빠뜨리면 조용히 실패하는 지점."""
    s = _sess(MEL)
    audio = (np.random.uniform(-0.5, 0.5, 1280) * 32767).astype(np.float32)
    raw = np.squeeze(s.run(None, {s.get_inputs()[0].name: audio[None, :]})[0])
    normed = raw / 10.0 + 2.0
    assert not np.allclose(raw, normed), "정규화가 무의미 — 상수 확인 필요"
```

- [ ] **Step 4: 테스트 실행**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_chain_shapes.py -v
```
Expected: PASS (3 passed). 파일이 아직 없으면 3 skipped — 그 경우 Step 1로 돌아간다.

- [ ] **Step 5: 커밋** *(git init 한 경우에만)*

```bash
git add tests/test_wake_chain_shapes.py && git commit -m "test(wake): 특징추출 ONNX 텐서 계약 고정"
```

> `models/` 는 용량이 커서 커밋하지 않는다. 젯슨 전송은 `push_model.sh` 를 쓴다.

---

## Task 3: OnnxWakeDetector 3단 체인과 링버퍼

**Files:**
- Create: `app/wake_onnx.py`
- Create: `tests/test_wake_onnx.py`

**Interfaces:**
- Consumes: `app.audio_source.AudioSource`, `FRAME`, `SAMPLE_RATE` (Task 1)
- Produces:
  - `WakeResult(preroll: np.ndarray, continued: bool, score: float)`
  - `OnnxWakeDetector(model_dir, classifier="jaehabot.onnx", threshold=0.5, trigger_frames=2, providers=None, source=None, continuation_window=0.5)`
  - `OnnxWakeDetector.push(frame: np.ndarray) -> float | None` — 워밍업 중엔 `None`
  - `OnnxWakeDetector.reset() -> None`
  - `OnnxWakeDetector.WARMUP_FRAMES: int` — 첫 점수가 나오기까지 필요한 프레임 수

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_wake_onnx.py`:

```python
"""OnnxWakeDetector 의 링버퍼·워밍업 로직. ONNX 세션은 가짜로 대체해 모델 없이 검증한다."""
import numpy as np
import pytest

from app.audio_source import FRAME
from app.wake_onnx import OnnxWakeDetector, WakeResult


class FakeSession:
    """onnxruntime InferenceSession 흉내. 정해진 shape 을 돌려준다."""

    def __init__(self, in_name, out_shape):
        self._in = in_name
        self._out_shape = out_shape
        self.calls = []

    def get_inputs(self):
        class _I:
            name = self._in
        return [_I()]

    def run(self, _out, feed):
        self.calls.append(feed[self._in])
        return [np.zeros(self._out_shape, dtype=np.float32)]


def make_detector(score=0.0, **kw):
    """ONNX 로드를 건너뛰고 가짜 세션을 꽂은 감지기."""
    d = OnnxWakeDetector.__new__(OnnxWakeDetector)
    d._mel_sess = FakeSession("x", (1, 1, 8, 32))
    d._emb_sess = FakeSession("x", (1, 1, 1, 96))

    class ScoreSession(FakeSession):
        def run(self, _out, feed):
            self.calls.append(feed[self._in])
            return [np.array([[score]], dtype=np.float32)]

    d._cls_sess = ScoreSession("embeddings", (1, 1))
    d._init_state(threshold=kw.get("threshold", 0.5),
                  trigger_frames=kw.get("trigger_frames", 2),
                  continuation_window=kw.get("continuation_window", 0.5),
                  source=kw.get("source"))
    return d


def _frame(v=0.1):
    return np.full(FRAME, v, dtype=np.float32)


def test_warmup_returns_none_until_buffers_fill():
    d = make_detector()
    # 멜 76프레임 채우는 데 76/8 = 9.5 -> 10 프레임,
    # 그 뒤 임베딩 16개 채우는 데 15 프레임 더 = 총 25 프레임째에 첫 점수.
    scores = [d.push(_frame()) for _ in range(OnnxWakeDetector.WARMUP_FRAMES - 1)]
    assert all(s is None for s in scores), "워밍업 중엔 점수를 내면 안 됨"


def test_first_score_after_warmup():
    d = make_detector(score=0.9)
    for _ in range(OnnxWakeDetector.WARMUP_FRAMES - 1):
        d.push(_frame())
    s = d.push(_frame())
    assert s is not None
    assert s == pytest.approx(0.9)


def test_classifier_receives_16_embeddings_of_96():
    d = make_detector(score=0.1)
    for _ in range(OnnxWakeDetector.WARMUP_FRAMES):
        d.push(_frame())
    fed = d._cls_sess.calls[-1]
    assert fed.shape == (1, 16, 96), f"분류기 입력 형상 오류: {fed.shape}"
    assert fed.dtype == np.float32


def test_melspectrogram_gets_int16_scaled_float():
    """[-1,1] 을 그대로 넣으면 조용히 실패한다. 32767 배율 확인."""
    d = make_detector()
    d.push(np.full(FRAME, 1.0, dtype=np.float32))
    fed = d._mel_sess.calls[0]
    assert fed.dtype == np.float32
    assert abs(float(np.max(np.abs(fed))) - 32767.0) < 1.0, \
        f"int16 범위로 스케일되지 않음: max={np.max(np.abs(fed))}"


def test_reset_clears_buffers():
    d = make_detector(score=0.9)
    for _ in range(OnnxWakeDetector.WARMUP_FRAMES):
        d.push(_frame())
    d.reset()
    assert d.push(_frame()) is None, "reset 후엔 다시 워밍업해야 함"


def test_wake_result_fields():
    r = WakeResult(preroll=np.zeros(4, dtype=np.float32), continued=True, score=0.8)
    assert r.preroll.size == 4
    assert r.continued is True
    assert r.score == pytest.approx(0.8)
```

- [ ] **Step 2: 테스트 실패 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_onnx.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.wake_onnx'`

- [ ] **Step 3: 구현 작성**

`app/wake_onnx.py`:

```python
"""ONNX 호출어 감지기 — 세션 트리거의 '감지기' 부분(교체 가능).

기존 STT 감지기(app/wake.py)는 whisper 가 받아쓴 '텍스트'를 '재하봇'과 자모 비교했다.
그 방식은 (1)'재하봇'이 OOV 라 유아 웅얼거림에서 딴판으로 적히고 (2)받아쓰는 4~6초 동안
호출을 놓쳐 구조적 한계가 있었다. 이 감지기는 받아쓰기를 하지 않고 음향 패턴만 본다.

체인(openWakeWord 규격, livekit-wakeword 호환):
    80ms 프레임(1280샘플)
      -> melspectrogram.onnx -> 멜 8프레임 -> [x/10+2] -> 멜 링버퍼(76)
      -> embedding_model.onnx(창 76, 보폭 8) -> 임베딩(96) -> 임베딩 링버퍼(16)
      -> <호출어>.onnx -> score

조용한 실패 지점 2가지(예외가 안 나고 점수만 망가진다):
  1) 멜 입력은 int16 '범위' 값을 float32 로 준다. 우리 오디오는 [-1,1] 이라 32767 배 해야 한다.
  2) 멜 출력에 x/10+2 정규화를 반드시 적용한다.
[[jaeha-bot-progress]]
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .audio_source import FRAME, SAMPLE_RATE

log = logging.getLogger("jaeha_bot.wake_onnx")

MEL_WINDOW = 76     # 임베딩 하나가 보는 멜 프레임 수
MEL_BANDS = 32
MEL_PER_FRAME = 8   # 1280샘플이 만드는 멜 프레임 수 = 보폭과 같음
EMB_WINDOW = 16     # 분류기가 보는 임베딩 개수
EMB_DIM = 96
INT16_SCALE = 32767.0


@dataclass
class WakeResult:
    """호출어가 걸렸을 때 상위(main)에 넘기는 것."""
    preroll: np.ndarray   # 감지 직전 오디오(+이어진 발화). continued=False 면 빈 배열
    continued: bool       # 감지 직후에도 말이 이어졌는가(인사말 생략 판단)
    score: float = 0.0


class OnnxWakeDetector:
    # 멜 76프레임 채우기 = ceil(76/8) = 10 프레임,
    # 그 뒤 임베딩 16개 채우기 = 15 프레임 더. 총 25번째 push 에서 첫 점수.
    WARMUP_FRAMES = -(-MEL_WINDOW // MEL_PER_FRAME) + EMB_WINDOW - 1

    def __init__(self, model_dir, classifier: str = "jaehabot.onnx",
                 threshold: float = 0.5, trigger_frames: int = 2,
                 providers=None, source=None,
                 continuation_window: float = 0.5) -> None:
        import pathlib

        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        d = pathlib.Path(model_dir)

        def _load(name):
            p = d / name
            if not p.exists():
                raise FileNotFoundError(f"ONNX 없음: {p}")
            return ort.InferenceSession(str(p), providers=providers)

        self._mel_sess = _load("melspectrogram.onnx")
        self._emb_sess = _load("embedding_model.onnx")
        self._cls_sess = _load(classifier)
        log.info("호출어 ONNX 로드: %s (providers=%s)", classifier, providers)

        self._init_state(threshold, trigger_frames, continuation_window, source)

    def _init_state(self, threshold, trigger_frames, continuation_window, source):
        """__init__ 과 테스트가 공유하는 순수 상태 초기화(ONNX 로드 없음)."""
        self.threshold = float(threshold)
        self.trigger_frames = int(trigger_frames)
        self.continuation_window = float(continuation_window)
        self.source = source
        self._mel: deque[np.ndarray] = deque(maxlen=MEL_WINDOW)
        self._emb: deque[np.ndarray] = deque(maxlen=EMB_WINDOW)
        self._hits = 0

    # ------------------------------------------------------------------ 체인
    def reset(self) -> None:
        """링버퍼를 비운다. 깨어난 뒤 다시 대기로 갈 때 호출."""
        self._mel.clear()
        self._emb.clear()
        self._hits = 0

    def push(self, frame: np.ndarray) -> float | None:
        """80ms 프레임 하나를 넣고 점수를 얻는다. 워밍업 중이면 None."""
        # 1) 멜: int16 범위 float32 로 넣고, 출력에 x/10+2 정규화
        audio = np.asarray(frame, dtype=np.float32).reshape(1, -1) * INT16_SCALE
        name = self._mel_sess.get_inputs()[0].name
        mel = np.squeeze(self._mel_sess.run(None, {name: audio})[0])
        mel = mel.reshape(-1, MEL_BANDS) / 10.0 + 2.0
        for row in mel:
            self._mel.append(row.astype(np.float32))
        if len(self._mel) < MEL_WINDOW:
            return None

        # 2) 임베딩: 멜 76프레임 창 -> 96차원 하나
        window = np.stack(list(self._mel))[None, :, :, None].astype(np.float32)
        name = self._emb_sess.get_inputs()[0].name
        emb = np.squeeze(self._emb_sess.run(None, {name: window})[0])
        self._emb.append(emb.reshape(EMB_DIM).astype(np.float32))
        if len(self._emb) < EMB_WINDOW:
            return None

        # 3) 분류기: 임베딩 16개 -> 점수
        feats = np.stack(list(self._emb))[None, :, :].astype(np.float32)
        name = self._cls_sess.get_inputs()[0].name
        return float(np.squeeze(self._cls_sess.run(None, {name: feats})[0]))
```

- [ ] **Step 4: 테스트 통과 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_onnx.py -v
```
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋** *(git init 한 경우에만)*

```bash
git add app/wake_onnx.py tests/test_wake_onnx.py && git commit -m "feat(wake): ONNX 3단 체인과 링버퍼"
```

---

## Task 4: wait_for_wake 와 인사말 분기

**Files:**
- Modify: `app/wake_onnx.py` (메서드 추가)
- Modify: `tests/test_wake_onnx.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 3의 `OnnxWakeDetector.push()`, Task 1의 `AudioSource.read()/preroll()/clear_preroll()/noise_floor`
- Produces: `OnnxWakeDetector.wait_for_wake(max_frames=None) -> WakeResult | None`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_wake_onnx.py` 끝에 추가:

```python
# ── wait_for_wake / 인사말 분기 ────────────────────────────────────────────

class ScriptedSource:
    """AudioSource 흉내 — 정해진 프레임을 돌려주고 프리롤을 흉내낸다."""

    def __init__(self, frames, preroll_frames=6):
        self._frames = list(frames)
        self._i = 0
        self._ring = []
        self.preroll_frames = preroll_frames
        self.noise_floor = 0.001

    def read(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        self._ring.append(f)
        self._ring = self._ring[-self.preroll_frames:]
        return f

    def preroll(self):
        if not self._ring:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._ring).astype(np.float32)

    def clear_preroll(self):
        self._ring = []


def test_requires_trigger_frames_consecutive_hits():
    """단발 점수 튐으로는 안 깨어난다."""
    src = ScriptedSource([_frame()] * 200)
    d = make_detector(score=0.9, trigger_frames=2, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None, "연속 2프레임이면 깨어나야 함"


def test_below_threshold_never_wakes():
    src = ScriptedSource([_frame()] * 200)
    d = make_detector(score=0.1, threshold=0.5, source=src)
    assert d.wait_for_wake(max_frames=150) is None


def test_continued_true_when_speech_follows():
    """감지 직후 큰 소리가 이어지면 continued=True 이고 오디오가 실려온다."""
    loud = np.full(FRAME, 0.5, dtype=np.float32)
    src = ScriptedSource([loud] * 200)
    d = make_detector(score=0.9, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None
    assert r.continued is True
    assert r.preroll.size > 0, "한 숨 패턴이면 오디오를 넘겨야 함"


def test_continued_false_when_silence_follows():
    """부르고 멈추면 continued=False 이고 프리롤은 버린다(인사말 경로)."""
    loud = np.full(FRAME, 0.5, dtype=np.float32)
    quiet = np.zeros(FRAME, dtype=np.float32)
    src = ScriptedSource([loud] * 40 + [quiet] * 160)
    d = make_detector(score=0.9, source=src)
    r = d.wait_for_wake(max_frames=150)
    assert r is not None
    assert r.continued is False
    assert r.preroll.size == 0, "인사말 경로에선 프리롤을 버려야 함"
```

- [ ] **Step 2: 테스트 실패 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_onnx.py -v -k "trigger or threshold or continued"
```
Expected: FAIL — `AttributeError: 'OnnxWakeDetector' object has no attribute 'wait_for_wake'`

- [ ] **Step 3: 구현 추가**

`app/wake_onnx.py` 의 `push()` 아래에 추가:

```python
    # -------------------------------------------------------------- 대기 루프
    def wait_for_wake(self, max_frames: int | None = None) -> WakeResult | None:
        """호출어가 걸릴 때까지 프레임을 읽는다. 걸리면 WakeResult, 아니면 None.

        max_frames 는 테스트·상위 타임아웃용. None 이면 걸릴 때까지 계속 듣는다.
        """
        if self.source is None:
            raise RuntimeError("source 가 없다 — AudioSource 를 넘겨야 한다")

        n = 0
        best = 0.0
        while max_frames is None or n < max_frames:
            n += 1
            score = self.push(self.source.read())
            if score is None:      # 워밍업
                continue
            best = max(best, score)

            if score < self.threshold:
                self._hits = 0
                # 진단: 임계값 튜닝 근거. 대기 중 주기적으로 최고 점수를 남긴다.
                if n % 250 == 0:   # 250프레임 = 20초
                    log.info("[대기] 최근 20초 최고 점수 %.3f (임계 %.2f)", best, self.threshold)
                    best = 0.0
                continue

            self._hits += 1
            if self._hits < self.trigger_frames:
                continue

            # ── 깨움 확정 ──
            log.info("[호출] 점수 %.3f → 깨어남", score)
            pre = self.source.preroll()          # 감지 직전 0.5초(호출어 포함)
            tail, continued = self._observe_continuation()
            self._hits = 0
            if continued:
                audio = np.concatenate([pre, tail]) if tail.size else pre
            else:
                # 부르고 기다리는 패턴 — 인사말을 하는 사이 낡으므로 버린다.
                audio = np.zeros(0, dtype=np.float32)
                self.source.clear_preroll()
            return WakeResult(preroll=audio.astype(np.float32),
                              continued=continued, score=score)
        return None

    def _observe_continuation(self) -> tuple[np.ndarray, bool]:
        """감지 직후 잠깐 들어 말이 이어지는지 본다. (읽은 오디오, 이어짐 여부)."""
        n = max(1, int(self.continuation_window * SAMPLE_RATE / FRAME))
        frames = []
        for _ in range(n):
            try:
                frames.append(self.source.read())
            except StopIteration:
                break
        if not frames:
            return np.zeros(0, dtype=np.float32), False
        tail = np.concatenate(frames).astype(np.float32)
        # 말소리 판정: STT VAD 와 같은 기준(소음 바닥의 2배, 하한 0.005)을 쓴다.
        floor = getattr(self.source, "noise_floor", 0.0)
        thr = max(floor * 2.0, 0.005)
        loudest = max(float(np.sqrt(np.mean(np.square(f)))) for f in frames)
        return tail, loudest >= thr
```

- [ ] **Step 4: 테스트 통과 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_onnx.py -v
```
Expected: PASS (10 passed)

- [ ] **Step 5: 커밋** *(git init 한 경우에만)*

```bash
git add app/wake_onnx.py tests/test_wake_onnx.py && git commit -m "feat(wake): wait_for_wake 와 인사말 분기"
```

---

## Task 5: SttWakeDetector 래퍼와 폴백 팩토리

**Files:**
- Modify: `app/wake.py` (파일 끝에 추가, 기존 함수는 건드리지 않음)
- Create: `tests/test_wake_factory.py`

**Interfaces:**
- Consumes: `app.wake_onnx.OnnxWakeDetector`, `WakeResult` (Task 3~4); 기존 `app.wake.is_wake_word`
- Produces:
  - `SttWakeDetector(stt, word, threshold, aliases)` with `.wait_for_wake() -> WakeResult | None`
  - `make_detector(wcfg: dict, stt, source) -> object` — `wcfg["detector"]` 에 따라 고르고, ONNX 실패 시 STT 로 폴백

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_wake_factory.py`:

```python
"""감지기 선택과 폴백. ONNX 파일이 없어도 봇이 죽지 않아야 한다."""
import numpy as np
import pytest

from app.wake import SttWakeDetector, make_detector
from app.wake_onnx import WakeResult


class FakeStt:
    def __init__(self, texts):
        self._texts = list(texts)
        self._i = 0

    def listen(self, **kw):
        if self._i >= len(self._texts):
            return "", 0.0
        t = self._texts[self._i]
        self._i += 1
        return t, 0.1


def test_stt_detector_wakes_on_wake_word():
    d = SttWakeDetector(FakeStt(["엄마 어디 있어", "재하봇"]), word="재하봇",
                        threshold=0.68, aliases=[])
    r = d.wait_for_wake(max_turns=5)
    assert isinstance(r, WakeResult)
    assert r.continued is False
    assert r.preroll.size == 0, "STT 감지기는 오디오를 넘기지 않는다"


def test_stt_detector_returns_none_when_never_called():
    d = SttWakeDetector(FakeStt(["엄마 어디 있어", "사과 먹고 싶어"]), word="재하봇",
                        threshold=0.68, aliases=[])
    assert d.wait_for_wake(max_turns=2) is None


def test_factory_selects_stt_when_configured():
    d = make_detector({"detector": "stt", "word": "재하봇", "threshold": 0.68},
                      stt=FakeStt([]), source=None)
    assert isinstance(d, SttWakeDetector)


def test_factory_falls_back_to_stt_when_onnx_missing(caplog):
    """분류기 파일이 없어도 예외 없이 STT 감지기로 내려와야 한다."""
    cfg = {"detector": "onnx", "word": "재하봇", "threshold": 0.68,
           "onnx": {"model_dir": "models/does-not-exist",
                    "classifier": "nope.onnx"}}
    d = make_detector(cfg, stt=FakeStt([]), source=object())
    assert isinstance(d, SttWakeDetector), "폴백하지 않았다"
```

- [ ] **Step 2: 테스트 실패 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_factory.py -v
```
Expected: FAIL — `ImportError: cannot import name 'SttWakeDetector' from 'app.wake'`

- [ ] **Step 3: 구현 추가**

`app/wake.py` 의 `_repl()` **앞**에 추가 (기존 함수들은 그대로 둔다):

```python
# ── 감지기 인터페이스 ─────────────────────────────────────────────────────────
# 두 감지기는 입력이 다르다(STT=텍스트, ONNX=오디오). 그래서 is_wake_word(text)
# 수준에선 교체가 안 되고, '깨어날 때까지 기다린다' 수준으로 경계를 올린다.

class SttWakeDetector:
    """기존 STT 자모 판정을 감지기 인터페이스에 맞춘 래퍼.

    오디오를 다루지 않으므로 프리롤은 항상 비어 있고 continued=False 다
    (= 늘 '부르고 기다리기' 경로 = 이 방식의 기존 동작과 같다).
    """

    def __init__(self, stt, word: str = "재하봇", threshold: float = 0.68,
                 aliases: list | None = None) -> None:
        self.stt = stt
        self.word = word
        self.threshold = threshold
        self.aliases = aliases or []

    def wait_for_wake(self, max_turns: int | None = None):
        from .wake_onnx import WakeResult

        turns = 0
        while max_turns is None or turns < max_turns:
            turns += 1
            text, _ = self.stt.listen()
            if not text:
                continue
            if is_wake_word(text, self.word, self.threshold, self.aliases):
                log.info("[호출] %s → 깨어남", text)
                return WakeResult(preroll=_np.zeros(0, dtype=_np.float32),
                                  continued=False, score=1.0)
            log.info("[대기] 안 깨움: %r (거리 %.2f / 임계 %.2f)",
                     text, best_wake_ratio(text, self.word), self.threshold)
        return None

    def reset(self) -> None:
        """ONNX 감지기와 인터페이스를 맞추기 위한 no-op."""


def make_detector(wcfg: dict, stt, source):
    """설정에 따라 감지기를 고른다. ONNX 로드 실패 시 STT 로 폴백한다.

    폴백하는 이유: 모델을 아직 안 넣었거나 파일이 깨져도 봇이 죽으면 안 된다.
    """
    word = wcfg.get("word", "재하봇")
    threshold = float(wcfg.get("threshold", 0.68))
    aliases = wcfg.get("aliases", [])
    fallback = SttWakeDetector(stt, word, threshold, aliases)

    if wcfg.get("detector", "stt") != "onnx":
        return fallback

    ocfg = wcfg.get("onnx", {}) or {}
    try:
        from .wake_onnx import OnnxWakeDetector
        return OnnxWakeDetector(
            model_dir=ocfg.get("model_dir", "models/wake"),
            classifier=ocfg.get("classifier", "jaehabot.onnx"),
            threshold=float(ocfg.get("threshold", 0.5)),
            trigger_frames=int(ocfg.get("trigger_frames", 2)),
            providers=ocfg.get("providers"),
            source=source,
        )
    except Exception as e:
        log.warning("ONNX 호출어 감지기 로드 실패(%s) → STT 감지기로 폴백", e)
        return fallback
```

파일 상단 import 에 다음 두 줄을 추가한다:

```python
import logging

import numpy as _np
```

그리고 import 아래에 로거를 만든다:

```python
log = logging.getLogger("jaeha_bot.wake")
```

- [ ] **Step 4: 테스트 통과 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_wake_factory.py -v
```
Expected: PASS (4 passed)

- [ ] **Step 5: 기존 STT 판정이 안 깨졌는지 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m app.wake
```
Expected: 기존 REPL 출력 그대로 — `[깨워야 함]` 7개가 전부 `True`, `[안 깨야 함]` 5개가 전부 `False`

- [ ] **Step 6: 커밋** *(git init 한 경우에만)*

```bash
git add app/wake.py tests/test_wake_factory.py && git commit -m "feat(wake): 감지기 인터페이스 통일과 ONNX 폴백"
```

---

## Task 6: stt_module 에 공유 스트림 인자 추가

**Files:**
- Modify: `app/stt_module.py:107-203` (`record_until_silence`, `listen`)
- Create: `tests/test_stt_source.py`

**Interfaces:**
- Consumes: `app.audio_source.AudioSource` (Task 1)
- Produces:
  - `STTModule.record_until_silence(verbose=False, *, source=None, prefix=None) -> np.ndarray`
  - `STTModule.listen(verbose=False, *, source=None, prefix=None) -> tuple[str, float]`

**동작 표:**

| source | prefix | 동작 |
|---|---|---|
| None | None | **기존 그대로** — 자체 스트림을 열고 소음 측정 후 말 시작 대기 |
| 있음 | None | 공유 스트림, 소음 바닥은 source 에서, 말 시작 대기부터 |
| 있음 | 있음 | 공유 스트림, prefix 로 시작, **말 시작 대기를 건너뛰고** 바로 무음 대기 |

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stt_source.py`:

```python
"""record_until_silence 의 공유 스트림·prefix 경로. 마이크·모델 불필요."""
import numpy as np
import pytest

from app.audio_source import FRAME
from app.stt_module import STTModule


class ScriptedSource:
    def __init__(self, frames, noise_floor=0.001):
        self._frames = list(frames)
        self._i = 0
        self.noise_floor = noise_floor

    def read(self):
        if self._i >= len(self._frames):
            raise StopIteration("프레임 소진")
        f = self._frames[self._i]
        self._i += 1
        return f


def _loud(n):
    return [np.full(FRAME, 0.3, dtype=np.float32) for _ in range(n)]


def _quiet(n):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n)]


def test_prefix_is_included_in_output():
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    prefix = np.full(FRAME * 2, 0.4, dtype=np.float32)
    src = ScriptedSource(_loud(3) + _quiet(30))
    out = stt.record_until_silence(source=src, prefix=prefix)
    assert out.size >= prefix.size, "prefix 가 결과에 포함되어야 함"


def test_prefix_path_skips_start_wait():
    """prefix 가 있으면 '말 시작 대기' 없이 곧바로 무음 판정으로 간다."""
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    prefix = np.full(FRAME, 0.4, dtype=np.float32)
    # 처음부터 조용해도 prefix 덕에 녹음이 성립하고 곧 종료된다.
    src = ScriptedSource(_quiet(30))
    out = stt.record_until_silence(source=src, prefix=prefix)
    assert out.size > 0


def test_source_without_prefix_waits_for_speech():
    stt = STTModule(silence_duration=0.3, max_duration=5.0, start_timeout=1.0)
    src = ScriptedSource(_quiet(5) + _loud(3) + _quiet(30))
    out = stt.record_until_silence(source=src)
    assert out.size > 0, "말이 시작되면 녹음돼야 함"


def test_source_returns_empty_when_frames_exhausted():
    stt = STTModule(silence_duration=0.3, max_duration=5.0)
    src = ScriptedSource(_quiet(3))
    out = stt.record_until_silence(source=src)
    assert out.size == 0


def test_signature_still_accepts_no_arguments():
    """기존 호출부 호환 — 인자 없이 부를 수 있어야 한다(실행은 안 함)."""
    import inspect
    sig = inspect.signature(STTModule.record_until_silence)
    assert sig.parameters["source"].default is None
    assert sig.parameters["prefix"].default is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_stt_source.py -v
```
Expected: FAIL — `TypeError: record_until_silence() got an unexpected keyword argument 'source'`

- [ ] **Step 3: 구현 — 프레임 루프를 소스 무관하게 분리**

`app/stt_module.py` 의 `record_until_silence` 를 아래로 **교체**한다. 기존 VAD 로직(상대 종료 임계, 관대한 카운터, pre_roll)은 그대로 옮긴다:

```python
    def record_until_silence(self, verbose: bool = False, *,
                             source=None, prefix=None) -> np.ndarray:
        """말이 시작되면 녹음, silence_duration 만큼 조용해지면 종료.

        source: AudioSource(공유 스트림). None 이면 예전처럼 자체 스트림을 연다.
                호출어 감지기와 마이크를 나눠 쓸 때 넘긴다(장치가 하나뿐이라 필수).
        prefix: 이미 확보한 앞부분 오디오(호출어 프리롤). 주면 '말 시작 대기'를
                건너뛰고 바로 무음 판정으로 들어간다(한 숨에 말한 경우).

        반환: float32 numpy 오디오(모노, 16kHz). 말이 없으면 빈 배열.
        """
        if source is None:
            return self._record_own_stream(verbose)
        return self._record_from_source(source, verbose, prefix)

    def _record_from_source(self, source, verbose: bool, prefix) -> np.ndarray:
        """공유 스트림에서 녹음. 소음 바닥은 source 가 이미 재 뒀다(재측정 안 함)."""
        frame = int(SAMPLE_RATE * 0.03)  # 로그·임계 계산 기준은 기존과 동일
        threshold = max(getattr(source, "noise_floor", 0.0) * self.silence_ratio,
                        self.min_start_rms)

        collected: list[np.ndarray] = []
        if prefix is not None and np.asarray(prefix).size:
            collected.append(np.asarray(prefix, dtype=np.float32).reshape(-1))
            started = True
        else:
            started = False

        # 1) 말 시작 대기(prefix 가 있으면 건너뜀)
        if not started:
            waited = 0
            max_wait = int(self.start_timeout * SAMPLE_RATE / FRAME)
            while waited < max_wait:
                waited += 1
                try:
                    block = source.read()
                except StopIteration:
                    return np.zeros(0, dtype=np.float32)
                if _rms(block) >= threshold:
                    collected.append(block)
                    started = True
                    break
            if not started:
                if verbose:
                    print("  (입력 없음 — 종료)")
                return np.zeros(0, dtype=np.float32)

        # 2) 무음이 이어지면 종료. 임계는 절대값과 '직전 최고음량의 12%' 중 큰 값.
        quiet_needed = max(1, int(self.silence_duration * SAMPLE_RATE / FRAME))
        max_frames = int(self.max_duration * SAMPLE_RATE / FRAME)
        quiet = 0
        speech_peak = threshold
        while len(collected) < max_frames:
            try:
                block = source.read()
            except StopIteration:
                break
            collected.append(block)
            r = _rms(block)
            speech_peak = max(speech_peak, r)
            if r < max(threshold, speech_peak * 0.12):
                quiet += 1
                if quiet >= quiet_needed:
                    break
            else:
                quiet = max(0, quiet - 3)

        if not collected:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(collected).astype(np.float32).reshape(-1)
        return _normalize(audio)
```

기존 본문은 `_record_own_stream(self, verbose)` 라는 이름의 메서드로 **그대로** 옮긴다 (첫 줄의 docstring만 "자체 스트림을 열어 녹음한다(예전 동작)."으로 바꾼다). 파일 상단 import 에 `FRAME` 을 추가한다:

```python
from .audio_source import FRAME
```

`listen()` 도 인자를 넘기도록 바꾼다:

```python
    def listen(self, verbose: bool = False, *, source=None, prefix=None):
        """마이크 -> (인식 텍스트, 인식 처리시간초). 말이 없으면 ('', 0.0)."""
        t0 = time.perf_counter()
        audio = self.record_until_silence(verbose=verbose, source=source, prefix=prefix)
        rec_dt = time.perf_counter() - t0
```

(나머지 본문은 그대로 둔다.)

- [ ] **Step 4: 테스트 통과 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/test_stt_source.py -v
```
Expected: PASS (5 passed)

- [ ] **Step 5: 전체 테스트로 회귀 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/ -v
```
Expected: 전부 PASS

- [ ] **Step 6: 커밋** *(git init 한 경우에만)*

```bash
git add app/stt_module.py tests/test_stt_source.py && git commit -m "feat(stt): 공유 스트림·prefix 인자 추가(기본 동작 불변)"
```

---

## Task 7: main.py 통합과 설정

**Files:**
- Modify: `app/main.py:109-161` (대기 분기)
- Modify: `configs/model_paths.yaml:45-60` (`wake:` 섹션)

**Interfaces:**
- Consumes: `app.wake.make_detector` (Task 5), `app.audio_source.AudioSource` (Task 1), `STTModule.listen(source=, prefix=)` (Task 6)
- Produces: 없음 (최종 통합)

- [ ] **Step 1: 설정 갱신**

`configs/model_paths.yaml` 의 `wake:` 섹션을 아래로 교체한다:

```yaml
wake:
  # 호출어(웨이크워드) 트리거 = '대기↔대화' 세션. false 면 항상 대화(옛 동작).
  # 목적: 부르기 전엔 LLM·TTS 안 돌려 쓸모없는 답변·자원 절약(교수님 요청).
  enabled: true
  # 감지기 선택. onnx = 전용 KWS(유아 웅얼거림 커버), stt = 옛 자모 판정(명확발화용).
  # 모델이 기대만큼 안 나오면 이 한 줄만 stt 로 되돌리면 된다.
  detector: onnx
  word: 재하봇
  threshold: 0.68      # (stt 감지기 전용) 자모거리 컷
  aliases: ["개하복", "제하모", "재하모사", "재보소"]  # (stt 감지기 전용)
  sleep_timeout: 30    # 대화 중 이만큼(초) 조용하면 다시 대기(잠듦)로
  preroll: 0.5         # 감지 직전 이만큼(초)을 보관해 '재하봇 이거 뭐야?' 뒷말 보존.
                       # Alexa 도 같은 값(500ms). 길게 잡으면 호출 직전 TV·부모 말소리를
                       # 끌어들여 whisper 가 엉뚱하게 받아쓴다.
  onnx:
    model_dir: models/wake
    classifier: jaehabot.onnx
    threshold: 0.5       # 실기 측정 후 확정(Task 8)
    trigger_frames: 2    # 점수가 threshold 이상인 프레임이 연속 2개 '이상'이어야 깨움
    providers: [CPUExecutionProvider]   # 모델이 작아 CPU 로 충분. GPU 는 LLM 에 양보.
```

- [ ] **Step 2: main.py 의 대기 분기 교체**

`app/main.py` 의 import 에 추가한다:

```python
from .audio_source import AudioSource
from .wake import make_detector
```

`main()` 안에서 wake 설정을 읽는 부분을 아래로 교체한다:

```python
    # 트리거(호출어) 설정: wake.enabled 면 '대기↔대화' 세션. 아니면 항상 대화(옛 동작).
    wcfg = settings.models.get("wake", {}) or {}
    wake_enabled = wcfg.get("enabled", True)
    sleep_timeout = float(wcfg.get("sleep_timeout", 30))
    sleep_words = wcfg.get("sleep_words")

    # 마이크 스트림은 하나만 연다(젯슨 ReSpeaker 는 장치가 하나뿐).
    # 감지기와 STT 가 이걸 나눠 쓴다. 대기 모드가 아니면 열 필요가 없다.
    source = None
    detector = None
    if wake_enabled:
        source = AudioSource(preroll=float(wcfg.get("preroll", 0.5))).open()
        detector = make_detector(wcfg, stt=stt, source=source)
        log.info("트리거 모드: '%s' 라고 부르면 깨어납니다 (감지기=%s)",
                 wcfg.get("word", "재하봇"), type(detector).__name__)
        tts.speak(READY_ASLEEP)
        awake = False
    else:
        tts.speak(GREETING)
        awake = True
    time.sleep(ECHO_COOLDOWN)
    last_active = time.time()
    pending_audio = None   # 한 숨 패턴에서 넘어온 오디오(다음 listen 의 prefix)
```

루프의 대기 분기(`if not awake:` 블록 전체)를 아래로 교체한다:

```python
            # ── 대기 모드: 호출어만 기다린다(LLM·TTS·놀이 안 함 = 자원 절약). ──
            if not awake:
                res = detector.wait_for_wake()
                if res is None:
                    continue
                awake = True
                last_active = time.time()
                if res.continued and res.preroll.size:
                    # '재하봇 이거 뭐야?' — 인사말을 건너뛰고 뒷말을 그대로 STT 로.
                    log.info("한 숨 패턴 감지 → 인사 생략")
                    pending_audio = res.preroll
                else:
                    # '재하봇!' 하고 기다리는 패턴 — 인사하고 새로 듣는다.
                    tts.speak(WAKE_GREETING)
                    time.sleep(ECHO_COOLDOWN)
                continue
```

루프 맨 위의 듣기 부분을 공유 스트림·prefix 를 쓰도록 바꾼다:

```python
            # 듣기(항상): listen 전체시간 - 인식연산(tr_dt) = 녹음대기.
            t_listen = time.perf_counter()
            text, tr_dt = stt.listen(source=source, prefix=pending_audio)
            pending_audio = None
            stt_wait = max(0.0, (time.perf_counter() - t_listen) - tr_dt)
```

잠들 때 감지기 상태를 비우도록, `awake = False` 로 가는 두 곳(무응답 타임아웃, 잠들기 명령) 뒤에 각각 추가한다:

```python
                detector.reset()
```

마지막으로 `finally` 블록에서 스트림을 닫는다:

```python
    finally:
        if source is not None:
            source.close()
        metrics.summary()  # 세션 요약(중앙값/p90/최대메모리) 출력·기록
```

- [ ] **Step 3: 설정 로드와 import 검증(마이크 불필요)**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -c "
from app.config import settings
from app.wake import make_detector, SttWakeDetector
w = settings.models['wake']
print('detector =', w['detector'])
print('preroll  =', w['preroll'])
print('onnx     =', w['onnx'])
# 모델이 아직 없으므로 폴백이 동작해야 한다
d = make_detector(w, stt=None, source=None)
print('실제 선택된 감지기:', type(d).__name__)
assert isinstance(d, SttWakeDetector), '모델이 없는데 폴백하지 않았다'
print('OK — 모델 없이도 기동 가능')
"
```
Expected: `detector = onnx`, 폴백 경고 로그 1줄, `실제 선택된 감지기: SttWakeDetector`, `OK`

- [ ] **Step 4: 전체 테스트 확인**

Run:
```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && "C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe" -m pytest tests/ -v
```
Expected: 전부 PASS

- [ ] **Step 5: 커밋** *(git init 한 경우에만)*

```bash
git add app/main.py configs/model_paths.yaml && git commit -m "feat(wake): main 대기 루프를 감지기 인터페이스로 교체"
```

---

## Task 8: 젯슨 배포와 실기 검증 (학습 완료 후)

**Files:**
- Create: `models/wake/jaehabot.onnx` (Colab 학습 결과)
- Create: `scripts/wake_score_probe.py`

**Interfaces:**
- Consumes: Task 1~7 전부
- Produces: 확정된 `wake.onnx.threshold` 값

> **선행조건:** Colab 학습 완료 + `jaehabot.onnx` 확보. 그 전까지는 Task 7까지만 진행하고, 봇은 STT 폴백으로 정상 동작한다.

- [ ] **Step 1: 점수 분포 측정 도구 작성**

`scripts/wake_score_probe.py`:

```python
"""호출어 점수 분포 측정 — 임계값을 실측으로 정하기 위한 도구.

말할 때마다 그 구간의 최고 점수를 찍어준다. 또박또박 / 웅얼거림 / 무관한 말을
각각 여러 번 해보고, 두 분포가 갈라지는 지점을 threshold 로 잡는다.

실행(젯슨):
  conda activate jaeha_bot && cd ~/jaeha_bot
  python scripts/wake_score_probe.py
종료: Ctrl+C
"""
import sys
import time

sys.path.insert(0, ".")

from app.audio_source import AudioSource
from app.config import settings
from app.main import _setup_audio_device
from app.wake_onnx import OnnxWakeDetector

_setup_audio_device()          # ReSpeaker 를 콕 집는다(빠뜨리면 무음 녹음됨)
w = settings.models["wake"]
o = w["onnx"]

src = AudioSource(preroll=float(w.get("preroll", 0.5))).open()
det = OnnxWakeDetector(model_dir=o["model_dir"], classifier=o["classifier"],
                       threshold=float(o["threshold"]),
                       trigger_frames=int(o["trigger_frames"]),
                       providers=o.get("providers"), source=src)

print("=" * 56)
print("말해보세요. 0.5초마다 그 구간 최고 점수를 찍습니다.")
print("  ① 또박또박 '재하봇'  ② 웅얼거리며  ③ 작게  ④ 무관한 말(엄마/사과)")
print("종료: Ctrl+C")
print("=" * 56)

try:
    best, t0 = 0.0, time.time()
    while True:
        s = det.push(src.read())
        if s is None:
            continue
        best = max(best, s)
        if time.time() - t0 >= 0.5:
            bar = "#" * int(best * 40)
            print(f"  {best:5.3f} |{bar}")
            best, t0 = 0.0, time.time()
except KeyboardInterrupt:
    print("\n종료")
finally:
    src.close()
```

- [ ] **Step 2: 모델과 코드를 젯슨으로 전송**

```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && bash push_model.sh models/wake
```

```bash
cd "C:/Users/Moon/Desktop/jaeha_bot" && bash push_code.sh
```

`scripts/` 는 `push_code.sh` 대상이 아니므로 따로 보낸다:

```bash
scp -r "C:/Users/Moon/Desktop/jaeha_bot/scripts" jaeha_bot@100.65.22.17:~/jaeha_bot/
```

- [ ] **Step 3: 젯슨에서 ONNX 로드 확인**

```bash
ssh jaeha_bot@100.65.22.17 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && cd ~/jaeha_bot && python -c "
from app.config import settings
from app.wake import make_detector
d = make_detector(settings.models[\"wake\"], stt=None, source=object())
print(\"감지기:\", type(d).__name__)
assert type(d).__name__ == \"OnnxWakeDetector\", \"폴백됨 — ONNX 로드 실패\"
print(\"OK\")
"'
```
Expected: `감지기: OnnxWakeDetector`, `OK`. `SttWakeDetector` 가 나오면 경고 로그에서 원인을 본다.

- [ ] **Step 4: 점수 분포 측정 → 임계값 확정**

```bash
ssh -t jaeha_bot@100.65.22.17 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && cd ~/jaeha_bot && python scripts/wake_score_probe.py'
```

각 조건을 **최소 5회씩** 말하고 점수를 기록한다:

| 조건 | 기대 | 기록 |
|---|---|---|
| 또박또박 "재하봇" | 높음 | |
| 웅얼거리며 "재하봇" | 중간 | |
| 작게 "재하봇" | 중간 | |
| 무관한 말(엄마/사과/이거 뭐야) | 낮음 | |
| 조용한 방 | 매우 낮음 | |

**임계값 = 호출어 최저점과 무관어 최고점 사이.** 겹치면 호출어 쪽을 살리는 값으로 잡고(놓치는 게 더 나쁨), `trigger_frames` 로 오탐을 억제한다. 정한 값을 `configs/model_paths.yaml` 의 `wake.onnx.threshold` 에 넣고 `push_code.sh` 로 보낸다.

- [ ] **Step 5: 전체 시나리오 실기 검증**

```bash
ssh -t jaeha_bot@100.65.22.17 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && cd ~/jaeha_bot && python -m app.main'
```

확인할 것:

- [ ] 무관한 말에 안 깨어난다
- [ ] "재하봇!" 하고 멈추면 → 인사 후 대화가 열린다
- [ ] "재하봇 이거 뭐야?" 를 한 숨에 → **인사 없이 바로 답하고, 뒷말이 안 잘린다**
- [ ] 깨어난 뒤엔 호출어 없이 대화가 이어진다
- [ ] 30초 조용하면 다시 대기로 잠든다
- [ ] "잘 자" 하면 즉시 대기로 잠든다
- [ ] 잠든 뒤 다시 부르면 깨어난다(reset 후 재감지)

- [ ] **Step 6: 결과를 메모리에 기록**

`jaeha-bot-progress.md` 에 확정 임계값, 조건별 점수 분포, 실기 검증 결과를 남긴다.

---

## Self-Review

**스펙 커버리지:**

| 스펙 절 | 담당 Task |
|---|---|
| 2. Python 3.10 제약 / 의존성 0개 | Global Constraints, Task 2 |
| 3. 전처리 체인 + `x/10+2` | Task 2(계약 고정), Task 3(구현) |
| 4.1 스트리밍 증분 | Task 3 |
| 4.2 단일 스트림 + 프리롤 0.5초 | Task 1, Task 6 |
| 4.3 인사말 분기 | Task 4, Task 7 |
| 4.4 세션 동작 유지 | Task 7 |
| 5. 컴포넌트 전부 | Task 1·3·4·5·6·7 |
| 6. 설정 | Task 7 |
| 7. 에러 처리(폴백·진단 로그) | Task 4(진단), Task 5(폴백) |
| 8. 테스트 1~6 (모델 없이) | Task 1~7의 각 테스트 |
| 8. 테스트 7~8 (실기) | Task 8 |
| 9. 완료 기준 | Task 8 Step 5 |

빠진 항목 없음.

**타입 일관성:** `WakeResult(preroll, continued, score)` 는 Task 3에서 정의되어 Task 4·5·7에서 같은 필드명으로 쓰인다. `wait_for_wake()` 는 두 감지기 모두 `WakeResult | None` 을 돌려준다. `reset()` 은 두 감지기 모두 갖는다(STT 쪽은 no-op). `record_until_silence(source=, prefix=)` 시그니처는 Task 6에서 정의되어 Task 7에서 같게 호출된다.

**주의:** Task 2 Step 2에서 실제 텐서 shape을 확인한 뒤, 값이 다르면 Task 3의 `MEL_PER_FRAME`·`MEL_BANDS` 상수와 Task 2의 테스트를 실제 값에 맞춘다. 계획의 값은 openWakeWord 규격 기준이다.
