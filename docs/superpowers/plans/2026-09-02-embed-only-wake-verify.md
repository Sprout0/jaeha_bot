# 임베딩 단독 호출어 검증 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 호출어 2단계 검증에서 whisper 의존을 제거해, 전면 API(S2S) 전환 시 로컬 STT 모델을 아예 안 올려도 되게 한다.

**Architecture:** 1단계 ONNX 감지는 그대로 둔다. 2단계 검증만 `whisper 전사 → 자모거리` 에서 `임베딩 열 → 본보기 코사인 유사도` 로 갈아 끼운다. 임베딩 모델(`embedding_model.onnx`, 96차원)은 **1단계 감지기가 이미 올려 둔 것**이라 메모리가 늘지 않고, 비용이 whisper 1.22초에서 수 밀리초가 된다. 스위치는 설정 한 줄(`verify.mode`)이고 기본값은 현행이라, 켜기 전까지 동작이 하나도 안 바뀐다.

**Tech Stack:** Python 3.12 / onnxruntime / numpy / pytest. 실행은 conda env `jaeha_bot`.

## Global Constraints

- 코드 실행은 반드시 conda env `jaeha_bot` (`C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe`). 시스템 python 에는 패키지가 없다.
- 브랜치를 만들지 않는다. `main` 에 직접 작은 커밋으로 쌓는다(사용자 방침).
- **기본 동작은 절대 안 바뀐다.** 새 설정의 기본값은 현행 그대로여야 하고, 각 작업 끝에 전체 스위트(현재 885개)가 통과해야 한다.
- 테스트는 ONNX 모델 파일 없이 돈다. 검증 흐름은 `tests/test_wake_verify.py` 의 `_det(scores, verifier=None, **kw)`(가짜 소스 + 점수 수열), 감지기 내부는 `tests/test_wake_onnx.py` 의 `make_detector()`(가짜 세션)를 쓴다. **새로 만들지 말고 있는 것을 쓴다.**
- 로그는 한국어. 조용한 실패를 만들지 않는다 — 장치가 꺼지면 반드시 경고를 남긴다(이 프로젝트는 지연적재에서 조용한 실패로 한 번 당했다).
- 임계값은 **추측으로 정하지 않는다.** 이 프로젝트는 3분짜리 소음 표본으로 두 번 틀렸다(08-26 우회컷: 3분 최고 0.173 → 같은 날 실기 0.725).

> **진행 상황 (2026-09-02):** Task 1~3 완료 — 커밋 `41fc471`, `053432e`, `8c9739c`.
> 전체 스위트 895개 통과. **동작은 아직 안 바뀐다**(`mode: whisper`,
> `embed_rescue.enabled: false`). Task 0(거실 녹음)과 Task 4(젯슨 판정)가 남았다.

---

## 🔴 Task 0 (선결, 코드 아님): 컷을 정할 데이터를 만든다

**이 작업 없이 Task 1~4 를 켜면 안 된다.** 코드는 먼저 짜도 되지만, `enabled` 로 바꾸는 것은 여기서 나온 수가 있어야 한다.

지금 `min_similarity: 0.85` 는 **거실 소음 3분** 표본으로 정한 값이고, 여유가 0.059 밖에 없다(소음 최대 0.814 ↔ 살린 것 중 최저 0.873). 그리고 whisper 를 **보강(OR)** 하던 값이라, 단독 관문으로 쓰면 정밀도 요구가 완전히 달라진다.

- [ ] **Step 1: 거실 소음 30~60분 녹음**

봇이 실제로 놓일 자리에서, 평소처럼 지내며 녹음한다(TV·대화·주방 소리 포함, 호출어는 **부르지 않는다**). 16kHz 모노 wav.

- [ ] **Step 2: 소음에서 유사도 분포를 뽑는다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe tools/enroll_wake.py --score-noise <녹음.wav>
```

기록할 것: 후보 자리 유사도의 최대·p99·p95. **이게 컷의 하한이다.**

- [ ] **Step 3: 실제로 쓸 사람 목소리로 본보기를 등록한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe tools/enroll_wake.py --record 5 --out models/wake/v6/templates_haitid.npy
```

⚠️ 지금 본보기는 어른 한 사람(`data/wake_real/adult_20260826_1445`) 것이고 **화자 종속**이다. 봇을 쓸 사람이 자기 목소리로 등록하는 것이 기본이다.

- [ ] **Step 4: 진짜 호출 재현율을 잰다**

등록에 안 쓴 호출 녹음 10건 이상으로 유사도를 재고, **가장 낮은 값**을 기록한다. 이게 컷의 상한이다.

- [ ] **Step 5: 컷을 정하고 근거를 적는다**

컷 = (소음 최대) 와 (진짜 최저) 사이. 둘이 겹치면 **임베딩 단독은 성립하지 않는다** — 그 사실을 `reports/` 에 적고 이 계획을 중단한다. 겹치지 않으면 그 수를 Task 4 에서 설정에 넣는다.

---

## Task 1: `_verify` 가 whisper 없이도 돌게 한다

**Files:**
- Modify: `app/wake_onnx.py` (`_verify`, 그리고 `_verify` 를 부르는 조건 — 현재 238행 `if self.verifier is not None and not self._verify(score)`)
- Test: `tests/test_wake_verify.py` (여기의 `_det(scores, verifier=None, **kw)` 헬퍼를 쓴다. `_verify` 는 `self.source` 에서 창을 읽으므로 직접 부르면 안 되고, 기존 테스트들처럼 `wait_for_wake()` 로 몰아야 한다.)

**Interfaces:**
- Consumes: `EmbedRescue.passes(embs) -> tuple[bool, float]` (`app/wake_embed.py`), `OnnxWakeDetector.embed_sequence(audio) -> np.ndarray`
- Produces: `verifier=None, embed_rescue=<EmbedRescue>` 로 만든 감지기가 임베딩만으로 통과/기각을 결정한다. 뒷 작업이 이 조합에 기댄다.

지금 구조의 문제: `verifier` 가 None 이면 `_verify` 자체를 안 부른다. 그래서 임베딩만 꽂아도 **1단계 단독**이 되어 버린다 — 그건 이미 기각된 길이다(실제 거실 시간당 160회 헛깨움).

- [x] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_wake_verify.py` 끝에 붙인다:

```python
# ── 임베딩 단독 검증 (2026-09-02) ────────────────────────────────────────
# 🔴 왜: 전면 API(S2S) 로 가면 로컬 STT 가 없어 verifier 가 None 이 된다. 지금은 그때
#    _verify 를 아예 안 불러 **1단계 단독**으로 떨어지는데, 그건 실제 거실에서 시간당
#    160회 깨어나는 이미 기각된 길이다.

class _Rescue:
    """EmbedRescue 대역 — passes() 와 min_similarity 만 있으면 된다."""

    def __init__(self, ok, sim=0.9):
        self.ok, self.sim = ok, sim
        self.min_similarity = 0.85
        self.calls = 0

    def passes(self, embs):
        self.calls += 1
        return self.ok, self.sim


def test_verifier_없이_임베딩이_통과시키면_깨운다():
    r = _Rescue(ok=True)
    d, _ = _det([0.9], verifier=None, embed_rescue=r)
    d.embed_sequence = lambda audio: object()      # 임베딩 계산은 여기 관심사가 아니다
    assert d.wait_for_wake(max_frames=3) is not None
    assert r.calls == 1, "임베딩 대조가 안 불렸다"


def test_verifier_없이_임베딩이_기각하면_안_깨운다():
    # 🔴 급소. 여기서 깨우면 1단계 단독과 똑같아진다.
    r = _Rescue(ok=False, sim=0.1)
    d, _ = _det([0.9], verifier=None, embed_rescue=r)
    d.embed_sequence = lambda audio: object()
    assert d.wait_for_wake(max_frames=3) is None
    assert r.calls == 1


def test_임베딩_계산이_터져도_봇이_안_죽는다():
    # 검증 장치가 봇을 죽이면 안 된다. 안전하게 기각으로 본다.
    r = _Rescue(ok=True)
    d, _ = _det([0.9], verifier=None, embed_rescue=r)

    def boom(audio):
        raise RuntimeError("onnx 죽음")

    d.embed_sequence = boom
    assert d.wait_for_wake(max_frames=3) is None
```

⚠️ `tests/test_wake_verify.py:59` 의 `test_no_verifier_behaves_exactly_like_before` 와
`:66` 의 `test_verifier_none_does_not_touch_the_audio_source` 는 **그대로 통과해야 한다**
(둘 다 `embed_rescue` 가 없는 구성이라 검증을 아예 안 탄다). 이게 롤백 보장이다.

- [x] **Step 2: 실패를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_wake_verify.py -k "임베딩" -v
```

기대: `test_verifier_없이_임베딩이_통과시키면_깨운다` 는 `r.calls == 1` 에서 FAIL(검증을 안 타서 0회), `test_verifier_없이_임베딩이_기각하면_안_깨운다` 도 FAIL(검증 없이 깨어난다).

- [x] **Step 3: `_verify` 를 고친다**

`app/wake_onnx.py` 의 `_verify` 안, RMS 게이트 **다음**부터를 아래로 바꾼다:

```python
        # 🔴 2026-09-02 whisper 없이도 돌 수 있어야 한다. 전면 API(S2S) 로 가면 로컬
        #    STT 를 안 올리는데, 그때 검증기가 None 이 되어 1단계 단독으로 떨어지면
        #    실제 거실에서 시간당 160회 깨어난다(이미 기각된 길이다).
        #    ➡️ verifier 가 없으면 임베딩이 **단독 관문**이 된다.
        if self.verifier is None:
            if self.embed_rescue is None:
                return True          # 검증 장치가 아예 없다 = 옛 동작(1단계 단독)
            try:
                ok, sim = self.embed_rescue.passes(self.embed_sequence(audio))
            except Exception as e:      # noqa: BLE001 — 검증기가 봇을 죽이면 안 된다
                log.warning("[검증] 임베딩 단독 실패(기각 처리): %s: %s",
                            type(e).__name__, e)
                return False
            finally:
                self._last_verify = time.monotonic()
            log.info("[검증] 임베딩 단독 %s — 유사도 %.3f %s %.2f (점수 %.3f)",
                     "통과" if ok else "기각", sim, ">=" if ok else "<",
                     self.embed_rescue.min_similarity, score)
            if ok:
                self._hits = 0
            return ok

        try:
            ok = bool(self.verifier(audio))
```

(그 아래 `except Exception ... finally ... self._last_verify = ...` 부터 기존 코드는 그대로 둔다.)

- [x] **Step 4: 호출 조건을 고친다**

`app/wake_onnx.py` 238행:

```python
            if self.verifier is not None and not self._verify(score):
```

를 아래로 바꾼다:

```python
            # 검증 장치가 **둘 중 하나라도** 있으면 태운다. verifier 만 보면
            # 임베딩 단독 구성에서 검증이 통째로 건너뛰어진다.
            if (self.verifier is not None or self.embed_rescue is not None) \
                    and not self._verify(score):
```

- [x] **Step 5: 통과를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_wake_verify.py -v
```

기대: 전부 PASS — 새 테스트 3개와, 롤백 보장 테스트 둘(`test_no_verifier_behaves_exactly_like_before`, `test_verifier_none_does_not_touch_the_audio_source`) 모두.

- [x] **Step 6: 전체 스위트로 회귀를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/ -q
```

기대: 885개 + 새 테스트 3개 = 888 passed.

- [x] **Step 7: 커밋**

```bash
git add app/wake_onnx.py tests/test_wake_verify.py
git commit -m "feat(wake): 검증기가 whisper 없이 임베딩만으로도 돌게 — S2S 로 가면 STT 가 없다

verifier 가 None 이면 _verify 를 아예 안 불러 1단계 단독으로 떨어졌다. 그건 실제
거실에서 시간당 160회 깨어나는, 이미 기각된 길이다. 이제 임베딩이 단독 관문이 된다.
기본 설정은 안 바뀐다(embed_rescue.enabled 는 여전히 false).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: 설정으로 검증 방식을 고른다

**Files:**
- Modify: `app/wake.py` (`make_detector` 안 `verifier=make_wake_verifier(...)` 부분, 현재 약 288행)
- Modify: `configs/model_paths.yaml` (`wake.onnx.verify` 절)
- Test: `tests/test_wake_factory.py`

**Interfaces:**
- Consumes: `make_wake_verifier(vcfg, stt, word)`, `_make_embed_rescue(vcfg)` (둘 다 `app/wake.py` 기존)
- Produces: `verify.mode` 설정값 세 가지 — `"whisper"`(기본, 현행) / `"embed"`(임베딩 단독) / `"both"`(현행 OR 보강). `make_detector` 가 이 값을 읽어 `verifier` 를 None 으로 만들지 결정한다.

- [x] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_wake_factory.py` 끝에 붙인다:

```python
import app.wake as wake_mod


def _capture_kwargs(monkeypatch):
    """OnnxWakeDetector 에 실제로 넘어간 인자를 잡아 둔다."""
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)

    import app.wake_onnx as wake_onnx
    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", Spy)
    return seen


def test_mode가_없으면_현행대로_whisper를_쓴다(monkeypatch):
    # 🔴 기본값이 바뀌면 아무도 모르게 동작이 달라진다. 여기서 못 박는다.
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True}}}
    make_detector(cfg, stt=FakeStt([]), source=object())
    assert seen["verifier"] is not None


def test_mode가_embed면_whisper를_안_만든다(monkeypatch):
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed"}}}
    make_detector(cfg, stt=FakeStt([]), source=object())
    assert seen["verifier"] is None


def test_mode가_embed면_stt가_없어도_만들어진다(monkeypatch):
    # S2S 구성에서는 stt 인스턴스 자체가 없다. 그때 죽으면 안 된다.
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed"}}}
    make_detector(cfg, stt=None, source=object())
    assert seen["verifier"] is None


def test_모르는_mode는_죽는다(monkeypatch):
    # 오타가 조용히 '현행'으로 떨어지면 켠 줄 알고 안 켜진다.
    _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embedding"}}}
    with pytest.raises(ValueError):
        make_detector(cfg, stt=FakeStt([]), source=object())
```

`tests/test_wake_factory.py` 맨 위에 `import pytest` 가 없으면 추가한다.

- [x] **Step 2: 실패를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_wake_factory.py -k "mode" -v
```

기대: `test_mode가_embed면_whisper를_안_만든다` 가 FAIL(아직 `mode` 를 안 읽으므로 verifier 가 생긴다).

- [x] **Step 3: `make_detector` 를 고친다**

`app/wake.py` 의 `make_detector` 안, `vcfg = ocfg.get("verify", {}) or {}` 바로 다음에 넣는다:

```python
        # 🔴 2026-09-02 검증 방식 스위치. 기본은 whisper = 현행 동작 그대로다.
        #   whisper : 후보를 전사해 자모거리로 판정(현행)
        #   embed   : 임베딩 본보기와 코사인 유사도로만 판정 — **로컬 STT 가 필요 없다**
        #   both    : whisper 로 판정하고, 기각된 것을 임베딩이 건진다(OR 보강)
        # ⚠️ 오타를 조용히 현행으로 떨어뜨리면 "켰는데 왜 그대로지"를 영영 못 찾는다.
        mode = vcfg.get("mode", "whisper")
        if mode not in ("whisper", "embed", "both"):
            raise ValueError(
                f"wake.onnx.verify.mode 가 이상하다: {mode!r} "
                "— whisper | embed | both 중 하나여야 한다")
        verifier = None if mode == "embed" else make_wake_verifier(vcfg, stt, word)
```

그리고 아래 `OnnxWakeDetector(...)` 호출의 `verifier=make_wake_verifier(vcfg, stt, word),` 를 `verifier=verifier,` 로 바꾼다.

- [x] **Step 4: 통과를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_wake_factory.py -v
```

기대: 전부 PASS.

- [x] **Step 5: 설정에 스위치를 적어 둔다 (기본값은 안 바꾼다)**

`configs/model_paths.yaml` 의 `verify:` 절 안, `enabled: true` 바로 아래에 넣는다:

```yaml
      # 🔴 2026-09-02 검증 방식. **기본 whisper = 현행 동작 그대로다.**
      #   whisper : 후보를 전사해 자모거리로 판정
      #   embed   : 임베딩 본보기와의 유사도로만 판정 — 로컬 STT 를 안 올려도 된다
      #             (전면 API/S2S 전환의 선결 조건)
      #   both    : whisper 판정 + 기각분을 임베딩이 건짐(OR 보강)
      # ⚠️ embed 로 바꾸기 전에 반드시 embed_rescue.min_similarity 를 **거실 30~60분
      #    녹음**으로 다시 잡을 것. 지금 0.85 는 3분 표본이고, whisper 를 보강하던
      #    값이라 단독 관문의 요구와 다르다. 이 프로젝트는 같은 크기의 표본으로
      #    두 번 틀렸다(08-26 우회컷: 3분 최고 0.173 -> 같은 날 실기 0.725).
      mode: whisper
```

- [x] **Step 6: 전체 스위트**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/ -q
```

기대: 892 passed (888 + 4).

- [x] **Step 7: 커밋**

```bash
git add app/wake.py configs/model_paths.yaml tests/test_wake_factory.py
git commit -m "feat(wake): verify.mode 로 검증 방식을 고른다 — 기본은 현행(whisper)

embed 로 두면 whisper 검증기를 아예 안 만든다. 오타는 조용히 현행으로 떨어지지 않고
ValueError 로 죽는다 — '켰는데 왜 그대로지'를 못 찾는 일이 없게.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: 임베딩 단독일 때 stt 없이 부팅되게 한다

**Files:**
- Modify: `app/wake.py` (`make_detector` 의 폴백 경로, 현재 약 280행 `fallback = SttWakeDetector(...)`)
- Test: `tests/test_wake_factory.py`

**Interfaces:**
- Consumes: Task 2 의 `mode` 값
- Produces: `stt=None` + `mode="embed"` 에서 ONNX 로드가 실패하면 **조용히 폴백하지 않고 죽는다**. 뒷 작업(운영 배선)이 이 동작에 기댄다.

지금은 ONNX 로드가 실패하면 `SttWakeDetector` 로 조용히 폴백한다. S2S 구성에는 stt 가 없으므로 그 폴백은 **부를 수 없는 객체**다. 조용히 넘어가면 봇이 영영 안 깨어나고 이유도 안 보인다.

- [x] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_stt가_없는데_onnx가_죽으면_조용히_넘어가지_않는다(monkeypatch):
    # 🔴 조용한 폴백이 제일 나쁘다. 봇이 안 깨어나는데 로그에 아무것도 안 남는다.
    import app.wake_onnx as wake_onnx

    def _boom(**kw):
        raise RuntimeError("모델 없음")

    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", _boom)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"enabled": True, "mode": "embed"}}}
    with pytest.raises(RuntimeError):
        make_detector(cfg, stt=None, source=object())


def test_stt가_있으면_예전처럼_폴백한다(monkeypatch):
    import app.wake_onnx as wake_onnx

    def _boom(**kw):
        raise RuntimeError("모델 없음")

    monkeypatch.setattr(wake_onnx, "OnnxWakeDetector", _boom)
    cfg = {"detector": "onnx", "word": "하이티드", "onnx": {"verify": {"enabled": True}}}
    d = make_detector(cfg, stt=FakeStt([]), source=object())
    assert isinstance(d, SttWakeDetector)
```

- [x] **Step 2: 실패를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_wake_factory.py -k "조용히" -v
```

기대: FAIL — 지금은 `SttWakeDetector(None, ...)` 를 만들어 돌려준다.

- [x] **Step 3: 폴백을 고친다**

`app/wake.py` 의 `make_detector` 에서 `fallback = SttWakeDetector(...)` 줄을 아래로 바꾼다:

```python
    # 🔴 stt 가 없으면 폴백이 성립하지 않는다(SttWakeDetector 는 stt 로 듣는다).
    #    전면 API 구성에는 로컬 STT 가 아예 없다. 그때 조용히 이걸 돌려주면 봇이
    #    영영 안 깨어나면서 로그에 아무것도 안 남는다 — 제일 나쁜 실패다.
    fallback = SttWakeDetector(stt, word, threshold, aliases) if stt is not None else None
```

그리고 `except Exception as e:` 로 시작하는 폴백 처리 블록 안, `return fallback` 앞에 넣는다:

```python
        if fallback is None:
            raise RuntimeError(
                "ONNX 호출어 모델을 못 올렸는데 STT 폴백도 없다"
                f"(stt=None) — 모델을 확인할 것: {type(e).__name__}: {e}") from e
```

`if wcfg.get("detector", "stt") != "onnx": return fallback` 앞에도 같은 보호를 넣는다:

```python
    if wcfg.get("detector", "stt") != "onnx":
        if fallback is None:
            raise RuntimeError("detector 가 stt 인데 stt 인스턴스가 없다")
        return fallback
```

- [x] **Step 4: 통과를 확인한다**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/test_wake_factory.py -v
```

- [x] **Step 5: 전체 스위트**

```bash
C:/Users/Moon/anaconda3/envs/jaeha_bot/python.exe -m pytest tests/ -q
```

기대: 894 passed.

- [x] **Step 6: 커밋**

```bash
git add app/wake.py tests/test_wake_factory.py
git commit -m "fix(wake): stt 없이 ONNX 가 죽으면 조용히 폴백하지 말고 죽는다

S2S 구성에는 로컬 STT 가 없다. 그때 SttWakeDetector(None) 을 돌려주면 봇이 영영
안 깨어나면서 로그에 아무것도 안 남는다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: 실기 판정 — 켜도 되는지 정한다

**Files:**
- Modify: `configs/model_paths.yaml` (`verify.mode`, `embed_rescue.enabled`, `min_similarity`)
- Create: `reports/wake/embed_only_20260902.md`

**Interfaces:**
- Consumes: Task 0 의 컷 값, Task 1~3 의 코드
- Produces: 켠다/안 켠다 결정과 그 근거. 이 문서가 없으면 켜지 않는다.

⚠️ **젯슨에서 한다.** 노트북 값은 마이크·소음 환경이 달라 옮겨가지 않는다.

- [ ] **Step 1: 설정을 켠다**

```yaml
      mode: embed
      ...
      embed_rescue:
        enabled: true
        templates: models/wake/v6/templates_haitid.npy
        min_similarity: <Task 0 Step 5 에서 정한 값>
```

- [ ] **Step 2: 기동해서 로그로 확인한다**

```bash
./run.sh check
```

기대: `임베딩 대조 켜짐 — 본보기 N개, 컷 X.XX` 가 뜨고, whisper 로드 로그가 **안** 뜬다.

- [ ] **Step 3: 재현율을 잰다**

실제로 쓸 사람이 평소 자리·평소 목소리로 20회 부른다. 깨어난 횟수를 센다.

**판정 기준: 18/20 이상.** 현행 whisper 검증의 실측 재현율이 그 언저리다(캐스케이드 80%, v6 화자 홀드아웃 97.6%@0.05). 이보다 낮으면 켜지 않는다.

- [ ] **Step 4: 헛깨움을 잰다**

아무도 안 부르는 상태로 **60분** 둔다(TV·대화 평소대로). 깨어난 횟수를 센다.

**판정 기준: 1회 이하/시간.** 참고로 1단계 단독은 시간당 160회, v6 1단계 헛후보는 시간당 6.95회다.

- [ ] **Step 5: 깨움 지연을 잰다**

로그의 `[검증]` 줄 시각차로 검증 비용을 낸다. whisper 는 1.22초였다.

**기대: 수십 밀리초.** 이게 안 나오면 임베딩 열을 잘못 만들고 있는 것이다.

- [ ] **Step 6: 결과를 적는다**

`reports/wake/embed_only_20260902.md` 에 재현율·헛깨움·지연·컷의 근거(소음 최대, 진짜 최저, 여유 폭)를 적는다. **여유 폭이 0.06 미만이면 "아직 못 켠다"로 적는다** — 08-27 에 0.059 였고 그건 얇다고 이미 판단했다.

- [ ] **Step 7: 커밋**

```bash
git add configs/model_paths.yaml reports/wake/embed_only_20260902.md
git commit -m "feat(wake): 임베딩 단독 검증으로 전환 — whisper 없이 깨어난다

재현율 N/20, 헛깨움 M회/시간, 검증 비용 1.22초 -> Xms. 컷 근거는 리포트에 있다.
이로써 전면 API(S2S) 전환에서 로컬 STT 를 안 올려도 된다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## 이 계획이 하지 않는 것

- **S2S 전환 자체는 하지 않는다.** 이건 그 전환의 선결 조건 하나만 푸는 계획이다. `stt_module`·`tts_module`·`main` 루프는 손대지 않는다.
- **화자 일반화를 풀지 않는다.** 본보기는 등록한 사람 것이고, 다른 사람이 부르면 어떻게 되는지는 여전히 미검증이다. 여러 사람이 쓸 거면 각자 등록해야 한다(`--append`).
- **whisper 코드를 지우지 않는다.** `mode: whisper` 가 기본으로 남아 있고, 임베딩 단독이 실기에서 무너지면 설정 한 줄로 되돌아간다.
