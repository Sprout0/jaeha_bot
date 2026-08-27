"""임베딩 대조 — whisper 가 지어낸 말 때문에 죽는 진짜 호출을 소리로 건진다.

## 왜 있나

2단계 검증은 whisper 가 받아쓴 **글자**를 호출어와 자모 비교한다. 그런데 whisper 는
자신 없는 짧은 음성을 받으면 **소리와 무관한 글을 지어낸다** — 학습 때 외운 유튜브
자막 문구다. 실기 로그에 나온 목록이 정확히 그것이다:

    안녕히계세요 · 고맙습니다 · 알겠습니다 · 다음 영상에서 만나요 · 오늘도 시청해주셔서 감사합니다

🔴 이건 '잘못 들었다'가 아니다. 증거가 디스크에 있다 —
   `data/wake_real/adult_20260826_1445/빠르게_00.wav` 는 어른이 '하이 티드'라고
   부른 걸 녹음한 파일인데 whisper 는 `안녕히계세요` 라고 적는다.

   ➡️ **자모컷을 어디로 옮겨도 못 고친다.** 글자가 소리와 무관하니 거리에 뜻이 없다.
      실제로 그 전사는 0.79 로, 진짜 소음들과 같은 자리에 있다.

## 무엇을 하나

v6 이 이미 쓰는 `embedding_model.onnx`(96차원)로 후보 구간을 벡터열로 만들고,
**미리 녹음해 둔 본보기**와 코사인 유사도를 잰다. 받아쓰기를 안 하므로 whisper 의
환각과 무관하고, 비용이 1.26초가 아니라 수 밀리초다.

whisper 를 **대체하지 않는다.** OR 이다 — whisper 가 통과시키면 지금처럼 깨우고,
whisper 가 기각해도 소리가 본보기와 맞으면 깨운다. 즉 재현율만 올리고 정밀도를
그만큼 내준다. 그 대가가 얼마인지는 **긴 소음 녹음으로만** 알 수 있다.

## 실측 (2026-08-27, 실음성 20건 + 거실 소음 3분)

등록 = 또박또박 5건, 시험 = **등록에 안 쓴 나머지 15건**. (화자 홀드아웃은 아니다 —
같은 사람이다. 그 한계는 아래 ⚠️ 참고.) 소음은 **1단계 후보가 뜬 자리만** 봤다.

    whisper 가 죽인 파일          전사            임베딩 유사도
    빠르게_00                    안녕히계세요        0.884
    흘려서_00                    화이팅             0.873
    흘려서_01                    한시들             0.884
    흘려서_02                    아주들             0.882
    흘려서_03                    알아듣듀           0.878
    흘려서_04                    안녕              0.842  <- 유일한 실패
    ─────────────────────────────────────────────────────
    거실 소음 3분, 후보 자리 최대치                   0.814

    컷 0.85 -> 진짜 **14/15**, 소음 0건
    같은 15건에 whisper(자모컷 0.45)는 **10/15** 였다.

⚠️ 여유가 얇다 — 소음 최대 0.814 ↔ 살린 것 중 최저 0.873, 폭이 0.059 뿐이다.

🔴🔴 **이 0.85 를 그대로 믿지 말 것.** "3분 소음에서 0건"은 이 프로젝트가 이미
   **두 번 속은 바로 그 증거**다(08-26 우회컷: 3분 최고 0.173 -> 같은 날 실기 0.725).
   컷은 반드시 **거실 30~60분 녹음**으로 다시 잡는다. `tools/enroll_wake.py --score-noise`.

⚠️ **화자 종속이다.** 위 숫자는 등록한 사람과 시험한 사람이 같다. 다른 사람이 부를 때도
   되는지는 **아직 아무도 재지 않았다** — 녹음이 한 사람 것뿐이라 잴 수가 없었다.
   ➡️ 봇을 실제로 쓸 사람이 자기 목소리로 등록하는 것을 기본으로 삼는다.

[[jaeha-bot-progress]]
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("jaeha_bot.wake_embed")

# 본보기 하나가 덮는 임베딩 개수. 분류기가 보는 창(EMB_WINDOW)과 같게 둔다 —
# 1단계가 '호출어 하나'로 보는 길이가 그 값이라, 여기서 다른 값을 쓰면 서로 다른
# 시간 폭을 비교하게 된다.
TEMPLATE_FRAMES = 16


def _flat_unit(embs: np.ndarray, i: int, k: int) -> np.ndarray:
    """임베딩 열의 [i, i+k) 창을 한 줄로 펴서 단위벡터로."""
    v = np.asarray(embs[i:i + k], dtype=np.float32).reshape(-1)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def make_template(embs, scores=None, k: int = TEMPLATE_FRAMES):
    """녹음 하나에서 본보기 벡터 하나를 뽑는다. 못 뽑으면 None.

    창을 고르는 기준은 **1단계 점수가 가장 높은 자리**다. 감지기가 실제로 반응하는
    지점과 본보기가 어긋나면 대조가 의미를 잃는다. 점수를 못 받으면(scores=None)
    에너지가 가장 큰 자리로 대신한다.
    """
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or len(embs) < k:
        return None
    last = len(embs) - k
    if scores is not None and len(scores) == len(embs):
        # 점수는 창의 **끝** 프레임에 붙는다(분류기가 직전 k개를 본다).
        tail = np.asarray(scores, dtype=np.float32)[k - 1:]
        i = int(np.argmax(tail))
    else:
        i = int(np.argmax([np.linalg.norm(embs[j:j + k]) for j in range(last + 1)]))
    return _flat_unit(embs, min(i, last), k)


def best_similarity(embs, templates, k: int = TEMPLATE_FRAMES) -> float:
    """임베딩 열 안에서 본보기들과 가장 잘 맞는 창의 코사인 유사도.

    창을 미끄러뜨리는 이유: 후보가 뜬 시각과 호출어의 실제 위치가 몇 프레임 어긋난다.
    """
    embs = np.asarray(embs, dtype=np.float32)
    T = np.asarray(templates, dtype=np.float32)
    if embs.ndim != 2 or len(embs) < k or T.size == 0:
        return 0.0
    if T.ndim == 1:
        T = T[None, :]
    if T.shape[1] != k * embs.shape[1]:
        # 본보기가 다른 규격(모델 교체·k 변경)으로 만들어졌다. 조용히 0 을 주면
        # '호출을 못 건졌다'와 구별이 안 되므로 반드시 말한다.
        log.warning("본보기 규격이 안 맞는다(%d != %d) — 임베딩 대조를 건너뛴다. "
                    "모델을 바꿨으면 tools/enroll_wake.py 로 다시 등록할 것",
                    T.shape[1], k * embs.shape[1])
        return 0.0
    return max(float(np.max(T @ _flat_unit(embs, i, k)))
               for i in range(len(embs) - k + 1))


class EmbedRescue:
    """본보기 묶음 + 컷. `passes(embs)` 하나만 쓰면 된다."""

    def __init__(self, templates, min_similarity: float) -> None:
        T = np.asarray(templates, dtype=np.float32)
        if T.ndim == 1:
            T = T[None, :]
        self.templates = T
        self.min_similarity = float(min_similarity)

    def __len__(self) -> int:
        return int(len(self.templates))

    def similarity(self, embs) -> float:
        return best_similarity(embs, self.templates)

    def passes(self, embs) -> tuple[bool, float]:
        s = self.similarity(embs)
        return s >= self.min_similarity, s


def save_templates(path, templates) -> None:
    from pathlib import Path
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), np.asarray(templates, dtype=np.float32))


def load_rescue(path, min_similarity: float):
    """설정의 경로에서 본보기를 읽어 EmbedRescue 를 만든다. 없으면 None.

    🔴 파일이 없으면 **경고를 남기고 None** 이다. 조용히 꺼지면 '켰는데 왜 그대로지'
       를 영영 못 알아낸다 — 이 프로젝트가 지연적재에서 똑같이 당한 적이 있다.
    """
    from pathlib import Path
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        log.warning("임베딩 본보기가 없다: %s — 대조를 못 쓴다. "
                    "tools/enroll_wake.py 로 먼저 등록할 것", p)
        return None
    try:
        T = np.load(str(p))
    except Exception as e:      # noqa: BLE001
        log.warning("임베딩 본보기를 못 읽었다(%s: %s) — 대조를 못 쓴다", type(e).__name__, e)
        return None
    if T.size == 0:
        log.warning("임베딩 본보기가 비어 있다: %s", p)
        return None
    log.info("임베딩 대조 켜짐 — 본보기 %d개, 컷 %.2f (%s)",
             len(T) if T.ndim > 1 else 1, min_similarity, p)
    return EmbedRescue(T, min_similarity)
