"""few-shot 예시가 우리 자신의 날조 판정기에 걸리면 안 된다.

2026-08-11 에 실제로 걸렸다: 대체 예시의 "곰돌이"가 audio_assets.yaml 에서
'곰 세 마리'의 별칭이고 음원 파일이 없어서
  find_fabrications("곰돌이가 자는구나! 코 자자.") == ['곰 세 마리']
가 됐다. 이 프로젝트 기록상 말투는 예시가 지배한다 — 즉 '못 트는 노래' 토큰을
프롬프트가 먼저 심고, 그 결과가 날조율로 되돌아온다(자기 오염).

여기서 막으면 새 예시를 넣을 때 자동으로 걸린다.
"""
import json
from pathlib import Path

import pytest

from app.claims import find_fabrications

_SEED = Path(__file__).resolve().parent.parent / "data" / "finetune_seed.jsonl"


def _rows() -> list[dict]:
    out = []
    for line in _SEED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def _replies() -> list[tuple[str, str]]:
    """(아이 말, 봇 답) 목록. few_shot 여부와 무관하게 전부 본다 —
    지금은 주입 안 되는 줄도 나중에 QLoRA 학습 데이터가 된다."""
    pairs = []
    for row in _rows():
        msgs = row["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        for m in msgs:
            if m["role"] == "assistant":
                pairs.append((user, m["content"]))
    return pairs


@pytest.mark.parametrize("ask,reply", _replies())
def test_fewshot_reply_is_fabrication_clean(ask, reply):
    assert find_fabrications(reply) == [], \
        f"예시가 못 하는 것을 약속한다 ({ask!r} -> {reply!r})"
