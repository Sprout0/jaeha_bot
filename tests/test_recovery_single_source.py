"""되묻기 문구는 설정 한 곳에서만 온다. 모델·마이크 불필요.

2026-08-11 리뷰: 스펙이 요구한 `recovery` 변경이 configs/prompt_templates.yaml 에만
들어갔는데, 운영(app/main.py)은 하드코딩한 복사본을 쓰고 있었다. 설정을 고쳐도
봇이 옛 문구를 그대로 말하는, 눈에 안 띄는 종류의 사고다.

문구 자체에도 조건이 있다. 호출부 하나는 **STT 결과를 환각으로 버린 직후**라
봇은 아무 말도 듣지 못한 상태다 — 아이 말에 맞장구치는 문구("그렇구나!")를 쓰면
하지 않은 말에 반응하는 셈이고, 특정 놀이 낱말('강아지')을 넣으면 따라 말하기
놀이가 돌아가는 중에 정답을 흔든다.
"""
from app.config import settings
from app.main import SAFE_RECOVERY


def test_main_reads_recovery_from_config():
    assert SAFE_RECOVERY == settings.prompts["recovery"]


def test_recovery_does_not_acknowledge_unheard_speech():
    # 아무 말도 못 들은 자리에서 쓰는 문구다.
    for echo in ("그렇구나", "맞아", "그랬구나"):
        assert echo not in SAFE_RECOVERY, SAFE_RECOVERY


def test_recovery_is_game_agnostic():
    # 놀이 카드의 낱말이 섞이면 진행 중인 놀이를 흔든다.
    from app.education_modes import ANIMAL_ITEMS, REPEAT_ITEMS

    words = [a for a, _ in ANIMAL_ITEMS] + [w for w, _ in REPEAT_ITEMS]
    for w in words:
        assert w not in SAFE_RECOVERY, f"놀이 낱말이 들어갔다: {w} / {SAFE_RECOVERY}"


def test_recovery_is_speakable():
    # TTS 로 그대로 읽힌다 — 기호가 섞이면 소리로 읽어버린다.
    for bad in ("*", "#", "-", "...", "…"):
        assert bad not in SAFE_RECOVERY, SAFE_RECOVERY
