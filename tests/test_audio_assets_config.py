"""실제 configs/audio_assets.yaml 점검 — 목록과 나머지 프로젝트가 어긋나지 않게.

앞의 test_audio_library.py 는 임시 yaml 로 '로직'을 본다. 이 파일은 **진짜 설정**을
본다. 오타 하나(id 중복, file 경로 오기)가 나면 봇이 조용히 그 소리를 못 내는데,
그건 아이 앞에서만 드러난다.
"""
import json

from app.audio_player import default_library


def test_project_registry_loads():
    """yaml 이 깨졌거나 필수 키가 빠지면 여기서 잡힌다."""
    lib = default_library()
    assert lib.songs and lib.sounds


def test_ids_are_unique():
    """id 가 겹치면 get() 이 먼저 것만 돌려줘 뒤엣것이 영영 안 나온다."""
    lib = default_library()
    ids = [a.id for a in lib.songs + lib.sounds]
    assert len(ids) == len(set(ids))


def test_asset_ids_and_filenames_are_ascii():
    """한글 파일명·id 는 scp 로 젯슨에 보낼 때 인코딩이 깨진다."""
    lib = default_library()
    for a in lib.songs + lib.sounds:
        assert a.id.isascii(), f"id 가 ASCII 가 아님: {a.id}"
        assert a.path.name.isascii(), f"파일명이 ASCII 가 아님: {a.path.name}"


def test_sound_effects_cover_every_animal_in_the_game():
    """🔴 동물소리 놀이 카드와 효과음 목록이 1:1이어야 한다.

    어긋나면 놀이가 "소 소리 들려줄게!"라고 해놓고 낼 소리가 없다.
    카드를 늘릴 때 효과음을 같이 안 늘리는 실수를 여기서 잡는다.
    """
    cards = json.load(open("scenarios/scenario_cards.json", encoding="utf-8"))
    animals = {c["answer"] for c in cards["sound_match"]["cards"]}
    titles = {a.title for a in default_library().sounds}
    assert animals == titles


def test_registry_records_a_license_decision_for_every_asset():
    """⚠️ 시연·논문에 들어간다. 라이선스 판단 없이 음원을 넣지 못하게 강제한다."""
    lib = default_library()
    for a in lib.songs + lib.sounds:
        assert a.license, f"license 미기재: {a.id}"
