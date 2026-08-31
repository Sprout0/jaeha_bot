"""오디오 에셋 레지스트리 — 봇이 '실제로 낼 수 있는 소리'의 목록.

왜 코드가 아니라 레지스트리인가:
동요는 TTS 로 부를 수 없다(읽을 뿐이다). 동물 소리도 "멍멍"을 읽는 것과 진짜
개 짖는 소리는 다르다. 그래서 노래·효과음은 **음원 파일**이고, 이 파일에 무엇이
있는지가 곧 봇의 능력이다.

🔴 이 레지스트리는 '현재활동 날조'(측정 15%)의 구조적 해결책이기도 하다.
지금은 프롬프트가 "색칠, 사물 찾기"를 광고하는데 실물이 없어서 모델이 없는 놀이를
지어낸다. 목록이 데이터로 존재하면 나중에 이걸 그대로 LLM 툴 스키마의 enum 으로
쓸 수 있고, 그러면 **없는 노래는 부를 방법 자체가 사라진다.** 프롬프트로 "지어내지
마"라고 부탁하는 것과는 다르다.
"""
import textwrap

import numpy as np
import pytest
import soundfile as sf

from app.audio_player import AudioLibrary, AudioPlayer

_YAML = """\
songs:
  - id: gom_three
    title: 곰 세 마리
    file: songs/gom_three.wav
    aliases: [곰세마리, 곰돌이, 세마리]
sounds:
  - id: dog
    title: 강아지
    file: sounds/dog.wav
    aliases: [멍멍이, 멍멍]
  - id: cat
    title: 고양이
    file: sounds/cat.wav
    aliases: [야옹이]
"""


def _wav(path, seconds=0.2, rate=16000):
    """진짜 wav 파일을 만든다(모킹 아님) — 로드 경로까지 실제로 태우기 위해."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    sf.write(path, np.sin(2 * np.pi * 440 * t) * 0.2, rate)


@pytest.fixture
def assets(tmp_path):
    """레지스트리 3개 중 2개만 파일이 실재한다(cat 은 일부러 없음)."""
    (tmp_path / "audio_assets.yaml").write_text(textwrap.dedent(_YAML), encoding="utf-8")
    _wav(tmp_path / "songs" / "gom_three.wav")
    _wav(tmp_path / "sounds" / "dog.wav")
    return tmp_path


@pytest.fixture
def lib(assets):
    return AudioLibrary.load(assets / "audio_assets.yaml", assets)


# ── 로드 ──────────────────────────────────────────────────────────────────────
def test_registry_separates_songs_from_sounds(lib):
    """노래와 효과음은 쓰임이 다르다(노래=길다·중단 가능, 효과음=짧다·블로킹)."""
    assert [a.id for a in lib.songs] == ["gom_three"]
    assert [a.id for a in lib.sounds] == ["dog", "cat"]


def test_asset_carries_title_and_resolved_path(lib, assets):
    asset = lib.get("gom_three")
    assert asset.title == "곰 세 마리"
    assert asset.path == assets / "songs" / "gom_three.wav"


def test_get_unknown_id_returns_none(lib):
    assert lib.get("없는아이디") is None


# ── find: 아이 발화에서 찾기 ──────────────────────────────────────────────────
def test_find_by_title_inside_a_sentence(lib):
    """아이는 '곰 세 마리'라고만 하지 않는다. '곰 세 마리 틀어줘'라고 한다."""
    assert lib.find("곰 세 마리 틀어줘").id == "gom_three"


def test_find_by_alias(lib):
    """정식 제목을 모르는 나이다. '곰돌이 노래'로도 찾아야 한다."""
    assert lib.find("곰돌이 노래 불러줘").id == "gom_three"


def test_find_tolerates_stt_mangling(lib):
    """2세 발음 + STT 오인식으로 글자가 뭉개져도 찾아야 한다.

    자모 편집거리 임계는 education_modes.match_trigger 와 같은 0.2 를 쓴다
    (같은 문제를 두 가지 기준으로 풀면 놀이마다 인식률이 달라진다).
    """
    assert lib.find("곰새마리").id == "gom_three"


def test_find_returns_none_for_song_not_in_registry(lib):
    """🔴 핵심: 없는 노래는 '없다'고 해야 한다.

    여기서 아무거나 비슷한 걸 돌려주면 아이가 '상어가족'을 달라 했는데 '곰 세 마리'가
    나온다. 유아 대상에선 엉뚱한 재생이 곧 사고다. 날조를 코드가 대신 하는 셈.
    """
    assert lib.find("상어가족 틀어줘") is None


def test_find_returns_none_for_unrelated_speech(lib):
    assert lib.find("배고파 밥 줘") is None


def test_find_can_be_limited_to_one_kind(lib):
    """'강아지 노래'를 요청했는데 강아지 '효과음'이 나오면 안 된다."""
    assert lib.find("강아지", kind="song") is None
    assert lib.find("강아지", kind="sound").id == "dog"


# ── 기동 점검 ─────────────────────────────────────────────────────────────────
def test_missing_lists_registered_assets_without_files(lib):
    """레지스트리에만 있고 파일이 없는 항목 — 기동 때 걸러야 한다.

    이걸 안 하면 봇이 "고양이 소리 들려줄게!"라고 말한 뒤 아무 소리도 안 난다.
    아이에게는 약속을 어긴 것과 같다.
    """
    assert [a.id for a in lib.missing()] == ["cat"]


def test_playable_excludes_missing_files(lib):
    """LLM 에 넘길 목록은 '파일이 실재하는 것'만이어야 한다."""
    assert [a.id for a in lib.playable("sound")] == ["dog"]


def test_titles_for_prompt_lists_only_playable(lib):
    """나중에 툴 스키마 enum / 프롬프트 주입에 그대로 쓰는 목록."""
    assert lib.titles("song") == ["곰 세 마리"]
    assert lib.titles("sound") == ["강아지"]


# ── 재생 ──────────────────────────────────────────────────────────────────────
class FakeSink:
    """실제 스피커 대신. 마이크·스피커 없는 노트북/CI 에서도 테스트가 돌아야 한다."""

    def __init__(self):
        self.calls = []
        self.stopped = 0
        self.playing = False

    def play(self, samples, rate, block):
        self.calls.append({"n": len(samples), "rate": rate, "block": block})
        self.playing = not block

    def stop(self):
        self.stopped += 1
        self.playing = False

    @property
    def is_playing(self):
        return self.playing


def test_play_sends_decoded_samples_to_device(lib):
    sink = FakeSink()
    player = AudioPlayer(lib, sink=sink)

    assert player.play("dog") is True
    assert sink.calls[0]["n"] > 0
    assert sink.calls[0]["rate"] == 16000


def test_short_sound_blocks_and_song_does_not(lib):
    """효과음(1~2초)은 끝까지 기다려도 되지만, 동요(2~3분)를 블로킹하면
    그동안 아이 말을 못 듣는다 — '그만'이라고 해도 못 멈춘다."""
    sink = FakeSink()
    player = AudioPlayer(lib, sink=sink)

    player.play("dog")
    player.play("gom_three")

    assert sink.calls[0]["block"] is True    # 효과음
    assert sink.calls[1]["block"] is False   # 노래


def test_missing_file_returns_false_without_raising(lib):
    """🔴 파일이 없어도 예외를 던지면 안 된다 — 봇 전체가 죽는다.

    호출어 감지 실패가 봇을 죽이지 않는 것과 같은 원칙(README '설계에서 알아둘 것').
    """
    sink = FakeSink()
    player = AudioPlayer(lib, sink=sink)

    assert player.play("cat") is False
    assert sink.calls == []


def test_unknown_id_returns_false_without_raising(lib):
    player = AudioPlayer(lib, sink=FakeSink())
    assert player.play("없는노래") is False


def test_stop_halts_playback(lib):
    sink = FakeSink()
    player = AudioPlayer(lib, sink=sink)

    player.play("gom_three")
    assert player.is_playing is True
    player.stop()

    assert sink.stopped == 1
    assert player.is_playing is False


def test_new_playback_stops_previous_one(lib):
    """노래 도중 다른 노래를 요청하면 겹쳐 나오면 안 된다."""
    sink = FakeSink()
    player = AudioPlayer(lib, sink=sink)

    player.play("gom_three")
    player.play("gom_three")

    assert sink.stopped == 1


# ── find 오탐: 짧은 이름이 평범한 문장에 걸린다 (2026-08-31) ────────────────────
# 실제 레지스트리(configs/audio_assets.yaml)에서 측정한 네 건이다. 여기 임시 yaml 은
# 그때 걸린 이름('소'·'양'·'하루')만 그대로 옮겨 담아 같은 로직을 태운다.
#
# 🔴 지금 당장 안 터지는 이유는 놀이/에이전트가 '노래 요청'이라 판단한 뒤에만 find()
#    를 부르기 때문이다. 자유대화에 물리는 순간 아이 말 아무거나 노래로 바뀐다.
#    claims.find_fabrications 는 같은 병을 _MIN_NAME(짧은 이름 무시)과 겹침 제거로
#    막아 뒀는데 find() 에는 가드가 하나도 없었다.
_TRAP_YAML = """\
songs:
  - id: one_day
    title: 하루
    file: songs/one_day.mp3
    aliases: [하루노래]
sounds:
  - id: cow
    title: 소
    file: sounds/cow.wav
    aliases: [음메, 젖소]
  - id: sheep
    title: 양
    file: sounds/sheep.wav
    aliases: [매애, 양이]
"""


@pytest.fixture
def trap(tmp_path):
    """짧은 이름만 담은 레지스트리. 파일은 없어도 된다 — find() 는 실재 여부를 안 본다."""
    p = tmp_path / "trap.yaml"
    p.write_text(textwrap.dedent(_TRAP_YAML), encoding="utf-8")
    return AudioLibrary.load(p, tmp_path)


def test_one_letter_name_inside_a_longer_word_is_not_a_request(trap):
    """'소'가 '소리' 안에 들어 있다고 소 울음을 틀면 안 된다."""
    assert trap.find("무슨 소리야", "sound") is None


def test_one_letter_name_as_a_whole_word_beats_one_buried_in_another(trap):
    """'양 소리' 는 양이다. 등록 순서가 앞선다는 이유로 '소'(cow)가 이기면 안 된다."""
    assert trap.find("양 소리", "sound").id == "sheep"


def test_one_letter_name_is_not_matched_inside_an_everyday_word(trap):
    """'양말'의 '양'은 양이 아니다."""
    assert trap.find("양말 신자", "sound") is None


def test_everyday_word_that_happens_to_be_a_song_title(trap):
    """제목 자체가 일상어인 경우('하루'). 문장 속에 그냥 나온 것은 요청이 아니다."""
    assert trap.find("오늘 하루 어땠어") is None


def test_short_title_still_works_when_it_is_actually_requested(trap):
    """오탐만 막고 기능은 남긴다 — 요청 단서가 붙으면 짧은 이름도 찾아야 한다."""
    assert trap.find("하루 틀어줘").id == "one_day"
    assert trap.find("하루 노래 들려줘").id == "one_day"


def test_bare_short_name_is_still_a_request(trap):
    """발화 전체가 이름 하나뿐이면 헷갈릴 여지가 없다."""
    assert trap.find("양", "sound").id == "sheep"


def test_particles_attached_to_the_name_still_match(trap):
    """한국어는 이름에 조사가 붙어 한 낱말이 된다('하루는'). 낱말 경계가 너무 빡빡하면
    이걸 놓친다 — '오리기'는 막되 '하루는'은 받아야 한다."""
    assert trap.find("하루는 틀어줘").id == "one_day"
