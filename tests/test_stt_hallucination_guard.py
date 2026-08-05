"""반복 환각 가드 — 모델·마이크 불필요.

배경(2026-08-04 실측, AI-Hub #003 9세 148발화): whisper-small-ko 는 정답을 맞힌 '뒤'에
"일찍 일찌 일참 일찀 일찱…" 같은 학습문장 잔해를 끝없이 덧붙인다. 발화의 12.2% 가
CER>50% 로 무너졌고 최악은 CER 1200% 였다. no_repeat_ngram_size=3 은 매 3-gram 이
달라서, dedupe_repeats 는 매 토큰이 달라서 둘 다 못 잡는다.

폭주분은 '오디오 길이 대비 글자수'가 비정상적으로 크다는 공통점이 있어 그걸로 자른다.
(정답 길이비가 더 정확하지만 운영에선 정답을 모르므로 쓸 수 없다.)
"""
import sys
import types

import numpy as np
import pytest

from app.audio_source import FRAME
from app.stt_module import SAMPLE_RATE, STTModule


@pytest.fixture
def fake_whisper(monkeypatch):
    """model.transcribe 가 지정한 텍스트를 뱉도록 가로챈다."""
    holder = {"text": ""}

    class FakeSegment:
        def __init__(self, text):
            self.text = text

    class FakeWhisperModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, audio, **kw):
            return [FakeSegment(holder["text"])], None

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return holder


def _audio(seconds: float) -> np.ndarray:
    """무음이 아닌(=_trim_edges 가 통째로 날리지 않는) 오디오."""
    n = int(seconds * SAMPLE_RATE)
    return np.full(n, 0.5, dtype=np.float32)


# 2026-08-04 실측에서 실제로 나온 폭주 출력(CER 1200% 사례의 꼬리 부분).
# ⚠️ 같은 글자 반복("가"*60)으로 테스트하면 안 된다 — 그건 dedupe_repeats 가 이미
#    잡아내는 형태라 가드가 없어도 통과해 버려, 가드를 전혀 검증하지 못한다.
#    실제 폭주는 이렇게 '매번 다른 토큰'이라 dedupe 도 no_repeat_ngram 도 못 잡는다.
RUNAWAY = ("일찍 일찌 일참 일찀 일찱 일찻 일참만 일찷 일찿 일찬만 일곱 일초부터 "
           "일찜만 일정만 일찜 일찾 일찰만 일초나 일찌만 일주일 동안 일찡 일차리")


def test_runaway_output_is_rejected(fake_whisper):
    """3초 발화에 70글자 넘게 나오면 사람이 낼 수 없는 속도라 폭주로 본다."""
    fake_whisper["text"] = RUNAWAY
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    text, _ = stt.transcribe(_audio(3.0))

    assert text == "", "폭주 출력은 버려야 함(LLM 으로 넘기면 헛소리로 답한다)"


def test_dedupe_alone_does_not_catch_runaway(fake_whisper):
    """가드가 필요한 이유의 증거 — 기존 후처리만으로는 이 폭주가 그대로 통과한다.

    이 테스트가 실패하면(=dedupe 가 잡아버리면) 위 테스트는 가드를 검증하지 못하는
    가짜 테스트가 된다. 그래서 별도로 못박아 둔다.
    """
    fake_whisper["text"] = RUNAWAY
    stt = STTModule(model_size="tiny", max_chars_per_sec=None)

    text, _ = stt.transcribe(_audio(3.0))

    assert len(text) > 50, f"기존 후처리는 이걸 못 잡아야 정상(현재 {len(text)}자)"


def test_normal_speech_passes_through(fake_whisper):
    """3초에 12글자 = 4글자/초. 실측 정답 말속도 중앙값 3.03 이라 정상 범위."""
    fake_whisper["text"] = "우리 이모는 미국 살아"
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    text, _ = stt.transcribe(_audio(3.0))

    assert text == "우리 이모는 미국 살아"


def test_guard_disabled_when_threshold_is_none(fake_whisper):
    """기본값(None)이면 아무것도 거르지 않는다 — 옛 동작 그대로."""
    fake_whisper["text"] = RUNAWAY
    stt = STTModule(model_size="tiny", max_chars_per_sec=None)

    text, _ = stt.transcribe(_audio(3.0))

    assert text == RUNAWAY


def test_whitespace_is_not_counted(fake_whisper):
    """글자수는 공백 제외로 센다 — 띄어쓰기 습관이 임계를 흔들면 안 된다.

    공백 포함 31자지만 실제 글자는 16자(=5.3글자/초)라 통과해야 한다.
    공백까지 세면 10.3글자/초가 되어 잘못 버려진다.
    """
    text_in = "나 는 오 늘 학 교 에 서 친 구 랑 놀 았 어 요 오"  # 글자 16, 전체 31자
    fake_whisper["text"] = text_in
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    text, _ = stt.transcribe(_audio(3.0))

    assert text == text_in, "공백까지 세면 임계를 넘어 잘못 버려진다"


def test_short_output_is_never_rejected(fake_whisper):
    """짧은 출력은 말속도가 빨라도 버리지 않는다.

    왜 필요한가: 2세 발화는 짧다("이거 뭐야"). 1.2초에 9글자면 7.5글자/초라 임계를
    넘지만 정상 발화다. 반면 진짜 폭주는 본질적으로 길다 — 실측에서 폭주 출력의
    최소 글자수는 16, 그중 임계를 넘긴 것은 전부 20자 이상이었다.
    하한 20자를 두면 오탐이 4건 -> 2건으로 줄면서 진짜 검출은 14건 그대로였다.
    """
    fake_whisper["text"] = "이거 뭐야 궁금해"  # 공백 제외 8글자 / 1.0초 = 8글자per초
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0, min_chars_to_reject=20)

    text, _ = stt.transcribe(_audio(1.0))

    assert text == "이거 뭐야 궁금해", "짧은 발화는 빨라도 정상으로 봐야 함"
    assert stt.last_rejected is False


def test_long_runaway_still_rejected_with_floor(fake_whisper):
    """하한을 둬도 진짜 폭주(길다)는 여전히 걸러야 한다."""
    fake_whisper["text"] = RUNAWAY
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0, min_chars_to_reject=20)

    text, _ = stt.transcribe(_audio(3.0))

    assert text == ""


def test_rejection_is_flagged_for_reask(fake_whisper):
    """버린 사실을 상위(main)가 알아야 '다시 말해줄래?' 로 되물을 수 있다.

    그냥 빈 문자열만 돌려주면 main 은 '말이 없었다'와 구분하지 못해 침묵한다.
    2세 상대로 침묵은 최악의 응답이다.
    """
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    fake_whisper["text"] = RUNAWAY
    stt.transcribe(_audio(3.0))
    assert stt.last_rejected is True

    fake_whisper["text"] = "안녕"
    stt.transcribe(_audio(3.0))
    assert stt.last_rejected is False, "다음 정상 발화에서 플래그가 남아 있으면 안 됨"


class _SilentSource:
    """아무 말도 없는 마이크(모든 프레임이 무음) — 곧 프레임이 소진된다."""

    noise_floor = 0.001

    def __init__(self):
        self._left = 3

    def read(self):
        if self._left <= 0:
            raise StopIteration("프레임 소진")
        self._left -= 1
        return np.zeros(FRAME, dtype=np.float32)


def test_last_rejected_cleared_when_nothing_is_recorded(fake_whisper):
    """무음 턴은 직전 거부 상태를 반드시 지운다.

    listen() 은 녹음이 비면 transcribe() 를 아예 호출하지 않는다. 그래서 플래그를
    transcribe() 안에서만 초기화하면, 한 번 거부된 뒤로는 아이가 조용히 있어도
    main 이 계속 'reask' 로 읽어 **"다시 말해줄래?"를 무한 반복**하고 영영 안 잠든다.
    예외가 안 나므로 테스트로만 잡힌다.
    """
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    fake_whisper["text"] = RUNAWAY
    stt.transcribe(_audio(3.0))
    assert stt.last_rejected is True  # 전제 확인

    text, _ = stt.listen(source=_SilentSource())

    assert text == ""
    assert stt.last_rejected is False, "무음 턴인데 직전 거부가 남으면 되묻기가 무한 반복된다"


def test_guard_skipped_for_path_input(fake_whisper, tmp_path):
    """wav '경로'로 넣으면 길이를 모르니 가드를 적용하지 않는다(_trim_edges 와 같은 제약).

    실파이프라인은 numpy 를 넘기므로 무영향. 조용히 다르게 동작하는 것을 기록해 둔다.
    """
    fake_whisper["text"] = RUNAWAY
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    text, _ = stt.transcribe(str(tmp_path / "nonexistent.wav"))

    assert text == RUNAWAY
    assert stt.last_rejected is False


def test_empty_audio_does_not_crash(fake_whisper):
    """길이 0 오디오에서 0 나눗셈이 나면 안 된다."""
    fake_whisper["text"] = RUNAWAY
    stt = STTModule(model_size="tiny", max_chars_per_sec=6.0)

    text, _ = stt.transcribe(np.zeros(0, dtype=np.float32))

    assert isinstance(text, str)
