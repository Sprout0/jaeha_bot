"""Realtime 실측기가 **거짓말을 안 하는지**.

이 도구의 결론 셋(빠르다 / 얼마다 / 아이 말을 알아듣는다)은 각각 다른 방식으로 무너진다:
- 말끝을 잘못 자르면 t_vad 가 0 으로 무너져 '엄청 빠르다'가 된다(실제로 8턴 중 4턴이 그랬다).
- 채점 규칙이 기존 결과와 다르면 로컬 medium 과의 비교가 통째로 무의미해진다.
- usage 모양이 바뀐 걸 눈치 못 채면 비용을 0 으로 보고한다.
"""
import numpy as np
import pytest

from tools.realtime_probe import (PRICE, SR, cer, clean_hyp, clean_ref, cost_usd,
                                  session_config, trim_edges, trim_speech)


def tone(dur_s, amp=0.5, sr=SR):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")


def silence(dur_s, sr=SR):
    return np.zeros(int(dur_s * sr), dtype="float32")


class TestTrim:
    def test_말끝에서_끊는다(self):
        # 🔴 급소. 뒤 무음을 남기면 서버 VAD 가 파일이 끝나기 전에 발화 종료를 잡아
        #    t_vad 가 0 이 되고, 도구가 '빠르다'고 거짓 보고한다.
        a = np.concatenate([silence(0.5), tone(0.8), silence(2.0)])
        out = trim_speech(a)
        assert 0.8 < out.size / SR < 1.1        # 말 0.8s + 앞뒤 pad 만

    def test_edges는_넉넉히_남긴다(self):
        # CER 측정용. 말끝을 깎으면 인식이 나빠져 비교가 불공정해진다.
        a = np.concatenate([silence(0.5), tone(0.8), silence(2.0)])
        assert trim_edges(a).size >= trim_speech(a).size

    def test_전부_무음이면_그대로_둔다(self):
        a = silence(1.0)
        assert trim_speech(a).size == a.size
        assert trim_edges(a).size == a.size

    def test_빈_배열도_안_터진다(self):
        empty = np.zeros(0, dtype="float32")
        assert trim_speech(empty).size == 0
        assert trim_edges(empty).size == 0


class TestScoring:
    def test_정답_태그는_속내용을_남긴다(self):
        # 153개 중 114개의 기록된 CER 이 이 규칙으로 재현된다. 버리는 규칙은 55개뿐이다.
        assert clean_ref("바다를 보고 있(SP:떠)어요") == "바다를보고있떠어요"

    def test_공백과_문장부호는_양쪽에서_지운다(self):
        # whisper 는 붙이고 AI-Hub 정답엔 없다. 안 지우면 표기 차이가 오류로 잡힌다.
        assert clean_hyp("엄마, 어디 갈 거예요?") == clean_ref("엄마 어디갈 거예요")

    def test_같으면_0(self):
        assert cer("과자는달콤해요", "과자는달콤해요") == 0.0

    def test_빈_전사는_100퍼센트다(self):
        # 🔴 빈 전사를 0% 로 세면 '실패할수록 잘한다'가 된다.
        assert cer("", "과자는달콤해요") == 100.0

    def test_정답이_비면_빈_전사만_0(self):
        assert cer("", "") == 0.0
        assert cer("무언가", "") == 100.0


class TestCost:
    USAGE = {"input_token_details": {"audio_tokens": 200, "text_tokens": 100,
                                     "cached_tokens_details": {"audio_tokens": 128}},
             "output_token_details": {"audio_tokens": 50, "text_tokens": 20}}

    def test_캐시된_오디오는_싼_값으로_센다(self):
        # 캐시분을 정가로 세면 비용이 부풀고, 빼먹으면 줄어든다. 둘 다 결정을 바꾼다.
        p = PRICE["gpt-realtime-mini"]
        want = ((200 - 128) * p["audio_in"] + 128 * p["cached"] + 100 * p["text_in"]
                + 50 * p["audio_out"] + 20 * p["text_out"]) / 1e6
        assert cost_usd("gpt-realtime-mini", self.USAGE) == pytest.approx(want)

    def test_모양이_낯설면_0이_아니라_None(self):
        # 🔴 스키마가 바뀌었을 때 조용히 $0 을 보고하면 "공짜네"라는 정반대 결론이 난다.
        assert cost_usd("gpt-realtime-mini", {"input_tokens": 300}) is None
        assert cost_usd("gpt-realtime-mini", {}) is None

    def test_모르는_모델은_None(self):
        assert cost_usd("gpt-4o-mini", self.USAGE) is None


class TestSessionConfig:
    def test_GA_스키마를_쓴다(self):
        # 베타의 평평한 input_audio_format/turn_detection 은 서버가 거부한다.
        s = session_config(1200, "whisper-1", "marin")["session"]
        assert s["type"] == "realtime"
        assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": SR}
        assert s["audio"]["output"]["format"]["rate"] == SR   # 빠뜨리면 서버가 거부한다

    def test_꼬리를_우리_값으로_줄_수_있다(self):
        assert session_config(1200, "whisper-1", "m")["session"]["audio"]["input"][
            "turn_detection"]["silence_duration_ms"] == 1200

    def test_한국어를_강제한다(self):
        assert session_config(500, "whisper-1", "m")["session"]["audio"]["input"][
            "transcription"]["language"] == "ko"
