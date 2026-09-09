"""Gemini 실측기가 **거짓말을 안 하는지**.

OpenAI 쪽에서 배운 것을 그대로 적용한다: usage 모양이 바뀐 걸 눈치 못 채면 비용을
0 으로 보고하고, 그러면 "공짜네"라는 정반대 결론이 난다.

여기에 Gemini 만의 함정이 하나 더 있다 — **캐시 단가가 공개돼 있지 않다.**
캐시분을 0 원으로 세면 비용이 과소평가되고, 그대로 정가로 세면 과대평가된다.
어느 쪽이든 조용히 틀리면 안 되므로 캐시 토큰 수를 따로 드러낸다.
"""
import pytest
from google.genai import types

from tools.gemini_live_probe import (PRICE, cached_tokens, cost_usd,
                                     session_config)


def usage(text_in=0, audio_in=0, text_out=0, audio_out=0, cached=0):
    def det(mod, n):
        return [types.ModalityTokenCount(modality=mod, token_count=n)] if n else []
    return types.UsageMetadata(
        prompt_tokens_details=det("TEXT", text_in) + det("AUDIO", audio_in),
        response_tokens_details=det("TEXT", text_out) + det("AUDIO", audio_out),
        cached_content_token_count=cached or None,
    )


class TestCost:
    def test_모달리티마다_단가가_다르다(self):
        # 🔴 오디오와 텍스트를 같은 값으로 세면 4배까지 틀린다.
        m = "gemini-2.5-flash-native-audio-preview-12-2025"
        p = PRICE[m]
        got = cost_usd(m, usage(text_in=1000, audio_in=2000,
                                text_out=100, audio_out=500))
        want = (1000 * p["text_in"] + 2000 * p["audio_in"]
                + 100 * p["text_out"] + 500 * p["audio_out"]) / 1e6
        assert got == pytest.approx(want)

    def test_모양이_낯설면_0이_아니라_None(self):
        assert cost_usd("gemini-2.5-flash-native-audio-preview-12-2025",
                        types.UsageMetadata()) is None
        assert cost_usd("gemini-2.5-flash-native-audio-preview-12-2025", None) is None

    def test_모르는_모델은_None(self):
        assert cost_usd("gemini-9-nonexistent", usage(audio_in=100)) is None

    def test_모르는_모달리티는_세지_않는다(self):
        # VIDEO 는 우리가 안 쓴다. 단가를 모르면서 아무 값이나 붙이면 안 된다.
        m = "gemini-2.5-flash-native-audio-preview-12-2025"
        u = usage(audio_in=100)
        u.prompt_tokens_details.append(
            types.ModalityTokenCount(modality="VIDEO", token_count=9999))
        assert cost_usd(m, u) == pytest.approx(100 * PRICE[m]["audio_in"] / 1e6)


class TestCached:
    def test_캐시_토큰_수를_드러낸다(self):
        # 🔴 캐시 단가가 공개돼 있지 않다. 수를 숨기면 비용이 상한인지 실값인지
        #    읽는 사람이 알 수 없다.
        assert cached_tokens(usage(audio_in=500, cached=200)) == 200

    def test_캐시가_없으면_0(self):
        assert cached_tokens(usage(audio_in=500)) == 0
        assert cached_tokens(None) == 0


class TestSessionConfig:
    def test_한국어를_지정할_수_있다(self):
        # 🔴 언어를 안 주면 "하이 티드"를 'hated' / 'はい てる' 로 적는다(09-09 실측).
        #    OpenAI 쪽은 language:"ko" 를 줬으므로, 안 주고 비교하면 자가 다르다.
        c = session_config(1200, "지시문", lang="ko-KR")
        assert c["input_audio_transcription"]["language_codes"] == ["ko-KR"]

    def test_언어를_안_주면_비워_둔다(self):
        assert session_config(1200, "지시문")["input_audio_transcription"] == {}

    def test_꼬리를_우리_값으로_준다(self):
        c = session_config(1200, "지시문")
        assert c["realtime_input_config"]["automatic_activity_detection"][
            "silence_duration_ms"] == 1200
