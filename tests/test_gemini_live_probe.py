"""Gemini 실측기가 **거짓말을 안 하는지**.

OpenAI 쪽에서 배운 것을 그대로 적용한다: usage 모양이 바뀐 걸 눈치 못 채면 비용을
0 으로 보고하고, 그러면 "공짜네"라는 정반대 결론이 난다.

여기에 Gemini 만의 함정이 하나 더 있다 — **캐시 단가가 공개돼 있지 않다.**
캐시분을 0 원으로 세면 비용이 과소평가되고, 그대로 정가로 세면 과대평가된다.
어느 쪽이든 조용히 틀리면 안 되므로 캐시 토큰 수를 따로 드러낸다.
"""
import pytest
from google.genai import types

from tools.gemini_live_probe import (PRICE, accumulation_report, cached_tokens,
                                     cost_usd, score_rows, session_config)


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


class TestAccumulation:
    """한 세션에서 여러 턴을 돌 때, usage 가 누적인지 증분인지 모르면 합계가 틀린다."""

    @staticmethod
    def rows(prompts, responses):
        return [{"tokens": {"prompt": p, "response": r}, "cost_usd": 0.001}
                for p, r in zip(prompts, responses)]

    def test_응답이_단조증가하면_누적으로_보고_경고한다(self):
        # 🔴 누적인데 턴별로 더하면 비용이 몇 배로 부풀고, "S2S 는 못 쓴다"는
        #    정반대 결론이 난다. 응답 토큰은 턴마다 새로 만드는 것이라
        #    단조증가한다면 그 값은 세션 누계다.
        out = accumulation_report(self.rows([100, 220, 350], [40, 85, 130]))
        assert any("누적" in line for line in out)

    def test_응답이_들쭉날쭉하면_증분으로_본다(self):
        out = accumulation_report(self.rows([100, 220, 350], [40, 25, 60]))
        assert not any("누적" in line for line in out)

    def test_입력이_늘어나는지_보여준다(self):
        # 이력이 매 턴 다시 실리는지가 비용의 핵심이다. 숫자를 그대로 보여야 한다.
        out = "\n".join(accumulation_report(self.rows([100, 220, 350], [40, 25, 60])))
        assert "100" in out and "350" in out

    def test_한_턴이면_판정하지_않는다(self):
        assert accumulation_report(self.rows([100], [40])) == []


class TestScoring:
    """빈 전사와 오류는 다르다. 섞으면 '실패할수록 잘한다'가 된다."""

    def test_빈_전사는_100퍼센트로_센다(self):
        # 🔴 OpenAI 쪽에서 whisper-1 이 39% 확률로 빈 문자열을 줬다. 빈 것을 0% 로
        #    세면 아무것도 못 알아들은 모델이 만점을 받는다.
        s = score_rows([{"heard": "", "ref": "과자는달콤해요"}])
        assert s["cer"] == 100.0
        assert s["empty"] == 1

    def test_연결_오류는_채점에서_빼고_따로_센다(self):
        # 오류는 전사 실패가 아니다. 100% 로 세면 모델을 부당하게 깎고,
        # 0% 로 세면 부당하게 올린다. 둘 다 틀리므로 아예 뺀다.
        s = score_rows([{"heard": "과자는달콤해요", "ref": "과자는달콤해요"},
                        {"error": "ConnectionClosed", "ref": "무언가"}])
        assert s["cer"] == 0.0
        assert s["scored"] == 1 and s["errors"] == 1

    def test_공백과_문장부호는_양쪽에서_지운다(self):
        # AI-Hub 정답엔 문장부호가 없고 모델은 붙인다. 안 지우면 표기 차이가 오류가 된다.
        assert score_rows([{"heard": "과자는, 달콤해요!",
                            "ref": "과자는 달콤해요"}])["cer"] == 0.0

    def test_정답_태그는_속내용을_남긴다(self):
        # realtime_probe 와 같은 규칙이어야 09-02 결과와 비교가 된다.
        assert score_rows([{"heard": "바다를보고있떠어요",
                            "ref": "바다를 보고 있(SP:떠)어요"}])["cer"] == 0.0

    def test_완벽일치_개수를_센다(self):
        s = score_rows([{"heard": "가", "ref": "가"}, {"heard": "나", "ref": "다"}])
        assert s["exact"] == 1 and s["scored"] == 2
