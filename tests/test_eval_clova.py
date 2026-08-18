"""클로바 스튜디오(하이퍼클로바X) 백엔드. 네트워크·키 불필요.

CLOVA Studio 는 OpenAI 호환 엔드포인트를 준다
(https://clovastudio.stream.ntruss.com/v1/openai). 그래서 어댑터를 새로 쓰지 않고
make_openai 의 base_url·키만 갈아끼운다 — 채점·날조·안전 판정은 그대로 재사용된다.

⚠️ 모델 이름이 'HCX-' 로 시작하는 것으로 갈래를 판단한다(HCX-DASH-002, HCX-005).
"""
import pytest

from tools import eval_llm


class _FakeOpenAI:
    """OpenAI SDK 대역. 생성 인자를 잡아 둔다."""
    last_kwargs = None

    def __init__(self, **kw):
        _FakeOpenAI.last_kwargs = kw
        self.chat = self


def _patch(monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    _FakeOpenAI.last_kwargs = None


def test_hcx_model_uses_clova_endpoint(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setenv("CLOVA_STUDIO_KEY", "테스트키")
    eval_llm.make_openai("HCX-DASH-002", "시스템", 80)
    kw = _FakeOpenAI.last_kwargs
    assert kw["base_url"] == "https://clovastudio.stream.ntruss.com/v1/openai"
    assert kw["api_key"] == "테스트키"


def test_openai_model_untouched(monkeypatch):
    """기존 경로는 인자 없이 만들어야 한다 — 환경변수 관례를 깨면 안 된다."""
    _patch(monkeypatch)
    eval_llm.make_openai("gpt-4o-mini", "시스템", 80)
    assert _FakeOpenAI.last_kwargs == {}


def test_missing_clova_key_is_reported_not_silent(monkeypatch):
    """키가 없으면 조용히 기본 엔드포인트로 새면 안 된다.

    08-10 에 '.env 를 안 읽어 API 를 켰다고 믿는데 로컬로 폴백'하던 사고가 있었다.
    같은 종류의 조용한 실패를 막는다.
    """
    _patch(monkeypatch)
    monkeypatch.delenv("CLOVA_STUDIO_KEY", raising=False)
    with pytest.raises(SystemExit):
        eval_llm.make_openai("HCX-DASH-002", "시스템", 80)


def test_required_key_check_covers_hcx():
    """모델 이름 -> 필요한 환경변수 매핑에 HCX 가 들어 있어야 한다."""
    assert eval_llm.required_key("HCX-DASH-002") == "CLOVA_STUDIO_KEY"
    assert eval_llm.required_key("gpt-4o-mini") == "OPENAI_API_KEY"
    assert eval_llm.required_key("gemini-3.5-flash-lite") == "GEMINI_API_KEY"
    assert eval_llm.required_key("local") is None
