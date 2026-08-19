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
    """기존 경로는 키·엔드포인트를 손대지 않는다 — 환경변수 관례를 깨면 안 된다."""
    _patch(monkeypatch)
    eval_llm.make_openai("gpt-4o-mini", "시스템", 80)
    kw = _FakeOpenAI.last_kwargs
    assert "base_url" not in kw and "api_key" not in kw


@pytest.mark.parametrize("model", ["gpt-4o-mini", "HCX-DASH-002"])
def test_sdk_retry_is_off_so_latency_is_the_models_own(monkeypatch, model):
    """🔴 2026-08-19: 이 도구의 지연 숫자가 재시도로 오염돼 있었다.

    같은 날 같은 gpt-4o-mini 를 두 도구로 쟀는데:
      bench_llm_latency (재시도 끔, 간격 있음)  중앙 1.668s
      eval_llm          (재시도 기본값 2)       중앙 3.885s  (min 1.38 / p25 3.68)
    분포가 '빠른 소수 + 3.7~4.0 에 몰린 다수'로 갈라져 있었다. SDK 자동 재시도는
    create() 안에서 일어나므로 **실패한 시도 + 백오프가 측정값에 통째로 들어간다.**
    어제 app/agent.py 와 벤치에서는 껐는데 여기만 남아 있었다.

    ⚠️ 명시적 재시도(_ask_with_retry)는 그대로 둔다 — 그건 대기 시간을 빼고
       마지막 시도만 재므로 측정을 오염시키지 않는다. 끄는 건 **숨은** 재시도다.
    """
    _patch(monkeypatch)
    monkeypatch.setenv("CLOVA_STUDIO_KEY", "테스트키")
    eval_llm.make_openai(model, "시스템", 80)
    assert _FakeOpenAI.last_kwargs["max_retries"] == 0, f"{model} 에 숨은 재시도가 살아 있다"


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
