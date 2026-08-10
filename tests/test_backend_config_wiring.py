"""configs/model_paths.yaml 의 백엔드 키가 생성자까지 닿는지.

main.build_pipeline() 은 설정을 그대로 `**` 로 펼쳐 넘긴다. 그래서 yaml 키 이름이
생성자 인자와 한 글자라도 어긋나면 **기동 즉시 TypeError** 로 죽는다 —
아이 앞에서 켰을 때 알게 될 종류의 실패라 여기서 잡는다.
값 자체도 확인한다(키만 맞고 엉뚱한 기본값이 실리면 조용히 로컬로 돈다).
"""
from app.agent import LLMAgent
from app.config import settings
from app.tts_module import TTSModule


def test_tts_config_keys_reach_constructor():
    tts = TTSModule(**settings.models["tts"])

    assert tts.backend in ("supertonic", "openai")
    assert tts.openai_voice, "목소리가 비면 API 호출이 실패한다"
    assert tts.openai_timeout > 0


def test_llm_config_keys_reach_constructor():
    cfg = dict(settings.models["llm"])
    cfg.pop("model_path")                      # main.build_pipeline 과 같은 방식
    agent = LLMAgent(model_path="dummy.gguf", system_prompt="테스트", **cfg)

    assert agent.backend in ("local", "openai")
    assert agent.api_model
    assert agent.api_timeout > 0


def test_defaults_shipped_in_repo_are_local():
    # 저장소 기본값은 로컬이어야 한다 — 받아서 켜자마자 과금되면 안 된다.
    assert settings.models["tts"]["backend"] == "supertonic"
    assert settings.models["llm"]["backend"] == "local"


def test_model_path_kept_for_fallback():
    # API 로 바꾸더라도 GGUF 경로는 남아 있어야 폴백이 가능하다.
    assert settings.models["llm"].get("model_path"), \
        "model_path 를 지우면 네트워크가 끊겼을 때 답할 수단이 없다"
