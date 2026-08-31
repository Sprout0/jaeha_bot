"""조립된 system 프롬프트가 로컬 폴백의 n_ctx 안에 들어가는지.

🔴 이걸 안 지켜서 2026-08-11~18 사이 **로컬 폴백이 죽어 있었다.**
   08-11 에 안전·품질 회귀를 잡느라 프롬프트를 1550 -> 3386자로 키웠고,
   `n_ctx: 2048` 은 그대로였다. 젯슨에서 이렇게 터졌다:
       ValueError: Requested tokens (2847) exceed context window of 2048
   경고 로그로만 나가는 자리(`agent.warm()` 의 폴백 예열)라 **아무도 못 봤다** —
   인터넷이 끊기는 날에야 '재하봇이 아무 말도 안 한다'로 발견될 종류였다.

여기서 지키려는 것: 프롬프트를 키우는 사람이 **그 커밋에서 바로** 알게 하는 것.
프롬프트는 앞으로도 커진다(안전 규칙은 계속 붙는다). 커지는 게 문제가 아니라
n_ctx 를 같이 안 올리는 게 문제다.

⚠️ 토큰 수는 GGUF 토크나이저 없이는 정확히 못 센다(노트북엔 모델이 없다).
   그래서 **글자당 토큰 상한**으로 보수적으로 추정한다. 젯슨 실측(EXAONE-3.5 토크나이저)은
   4185자 = 2356토큰 = 0.563 토큰/자였다. 여기선 0.7 을 쓴다(24% 여유).
   추정이 실제보다 크게 나오므로, 이 테스트가 통과하면 실제로도 들어간다.
"""
from app.agent import MAX_HISTORY_TURNS, LLMAgent
from app.config import settings

# 젯슨 실측 0.563 토큰/자에 여유를 얹은 상한. 실제보다 크게 잡아야 안전하다.
TOKENS_PER_CHAR = 0.7
# 이력 한 메시지가 이보다 길어질 일은 없다(아이 발화는 짧고 답변은 2문장 클램프).
MAX_CHARS_PER_HISTORY_MSG = 100


def _est_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


def _assembled_system() -> str:
    """운영이 **매 턴 실제로 보내는** 앞머리(system + few-shot)를 그대로 조립한다.

    🔴 예전엔 여기서 `_augment_system` 을 직접 불렀다. 그러면 few-shot 을 담는 그릇을
       바꾼 날(fewshot_form) 이 시험만 옛 형식을 재고 통과한다 — 지키려던 것을 놓친다.
       그래서 운영 코드(_build_messages)를 거쳐 조립한다.
    ⚠️ backend 만 local 로 고정한다. 원격 키 없는 기계에서도 예산은 재야 하고,
       앞머리 조립은 backend 와 무관하다.
    """
    cfg = dict(settings.models["llm"])
    cfg.pop("model_path", None)
    cfg.update(backend="local", api_model="gpt-4o-mini")
    agent = LLMAgent(model_path="없어도 된다.gguf",
                     system_prompt=settings.prompts["system"], **cfg)
    prefix = agent._build_messages("", "")[:-1]   # 마지막은 이번 아이 말이라 뺀다
    return "".join(m["content"] for m in prefix)


def test_system_prompt_fits_in_n_ctx():
    n_ctx = settings.models["llm"]["n_ctx"]
    est = _est_tokens(_assembled_system())

    assert est < n_ctx, (
        f"system 프롬프트 추정 {est}토큰이 n_ctx {n_ctx} 를 넘는다 — "
        f"로컬 폴백이 죽는다. 프롬프트를 줄이거나 n_ctx 를 올릴 것."
    )


def test_n_ctx_leaves_room_for_history_and_answer():
    """프롬프트만 겨우 들어가면 안 된다. 대화 이력과 답변 자리가 남아야 한다.

    system 만 재면 첫 턴은 통과하고 **여섯 턴째에** 죽는다 — 놀이 도중에.
    """
    cfg = settings.models["llm"]
    n_ctx = cfg["n_ctx"]
    system = _est_tokens(_assembled_system())
    history = _est_tokens("x" * MAX_CHARS_PER_HISTORY_MSG) * MAX_HISTORY_TURNS * 2
    answer = cfg["max_tokens"]

    need = system + history + answer
    assert need <= n_ctx, (
        f"system {system} + 이력 {history}({MAX_HISTORY_TURNS}턴) + 답변 {answer} "
        f"= {need}토큰 > n_ctx {n_ctx}. 대화가 길어지면 폴백이 죽는다."
    )


def test_agent_default_n_ctx_matches_config():
    """생성자 기본값이 설정보다 작으면, 설정을 안 넘기는 경로(도구·테스트)가 조용히 죽는다."""
    import inspect

    default = inspect.signature(LLMAgent.__init__).parameters["n_ctx"].default
    assert default >= _est_tokens(_assembled_system()), (
        f"LLMAgent 기본 n_ctx={default} 로는 현재 프롬프트가 안 들어간다"
    )
