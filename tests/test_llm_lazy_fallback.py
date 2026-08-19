"""폴백 EXAONE 을 기동 때 올리지 않는다 — 젯슨 메모리 2.3GB 를 되찾기 위해.

🔴 왜 (2026-08-19 젯슨 실측): 앱 RSS 6,259MB 중 **EXAONE GGUF 가 2,319MB(37%)** 다.
   그런데 `backend: openai` 라 網이 죽지 않는 한 한 번도 안 쓰인다. 세션 시스템
   메모리가 7,348/7,607MB(96.6%)까지 차서 다음 최적화(선행 TTS 합성)를 넣을 자리가
   없다. 되찾을 수 있는 유일한 큰 덩어리가 여기다.

   | 항목 | 증가 |
   |---|---:|
   | whisper medium (CUDA) | 1,881MB |
   | Supertonic (CUDA) | 1,321MB |
   | **EXAONE GGUF 폴백** | **2,319MB** |

⚠️ 그냥 안 올리면 網이 끊긴 **그 턴**에 콜드 폴백 6~7초를 문다(08-10 실측 7.18s).
   그래서 기동 때 API 를 한 번 찔러 보고 **원격이 안 되면 그때는 올린다** —
   원격이 이미 죽은 상태로 켜졌다면 폴백이 곧 필요하다는 뜻이기 때문이다.
   원격이 정상이면 안 올리고, 운영 중 실패하는 첫 턴에만 대가를 치른다.
"""
from tests.test_agent_api_backend import (  # noqa: F401  (픽스처 재사용)
    LOCAL_REPLY, _agent, fake_llama, fake_openai,
)


def test_remote_healthy_means_the_gguf_stays_off_memory(fake_llama, fake_openai):
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert fake_llama["loaded"] == 0, "원격이 멀쩡한데 폴백 2.3GB 를 올렸다"
    assert len(fake_openai["calls"]) == 1, "원격 커넥션 예열은 그대로 해야 한다"


def test_remote_dead_at_boot_loads_the_fallback_now(fake_llama, fake_openai):
    """🔴 이게 이 설계의 핵심이다.

    網 없이 켜졌다 = 폴백이 곧 필요하다. 그때까지 미루면 아이의 **첫 질문**이
    콜드 로드 7초를 문다. 기동 때 치르면 아이는 못 느낀다.
    """
    fake_openai["fail"] = RuntimeError("no network")
    agent = _agent(fake_llama, backend="openai")

    agent.warm()

    assert fake_llama["loaded"] == 1, "원격이 죽었는데도 폴백을 안 올렸다"
    assert fake_llama["calls"], "로드만으로는 부족하다 — 추론을 한 번 돌려야 한다"


def test_local_backend_always_preloads(fake_llama, fake_openai):
    """backend: local 은 폴백이 아니라 본진이다. 미루면 첫 대답이 7초가 된다."""
    _agent(fake_llama).warm()

    assert fake_llama["loaded"] == 1


def test_preload_can_be_forced_back_on(fake_llama, fake_openai):
    """메모리가 넉넉한 기계에서는 예전처럼 미리 데울 수 있어야 한다."""
    agent = _agent(fake_llama, backend="openai", preload_local=True)

    agent.warm()

    assert fake_llama["loaded"] == 1


def test_the_fallback_still_works_when_it_was_never_preloaded(fake_llama, fake_openai):
    """안 올려 뒀어도 원격이 죽으면 그 자리에서 올라와 답해야 한다.

    이게 깨지면 메모리를 아끼려다 **網이 끊긴 순간 봇이 벙어리가 된다** —
    가이드라인 1 위반이다.
    """
    agent = _agent(fake_llama, backend="openai")
    agent.warm()
    assert fake_llama["loaded"] == 0

    fake_openai["fail"] = RuntimeError("connection reset")
    reply = agent.respond("안녕")["text"]

    assert reply == LOCAL_REPLY
    assert fake_llama["loaded"] == 1, "그 자리에서 올라와야 한다"


def test_it_loads_only_once_even_after_repeated_outages(fake_llama, fake_openai):
    agent = _agent(fake_llama, backend="openai")
    agent.warm()
    fake_openai["fail"] = RuntimeError("connection reset")

    agent.respond("안녕")
    agent.respond("또 안녕")

    assert fake_llama["loaded"] == 1, "실패할 때마다 다시 로드하면 매 턴 7초다"
