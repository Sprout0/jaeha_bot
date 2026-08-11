"""테스트가 진짜 네트워크를 타지 못하게 막는다.

2026-08-11 에 실제로 뚫렸다 — `agent.warm()` 에 API 커넥션 예열을 넣었더니,
`fake_openai` 를 안 받은 테스트 3개가 **진짜 OpenAI 를 호출**했다(노트북에 키가
환경변수로 있어서 성공까지 했다). 스위트가 1.4s -> 9s 로 느려진 걸로 겨우 알아챘다.

🔴 일반 예외로 막으면 안 된다. 프로덕션 코드가 예열·폴백 실패를 의도적으로 삼키므로
   (`except Exception` -> 경고 로그), 가드가 Exception 이면 테스트는 **조용히 통과**한다.
   그래서 BaseException 을 상속해 except Exception 을 통과해 나가게 한다.

가짜가 필요한 테스트는 자기 픽스처로 `sys.modules["openai"]` 를 덮어쓴다(그쪽이 이긴다).
"""
import sys
import types

import pytest


class NetworkBlockedInTest(BaseException):
    """테스트에서 실제 네트워크를 시도했다. BaseException 상속은 의도적이다(위 설명 참조)."""


@pytest.fixture(autouse=True)
def block_real_openai(monkeypatch):
    """`openai` 를 가짜로 갈아끼운다. 진짜 클라이언트를 만들려 하면 즉시 터진다."""

    def _blocked(*args, **kwargs):
        raise NetworkBlockedInTest(
            "테스트가 실제 OpenAI 를 호출하려 했다. fake_openai 픽스처를 받거나, "
            "네트워크를 타지 않는 경로로 바꿀 것."
        )

    mod = types.ModuleType("openai")
    mod.OpenAI = _blocked
    monkeypatch.setitem(sys.modules, "openai", mod)
