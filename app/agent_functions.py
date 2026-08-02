"""LLM Agent + Function Calling 함수 정의(가이드 4-3).

함수: start_game, ask_hint, check_answer, use_camera_context, generate_parent_briefing
실제 LLM 바인딩은 agent.py 에서. 여기서는 함수 본체와 스키마만 정의.
"""
from __future__ import annotations

# OpenAI/llama.cpp 호환 function-calling 스키마
FUNCTION_SCHEMAS = [
    {
        "name": "start_game",
        "description": "교육 놀이 모드를 시작한다.",
        "parameters": {
            "type": "object",
            "properties": {"mode": {"type": "string",
                                     "enum": ["sound_match", "color_play",
                                              "find_object", "repeat_word",
                                              "emotion_talk"]}},
            "required": ["mode"],
        },
    },
    {
        "name": "ask_hint",
        "description": "현재 놀이에서 아이에게 힌트를 준다.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "check_answer",
        "description": "아이의 답을 확인한다. 정답/오답 무관 긍정적으로 반응.",
        "parameters": {"type": "object",
                       "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]},
    },
    {
        "name": "use_camera_context",
        "description": "카메라 비전 상태값을 요청해 대화에 활용한다.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_parent_briefing",
        "description": "대화/놀이 로그를 요약해 부모 브리핑을 생성한다.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def start_game(mode: str) -> dict:
    return {"action": "start_game", "mode": mode}


def ask_hint() -> dict:
    return {"action": "ask_hint"}


def check_answer(answer: str) -> dict:
    return {"action": "check_answer", "answer": answer}


def use_camera_context() -> dict:
    # 상위에서 VisionDetector.detect() 결과를 주입
    return {"action": "use_camera_context"}


def generate_parent_briefing() -> dict:
    return {"action": "generate_parent_briefing"}
