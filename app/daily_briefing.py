"""일정/대화 브리핑(가이드 4-1 과제4). 로그 요약 -> 부모 메시지.

민감 원본 대화 전체가 아니라 요약/결과만 전달(가이드 6).
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TEMPLATE = BASE / "reports" / "parent_message_template.md"
LOG = BASE / "reports" / "briefing_log.json"


def summarize_logs(conversation_log: list[dict]) -> dict:
    """대화 로그 -> {활동요약, 학습포인트, 주의사항(<=3)}."""
    raise NotImplementedError("LLM 요약 기반 브리핑 구현 예정")


def render_parent_message(summary: dict) -> str:
    raise NotImplementedError("템플릿 채우기 구현 예정")
