"""Vision: YOLOv8 / MediaPipe 로 사물·색상·위치 등 최소 정보만 추출.

가이드 4-3: 결과를 LLM에 그대로 넘기지 않고 상태값(예: red_ball_detected=true)으로 압축.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class VisionState:
    detected: dict = field(default_factory=dict)  # {"red_ball": True, ...}

    def as_context(self) -> str:
        if not self.detected:
            return "no_object_detected"
        return ", ".join(f"{k}={str(v).lower()}" for k, v in self.detected.items())


class VisionDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt") -> None:
        self.model_path = model_path
        self._model = None

    def load(self) -> None:
        raise NotImplementedError("YOLOv8/MediaPipe 로딩 구현 예정")

    def detect(self, frame) -> VisionState:
        """프레임 -> 압축된 VisionState. 실패해도 빈 상태 반환(대화 끊김 방지)."""
        return VisionState()
