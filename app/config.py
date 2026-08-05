"""설정 로더. configs/*.yaml 을 읽어 단일 객체로 제공한다.

⚠️ `model_paths.yaml` 은 **젯슨(운영) 기준값**이다. push_code.sh 가 configs/ 를 통째로
   젯슨에 밀어넣으므로 두 기계가 이 파일 한 벌을 공유하는데, 기계마다 정반대여야 하는
   항목이 있다(노트북 ctranslate2 는 CPU 전용 휠이라 `stt.device: cuda` 면 죽는다,
   `metrics.tag` 는 PC/젯슨 로그 구분용).
   → `configs/local.yaml` 이 있으면 그 값만 덧씌운다. 이 파일은 git 미추적이고
     push_code.sh 전송에서도 제외되므로 기계마다 다르게 둘 수 있다.
     예시는 `configs/local.example.yaml`.
"""
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
LOCAL_NAME = "local.yaml"


def load_yaml(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or CONFIG_DIR) / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, over: dict | None) -> dict:
    """over 의 값을 base 에 재귀적으로 덮어쓴다(base 를 제자리 수정).

    얕은 병합(update)이면 `stt: {device: cpu}` 한 줄이 stt 섹션을 통째로 갈아치워
    model_size·가드 임계까지 사라진다. 그래서 dict 끼리는 파고들고, 그 외(스칼라·
    리스트)는 교체한다. 리스트를 '합치기'로 두면 providers 를 줄이려는 의도가 안 먹는다.
    """
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_models(config_dir: Path | None = None) -> dict:
    """model_paths.yaml(운영 기준) + local.yaml(이 기계 전용 덮어쓰기)."""
    cfg = load_yaml("model_paths.yaml", config_dir)
    local = (config_dir or CONFIG_DIR) / LOCAL_NAME
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            deep_merge(cfg, yaml.safe_load(f))
    return cfg


class Settings:
    def __init__(self) -> None:
        self.models = load_models()
        self.prompts = load_yaml("prompt_templates.yaml")
        # 안전·윤리 규칙(가이드 6). ⚠️ 현재 코드에서 소비하는 곳이 없다 — 필터/에스컬레이션을
        # 구현할 때 쓸 자리로 남겨 둔 것이다. 없앨 게 아니라 아직 안 쓴 것.
        self.safety = load_yaml("safety_rules.yaml")


settings = Settings()
