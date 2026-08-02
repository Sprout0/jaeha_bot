"""설정 로더. configs/*.yaml 을 읽어 단일 객체로 제공한다."""
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"


def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings:
    def __init__(self) -> None:
        self.models = load_yaml("model_paths.yaml")
        self.prompts = load_yaml("prompt_templates.yaml")
        self.safety = load_yaml("safety_rules.yaml")


settings = Settings()
