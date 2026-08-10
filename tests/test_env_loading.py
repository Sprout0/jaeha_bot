"""앱이 .env 를 읽는지. API 백엔드를 켜는 순간 이게 없으면 조용히 로컬로 폴백한다.

2026-08-10 에 실제로 겪은 함정: 노트북은 OPENAI_API_KEY 가 Windows 환경변수(setx)라
잘 돌았는데 .env 에는 없었다. 젯슨에 .env 를 그대로 복사해도 키가 안 넘어갔다.
앱이 .env 를 읽지 않으면 같은 혼선이 운영에서 반복된다 — 게다가 폴백은 '동작은 하는'
실패라 알아채기 더 어렵다.
"""
import os

from app.config import load_env


def test_load_env_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("JAEHA_TEST_TOKEN", raising=False)
    (tmp_path / ".env").write_text("JAEHA_TEST_TOKEN=from-file\n", encoding="utf-8")

    load_env(tmp_path)

    assert os.environ.get("JAEHA_TEST_TOKEN") == "from-file"


def test_existing_env_var_wins_over_dotenv(tmp_path, monkeypatch):
    # 노트북은 setx 로 넣은 실제 환경변수가 있다. 파일이 그걸 덮으면 혼란스럽다.
    monkeypatch.setenv("JAEHA_TEST_TOKEN", "from-shell")
    (tmp_path / ".env").write_text("JAEHA_TEST_TOKEN=from-file\n", encoding="utf-8")

    load_env(tmp_path)

    assert os.environ["JAEHA_TEST_TOKEN"] == "from-shell", \
        "이미 설정된 환경변수를 .env 가 덮어쓰면 안 된다"


def test_missing_dotenv_is_harmless(tmp_path):
    load_env(tmp_path)   # .env 가 없는 디렉터리 — 예외 없이 지나가야 한다
