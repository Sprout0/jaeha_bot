# =====================================================================
# 재하봇 — 전면 API 경로(pipeline: realtime) 이미지
#
# GPU 가 필요 없는 경로라 일반 파이썬 이미지를 쓴다. python:3.10-slim 은 arm64(젯슨)와
# amd64(PC)를 모두 제공하므로 같은 파일로 두 기계에서 빌드된다.
#
# 로컬 경로(pipeline: local — STT·TTS·LLM 을 보드 GPU 로)는 이미지로 만들지 않는다.
# CTranslate2·llama-cpp 를 CUDA 로 소스 빌드해야 해서(tools/jetson/) 네이티브 conda 로 설치한다.
#
# 이미지에 넣지 않는 것: .env(키), configs/local.yaml(기계별 설정), models/(가중치)
#   → docker-compose.yml 이 실행 때 붙인다.
# =====================================================================
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# sounddevice → PortAudio, soundfile → libsndfile, 장치 확인용 alsa-utils(aplay -l)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 libsndfile1 alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-realtime.txt .
RUN pip install --no-cache-dir -r requirements-realtime.txt

COPY app/ app/
COPY configs/ configs/
COPY scenarios/ scenarios/

CMD ["python", "-m", "app.main_realtime"]
