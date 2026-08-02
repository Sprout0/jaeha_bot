# =====================================================================
# 재하봇 1 — Jetson Orin Nano (JetPack 6 / L4T r36.x) 용 이미지
# 베이스: NVIDIA L4T PyTorch (ARM64 + CUDA 포함)
# 태그는 보드의 JetPack 버전에 맞춰 조정하세요. (예: r36.2.0)
# =====================================================================
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 오디오(마이크/스피커) + 영상 처리용 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libportaudio2 portaudio19-dev \
        libgl1 libglib2.0-0 alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사 -> 레이어 캐시 활용
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 모델/로그는 볼륨으로 마운트 (이미지에 굽지 않음)
VOLUME ["/app/models", "/app/logs"]

CMD ["python3", "-m", "app.main"]
