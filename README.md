# 재하봇 1 (jaeha_bot_1) — 교육용 AI

Jetson Orin Nano(8GB)에서 **독립 구동**하는 2세 유아용 음성·비전 교육 놀이 AI.
파이프라인: `카메라/마이크 → faster-whisper(STT) → LLM Agent(Function Calling) → Piper(TTS)`
상세 사양은 [`가이드라인_요약.md`](./가이드라인_요약.md) 참조.

## 폴더 구조
```
jaeha_bot_1/
├── app/                  # 실행 코드
│   ├── main.py           # 엔트리포인트 (파이프라인 루프)
│   ├── config.py         # configs/*.yaml 로더
│   ├── stt_module.py     # faster-whisper STT
│   ├── tts_module.py     # Piper TTS
│   ├── vision_module.py  # YOLOv8/MediaPipe 비전(상태값 압축)
│   ├── agent.py          # 로컬 LLM 추론
│   ├── agent_functions.py# Function Calling 정의
│   ├── education_modes.py# 놀이 모드 진행
│   └── daily_briefing.py # 부모 브리핑 생성
├── configs/              # 모델 경로·프롬프트·안전 규칙
├── scenarios/            # 교육 놀이 시나리오 카드
├── logs/                 # 대화/지연/메모리 로그 (git 미추적)
├── reports/              # 주차 보고서·브리핑 템플릿
├── models/               # 모델 가중치 (git 미추적, 별도 배포)
├── Dockerfile            # Jetson(L4T) 배포 이미지
├── Dockerfile.dev        # PC 개발용 이미지
├── docker-compose.yml    # Jetson 실행 (장치·볼륨 연결)
├── requirements.txt
└── run.sh                # 실행 스크립트
```

## 빠른 시작

### 방법 A. 네이티브 실행 (보드/PC)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# models/ 에 LLM(gguf)·YOLO(pt)·Piper 음성 모델을 넣은 뒤
./run.sh local        # = python3 -m app.main
```

### 방법 B. Docker (권장 — 환경 공유)
PC 개발용:
```bash
./run.sh dev          # Dockerfile.dev 빌드 후 실행
```
Jetson 보드:
```bash
./run.sh docker       # = docker compose up --build
```

## Docker로 환경 공유하는 방법

목표: "내 PC에서 되는데 보드에서 안 됨" 문제를 없애고, 팀원이 **동일한 환경을 한 줄로 재현**하게 만드는 것. 핵심은 Jetson이 **ARM64 + NVIDIA JetPack(L4T)** 라는 점이라, 일반 `python:slim` 이미지가 아니라 NVIDIA 공식 L4T 이미지를 베이스로 써야 GPU/CUDA가 잡힙니다.

**1) 두 개의 이미지로 분리**
개발은 PC(x86)에서 빠르게, 배포는 Jetson에서. 그래서 `Dockerfile`(Jetson, `nvcr.io/nvidia/l4t-pytorch` 베이스)과 `Dockerfile.dev`(PC, `python:3.11-slim`)를 나눠 뒀습니다. 1~2주차 PC 데모는 `dev`로, 4주차 보드 통합부터 `Dockerfile`로 갑니다.

**2) 코드·의존성은 이미지에, 모델·로그는 볼륨으로**
모델 가중치(gguf/pt)는 수 GB라 이미지에 굽지 않고 `./models`를 컨테이너에 마운트합니다(`docker-compose.yml`의 `volumes`). 덕분에 이미지는 가볍게 공유하고, 모델만 따로 배포하면 됩니다. `logs/`도 마운트해 호스트에서 지연·메모리 측정 결과를 바로 봅니다.

**3) 장치 연결**
유아와 상호작용하려면 컨테이너가 마이크·스피커·카메라에 접근해야 합니다. compose의 `devices`에서 `/dev/snd`(오디오)와 `/dev/video0`(카메라)를 넘기고, Jetson GPU는 `runtime: nvidia`로 잡습니다.

**4) 버전 고정으로 재현성 확보**
베이스 이미지 태그(`r36.2.0` 등)를 **보드의 JetPack 버전에 정확히 맞춰** 고정하고, `requirements.txt`도 하한 버전을 명시했습니다. 팀원은 같은 태그로 빌드하면 동일 환경을 얻습니다. `docker --version` / `dpkg-query --show nvidia-l4t-core`로 보드 버전을 먼저 확인하세요.

**5) 공유 방법 두 가지**
- 소스 공유: 이 레포를 클론 → `./run.sh docker` (각자 빌드, 가장 단순)
- 이미지 공유: `docker save jaeha_bot:jetson | gzip > jaeha_bot.tar.gz` 로 파일 전달하거나, 레지스트리(`docker push`)에 올려 `docker pull`. 오프라인 보드엔 `docker load` 가 편합니다.

> 주의: Jetson 이미지는 **반드시 ARM64**여야 합니다. PC(x86)에서 빌드해 보드로 옮기려면 `docker buildx build --platform linux/arm64 ...` 가 필요하니, 가능하면 보드에서 직접 빌드하는 것을 권장합니다.

## 개발 우선순위 (가이드 5)
"가장 똑똑한 답변"보다 **8GB 안에서 안 멈추고 3초 내 반응하는 안정성**이 최우선.
양자화(INT4/8) · 작은 모델 · 모듈 순차 로딩으로 OOM을 방지합니다.

## Known issues / TODO
- 각 모듈은 현재 **스텁(인터페이스만)** 상태 — `NotImplementedError` 자리부터 구현 시작.
- Piper 한국어 음성 모델, YOLOv8 가중치, 양자화 LLM(gguf)을 `models/`에 준비해야 실행됩니다.
- Jetson 베이스 이미지 태그는 보드 JetPack 버전에 맞춰 수정 필요.
