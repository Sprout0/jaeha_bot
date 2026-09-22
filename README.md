# 재하봇 (jaeha_bot) — 2세 유아용 음성 대화·놀이 로봇

아이가 **"하이 티드"** 라고 부르면 깨어나 대화하고, 동물 소리 흉내·따라 말하기 놀이를 하고,
동요를 틀어 주는 한국어 음성 로봇이다. NVIDIA Jetson Orin Nano 8GB + ReSpeaker USB 마이크
어레이에서 돈다. 30초 동안 말이 없으면 다시 호출어만 듣는 대기로 돌아간다.

> 상태(2026-09-22): 개발 종료·인수인계 단계. 모든 실기 수치는 **성인(개발자) 목소리**로
> 얻었고, 대상 아동이 실제로 쓴 적은 없다.

---

## 1. 대화 경로 두 가지

같은 저장소에 두 경로가 있고, **설정 한 줄(`pipeline`)** 로 고른다. 호출어 검출·놀이
상태머신·노래·안전 규칙은 두 경로가 함께 쓴다.

```
[로컬 경로]  pipeline: local  →  app/main.py            (저장소 기본값)
마이크 → 호출어(ONNX 2단계) → 음성 인식(faster-whisper medium, 보드 GPU)
       → 놀이 상태머신 / LLM(gpt-4o-mini 또는 로컬 EXAONE 2.4B) → 음성 합성(Supertonic + TensorRT) → 스피커

[전면 API 경로]  pipeline: realtime  →  app/main_realtime.py   (지금 젯슨 운영값)
마이크 → 호출어(ONNX 2단계, 보드) → OpenAI Realtime(gpt-realtime-mini: 받아 적기·대답·목소리 marin)
       → 턴 판단(잠들기·노래·놀이·대화) → 응답 관문(위험 턴 붙잡기·놀이 이탈 바로잡기) → 스피커
```

| | 로컬 경로 | 전면 API 경로 |
|---|---|---|
| 체감 응답(말 끝 → 첫 소리, 젯슨 실기) | 중앙 4.14초, 3초 이내 0/5턴 (09-07) | 중앙 **2.56초**, 3초 이내 14/18턴 (09-19)<sup>1</sup> |
| 시스템 메모리 / GPU | 85% / 99.9% | 13~33% / 0% |
| 아이 음성 | **기기 밖으로 안 나간다**(LLM 에는 글자만) | OpenAI 서버로 간다<sup>2</sup> |
| 인터넷이 끊기면 | 로컬 EXAONE 으로 이어서 대답 | 3번 다시 붙고, 안 되면 안내 문구 뒤 대기 |
| 목소리 | Supertonic F2 (블라인드 평가로 고름) | Realtime `marin`, 속도 1.05 |
| 비용 | LLM 만 — 무시할 수준 | 턴당 중앙 $0.003 |
| 설치 | CUDA 소스 빌드 필요(4절 C) | pip 9개(4절 A) 또는 Docker(4절 B) |

<sup>1</sup> 말 끝 기다리기 1,200ms 시점의 값이다. 지금은 900ms 이고(아이 녹음 153개로 −0.28초·잘림
2→3% 측정, `reports/turn/realtime_turn_end.md`), 젯슨에서 다시 재지 않았다.
<sup>2</sup> OpenAI 는 만 13세 미만 개인정보에 데이터 무보관(ZDR) 적용을 요구하고, Realtime 의 ZDR
은 **기관 명의 사전 승인**이 필요하다. 승인은 받지 않았다. **실제 아동에게 쓰려면 이게 먼저다.**
받기 전에는 로컬 경로를 쓴다.

### 경로 바꾸기

`configs/local.yaml`(기계별 덮어쓰기, git 미추적)에 적는다. 예시는 `configs/local.example.yaml` 의 A 블록이고,
지금 젯슨이 쓰는 설정 전체는 `configs/local.jetson.example.yaml` 이다(전면 API + 노래 켬 + 로컬 경로일 때 LLM 은 API).

```yaml
pipeline: realtime
wake:
  onnx:
    verify:
      mode: embed            # 전면 API 에는 whisper 가 없어 임베딩 대조만 쓴다
      embed_rescue: {enabled: true, min_similarity: 0.85}
```

되돌리려면 이 블록을 지운다. `./run.sh` 가 `pipeline` 을 읽어 진입점을 고르고,
`./run.sh check` 가 지금 어느 경로인지와 준비 상태를 보여 준다.
🔴 `local.yaml` 이 지워지면 **경고 없이** 저장소 기본값(로컬 경로 + 로컬 EXAONE)으로 돈다.

---

## 2. 무엇이 되고 무엇이 안 되나

| 기능 | 로컬 | 전면 API | 비고 |
|---|---|---|---|
| 호출어 "하이 티드" | ✅ | ✅ | 임베딩 대조 호출 25/35(성인). 거실 45분 녹음에 헛깨움 1~2회 — 합격선(시간당 1회) 근처 |
| 자유 대화 | ✅ | ✅ | |
| 놀이 2종 (동물 소리·따라 말하기) | ✅ | ✅ | 놀이 진행·정답 판정은 코드(상태머신)가 한다. 색칠·사물찾기·감정은 카드만 있음 |
| 놀이 이탈 바로잡기 | 틀린 문장을 합성 전에 템플릿으로 교체 | 글자를 보고 틀린 이름이 들리기 전에 잘라 미리 녹음한 문장으로 이음 | API: 실서버 24턴 중 8턴 바로잡음, 샘 0 |
| 안전 | 프롬프트 규칙 | 프롬프트 규칙 + 위험 신호 턴만 소리를 붙잡아 검사, 막으면 안전 문장 + `logs/guardian_*.jsonl` | API 가드는 젯슨 실기 전 |
| 명령(잠들기·노래·놀이) | 글자 규칙 | 글자 규칙 + 모델 도구 호출 | API 도구 11/13 맞음 |
| 노래 틀기(유튜브 아동용 영상만) | 🔶 기본 꺼짐 | 🔶 기본 꺼짐 | 젯슨만 켬. 크롬·Xvfb·PulseAudio 필요(`./run.sh setup-youtube`) |
| 부모 알림 전송 · 비전 · 비용 상한 | 🚧 | 🚧 | 구현 안 함(빈 껍데기 모듈은 09-22 삭제) |

---

## 3. 저장소에 없는 것 (따로 받아야 한다)

| 무엇 | 어디에 | 누가 쓰나 | 구하는 법 |
|---|---|---|---|
| `.env` | 루트 | 둘 다 | `cp .env.example .env` 후 키 입력 (5절) |
| `configs/local.yaml` | `configs/` | 둘 다 | 젯슨: `cp configs/local.jetson.example.yaml configs/local.yaml` / PC: `local.example.yaml` 에서 필요한 블록만 |
| 호출어 모델 5개 | `models/wake/v6/` — `jaehabot_v6.onnx`, `embedding_model.onnx`, `melspectrogram.onnx`, `jaeha_v6.yaml`, `jaehabot_v6.pt`(학습 원본, 실행엔 불필요) | 둘 다 | 인계 압축본 `jaeha_bot_wake_v6.tar.gz`(3.5MB, 본보기 포함 — 루트에서 `tar -xzf`). 다시 만들려면 `tools/colab_wake_train_v6.ipynb`(Colab, HF_TOKEN 은 Colab 비밀값) |
| 호출어 본보기 `templates_haitid.npy` | `models/wake/v6/` | 전면 API 필수, 로컬은 embed 모드일 때 | **쓸 사람 목소리로 새로 만든다**: `python tools/enroll_wake.py --record 10` (지금 것은 개발자 가족 한 명의 목소리) |
| EXAONE 3.5 2.4B Q4 GGUF (1.6GB) | `models/exaone-3.5-2.4b-q4.gguf` | 로컬 경로의 폴백 | Hugging Face `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF`. ⚠️ 비상업 라이선스 |
| faster-whisper medium, Supertonic | 캐시(`~/.cache/huggingface`) | 로컬 경로 | 첫 실행 때 자동으로 받는다 |
| 동요 mp3 | `assets/songs/` | 노래 | `assets/README.md`. 목록은 `configs/audio_assets.yaml` |

젯슨으로 모델을 보낼 때는 `JETSON=사용자@주소 bash push_model.sh <파일>`.

---

## 4. 환경 만들기

검증된 조합: **Jetson Orin Nano 8GB, L4T R36.5(JetPack 6 계열), CUDA 12.6, Python 3.10.20**
(miniforge conda env `jaeha_bot`). 개발 PC 는 Windows 11 + Anaconda(Python 3.12).

### A. 전면 API 경로 — 어느 기계든 (권장)

GPU 가 필요 없다.

```bash
sudo apt install libportaudio2 libsndfile1          # 리눅스
conda create -n jaeha_bot python=3.10 -y && conda activate jaeha_bot
pip install -r requirements-realtime.txt            # 젯슨 실사용 버전으로 고정
```

### B. 전면 API 경로 — Docker

`python:3.10-slim` 기반이라 젯슨(arm64)과 리눅스 PC(amd64)에서 같은 파일로 빌드된다.
키·기계별 설정·모델은 이미지에 굽지 않고 실행 때 붙인다(`docker-compose.yml` 머리말).

```bash
docker compose up --build
```

⚠️ 이 Dockerfile 은 2026-09-22 에 새로 썼고 **이미지 빌드는 아직 못 해 봤다**(설치 파일이
arm64·amd64 로 모두 받아지는 것까지만 확인). 노래 틀기는 컨테이너에서 켜지 않는다. 젯슨에는
Docker 가 깔려 있지 않다.

### C. 로컬 경로 — 젯슨 네이티브 (GPU)

`requirements.txt` 를 그냥 설치하면 GPU 가 안 잡힌다. 순서대로:

1. miniforge 설치 → `conda create -n jaeha_bot python=3.10`
2. 젯슨 전용 pip 인덱스에서 `onnxruntime-gpu==1.24.0` (TensorRT 포함):
   `pip install onnxruntime-gpu==1.24.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126`
3. `pip install -r requirements.txt` — 단 `onnxruntime`·`llama-cpp-python` 줄은 빼고,
   `faster-whisper` 는 `--no-deps` 로
4. CTranslate2 CUDA 소스 빌드: `bash tools/jetson/build_ct2_cuda.sh` (sm_87, 1시간 남짓,
   `~/.local/ct2` 에 설치. 인덱스 휠은 CPU 전용이라 STT 가 발화당 3.6초 걸린다)
5. llama-cpp-python CUDA 소스 빌드: `bash tools/jetson/build_llama.sh` (0.3.34. 인덱스의
   0.3.14 는 생성 때 죽는다)
6. conda 활성화 훅에 라이브러리 경로 추가 —
   `$CONDA_PREFIX/etc/conda/activate.d/zz_ld.sh`:
   `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:$HOME/.local/ct2/lib:${LD_LIBRARY_PATH:-}"`
   (`./run.sh` 는 훅 없이도 같은 경로를 붙인다)

젯슨에서 돌고 있는 버전: faster-whisper 1.2.1, ctranslate2 4.8.0(소스), llama_cpp_python 0.3.34(소스),
supertonic 1.3.1, openai 2.53.0, 나머지는 `requirements-realtime.txt` 와 같다.

### D. 개발 PC (Windows)

`pip install -r requirements.txt` (llama-cpp 는 파일 안 설명대로 CPU 휠 0.3.2).
`configs/local.yaml` 에 D 블록(`stt.device: cpu` 등)을 둔다. 마이크 없이 확인할 때:

```bash
python -m app.main --text              # 로컬 경로, 글자로 대화
python -m app.main_realtime --no-wake  # 전면 API, 호출어 없이 바로 대화
```

---

## 5. 비밀 값 (`.env`)

| 키 | 필요할 때 | 없으면 |
|---|---|---|
| `OPENAI_API_KEY` | 전면 API 경로(필수), 로컬 경로에서 `llm.backend: openai` | 전면 API 는 기동에서 멈춘다. 로컬은 **조용히** EXAONE 으로 내려간다 |
| `YOUTUBE_DATA_KEY` | 노래 틀기(`youtube.enabled: true`) | 검색 못 하고 `assets/songs` 만 튼다 |
| `CLOVA_STUDIO_KEY` | 비교 평가(`tools/eval_llm.py`), `llm.api_model` 이 HCX 일 때 | 해당 기능만 안 된다 |
| `GEMINI_API_KEY` | 비교 평가 전용 | — (Gemini 약관은 18세 미만 대상 앱 금지) |

키는 인계할 때 **새로 발급**해서 넘긴다. 개발 중 쓰던 키는 폐기할 것.

---

## 6. 실행

```bash
./run.sh check            # 봇을 띄우지 않고 준비 상태만: 모듈, 경로, 키, 호출어 설정, 오디오 장치
./run.sh                  # pipeline 을 보고 app.main 또는 app.main_realtime 을 띄운다
./run.sh setup-youtube    # 노래 틀기 준비물 점검 + ~/.asoundrc(공유 출력 respk) 설치
```

- 전면 API 는 첫 기동 때 고정 문구(안내·안전 문장, 놀이 바로잡기 문장 약 70개)를 `marin` 목소리로
  합성해 `~/.cache/jaeha_voice` 에 둔다. **첫 기동만 4~5분** 걸린다.
- 로그: `logs/jaeha_날짜.log`(대화), `logs/metrics_날짜.jsonl`(턴별 지연·비용),
  `logs/guardian_날짜.jsonl`(막거나 정정한 일, 글자만).
- 시험: `python -m pytest -q` — 1,204개, 마이크·모델·젯슨·네트워크 없이 30초 안에 돈다.

### 젯슨으로 배포

젯슨의 `~/jaeha_bot` 은 git 저장소가 아니다. 노트북에서 코드만 밀어 넣는다.

```bash
JETSON=사용자@주소 bash push_code.sh   # 젯슨 주소는 저장소에 두지 않고 이렇게 넘긴다
```

보내는 것: `app/` `configs/`(`local.yaml` 제외) `scenarios/`, 실행·환경 파일, 젯슨에서 쓰는 도구 10개
(`tools/README.md` 1절). 평가·학습 도구와 평가셋은 노트북 전용이라 보내지 않는다.

---

## 7. 설정 파일

| 파일 | 내용 |
|---|---|
| `configs/model_paths.yaml` | 모든 설정의 기준값(로컬 경로 기준). 값마다 정한 근거가 주석으로 붙어 있다 — 숫자를 바꾸기 전에 그 줄을 읽을 것. 전면 API 설정은 `realtime:` 절(모델·목소리·속도·말 끝 900ms·가드) |
| `configs/local.yaml` | 이 기계만의 덮어쓰기(경로 선택 포함). git 미추적. 예시: `local.example.yaml`(PC), `local.jetson.example.yaml`(젯슨 운영값) |
| `configs/prompt_templates.yaml` | 시스템 프롬프트 — 말투·길이·안전 규칙 |
| `configs/safety_rules.yaml` | 위험 낱말·먹기 표현 등. `app/safety.py` 가 읽는다 |
| `configs/audio_assets.yaml` | 로컬 음원·효과음 목록 |
| `configs/alsa/respk.asoundrc` | 노래와 봇 목소리를 한 스피커로 섞는 ALSA 장치 |
| `scenarios/scenario_cards.json` | 놀이 카드(동물 8, 낱말 10) |
| `configs/wake/` | 호출어 학습 설정(Colab 용 — 봇은 읽지 않는다) |

---

## 8. 폴더 구조

```
app/
  main.py                 로컬 경로 진입점 (대기 ↔ 대화 루프)
  main_realtime.py        전면 API 진입점 (기동 검사, 바깥 루프)
  realtime_conversation.py  깨어 있는 한 구간 — 턴 판단·노래·놀이·가드·끊김 복구
  realtime_session.py / realtime_protocol.py / realtime_audio.py / realtime_turn.py
  reply_gate.py           응답 관문(위험 턴 붙잡기·도중 검사)
  game_repair.py          놀이 이탈을 글자로 찾아 소리 자를 곳 계산
  guardian_log.py         보호자 기록
  voice_cache.py          고정 문구 wav 캐시
  wake.py / wake_onnx.py / wake_embed.py   호출어 2단계
  audio_source.py / audio_device.py / audio_player.py   마이크 공유·장치 선택·재생
  stt_module.py / tts_module.py / agent.py / filler.py  로컬 경로 전용
  education_modes.py      놀이 상태머신(두 경로 공용)
  music.py / youtube.py / song_names.py / loudness.py   노래 틀기
  safety.py / claims.py / text_norm.py / metrics.py / config.py
configs/             설정 (7절)
scenarios/           놀이 카드
data/                LLM few-shot 예시(로컬 경로가 읽는다), 평가셋, 목소리 블라인드 정답표, 호출어 녹음 메타
tests/               시험 1,204개
tools/               젯슨 관리 도구 · 측정/평가 · 호출어 학습 · 문서 생성 — 목록은 tools/README.md
reports/             측정 결과 — 주제별 README 에 요약, 일부 원자료(큰 파일은 git 미추적)
docs/submission-api/ 최종 제출본(전면 API 기준) 원고
docs/final/          로컬 구성 시점의 보고서·발명신고 초안(09-12 기준)
docs/superpowers/    기능별 설계(specs)·구현 계획(plans)
assets/              동요·효과음(파일은 git 미추적)
models/              가중치(git 미추적, 3절)
run.sh · push_code.sh · push_model.sh · Dockerfile · docker-compose.yml · requirements*.txt
```

---

## 9. 알아둘 설계

- **놀이의 순서와 정답은 코드가 쥔다.** 모델은 문장만 만든다. 모델이 문제를 바꿔 말하면
  로컬은 합성 전에 템플릿으로 바꾸고, 전면 API 는 글자가 소리보다 먼저 오는 틈(중앙 0.98초)에
  잘라 미리 녹음한 문장으로 잇는다.
- **전면 API 에서도 턴은 우리가 쥔다.** 서버의 자동 응답을 끄고(`create_response: false`),
  받아 적은 글자로 잠들기·노래·놀이를 먼저 가로챈 뒤 응답을 요청한다.
- **위험 신호가 있는 턴만 붙잡는다.** 아이 말에 위험 낱말이 있거나 '이거'+먹기 표현이면 소리를
  모아 글자를 끝까지 검사한다(약 +0.3초). 나머지는 흘려보내며 검사한다.
- **마이크 스트림은 하나를 닫지 않고 나눠 쓴다**(`AudioSource`). 다시 열면 0.7초짜리 '귀 먹은
  구간'이 아이가 말을 시작하는 순간에 생긴다.
- **ReSpeaker 는 16kHz 전용이고 장치가 하나다.** 봇이 도는 동안 다른 도구가 마이크를 못 연다.
- 🔴 **`unsafe=0` 은 "안전하다"가 아니라 "규칙에 걸린 게 없다"다.** 안전 결함은 매번 사람이
  답변을 읽어서 찾았다. 안전 관련 변경 뒤에는 `reports/eval/` 의 답변 전문을 읽을 것.

---

## 10. 넘겨받는 사람이 할 일

1. **아동 음성 국외 전송 해결** — OpenAI ZDR(기관 명의 사전 승인). 없으면 아동에게는 로컬 경로.
   아동 대상 시험 자체는 IRB 와 법정대리인 동의가 먼저다.
2. **젯슨 실기 확인** — 응답 관문, 놀이 바로잡기, 명령 도구("인형이 잘 자래"를 잠들기로 오판),
   말 끝 900ms, 공유기 뽑았다 꽂기(재연결).
3. **스피커 확보** — 지금까지 이어폰으로만 들었다. 노래 중 호출어·재생 여유(`PLAY_PAD_S`) 미측정.
4. **호출어 본보기를 실제 사용자 목소리로 다시 등록**, 헛깨움을 시간당 1회 아래로.
5. Docker 이미지 빌드 검증, 부모 알림 전송(텔레그램 계획만), 비용 상한, 놀이 추가.

## 11. 문서

- [docs/submission-api/README.md](docs/submission-api/README.md) — **최종 제출본**(전면 API 기준): 결과보고서·기술상세보고서·발명신고
- [docs/final/README.md](docs/final/README.md) — 로컬 구성 시점(09-12)의 보고서와 근거 문서
- [docs/final/api-개발보고서.md](docs/final/api-개발보고서.md) — 전면 API 경로를 만든 경과와 실측
- [tools/README.md](tools/README.md) — 도구 목록
- `docs/superpowers/specs/` — 기능별 설계 문서(결정의 이유와 실측 기록)
- `reports/*/README.md` — 측정 결과 요약
