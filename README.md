# 재하봇 1 (jaeha_bot) — 2세 유아용 음성 놀이 AI

Jetson Orin Nano(8GB)에서 **독립 구동**하는 한국어 음성 교육 놀이 로봇.

```
마이크 ─▶ 호출어 KWS(ONNX) ─▶ STT(faster-whisper) ─▶ 놀이 상태머신 ─▶ TTS(Supertonic)
                                                    └▶ LLM ─┬ gpt-4o-mini (운영 기본)
                                                            └ EXAONE 2.4B GGUF (오프라인 폴백)
```

부르기 전에는 **대기 모드**로 호출어만 듣는다(LLM·TTS 를 안 돌려 자원 절약).
**"하이 티드"** 가 들리면 대화 모드로 깨어나고, 30초 조용하면 다시 잠든다.
(2026-08-25 에 `재하봇` 에서 바꿨다 — whisper 가 고유명사 `재하봇` 을 힌트 없이는
41% 밖에 못 읽어 2단계 검증이 무거웠다. `하이 티드` 는 36개 중 35개를 정확히 읽는다.)

## 지금 어디까지 됐나

| 부분 | 상태 | 비고 |
|---|---|---|
| 호출어(KWS) | ✅ 동작 | ONNX 2단계. v6 배포 완료(08-26), 임계 0.05. 젯슨 실기 5턴 1패스 통과(09-07). 🔴 거실 헛깨움 시간당 1.33~2.67회로 **합격선(1회) 초과** |
| STT | ✅ 동작 | **`medium`**, 젯슨 GPU. 3~6세 CER **31.46%**<sup>*</sup> |
| LLM | ✅ 동작 | **하이브리드** — gpt-4o-mini 기본, 끊기면 EXAONE-3.5-2.4B Q4 폴백(지연 적재) |
| TTS | ✅ 동작 | Supertonic ONNX + **TensorRT@24**. 합성 817→454ms, 젯슨 첫 소리 **721ms** |
| 놀이 | 🔶 2개 | 동물소리·따라말하기. 색칠·사물찾기·감정대화는 카드만 |
| 노래 틀기 | 🔶 기본 꺼짐 | 로컬 음원 우선 + 유튜브 IFrame. 젯슨만 켬(09-11). ⏳ 노래 중 호출어는 스피커 없어 미측정 |
| 안전 판정기 | 🔶 평가 전용 | `app/safety.py`·`claims.py` 는 `tools/eval_llm.py` 에서만 쓴다 — **런타임 가드 미적용**. 운영 안전은 프롬프트 규칙이다 |
| 비전 | 🚧 미구현 | STEP 9. `vision_module.py` 는 스텁. 제약은 메모리(TRT 2,887MB) |
| 부모 브리핑 | 🚧 미구현 | `daily_briefing.py` 는 스텁(참조하는 코드 없음) |

<sup>*</sup> 운영값은 `medium`(`configs/model_paths.yaml`)이다. **3~6세 153발화·화자 114명
실측 CER 31.46%** (`reports/stt/README.md`). 실제 대상인 **2세는 공개 데이터가 없어 여전히
미측정**이고, 이 프로젝트에서 가장 큰 미검증 영역이다. (예전에 여기 적혀 있던
`large-v3-turbo` · CER 2.96% 는 9세 음성 기준의 다른 모델 값이다.)

📄 **마무리 문서는 [`docs/final/`](docs/final/README.md) 에 있다** — 보고서 본문
([report.md](docs/final/report.md)), 특허 정리, 그림. 서술 대상은 태그
`doc-baseline-2026-09-12` 다.

## 폴더 구조

```
jaeha_bot/
├── app/
│   ├── main.py            # 엔트리포인트 (대기↔대화 루프)
│   ├── config.py          # configs/*.yaml 로더 (+ local.yaml 오버레이)
│   ├── audio_source.py    # 마이크 단일 스트림 소유 (감지기·STT 가 공유)
│   ├── wake.py            # 호출어 판정 + 감지기 선택(ONNX / STT 폴백)
│   ├── wake_onnx.py       # ONNX KWS 감지기
│   ├── stt_module.py      # faster-whisper STT + 에너지 VAD + 환각 가드
│   ├── tts_module.py      # Supertonic TTS
│   ├── agent.py           # LLM 대화·놀이 대사 렌더 (API/로컬 공통 후처리)
│   ├── education_modes.py # 놀이 상태머신
│   ├── safety.py          # 답변 안전 판정 (평가 지표용. 런타임 가드는 미적용)
│   ├── claims.py          # 날조 판정 (없는 놀이·음원을 약속하는지)
│   ├── text_norm.py       # 자모 거리, 오인식 교정, 반복 제거
│   ├── metrics.py         # 턴별 지연·메모리 계측
│   ├── vision_module.py   # 🚧 스텁
│   └── daily_briefing.py  # 🚧 스텁
├── configs/               # 런타임 설정 (아래 '설정' 참조)
│   └── wake/              # 호출어 학습 설정 (Colab 용, 런타임 아님)
├── scenarios/             # 놀이 카드(JSON)
├── data/                  # few-shot 예시(= QLoRA 씨앗) + 평가셋 3종
├── tests/                 # 1,028건. 마이크·모델·젯슨·네트워크 없이 전부 돈다
├── tools/                 # 런타임 아님 — 호출어 데이터 생성·검수, LLM 평가, 벤치
├── reports/eval/          # 평가 원본 로그(답변 전문). 판정 규칙이 바뀌면 재채점한다
├── models/                # 가중치 (git 미추적, push_model.sh 로 별도 배포)
├── logs/                  # 계측·녹음 (git 미추적)
└── docs/superpowers/      # 설계·계획 문서 (호출어 감지기 / 놀이 상호작용)
```

## 실행

노트북·젯슨 모두 conda env `jaeha_bot` 을 쓴다.

```bash
conda activate jaeha_bot && python -m app.main
```

마이크·TTS 없이 대화와 놀이만 확인하려면:

```bash
python -m app.main --text
```

모듈별 단독 테스트: `python -m app.stt_module` (마이크), `python -m app.agent` (LLM),
`python -m app.wake` (문자열 판정, 마이크 불필요).

## 설정 — ⚠️ 이 부분을 먼저 읽을 것

`configs/model_paths.yaml` 은 **젯슨(운영) 기준값**이다. 값마다 그렇게 정한 실측 근거가
주석으로 붙어 있으니, 숫자를 바꾸기 전에 그 줄을 읽는 편이 빠르다.

`push_code.sh` 가 `configs/` 를 통째로 젯슨에 밀어넣기 때문에 **두 기계가 이 파일 한 벌을
공유한다.** 그런데 `stt.device` 는 젯슨=`cuda` / 노트북=`cpu` 로 정반대여야 한다(노트북
ctranslate2 는 CPU 전용 휠이라 `cuda` 면 죽는다). `metrics.tag` 도 마찬가지.

→ **기계마다 다른 값은 `model_paths.yaml` 을 고치지 말고 `configs/local.yaml` 에 적는다.**
git 미추적이고 push 전송에서도 빠진다. 키 단위 deep merge 라 거기 안 적은 값은 운영값 그대로.

```bash
cp configs/local.example.yaml configs/local.yaml   # 필요한 줄만 남기면 됨
```

| 파일 | 내용 |
|---|---|
| `model_paths.yaml` | 모델 선택·디코딩·VAD·호출어 임계 등 전부 (젯슨 기준) |
| `local.yaml` | 이 기계 전용 덮어쓰기 (git 미추적) |
| `prompt_templates.yaml` | system 프롬프트. 말투·길이·안전 규칙 |
| `safety_rules.yaml` | `reply_check` 는 ✅ `app/safety.py` 가 읽는다. 나머지 항목은 🚧 소비처 없음 |
| `object_labels.json` | 비전 라벨 매핑. 🚧 미사용 |
| `wake/*.yaml` | 호출어 학습 설정(Colab). 런타임이 읽지 않음 |

🔴 **운영 LLM 백엔드는 `model_paths.yaml` 이 아니라 젯슨의 `local.yaml` 에 있다.**
기본값은 `llm.backend: local`(EXAONE)이고, 젯슨만 `openai` 로 덮어쓴다. 그 파일은
git 미추적 + push 제외라 **젯슨에서 지워지면 아무 경고 없이 로컬 EXAONE 으로 되돌아간다.**
백엔드를 확인하려면 젯슨에서 `grep -A1 '^llm:' configs/local.yaml`.

## 젯슨 배포

Docker 가 아니라 **SSH 로 코드만 밀어넣는다**(젯슨은 실행 전용).

```bash
bash push_code.sh                          # app/ configs/ scenarios/ data/
bash push_model.sh models/새모델.gguf       # 모델은 새로 만들었을 때만
```

젯슨에서:

```bash
conda activate jaeha_bot && cd ~/jaeha_bot && python -m app.main
```

젯슨 환경은 노트북과 다르다 — py3.10, 전용 pip 인덱스, ctranslate2·llama-cpp 는 CUDA
소스빌드다. `requirements.txt` 로 그냥 설치하면 GPU 가 안 잡힌다. 절차와 함정은
`push_model.sh` 주석과 프로젝트 메모리를 참조.

> Dockerfile·docker-compose 는 **일상 배포용이 아니다.** 실제 운용은 위 SSH 경로다.
> 남겨 두는 이유는 **환경 공유용** — 다른 사람(교수님·팀원)에게 "이 환경 그대로 돌려보세요"를
> 한 줄로 전달해야 할 때 쓴다. 그때는 베이스 이미지 태그를 보드 JetPack 버전에 맞추고
> (`dpkg-query --show nvidia-l4t-core`), Jetson 이미지는 **반드시 ARM64** 로 빌드해야 한다
> (PC 에서 만들려면 `docker buildx build --platform linux/arm64`, 가급적 보드에서 직접 빌드).
> ⚠️ 젯슨의 실제 구성(ctranslate2·llama-cpp CUDA 소스빌드)은 이 Dockerfile 에 반영돼 있지 않다 —
> 공유 전에 갱신이 필요하다.

## 설계에서 알아둘 것

- **놀이의 흐름·정답판정은 코드(상태머신)가 갖고, LLM 은 칭찬·질문 '문장'만 렌더한다.**
  2.4B 에 상태를 맡기면 깨지므로 function-calling 은 쓰지 않는다. 렌더가 실패하거나
  설명으로 새면 템플릿으로 폴백해 놀이가 멈추지 않는다.
- **마이크 스트림을 닫지 않는다.** 닫고 STT 가 새로 열면 ALSA 재오픈 + 소음 재측정 +
  에코 쿨다운으로 0.7초짜리 '귀 먹은 구간'이 아이가 말을 시작하는 순간에 생긴다.
  `AudioSource` 하나가 스트림을 소유하고 감지기와 STT 가 나눠 쓴다.
- **호출어 감지 실패는 봇을 죽이지 않는다.** ONNX 로드가 안 되면 옛 STT 자모 매칭으로
  자동 폴백한다(`wake.detector: onnx | stt`).
- **TTS 응답이 길면 그만큼 아이가 기다린다.** `speak()` 는 재생이 끝날 때까지 블로킹한다.
  답변을 짧게 만드는 프롬프트·`max_sentences` 가 곧 지연 단축이다.
- 🔴 **`unsafe=0` 은 "안전하다"가 아니라 "규칙에 걸린 게 없다"는 뜻이다.** `app/safety.py`
  는 규칙 기반이라 반드시 놓친다. 실제로 안전 결함은 매번 **답변을 사람이 읽어서** 나왔고
  자동 판정기는 그때마다 통과시켰다. 안전 관련 변경 뒤에는 `reports/eval/` 의 답변 전문을
  읽을 것. 판정기를 고치면 `JUDGE_VER` 을 올리고 옛 수치와 직접 비교하지 않는다.
- **놀이의 템플릿 폴백은 예외 경로가 아니라 평상시 경로다.** LLM 렌더가 `require` 검증에
  걸리면 템플릿으로 내려오는데, 그 일이 자주 일어난다. 폴백 문구를 '비상용'으로 여기고
  대충 쓰면 그게 아이가 듣는 말이 된다.

## 남은 일

1. **대상 아동 실사용 시험** — 모든 실기 수치가 **성인 발화**로 얻은 것이다. 2세에서의
   동작은 미확인이고, 하려면 IRB 심의와 법정대리인 동의가 먼저다
2. **스피커 확보** — `PLAY_PAD_S=0.15` 는 도입 이래 미계측이고, 노래 중 호출어도 못 잰다.
   이어폰으로는 음향 누설이 없어 8회 중 0회 검출이라 대체가 안 된다
3. **헛깨움을 합격선 안으로** — 거실 시간당 1.33~2.67회(합격선 1회 이하). 수 시간 연속
   측정과 임계값 재조정이 필요하다
4. **안전 판정기를 런타임에 붙이기** — 지금은 평가 지표로만 돈다. 프롬프트로 안전을
   확보하는 방식이 용량 한계다(세 줄 더하면 다른 문항이 깨진다)
5. **실음성 녹음** — 호출어 학습 데이터에 실제 사람 목소리가 **한 건도 없다**
6. 놀이 카드 확대 + 껍데기 놀이 3개 구현, 비전(STEP 9)
