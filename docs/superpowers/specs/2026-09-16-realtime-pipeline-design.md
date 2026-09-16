# 전면 API(OpenAI Realtime) 대화 경로 — 설계

- 날짜: 2026-09-16
- 상태: 설계 확정, 구현 전
- 결정 배경: 2026-09-09 회의 "개발은 전부 API". 선결 조건 다섯 가지는 사용자가 09-16 에 해소로 판단했다
  (정책 OK / 호출어 임베딩 단독 실기 18/20 / 로컬 폴백 불필요 / 재생 전 여유 OpenAI 가 Gemini 보다 낫다 /
  비용은 캐시로 40턴 월 21,200원 — `docs/final/report.md` 7.2절 정정판).
- 보고서가 서술하는 로컬 경로는 태그 `doc-baseline-2026-09-12` 와 젯슨 `~/jaeha_bot_baseline` 에 얼려 두었다.

## 1. 범위

**넣는다** — 지금 봇이 대화 중에 하는 일 전부:
호출어(로컬, 임베딩 검증) → Realtime 대화, 인사말, 노래 틀기/그만/노래 중 호출, 놀이 2종, 맞장구,
잠들기 명령, 30초 무응답이면 대기, 계측.

**안 넣는다**
- 소리 효과 — 지금 봇에도 연결돼 있지 않다(`AudioLibrary.find` 는 측정 도구만 부른다).
- 재생 전 안전 차단 — 이번 판은 **로그만**. 걸린 답이 잦으면 차단(전체 글자까지 소리 보류)으로 바꾼다.
- 로컬 폴백 — 끊기면 캐시 문구를 틀고 대기로 간다.
- 로컬 경로 수정 — `app/main.py` 는 한 줄도 안 건드린다.

**진행 원칙** — 구현 중 막히는 것은 멈추지 말고 9절 "보류 기록"에 적고 넘어간다.

## 2. 확정한 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 목소리 | 전부 `marin` 하나. 고정 문구도 marin wav 캐시 | 09-07 블라인드. F2(Supertonic)는 잃는다. 로컬 STT·TTS 모델을 안 올려 메모리가 빈다 |
| 안전 | `safety.check_reply` 결과를 로그로만 | 지금 봇도 프롬프트만으로 지킨다. 지연 0 |
| 코드 구성 | 새 진입점 `app/main_realtime.py`, 공용 모듈은 가져다 씀 | 보고서가 인용하는 로컬 코드를 `main` 에 그대로 둔다 |
| 대화 모델 | `gpt-realtime-mini` | 40턴 월 21,200원(큰 모델 73,300원) |
| 받아 적기 | `gpt-4o-transcribe` | `whisper-1` 은 아동 발화 39% 빈 문자열 |
| 말끝 대기 | 서버 VAD 1,200ms | 현행과 같게 — 비교 가능 |
| 마이크 | 반이중. 재생 중 + 0.15초 닫음 | 라이브 루프 에코 0/11턴 |
| 연결 단위 | 깰 때 연결, 대기로 갈 때 끊음. 최근 6턴을 글자로 넣어 이음 | 대기 중 비용 0 |
| 턴 진행 | `create_response: false` — 받아 적기를 보고 우리가 요청 | 노래·놀이 명령을 모델보다 먼저 가로채야 한다(모델은 "틀어줄게"라고 약속만 한다, bench README 5절) |

## 3. 구성

### 새 파일
| 파일 | 책임 | 의존 |
|---|---|---|
| `app/main_realtime.py` | 바깥 루프: 대기 → 호출어 → 대화 구간 → 대기. 기동 검사 | 아래 전부 |
| `app/realtime_session.py` | 연결 하나: 열기/닫기, `session.update`, 입력 오디오 전송, 이벤트 수신, `response.create`/`cancel`, 끊김 감지 | websockets |
| `app/realtime_turn.py` | **순수 함수.** 받아 적은 글자 → `sleep / music / game / chat` 판단과 요청 내용(지시문) 생성. 이력 항목 만들기 | `music`, `education_modes`, 잠들기 규칙 |
| `app/realtime_audio.py` | `MicGate`, `PcmAccumulator`, 16k→24k 변환, 스트리밍 재생기 — `tools/realtime_live.py` 에서 옮김 | numpy, sounddevice |
| `app/voice_cache.py` | 고정 문구 → marin wav. 없으면 Realtime 으로 만들어 저장, 있으면 틀기 | `realtime_session` |

`tools/realtime_live.py` 는 옮긴 모듈을 import 하도록 바꾼다(동작 불변, 기존 시험 유지).

### 가져다 쓰는 것(수정 없음)
`AudioSource`, `make_detector`, `MusicController`, 잠들기 낱말 규칙, `MetricsLogger`, 출력 장치 설정(`audio_device`), `safety.check_reply`, 프롬프트(`configs/prompt_templates.yaml`).

### 추가만 하는 것
`GameManager` 에 **지시(beat)를 문장으로 바꾸지 않고 그대로 돌려주는** `maybe_start_beat(text)` / `handle_beat(text)`.
기존 `maybe_start` / `handle` 은 그대로 — 로컬 봇 동작 불변.

### 실행 구조
호출어 대기는 동기. 깨면 대화 구간만 `asyncio.run(conversation(...))`, 대기로 가면 반환.

## 4. 한 턴의 흐름

1. 호출어 통과 → 연결 → `session.update`(프롬프트, marin, 받아 적기 모델, VAD 1,200ms, `create_response:false`)
   → 최근 6턴을 `conversation.item.create` 로 넣는다.
   - 호출 직후 이어 말했으면(`result.continued`) 그 프리롤을 24k 로 바꿔 먼저 보낸다. 아니면 인사말 캐시를 튼다.
2. `AudioSource` 에서 읽은 16k 를 24k 로 바꿔 `input_audio_buffer.append`. `MicGate` 가 닫혀 있으면 안 보낸다.
3. `conversation.item.input_audio_transcription.completed` → `realtime_turn.route(text, state)`:
   - 빈 글자 → 아무것도 안 함, 계속 듣는다
   - **sleep** → 캐시 문구 → 끊고 대기
   - **music** → `music.handle(text)` 의 안내 문장을 "이 문장 그대로 말해" 로 **대화 기록 밖 응답**(`conversation: "none"`)으로 요청
     → 재생 끝 → `action()` → 실패면 캐시 문구 → `standby` 면 끊고 대기
   - **game** → `handle_beat`/`maybe_start_beat` 의 지시를 그 응답의 `instructions` 로 붙여 `response.create`
     → 받은 글자에 `require` 낱말이 빠졌으면 로그(R2)
   - **chat** → `response.create`. 0.7초 안에 첫 소리가 없으면 맞장구 캐시를 튼다. 맞장구 중 답이 오면 맞장구 끝까지 기다렸다가 튼다
4. `response.output_audio.delta` 를 받는 대로 재생. 재생 상태를 `MicGate` 에 넘긴다.
5. `response.done` → 글자로 `safety.check_reply(reply, child_text=text)`(로그만), 이력에 추가, 계측 한 줄.
6. 노래 중 호출이었으면 답한 뒤 `music.resume_after_wake()` → 끊고 대기.
7. 마지막 활동 후 30초 → 캐시 문구 → 끊고 대기.

## 5. 고정 문구 캐시

인사말(대기 진입, 깨움), 잠들기, 무응답 대기, 되묻기, 노래 실패(`MUSIC_FAILED`), 끊김("잠깐 쉬었다 올게"), 맞장구 목록.
- 문구 목록은 기존 상수·`filler` 설정에서 읽는다(새로 쓰지 않는다).
- 키 = 문구 + 목소리 + 모델. 파일은 `~/.cache/jaeha_voice/<key>.wav`(24k).
- 기동 때 빠진 것만 만든다. 만들다 실패하면 경고하고 그 문구는 건너뛴다(기동은 계속).
- 이미 있는 `filler.FillerBank` 의 정규화·재생 방식을 따른다.

## 6. 오류 처리

| 상황 | 동작 |
|---|---|
| 깰 때 연결 실패 / 대화 중 끊김 | 끊김 캐시 문구 → 대기. 다음 호출에 재연결 |
| 서버 `error` 이벤트 | 코드·메시지 로그 → 위와 같이 |
| `response.create` 뒤 8초간 소리 없음 | `response.cancel` → 되묻기 캐시 → 계속 듣기 |
| API 키 없음 | 기동 중단(RuntimeError) |
| `wake.onnx.verify.mode` 가 embed 아님 | 기동 중단 — 전면 API 에는 whisper 가 없어 2단계 없이 돈다 |
| Ctrl+C / SIGTERM | 연결·노래·마이크 닫고 계측 요약 |

## 7. 계측

턴마다 기존 `logs/metrics_*.jsonl` 에 한 줄, 태그 `jetson-rt`(노트북 `pc-rt`), `metric_ver` 새 번호.

| 필드 | 뜻 |
|---|---|
| `perceived_s` | 말끝(`speech_stopped`) → 첫 소리. 로컬 체감 4.14s 와 비교 |
| `transcribe_s` | 말끝 → 받아 적기 완료 (R1) |
| `respond_first_s` | `response.create` → 첫 오디오 조각 |
| `kind` | chat / game / game_start / music / sleep |
| `filler` | 맞장구를 틀었는지 |
| `cost_usd`, `cached_tokens` | `response.done` 의 사용량 그대로(`realtime_probe.cost_usd`) |
| `safety` | `check_reply` 결과 목록 |
| `game_missing` | 놀이에서 빠진 `require` 낱말 (R2) |
| `rss_mb` | 메모리 |

## 8. 설정 · 배포 · 시험

### 설정 (`configs/model_paths.yaml`)
```yaml
pipeline: local            # local | realtime — 저장소 기본은 local
realtime:
  model: gpt-realtime-mini
  voice: marin
  transcribe_model: gpt-4o-transcribe
  silence_ms: 1200
  mic_pad_s: 0.15
  history_turns: 6
  filler_after_s: 0.7
  response_timeout_s: 8
  voice_cache_dir: ~/.cache/jaeha_voice
```
젯슨 `configs/local.yaml` 에만 `pipeline: realtime`. 되돌리기 = 그 한 줄 삭제.
`run.sh` 는 `pipeline` 을 읽어 `app.main` / `app.main_realtime` 을 고른다. `./run.sh check` 에 키·연결 확인 추가.

### 시험
- 단위(소켓 없음): `route` 네 갈래와 빈 글자, 이벤트 해석, 16k→24k, `MicGate`/`PcmAccumulator`(옮긴 뒤 기존 시험 그대로),
  맞장구 판단, 이력 항목, 캐시 키, 계측 한 줄, 기동 검사 두 가지.
- 흐름(가짜 소켓): 받아 적기 완료 → 올바른 `response.create`(놀이면 지시 포함, 노래면 `conversation:none`),
  끊김 → 대기, 8초 무응답 → cancel.
- **기존 시험 1,032개가 그대로 통과**해야 한다 — 로컬 경로 불변의 증거.

### 실기 순서
1. 노트북: `python -m app.main_realtime --no-wake` 로 자유대화·노래·놀이·잠들기 한 번씩.
2. 젯슨: 파일 전송, `local.yaml` 에 `pipeline: realtime`, 캐시 생성.
3. 젯슨 실기: 자유대화 5턴, 노래 틀기 + 노래 중 호출, 놀이 한 판, 잠들기, 30초 무응답.
   체감 지연·메모리·R1 을 보고서 기준(체감 4.14s, 시스템 메모리 85%)과 비교.
4. 결과와 보류를 9절에 기록.

## 9. 보류 기록

### 미리 예상한 위험
| # | 위험 | 막혔을 때 대안 |
|---|---|---|
| R1 | 말끝 → 받아 적기 완료 지연이 모든 턴에 붙는다(미측정) | 놀이 중이 아니면 답을 먼저 시작하고, 명령으로 판명되면 `response.cancel` |
| R2 | 놀이 필수 낱말을 소리가 나간 뒤에야 확인 | 빈도를 세고, 잦으면 놀이 턴만 전체 글자까지 소리 보류 |
| R3 | "그대로 말해" 를 모델이 바꿔 말함 | 곡 제목을 뺀 나머지를 캐시 문구로 |
| R4 | 깰 때마다 새 연결 시간 | 호출어 2단계 검증과 동시에 연결 시작 |
| R5 | 젯슨에서 재생 끊김 | 재생 버퍼 조정 |

### 진행 중 발생
(날짜 · 증상 · 원인 · 처리/대안 순으로 추가한다)
