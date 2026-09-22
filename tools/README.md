# tools/ — 봇 실행에는 쓰지 않는 도구

봇 자체는 `app/` 만으로 돈다. 여기 있는 것은 **젯슨을 관리하는 도구**와, 설계를 정할 때
쓴 **측정·평가 도구**(보고서 수치의 근거라 남겨 둔다)다. 대부분 저장소 루트에서
`python tools/<이름>.py --help` 로 쓰는 법이 나온다.

## 1. 젯슨에서 쓰는 것 — `push_code.sh` 가 젯슨으로 보낸다

| 도구 | 하는 일 |
|---|---|
| `enroll_wake.py` | **호출어 본보기 등록**(`templates_haitid.npy`). 봇을 쓸 사람이 바뀌면 먼저 한다. 소음 녹음으로 컷도 정한다 |
| `record_noise.py` | 거실 소음을 길게 녹음 — 헛깨움 컷의 하한을 정할 데이터 |
| `measure_wake_fp.py` | 실제 소음으로 헛깨움 횟수를 잰다 |
| `record_wake_real.py` | 실제 목소리로 호출어를 녹음하고 감지기가 못 잡는 이유를 가른다 (`gen_wake_supertonic.py` 를 가져다 쓴다) |
| `probe_audio_out.py` | 소리가 안 날 때 출력이 어디서 막히는지 층별로 본다 |
| `realtime_live.py` | Realtime 목소리·속도를 마이크↔스피커로 바로 들어 본다 (`realtime_probe.py` 를 가져다 쓴다) |
| `wake_under_music.py` | 노래가 나오는 중에 호출어를 알아듣는가. `python -m tools.wake_under_music` 으로 실행. **스피커가 생기면 잴 것** |
| `measure_play_pad.py` | 재생 앞 무음(`PLAY_PAD_S`)이 얼마나 필요한가. **스피커가 생기면 잴 것** |
| `jetson/build_ct2_cuda.sh`, `jetson/build_llama.sh` | 로컬 경로용 CTranslate2·llama-cpp CUDA 소스 빌드 (README 4절 C) |

## 2. 측정·평가 — 결정의 근거 (노트북에서)

| 주제 | 도구 | 결과가 있는 곳 |
|---|---|---|
| 말 끝 판정 | `turn_end_probe.py`(전면 API, 아이 153발화), `eval_turn_detect.py`(Smart Turn — 기각) | `reports/turn/` |
| 전면 API | `realtime_probe.py`(지연·비용·아이 말 인식) | `reports/bench/`, `docs/final/api-전환안.md` |
| LLM | `eval_llm.py`(품질·안전 채점), `bench_llm_latency.py`, `bench_pipeline.py`, `bench_fewshot_form.py`, `bench_fewshot_leak.py` | `reports/eval/`, `reports/bench/` |
| 음성 합성 | `profile_tts.py`, `measure_first_sentence.py`(문장 스트리밍 — 기각), `check_filler.py` | `reports/bench/` |
| 목소리 고르기 | `voice_try.py`, `voice_tournament.py`, `voice_knockout.py`, `tts_audition.py` (젯슨 스피커로 들어야 한다 — 필요하면 scp) | `data/voice_blind/` |
| 호출어 검증 | `wake_ab_verify.py`, `probe_verify_model.py`, `probe_verify_precision.py`, `probe_wake_word.py`, `probe_wake_speed.py` | `reports/wake/` |
| 메모리 | `probe_memory.py` | `docs/final/report.md` |

## 3. 호출어 학습 데이터 (노트북 → Colab)

`gen_wake_supertonic.py`(합성) → `wake_qc.py`(검수) → `check_negative_labels.py` → `merge_wake_clips.py` →
`wake_cell_balance.py` → `split_wake_dataset.py` → `sanity_wake_data.py` → Colab `colab_wake_train_v6.ipynb`
(학습, Hugging Face 토큰은 Colab 비밀값). `colab_wake_ab.ipynb` 는 합성 음성 A/B 진단. 설정은 `configs/wake/`.

## 4. 제출 문서 만들기

| 도구 | 하는 일 |
|---|---|
| `build_docs_docx.py`, `docx_reference.py`, `update_docx_fields.ps1` | markdown 보고서 → docx/pdf (`docs/final/README.md`, `docs/submission-api/README.md`) |
