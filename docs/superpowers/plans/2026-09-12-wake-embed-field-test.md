# 09-12 할 일 — 임베딩 호출어 실기 판정 → 저장소 기본값 결정

전제: 젯슨은 `configs/local.yaml` 로 `mode: embed`, 컷 0.85 가 켜져 있다(백업 `~/local.yaml.bak_20260911_embed`).
저장소 기본은 `mode: whisper` 그대로. 근거 자료는 `reports/wake/ab_verify_20260911.md`.

## 순서

| # | 할 일 | 누가 | 걸리는 시간 | 합격선 |
|---|---|---|---|---|
| A | 20번 부르기 시험 (젯슨) | 사람이 부른다 | 3분 | 18/20 이상 |
| B | 헛깨움 세기 (봇 켜 두기) | 켜 두기만 | 1시간 이상 | 시간당 1회 이하 |
| C | 1단계 단독 함정 막기 (코드) | Claude | 30분 | 테스트 통과 |
| D | 저장소 기본값 결정 + 기록 | 같이 | 15분 | — |
| E | `wake_under_music.py` 임베딩 창 넘기기 (코드, 작음) | Claude | 10분 | — |
| F | 마무리 문서에 09-11 결과 넣기 | Claude | 1시간 | — |
| G | KIPRIS 선행조사 | 사람 | — | — |
| H | 교수님께 물을 것 | 사람 | — | — |

A·B 는 C 를 기다리지 않는다(젯슨 설정은 이미 올바르다 — 본보기 있고 `enabled: true`).

---

## A. 20번 부르기 시험

봇을 끈 상태에서(마이크 장치가 하나라서 둘이 같이 못 연다):

```bash
cd ~/jaeha_bot && source ~/miniforge3/etc/profile.d/conda.sh && conda activate jaeha_bot && python -m tools.wake_under_music --volumes 0 --trials 20 --force
```

- 시작할 때 음악이 5초 나오는데, 음악 시험용 점검이라 무시해도 된다(`--force`).
- `Enter` 한 번 누르고, `[k/20] 지금 부르세요` 가 뜰 때마다 "하이 티드".
- 🟢 잡음이면 0.4초 뒤 다음 차례, 🔴 놓침이면 12초를 기다린다.
- 마지막 줄 `볼륨 0%: N/20`. 원자료는 `reports/jetson/wake_under_music_<시각>.jsonl`.
- 콘솔의 `[검증] 임베딩 단독 통과/기각 — 유사도 0.xxx` 로 컷과의 여유를 본다.

떨어졌을 때(17 이하): 기각 줄의 유사도를 모은다.
- 0.80~0.85 에 몰려 있으면 → 컷 문제. 0.85 는 이미 소음 최대치 0.872 보다 아래라 더 내리면 안 된다.
  본보기를 다시 등록한다(오늘 목소리로 5건).
- 1단계에서 떨어졌으면(검증 줄이 아예 없음) → 임베딩 탓이 아니다. 흘려서 부른 한계로 기록한다.

가져오기(노트북):

```bash
scp jaeha_bot@<젯슨IP>:~/jaeha_bot/reports/jetson/wake_under_music_*.jsonl reports/jetson/
```

## B. 헛깨움 세기

```bash
cd ~/jaeha_bot && ./run.sh
```

음악은 틀지 않고 평소대로 지낸다. 끝나면 B = 봇 켠 시각, S/E = 일부러 부른 구간을 채운다(안 불렀으면 S=E=99:99).

```bash
B=14:00; S=14:02; E=14:17; L=~/jaeha_bot/logs/jaeha_$(date +%Y%m%d).log; grep '→ 깨어남' $L | awk -v b="$B" -v s="$S" -v e="$E" '{t=substr($2,1,5); if (t<b) next; if (t>=s && t<=e) a++; else f++} END {print "부른 구간 깨어남:", a+0, "  헛깨움(그 밖):", f+0}'
```

헛깨움 ÷ 켜 둔 시간 = 시간당 헛깨움. 합격선은 1회 이하.
시간을 못 내면 오프라인 45분 결과(1회 = 1.33/h, 같은 코드)를 근거로 쓰되, 문서에는 "실기 미계측"으로 적는다.

## C. 1단계 단독 함정 막기 — ✅ 09-14 함(스위트 1,032개 통과)

`make_detector` 가 try 밖에서 임베딩 대조를 먼저 만들고, embed 인데 None 이면 RuntimeError 로 멈춘다.
기존 시험 4개는 본보기 없이 embed 를 쓰고 있어 `_rescue_cfg(tmp_path)` 를 넣어 고쳤다.
🔴 **아직 젯슨에 안 올렸다** — 배포는 A·B 판정 뒤에 `app/wake.py` 한 파일만 scp.


**문제:** `mode: embed` 인데 `embed_rescue.enabled: false` 거나 본보기 파일이 없으면
`_make_embed_rescue` 가 None 을 준다. whisper 검증기도 None 이라 **2단계 없이 1단계 단독**으로 돈다
(거실 시간당 160회 헛깨움, 08-26 실측). 로그엔 경고 한 줄뿐이다.

**고칠 곳:** `app/wake.py` `make_detector`. 임베딩 장치를 **try 밖에서** 먼저 만들고, embed 인데 None 이면 멈춘다.
try 안에 두면 폴백 except 가 삼켜 "STT 폴백"으로 둔갑한다(모드 오타 때 실제로 그랬다).

```python
    # (mode 검사 바로 아래, try 위에)
    # 🔴 2026-09-12 embed 는 임베딩 대조가 유일한 2단계다. 그게 없으면 whisper 도 없으니
    #    1단계 단독(거실 시간당 160회)으로 조용히 돈다. 기동을 멈춘다.
    embed_rescue = _make_embed_rescue(vcfg)
    if mode == "embed" and embed_rescue is None:
        raise RuntimeError(
            "wake.onnx.verify.mode 가 embed 인데 임베딩 대조를 못 만들었다 — "
            "embed_rescue.enabled 가 true 인지, 본보기(templates) 파일이 있는지 확인할 것. "
            "이대로면 2단계 없이 1단계 단독으로 돈다")

    try:
        ...
            embed_rescue=embed_rescue,      # 기존 _make_embed_rescue(vcfg) 호출을 바꾼다
```

`load_rescue` 는 파일 없음·깨짐을 자기 안에서 잡아 None 을 주므로, try 밖으로 옮겨도 새 예외는 안 생긴다.

**테스트 (`tests/test_wake_factory.py`, 먼저 빨갛게):**

```python
def _templates(tmp_path):
    p = tmp_path / "t.npy"
    np.save(p, np.ones((1, 16 * 96), dtype=np.float32))  # 본보기 = 임베딩 16개를 펼친 줄(wake_embed.make_template)
    return str(p)


def test_mode가_embed인데_rescue가_꺼져있으면_죽는다(monkeypatch):
    _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"mode": "embed"}}}
    with pytest.raises(RuntimeError):
        make_detector(cfg, stt=FakeStt([]), source=object())


def test_mode가_embed인데_본보기가_없으면_죽는다(monkeypatch, tmp_path):
    _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드",
           "onnx": {"verify": {"mode": "embed", "embed_rescue": {
               "enabled": True, "templates": str(tmp_path / "없음.npy")}}}}
    with pytest.raises(RuntimeError):      # STT 폴백으로 둔갑하면 안 된다
        make_detector(cfg, stt=FakeStt([]), source=object())


def test_mode가_whisper면_rescue가_꺼져도_그대로다(monkeypatch):
    seen = _capture_kwargs(monkeypatch)
    cfg = {"detector": "onnx", "word": "하이티드", "onnx": {"verify": {}}}
    make_detector(cfg, stt=FakeStt([]), source=object())
    assert seen["embed_rescue"] is None
```

⚠️ **기존 테스트 4개가 같이 깨진다** — `mode: embed` 를 본보기 없이 쓰고 있다:
`test_mode가_embed면_whisper를_안_만든다`, `test_mode가_embed면_stt가_없어도_만들어진다`,
`test_embed_settle_s_가_감지기로_넘어간다`, `test_stt가_없는데_onnx가_죽으면_조용히_넘어가지_않는다`.
이들 설정에 `"embed_rescue": {"enabled": True, "templates": _templates(tmp_path)}` 를 넣는다
(마지막 것은 RuntimeError 를 기대하므로 지금도 통과하지만, 이유가 바뀌지 않게 같이 넣는다).

`tools/wake_ab_verify.py` 의 `build` 는 이미 rescue 없음을 따로 막고 있어 영향 없음 — 확인만.

젯슨 배포: `app/wake.py` 한 파일 md5 비교 후 scp. 젯슨 설정은 조건을 만족하므로 기동이 그대로여야 한다
(봇 켜서 `임베딩 단독` 줄이 뜨는지 확인).

## D. 저장소 기본값 결정

A 합격(+ B 합격 또는 오프라인 근거) 이면:

- **권장: 저장소 기본은 whisper 로 두고, 임베딩은 기계별 `local.yaml` 로 켠다.**
  이유 — 본보기는 화자 종속이고 노트북엔 본보기 파일이 없다. 기본을 embed 로 바꾸면 C 의 검사 때문에
  노트북 기동이 멈춘다(그게 맞는 동작이지만, 쓰는 사람마다 등록이 먼저 필요해진다).
- 시연 기계(젯슨) 설정은 이미 embed. `configs/model_paths.yaml` 의 verify 주석에
  "젯슨은 local.yaml 로 embed(컷 0.85) — 09-12 실기 N/20" 을 적는다.
- 기록: `reports/wake/ab_verify_20260911.md` 끝에 "실기 판정" 절 추가 + 계획 문서
  `2026-09-02-embed-only-wake-verify.md` Task 4 체크. 메모리 갱신.

A 불합격이면: 젯슨 `local.yaml` 을 백업으로 되돌린다(`cp ~/local.yaml.bak_20260911_embed ~/jaeha_bot/configs/local.yaml`
— 되돌리기 전에 그 뒤 추가된 `youtube: enabled: true` 가 백업에 있는지 먼저 본다).

## E. `tools/wake_under_music.py` — 봇과 같은 창

지금은 `AudioSource(...)` 에 `embed_window` 를 안 넘겨 기본 3.0 을 쓴다. 설정과 우연히 같을 뿐이다.
`app/main.py` 와 똑같이:

```python
vcfg = (wcfg.get("onnx") or {}).get("verify") or {}
AudioSource(..., verify_window=float(vcfg.get("window_s", 2.0)),
            embed_window=float(vcfg.get("embed_window_s", 3.0)))
```

다른 세션이 만든 파일 — 작은 독립 커밋으로.

## F. 마무리 문서

`docs(final)` 커밋들이 건드린 파일을 찾아(`git log --stat --grep "docs(final)"`) 호출어 장에 넣을 것:

1. A/B 표(whisper 23/35·2.67/h vs 임베딩 0.85 25/35·1.33/h, 검증 0.78s → 0.08s)
2. 임베딩 결함과 수정(2초 창 → 임베딩 10개 < 16개, 유사도 늘 0) — 테스트가 `passes()` 를 가짜로 바꿔 못 잡았다
3. 지금 봇(whisper)도 헛깨움 합격선 초과
4. 흘려서 부른 호출은 1단계에서 떨어진다 — 한계
5. A·B 실기 결과(나오면)

숫자는 원자료 JSON 과 대조해서 넣는다(09-11 원본 대조에서 틀린 숫자 셋이 나왔다).

## G. KIPRIS 선행조사

후보 둘: ③ 호출어 "1단계 임베딩 재사용 + 후보 뒤 0.48초 창"(CN116884407A 와 선을 그을 것),
⑤ S2S 재생 전 게이팅(Moshi 2024 와 선을 그을 것). 검색어 예: `호출어 AND 임베딩 AND 검증`,
`웨이크업 AND 2단계`, `음성합성 AND 출력 AND 차단 AND 텍스트`.

## H. 교수님께 물을 것

1. 합격선 — 실기 시연 / 영상 / 문서 중 무엇인가
2. 아이 목소리를 해외 서버(S2S)로 보내도 되는가
3. 특허 범위 — 좁게라도 낼지, 논문형으로 갈지
4. TensorRT 유지/해제 (메모리 85%, 폴백이 못 올라온다)

## 나중 / 위험

- 노래 중 호출어 — 스피커가 있어야 잰다.
- 젯슨 메모리 다시 재기(유튜브 플레이어 추가 뒤).
- 설정 주석의 "44건"(08-12) — 젯슨에만 있는 64건 녹음과 맞는지 확인.
