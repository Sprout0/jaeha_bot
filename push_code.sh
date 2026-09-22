#!/usr/bin/env bash
# ============================================================
# push_code.sh — 노트북 코드를 젯슨으로 밀어넣기
# ------------------------------------------------------------
# 대상: app/ configs/ scenarios/ + 아래 FILES (봇 실행과 젯슨에서 쓰는 도구만)
# 안 건드림: models/ (크고 드물게 바뀜 → push_model.sh 로)
#
# 사용법:  JETSON=사용자@주소 bash push_code.sh
# ============================================================
set -e

JETSON="${JETSON:?젯슨 주소를 지정할 것 — 예: JETSON=사용자@주소 bash push_code.sh}"   # 계정@주소(개인 주소는 저장소에 두지 않는다)
REMOTE='~/jaeha_bot'              # 젯슨 안의 프로젝트 폴더

# 이 스크립트가 있는 폴더(=프로젝트 루트)로 이동 → 어디서 실행하든 동작
cd "$(dirname "$0")"

DIRS="app configs scenarios"
# 폴더째 보내지 않는 것 — 평가·학습·비교 도구와 평가셋은 노트북(또는 Colab)에서만 쓴다(2026-09-22 정리).
# 젯슨에서 쓰는 도구: 호출어 본보기 등록·소음 녹음·헛깨움 측정·출력 진단·라이브 확인·음악 중 호출어·재생 여유.
FILES="run.sh requirements.txt requirements-realtime.txt README.md .env.example
Dockerfile docker-compose.yml .dockerignore data/finetune_seed.jsonl
tools/enroll_wake.py tools/record_noise.py tools/record_wake_real.py tools/gen_wake_supertonic.py
tools/measure_wake_fp.py tools/probe_audio_out.py tools/realtime_live.py tools/realtime_probe.py
tools/wake_under_music.py tools/measure_play_pad.py tools/jetson"

echo "== 코드 밀어넣기 -> $JETSON:$REMOTE =="

# 원격 폴더 없으면 만들어 둠(첫 전송 안전)
ssh "$JETSON" "mkdir -p $REMOTE/app $REMOTE/configs $REMOTE/scenarios $REMOTE/data $REMOTE/tools"

# configs/local.yaml 은 '이 기계 전용' 덮어쓰기(노트북=cpu 등)라 절대 보내면 안 된다.
# 보내는 순간 젯슨 STT 가 CPU 로 떨어져 GPU 화가 조용히 무효가 된다.
SKIP="local.yaml"

for d in $DIRS; do
  if [ -d "$d" ]; then
    echo "-- $d/ 전송..."
    for f in "$d"/*; do
      [ -e "$f" ] || continue
      [ "$(basename "$f")" = "__pycache__" ] && continue
      if [ "$(basename "$f")" = "$SKIP" ]; then
        echo "   (건너뜀: $f — 기계 전용 설정)"
        continue
      fi
      scp -r "$f" "$JETSON:$REMOTE/$d/"
    done
  fi
done

echo "-- 개별 파일 전송..."
for f in $FILES; do
  [ -e "$f" ] || { echo "   (없음: $f)"; continue; }
  scp -r "$f" "$JETSON:$REMOTE/$(dirname "$f")/"
done
ssh "$JETSON" "chmod +x $REMOTE/run.sh $REMOTE/tools/jetson/*.sh"

echo ""
echo "== 완료 =="
echo "젯슨에서 실행:"
echo "  ssh $JETSON"
echo "  cd ~/jaeha_bot && ./run.sh check && ./run.sh"
