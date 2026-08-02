#!/usr/bin/env bash
# ============================================================
# push_model.sh — 새 모델 파일 하나를 젯슨으로 밀어넣기
# ------------------------------------------------------------
# 모델은 크니까 "새로 만들었을 때만" 이걸로 밀어넣는다.
# 규칙: 모델은 항상 '새 이름'으로 저장 → 덮어쓰기/중첩 사고 방지 + 롤백 쉬움.
#
# 사용법:
#   bash push_model.sh models/exaone-3.5-2.4b-ft-v1.gguf   (파일)
#   bash push_model.sh models/whisper-small-ko-ct2          (폴더도 가능)
#
# 밀어넣은 뒤:
#   1) configs/model_paths.yaml 의 경로를 이 새 모델로 교체(옛 경로는 주석으로 남겨 롤백)
#   2) bash push_code.sh  로 그 설정 변경을 젯슨에 반영
# ============================================================
set -e

JETSON=jaeha_bot@100.65.22.17
REMOTE='~/jaeha_bot'

cd "$(dirname "$0")"

SRC="$1"
if [ -z "$SRC" ]; then
  echo "사용법: bash push_model.sh <모델경로>"
  echo "  예:   bash push_model.sh models/exaone-3.5-2.4b-ft-v1.gguf"
  exit 1
fi
if [ ! -e "$SRC" ]; then
  echo "그런 파일 없음: $SRC"
  exit 1
fi

echo "== 모델 전송: $SRC -> $JETSON:$REMOTE/models/ =="
ssh "$JETSON" "mkdir -p $REMOTE/models"
scp -r "$SRC" "$JETSON:$REMOTE/models/"

echo ""
echo "== 전송 완료 =="
echo "다음:"
echo "  1) configs/model_paths.yaml 경로를 새 모델로 교체 (옛 경로는 주석으로 남기기)"
echo "  2) bash push_code.sh   (설정 변경을 젯슨에 반영)"
