#!/usr/bin/env bash
# ============================================================
# push_code.sh — 노트북 코드를 젯슨으로 밀어넣기
# ------------------------------------------------------------
# 대상: app/ configs/ scenarios/ data/  (전부 작은 텍스트/코드)
# 안 건드림: models/ (크고 드물게 바뀜 → push_model.sh 로)
#
# 사용법:  bash push_code.sh        (또는  ./push_code.sh)
# ============================================================
set -e

JETSON=jaeha_bot@100.65.22.17     # 젯슨 계정@Tailscale IP (바뀌면 여기만 수정)
REMOTE='~/jaeha_bot'              # 젯슨 안의 프로젝트 폴더

# 이 스크립트가 있는 폴더(=프로젝트 루트)로 이동 → 어디서 실행하든 동작
cd "$(dirname "$0")"

DIRS="app configs scenarios data"

echo "== 코드 밀어넣기 -> $JETSON:$REMOTE =="

# 원격 폴더 없으면 만들어 둠(첫 전송 안전)
ssh "$JETSON" "mkdir -p $REMOTE/app $REMOTE/configs $REMOTE/scenarios $REMOTE/data"

for d in $DIRS; do
  if [ -d "$d" ]; then
    echo "-- $d/ 전송..."
    scp -r "$d"/* "$JETSON:$REMOTE/$d/"
  fi
done

echo ""
echo "== 완료 =="
echo "젯슨에서 실행:"
echo "  ssh $JETSON"
echo "  conda activate jaeha_bot && cd ~/jaeha_bot && python -m app.main"
