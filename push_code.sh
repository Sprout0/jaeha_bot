#!/usr/bin/env bash
# ============================================================
# push_code.sh — 노트북 코드를 젯슨으로 밀어넣기
# ------------------------------------------------------------
# 대상: app/ configs/ scenarios/ data/  (전부 작은 텍스트/코드)
# 안 건드림: models/ (크고 드물게 바뀜 → push_model.sh 로)
#
# 사용법:  JETSON=사용자@주소 bash push_code.sh
# ============================================================
set -e

JETSON="${JETSON:?젯슨 주소를 지정할 것 — 예: JETSON=사용자@주소 bash push_code.sh}"   # 계정@주소(개인 주소는 저장소에 두지 않는다)
REMOTE='~/jaeha_bot'              # 젯슨 안의 프로젝트 폴더

# 이 스크립트가 있는 폴더(=프로젝트 루트)로 이동 → 어디서 실행하든 동작
cd "$(dirname "$0")"

# tools: 평가·점검 스크립트. 런타임은 아니지만 젯슨에서 돌려야 할 때가 있다
# (로컬 EXAONE 기준선 평가는 이 노트북 CPU 로는 너무 느려 젯슨에서 재야 한다).
DIRS="app configs scenarios data tools"

echo "== 코드 밀어넣기 -> $JETSON:$REMOTE =="

# 원격 폴더 없으면 만들어 둠(첫 전송 안전)
ssh "$JETSON" "mkdir -p $REMOTE/app $REMOTE/configs $REMOTE/scenarios $REMOTE/data"

# configs/local.yaml 은 '이 기계 전용' 덮어쓰기(노트북=cpu 등)라 절대 보내면 안 된다.
# 보내는 순간 젯슨 STT 가 CPU 로 떨어져 GPU 화가 조용히 무효가 된다.
SKIP="local.yaml"

for d in $DIRS; do
  if [ -d "$d" ]; then
    echo "-- $d/ 전송..."
    for f in "$d"/*; do
      [ -e "$f" ] || continue
      if [ "$(basename "$f")" = "$SKIP" ]; then
        echo "   (건너뜀: $f — 기계 전용 설정)"
        continue
      fi
      scp -r "$f" "$JETSON:$REMOTE/$d/"
    done
  fi
done

echo ""
echo "== 완료 =="
echo "젯슨에서 실행:"
echo "  ssh $JETSON"
echo "  cd ~/jaeha_bot && ./run.sh check && ./run.sh"
