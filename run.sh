#!/usr/bin/env bash
# 재하봇 1 실행 스크립트 (독립 기동 — 가이드 5)
#
# 🔴 2026-09-01 고침. 여기가 `python3 -m app.main` 이었다.
#    젯슨의 시스템 python3 에는 numpy 말고 아무것도 없어서 첫 import 에서
#    `ModuleNotFoundError: No module named 'sounddevice'` 로 즉사한다.
#    여태 안 드러난 이유는 **아무도 이 스크립트를 안 썼기 때문**이다 —
#    손으로 `conda activate jaeha_bot && python -m app.main` 을 치고 있었다
#    (젯슨 bash_history: 그 형태 11회, run.sh 0회).
#    그런데 이 파일의 목적이 '독립 기동'이라, 시연에서 이걸 누르는 순간 실패한다.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-local}"                  # local | docker | dev | check | setup-youtube
ENV_NAME="${JAEHA_ENV:-jaeha_bot}"  # 다른 이름을 쓰면 JAEHA_ENV=... ./run.sh

# 이 환경의 python 을 찾는다.
# `conda activate` 를 부르지 않고 **실행 파일을 직접** 쓴다 — 비대화형 셸에서는
# conda 훅이 안 걸리는 일이 잦아서(그게 바로 위 사고의 사촌이다) 이 편이 튼튼하다.
find_python() {
  local base
  for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -x "$base/envs/$ENV_NAME/bin/python" ] && { printf '%s\n' "$base/envs/$ENV_NAME/bin/python"; return 0; }
    [ -x "$base/envs/$ENV_NAME/python.exe" ] && { printf '%s\n' "$base/envs/$ENV_NAME/python.exe"; return 0; }
  done
  # 이미 그 환경을 켜 둔 채로 부른 경우
  [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] && { printf 'python\n'; return 0; }
  return 1
}

resolve_or_die() {
  local py
  if ! py="$(find_python)"; then
    echo "❌ conda 환경 '$ENV_NAME' 의 python 을 못 찾았다." >&2
    echo "   환경을 만들었는지 확인하거나, 이름이 다르면 JAEHA_ENV=<이름> ./run.sh" >&2
    exit 1
  fi
  printf '%s\n' "$py"
}

case "$MODE" in
  local)   # 보드/PC 네이티브 실행
    PY="$(resolve_or_die)"
    echo "▶ $PY -m app.main"
    exec "$PY" -m app.main ;;
  check)   # 봇을 띄우지 않고 **기동 준비만** 확인한다 (시연 직전에 쓰라고 만든 것)
    PY="$(resolve_or_die)"
    echo "python: $PY"
    "$PY" - <<'PYEOF'
import importlib.util, sys
need = ["numpy", "yaml", "sounddevice", "soundfile", "onnxruntime"]
bad = [m for m in need if not importlib.util.find_spec(m)]
print("모듈:", "전부 있음" if not bad else f"❌ 없음 -> {bad}")
sys.exit(1 if bad else 0)
PYEOF
    "$PY" -m app.audio_player | head -4 ;;
  setup-youtube)  # 노래 틀기 준비물 점검 + ~/.asoundrc 설치 (2026-09-11)
    # 공유 출력 장치(respk)를 ~/.asoundrc 에 넣는다. 기본 장치는 안 바꾸고 이름만 추가한다.
    # 🔴 이미 있는 ~/.asoundrc 는 덮어쓰지 않는다 — 누가 무엇을 넣어 뒀는지 모른다.
    SRC=configs/alsa/respk.asoundrc
    if [ -f "$HOME/.asoundrc" ]; then
      if grep -q "pcm.respk_dmix" "$HOME/.asoundrc"; then
        echo "✅ ~/.asoundrc 에 respk 가 이미 있다"
      else
        echo "❌ ~/.asoundrc 가 이미 있고 respk 가 없다 — 덮어쓰지 않는다. $SRC 를 직접 합칠 것"; exit 1
      fi
    else
      cp "$SRC" "$HOME/.asoundrc" && echo "✅ ~/.asoundrc 설치 ($SRC)"
    fi
    for b in Xvfb pulseaudio pactl snap; do
      command -v "$b" >/dev/null && echo "✅ $b" || echo "❌ $b 없음"
    done
    snap list chromium >/dev/null 2>&1 && echo "✅ chromium (snap)" || echo "❌ chromium 없음 — sudo snap install chromium"
    grep -qE '^\s*YOUTUBE_DATA_KEY\s*=' .env 2>/dev/null && echo "✅ .env 에 YOUTUBE_DATA_KEY" \
      || echo "❌ .env 에 YOUTUBE_DATA_KEY 없음 — 검색이 안 된다(로컬 음원만 튼다)"
    PY="$(resolve_or_die)"
    "$PY" -c "import sounddevice as sd; n=[d['name'] for d in sd.query_devices()]; print('✅ 공유 출력 respk 보임' if 'respk' in n else '❌ respk 가 장치 목록에 없다')" 2>/dev/null
    echo "켜기: 이 기계의 configs/local.yaml 에  youtube: {enabled: true}" ;;
  docker)  # Jetson 컨테이너
    docker compose up --build ;;
  dev)     # PC 개발용 컨테이너
    docker build -f Dockerfile.dev -t jaeha_bot:dev .
    docker run --rm -it --device /dev/snd jaeha_bot:dev ;;
  *)
    echo "usage: ./run.sh [local|check|setup-youtube|docker|dev]"; exit 1 ;;
esac
