#!/usr/bin/env bash
# CTranslate2 CUDA 소스빌드 (Jetson Orin Nano / JetPack 6.2 / CUDA 12.6 / aarch64)
#
# 왜: 젯슨 pip 인덱스의 ctranslate2 4.8.0 은 CPU 전용 빌드라 get_cuda_device_count()==0.
#     STT 가 CPU 로만 돌아 발화당 3.6초가 걸린다. 이게 현재 유일한 병목.
#
# 안전장치: 기존 CPU 휠을 먼저 받아 두고(~/ct2_rollback), 실패 시 즉시 복구 가능.
# 설치 위치: ~/.local/ct2 (sudo 불필요, 시스템 오염 없음)
# ⚠️ set -u 를 쓰면 안 된다. 젯슨 conda 의 activate.d/zz_ld.sh 가 LD_LIBRARY_PATH 를
#    미설정 상태로 참조해 activate 단계에서 즉사한다(실측 2026-08-05).
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

LOG_PREFIX() { echo "[$(date +%H:%M:%S)] $*"; }

SRC=$HOME/src/CTranslate2
PREFIX=$HOME/.local/ct2
ROLLBACK=$HOME/ct2_rollback
VERSION=v4.8.0          # 설치본과 같은 버전 — faster-whisper 호환을 바꾸지 않기 위해
JOBS=3                  # 7GB RAM 에서 nvcc 는 무겁다. -j6 은 OOM 위험

source ~/miniforge3/etc/profile.d/conda.sh
conda activate jaeha_bot
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

LOG_PREFIX "=== 0. 롤백용 CPU 휠 확보 ==="
mkdir -p "$ROLLBACK"
if [ -z "$(ls -A "$ROLLBACK" 2>/dev/null)" ]; then
  pip download "ctranslate2==4.8.0" --no-deps -d "$ROLLBACK" \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 || \
  pip download "ctranslate2==4.8.0" --no-deps -d "$ROLLBACK" || \
    LOG_PREFIX "경고: 롤백 휠 확보 실패(수동 복구 필요)"
fi
ls -la "$ROLLBACK"

LOG_PREFIX "=== 1. 빌드 도구 (conda env 안에만 설치) ==="
pip install -q cmake ninja
cmake --version | head -1

LOG_PREFIX "=== 2. 소스 받기 ($VERSION) ==="
mkdir -p "$(dirname "$SRC")"
if [ ! -d "$SRC" ]; then
  git clone --recursive --depth 1 -b "$VERSION" \
    https://github.com/OpenNMT/CTranslate2.git "$SRC"
else
  LOG_PREFIX "이미 존재 — 그대로 사용"
fi

LOG_PREFIX "=== 3. cmake 구성 (CUDA arch 87 = Orin) ==="
rm -rf "$SRC/build"
mkdir -p "$SRC/build"
cd "$SRC/build"
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DBUILD_CLI=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"

LOG_PREFIX "=== 4. 빌드 (-j$JOBS, 오래 걸림) ==="
cmake --build . -j "$JOBS"

LOG_PREFIX "=== 5. 설치 -> $PREFIX ==="
cmake --install .

LOG_PREFIX "=== 6. 파이썬 바인딩 ==="
cd "$SRC/python"
export CTRANSLATE2_ROOT="$PREFIX"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
pip install -q -r install_requirements.txt
python setup.py -q bdist_wheel
ls -la dist/

LOG_PREFIX "=== 7. 설치 및 검증 ==="
pip install --force-reinstall --no-deps dist/*.whl
python - <<'EOF'
import ctranslate2 as c
n = c.get_cuda_device_count()
print("ctranslate2", c.__version__, "| cuda_device_count =", n)
print("성공: GPU 사용 가능" if n > 0 else "실패: 여전히 0 — CPU 빌드")
EOF

LOG_PREFIX "=== 완료 ==="
echo "런타임에 LD_LIBRARY_PATH 에 $PREFIX/lib 이 필요할 수 있음"
echo "롤백: pip install --force-reinstall --no-deps $ROLLBACK/*.whl"
