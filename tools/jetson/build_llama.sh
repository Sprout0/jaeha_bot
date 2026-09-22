#!/bin/bash
# 젯슨(Orin, sm_87)에서 llama-cpp-python을 CUDA로 소스 빌드.
# prebuilt 0.3.14가 unified KV cache 회귀 버그로 생성 시 크래시 → 최신 소스로 교체.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate jaeha_bot
export PATH=/usr/local/cuda/bin:$PATH
echo "=== nvcc ==="; nvcc --version | tail -1
echo "=== uninstall broken wheel ==="
pip uninstall -y llama-cpp-python || true
echo "=== build from source (GGML_CUDA=on, arch 87) ==="
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87"
export FORCE_CMAKE=1
pip install llama-cpp-python \
  --no-binary llama-cpp-python \
  --no-cache-dir --force-reinstall --no-deps \
  --index-url https://pypi.org/simple
echo "BUILD_EXIT=$?"
python -c "import llama_cpp; print('INSTALLED', llama_cpp.__version__)"
