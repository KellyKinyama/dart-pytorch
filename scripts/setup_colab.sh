#!/usr/bin/env bash
# setup_colab.sh - one-shot setup for running dart-pytorch on a free
# cloud GPU (Google Colab, Kaggle Notebooks, Paperspace Gradient,
# Lightning AI Studio, etc.). All of these are Debian/Ubuntu-based and
# ship with a preinstalled CUDA toolkit + NVIDIA driver, so all we need
# to add is (a) the Dart SDK and (b) the compiled native library.
#
# Usage from the repo root:
#
#   bash scripts/setup_colab.sh
#
# Or in a Colab cell:
#
#   !bash scripts/setup_colab.sh
#
# See docs/colab.md for full copy/paste-ready notebook cells.

set -euo pipefail

echo ">>> [1/4] verifying CUDA toolkit"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "!! nvcc not on PATH."
    echo "!! On Colab: Runtime -> Change runtime type -> Hardware accelerator = GPU."
    echo "!! On Kaggle: Settings (right sidebar) -> Accelerator -> GPU T4 x2 (or P100)."
    exit 1
fi
nvcc --version | tail -n 1
nvidia-smi -L 2>/dev/null | head -n 1 || echo "(no nvidia-smi output, but nvcc present -> compile-only ok)"

echo ">>> [2/4] installing Dart SDK (if missing)"
if ! command -v dart >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq apt-transport-https wget gnupg ca-certificates
    wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/dart.gpg
    echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' \
        > /etc/apt/sources.list.d/dart_stable.list
    apt-get update -qq
    apt-get install -y -qq dart
fi
dart --version

echo ">>> [3/4] building native/lib/libmat_mul.so with nvcc"
mkdir -p native/lib
nvcc --shared -Xcompiler -fPIC \
    -o native/lib/libmat_mul.so \
    lib/native/src/engine.cu
ls -lh native/lib/libmat_mul.so

echo ">>> [4/4] dart pub get"
dart pub get

echo
echo "Done. Try:"
echo "  dart run bin/word2vec_demo.dart --max-chars=1200000 --epochs=8 --window=4 --vocab-size=2048"
