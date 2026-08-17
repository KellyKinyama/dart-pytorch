#!/usr/bin/env bash
# Build the native CUDA library `libmat_mul.{so,dll,dylib}` for the current
# host platform. Requires a working `nvcc` on PATH.
#
# Usage:
#   scripts/build_native.sh              # Release build for host OS
#   scripts/build_native.sh --debug      # Debug build (unoptimized, -G)
#   scripts/build_native.sh --arch=sm_86 # Force a specific SM arch
#
# Outputs to `native/lib/`. On WSL2 build for Linux; use the .bat script
# on Windows and the .command script on macOS (though CUDA on macOS is
# unsupported since CUDA 11).

set -euo pipefail

cd "$(dirname "$0")/.."

DEBUG=0
ARCH=""
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG=1 ;;
        --arch=*) ARCH="${arg#--arch=}" ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH. Install the CUDA toolkit first." >&2
    exit 1
fi

case "$(uname -s)" in
    Linux*)  OUT="native/lib/libmat_mul.so" ;;
    Darwin*) OUT="native/lib/libmat_mul.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) OUT="native/lib/mat_mul.dll" ;;
    *) echo "unsupported host OS: $(uname -s)" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$OUT")"

FLAGS=(--shared -Xcompiler -fPIC -o "$OUT")
if [[ $DEBUG -eq 1 ]]; then
    FLAGS+=(-G -O0)
else
    FLAGS+=(-O3)
fi
if [[ -n "$ARCH" ]]; then
    FLAGS+=(-arch="$ARCH")
fi

echo "building $OUT with nvcc ${FLAGS[*]}"
nvcc "${FLAGS[@]}" lib/native/src/engine.cu

ls -lh "$OUT"
