#!/bin/bash
# Builds the offline golden generator against a llama.cpp build tree. Not part of cargo.
set -euo pipefail
LLAMA=${LLAMA:-$HOME/llama.cpp}
BUILD=${BUILD:-$LLAMA/build-qx}
OUT=${OUT:-$(dirname "$0")/golden}
g++ -O2 -std=c++17 "$(dirname "$0")/golden.cpp" -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
    -L"$BUILD/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$BUILD/bin" -o "$OUT"
echo "built $OUT"
