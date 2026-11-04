#!/bin/bash
# Builds the offline routing-trace recorder against the DeepSeek-V4.1 llama.cpp build tree. Not part of cargo.
set -euo pipefail
LLAMA=${LLAMA:-$HOME/git/llama.cpp-dsv41}
BUILD=${BUILD:-$LLAMA/build}
OUT=${OUT:-$(dirname "$0")/route_trace}
g++ -O2 -std=c++17 "$(dirname "$0")/route_trace.cpp" -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
    -L"$BUILD/bin" -lllama -lggml -lggml-base -lggml-cpu -Wl,--disable-new-dtags -Wl,-rpath,"$BUILD/bin" -o "$OUT"
echo "built $OUT"
