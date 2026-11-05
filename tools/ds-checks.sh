#!/bin/bash
# The DeepSeek engine's regression checks, in one command (each prints a line to compare with docs/progress.md):
#   1. exact f32 mode against the CPU reference (p0, 6 positions): expect 6/6, KLD 0.000000
#   2. production numerics against the reference (p1, 59 positions): expect about 57/59, KLD about 0.13
#   3. windows of 8 and 64 against one-token steps (route1k, 128 positions): expect 128/128 bit-identical rows
#   4. tokenization against llama.cpp's llama-tokenize (when QX_LLAMA_TOKENIZE points at it): expect identical
# Heavy (the pack's RAM tier): run alone, under a memory cap: `tools/capped.sh 112 tools/ds-checks.sh`.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
D=${QX_DS_DIR:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS}
M=${QX_DS_MODEL:-$D/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf}
export QX_DS_PACK=${QX_DS_PACK:-$D/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp}
T=${TMPDIR:-/tmp}/ds-checks.$$; mkdir -p "$T"; trap 'rm -rf "$T"' EXIT
B=target/release/qx-ds-check
[ -x $B ] || cargo build --release -p qx-engine --bin qx-ds-check --bin qx-tok >/dev/null
# a process that pinned the RAM tier takes 10-30 s to tear down after it exits and holds its VRAM meanwhile
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done; }
gpu_free
echo "1. exact f32 vs reference:      $(env -u QX_DS_PACK QX_MGEMV_INT8=0 QX_EXACT_EXPERTS=1 QX_REF_COMPAT=0 $B "$M" golden/out-dsv41/p0.tokens 6 2>&1 | grep '^positions')"
gpu_free
echo "2. production vs reference:     $(QX_DS_VRAM=20 QX_DS_RAM=60 QX_REF_COMPAT=engine $B "$M" golden/out-dsv41/p1.tokens 59 2>&1 | grep '^positions')"
B0="QX_DS_VRAM=20 QX_DS_RAM=60 QX_DS_BENCH=0 QX_DS_ADAPT=0"
gpu_free
env $B0 QX_DS_DUMP_LOGITS="$T/t1.bin" $B "$M" golden/data/route1k.tokens 128 >/dev/null 2>&1
for W in 8 64; do
  gpu_free
  env $B0 QX_DS_WINDOW=$W QX_DS_DUMP_LOGITS="$T/w$W.bin" $B "$M" golden/data/route1k.tokens 128 >/dev/null 2>&1
  echo "3. windows of $W vs single steps: $(python3 -I tools/cmp_logits.py "$T/t1.bin" "$T/w$W.bin" 129280 | grep identical)"
done
if [ -n "${QX_LLAMA_TOKENIZE:-}" ]; then
  printf 'The quick brown fox jumps over the lazy dog. 1234567 ok!\n<｜User｜>Hi<｜Assistant｜></think>' > "$T/tok.txt"
  "$QX_LLAMA_TOKENIZE" -m "$M" -f "$T/tok.txt" --ids --no-bos 2>/dev/null | tail -1 > "$T/tok.ids"
  python3 -I -c "import sys,struct,ast; ids=ast.literal_eval(open(sys.argv[1]).read().strip()); open(sys.argv[2],'wb').write(struct.pack('<%dI'%len(ids),*ids))" "$T/tok.ids" "$T/tok.bin"
  echo "4. tokenizer vs llama.cpp:      $(QX_MODEL="$M" target/release/qx-tok check "$T/tok.txt" "$T/tok.bin" 2>/dev/null | tr '\n' ' ')"
fi
