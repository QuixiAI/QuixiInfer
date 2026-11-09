#!/bin/bash
# The DeepSeek engine's regression checks, in one command (each prints a line to compare with docs/progress.md):
#   1. exact f32 mode against the CPU reference (p0, 6 positions): expect 6/6, KLD 0.000000
#   2. production numerics against the reference (p1, 59 positions): expect about 57/59, KLD about 0.13
#   3. windows of 3, 8 and 64 against one-token steps (route1k, 128 positions): expect 128/128 bit-identical rows
#   3b. windows of 3 with the tiers adapting (promotions on) against the same steps: expect 128/128
#   4. tokenization against llama.cpp's llama-tokenize (when QX_LLAMA_TOKENIZE points at it): expect identical
#   5. rewind to 64 and park + unpark to 64 of route1k 128 against the straight pass: expect 64 / 64 twice
#   6. mixed sequences with cold widths 2,3,4: every slot's 3 rows match single steps (pending-pair regression)
# Heavy (the pack's RAM tier): run alone, under a memory cap: `tools/capped.sh 112 tools/ds-checks.sh`.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
D=${QX_DS_DIR:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS}
M=${QX_DS_MODEL:-$D/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf}
export QX_DS_PACK=${QX_DS_PACK:-$D/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp}
T=${TMPDIR:-/tmp}/ds-checks.$$; mkdir -p "$T"; trap 'rm -rf "$T"' EXIT
B=${QX_DS_CHECK_BIN:-target/release/qx-ds-check}
BM=$(dirname "$B")/qx-ds-multi
[ -x "$B" ] || cargo build --release -p qx-engine --bin qx-ds-check --bin qx-ds-multi --bin qx-tok >/dev/null
[ -x "$BM" ] || { echo "missing $BM; build qx-ds-multi beside qx-ds-check" >&2; exit 1; }
# a process that pinned the RAM tier takes 10-30 s to tear down after it exits and holds its VRAM meanwhile
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done; }
gpu_free
echo "1. exact f32 vs reference:      $(env -u QX_DS_PACK QX_MGEMV_INT8=0 QX_EXACT_EXPERTS=1 QX_REF_COMPAT=0 $B "$M" golden/out-dsv41/p0.tokens 6 2>&1 | grep '^positions')"
gpu_free
echo "2. production vs reference:     $(QX_DS_VRAM=20 QX_DS_RAM=60 QX_REF_COMPAT=engine $B "$M" golden/out-dsv41/p1.tokens 59 2>&1 | grep '^positions')"
B0="QX_DS_VRAM=20 QX_DS_RAM=60 QX_DS_BENCH=0 QX_DS_ADAPT=0"
gpu_free
env $B0 QX_DS_DUMP_LOGITS="$T/t1.bin" $B "$M" golden/data/route1k.tokens 128 >/dev/null 2>&1
for W in 3 8 64; do
  gpu_free
  env $B0 QX_DS_WINDOW=$W QX_DS_DUMP_LOGITS="$T/w$W.bin" $B "$M" golden/data/route1k.tokens 128 >/dev/null 2>&1
  echo "3. windows of $W vs single steps: $(python3 -I tools/cmp_logits.py "$T/t1.bin" "$T/w$W.bin" 129280 | grep identical)"
done
# 3b. the same width-3 windows with the tiers adapting (promotions, evictions, quarantine), which the lines above
# hold still (QX_DS_ADAPT=0): a timing-dependent tier race shows here as a non-identical row (added 2026-11-08 after a
# late-token flip under three concurrent streams in the server)
gpu_free
env ${B0/QX_DS_ADAPT=0/QX_DS_ADAPT=1} QX_DS_WINDOW=3 QX_DS_PROMOS=32 QX_DS_DUMP_LOGITS="$T/w3a.bin" $B "$M" golden/data/route1k.tokens 128 >/dev/null 2>&1
echo "3b. windows of 3, tiers adapting, vs single steps: $(python3 -I tools/cmp_logits.py "$T/t1.bin" "$T/w3a.bin" 129280 | grep identical)"
if [ -n "${QX_LLAMA_TOKENIZE:-}" ]; then
  printf 'The quick brown fox jumps over the lazy dog. 1234567 ok!\n<｜User｜>Hi<｜Assistant｜></think>' > "$T/tok.txt"
  "$QX_LLAMA_TOKENIZE" -m "$M" -f "$T/tok.txt" --ids --no-bos 2>/dev/null | tail -1 > "$T/tok.ids"
  python3 -I -c "import sys,struct,ast; ids=ast.literal_eval(open(sys.argv[1]).read().strip()); open(sys.argv[2],'wb').write(struct.pack('<%dI'%len(ids),*ids))" "$T/tok.ids" "$T/tok.bin"
  echo "4. tokenizer vs llama.cpp:      $(QX_MODEL="$M" target/release/qx-tok check "$T/tok.txt" "$T/tok.bin" 2>/dev/null | tr '\n' ' ')"
fi
# 5. a rewind (the ring still holds the window) and a park + unpark (the window rebuilt by replay) against the
# straight pass: every later row bit-identical (the replay-based rewind, 2026-11-06)
gpu_free
echo "5. rewind and park vs straight: $(env $B0 QX_DS_REWIND=64 QX_DS_PARK=64 $B "$M" golden/data/route1k.tokens 128 2>/dev/null | grep -E '^(rewind|park)' | sed -E 's/ \(restore.*//; s/ then positions 64..128//' | tr '\n' ' ')"
gpu_free
if ! env QX_DS_VRAM=20 QX_DS_RAM=60 QX_DS_MULTI_WIDTHS=2,3,4 QX_DS_MULTI_ROTATE=1 QX_DS_MULTI_CHECK_ALL=1 QX_DS_MULTI_WARM=0 \
    "$BM" "$QX_DS_PACK" golden/data/route1k.tokens 3 > "$T/multi.log" 2>&1; then
  cat "$T/multi.log"
  exit 1
fi
echo "6. mixed sequences, cold widths: $(grep '^multi-sequence' "$T/multi.log" | tr '\n' ' ')"
