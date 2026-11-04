"""Checks every qx-quant block decoder against gguf-py on real tensors of a GGUF file (shards welcome).

    python3 -I tools/dequant_check.py ~/git/llama.cpp/gguf-py MODEL-0000N-of-0000M.gguf [BLOCKS_PER_TYPE]

For each tensor type present it takes BLOCKS_PER_TYPE blocks (default 4096) from the first tensor of that type
(and the same count from the last one), dequantizes them with gguf-py and with `target/release/examples/dequant`
(build: `cargo build --release -p qx-quant --example dequant`), and reports the largest difference. gguf-py is
the reference llama.cpp's own tests check against, so agreement to float rounding means the decoder is right.
"""
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from gguf import GGUFReader, GGMLQuantizationType, quants  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

root = pathlib.Path(__file__).resolve().parent.parent
exe = root / "target/release/examples/dequant"
path = sys.argv[2]
nblk = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
reader = GGUFReader(path)
by_type = {}
for t in reader.tensors:
    by_type.setdefault(t.tensor_type, []).append(t)
skip = {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}
tmp = tempfile.mkdtemp()
ok = True
print("| type | tensor | blocks | max abs diff | max ref |")
print("|---|---|---|---|---|")
for ty, ts in sorted(by_type.items(), key=lambda kv: kv[0].name):
    if ty in skip:
        continue
    bs, bbytes = GGML_QUANT_SIZES[ty]
    for t in (ts[0], ts[-1]) if len(ts) > 1 else (ts[0],):
        raw = np.asarray(t.data).reshape(-1).view(np.uint8)
        n = min(nblk, raw.size // bbytes)
        blocks = raw[: n * bbytes]
        ref = quants.dequantize(blocks.reshape(n, bbytes), ty).reshape(-1).astype(np.float32)
        fin = os.path.join(tmp, "in.bin")
        fout = os.path.join(tmp, "out.f32")
        blocks.tofile(fin)
        r = subprocess.run([str(exe), ty.name, fin, fout], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"| {ty.name} | {t.name} | {n} | FAILED: {r.stderr.strip()[:80]} | |")
            ok = False
            continue
        got = np.fromfile(fout, dtype=np.float32)
        if got.size != ref.size:
            print(f"| {ty.name} | {t.name} | {n} | size {got.size} != {ref.size} | |")
            ok = False
            continue
        diff = np.abs(got - ref).max()
        mref = np.abs(ref).max()
        good = diff <= 1e-6 * max(1.0, mref) + 1e-7
        ok &= bool(good)
        print(f"| {ty.name} | {t.name} | {n} | {diff:.3g}{'' if good else ' MISMATCH'} | {mref:.3g} |")
print("all decoders agree" if ok else "MISMATCHES")
sys.exit(0 if ok else 1)
