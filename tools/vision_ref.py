#!/usr/bin/env python3
"""A NumPy + Pillow reference of DeepSeek-V4.1-Flash's vision tower, to check `qx-vision` against.

`vision_ref.py MMPROJ.gguf IMAGE PREFIX [--pixels-only]`: fits IMAGE as the model's `image_processor.py` does (real
Pillow: convert("RGB"), ImageOps.pad on grey 127), runs the ViT and the aligner as `vision.py` does, and compares
with `PREFIX.rgb` / `PREFIX.f32` written by `qx-vision-check MMPROJ.gguf IMAGE PREFIX`. Needs gguf-py on the path
(GGUF_PY, default ~/git/llama.cpp-dsv41/gguf-py)."""
import math, os, sys
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/git/llama.cpp-dsv41/gguf-py")))
from gguf import GGUFReader

P, R, MAXTOK, MINPIX, THETA, EPS = 14, 3, 1024, 544 * 544, 10000.0, 1e-6

def plan(width, height):
    if 0 < width * height < MINPIX:
        ratio = (MINPIX / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    bw, bh = math.ceil(width / P) * P, math.ceil(height / P) * P
    grid = lambda bh, bw: (math.ceil((bh // P) / R), math.ceil((bw // P) / R))
    ntok = lambda g: g[0] * (g[1] + 1) + 2
    if ntok(grid(bh, bw)) > MAXTOK:
        r = height / width
        mw = math.sqrt((MAXTOK - 2) / r + 0.25) - 0.5
        mh = mw * r
        cell = P * R
        if mw < 1.0:
            bh, bw = (MAXTOK - 2) // 2 * cell, cell
        elif mh < 1.0:
            bh, bw = cell, (MAXTOK - 3) * cell
        else:
            beta = min(math.floor(mw) * cell / width, math.floor(mh) * cell / height)
            bh, bw = math.floor(height * beta / P) * P, math.floor(width * beta / P) * P
        assert ntok(grid(bh, bw)) <= MAXTOK
    return bh, bw

def main():
    mm, path, prefix = sys.argv[1:4]
    with Image.open(path) as src:
        image = src.convert("RGB")
    bh, bw = plan(image.width, image.height)
    image = ImageOps.pad(image, (bw, bh), color=(127, 127, 127))
    rgb = np.asarray(image, dtype=np.uint8)
    ours = np.fromfile(prefix + ".rgb", dtype=np.uint8)
    same = ours.size == rgb.size and np.array_equal(ours, rgb.reshape(-1))
    print(f"fitted {image.width}x{image.height}: pixels {'identical to Pillow' if same else 'DIFFERENT from Pillow'}" + ("" if same or ours.size != rgb.size else f" ({int((ours != rgb.reshape(-1)).sum())} bytes, max |d| {int(np.abs(ours.astype(int) - rgb.reshape(-1).astype(int)).max())})"))
    if "--pixels-only" in sys.argv:
        return 0 if same else 1
    r = GGUFReader(mm)
    T = {t.name: t for t in r.tensors}
    # a GGUF tensor's dims are reversed: its data is the torch tensor in C order
    def w(name):
        t = T[name]
        return np.asarray(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
    nh, nw = bh // P, bw // P
    n = nh * nw
    x = rgb.astype(np.float32).transpose(2, 0, 1) / np.float32(255)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    # to bfloat16 (round to nearest even) and back
    b = x.view(np.uint32)
    b = (b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    x = b.view(np.float32)
    patches = x.reshape(3, nh, P, nw, P).transpose(1, 3, 0, 2, 4).reshape(n, 3 * P * P)
    h = patches @ w("v.patch_embd.weight").reshape(1024, -1).T + w("v.patch_embd.bias")
    dim, heads = 1024, 16
    hd = dim // heads
    rd = hd // 2
    inv = (1.0 / (THETA ** (np.arange(0, rd, 2, dtype=np.float32) / rd))).astype(np.float32)
    hpos = np.repeat(np.arange(nh), nw).astype(np.float32)
    wpos = np.tile(np.arange(nw), nh).astype(np.float32)
    fr = np.concatenate([hpos[:, None] * inv, wpos[:, None] * inv], axis=1)
    cos, sin = np.cos(fr)[:, None, :], np.sin(fr)[:, None, :]
    def rot(t):
        a, c = t[..., :rd], t[..., rd:]
        return np.concatenate([a * cos - c * sin, c * cos + a * sin], axis=-1)
    def rms(t, g):
        return g * (t / np.sqrt((t * t).mean(-1, keepdims=True) + np.float32(EPS)))
    for l in range(32):
        p = f"v.blk.{l}."
        y = rms(h, w(p + "ln1.weight"))
        qkv = y @ w(p + "attn_qkv.weight").T + w(p + "attn_qkv.bias")
        q, k, v = (t.reshape(n, heads, hd) for t in np.split(qkv, 3, axis=-1))
        q, k = rot(q), rot(k)
        o = np.empty((n, heads, hd), dtype=np.float32)
        for hh in range(heads):
            s = (q[:, hh] @ k[:, hh].T) / np.float32(math.sqrt(hd))
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o[:, hh] = s @ v[:, hh]
        h = h + (o.reshape(n, dim) @ w(p + "attn_out.weight").T + w(p + "attn_out.bias"))
        y = rms(h, w(p + "ln2.weight"))
        g, u = y @ w(p + "ffn_gate.weight").T, y @ w(p + "ffn_up.weight").T
        h = h + ((g / (1 + np.exp(-g))) * u) @ w(p + "ffn_down.weight").T
    h = rms(h, w("v.post_ln.weight"))
    # the aligner: pad, unfold 3x3 (feature-major), two linears with an erf GELU
    grid = h.reshape(nh, nw, dim)
    gh, gw = math.ceil(nh / R), math.ceil(nw / R)
    pad = np.zeros((gh * R, gw * R, dim), dtype=np.float32)
    pad[:nh, :nw] = grid
    cells = pad.reshape(gh, R, gw, R, dim).transpose(0, 2, 4, 1, 3).reshape(gh * gw, dim * R * R)
    a = cells @ w("mm.1.weight").T + w("mm.1.bias")
    erf = np.vectorize(math.erf)
    a = (0.5 * a.astype(np.float64) * (1 + erf(a.astype(np.float64) / math.sqrt(2)))).astype(np.float32)
    feat = (a @ w("mm.2.weight").T + w("mm.2.bias")).reshape(gh, gw, -1)
    rows = [w("v.token_embd.img_start")]
    for yy in range(gh):
        rows += list(feat[yy]) + [w("v.image_newline")]
    rows.append(w("v.token_embd.img_end"))
    ref = np.stack(rows).astype(np.float32)
    got = np.fromfile(prefix + ".f32", dtype=np.float32)
    if got.size != ref.size:
        print(f"block: {got.size // 5120} rows against the reference's {ref.shape[0]}: DIFFERENT")
        return 1
    got = got.reshape(ref.shape)
    d = np.abs(got - ref)
    cosr = (got * ref).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-30)
    print(f"block {ref.shape[0]} rows (grid {gh}x{gw}): max |d| {d.max():.3e}, mean |d| {d.mean():.3e}, reference mean |x| {np.abs(ref).mean():.3e}, row cosine min {cosr.min():.7f}")
    ok = same and cosr.min() > 0.9999 and d.max() < 0.05 * np.abs(ref).max()
    print("ok" if ok else "MISMATCH")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
