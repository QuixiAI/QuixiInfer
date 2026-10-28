"""Simulates VRAM expert-cache policies on a routing trace written by qx-profile.

    python3 -I tools/cache_sim.py golden/profile-calib.txt   (reads .txt ranking and .txt.trace)

Every miss is computed on the CPU from RAM and every swap-in copies an expert from RAM, so both cost about
one expert of DRAM traffic; the score is (misses + swaps) per token, and the hit rate.
"""
import sys
from collections import OrderedDict

import numpy as np

rank_path = sys.argv[1]
ranking = [tuple(map(int, l.split()[:2])) for l in open(rank_path)]
tr = np.fromfile(rank_path + ".trace", dtype=np.uint16).reshape(-1, 11)
L, NE, K = 48, 512, 10
n_tok = len(tr) // L
print(f"trace: {n_tok} tokens")
keys = tr[:, 0].astype(np.int64) * NE  # layer * NE
ids = keys[:, None] + tr[:, 1:].astype(np.int64)  # (n_rows, 10) global expert ids


def static(n):
    res = np.zeros(L * NE, bool)
    for l, e in ranking[:n]:
        res[l * NE + e] = True
    miss = (~res[ids]).sum()
    return miss / n_tok, 0.0


def lru(n):
    cache = OrderedDict((l * NE + e, None) for l, e in ranking[:n])
    miss = swaps = 0
    for row in ids:
        for g in row:
            if g in cache:
                cache.move_to_end(g)
            else:
                miss += 1
                swaps += 1
                cache[g] = None
                cache.popitem(last=False)
    return miss / n_tok, swaps / n_tok


def decayed(n, half_life_tokens, max_swaps_per_token, margin):
    """Decayed-frequency admission: a missed expert replaces the coldest resident one when its score is higher
    by `margin`, at most `max_swaps_per_token` per token. Scores start from the calibration ranking."""
    decay = 0.5 ** (1.0 / (half_life_tokens * L))  # per routing row
    score = np.zeros(L * NE)
    for i, (l, e) in enumerate(ranking):
        score[l * NE + e] = 1.0 / (1 + i / 2000)
    res = np.zeros(L * NE, bool)
    for l, e in ranking[:n]:
        res[l * NE + e] = True
    import heapq
    miss = swaps = 0
    budget = 0.0
    scale = 1.0  # lazy decay: stored scores are divided by `scale`
    for r, row in enumerate(ids):
        if r % L == 0:
            budget = min(budget + max_swaps_per_token, 4 * max_swaps_per_token)
        scale /= decay
        for g in row:
            score[g] += scale
            if not res[g]:
                miss += 1
        if budget >= 1 and r % L == L - 1:
            # once per token: swap in the hottest non-residents over the coldest residents
            cand = np.flatnonzero(~res)
            hot = cand[np.argsort(-score[cand])[: int(budget)]]
            resi = np.flatnonzero(res)
            cold = resi[np.argsort(score[resi])[: len(hot)]]
            for h, c in zip(hot, cold):
                if score[h] > score[c] * (1 + margin):
                    res[h], res[c] = True, False
                    swaps += 1
                    budget -= 1
        if scale > 1e100:
            score /= scale
            scale = 1.0
    return miss / n_tok, swaps / n_tok


total = K * L
print("| policy | slots | misses/token | swaps/token | hit rate | RAM MB/token |")
print("|---|---|---|---|---|---|")
for n in [11000, 12000, 13500]:
    for name, fn in [("static", lambda: static(n)), ("lru", lambda: lru(n)),
                     ("decayed hl=64 s=4", lambda: decayed(n, 64, 4, 0.2)),
                     ("decayed hl=256 s=8", lambda: decayed(n, 256, 8, 0.2)),
                     ("decayed hl=32 s=16", lambda: decayed(n, 32, 16, 0.1))]:
        m, s = fn()
        print(f"| {name} | {n} | {m:.1f} | {s:.1f} | {100 * (1 - m / total):.1f}% | {(m + s) * 1.88:.0f} |")
