"""Simulates a three-tier expert cache (VRAM, pinned RAM, NVMe) on a routing trace from tools/golden/route_trace.

    python3 -I tools/cache_sim3.py TRACE_PREFIX EXPERTS.json [--vram GIB] [--ram GIB] [--calib FRACTION]

TRACE_PREFIX.trace holds, token-major, one uint16 row [layer, id_0..id_{k-1}] per (token, layer).
EXPERTS.json comes from `tools/gguf_layout.py --experts` and gives each layer's bytes per expert.

Policy: one decayed routing count per (layer, expert); the VRAM set and the RAM set are refilled from the ranking
with hysteresis, a limited number of promotions per token. The first --calib fraction of the trace only
calibrates (static placement by frequency); the rest is measured. Costs: a RAM miss is computed by the CPU pool
at RAM bandwidth; an NVMe miss is read at drive bandwidth (then computed from staging, which costs RAM bandwidth
too); a promotion into RAM reads the drive, into VRAM copies over PCIe (counted as RAM bandwidth). Also reported:
the drive bytes a perfect one-layer lookahead would hide (NVMe misses whose read overlaps the previous layer).
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('prefix')
ap.add_argument('experts')
ap.add_argument('--vram', type=float, default=20.0, help='GiB of VRAM for experts')
ap.add_argument('--ram', type=float, default=88.0, help='GiB of pinned RAM for experts')
ap.add_argument('--calib', type=float, default=0.2)
ap.add_argument('--ram-gbs', type=float, default=55.0)
ap.add_argument('--nvme-gbs', type=float, default=12.0)
ap.add_argument('--half-life', type=float, default=256.0, help='decay half-life in tokens')
ap.add_argument('--swaps', type=int, default=4, help='max promotions per token into each tier')
ap.add_argument('--margin', type=float, default=0.2)
a = ap.parse_args()

layers = json.load(open(a.experts))['layers']
L = len(layers)
NE = layers[0]['gate']['n_expert']
rec = np.array([sum(p['bytes_per_expert'] for k, p in l.items() if k in ('gate', 'up', 'down')) for l in layers], np.float64)
raw = np.fromfile(a.prefix + '.trace', dtype=np.uint16)
K = None
for k in range(1, 17):
    if raw.size % (1 + k) == 0 and raw.size // (1 + k) % L == 0:
        rows = raw.reshape(-1, 1 + k)
        if (rows[:L, 0] == np.arange(L)).all():
            K = k
            break
assert K, 'cannot infer experts per token from the trace'
rows = raw.reshape(-1, 1 + K)
n_tok = rows.shape[0] // L
gid = rows[:, :1].astype(np.int64) * NE + rows[:, 1:].astype(np.int64)  # (rows, K) global expert ids
gid = gid.reshape(n_tok, L, K)
bytes_of = np.repeat(rec, NE)  # per global expert
print(f'trace: {n_tok} tokens, {L} layers, {NE} experts, {K} per token; record {rec.mean() / 2 ** 20:.2f} MiB mean, '
      f'{rec.sum() * K / 2 ** 30:.2f} GiB of expert weights per token')

n_cal = int(n_tok * a.calib)
calib = np.bincount(gid[:n_cal].reshape(-1), minlength=L * NE).astype(np.float64)
order = np.argsort(-calib, kind='stable')


def fill(order, budget_bytes, exclude):
    """Takes experts in ranking order until the byte budget is spent."""
    res = np.zeros(L * NE, bool)
    used = 0.0
    for g in order:
        if exclude[g]:
            continue
        if used + bytes_of[g] > budget_bytes:
            continue
        res[g] = True
        used += bytes_of[g]
    return res


vram = fill(order, a.vram * 2 ** 30, np.zeros(L * NE, bool))
ram = fill(order, a.ram * 2 ** 30, vram)
print(f'static from calibration: VRAM {vram.sum()} experts ({bytes_of[vram].sum() / 2 ** 30:.1f} GiB), '
      f'RAM {ram.sum()} ({bytes_of[ram].sum() / 2 ** 30:.1f} GiB), NVMe {L * NE - vram.sum() - ram.sum()}')

score = calib / max(1.0, calib.max()) * 10.0
decay = 0.5 ** (1.0 / a.half_life)
stats = dict(vram=0.0, ram=0.0, nvme=0.0, ram_bytes=0.0, nvme_bytes=0.0, hidden_bytes=0.0, promo_ram=0, promo_vram=0,
             promo_ram_bytes=0.0, promo_vram_bytes=0.0)
vram_bytes = bytes_of[vram].sum()
ram_bytes_used = bytes_of[ram].sum()
for t in range(n_cal, n_tok):
    score *= decay
    ids = gid[t]  # (L, K)
    flat = ids.reshape(-1)
    np.add.at(score, flat, 1.0)
    in_v = vram[flat]
    in_r = ram[flat] & ~in_v
    in_n = ~(in_v | in_r)
    stats['vram'] += in_v.sum()
    stats['ram'] += in_r.sum()
    stats['nvme'] += in_n.sum()
    stats['ram_bytes'] += bytes_of[flat[in_r]].sum() + bytes_of[flat[in_n]].sum()
    stats['nvme_bytes'] += bytes_of[flat[in_n]].sum()
    # a one-layer lookahead hides an NVMe read when the previous layer's work (its RAM misses) takes at least as long
    nv = in_n.reshape(L, K)
    for l in range(1, L):
        if nv[l].any():
            prev_ms = bytes_of[ids[l - 1][in_r.reshape(L, K)[l - 1]]].sum() / (a.ram_gbs * 1e6)
            need_ms = bytes_of[ids[l][nv[l]]].sum() / (a.nvme_gbs * 1e6)
            stats['hidden_bytes'] += bytes_of[ids[l][nv[l]]].sum() * min(1.0, prev_ms / need_ms if need_ms else 1.0)
    # promotions: the hottest non-resident experts over the coldest residents, with hysteresis
    for tier, res, budget, key in ((0, vram, a.vram * 2 ** 30, 'promo_vram'), (1, ram, a.ram * 2 ** 30, 'promo_ram')):
        cand_mask = ~res if tier == 1 else ~res
        if tier == 1:
            cand_mask &= ~vram
        cand = np.flatnonzero(cand_mask)
        if cand.size == 0:
            continue
        top = cand[np.argpartition(-score[cand], min(a.swaps, cand.size - 1))[: a.swaps]]
        resi = np.flatnonzero(res)
        cold = resi[np.argpartition(score[resi], min(a.swaps, resi.size - 1))[: a.swaps]]
        cold = cold[np.argsort(score[cold])]
        top = top[np.argsort(-score[top])]
        for h, c in zip(top, cold):
            if score[h] > score[c] * (1 + a.margin) + 1e-9:
                res[h], res[c] = True, False
                stats[key] += 1
                stats[key + '_bytes'] += bytes_of[h]
                if tier == 0 and ram[h]:
                    ram[h], ram[c] = False, True  # the evicted VRAM expert takes the promoted one's RAM place
                elif tier == 0:
                    ram[c] = True  # evicted from VRAM straight to RAM (budget drifts a little; reported below)

n = n_tok - n_cal
tot = n * L * K
ms_ram = stats['ram_bytes'] / n / (a.ram_gbs * 1e6) + stats['promo_vram_bytes'] / n / (a.ram_gbs * 1e6)
ms_nvme = (stats['nvme_bytes'] + stats['promo_ram_bytes']) / n / (a.nvme_gbs * 1e6)
ms_nvme_hidden = (stats['nvme_bytes'] - stats['hidden_bytes'] + stats['promo_ram_bytes']) / n / (a.nvme_gbs * 1e6)
print()
print(f'measured over {n} tokens, half-life {a.half_life:.0f} tokens, {a.swaps} promotions per token per tier')
print('| tier | share of routings | bytes per token |')
print('|---|---|---|')
for k, label in (('vram', 'VRAM'), ('ram', 'RAM (CPU pool)'), ('nvme', 'NVMe')):
    b = {'vram': 0.0, 'ram': stats['ram_bytes'] - stats['nvme_bytes'], 'nvme': stats['nvme_bytes']}[k]
    print(f'| {label} | {100 * stats[k] / tot:.1f}% | {b / n / 2 ** 20:.0f} MiB |')
print(f'promotions per token: {stats["promo_vram"] / n:.2f} into VRAM ({stats["promo_vram_bytes"] / n / 2 ** 20:.1f} MiB), '
      f'{stats["promo_ram"] / n:.2f} into RAM ({stats["promo_ram_bytes"] / n / 2 ** 20:.1f} MiB)')
print(f'RAM-bandwidth time per token: {ms_ram:.1f} ms; drive time: {ms_nvme:.1f} ms '
      f'(with a perfect one-layer lookahead: {ms_nvme_hidden:.1f} ms on the critical path)')
print(f'single-stream estimate: {ms_ram + ms_nvme + 6:.0f} ms/token -> {1000 / (ms_ram + ms_nvme + 6):.0f} tok/s; '
      f'with lookahead {ms_ram + ms_nvme_hidden + 6:.0f} ms -> {1000 / (ms_ram + ms_nvme_hidden + 6):.0f} tok/s '
      f'(6 ms of dense, attention and resident experts assumed)')
print(f'final residency: VRAM {bytes_of[vram].sum() / 2 ** 30:.1f} GiB, RAM {bytes_of[ram].sum() / 2 ** 30:.1f} GiB')
