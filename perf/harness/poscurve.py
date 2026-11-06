#!/usr/bin/env python3
"""Cost by position from a forced run's `pos N:` lines: per bucket of windows, ms per token, the tier shares of the
selections made in the bucket, the pool's ms per token and the staging wait. Usage: poscurve.py LOG [bucket_windows]"""
import re, sys
rx = re.compile(r"^pos (\d+): (\d+) ms per window of (\d+) \(([\d.]+) ms per token\)\s+vram (\d+) staged (\d+) pcie (\d+) ram (\d+) nvme (\d+) promos vram (\d+) ram (\d+) \(([\d.]+) ms, pool ([\d.]+) ms of which fetch ([\d.]+), stage wait ([\d.]+)\)", re.M)
rows = [tuple(float(x) for x in m.groups()) for m in rx.finditer(open(sys.argv[1]).read())]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
print(f"{len(rows)} pos lines; buckets of {B} printed windows (each line is one window every 4; a bucket spans {B*4} windows)")
print(f"{'pos':>7} {'ms/tok':>7} {'vram%':>6} {'stg%':>5} {'ram%':>5} {'nvme%':>6} {'pool ms/tok':>11} {'fetch':>6} {'stgwait':>7}")
prev = None
for i in range(0, len(rows), B):
    r = rows[min(i + B - 1, len(rows) - 1)]
    if prev is None:
        prev = r; continue
    pos, w = int(r[0]), r[2]
    toks = (r[0] - prev[0])
    d = [r[k] - prev[k] for k in range(len(r))]
    sel = max(d[4] + d[5] + d[7] + d[8], 1)
    mt = sum(x[3] for x in rows[i:i + B]) / len(rows[i:i + B])
    print(f"{pos:7d} {mt:7.1f} {100*d[4]/sel:6.1f} {100*d[5]/sel:5.1f} {100*d[7]/sel:5.1f} {100*d[8]/sel:6.1f} {d[12]/toks:11.1f} {d[13]/toks:6.1f} {d[14]/toks:7.1f}")
    prev = r
