import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
n = int(sys.argv[3])
a = a.reshape(-1, n); b = b.reshape(-1, n)
print(f"rows {a.shape[0]} vs {b.shape[0]}")
m = min(a.shape[0], b.shape[0]); a = a[:m]; b = b[:m]
eq = (a == b).all(axis=1)
print(f"bit-identical rows: {eq.sum()} / {m}")
if not eq.all():
    bad = np.where(~eq)[0]
    print("first differing rows:", bad[:10].tolist())
    for r in bad[:3]:
        d = np.abs(a[r] - b[r]); print(f"  row {r}: max abs diff {d.max():.3e} at {d.argmax()}, argmax {a[r].argmax()} vs {b[r].argmax()}")
