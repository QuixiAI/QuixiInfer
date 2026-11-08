#!/usr/bin/env python3
"""Compare two `tools/loadgen.py --dump` files generation by generation, matched by (concurrency, prompt, occurrence):
the engine's results do not depend on scheduling, so the same prompt at temperature 0 must produce the same text in
both runs. Completion order differs between runs, which is why a hash of the whole file is not the test.

    python3 tools/compare_dumps.py ref.dump.jsonl other.dump.jsonl
Exit 0 when every matched generation is identical, 1 otherwise (each difference printed with its first divergence)."""
import json
import sys


def load(path):
    d = {}
    for line in open(path):
        r = json.loads(line)
        d.setdefault((r["c"], r["prompt"]), []).append(r.get("reasoning", "") + r.get("text", ""))
    return d


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    bad = 0
    for key in sorted(a):
        ta, tb = a[key], b.get(key, [])
        for i, t in enumerate(ta):
            if i >= len(tb):
                print(f"c{key[0]} {key[1][:40]!r}: missing in the second file")
                bad += 1
                continue
            u = tb[i]
            if t == u:
                print(f"c{key[0]} {key[1][:40]!r}: same ({len(t)} chars)")
                continue
            j = 0
            while j < min(len(t), len(u)) and t[j] == u[j]:
                j += 1
            print(f"c{key[0]} {key[1][:40]!r}: DIFFER at {j}/{len(t)}: {t[j - 15:j + 25]!r} vs {u[j - 15:j + 25]!r}")
            bad += 1
    for key in b:
        if key not in a:
            print(f"c{key[0]} {key[1][:40]!r}: only in the second file")
    print("identical" if not bad else f"{bad} difference(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
