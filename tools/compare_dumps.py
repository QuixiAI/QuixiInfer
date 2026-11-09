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
    for number, line in enumerate(open(path), 1):
        r = json.loads(line)
        if (not isinstance(r, dict) or type(r.get("c")) is not int
                or not isinstance(r.get("prompt"), str) or not isinstance(r.get("text"), str)
                or not isinstance(r.get("reasoning", ""), str)):
            raise ValueError(f"{path}:{number}: invalid generation record")
        d.setdefault((r["c"], r["prompt"]), []).append((r.get("reasoning", ""), r["text"]))
    return d


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    bad = 0
    if not a or not b:
        print("empty generation dump")
        bad += 1
    for key in sorted(a.keys() | b.keys()):
        ta, tb = a.get(key, []), b.get(key, [])
        for i in range(max(len(ta), len(tb))):
            if i >= len(ta):
                print(f"c{key[0]} {key[1][:40]!r}: extra response in the second file")
                bad += 1
                continue
            if i >= len(tb):
                print(f"c{key[0]} {key[1][:40]!r}: missing in the second file")
                bad += 1
                continue
            t = ta[i]
            u = tb[i]
            if t == u:
                print(f"c{key[0]} {key[1][:40]!r}: same ({sum(map(len, t))} chars)")
                continue
            field = 0 if t[0] != u[0] else 1
            t, u = t[field], u[field]
            j = 0
            while j < min(len(t), len(u)) and t[j] == u[j]:
                j += 1
            print(f"c{key[0]} {key[1][:40]!r}: DIFFER in {('reasoning', 'text')[field]} at {j}/{len(t)}: {t[max(0, j - 15):j + 25]!r} vs {u[max(0, j - 15):j + 25]!r}")
            bad += 1
    print("identical" if not bad else f"{bad} difference(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
