#!/usr/bin/env python3
"""Compare teacher-forced next-token distributions between runs over the same token ids.

Inputs are logits dumps (QX_DS_DUMP_LOGITS: f32 LE rows of the vocabulary, one per position from --from) and
Strata score rows (--score-out TSV: pos, target, logp(target), argmax, logp(argmax), hit, logp(eos),
logp(target|no eos), then 64 `id:logp`). Metrics over the positions both sides cover, in [--pos-from, --pos-to):
top-1 agreement, mean and p99 KL(A || B), and the perplexity of the text under each side. KL against a Strata row
is over A's 64 most likely tokens, both renormalized to that set, a token B does not list taking B's 64th
log-probability (Strata's `quant_quality.py compare` rule); between two dumps it is the full-vocabulary KL.

  tools/logits_compare.py --tokens perf/data/route16k.tokens --pos-from 12288 --pos-to 16384 \
      --a ours.bin --a-from 12288 [--b other.bin --b-from 12288 | --b-strata run.tsv]
"""
import argparse
import math
import sys

import numpy as np


def read_tokens(path):
    return np.fromfile(path, dtype="<u4").astype(np.int64)


class Dump:
    def __init__(self, path, first, vocab):
        self.m = np.memmap(path, dtype="<f4", mode="r")
        assert self.m.size % vocab == 0, f"{path}: {self.m.size} floats is not a multiple of the vocabulary {vocab}"
        self.rows = self.m.size // vocab
        self.first = first
        self.vocab = vocab

    def has(self, pos):
        return self.first <= pos < self.first + self.rows

    def logp(self, pos):
        r = np.asarray(self.m[(pos - self.first) * self.vocab:(pos - self.first + 1) * self.vocab], dtype=np.float64)
        mx = r.max()
        z = np.log(np.exp(r - mx).sum()) + mx
        return r - z


class StrataRows:
    def __init__(self, path):
        self.rows = {}
        for line in open(path):
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            pos = int(f[0])
            top = {}
            for kv in f[8:]:
                i, lp = kv.split(":")
                top[int(i)] = float(lp)
            self.rows[pos] = (int(f[1]), float(f[2]), int(f[3]), float(f[4]), top)

    def has(self, pos):
        return pos in self.rows


def kl_full(la, lb):
    pa = np.exp(la)
    return float((pa * (la - lb)).sum())


def kl_top64(la, strow):
    """KL(A || B) over A's 64 most likely tokens, both renormalized; B's missing tokens take B's 64th log-prob."""
    top = np.argpartition(-la, 64)[:64]
    _, _, _, _, btop = strow
    bfloor = min(btop.values())
    a = la[top]
    b = np.array([btop.get(int(i), bfloor) for i in top])
    a = a - np.log(np.exp(a).sum())
    b = b - np.log(np.exp(b).sum())
    return float((np.exp(a) * (a - b)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--pos-from", type=int, required=True)
    ap.add_argument("--pos-to", type=int, required=True)
    ap.add_argument("--a", required=True, help="logits dump A (the reference side of the KL)")
    ap.add_argument("--a-from", type=int, required=True)
    ap.add_argument("--b")
    ap.add_argument("--b-from", type=int)
    ap.add_argument("--b-strata")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    toks = read_tokens(a.tokens)
    A = Dump(a.a, a.a_from, a.vocab)
    B = Dump(a.b, a.b_from, a.vocab) if a.b else StrataRows(a.b_strata)
    agree = 0
    n = 0
    kls = []
    nll_a = 0.0
    nll_b = 0.0
    for pos in range(a.pos_from, a.pos_to):
        if pos + 1 >= len(toks) or not A.has(pos) or not B.has(pos):
            continue
        tgt = int(toks[pos + 1])
        la = A.logp(pos)
        arg_a = int(la.argmax())
        if isinstance(B, Dump):
            lb = B.logp(pos)
            arg_b = int(lb.argmax())
            kls.append(kl_full(la, lb))
            nll_b -= float(lb[tgt])
        else:
            row = B.rows[pos]
            assert row[0] == tgt, f"position {pos}: Strata's target {row[0]} is not the text's {tgt}"
            arg_b = row[2]
            kls.append(kl_top64(la, row))
            nll_b -= row[1]
        nll_a -= float(la[tgt])
        agree += arg_a == arg_b
        n += 1
    if n == 0:
        sys.exit("no positions in common")
    kls = np.array(kls)
    print(f"{a.label} positions {n}: top-1 agreement {agree / n * 100:.1f}%, KL(A||B) mean {kls.mean():.4f} "
          f"p99 {np.percentile(kls, 99):.4f} max {kls.max():.4f}; perplexity A {math.exp(nll_a / n):.3f} "
          f"B {math.exp(nll_b / n):.3f}")


if __name__ == "__main__":
    main()
