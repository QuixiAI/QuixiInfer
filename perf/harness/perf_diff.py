#!/usr/bin/env python3
"""Parse, summarize, compare and gate QuixiInfer benchmark runs (schema 1). See perf/perf.md.

A run directory (perf/results/<date>/<label>-<time>/, written by perf/harness/bench.sh) holds run.json
(provenance) and results.jsonl (one row per case and repeat). This tool is the mechanical half of the handbook's
decision rules: the 3% / 8% keep bars, the noise guard (a change inside the repeats' spread is not a result), and
the rule that runs from different hosts or settings are not comparable.

Subcommands:
  parse LOG --case C --repeat R [--exit N]   one results.jsonl row from a qx-ds-check log (used by bench.sh)
  summarize RUN                  per-case median / min / max / spread and the score, as markdown
  validate RUN                   failed repeats, NaN, noisy cases, correctness status; exit 1 when unusable
  diff BASE CAND                 case-by-case comparison with verdicts; exit 1 on a regression, 4 on a host or
                                 settings mismatch
  gate RUN                       diff RUN against perf/baselines/<fingerprint>/ (exit 3 when there is none)
  promote RUN                    copy RUN into perf/baselines/<fingerprint>/ (the only subcommand that writes
                                 outside the run directory, besides score --record and index)
  score RUN [--record FILE --note TEXT]   the geometric mean of the cases' median ms per token; --record appends
                                 it to the scoreboard
  entry RUN                      a pre-filled optimization_status.md entry
  fingerprint RUN                the host fingerprint
  index NOTEBOOK                 regenerate the notebook's entry index (between the qx:index markers)
"""

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

SCHEMA = 1
LOW_RISK_BAR = 0.03      # a low-risk change must beat the baseline median by this much
COMPLEX_BAR = 0.08       # a complexity-adding change by this much
SPREAD_LIMIT = 0.10      # (max - min) / median above this: the case is too noisy to decide anything
ROOT = Path(__file__).resolve().parents[2]


def die(msg, code=2):
    print(msg, file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------------------------------------------
# parse

NUM = r"([0-9]+(?:\.[0-9]+)?)"
RE_DECODE = re.compile(r"^generated (\d+) tokens: " + NUM + r" ms per token.*?last (\d+): " + NUM + r" ms per token", re.M)
RE_WINDOWS = re.compile(r"^windows of (\d+): (\d+) tokens, " + NUM + r" ms per token.*?last (\d+): " + NUM + r" ms per token", re.M)
RE_POS = re.compile(r"^pos (\d+):.*?vram (\d+) staged (\d+) pcie (\d+) ram (\d+) nvme (\d+) promos vram (\d+) ram (\d+) \("
                    + NUM + r" ms, pool " + NUM + r" ms of which fetch " + NUM + r", stage wait " + NUM + r"\)", re.M)
RE_PHASES = re.compile(r"phases per step \(last \d+\): uploads " + NUM + r" ms, launch/replay " + NUM + r", device wait "
                       + NUM + r", history flush " + NUM + r", logits download " + NUM)


def parse_log(text):
    row = {"metric": "ms_per_token_last32", "unit": "ms/token", "value": None}
    m = RE_DECODE.search(text)
    if m:
        row.update(kind="decode", tokens=int(m.group(1)), mean_ms_per_token=float(m.group(2)), value=float(m.group(4)))
    m = RE_WINDOWS.search(text)
    if m:
        row.update(kind="windows", window=int(m.group(1)), tokens=int(m.group(2)), mean_ms_per_token=float(m.group(3)),
                   value=float(m.group(5)))
    last = None
    for last in RE_POS.finditer(text):
        pass
    if last:
        g = last.groups()
        row["tiers_at_end"] = {"pos": int(g[0]), "vram": int(g[1]), "staged": int(g[2]), "pcie": int(g[3]), "ram": int(g[4]),
                               "nvme": int(g[5]), "promos_vram": int(g[6]), "promos_ram": int(g[7]), "promo_ms": float(g[8]),
                               "pool_ms": float(g[9]), "fetch_ms": float(g[10]), "stage_wait_ms": float(g[11])}
    m = RE_PHASES.search(text)
    if m:
        row["phases_ms"] = dict(zip(["uploads", "launch_replay", "device_wait", "history_flush", "logits_download"],
                                    map(float, m.groups())))
    row["nan"] = "NaN" in text
    row["panicked"] = "panicked" in text
    if row["value"]:
        row["tok_s"] = round(1000.0 / row["value"], 2)
    return row


def cmd_parse(a):
    text = Path(a.log).read_text(errors="replace")
    row = {"schema": SCHEMA, "case": a.case, "repeat": a.repeat, "exit": a.exit, "log": str(a.log)}
    row.update(parse_log(text))
    print(json.dumps(row))


# ---------------------------------------------------------------------------------------------------------------
# loading runs

def load(run):
    p = Path(run)
    if p.is_file():
        rows_path, info_path = p, p.with_name("run.json")
    else:
        rows_path, info_path = p / "results.jsonl", p / "run.json"
    if not rows_path.exists():
        die(f"no results.jsonl in {run}")
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    for r in rows:
        if r.get("schema") != SCHEMA:
            die(f"{rows_path}: schema {r.get('schema')} (want {SCHEMA})")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return rows, info


def fingerprint(info):
    h = info.get("host", {})
    def slug(s):
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    gpu = slug(h.get("gpu", "unknown-gpu").replace("NVIDIA GeForce ", "").replace("NVIDIA ", ""))
    cpu = slug(re.sub(r"\d+-Core Processor", "", h.get("cpu", "unknown-cpu")).replace("AMD Ryzen ", "ryzen-"))
    return f"{gpu}_{cpu}_{h.get('ram_gib', 0)}g"


def per_case(rows):
    cases = {}
    for r in rows:
        cases.setdefault(r["case"], []).append(r)
    out = {}
    for c, rs in cases.items():
        ok = [r["value"] for r in rs if r.get("exit") == 0 and r.get("value") and not r.get("nan")]
        s = {"n": len(rs), "ok": len(ok)}
        if ok:
            med = statistics.median(ok)
            s.update(median=med, min=min(ok), max=max(ok), spread=(max(ok) - min(ok)) / med,
                     mean_ms=statistics.median([r["mean_ms_per_token"] for r in rs if r.get("mean_ms_per_token")] or [0]))
            pools = [r["tiers_at_end"]["pool_ms"] / r["tiers_at_end"]["pos"] for r in rs if r.get("tiers_at_end") and r["tiers_at_end"]["pos"]]
            if pools:
                s["pool_ms_per_token"] = statistics.median(pools)
        out[c] = s
    return out


def score_of(stats):
    meds = [s["median"] for s in stats.values() if "median" in s]
    if not meds or len(meds) != len(stats):
        return None
    return math.exp(sum(math.log(m) for m in meds) / len(meds))


# ---------------------------------------------------------------------------------------------------------------
# summarize / validate

def cmd_summarize(a):
    rows, info = load(a.run)
    st = per_case(rows)
    print(f"# {info.get('label', '?')} ({info.get('date', '?')})\n")
    g = info.get("git", {})
    print(f"git {g.get('commit', '?')}{' + working tree ' + str(g.get('diff_sha')) if g.get('dirty') else ''}, host `{fingerprint(info)}`, "
          f"driver {info.get('host', {}).get('driver', '?')}, correctness {info.get('correctness', '?')}")
    if info.get("experiment_env"):
        print(f"experiment env: `{info['experiment_env']}`")
    print("\n| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for c, s in st.items():
        if "median" in s:
            print(f"| {c} | {s['ok']}/{s['n']} | **{s['median']:.2f}** | {1000 / s['median']:.1f} | {s['min']:.2f} | {s['max']:.2f} | "
                  f"{100 * s['spread']:.1f}% | {s['mean_ms']:.2f} | {s.get('pool_ms_per_token', float('nan')):.2f} |")
        else:
            print(f"| {c} | 0/{s['n']} | failed | | | | | | |")
    sc = score_of(st)
    print(f"\nscore (geomean of the case medians, ms/token): **{sc:.3f}**" if sc else "\nscore: n/a (a case failed)")


def cmd_validate(a):
    rows, info = load(a.run)
    bad = 0
    for r in rows:
        if r.get("exit") != 0 or not r.get("value") or r.get("nan") or r.get("panicked"):
            print(f"FAILED: {r['case']} repeat {r['repeat']}: exit {r.get('exit')}, nan {r.get('nan')}, value {r.get('value')} ({r.get('log')})")
            bad += 1
    for c, s in per_case(rows).items():
        if s.get("ok", 0) < 2:
            print(f"WARN: {c}: {s.get('ok', 0)} usable repeat(s); decisions need at least 2 (3 by default)")
        if s.get("spread", 0) > SPREAD_LIMIT:
            print(f"NOISY: {c}: spread {100 * s['spread']:.1f}% > {100 * SPREAD_LIMIT:.0f}%: re-run on an idle box before using it")
            bad += 1
    if info.get("correctness") != "pass":
        print(f"NOTE: correctness {info.get('correctness', 'unknown')}: measurable, not landable (run with --check)")
    if info.get("git", {}).get("dirty"):
        print("NOTE: working tree was dirty; the diff is saved as working-tree.diff in the run directory")
    if bad:
        sys.exit(1)
    print("valid")


# ---------------------------------------------------------------------------------------------------------------
# diff / gate / promote

def diff(base_dir, cand_dir):
    brows, binfo = load(base_dir)
    crows, cinfo = load(cand_dir)
    if fingerprint(binfo) != fingerprint(cinfo):
        die(f"host fingerprint mismatch: {fingerprint(binfo)} vs {fingerprint(cinfo)}: not comparable", 4)
    if binfo.get("settings") != cinfo.get("settings"):
        die(f"settings differ:\n  base {binfo.get('settings')}\n  cand {cinfo.get('settings')}\nnot comparable", 4)
    if binfo.get("host", {}).get("driver") != cinfo.get("host", {}).get("driver"):
        print(f"WARN: driver differs ({binfo['host'].get('driver')} vs {cinfo['host'].get('driver')})")
    bs, cs = per_case(brows), per_case(crows)
    print("| case | base median | cand median | change | noise (max spread) | verdict |")
    print("|---|--:|--:|--:|--:|---|")
    regress = False
    for c in bs:
        if c not in cs or "median" not in bs[c] or "median" not in cs[c]:
            print(f"| {c} | | | | | missing or failed |")
            continue
        b, n = bs[c]["median"], cs[c]["median"]
        gain = (b - n) / b  # positive = faster
        noise = max(bs[c]["spread"], cs[c]["spread"])
        if abs(gain) <= max(noise, 0.01) or abs(gain) < LOW_RISK_BAR:
            v = "within noise" if abs(gain) <= noise else "below the 3% bar"
        elif gain > 0:
            v = "WIN (clears the 8% bar)" if gain >= COMPLEX_BAR else "win (low-risk bar only: 3-8%)"
        else:
            v = "REGRESSION"
            regress = True
        print(f"| {c} | {b:.2f} | {n:.2f} | {-100 * gain:+.1f}% | {100 * noise:.1f}% | {v} |")
    sb, sc = score_of(bs), score_of(cs)
    if sb and sc:
        print(f"\nscore {sb:.3f} -> {sc:.3f} ({100 * (sc - sb) / sb:+.1f}%)")
    if cinfo.get("correctness") != "pass":
        print(f"candidate correctness: {cinfo.get('correctness', 'unknown')} (landing needs pass)")
    return regress


def cmd_diff(a):
    sys.exit(1 if diff(a.base, a.cand) else 0)


def cmd_gate(a):
    _, info = load(a.run)
    base = ROOT / "perf" / "baselines" / fingerprint(info)
    if not (base / "results.jsonl").exists():
        die(f"no baseline for {fingerprint(info)} (perf/harness/perf_diff.py promote <run> creates one)", 3)
    print(f"baseline: {base.relative_to(ROOT)}")
    sys.exit(1 if diff(base, a.run) else 0)


def cmd_promote(a):
    rows, info = load(a.run)
    if info.get("correctness") != "pass":
        die("refusing to promote a run without a passing correctness check (bench.sh --check)", 1)
    dst = ROOT / "perf" / "baselines" / fingerprint(info)
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(a.run)
    for f in ("run.json", "results.jsonl", "summary.md"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    print(f"promoted {a.run} -> {dst.relative_to(ROOT)}")


# ---------------------------------------------------------------------------------------------------------------
# score / entry / index

def cmd_score(a):
    rows, info = load(a.run)
    st = per_case(rows)
    sc = score_of(st)
    if not sc:
        die("no score: a case failed", 1)
    print(f"{sc:.3f}")
    if a.record:
        if info.get("correctness") != "pass" and not a.force:
            die("refusing to record a score without a passing correctness check (use --force for a non-landing note)", 1)
        path = Path(a.record)
        text = path.read_text()
        cases = " ".join(f"{c} {s['median']:.2f}" for c, s in st.items())
        g = info.get("git", {})
        commit = g.get("commit", "?") + ("+wt" if g.get("dirty") else "")
        line = (f"| {info.get('date', '?')[:10]} | {sc:.3f} | {cases} | {fingerprint(info)} | {commit} | "
                f"{info.get('correctness', '?')} | {a.note or ''} |\n")
        text = text.replace("No scores recorded yet.\n", "")
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text + line)
        print(f"recorded in {path}")


def cmd_entry(a):
    rows, info = load(a.run)
    st = per_case(rows)
    g = info.get("git", {})
    lines = [f"## {info.get('date', 'YYYY-MM-DD')[:10]}: <change> — <VERDICT>", "",
             "Status: baselining | experimenting | candidate | landed | deferred.",
             "Bottleneck hypothesis (named, with the counter that shows it):",
             "Change (one factor):",
             f"Correctness: tools/ds-checks.sh {info.get('correctness', 'not run')}",
             f"Run: {info.get('label')} at git {g.get('commit')}{' + working tree ' + str(g.get('diff_sha')) if g.get('dirty') else ''}, "
             f"host {fingerprint(info)}, {info.get('repeats')} repeats, env `{info.get('experiment_env') or 'standard'}`", "",
             "| case | baseline median | this run median | change | spread |", "|---|--:|--:|--:|--:|"]
    for c, s in st.items():
        if "median" in s:
            lines.append(f"| {c} | | {s['median']:.2f} | | {100 * s['spread']:.1f}% |")
    sc = score_of(st)
    lines += ["", f"Score: {sc:.3f}" if sc else "Score: n/a", "Counters that explain it (pool ms/token, stage wait, phases):",
              "Decision (with the bar it cleared or missed):", "Rejected alternatives:", "Open questions:"]
    print("\n".join(lines))


RE_ENTRY = re.compile(r"^## (\d{4}-\d{2}-\d{2}): (.+?)(?: — ([A-Z]+))?\s*$")


def cmd_index(a):
    path = Path(a.notebook)
    text = path.read_text().splitlines(keepends=True)
    begin = next((i for i, l in enumerate(text) if l.startswith("<!-- qx:index:begin")), None)
    end = next((i for i, l in enumerate(text) if l.startswith("<!-- qx:index:end")), None)
    if begin is None or end is None:
        die("no qx:index markers in the notebook")
    head = text[:begin + 1]
    tail = text[end:]
    # line numbers refer to the file after regeneration: compute the index size first
    entries = []
    for i, l in enumerate(tail):
        m = RE_ENTRY.match(l)
        if m:
            entries.append((m.group(1), m.group(2), m.group(3) or "UNLABELED", i))
    block = ["### Index (generated)\n", "\n", "| date | entry | verdict | line |\n", "|---|---|---|--:|\n"]
    offset = len(head) + len(block) + len(entries) + 1
    for d, t, v, i in entries:
        block.append(f"| {d} | {t} | {v} | {offset + i} |\n")
    path.write_text("".join(head + block + tail))
    print(f"indexed {len(entries)} entries")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("parse"); s.add_argument("log"); s.add_argument("--case", required=True)
    s.add_argument("--repeat", type=int, required=True); s.add_argument("--exit", type=int, default=0); s.set_defaults(f=cmd_parse)
    for name, f in (("summarize", cmd_summarize), ("validate", cmd_validate), ("gate", cmd_gate), ("promote", cmd_promote),
                    ("entry", cmd_entry)):
        s = sub.add_parser(name); s.add_argument("run"); s.set_defaults(f=f)
    s = sub.add_parser("fingerprint"); s.add_argument("run"); s.set_defaults(f=lambda a: print(fingerprint(load(a.run)[1])))
    s = sub.add_parser("diff"); s.add_argument("base"); s.add_argument("cand"); s.set_defaults(f=cmd_diff)
    s = sub.add_parser("score"); s.add_argument("run"); s.add_argument("--record"); s.add_argument("--note")
    s.add_argument("--force", action="store_true"); s.set_defaults(f=cmd_score)
    s = sub.add_parser("index"); s.add_argument("notebook"); s.set_defaults(f=cmd_index)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
