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
# the case set a scoreboard row or a promoted baseline must cover (bench.sh's default); a run with fewer cases is a
# screening run: it can be diffed against the baseline case by case, not recorded or promoted without --partial
STANDARD_CASES = ("decode", "prefill64", "long16k", "serve3")
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


# QX_DS_TAIL: single steps after the windows (decode at the context the windows built)
RE_TAIL = re.compile(r"^decode tail of (\d+) single steps at position (\d+)\+: " + NUM + r" ms per token", re.M)
RE_TAIL_PHASES = re.compile(r"phases per step: engram " + NUM + r" ms, uploads " + NUM + r", launch/replay " + NUM
                            + r", device wait " + NUM + r", history flush " + NUM + r", logits download " + NUM)
# tools/loadgen.py's table: conc reqs tokens prompt wall agg per-req ttft_med ttft_p90
RE_SCHED_PHASES = re.compile(r"sched: per step over the last (\d+): forward " + NUM + r" ms \(engram " + NUM + r", uploads " + NUM
                             + r", launch/replay " + NUM + r", device wait " + NUM + r", history flush " + NUM + r", logits download " + NUM
                             + r"\), (?:between forwards|outside the engine) " + NUM)
RE_REQ_TIERS = re.compile(r"^req: .*? vram (\d+) staged (\d+) pcie (\d+) ram (\d+) nvme (\d+)", re.M)
RE_LOADGEN = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s*$", re.M)


def parse_log(text, metric="last32", server_text=None):
    """One results row from an engine log. `metric` picks the number that becomes `value`:
    last32    ms per token over the last 32 tokens (decode and prefill windows alike)
    tail      the QX_DS_TAIL single steps' ms per token (decode at a long context)
    prefill   the windows' ms per token with the tail removed (prefill at a long context)
    serve:C   from a loadgen table: 1000 / per-request decode tok/s at concurrency C"""
    row = {"metric": {"last32": "ms_per_token_last32", "tail": "ms_per_token_tail", "prefill": "ms_per_token_windows"}
           .get(metric, "ms_per_token_per_request"), "unit": "ms/token", "value": None}
    if metric.startswith("serve:"):
        want = int(metric.split(":", 1)[1])
        for m in RE_LOADGEN.finditer(text):
            if int(m.group(1)) == want:
                per_req = float(m.group(7))
                row.update(kind="serve", concurrency=want, requests=int(m.group(2)), tokens=int(m.group(3)),
                           prompt_tokens=int(m.group(4)), wall_s=float(m.group(5)), agg_tok_s=float(m.group(6)),
                           per_req_tok_s=per_req, ttft_med_s=float(m.group(8)), ttft_p90_s=float(m.group(9)),
                           value=round(1000.0 / per_req, 2) if per_req > 0 else None)
        row["nan"] = False
        row["panicked"] = "panicked" in text or "Traceback" in text
        if row["value"]:
            row["tok_s"] = round(1000.0 / row["value"], 2)
        if server_text:
            # the server's QX_SCHED_STATS lines: the last per-step phase line and the last cumulative tier counters
            ph = None
            for ph in RE_SCHED_PHASES.finditer(server_text):
                pass
            if ph:
                row["step_phases_ms"] = dict(zip(["forward", "engram", "uploads", "launch_replay", "device_wait", "history_flush",
                                                  "logits_download", "outside_engine"], map(float, ph.groups()[1:])))
                row["step_phases_ms"]["steps"] = int(ph.group(1))
            rq = None
            for rq in RE_REQ_TIERS.finditer(server_text):
                pass
            if rq:
                g = list(map(int, rq.groups()))
                tot = max(g[0] + g[1] + g[2] + g[3] + g[4], 1)
                row["tiers_cumulative"] = {"vram": g[0], "staged": g[1], "pcie": g[2], "ram": g[3], "nvme": g[4],
                                           "vram_share": round(g[0] / tot, 4), "staged_share": round(g[1] / tot, 4),
                                           "ram_share": round(g[3] / tot, 4), "nvme_share": round(g[4] / tot, 4)}
            w = None
            for w in re.finditer(r"window widths (\{[^}]*\})", server_text):
                pass
            if w:
                row["window_widths"] = w.group(1)
            row["panicked"] = row["panicked"] or "panicked" in server_text or "fatal:" in server_text
        return row
    m = RE_DECODE.search(text)
    if m:
        row.update(kind="decode", tokens=int(m.group(1)), mean_ms_per_token=float(m.group(2)), value=float(m.group(4)))
    m = RE_WINDOWS.search(text)
    if m:
        row.update(kind="windows", window=int(m.group(1)), tokens=int(m.group(2)), mean_ms_per_token=float(m.group(3)),
                   value=float(m.group(5)))
    t = RE_TAIL.search(text)
    if t:
        row.update(tail_steps=int(t.group(1)), tail_position=int(t.group(2)), tail_ms_per_token=float(t.group(3)))
        tp = RE_TAIL_PHASES.search(text)
        if tp:
            row["tail_phases_ms"] = dict(zip(["engram", "uploads", "launch_replay", "device_wait", "history_flush",
                                              "logits_download"], map(float, tp.groups())))
        n, tn = row.get("tokens", 0), int(t.group(1))
        if n > tn and row.get("mean_ms_per_token"):
            # the windows' own mean: the whole run's time less the tail's, over the windowed tokens
            row["windows_ms_per_token"] = round((row["mean_ms_per_token"] * n - float(t.group(3)) * tn) / (n - tn), 2)
    if metric == "tail":
        row["value"] = row.get("tail_ms_per_token")
        row["kind"] = "tail"
    elif metric == "prefill":
        row["value"] = row.get("windows_ms_per_token")
        row["kind"] = "windows-prefill"
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
    # a serve case's companion server log sits next to the loadgen log as <name>.server.log
    companion = Path(str(a.log)[: -len(".log")] + ".server.log") if str(a.log).endswith(".log") else None
    server_text = companion.read_text(errors="replace") if companion and companion.exists() else None
    row.update(parse_log(text, a.metric, server_text))
    print(json.dumps(row))


def case_set_note(rows):
    """None when the run covers the standard case set; otherwise a sentence naming what is missing or extra."""
    have = {r["case"] for r in rows}
    missing = [c for c in STANDARD_CASES if c not in have]
    return f"partial case set: missing {', '.join(missing)} (standard: {', '.join(STANDARD_CASES)})" if missing else None


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
    """The geometric mean of the standard cases' medians when the run has them all; of every case the run has
    otherwise (a screening run compares on its own cases). Companion rows such as long16k-prefill and serve1 are
    recorded, never scored. None when a case failed."""
    if all(c in stats for c in STANDARD_CASES):
        picked = {c: stats[c] for c in STANDARD_CASES}
    else:
        picked = stats
    meds = [s["median"] for s in picked.values() if "median" in s]
    if not meds or len(meds) != len(picked):
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
    note = case_set_note(rows)
    if note:
        print(f"\nNOTE: {note}; this score is not comparable with the scoreboard's")


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
    # the score over the cases both runs have: a screening run (bench.sh --quick) compares on its own cases only
    common = {c: s for c, s in bs.items() if c in cs and "median" in s and "median" in cs[c]}
    sb, sc = score_of(common), score_of({c: cs[c] for c in common})
    if sb and sc:
        tag = "" if len(common) == len(bs) else f" over {', '.join(common)} only"
        print(f"\nscore {sb:.3f} -> {sc:.3f} ({100 * (sc - sb) / sb:+.1f}%){tag}")
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
    note = case_set_note(rows)
    if note and not a.partial:
        die(f"refusing to promote a {note}; a baseline must cover every standard case (--partial overrides)", 1)
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
        note = case_set_note(rows)
        if note and not a.partial:
            die(f"refusing to record a {note}; scores are comparable only over the standard set (--partial overrides, "
                "and the note must say which cases the row covers)", 1)
        path = Path(a.record)
        text = path.read_text()
        scored = [c for c in STANDARD_CASES if c in st] or list(st)
        cases = " ".join(f"{c} {st[c]['median']:.2f}" for c in scored if "median" in st[c])
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
    s.add_argument("--repeat", type=int, required=True); s.add_argument("--exit", type=int, default=0)
    s.add_argument("--metric", default="last32", help="last32 | tail | prefill | serve:C (see parse_log)"); s.set_defaults(f=cmd_parse)
    for name, f in (("summarize", cmd_summarize), ("validate", cmd_validate), ("gate", cmd_gate), ("entry", cmd_entry)):
        s = sub.add_parser(name); s.add_argument("run"); s.set_defaults(f=f)
    s = sub.add_parser("promote"); s.add_argument("run"); s.add_argument("--partial", action="store_true"); s.set_defaults(f=cmd_promote)
    s = sub.add_parser("fingerprint"); s.add_argument("run"); s.set_defaults(f=lambda a: print(fingerprint(load(a.run)[1])))
    s = sub.add_parser("diff"); s.add_argument("base"); s.add_argument("cand"); s.set_defaults(f=cmd_diff)
    s = sub.add_parser("score"); s.add_argument("run"); s.add_argument("--record"); s.add_argument("--note")
    s.add_argument("--force", action="store_true"); s.add_argument("--partial", action="store_true"); s.set_defaults(f=cmd_score)
    s = sub.add_parser("index"); s.add_argument("notebook"); s.set_defaults(f=cmd_index)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
