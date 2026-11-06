#!/usr/bin/env bash
# The standard end-to-end benchmark for the DeepSeek-V4.1-Flash engine. See perf/perf.md ("Measurement harness").
#
#   perf/harness/bench.sh [--label L] [--repeats N] [--cases decode,prefill64] [--check] [--no-build] [-- ENV=V ...]
#
# Each run writes perf/results/<date>/<label>-<HHMMSS>/ (git-ignored):
#   run.json       provenance: git commit and dirty state, host fingerprint, driver, knobs, command lines
#   results.jsonl  one row per (case, repeat): the metric plus the tier and phase counters the run printed
#   summary.md     per case median / min / max / spread, and the score
#   logs/          every engine run's full output
# then prints the summary and a pre-filled notebook entry. Trailing ENV=V arguments are passed to every engine run
# (an experiment's knobs); they are recorded in run.json.
#
# Guards: re-executes itself inside a memory-capped systemd scope when it is not in one (an OOM takes the agent
# harness down); refuses to start while another process holds the GPU (contention invalidates absolute numbers);
# waits for the GPU to be released between runs (a pinned process takes 10-30 s to tear down); gates on cargo's
# exit status. --check runs tools/ds-checks.sh first and records whether it passed: a run without a passing check
# can be measured but not landed.
set -euo pipefail
cd "$(dirname "$0")/../.."
ORIG_ARGS=("$@")

LABEL=run
REPEATS=3
CASES=decode,prefill64
CHECK=0
BUILD=1
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL=$2; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --cases) CASES=$2; shift 2 ;;
    --check) CHECK=1; shift ;;
    --no-build) BUILD=0; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

# the memory cap: re-exec inside a scope unless this process already has a limit
cg=$(cut -d: -f3 /proc/self/cgroup)
lim=$(cat "/sys/fs/cgroup$cg/memory.max" 2>/dev/null || echo max)
if [ "$lim" = max ] && [ -z "${QX_PERF_CAPPED:-}" ]; then
  exec env QX_PERF_CAPPED=1 systemd-run --user --scope --quiet -p MemoryMax=118G -p MemorySwapMax=0 -- "$0" "${ORIG_ARGS[@]}"
fi

export PATH=$HOME/.cargo/bin:$PATH
D=${QX_DS_DIR:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS}
M=${QX_DS_MODEL:-$D/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf}
PACK=${QX_DS_PACK:-$D/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS.qxp}
TOKENS=golden/data/route1k.tokens
B=target/release/qx-ds-check
# the standard settings every recorded DeepSeek number since 2026-11-04 used; change them only with a note in the
# scoreboard (scores are comparable only at the same settings)
BASE=(QX_DS_PACK="$PACK" QX_DS_VRAM=20 QX_DS_RAM=100 QX_DS_BENCH=0 QX_DS_PROMOS=32 QX_DS_VERBOSE=1)

for f in "$M" "$PACK" "$TOKENS"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 2; }; done

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_free() { local i=0; until [ "$(gpu_used)" -lt 1000 ]; do sleep 2; i=$((i + 1)); [ $i -lt 90 ] || { echo "GPU still busy after 180 s" >&2; return 1; }; done; }
if [ "$(gpu_used)" -ge 1000 ]; then
  echo "waiting for the GPU (in use: $(gpu_used) MiB):" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
  gpu_free
fi

if [ $BUILD = 1 ]; then
  set +e
  cargo build --release -p qx-engine --bin qx-ds-check > /tmp/qx-perf-build.$$ 2>&1
  rc=$?
  set -e
  if [ $rc != 0 ]; then tail -30 /tmp/qx-perf-build.$$ >&2; rm -f /tmp/qx-perf-build.$$; echo "build failed ($rc)" >&2; exit 1; fi
  rm -f /tmp/qx-perf-build.$$
fi

RUN=perf/results/$(date +%F)/$LABEL-$(date +%H%M%S)
mkdir -p "$RUN/logs"
echo "run directory: $RUN"
# provenance of the binary just built, taken now (edits made while the run goes on must not be attributed to it)
git rev-parse --short HEAD > "$RUN/git-commit"
# tools/golden/golden-dsv41 is a tracked build output left modified on purpose: not engine state
git status --porcelain --untracked-files=no -- . ':!tools/golden/golden-dsv41' > "$RUN/git-status"
git diff HEAD -- . ':!tools/golden/golden-dsv41' > "$RUN/working-tree.diff"
[ -s "$RUN/working-tree.diff" ] || rm -f "$RUN/working-tree.diff"

check=skipped
if [ $CHECK = 1 ]; then
  echo "correctness: tools/ds-checks.sh"
  if tools/ds-checks.sh > "$RUN/logs/ds-checks.log" 2>&1 && ! grep -q "^[0-9]\..*: *$" "$RUN/logs/ds-checks.log"; then
    check=pass
  else
    check=fail
  fi
  cat "$RUN/logs/ds-checks.log"
  # exact-match expectations of the regression set (perf/perf.md, "Correctness gate")
  if [ $check = pass ]; then
    grep -q "top-1 agreement 6/6, KLD mean 0.000000" "$RUN/logs/ds-checks.log" || check=fail
    [ "$(grep -c 'bit-identical rows: 128 / 128' "$RUN/logs/ds-checks.log")" = 2 ] || check=fail
    grep -q "^4\. .*identical" "$RUN/logs/ds-checks.log" || check=fail
  fi
  echo "correctness: $check"
fi

declare -A ARGS=( [decode]="" [prefill64]="QX_DS_WINDOW=64" [prefill32]="QX_DS_WINDOW=32" [prefill128]="QX_DS_WINDOW=128" )
IFS=, read -ra CASE_LIST <<< "$CASES"
for c in "${CASE_LIST[@]}"; do [ -n "${ARGS[$c]+x}" ] || { echo "unknown case $c (known: ${!ARGS[*]})" >&2; exit 2; }; done

cmds=()
for r in $(seq 1 "$REPEATS"); do
  for c in "${CASE_LIST[@]}"; do
    gpu_free
    log="$RUN/logs/$c-$r.log"
    # shellcheck disable=SC2206
    env_args=("${BASE[@]}" ${ARGS[$c]} "${EXTRA[@]}")
    cmd="env ${env_args[*]} $B $M $TOKENS 1024"
    cmds+=("$cmd")
    echo "[$c $r/$REPEATS] $(date +%T)"
    set +e
    timeout 1200 env "${env_args[@]}" $B "$M" "$TOKENS" 1024 > "$log" 2>&1
    rc=$?
    set -e
    [ $rc = 0 ] || echo "  exit $rc (see $log)"
    python3 -I perf/harness/perf_diff.py parse "$log" --case "$c" --repeat "$r" --exit "$rc" >> "$RUN/results.jsonl"
  done
done

python3 -I - "$RUN" "$LABEL" "$REPEATS" "$CASES" "$check" "${EXTRA[*]:-}" "${BASE[*]}" "${cmds[@]}" <<'EOF'
import json, os, subprocess, sys, datetime, platform, hashlib
run, label, repeats, cases, check, extra, base, *cmds = sys.argv[1:]
def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        return ""
gpu = sh("nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total", "--format=csv,noheader").split(", ")
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
mem_gib = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2**20)
def rd(name):
    p = os.path.join(run, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""
diff = rd("working-tree.diff")
status = rd("git-status").strip()
info = {
    "schema": 1,
    "label": label,
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "git": {"commit": rd("git-commit").strip(), "dirty": bool(status), "modified": status.splitlines(),
            "diff_sha": hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else None},
    "host": {"gpu": gpu[0] if gpu else "", "driver": gpu[1] if len(gpu) > 1 else "", "power_limit": gpu[2] if len(gpu) > 2 else "",
             "vram": gpu[3] if len(gpu) > 3 else "", "cpu": cpu, "ram_gib": mem_gib, "kernel": platform.release(),
             "rustc": sh("rustc", "--version")},
    "settings": base, "experiment_env": extra, "cases": cases.split(","), "repeats": int(repeats),
    "correctness": check, "commands": sorted(set(cmds)),
}
json.dump(info, open(os.path.join(run, "run.json"), "w"), indent=2)
EOF

python3 -I perf/harness/perf_diff.py summarize "$RUN" | tee "$RUN/summary.md"
python3 -I perf/harness/perf_diff.py validate "$RUN" || echo "do NOT record this run as a measured result: see the warnings above"
echo
python3 -I perf/harness/perf_diff.py entry "$RUN"
