#!/bin/bash
# Runs a command inside a memory-capped systemd scope, so a runaway job is killed by itself instead of taking the
# machine (and the agent harness) down. Usage: tools/capped.sh [GIB] command args...  (default cap 90 GiB).
set -euo pipefail
cap=90
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then cap=$1; shift; fi
exec systemd-run --user --scope --quiet -p MemoryMax="${cap}G" -p MemorySwapMax=0 -- "$@"
