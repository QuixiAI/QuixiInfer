# Scoreboard

This is the engine's one hill-climbable number. It is the geometric mean of the median ms per token of the
standard cases, three repeats each. Each row is recorded at a landed win, and lower is better.

Scores are comparable only within one host fingerprint, one settings line and one case set. The standard
settings are `QX_DS_VRAM=20 QX_DS_RAM=100 QX_DS_BENCH=0 QX_DS_PROMOS=32`. Add a row with a note when the case set
or settings change.

Case sets so far:

- **Set A (to 2026-11-05):** `decode` and `prefill64`, last 32 tokens of route1k 1024.
- **Set B (from 2026-11-06):** set A plus `long16k` (the last 32 single steps of a 64-step tail after 16,384 tokens
  in windows of 64) and `serve3` (per-request decode under `loadgen.py` at concurrency 3, 3 slots). See `perf.md`,
  "The Metrics". The serve tier was 90 GiB in the first set B row and is 95 GiB from the second (a measured
  -5.4% on serve3; the tier is in the recorded settings).
- **Set D (from 2026-11-06, the user's priority: performance at 256k):** `decode256k` (64 single steps from a
  saved 262,144-token state, last 32), `prefill256k` (the last 16k tokens of the 256k input in windows of 128
  from a saved 245,760-token state, last 32), `serve256k` (the server at `--ctx 262144 --slots 3`), with
  `decode`, `prefill128` and `serve3` kept as short-context guards. Set D scores are not comparable with earlier
  sets.
- **Set C (from 2026-11-06, the user's decision):** the 128-token prompt chunk with the full staged share (8/8)
  and a 256-record staging area is the default, so the standard prefill case is `prefill128`, `long16k` runs in
  windows of 128, and the server serves at `--window 128`. Set C scores are not comparable with set B's: prefill
  is faster (10.8 against 12.3) and serve3 slower (about +10%) by construction.
  Set B scores are larger than set A's, since the two new cases are the slow regimes; a set B score is not
  comparable with a set A one. `perf_diff.py` refuses to record a run that does not cover set B.

Record a row with this command, which refuses a run without a passing correctness check:

```bash
python3 perf/harness/perf_diff.py score <run-dir> --record perf/scoreboard.md --note "<what changed>"
```

Historical points from before the harness use comparable settings, a single run each, and `docs/progress.md` as
the source:

| date | decode ms/token | prefill64 ms/token | score | note |
|---|--:|--:|--:|---|
| 2026-11-04 | 31.9 | 21.8 | 26.4 | flash-decoding attention; windows without staging |
| 2026-11-04 | 31.4 | 11.8 | 19.2 | DMA-staged GPU share |

Harness-recorded scores:

| date | score (geomean ms) | case medians | host | git | correctness | what changed |
|---|---|---|---|---|---|---|
| 2026-11-05 | 19.514 | decode 32.00 prefill64 11.90 | rtx-5090_ryzen-9-9950x_121g | f359604 | pass | set A. first harness baseline (f359604: staged share, kernel-load deadlock fix) |
| 2026-11-06 | 34.327 | decode 33.10 prefill64 12.10 long16k 36.40 serve3 95.24 | rtx-5090_ryzen-9-9950x_121g | 0dc3dbd+wt | pass | set B (first row): harness long16k+serve3; capture-race fix; warm-up of every width, sealed kernel set, 2.5 GiB VRAM headroom; stage share 6/8 from width 2; allocator thresholds pinned |
| 2026-11-06 | 34.321 | decode 33.00 prefill64 12.40 long16k 37.30 serve3 90.91 | rtx-5090_ryzen-9-9950x_121g | 4e782fb+wt | pass | set B at the serve tier 95 GiB (settings change: serve3 90.9 against 95.2 at 90; decode, prefill64 and long16k within noise of the first set B row) |
| 2026-11-06 | 34.621 | decode 34.00 prefill128 11.00 long16k 38.80 serve3 99.01 | rtx-5090_ryzen-9-9950x_121g | b543bce+wt | pass | set C (first row): the 128-token chunk with share 8/8 and cap 256 as defaults (the user's decision); prefill128 11.0 against prefill64 12.4 at set B; serve3 99.0 against 90.9 at set B (the staging area's VRAM); decode and long16k (now windows of 128) within noise |

- 2026-11-06, set D: no baseline row yet. The states the set D cases resume from are being regenerated (rings in
  the state file, the tier ranking beside it; see the notebook's "The set D cases resume into a cold tier"). The
  first 256k win is recorded from its A/B instead: `qx_ds_idx_score2`, decode256k 56.6 → 43.6, prefill256k
  34.5 → 23.2 ms per token (notebook, "the indexer scorer with the queries in shared memory").
