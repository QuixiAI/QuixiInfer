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
