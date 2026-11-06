# Scoreboard

This is the engine's one hill-climbable number. It is the geometric mean of the median ms per token of the
standard cases (`decode` and `prefill64`, last 32 tokens of route1k 1024, three repeats). Each row is recorded at
a landed win, and lower is better.

Scores are comparable only within one host fingerprint, one settings line and one case set. The standard
settings are `QX_DS_VRAM=20 QX_DS_RAM=100 QX_DS_BENCH=0 QX_DS_PROMOS=32`. Add a row with a note when the case set
or settings change.

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
| 2026-11-05 | 19.514 | decode 32.00 prefill64 11.90 | rtx-5090_ryzen-9-9950x_121g | f359604 | pass | first harness baseline (f359604: staged share, kernel-load deadlock fix) |
