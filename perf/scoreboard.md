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

- 2026-11-07, set D first row (49.601): taken after the states were regenerated with rings (exact resume) and the
  tier ranking beside them. `qx_ds_idx_score2` (decode256k 56.6 → 43.6, prefill256k 34.5 → 23.2 in its A/B) and
  the layer skip landed before this row, so the row already includes them; the saved ranking is a worse placement
  for the decode tail than the golden profile (52.5 against 46.1; notebook 2026-11-07).
| 2026-11-07 | 49.601 | decode256k 52.50 prefill256k 21.50 serve256k3 108.11 | rtx-5090_ryzen-9-9950x_121g | 642433a+wt | pass | set D (first row): decode256k, prefill256k, serve256k3 scored; decode, prefill128, serve3 kept as guards. The cases resume from the regenerated states (rings, exact) with the saved tier ranking (QX_DS_PROFILE); prefill256k and long16k opt into the layer skip. Not comparable with sets A-C. decode256k 52.5 here against 43.6 in the idx2 A/B: the saved ranking is a worse placement for the decode tail than the golden profile (46.1 without it; notebook) |
| 2026-11-07 | 34.447 | decode256k 53.20 prefill256k 6.80 serve256k3 112.99 | rtx-5090_ryzen-9-9950x_121g | 0dad410+wt | pass | set D, the chunk path (6d8ce02 tree): prefill256k in 4096-row chunks (21.5 -> 6.8, exact against the window pass), the server's prompt-only steps in 2048-row chunks at ctx >= 32k, int8 tensor-core expert kernels, in-block GEMV token groups. decode256k 53.2 and serve256k3 113.0 (+4.5% against 108.1, spread 2%: the chunk path did not run there, chunk steps 0; a drift shared by every decode-side case, +1 to +4%; the in-block GEMV groups are the suspect, A/B queued). Guards within noise of set C |
| 2026-11-07 | 31.470 | decode256k 46.40 prefill256k 6.20 serve256k3 108.34 | rtx-5090_ryzen-9-9950x_121g | 0ced38e | pass | set D on 0ced38e: case settings changed (the 256k resume cases rank the tiers by the golden profile instead of the saved ranking: decode256k 53.2 -> 46.4; prefill256k in 8192-row chunks: 6.8 -> 6.2), plus the GEMV split (width 1-8 entries back to 80 registers: decode 35.4 -> 33.5, serve256k3 113.0 -> 108.3, serve1/serve3 -2 to -4%). decode256k is not comparable with the 34.447 row on the profile account; the rest is the same cases |
| 2026-11-07 | 30.210 | decode256k 44.90 prefill256k 5.60 serve256k3 109.65 | rtx-5090_ryzen-9-9950x_121g | fd2473e | pass | set D on fd2473e with QX_RAW_LAUNCH=1 (now the default): the dense GEMV as one kernel per segment kind (default since 7a3c828) and the raw launcher for the chunk path. prefill256k 6.2 -> 5.6 (-9.7%), decode256k 46.4 -> 44.9 (the tiled top-k, 8fae0e7), serve256k3 +1.2%, guards within noise. Same cases as the 31.470 row |
