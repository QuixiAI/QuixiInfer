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
| 2026-11-08 | 30.047 | decode256k 45.10 prefill256k 5.60 serve256k3 107.41 | rtx-5090_ryzen-9-9950x_121g | 635ab36+wt | pass | set D on 328338b: the elastic flex tail (chunk buffers borrow the arena's tail for a prompt; warm-up fix 251d142), server chunks of 8192 by default; serve256k3 109.65 -> 107.41, the rest unchanged within spread |
| 2026-11-10 | 26.925 | decode256k 38.25 prefill256k 5.70 serve256k3 89.53 | rtx-5090_ryzen-9-9950x_121g | 4ee0e73 | pass | set D on 4ee0e73, NEW HOST STATE: driver 620.70.06 (was 595.99.02), win10 VM off, vm.compaction_proactiveness 0 (not persistent). Not comparable with earlier rows: the -10.4% against 30.047 includes the host change. 10-minute protocol: decode256k 2 repeats, other cases 1 (bench.sh --promo); serve1/serve3 from val4-serve-053805 (same commit, binaries, settings); ds-checks in two parts on the same binaries; identity identical in 3 of 3 serve256k sessions. Changes since 30.047: engine thread on its own core and server threads off it (decode256k 44.1 -> 42.0 same day), widths 2-3 on a 4/8 staged share (serve256k3 106.3 -> 95.3), exclusive tiers QX_DS_RAM_DEDUP=2 (decode256k 42.0 -> 38.3 and serve256k3 95.06 -> 90.01, same-binary controls). prefill256k 5.7 = 5.57-5.71 on this host state in both tier modes. serve3 78.62 is one session, the 87.9 -> 78.6 step unattributed. Open: slow first token about 1 session in 10, unexplained. |
| 2026-11-10 | 25.447 | decode256k 38.50 prefill256k 4.70 serve256k3 91.07 | rtx-5090_ryzen-9-9950x_121g | 41dec22 | pass | set D on 013f98c (run.json records 41dec22, notebook-only; HEAD 3e0d906 differs in code by one comment in worker_ds.rs; binaries qx-ds-check 1a1d5cf9cb9d, quixiinfer bc1b41e0bb3d). NUMERICS CHANGE FOR PROMPT CHUNKS (the user's contract of 2026-11-10): the chunk path's dense products are f16 x f16 -> f32 tensor-core GEMMs (own kernel qx_gemm16), no longer bit-identical to single steps: KL mean 0.0089, top-1 97.5% over 511 positions after a 16k prompt; KL mean 0.0048, max 0.044, top-1 100% over 64 positions at the prefill256k configuration (bound 0.03 / 95%). Decode, windows and QX_DS_PREFILL_EXACT=1 stay exact. Consequence: the server takes chunks only while no slot decodes, so the temperature-0 output of a long prompt can depend on server load. prefill256k 5.70 -> 4.7 (-17.5%): expert side streams (exact) to 5.2, the f16 kernel to 4.7. Costs, inside the measured spreads: serve256k3 91.07 (+1.7%; candidate-tree sessions 90.3-91.7 against 89.5-90.0: the larger flex tail leaves 102 VRAM slots empty after warm-up instead of 77, not fixed), decode256k 38.3 / 38.7 (+0.7%), decode 31.0 (+1.6%). 10-minute protocol: decode256k 2 repeats, other cases 1; serve1/serve3 from v7-serve-212321; ds-checks in two parts on the same binaries; identity identical in 2 of 2 serve256k sessions (short prompts, chunk steps 0: decode only); one three-slot 256k session with a 13,283-token prompt completed on the f16 path (chunk steps 2; second chunk 14.5 s against 16.2 s exact; TTFT 30.6 s against 31.6 s with the exact knob, one session each). Kernels: gemm.cu (the only kernel file changed) compiles for sm_86, sm_89, sm_120a. Open: v7-serve had the slow first token (TTFT median 3.01 s; its serve256k3 88.81 is not comparable); a slow decode256k mode (40.8-42.1) in 5 of 11 candidate runs 19:20-20:02, 0 of 11 after 20:04 and not in this run, cause unknown; serve256k1 +2% unattributed. Strata's prefill remains about 2x ahead. |
