# Conjecturer pass, 2026-11-10 (perf-conjecturer agent, read-only, facts only; condensed by the caller)

## Corrections to the caller's numbers (arithmetic on existing logs)
1. `host call ~705 us / gap ~1,180 us` are cumulative over 64 tokens incl. warm-up. Scored last 32 tokens
   (mix-off / mix-predmix / mix-mk6c1, 2026-11-09), per token: off 41.8 = host call 22.8 + gap 19.1, pool 20.2 (of
   which NVMe fetch 7.0), staged/RAM/NVMe 24 / 40.5 / 7.0, 20 VRAM promotions; predictor only 43.0 (gap 20.1);
   mix K6 one copy 40.6 = 19.6 + 21.1, pool 16.9 (fetch 7.0), 17 / 29 / 6.3. Prefetch residue on the GPU side is
   2.0 ms (predictor 1.0, ~0.5 for ~20 more GPU-run selections, ~0.5 unattributed), not 3.6.
   NVMe fetch wait = 7.0 ms per token (17% of decode256k) for 7 records; the prefetch does not touch it.
2. Concurrency 3 (wide-s256-min4): a width-3 step = ~82 ms host call (pool wall 78) + 23 ms gap; ~445 host records
   = 4.2 GB in 82 ms = 51 GB/s = the DRAM ceiling. The server's single stream has a 12.3 GiB tier, 47% VRAM hit
   (70% in decode256k), ~127 host records per token at ~30 GB/s. Three streams = 3x bytes at 100% of the ceiling:
   ~47 of the extra 57 ms; DRAM idles during the 23 ms gap (22% of the step): the only slack.
3. **Tier leak (code read, unmeasured):** `ExpertStore::build(.., &in_vram, ..)` leaves the initial VRAM residents
   out of RAM (10,106 + 3,641 + 1,613 = 15,360); a promotion (driver.rs ~712-740) needs the candidate in RAM and
   zeroes the victim's entry without writing it back. An evicted initial resident is NVMe-only; its replacement
   holds VRAM and RAM both. decode256k: 1,467 VRAM promotions per 64 tokens against 1,613 slots; server run 8,945
   against 1,275. Steady state: unique capacity falls toward RAM alone (-15 GiB decode, -12 GiB server).
   Backlog 15.15 dismissed write-back on "inclusive at the VRAM/RAM boundary": true only for promoted records.

## Ranked experiments
- E1 tier leak: measure (NVMe-served selections that were initial residents; `QX_DS_VRAM_PROMOS=0` arms reading NVMe
  per token), then exclusive tiers (victim takes the promoted record's RAM place). decode256k up to -3 ms (-7%),
  serve256k3 -5..-12%, prefill256k -2..-4%. Exact: read-only weights. Row 15.
- E2 prefetch at width 3 in the server (copies in the 23 ms gap): -8..-20% serve256k3 if copies land. Rows 25/28.
- E3 unused context capacity as an expert cache (long shot; gain only at short contexts). Row 8.
- E4 NVMe read-ahead from the `next` predictor on a branch parallel to the wait kernel: -3..-4 ms decode256k. Rows 6, 14.
- E5 promotion churn at 256k: `QX_DS_VRAM_PROMOS=4/8/16`, `QX_DS_MARGIN`. 0..-1.5 ms; separates causes for E1.
- E6 counter repair: scored-window counters, per-kernel split of the gap. Row 24.
- E7 width-1 GEMV fusion (6.5 ms over 511 launches, floor ~2): -3..-4 ms decode256k. Row 19.
- E8 is the chunk's "slab enqueue 32.9 of 46.4 s" host cost or GPU back-pressure? One nsys run. Rows 10/21.
Rejected/kept closed: per-width share tuning beyond the sweep, two copies per layer on decode, latent rows to host
for decode256k; staggered slots behind E2; idea: drop the pack's page cache after the fill (no sysctl) for the slow mode.
