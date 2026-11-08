# Adjudicator verdict on perf/results/2026-11-08/setd-pin-164650 (chunk36), 2026-11-08

**Verdict: HOLD** — one missing check (point 2); every other point holds. What lifts it is named under point 2.

Candidate: `/home/eric/git/QuixiInfer/perf/results/2026-11-08/setd-pin-164650` (df104c7, clean, chunk36) against the baseline `/home/eric/git/QuixiInfer/perf/baselines/rtx-5090_ryzen-9-9950x_121g` (setd-flex-145532, 635ab36+wt, 30.047).

**2. Correctness — the server-level identity check is missing (the HOLD).**
`run.json` records `"correctness": "pass"`, and `logs/ds-checks.log` has all six lines exact (6/6 KLD 0.000000; 57/59 KLD 0.128643; 128/128 twice; tokenizer identical; rewind and park 64/64). But df104c7 touches the server: `git show df104c7 -- crates/qx-serve/src/worker_ds.rs` adds `pin_thread(cpu)` at the top of the worker's engine thread, before `load_tiered_slots`. The brief requires a `tools/loadgen.py --dump` vs reference comparison through `tools/compare_dumps.py` ("identical") for changes touching the server, and neither the notebook entry (lines 2086-2099 of `/home/eric/git/QuixiInfer/perf/optimization_status.md`) nor `handoff.md` has one for chunk36 or chunk37 (`grep -i 'compare_dumps\|--dump'` after line 1900 finds only the chunk32 flex-tail check). The breaker's "bit-identical by construction" argument and the identical decode256k token lists in the logs are engine-level evidence, not the server-level check the rule asks for.
*Lifts the HOLD:* a `loadgen.py --dump` run of the 8 temperature-0 prompts on the candidate server (262144 / 3 slots) compared with `tools/compare_dumps.py` against the existing no-chunk reference `perf/results/2026-11-07/srv2-chunk32/srv2-0.dump.jsonl`, reporting "identical", quoted in the entry. Given that chunk37 (5d185f2) already replaces this placement, the practical route is to run that check on chunk37 and adjudicate `setd-place` instead; df104c7's row stays RECORDED.

**1. Gate arithmetic — clears the 3% low-risk bar on the targeted case.**
```
python3 perf/harness/perf_diff.py gate perf/results/2026-11-08/setd-pin-164650
decode256k   45.10 -> 42.20  -6.4%  noise 1.7%   win (low-risk bar only: 3-8%)
prefill256k   5.60 ->  5.60  -0.0%
serve256k3  107.41 -> 106.61 -0.7%  noise 0.8%   within noise
decode       34.60 -> 33.80  -2.3%  noise 4.7%   within noise
prefill128   11.00 -> 10.80  -1.8%  noise 0.9%   below the 3% bar
serve3       96.81 -> 96.43  -0.4%  noise 0.9%   within noise
score 30.047 -> 29.316 (-2.4%)
python3 perf/harness/perf_diff.py score ...  -> 29.316
```
decode256k repeats: baseline 45.0/45.7/45.1 (spread 1.6%), candidate 42.0/42.7/42.2 (1.7%); the -6.4% is outside both spreads. The score itself moves only -2.4%, below 3%, but perf.md's rule is "the score or the targeted case", and the change is a placement (3% bar). Guards within noise; nothing regresses. serve256k1 -9.8% is unscored and inside its 7.2% spread.

**3. Like for like — holds, with one note.**
Settings strings identical (`QX_DS_VRAM=20 QX_DS_RAM=100 ... serve:QX_DS_RAM=95 serve:window=128`, `QX_DS_CHUNK=8192` on prefill256k, golden profile), same tier state in every decode256k log (1609 records / 15.4 GiB VRAM, 10105 in RAM, promos 1442/533, same generated tokens). Binary state constant: bench ran `--no-build` (journal, scope started 16:46:50), the logs' `cpus: pool on [0..14], engine thread on 15` line is df104c7's format (chunk37 prints `... other threads on ...`), `~/qx-bins/chunk36` built 16:34:09, and `target/release/*` was only overwritten at 17:34:52, after `run.json` (17:34:29). No other result directory overlaps the run; the journal shows no other GPU process. **Note:** a chunk37 build (`cargo build -j4 --release`, nice 15, 8 s) ran 17:31:05-17:31:13, between `prefill128-3.log` closing (17:31:03.7) and serve-3's loadgen phase (server load 11.5 s + warm-up 21.5 s + 6 warm requests after that). It overlapped only serve-3's tier fill, not any measured window, so the numbers stand, but it is a compile on the CPU during the run.

**4. The entry's claims — all found.**
decode256k 42.20 (1.7%), prefill256k 5.60, serve256k3 106.61 (0.8%), decode 33.80 (4.7%), prefill128 10.80 (0.9%), serve3 96.43 (0.3%), score 29.316 (-2.4%), decode256k -6.4%: all in `summary.md`/gate output. "Tier fill 8.3 -> 12.0 s": 8.3/8.4 s vs 12.0/12.1 s in the decode256k logs. "Warm-up 15 -> 22 s": 14.9/14.9/17.1 vs 21.5/21.6/22.3 s (1k server); the 256k server also rose 22.4-24.0 -> 26.2-29.1 s. Counters are consistent with "the pool got faster": pool per 64 tokens 1638/1648/1647 ms vs 1829/1907/1851; fetch up 450-455 -> 512-526 (the breaker's exposed-NVMe effect); device wait 41.4-41.5 vs 43.9-44.5 ms. Attribution between the engine thread and the CUDA helper that moved with it is unresolved, and the entry says so.

**5. Scope — holds, with a caveat.** df104c7 is a committed, clean tree (`dirty: false`, `git-status` empty). But HEAD (5d185f2) has already replaced this placement mechanism, so promoting this baseline would pin the hill to a superseded default; acceptable as a stepping stone, better replaced by chunk37's run.

**6. Exploration cadence — no flag.** The four experiments before the candidate entry (timeline counters + `qx-poolbw` by core count; core clocks; DMA contention and page size; the engine thread on its own core) directly ran the killing tests of `perf/presuppositions.md` rows 2 (overturned), 3 (held) and 18 (partly). The next experiment is not forced to be a long shot.
