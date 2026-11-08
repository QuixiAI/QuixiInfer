# Breaker report on df104c7 (thread placement), 2026-11-08

## Findings on df104c7 (pool on cores 0-14, engine thread pinned to core 15)

### Would corrupt results

Nothing found. The diff is placement only: the pool's per-element arithmetic is independent of the thread split (`split(n, t, nt)` in `/home/eric/git/QuixiInfer/crates/qx-cpu/src/pool.rs` assigns whole rows/experts to single threads; no reductions cross threads), so 15 vs 16 workers is bit-identical by construction, and `perf/results/2026-11-08/setd-pin-164650/logs/ds-checks.log` passes all five on chunk36. Checked: no new memory written or re-mapped; `Pool::new` still pins the driver (worker 0) to `cpus[0]`; the mailbox, flex tail and chunk paths are untouched.

### Would invalidate the number or the default

**1. Affinity inheritance: the pin also confines the CUDA driver's busy thread and rayon to CPU 15 (confirmed on the live setd-pin server, PID 555833).** `sched_setaffinity` is inherited by every thread spawned afterwards, and in both `qx-ds-check.rs:48` and `worker_ds.rs:169` the pin precedes `load_tiered_slots`, which initializes CUDA, runs the rayon fill and starts the driver. Live `/proc/<pid>/task/*/status`:

```
555836 quixiinfer       allowed=15  utime=2179   <- engine thread
555837 DSD-0-0          allowed=15  utime=1454   <- CUDA driver thread, 2/3 of the engine thread's CPU time
555838 cuda00001400006  allowed=15
555843 cuda-EvtHandlr   allowed=15
555846 quixiinfer       allowed=15  utime=68     <- rayon's single worker (available_parallelism()=1 under the mask)
555899..913            allowed=0..14             <- pool (worker 0 = driver, utime 1590)
```

CPU 31 (the sibling) is idle and unavailable to any of them. Three consequences:

- *Mechanism attribution is confounded.* In the unpinned arms the note (AFFIN2) saw "a CUDA helper thread on CPU 16" — that is this DSD thread, on the sibling of pool core 0, i.e. the driver's core. The pinned arm moved both the engine thread and the DSD thread off the pool cores at once. The 45→42 may be mostly the DSD thread, not "the engine's sync spins". The sib31 arm (44.5) is consistent with either reading.
- *A hidden serialization:* two busy threads (21.8 s and 14.5 s of CPU) on one hardware thread. This could be costing GPU-side latency now and is the first suspect if the server's "outside"/chunk-enqueue paths ever regress.
- *Measured startup regression:* RAM-tier fill 8.3-8.4 s → 12.0-12.1 s in every pinned qx-ds-check log, 7.9 → 11.5 s in the server; server warm-up 14.9-17.1 → 21.6 s (1k) and 22.4-24.0 → 26.2 s (256k). Compare `grep "filled in\|warm-up" perf/results/2026-11-08/setd-flex-145532/logs/*` against `setd-pin-164650/logs/*`.

Tests: (a) during a run, `taskset -pc 31 <DSD-0-0 tid>` and watch the per-8-token stats line or the `sched:` line; (b) move the pin after `load_tiered_slots` (or re-pin after CUDA init) so the helpers keep the wide mask, and run decode256k three ways: both on 15 (now), engine 15 + helpers free, engine 15 + helpers on 31. If the gain shrinks when the DSD thread is freed, the mechanism is not the engine's spin.

**2. The default ignores the process's allowed CPU set, and `pin()` ignores EINVAL.** `physical_cores()` (`/home/eric/git/QuixiInfer/crates/qx-engine/src/dsv41/mod.rs:447`) reads host sysfs topology, which containers and cgroup cpusets still expose in full; `pin()` in `pool.rs:112` and `mod.rs:500` discard `sched_setaffinity`'s return. In a restricted cpuset the pins silently fail and 15 spinning workers plus the driver float on a few CPUs, with `wait_barrier` never yielding — a convoy that looks like a hang. That is the "confined" arm: `taskset -c 14,15,30,31` made pins to 0-13 fail, and `perf/results/2026-11-08/affin-confined-161857/logs/decode256k-1.log` ends at `driver idle 15 s: MB_SEQ 0`. The notebook calls it "not the way to do it"; it is a latent bug the new default now exposes to every container user. Test: `taskset -c 0-3 target/release/qx-ds-check ...` with defaults (expect the idle line) vs `QX_DS_CPUS=0-2 QX_DS_MAIN_CPU=3` (expect it to run). Fix: intersect with `sched_getaffinity(0)`, check the pin's return, print the final placement.

**3. `physical_cores()` stops at the first CPU without a topology directory** (`else { break }`). Offline CPUs have no `topology/`; with Intel's adjacent sibling numbering and `nosmt`, cpu1 is offline → `cores = [0]` → a one-core pool. Any hotplug hole below the last core truncates the pool. Fix: `continue`, or walk `/sys/devices/system/cpu/online`.

**4. No cap on the pool size.** The old default was 16; the new one is "all physical cores but one". On a 64-core EPYC that is 63 spinning workers across NUMA nodes, with two barriers per job, where `qx-poolbw` showed 16 cores already at 43 GB/s against a 52-59 GB/s DRAM ceiling. Untested anywhere; suggest `min(cores-1, 16)` or an explicit statement that the default is for ≤16-core parts. Also note `qx-poolbw` now defaults to 15 cores, so the recorded 0.230 ms/record 16-core figure is no longer the default's.

### Measurement provenance (checked, cleared)

- Build confound: chunk36 was an 8-second incremental build (`.fingerprint/qx-engine-*/invoked.timestamp` 16:34:01, binaries 16:34:06-09), after the control arm (16:30-16:33) and during only the load phase of core15b. Clear.
- Arms identical in settings: same `run.json` settings string, same VRAM budget cut (20 → 15.4 GiB, 1609 records), same promos (1442/533), same generated token ids. The control (45.00, 3.3% spread from one 43.9 repeat) is corroborated by the clean 16:15 run (45.10) and the morning setd-flex row (45.10). Arm order was interleaved with a repeated control. Clear.
- The arms' binary (chunk35, `QX_DS_MAIN_CPU=15` env) pinned at the same code point as chunk36's default, so the inheritance effect above is inside the measured 42.0 — the number is reproducible, its attribution is not.

### Hidden costs and collateral

- `fetch_ms` rose 15-19% (453 → 537 ms per 64 tokens) in every pinned arm. The code explains it: in `driver.rs:548-557` the pool runs the RAM records first, then `fetch_wait` is timed — a faster pool exposes more NVMe latency. So the pool actually gained ~4.4 ms/token and NVMe took back ~1.3; the next critical path at 256k is the ~7 NVMe records per token, and further pool wins will not translate 1:1.
- "mean over all tokens" is 200-400 ms unpinned vs 111 pinned (every repeat): the cold first tokens (JIT, first forward) are 2-4x faster with the engine thread pinned. A win and not scored, but unexplained in the notebook.
- serve256k3 is flat (106.6/107.3 vs 107.41), as expected: width ≥2 uses the 8/8 staging share and the pool idles. The score gain is decode256k alone (~-2.3% on the geomean).
- The non-tiered path (`DsEngine::load`, used by the ds-checks' f32 reference) now pins the engine thread and the inherited CUDA threads to one hardware thread with no pool to protect: a pure cost there.

Files: `/home/eric/git/QuixiInfer/crates/qx-engine/src/dsv41/mod.rs` (447-505), `/home/eric/git/QuixiInfer/crates/qx-serve/src/worker_ds.rs` (166-172), `/home/eric/git/QuixiInfer/crates/qx-engine/src/bin/qx-ds-check.rs` (46-53), `/home/eric/git/QuixiInfer/crates/qx-cpu/src/pool.rs` (112-125, 222-228), `/home/eric/git/QuixiInfer/crates/qx-engine/src/tiers.rs` (121-130), `/home/eric/git/QuixiInfer/crates/qx-engine/src/dsv41/driver.rs` (538-560).
