//! The CPU expert pool: persistent worker threads, one per physical core, that compute a layer's routed
//! experts the GPU does not hold.
//!
//! A job runs in three phases separated by spin barriers:
//! 1. gate and up rows of every expert, split evenly across threads;
//! 2. `h = silu(gate) * up` (clamped for DeepSeek) quantized to Q8_0 or Q8_K as the down format needs, one expert
//!    per thread;
//! 3. down rows: thread t owns output rows [t*n/T, (t+1)*n/T) of every expert.
//! Outputs are unweighted and per expert, so the GPU combines all of a layer's experts in one fixed order
//! whichever tier computed each; the arithmetic matches the GPU expert kernels bit for bit.
//! The calling thread is worker 0, so a job costs no wake-up latency on the caller's side.
use crate::act::{quantize_q8_0, quantize_q8_k, Q80, Q8K};
use crate::dot::avx512;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// A weight format the pool has dot products for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WType {
    Iq2S,
    Iq3S,
    Iq4Nl,
    /// With the per-tensor scales in `Expert::scale`.
    Nvfp4,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Q2K,
}

impl WType {
    /// Whether this format is dotted with Q8_K activations (256-element blocks) rather than Q8_0.
    pub fn q8k_activation(self) -> bool {
        !matches!(self, WType::Iq4Nl | WType::Nvfp4)
    }
}

/// The formats of one layer's routed experts: gate and up share `gate`; `clamp` > 0 selects the clamped SwiGLU
/// `silu(min(g, clamp)) * clamp(u, -clamp, clamp)` (DeepSeek-V4.1), 0 the plain `silu(g) * u`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExpertFmt {
    pub gate: WType,
    pub down: WType,
    pub clamp: f32,
}

impl ExpertFmt {
    /// IQ2_S gate and up, IQ4_NL down.
    pub const IQ2S: Self = Self { gate: WType::Iq2S, down: WType::Iq4Nl, clamp: 0.0 };
    /// IQ3_S gate and up, IQ4_NL down.
    pub const IQ3S: Self = Self { gate: WType::Iq3S, down: WType::Iq4Nl, clamp: 0.0 };
    /// NVFP4 throughout.
    pub const NVFP4: Self = Self { gate: WType::Nvfp4, down: WType::Nvfp4, clamp: 0.0 };
}

/// One expert to compute for one token; its unweighted output goes to row `out` of the output matrix.
#[derive(Clone, Copy)]
pub struct Expert {
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
    pub out: usize,
    /// The token (index into the job's inputs) it applies to.
    pub tok: usize,
    /// NVFP4: the expert's per-tensor scales {gate, up, down}, applied to the gate and up outputs before the
    /// activation and to the down output (ignored by the other formats).
    pub scale: [f32; 3],
}
unsafe impl Send for Expert {}

/// Experts (expert, token pairs) per job: a window of tokens times their selections.
pub const MAX_EXPERTS: usize = 1280;

struct Job {
    fmt: ExpertFmt,
    n_embd: usize,
    ff: usize,
    gate_row_bytes: usize,
    down_row_bytes: usize,
    /// Sorted by weights, so the pairs of one expert are adjacent; `groups` are their ranges.
    experts: [Expert; MAX_EXPERTS],
    n_exp: usize,
    groups: Vec<(usize, usize)>,
    xq: Vec<Vec<Q8K>>,
    gu: Vec<f32>,
    hq: Vec<Vec<Q80>>,
    /// The hidden activations as Q8_K, for down formats with 256-element blocks.
    hqk: Vec<Vec<Q8K>>,
    y: *mut f32,
}

struct Shared {
    job: UnsafeCell<Job>,
    /// Incremented by the caller to post a job; workers run the job whose number they have not yet seen.
    seq: AtomicU64,
    /// Arrivals at each barrier, never reset: job number s is past barrier k once bar[k] reaches s * n_threads.
    /// Resetting them would race with a worker still reading the previous job's count.
    bar: [AtomicU64; 3],
    n_threads: usize,
    stop: AtomicU32,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn wait_barrier(b: &AtomicU64, target: u64) {
    b.fetch_add(1, Ordering::AcqRel);
    while b.load(Ordering::Acquire) < target {
        std::hint::spin_loop();
    }
}

fn split(n: usize, t: usize, nt: usize) -> std::ops::Range<usize> {
    (n * t / nt)..(n * (t + 1) / nt)
}

impl Shared {
    /// Worker `t`'s share of the posted job.
    unsafe fn run(&self, t: usize, seq: u64) {
        let job = &mut *self.job.get();
        let nt = self.n_threads;
        let target = seq * nt as u64;
        let (ff, n) = (job.ff, job.n_embd);
        // 1. gate and up rows: index r over (distinct expert, row); each row is read once for all its tokens
        let ng = job.groups.len();
        for r in split(ng * 2 * ff, t, nt) {
            let (gi, rr) = (r / (2 * ff), r % (2 * ff));
            let (a, b) = job.groups[gi];
            let e = job.experts[a];
            let row = if rr < ff { e.gate.add(rr * job.gate_row_bytes) } else { e.up.add((rr - ff) * job.gate_row_bytes) };
            let row = std::slice::from_raw_parts(row, job.gate_row_bytes);
            for j in a..b {
                let xq = &job.xq[job.experts[j].tok];
                let v = match job.fmt.gate {
                    WType::Iq2S => avx512::iq2s_q8k_gather(row, xq),
                    WType::Iq3S => avx512::iq3s_q8k(row, xq),
                    WType::Nvfp4 => avx512::nvfp4_q8k(row, xq) * e.scale[if rr < ff { 0 } else { 1 }],
                    WType::Iq2Xxs => avx512::iq2xxs_q8k(row, xq),
                    WType::Iq2Xs => avx512::iq2xs_q8k(row, xq),
                    WType::Iq3Xxs => avx512::iq3xxs_q8k(row, xq),
                    WType::Iq1S => avx512::iq1s_q8k(row, xq),
                    WType::Q2K => avx512::q2k_q8k(row, xq),
                    WType::Iq4Nl => panic!("IQ4_NL gate rows are not supported"),
                };
                *job.gu.as_ptr().add(j * 2 * ff + rr).cast_mut() = v;
            }
        }
        wait_barrier(&self.bar[0], target);
        // 2. hidden activations, one expert per thread
        let mut j = t;
        while j < job.n_exp {
            let gu = &job.gu[j * 2 * ff..(j + 1) * 2 * ff];
            let c = job.fmt.clamp;
            let h: Vec<f32> = if c > 0.0 {
                (0..ff).map(|i| crate::math::qx_silu_mul(gu[i].min(c), gu[ff + i].clamp(-c, c))).collect()
            } else {
                (0..ff).map(|i| crate::math::qx_silu_mul(gu[i], gu[ff + i])).collect()
            };
            if job.fmt.down.q8k_activation() {
                quantize_q8_k(&h, &mut *(job.hqk.as_ptr().add(j).cast_mut()));
            } else {
                quantize_q8_0(&h, &mut *(job.hq.as_ptr().add(j).cast_mut()));
            }
            j += nt;
        }
        wait_barrier(&self.bar[1], target);
        // 3. down rows
        for r in split(n, t, nt) {
            for &(a, b) in &job.groups {
                let row = std::slice::from_raw_parts(job.experts[a].down.add(r * job.down_row_bytes), job.down_row_bytes);
                for j in a..b {
                    let v = match job.fmt.down {
                        WType::Nvfp4 => avx512::nvfp4_q80(row, &job.hq[j]) * job.experts[a].scale[2],
                        WType::Iq4Nl => avx512::iq4nl_q80(row, &job.hq[j]),
                        WType::Iq2S => avx512::iq2s_q8k_gather(row, &job.hqk[j]),
                        WType::Iq3S => avx512::iq3s_q8k(row, &job.hqk[j]),
                        WType::Iq2Xxs => avx512::iq2xxs_q8k(row, &job.hqk[j]),
                        WType::Iq2Xs => avx512::iq2xs_q8k(row, &job.hqk[j]),
                        WType::Iq3Xxs => avx512::iq3xxs_q8k(row, &job.hqk[j]),
                        WType::Iq1S => avx512::iq1s_q8k(row, &job.hqk[j]),
                        WType::Q2K => avx512::q2k_q8k(row, &job.hqk[j]),
                    };
                    *job.y.add(job.experts[j].out * n + r) = v;
                }
            }
        }
        wait_barrier(&self.bar[2], target);
    }
}

impl Pool {
    /// `cpus`: the cores to run on; the caller becomes worker 0 and is pinned to `cpus[0]`.
    pub fn new(cpus: &[usize]) -> Self {
        let n_threads = cpus.len();
        let empty = Expert { gate: std::ptr::null(), up: std::ptr::null(), down: std::ptr::null(), out: 0, tok: 0, scale: [1.0; 3] };
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(Job {
                fmt: ExpertFmt::IQ2S,
                n_embd: 0,
                ff: 0,
                gate_row_bytes: 0,
                down_row_bytes: 0,
                experts: [empty; MAX_EXPERTS],
                n_exp: 0,
                groups: Vec::new(),
                xq: Vec::new(),
                gu: Vec::new(),
                hq: vec![Vec::new(); MAX_EXPERTS],
                hqk: vec![Vec::new(); MAX_EXPERTS],
                y: std::ptr::null_mut(),
            }),
            seq: AtomicU64::new(0),
            bar: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            n_threads,
            stop: AtomicU32::new(0),
        });
        pin(cpus[0]);
        let mut threads = Vec::new();
        for (t, &cpu) in cpus.iter().enumerate().skip(1) {
            let sh = shared.clone();
            threads.push(std::thread::spawn(move || {
                pin(cpu);
                let mut seen = 0u64;
                let mut idle = 0u32;
                loop {
                    let s = sh.seq.load(Ordering::Acquire);
                    if s != seen {
                        seen = s;
                        idle = 0;
                        unsafe { sh.run(t, s) };
                        continue;
                    }
                    if sh.stop.load(Ordering::Relaxed) != 0 {
                        return;
                    }
                    idle += 1;
                    if idle < 200_000 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                }
            }));
        }
        Self { shared, threads }
    }

    pub fn threads(&self) -> usize {
        self.shared.n_threads
    }

    /// Row `e.out` of `y` (n_embd wide) = down_e(silu(gate_e x) * up_e x) for each expert e, for one token.
    /// Blocks until done.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, fmt: ExpertFmt, x: &[f32], ff: usize, gate_row_bytes: usize, down_row_bytes: usize, experts: &[Expert], y: &mut [f32]) {
        self.run_multi(fmt, &[x], ff, gate_row_bytes, down_row_bytes, experts, y);
    }

    /// `run` for several tokens at once: expert `e` reads input `xs[e.tok]`. Each weight row is read from memory
    /// once for all the tokens its expert serves; each (expert, token) result is the one-token result.
    #[allow(clippy::too_many_arguments)]
    pub fn run_multi(&self, fmt: ExpertFmt, xs: &[&[f32]], ff: usize, gate_row_bytes: usize, down_row_bytes: usize, experts: &[Expert], y: &mut [f32]) {
        assert!(experts.len() <= MAX_EXPERTS);
        assert!(!fmt.down.q8k_activation() || ff % 256 == 0, "Q8_K hidden activations need ff to be a multiple of 256");
        if experts.is_empty() {
            return;
        }
        let sh = &*self.shared;
        unsafe {
            let job = &mut *sh.job.get();
            job.fmt = fmt;
            job.n_embd = xs[0].len();
            job.ff = ff;
            job.gate_row_bytes = gate_row_bytes;
            job.down_row_bytes = down_row_bytes;
            let mut sorted = experts.to_vec();
            sorted.sort_by_key(|e| (e.gate as usize, e.tok));
            job.experts[..sorted.len()].copy_from_slice(&sorted);
            job.n_exp = sorted.len();
            job.groups.clear();
            let mut a = 0;
            for j in 1..=sorted.len() {
                if j == sorted.len() || sorted[j].gate != sorted[a].gate {
                    job.groups.push((a, j));
                    a = j;
                }
            }
            job.xq.resize(xs.len(), Vec::new());
            for (q, x) in job.xq.iter_mut().zip(xs) {
                quantize_q8_k(x, q);
            }
            job.gu.resize(sorted.len() * 2 * ff, 0.0);
            job.y = y.as_mut_ptr();
        }
        let s = sh.seq.fetch_add(1, Ordering::AcqRel) + 1;
        unsafe { sh.run(0, s) };
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(1, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dot::{iq2xxs_q8k_scalar, iq3xxs_q8k_scalar};

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn blocks(n: usize, bb: usize, seed: &mut u64) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n * bb).map(|_| rng(seed) as u8).collect();
        for b in v.chunks_exact_mut(bb) {
            let d = half::f16::from_f32(((rng(seed) % 1000) as f32 + 1.0) * 1e-4);
            b[..2].copy_from_slice(&d.to_le_bytes());
        }
        v
    }

    /// DeepSeek-style experts (IQ2_XXS gate and up, IQ3_XXS down, clamped SwiGLU, Q8_K hidden activations):
    /// the pool's output for several tokens equals the scalar dots applied token by token.
    #[test]
    fn pool_deepseek_formats_exact() {
        let (n, ff, ne, nt) = (512usize, 256usize, 5usize, 3usize);
        let (gb, db) = (66 * n / 256, 98 * ff / 256);
        let mut s = 0x1111_2222_3333_4444u64;
        let recs: Vec<Vec<u8>> = (0..ne).map(|_| [blocks(ff * n / 256, 66, &mut s), blocks(ff * n / 256, 66, &mut s), blocks(n * ff / 256, 98, &mut s)].concat()).collect();
        let xs: Vec<Vec<f32>> = (0..nt).map(|_| (0..n).map(|_| ((rng(&mut s) % 20001) as f32 - 10000.0) * 1e-3).collect()).collect();
        let fmt = ExpertFmt { gate: WType::Iq2Xxs, down: WType::Iq3Xxs, clamp: 10.0 };
        // every expert for every token, in a scrambled order
        let mut experts = Vec::new();
        for t in 0..nt {
            for e in (0..ne).rev() {
                let r = recs[e].as_ptr();
                experts.push(Expert { gate: r, up: unsafe { r.add(ff * gb) }, down: unsafe { r.add(2 * ff * gb) }, out: t * ne + e, tok: t, scale: [1.0; 3] });
            }
        }
        let pool = Pool::new(&[0, 1, 2, 3]);
        let mut y = vec![0f32; nt * ne * n];
        let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
        pool.run_multi(fmt, &xr, ff, gb, db, &experts, &mut y);
        for t in 0..nt {
            let mut xq = Vec::new();
            quantize_q8_k(&xs[t], &mut xq);
            for e in 0..ne {
                let r = &recs[e];
                let h: Vec<f32> = (0..ff)
                    .map(|i| {
                        let g = iq2xxs_q8k_scalar(&r[i * gb..(i + 1) * gb], &xq);
                        let u = iq2xxs_q8k_scalar(&r[ff * gb + i * gb..ff * gb + (i + 1) * gb], &xq);
                        crate::math::qx_silu_mul(g.min(10.0), u.clamp(-10.0, 10.0))
                    })
                    .collect();
                let mut hq = Vec::new();
                quantize_q8_k(&h, &mut hq);
                for row in 0..n {
                    let want = iq3xxs_q8k_scalar(&r[2 * ff * gb + row * db..2 * ff * gb + (row + 1) * db], &hq);
                    let got = y[(t * ne + e) * n + row];
                    assert!(want.to_bits() == got.to_bits(), "token {t} expert {e} row {row}: {got} vs {want}");
                }
            }
        }
    }
}
