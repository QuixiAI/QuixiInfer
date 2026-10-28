//! `qx-rambw`: how close the CPU expert pool gets to the RAM's read bandwidth, and which thread layout gets it
//! there. Allocates the records the way `host_experts` does (anonymous memory with MADV_HUGEPAGE), then measures
//! a plain AVX-512 streaming read and the pool's NVFP4 decode job for several CPU sets.
use qx_cpu::pool::{Expert, GateFmt, Pool};
use std::arch::x86_64::*;

const E: usize = 2560;
const FF: usize = 640;

fn huge(bytes: usize) -> *mut u8 {
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
        assert!(p != libc::MAP_FAILED);
        libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
        p as *mut u8
    }
}

fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn read_span(p: *const u8, n: usize, prefetch: usize) -> u64 {
    let mut a = _mm512_setzero_si512();
    let mut b = _mm512_setzero_si512();
    let mut o = 0;
    while o < n {
        if prefetch > 0 {
            _mm_prefetch::<_MM_HINT_T0>(p.add(o + prefetch) as *const i8);
        }
        a = _mm512_xor_si512(a, _mm512_load_si512(p.add(o) as *const _));
        b = _mm512_xor_si512(b, _mm512_load_si512(p.add(o + 64) as *const _));
        o += 128;
    }
    _mm512_reduce_add_epi64(_mm512_add_epi64(a, b)) as u64
}

/// Threads in `cpus` each stream their share of `bytes`, 2 MiB pieces handed out round-robin; GB/s.
fn raw(base: *const u8, bytes: usize, cpus: &[usize], prefetch: usize) -> f64 {
    let piece = 2 << 20;
    let pieces = bytes / piece;
    let t0 = std::time::Instant::now();
    let mut total = 0usize;
    for _ in 0..3 {
        std::thread::scope(|s| {
            for (t, &c) in cpus.iter().enumerate() {
                let base = base as usize;
                let nt = cpus.len();
                s.spawn(move || {
                    pin(c);
                    let mut x = 0u64;
                    let mut i = t;
                    while i < pieces {
                        x ^= unsafe { read_span((base + i * piece) as *const u8, piece - if prefetch > 0 { 1024 } else { 0 }, prefetch) };
                        i += nt;
                    }
                    std::hint::black_box(x);
                });
            }
        });
        total += pieces * piece;
    }
    total as f64 / t0.elapsed().as_secs_f64() / 1e9
}

fn main() {
    let gate_row = E / 64 * 36;
    let down_row = FF / 64 * 36;
    let (gb, db) = (FF * gate_row, E * down_row);
    let rec = (2 * gb + db + 16).next_multiple_of(4096);
    let n_rec = 2400; // 6.6 GB: far past every cache
    let bytes = n_rec * rec;
    let host = huge(bytes);
    let buf = unsafe { std::slice::from_raw_parts_mut(host, bytes) };
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i as u32).wrapping_mul(2654435761).to_le_bytes()[3] & 0x77;
    }
    let ahp = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    let ahp = ahp.lines().find(|l| l.starts_with("AnonHugePages")).unwrap_or("?").to_string();
    println!("{:.1} GB of records; {ahp}", bytes as f64 / 1e9);
    let sets: Vec<(&str, Vec<usize>)> = vec![
        ("CCD0 8 cores", (0..8).collect()),
        ("CCD1 8 cores", (8..16).collect()),
        ("16 cores", (0..16).collect()),
        ("16 cores + SMT (32)", (0..32).collect()),
    ];
    for (name, cpus) in &sets {
        println!("raw read  {name:<22} {:5.1} GB/s   prefetch 512 B ahead {:5.1} GB/s", raw(host, bytes, cpus, 0), raw(host, bytes, cpus, 512));
    }
    let x: Vec<f32> = (0..E).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut y = vec![0f32; 320 * E];
    for (name, cpus) in &sets {
        let pool = Pool::new(cpus);
        let mut next = 0usize;
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        while t0.elapsed().as_secs_f64() < 3.0 {
            let ex: Vec<Expert> = (0..64)
                .map(|i| {
                    let r = unsafe { host.add(((next + i) % n_rec) * rec) };
                    Expert { gate: r, up: unsafe { r.add(gb) }, down: unsafe { r.add(2 * gb) }, out: i, tok: 0, scale: [1.0; 3] }
                })
                .collect();
            next = (next + 64) % n_rec;
            pool.run_multi(GateFmt::Nvfp4, &[&x], FF, gate_row, down_row, &ex, &mut y);
            n += 64;
        }
        println!("pool      {name:<22} {:5.1} GB/s", (n * (2 * gb + db)) as f64 / t0.elapsed().as_secs_f64() / 1e9);
    }
}
