//! Phase 0 hardware probes: PCIe copy bandwidth and host memory read bandwidth.
//!
//! `qx-probe pcie` times pinned host<->device copies at transfer sizes the engine uses.
//! `qx-probe ram` times multi-threaded sequential reads of a large buffer, per CCD and for both.
use std::time::Instant;

fn pcie() -> anyhow::Result<()> {
    use cudarc::driver::{result, sys};
    let ctx = cudarc::driver::CudaContext::new(0)?;
    ctx.bind_to_thread()?;
    let max = 512usize << 20;
    unsafe {
        let host = result::malloc_host(max, sys::CU_MEMHOSTALLOC_PORTABLE)? as *mut u8;
        std::ptr::write_bytes(host, 1, max);
        let dev = result::malloc_sync(max)?;
        let stream = result::stream::create(result::stream::StreamKind::NonBlocking)?;
        println!("| Transfer | H2D GB/s | D2H GB/s |\n|---|---|---|");
        for size in [64usize << 10, 2 << 20, 8 << 20, 64 << 20, 512 << 20] {
            let reps = (2usize << 30) / size;
            let mut rates = [0f64; 2];
            for (d, rate) in rates.iter_mut().enumerate() {
                let run = |n: usize| {
                    for _ in 0..n {
                        if d == 0 {
                            sys::cuMemcpyHtoDAsync_v2(dev, host as *const _, size, stream).result().unwrap();
                        } else {
                            sys::cuMemcpyDtoHAsync_v2(host as *mut _, dev, size, stream).result().unwrap();
                        }
                    }
                    result::stream::synchronize(stream).unwrap();
                };
                run(reps.min(8));
                let t = Instant::now();
                run(reps);
                *rate = (reps * size) as f64 / t.elapsed().as_secs_f64() / 1e9;
            }
            println!("| {} KiB | {:.1} | {:.1} |", size >> 10, rates[0], rates[1]);
        }
        result::free_sync(dev)?;
        result::free_host(host as *mut _)?;
    }
    Ok(())
}

fn pin_to(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn sum_avx512(p: *const u8, len: usize) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = [_mm512_setzero_si512(); 4];
    let mut i = 0;
    while i + 256 <= len {
        for (k, a) in acc.iter_mut().enumerate() {
            let v = unsafe { _mm512_load_si512(p.add(i + 64 * k) as *const _) };
            *a = _mm512_add_epi64(*a, v);
        }
        i += 256;
    }
    let s = _mm512_add_epi64(_mm512_add_epi64(acc[0], acc[1]), _mm512_add_epi64(acc[2], acc[3]));
    let mut total = _mm512_reduce_add_epi64(s) as u64;
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` keeps the read inside the span; the load is unaligned.
        total = total.wrapping_add(unsafe { (p.add(i) as *const u64).read_unaligned() });
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len` keeps the byte read inside the span.
        total = total.wrapping_add(unsafe { *p.add(i) } as u64);
        i += 1;
    }
    total
}

/// The same sum with 32-byte registers, for a CPU without AVX-512.
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(p: *const u8, len: usize) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut i = 0;
    while i + 128 <= len {
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: `i + 128 <= len` keeps each 32-byte load inside the span; loads are unaligned.
            let v = unsafe { _mm256_loadu_si256(p.add(i + 32 * k) as *const __m256i) };
            *a = _mm256_add_epi64(*a, v);
        }
        i += 128;
    }
    let mut t = [0u64; 16];
    for (k, a) in acc.iter().enumerate() {
        // SAFETY: `k < 4`, so `4 * k..4 * k + 4` stays inside `t`.
        unsafe { _mm256_storeu_si256(t.as_mut_ptr().add(4 * k) as *mut __m256i, *a) };
    }
    let mut total = t.iter().fold(0u64, |x, &y| x.wrapping_add(y));
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` keeps the read inside the span; the load is unaligned.
        total = total.wrapping_add(unsafe { (p.add(i) as *const u64).read_unaligned() });
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len` keeps the byte read inside the span.
        total = total.wrapping_add(unsafe { *p.add(i) } as u64);
        i += 1;
    }
    total
}

/// The same sum without vector code, for very old CPUs.
unsafe fn sum_plain(p: *const u8, len: usize) -> u64 {
    let mut total = 0u64;
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` keeps the read inside the span; the load is unaligned.
        total = total.wrapping_add(unsafe { (p.add(i) as *const u64).read_unaligned() });
        i += 8;
    }
    while i < len {
        // SAFETY: `i < len` keeps the byte read inside the span.
        total = total.wrapping_add(unsafe { *p.add(i) } as u64);
        i += 1;
    }
    total
}

/// The best sum the CPU can do, so the probe does not SIGILL on a machine without AVX-512.
unsafe fn sum(p: *const u8, len: usize) -> u64 {
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: detected above; `p[..len]` is the caller's contract.
        unsafe { sum_avx512(p, len) }
    } else if is_x86_feature_detected!("avx2") {
        // SAFETY: detected above; `p[..len]` is the caller's contract.
        unsafe { sum_avx2(p, len) }
    } else {
        // SAFETY: `p[..len]` is the caller's contract.
        unsafe { sum_plain(p, len) }
    }
}

fn ram() -> anyhow::Result<()> {
    let total = 8usize << 30;
    let layout = std::alloc::Layout::from_size_align(total, 2 << 20)?;
    let buf = unsafe { std::alloc::alloc(layout) };
    unsafe { libc::madvise(buf as *mut _, total, libc::MADV_HUGEPAGE) };
    // Touch from many threads so pages land spread across the memory controllers.
    std::thread::scope(|s| {
        for t in 0..16 {
            let p = buf as usize;
            s.spawn(move || unsafe { std::ptr::write_bytes((p + t * (total / 16)) as *mut u8, 1, total / 16) });
        }
    });
    let ccd0: Vec<usize> = (0..8).chain(16..24).collect();
    let ccd1: Vec<usize> = (8..16).chain(24..32).collect();
    let cases: Vec<(&str, Vec<usize>)> = vec![
        ("1 thread", vec![0]),
        ("CCD0, 8 cores", (0..8).collect()),
        ("CCD0, 16 threads", ccd0.clone()),
        ("CCD1, 8 cores", (8..16).collect()),
        ("both CCDs, 16 cores", (0..16).collect()),
        ("both CCDs, 32 threads", (0..32).collect()),
    ];
    println!("| Threads | Read GB/s |\n|---|---|");
    for (name, cpus) in cases {
        let n = cpus.len();
        let chunk = total / n / 256 * 256;
        let mut best = 0f64;
        for _ in 0..3 {
            let t = Instant::now();
            std::thread::scope(|s| {
                for (i, &cpu) in cpus.iter().enumerate() {
                    let p = buf as usize + i * chunk;
                    s.spawn(move || {
                        pin_to(cpu);
                        std::hint::black_box(unsafe { sum(p as *const u8, chunk) });
                    });
                }
            });
            best = best.max((chunk * n) as f64 / t.elapsed().as_secs_f64() / 1e9);
        }
        println!("| {name} | {best:.1} |");
    }
    unsafe { std::alloc::dealloc(buf, layout) };
    Ok(())
}

fn main() -> anyhow::Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("pcie") => pcie(),
        Some("ram") => ram(),
        _ => {
            eprintln!("usage: qx-probe pcie|ram");
            std::process::exit(2)
        }
    }
}
