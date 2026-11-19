//! Weight bytes per second of the expert dots: the scalar reference against the AVX2 kernels (and AVX-512 when the
//! CPU has it), over 256 MiB of rows (out of cache), 1 and up to 16 threads. Prints the detected kernel set first.

#[cfg(target_arch = "x86_64")]
mod bench {
    use qx_cpu::act::{quantize_q8_0, quantize_q8_k, Q80, Q8K};
    use qx_cpu::dot::{avx2, avx512, isa, Isa};
    use std::time::Instant;

    fn pin(cpu: usize) {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }

    /// `f` over every row of `buf` split across `nt` threads, pinned to cpu 0..nt; GB/s of weight bytes.
    fn bench<A: Sync>(buf: &[u8], rb: usize, act: &[A], f: unsafe fn(&[u8], &[A]) -> f32, nt: usize) -> f64 {
        let rows = buf.len() / rb;
        let per = rows / nt;
        let rem = rows % nt;
        let t = Instant::now();
        std::thread::scope(|sc| {
            for tid in 0..nt {
                sc.spawn(move || {
                    pin(tid);
                    let start = tid * per + tid.min(rem);
                    let end = start + per + usize::from(tid < rem);
                    let mut acc = 0f32;
                    for r in start..end {
                        acc += unsafe { f(&buf[r * rb..(r + 1) * rb], act) };
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        (rows * rb) as f64 / t.elapsed().as_secs_f64() / 1e9
    }

    pub fn run() {
        let total = 1usize << 28;
        let mut buf = vec![0u8; total];
        let mut s = 0x1234_5678u64;
        for b in buf.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = s as u8;
        }
        let x: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.37).sin()).collect();
        let (mut qk, mut q0) = (Vec::new(), Vec::new());
        quantize_q8_k(&x, &mut qk);
        quantize_q8_0(&x[..640], &mut q0);

        let avx512_on = isa() == Isa::Avx512;
        let cpu_has_avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl");
        let note = match (avx512_on, cpu_has_avx512) {
            (true, _) => "",
            (false, true) => " (AVX-512 not selected)",
            (false, false) => " (no AVX-512 on this CPU)",
        };
        println!("kernel set: {:?}{}", isa(), note);
        println!("| format | kernel | threads | GB/s |\n|---|---|---|---|");

        let max = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut nts = vec![1usize];
        let cap = 16.min(max);
        if cap > 1 {
            nts.push(cap);
        }

        type K = unsafe fn(&[u8], &[Q8K]) -> f32;
        let qk_cases: [(&str, usize, K, K, Option<K>); 8] = [
            ("IQ2_S", 82 * 10, qx_cpu::dot::iq2s_q8k_scalar, avx2::iq2s_q8k, Some(avx512::iq2s_q8k_gather)),
            ("IQ3_S", 110 * 10, qx_cpu::dot::iq3s_q8k_scalar, avx2::iq3s_q8k, Some(avx512::iq3s_q8k)),
            ("NVFP4", 144 * 10, qx_cpu::dot::nvfp4_q8k_scalar, avx2::nvfp4_q8k, Some(avx512::nvfp4_q8k)),
            ("IQ2_XXS", 66 * 10, qx_cpu::dot::iq2xxs_q8k_scalar, avx2::iq2xxs_q8k, Some(avx512::iq2xxs_q8k)),
            ("IQ2_XS", 74 * 10, qx_cpu::dot::iq2xs_q8k_scalar, avx2::iq2xs_q8k, Some(avx512::iq2xs_q8k)),
            ("IQ3_XXS", 98 * 10, qx_cpu::dot::iq3xxs_q8k_scalar, avx2::iq3xxs_q8k, Some(avx512::iq3xxs_q8k)),
            ("IQ1_S", 50 * 10, qx_cpu::dot::iq1s_q8k_scalar, avx2::iq1s_q8k, Some(avx512::iq1s_q8k)),
            ("Q2_K", 84 * 10, qx_cpu::dot::q2k_q8k_scalar, avx2::q2k_q8k, Some(avx512::q2k_q8k)),
        ];
        for (name, rb, scalar, a2, a5) in qk_cases {
            for &nt in &nts {
                println!("| {name} | scalar | {nt} | {:.1} |", bench(&buf, rb, &qk, scalar, nt));
                println!("| {name} | avx2 | {nt} | {:.1} |", bench(&buf, rb, &qk, a2, nt));
                if let (true, Some(k)) = (avx512_on, a5) {
                    println!("| {name} | avx512 | {nt} | {:.1} |", bench(&buf, rb, &qk, k, nt));
                }
            }
        }

        type O = unsafe fn(&[u8], &[Q80]) -> f32;
        let q80_cases: [(&str, usize, O, O, Option<O>); 2] = [
            ("NVFP4 q80", 36 * 10, qx_cpu::dot::nvfp4_q80_scalar, avx2::nvfp4_q80, Some(avx512::nvfp4_q80)),
            ("IQ4_NL", 18 * 20, qx_cpu::dot::iq4nl_q80_scalar, avx2::iq4nl_q80, Some(avx512::iq4nl_q80)),
        ];
        for (name, rb, scalar, a2, a5) in q80_cases {
            for &nt in &nts {
                println!("| {name} | scalar | {nt} | {:.1} |", bench(&buf, rb, &q0, scalar, nt));
                println!("| {name} | avx2 | {nt} | {:.1} |", bench(&buf, rb, &q0, a2, nt));
                if let (true, Some(k)) = (avx512_on, a5) {
                    println!("| {name} | avx512 | {nt} | {:.1} |", bench(&buf, rb, &q0, k, nt));
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    bench::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("dotbench: the CPU kernels are x86-64 only, nothing to measure");
}
