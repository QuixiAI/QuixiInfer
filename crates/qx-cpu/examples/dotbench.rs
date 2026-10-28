//! Weight bytes per second of the AVX-512 expert dots, over 1 GiB of rows (out of cache), 1..N threads.
use qx_cpu::act::{quantize_q8_0, quantize_q8_k};
use qx_cpu::dot::avx512;
use std::time::Instant;

fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn main() {
    let total = 1usize << 30;
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
    type K = unsafe fn(&[u8], &[qx_cpu::act::Q8K]) -> f32;
    let cases: [(&str, usize, Option<K>); 4] = [("IQ2_S", 82 * 10, Some(avx512::iq2s_q8k)), ("IQ2_S gather", 82 * 10, Some(avx512::iq2s_q8k_gather)), ("IQ3_S", 110 * 10, Some(avx512::iq3s_q8k)), ("IQ4_NL", 18 * 20, None)];
    println!("| format | threads | GB/s |\n|---|---|---|");
    for (name, rb, k) in cases {
        let rows = total / rb;
        for nt in [1usize, 8, 16, 32] {
            let t = Instant::now();
            let reps = if nt == 1 { 1 } else { 4 };
            for _ in 0..reps {
                std::thread::scope(|sc| {
                    for tid in 0..nt {
                        let (buf, qk, q0) = (&buf, &qk, &q0);
                        sc.spawn(move || {
                            pin(tid);
                            let mut acc = 0f32;
                            let per = rows / nt;
                            for r in tid * per..(tid + 1) * per {
                                let row = &buf[r * rb..(r + 1) * rb];
                                acc += unsafe { match k { Some(f) => f(row, qk), None => avx512::iq4nl_q80(row, q0) } };
                            }
                            std::hint::black_box(acc);
                        });
                    }
                });
            }
            let gbs = (rows * rb * reps) as f64 / t.elapsed().as_secs_f64() / 1e9;
            println!("| {name} | {nt} | {gbs:.1} |");
        }
    }
}
