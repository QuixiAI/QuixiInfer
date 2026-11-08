//! `qx-poolbw`: the CPU expert pool's throughput on DeepSeek-V4.1-Flash expert records (E 5120, ff 2304) for the
//! pack's three main format mixes, at the batch sizes one decode layer hands it (1, 2, 6 experts) and at 64 for
//! the streaming rate. Synthetic records in huge pages, far past the caches. `QX_DS_CPUS` picks the cores.
use qx_cpu::pool::{Expert, ExpertFmt, Pool, WType};

const E: usize = 5120;
const FF: usize = 2304;

fn huge(bytes: usize) -> *mut u8 {
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
        assert!(p != libc::MAP_FAILED);
        // QX_POOLBW_HUGE=0: 4 KiB pages (as a pinned tier has), else transparent huge pages
        if !std::env::var("QX_POOLBW_HUGE").is_ok_and(|v| v == "0") {
            libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
        }
        p as *mut u8
    }
}

fn row_bytes(t: WType, cols: usize) -> usize {
    let per_block = match t {
        WType::Iq1S => 50,
        WType::Iq2Xxs => 66,
        WType::Iq2Xs => 74,
        WType::Iq2S => 82,
        WType::Iq3Xxs => 98,
        WType::Q2K => 84,
        WType::Iq3S => 110,
        WType::Iq4Nl | WType::Nvfp4 => unreachable!(),
    };
    cols / 256 * per_block
}

fn main() {
    let cpus = qx_engine::dsv41::pool_cpus();
    let mixes: [(&str, WType, WType); 3] = [("IQ2_XXS gate/up, IQ2_S down (10 layers)", WType::Iq2Xxs, WType::Iq2S), ("IQ2_S gate/up, IQ3_XXS down (10 layers)", WType::Iq2S, WType::Iq3Xxs), ("IQ1_S gate/up, IQ2_XS down (6 layers)", WType::Iq1S, WType::Iq2Xs)];
    let n_rec = 1024;
    let x: Vec<f32> = (0..E).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut y = vec![0f32; 320 * E];
    let pool = Pool::new(&cpus);
    println!("pool on {} cores", cpus.len());
    for (name, g, d) in mixes {
        let (gr, dr) = (row_bytes(g, E), row_bytes(d, FF));
        let (gb, db) = (FF * gr, E * dr);
        let rec = (2 * gb + db).next_multiple_of(4096);
        let bytes = n_rec * rec;
        let host = huge(bytes);
        let buf = unsafe { std::slice::from_raw_parts_mut(host, bytes) };
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2654435761).to_le_bytes()[3];
        }
        let fmt = ExpertFmt { gate: g, down: d, clamp: 10.0 };
        println!("{name}: {:.2} MiB per expert", (2 * gb + db) as f64 / (1 << 20) as f64);
        for batch in [1usize, 2, 6, 64] {
            let mut next = 0usize;
            let t0 = std::time::Instant::now();
            let mut n = 0usize;
            let mut calls = 0usize;
            while t0.elapsed().as_secs_f64() < 2.0 {
                let ex: Vec<Expert> = (0..batch)
                    .map(|i| {
                        let r = unsafe { host.add(((next + i) % n_rec) * rec) };
                        Expert { gate: r, up: unsafe { r.add(gb) }, down: unsafe { r.add(2 * gb) }, out: i, tok: 0, scale: [1.0; 3] }
                    })
                    .collect();
                next = (next + batch) % n_rec;
                pool.run_multi(fmt, &[&x], FF, gr, dr, &ex, &mut y);
                n += batch;
                calls += 1;
            }
            let secs = t0.elapsed().as_secs_f64();
            println!("  batch {batch:>2}: {:6.2} ms per call, {:5.3} ms per expert, {:5.1} GB/s", secs * 1e3 / calls as f64, secs * 1e3 / n as f64, (n * (2 * gb + db)) as f64 / secs / 1e9);
        }
        unsafe { libc::munmap(host as *mut _, bytes) };
    }
}
