//! `qx-membw`: the CPU expert pool's throughput on NVFP4 records alone, and while the GPU's copy engine reads
//! pinned host memory over PCIe at the same time: whether the two share the RAM's bandwidth or add up.
use qx_cpu::pool::{Expert, GateFmt, Pool};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const E: usize = 2560;
const FF: usize = 640;

fn main() {
    qx_gpu::init();
    let gate_row = E / 64 * 36;
    let down_row = FF / 64 * 36;
    let (gb, db) = (FF * gate_row, E * down_row);
    let rec = (2 * gb + db + 16).next_multiple_of(4096);
    let n_rec = 2400; // 6.6 GB: far past every cache
    let host = qx_gpu::mem::HostMapped::new(n_rec * rec);
    let bytes = host.slice::<u8>(0, n_rec * rec);
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i as u32).wrapping_mul(2654435761).to_le_bytes()[3] & 0x77;
    }
    let cpus: Vec<usize> = (0..16).collect();
    let pool = Pool::new(&cpus);
    let x: Vec<f32> = (0..E).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut y = vec![0f32; 320 * E];
    let mut next = 0usize;
    let mut run_pool = |secs: f64| -> f64 {
        let t0 = std::time::Instant::now();
        let mut n = 0usize;
        while t0.elapsed().as_secs_f64() < secs {
            // a decode job: 64 distinct experts for one token
            let ex: Vec<Expert> = (0..64)
                .map(|i| {
                    let r = unsafe { host.ptr.add(((next + i) % n_rec) * rec) };
                    Expert { gate: r, up: unsafe { r.add(gb) }, down: unsafe { r.add(2 * gb) }, out: i, tok: 0, scale: [1.0; 3] }
                })
                .collect();
            next = (next + 64) % n_rec;
            pool.run_multi(GateFmt::Nvfp4, &[&x], FF, gate_row, down_row, &ex, &mut y);
            n += 64;
        }
        (n * (2 * gb + db)) as f64 / t0.elapsed().as_secs_f64() / 1e9
    };
    println!("pool alone: {:.1} GB/s", run_pool(3.0));
    // the copy engine streaming another part of the records into VRAM
    let stop = Arc::new(AtomicBool::new(false));
    let copied = Arc::new(AtomicU64::new(0));
    let (st, cp, src) = (stop.clone(), copied.clone(), host.ptr as usize + (n_rec / 2) * rec);
    let chunk = 64 << 20;
    let th = std::thread::spawn(move || {
        qx_gpu::mem::bind();
        let dst = qx_gpu::mem::DeviceBuf::new(chunk);
        let stream = unsafe { cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::NonBlocking) }.unwrap();
        let mut k = 0usize;
        while !st.load(Ordering::Relaxed) {
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst.ptr, (src + (k % 40) * chunk) as *const _, chunk, stream).result().unwrap();
                cudarc::driver::sys::cuStreamSynchronize(stream).result().unwrap();
            }
            cp.fetch_add(chunk as u64, Ordering::Relaxed);
            k += 1;
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    let c0 = copied.load(Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let pool_bw = run_pool(3.0);
    let dma_bw = (copied.load(Ordering::Relaxed) - c0) as f64 / t0.elapsed().as_secs_f64() / 1e9;
    stop.store(true, Ordering::Relaxed);
    th.join().unwrap();
    println!("pool with DMA beside it: {pool_bw:.1} GB/s, DMA {dma_bw:.1} GB/s, together {:.1} GB/s", pool_bw + dma_bw);
}
