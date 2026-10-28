//! `qx-batch-bench TOKENS_FILE [PROMPT] [STEPS] [BATCHES]`: decode throughput of B sequences together, for each B in
//! BATCHES (default 1,4,8,16): B slots prefilled with different PROMPT-token texts (default 256), then STEPS
//! (default 64) batched steps of one token each. Reports tokens/s, step time, and where the CPU expert pool and
//! the PLE reader spent theirs.
use anyhow::Result;
use qx_engine::engine::{Engine, Sampling};
use qx_pack::gguf::Gguf;
use std::sync::atomic::Ordering;

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = std::env::var("QX_MODEL").unwrap_or_else(|_| qx_engine::default_model_path());
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let plen: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(256);
    let steps: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(64);
    let batches: Vec<usize> = a.get(4).map_or("1,4,8,16", |s| s.as_str()).split(',').map(|x| x.parse().unwrap()).collect();
    let slots = *batches.iter().max().unwrap();
    let order = qx_engine::routing::default_order(&g, 48, 512);
    let ctx: usize = std::env::var("QX_BENCH_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or((plen + steps + 64).next_multiple_of(256));
    let mut eng = Engine::load_slots(&g, ctx, qx_gpu::mem::mem_info().0, &order, slots)?;
    eng.warm_up();
    let sp = Sampling { temperature: 0.8, top_p: 0.95, top_k: 40, seed: 7, ..Default::default() };
    println!("{:>3} {:>9} {:>9} {:>8} {:>10} {:>9} {:>9}", "B", "tok/s", "step ms", "hits %", "misses/st", "cpu ms/st", "ple rows/st");
    for &b in &batches {
        for i in 0..b {
            let text = toks[(i * 4099) % (toks.len() - plen)..][..plen].to_vec();
            eng.begin_seq(i, &text, &sp);
        }
        let segs: Vec<(usize, usize)> = (0..b).map(|i| (i, 1)).collect();
        // two warm steps, then the timed ones
        for _ in 0..2 {
            eng.step_segments(&segs);
        }
        let (h0, r0, _) = eng.cache_stats();
        let (pr0, _, _) = eng.ple_stats();
        let cpu = |e: &Engine| (e.shared().cpu_ns.load(Ordering::Relaxed), e.shared().cpu_misses.load(Ordering::Relaxed));
        let (ns0, m0) = cpu(&eng);
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            eng.step_segments(&segs);
        }
        let dt = t0.elapsed().as_secs_f64();
        let (h1, r1, _) = eng.cache_stats();
        let (pr1, _, _) = eng.ple_stats();
        let (ns1, m1) = cpu(&eng);
        let st = steps as f64;
        if std::env::var("QX_TABLE_CHECK").is_ok() {
            let t: Vec<u64> = eng.cache.table.download(48 * 512);
            let nz: Vec<u64> = t.iter().copied().filter(|&v| v != 0).collect();
            let mut u = nz.clone();
            u.sort_unstable();
            u.dedup();
            let (lo, hi) = (eng.cache.arena.ptr, eng.cache.arena.ptr + eng.cache.arena.bytes as u64);
            let outside = nz.iter().filter(|&&v| v < lo || v >= hi).count();
            eprintln!("table: {} resident, {} distinct slots, {} outside the arena (cache built with {})", nz.len(), u.len(), outside, eng.cache.resident);
        }
        println!("{b:>3} {:>9.1} {:>9.2} {:>8.1} {:>10.1} {:>9.2} {:>9.1}", (b * steps) as f64 / dt, 1e3 * dt / st, 100.0 * (h1 - h0) as f64 / (r1 - r0).max(1) as f64, (m1 - m0) as f64 / st, (ns1 - ns0) as f64 * 1e-6 / st, (pr1 - pr0) as f64 / st);
    }
    Ok(())
}
