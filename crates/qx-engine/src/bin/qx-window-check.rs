//! `qx-window-check TOKENS_FILE N [WIDTHS]`: logits of N forced tokens one at a time, then the same tokens in
//! windows cycling through WIDTHS (default 2,3,4,5,8); every window row must equal its one-token row bit for bit.
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n: usize = a[2].parse::<usize>()?.min(toks.len());
    let widths: Vec<usize> = a.get(3).map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![2, 3, 4, 5, 8]);
    let budget = (std::env::var("QX_CACHE_GIB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(4.0) * (1u64 << 30) as f64) as usize;
    // every other expert resident, so windows mix GPU and CPU experts
    let order: Vec<(usize, usize)> = (0..512).step_by(2).flat_map(|e| (0..48).map(move |l| (l, e))).collect();
    let mut eng = Engine::load(&g, (n + 64).next_multiple_of(256), budget, &order)?;
    let mut one = Vec::with_capacity(n);
    let t0 = std::time::Instant::now();
    for &t in &toks[..n] {
        one.push(eng.forward(t, None));
    }
    let t1 = t0.elapsed().as_secs_f64();
    eng.reset();
    let t0 = std::time::Instant::now();
    let (mut p, mut wi, mut bad, mut windows) = (0usize, 0usize, 0usize, 0usize);
    let mut worst = 0f32;
    while p < n {
        let w = widths[wi % widths.len()].min(n - p);
        wi += 1;
        windows += 1;
        let rows = eng.forward_window(&toks[p..p + w]);
        for (i, r) in rows.iter().enumerate() {
            let o = &one[p + i];
            let d = r.iter().zip(o).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            if r.iter().zip(o).any(|(a, b)| a.to_bits() != b.to_bits()) {
                bad += 1;
                worst = worst.max(d);
                if bad <= 5 {
                    println!("pos {} (window of {w}, row {i}): differs, max |d| {d:.3e}", p + i);
                }
            }
        }
        p += w;
    }
    let t2 = t0.elapsed().as_secs_f64();
    println!("{n} positions in {windows} windows: {bad} rows differ (max |d| {worst:.3e}); one-token {:.1} tok/s, windows {:.1} tok/s (eager, synced)", n as f64 / t1, n as f64 / t2);
    Ok(())
}
