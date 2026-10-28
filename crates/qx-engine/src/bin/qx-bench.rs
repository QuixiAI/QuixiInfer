//! `qx-bench PROMPT_TOKENS N_NEW [RANKING]`: free-running greedy generation; reports decode tok/s and the
//! VRAM expert cache's hit rate. The cache takes all free VRAM but a margin, filled in RANKING order
//! (`layer expert count` lines from qx-profile), or expert order without one.
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let prompt: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_new: usize = a[2].parse()?;
    let order: Vec<(usize, usize)> = match a.get(3) {
        Some(p) => std::fs::read_to_string(p)?.lines().map(|l| {
            let v: Vec<usize> = l.split_whitespace().take(2).map(|x| x.parse().unwrap()).collect();
            (v[0], v[1])
        }).collect(),
        None => qx_engine::routing::default_order(&g, 48, 512),
    };
    let margin_gib: f64 = std::env::var("QX_VRAM_MARGIN_GIB").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    // Dense weights and state take about 4.6 GiB; the cache gets what is left once they load.
    let (free, total) = qx_gpu::mem::mem_info();
    let dense_est = (4.9 * (1u64 << 30) as f64) as usize;
    let budget = free.saturating_sub(dense_est + (margin_gib * (1u64 << 30) as f64) as usize);
    eprintln!("VRAM free {:.1} / {:.1} GiB, expert cache budget {:.1} GiB", free as f64 / 1073741824.0, total as f64 / 1073741824.0, budget as f64 / 1073741824.0);
    let max_ctx = (prompt.len() + n_new + 64).next_multiple_of(256);
    let t_load = std::time::Instant::now();
    let mut eng = Engine::load(&g, max_ctx, budget, &order)?;
    eprintln!("engine load: {:.1}s", t_load.elapsed().as_secs_f64());
    let (free2, _) = qx_gpu::mem::mem_info();
    eprintln!("loaded: {} experts resident ({:.1} GiB), VRAM free after load {:.2} GiB", eng.cache.resident, eng.cache.used as f64 / 1073741824.0, free2 as f64 / 1073741824.0);
    // Warm up (and capture the step graph), then time one run: the prompt's steps and the generated steps.
    eng.warm_up();
    let (h0, r0, s0) = eng.cache_stats();
    let (out, t_prompt, t_end) = eng.generate_timed(&prompt, n_new);
    println!("prompt {} tokens: {:.1} tok/s", prompt.len(), (prompt.len() - 1) as f64 / t_prompt);
    println!("decode {} tokens: {:.1} tok/s", n_new, n_new as f64 / (t_end - t_prompt));
    let (h1, r1, s1) = eng.cache_stats();
    println!("expert cache: {:.1}% of routings served by the GPU, {:.1} swaps/token", 100.0 * (h1 - h0) as f64 / (r1 - r0).max(1) as f64, (s1 - s0) as f64 / (prompt.len() + n_new) as f64);
    let (pr, ph, pread) = eng.ple_stats();
    println!("PLE rows: {pr} asked, {:.1}% from the row cache, {pread} read from the drive", 100.0 * ph as f64 / pr.max(1) as f64);
    println!("first tokens: {:?}", &out[..out.len().min(16)]);
    println!("last tokens: {:?}", &out[out.len().saturating_sub(16)..]);
    Ok(())
}
