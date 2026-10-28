//! `qx-profile CALIB_TOKENS HELDOUT_TOKENS OUT [N_HELDOUT]`: counts expert routing on a calibration set,
//! writes the ranking (most routed first) as `layer expert count` lines to OUT, then replays a held-out set
//! and reports what share of its routings a cache holding the top-N ranked experts would have served.
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;

fn tokens(p: &str) -> Result<Vec<u32>> {
    Ok(std::fs::read(p)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn run(eng: &mut Engine, toks: &[u32], window: usize) {
    let t0 = std::time::Instant::now();
    for (i, &t) in toks.iter().enumerate() {
        if eng.pos == window {
            eng.reset();
        }
        eng.forward(t, None);
        if (i + 1) % 500 == 0 {
            eprintln!("  {} / {} tokens, {:.1} tok/s", i + 1, toks.len(), (i + 1) as f64 / t0.elapsed().as_secs_f64());
        }
    }
}

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let calib = tokens(&a[1])?;
    let held: Vec<u32> = tokens(&a[2])?.into_iter().take(a.get(4).map(|s| s.parse().unwrap()).unwrap_or(4096)).collect();
    let window = 2048;
    let budget = 12usize << 30;
    let order: Vec<(usize, usize)> = (0..512).flat_map(|e| (0..48).map(move |l| (l, e))).collect();
    let mut eng = Engine::load(&g, window, budget, &order)?;
    let (nl, ne) = (eng.cfg.n_layer, eng.cfg.n_expert);
    eprintln!("calibration: {} tokens", calib.len());
    run(&mut eng, &calib, window);
    let counts = eng.route_counts.borrow().clone();
    let mut ranked: Vec<(usize, usize, u64)> = (0..nl).flat_map(|l| (0..ne).map(move |e| (l, e))).map(|(l, e)| (l, e, counts[l * ne + e])).collect();
    ranked.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let mut out = String::new();
    for (l, e, c) in &ranked {
        out.push_str(&format!("{l} {e} {c}\n"));
    }
    std::fs::write(&a[3], out)?;
    // Held-out replay: count its routings, then score cache sizes against the calibration ranking.
    eng.reset();
    *eng.route_counts.borrow_mut() = vec![0; nl * ne];
    eprintln!("held-out: {} tokens", held.len());
    *eng.route_trace.borrow_mut() = Some(Vec::new());
    run(&mut eng, &held, window);
    if let Some(tr) = eng.route_trace.borrow().as_ref() {
        let bytes: Vec<u8> = tr.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{}.trace", &a[3]), bytes)?;
    }
    let hc = eng.route_counts.borrow().clone();
    let total: u64 = hc.iter().sum();
    let mib = 1.88f64;
    println!("| cache experts | cache GiB | held-out hit rate |\n|---|---|---|");
    for n in [2000usize, 4000, 6000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 20000, 24576] {
        let hit: u64 = ranked.iter().take(n).map(|(l, e, _)| hc[l * ne + e]).sum();
        println!("| {n} | {:.1} | {:.1}% |", n as f64 * mib / 1024.0, 100.0 * hit as f64 / total as f64);
    }
    let zero = ranked.iter().filter(|r| r.2 == 0).count();
    println!("experts never routed in calibration: {zero}");
    Ok(())
}
