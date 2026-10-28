//! `qx-invariance TOKENS N GIB_A GIB_B`: runs the same forced tokens with two VRAM expert-cache sizes (so
//! different experts run on the GPU and on the CPU) and checks the logits are bitwise identical.
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;

fn run(g: &Gguf, toks: &[u32], gib: f64) -> Result<(Vec<Vec<f32>>, f64)> {
    let order: Vec<(usize, usize)> = (0..512).flat_map(|e| (0..48).map(move |l| (l, e))).collect();
    let mut eng = Engine::load(g, (toks.len() + 64).next_multiple_of(256), (gib * (1u64 << 30) as f64) as usize, &order)?;
    let out: Vec<Vec<f32>> = toks.iter().map(|&t| eng.forward(t, None)).collect();
    let (h, r, _) = eng.cache_stats();
    Ok((out, h as f64 / r.max(1) as f64))
}

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n: usize = a[2].parse()?;
    let toks = &toks[..n.min(toks.len())];
    let (la, ha) = run(&g, toks, a[3].parse()?)?;
    let (lb, hb) = run(&g, toks, a[4].parse()?)?;
    let mut first_diff = None;
    for (p, (x, y)) in la.iter().zip(&lb).enumerate() {
        if x.iter().zip(y).any(|(u, v)| u.to_bits() != v.to_bits()) {
            first_diff = Some(p);
            break;
        }
    }
    println!("GPU share of routings: {:.1}% vs {:.1}%", 100.0 * ha, 100.0 * hb);
    match first_diff {
        None => println!("{} positions: logits bitwise identical", toks.len()),
        Some(p) => println!("logits differ from position {p}"),
    }
    Ok(())
}
