//! `qx-gen-check TOKENS_FILE [N_NEW]`: the eager `forward` path's argmax at every prompt position against the
//! production paths' greedy generation from the same prompt (batched stream, then speculative). The first
//! generated token must be the last position's argmax; the texts are printed.
use anyhow::Result;
use qx_engine::engine::{Engine, Sampling};
use qx_pack::gguf::Gguf;

fn argmax(x: &[f32]) -> u32 {
    x.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
}

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_new: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let (free, _) = qx_gpu::mem::mem_info();
    let budget = free.saturating_sub((6.2 * (1u64 << 30) as f64) as usize);
    let order: Vec<(usize, usize)> = (0..512).flat_map(|e| (0..48).map(move |l| (l, e))).collect();
    let mut eng = Engine::load(&g, (toks.len() + n_new + 64).next_multiple_of(256), budget, &order)?;
    eng.warm_up();
    let tok = qx_tok::Tok::from_gguf(&g)?;
    let mut am = Vec::new();
    let mut nan = 0;
    for &t in &toks {
        let l = eng.forward(t, None);
        nan += l.iter().filter(|v| !v.is_finite()).count();
        am.push(argmax(&l));
    }
    println!("eager forward: argmax per position {:?} ({nan} non-finite logits) -> {:?}", am, tok.decode(&am)?);
    eng.reset();
    let greedy = Sampling { temperature: 0.0, ..Default::default() };
    let mut out = Vec::new();
    eng.spec = false;
    eng.stream(&toks, n_new, &greedy, &[], |t| {
        out.push(t);
        true
    });
    println!("stream greedy: {:?} -> {:?}", out, tok.decode(&out)?);
    eng.reset();
    eng.spec = true;
    let (spec, _) = eng.generate_spec(&toks, n_new, &greedy, 4, &[], None);
    println!("speculative greedy: {:?} -> {:?}", spec, tok.decode(&spec)?);
    let ok = out.first() == am.last() && spec.first() == am.last();
    println!("{}", if ok { "first generated token matches the eager argmax" } else { "MISMATCH: generation disagrees with the eager forward pass" });
    Ok(())
}
