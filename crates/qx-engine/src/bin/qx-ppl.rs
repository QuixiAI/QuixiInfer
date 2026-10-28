//! `qx-ppl TOKENS_FILE OUT [REF] [SEGMENTS] [LEN]`: teacher-forces SEGMENTS texts of LEN tokens (default 4 x 512,
//! spread over the file; the slot reset between them, the expert cache kept, as between a server's requests)
//! through the decode path in windows of 8, and reports the mean negative log-likelihood of each next token and
//! the VRAM cache's hit rate. Writes every position's log-probabilities (f16) to OUT; with REF (an earlier OUT,
//! say of exact routing) also the KL divergence from REF and how often the argmax agrees. For lossy modes such as
//! cache-aware routing (`QX_ROUTE_CACHE`). The cache is sized as in production (`QX_CACHE_GIB`, default 16.5).
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;
use std::io::{Read, Write};

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = std::env::var("QX_MODEL").unwrap_or_else(|_| qx_engine::default_model_path());
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let refp = a.get(3).filter(|s| !s.is_empty());
    let segs: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    let len: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(512);
    let budget = (std::env::var("QX_CACHE_GIB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(16.5) * (1u64 << 30) as f64) as usize;
    let order = qx_engine::routing::default_order(&g, 48, 512);
    let mut eng = Engine::load(&g, (len + 64).next_multiple_of(256), budget, &order)?;
    eng.warm_up();
    let n = eng.cfg.n_vocab;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    let mut rf = refp.map(|p| std::fs::File::open(p).map(std::io::BufReader::new)).transpose()?;
    let (h0, r0, _) = eng.cache_stats();
    let (mut nll, mut kl, mut agree, mut count) = (0f64, 0f64, 0usize, 0usize);
    let mut row = vec![0u8; n * 2];
    let t0 = std::time::Instant::now();
    for sgi in 0..segs {
        let at = (sgi * 7919 * 131) % (toks.len() - len - 1);
        let text = &toks[at..at + len + 1];
        eng.reset();
        for w in (0..len).step_by(8) {
            let win = &text[w..(w + 8).min(len)];
            let logits = eng.forward_window(win);
            for (i, lg) in logits.iter().enumerate() {
                let m = lg.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
                let z: f64 = lg.iter().map(|&x| ((x as f64) - m).exp()).sum::<f64>().ln() + m;
                let lp: Vec<f64> = lg.iter().map(|&x| x as f64 - z).collect();
                nll -= lp[text[w + i + 1] as usize];
                let am = (0..n).max_by(|&x, &y| lp[x].partial_cmp(&lp[y]).unwrap()).unwrap();
                for (j, &v) in lp.iter().enumerate() {
                    row[2 * j..2 * j + 2].copy_from_slice(&half::f16::from_f64(v.max(-60.0)).to_le_bytes());
                }
                out.write_all(&row)?;
                if let Some(r) = rf.as_mut() {
                    let mut rb = vec![0u8; n * 2];
                    r.read_exact(&mut rb)?;
                    let rp: Vec<f64> = rb.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f64()).collect();
                    // KL(ref || this)
                    kl += rp.iter().zip(&lp).map(|(&p, &q)| if p > -30.0 { p.exp() * (p - q) } else { 0.0 }).sum::<f64>();
                    let ram = (0..n).max_by(|&x, &y| rp[x].partial_cmp(&rp[y]).unwrap()).unwrap();
                    agree += (ram == am) as usize;
                }
                count += 1;
            }
        }
        eprintln!("  segment {} / {segs}: {count} positions, {:.0}s", sgi + 1, t0.elapsed().as_secs_f64());
    }
    let (h1, r1, _) = eng.cache_stats();
    println!("positions {count}  NLL {:.4} (ppl {:.3})  hit rate {:.1}%", nll / count as f64, (nll / count as f64).exp(), 100.0 * (h1 - h0) as f64 / (r1 - r0).max(1) as f64);
    if rf.is_some() {
        println!("against REF: KL mean {:.3e}  argmax agrees {agree}/{count}", kl / count as f64);
    }
    Ok(())
}
