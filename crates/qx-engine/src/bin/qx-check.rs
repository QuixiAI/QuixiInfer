//! `qx-check TOKENS_FILE [N] [DEBUG_POS]`: runs the GPU engine and the CPU reference side by side on the
//! same tokens and compares logits at every position, and activations at DEBUG_POS.
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;
use qx_ref::model::Model;
use std::collections::HashMap;

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
    let d: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    n / d.max(1e-30)
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|v| *v as f64 - z).collect()
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(toks.len()).min(toks.len());
    let dbg_pos: Option<usize> = a.get(3).map(|s| s.parse().unwrap());
    let t0 = std::time::Instant::now();
    // Every other expert resident (QX_CACHE_GIB of VRAM), so both the GPU and the CPU expert paths run.
    let budget = (std::env::var("QX_CACHE_GIB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(4.0) * (1u64 << 30) as f64) as usize;
    let order: Vec<(usize, usize)> = (0..512).step_by(2).flat_map(|e| (0..48).map(move |l| (l, e))).collect();
    let mut eng = Engine::load(&g, (n + 64).next_multiple_of(256), budget, &order)?;
    eprintln!("expert cache: {} experts, {:.2} GiB", eng.cache.resident, eng.cache.used as f64 / (1u64 << 30) as f64);
    eprintln!("engine loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let reference = Model::load(&g)?;
    let mut st = reference.new_state();
    let (mut agree, mut kmax, mut ksum, mut rmax) = (0, 0f64, 0f64, 0f64);
    let mut t_eng = 0f64;
    for (p, &t) in toks.iter().take(n).enumerate() {
        let mut want: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let lr = reference.forward(t, &mut st, &mut |name, v| {
            if Some(p) == dbg_pos {
                want.entry(name.to_string()).or_default().push(v.to_vec());
            }
        });
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut cb = |name: &str, v: Vec<f32>| {
            let c = seen.entry(name.to_string()).or_insert(0);
            if let Some(w) = want.get(name).and_then(|l| l.get(*c)) {
                println!("  pos {p} {name}#{c}: rel err {:.2e}", rel(&v, w));
            }
            *c += 1;
        };
        let te = std::time::Instant::now();
        let le = if Some(p) == dbg_pos { eng.forward(t, Some(&mut cb)) } else { eng.forward(t, None) };
        t_eng += te.elapsed().as_secs_f64();
        let r = rel(&le, &lr);
        let (lp, lq) = (log_softmax(&lr), log_softmax(&le));
        let kld: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
        let same = argmax(&le) == argmax(&lr);
        agree += same as usize;
        kmax = kmax.max(kld);
        ksum += kld;
        rmax = rmax.max(r);
        println!("pos {p:4} tok {t:6}  logits rel err {r:.2e}  KLD {kld:.2e}  top1 {}", if same { "same" } else { "DIFF" });
    }
    println!("{n} positions: top-1 {agree}/{n}, logits rel err max {rmax:.2e}, KLD mean {:.2e} max {kmax:.2e}, engine {:.3} s/token", ksum / n as f64, t_eng / n as f64);
    Ok(())
}
