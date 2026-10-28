//! `qx-prefill-check TOKENS_FILE N [AFTER]`: logits of positions N-1 .. N-1+AFTER from token-by-token decode
//! against batched prefill of the first N-1 tokens followed by decode steps; also times the prefill.
use anyhow::Result;
use qx_engine::engine::Engine;
use qx_pack::gguf::Gguf;

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
    // let a debugger attach (yama ptrace_scope 1 otherwise allows only ancestors)
    unsafe { libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0) };
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n: usize = a[2].parse()?;
    let after: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    assert!(n + after <= toks.len());
    let budget = (std::env::var("QX_CACHE_GIB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(4.0) * (1u64 << 30) as f64) as usize;
    let order: Vec<(usize, usize)> = (0..512).step_by(2).flat_map(|e| (0..48).map(move |l| (l, e))).collect();
    let mut eng = Engine::load(&g, (n + after + 64).next_multiple_of(256), budget, &order)?;
    eprintln!("expert cache: {} experts, {:.2} GiB", eng.cache.resident, eng.cache.used as f64 / (1u64 << 30) as f64);
    let skip_ref = std::env::var("QX_SKIP_DECODE_REF").is_ok();
    let mut want = Vec::new();
    if !skip_ref {
        for (p, &t) in toks.iter().take(n + after).enumerate() {
            let l = eng.forward(t, None);
            if p + 1 >= n {
                want.push(l);
            }
        }
        eng.reset();
    }
    // warm-up of the prefill kernels (compiles them), then a clean run
    eng.prefill_forced(&toks[..n.min(64) - 1]);
    eng.reset();
    let t0 = std::time::Instant::now();
    eng.prefill_forced(&toks[..n - 1]);
    let dt = t0.elapsed().as_secs_f64();
    println!("prefill {} tokens in {:.3}s: {:.0} tok/s", n - 1, dt, (n - 1) as f64 / dt);
    let mut got = Vec::new();
    for &t in &toks[n - 1..n + after] {
        got.push(eng.forward(t, None));
    }
    // QX_DUMP_LOGITS=FILE: the logits after the prefill (f32 rows), to compare runs bit for bit
    if let Ok(f) = std::env::var("QX_DUMP_LOGITS") {
        let flat: Vec<f32> = got.iter().flatten().copied().collect();
        std::fs::write(f, bytemuck::cast_slice(&flat))?;
    }
    if skip_ref {
        return Ok(());
    }
    // Optional ground truth: llama.cpp logits (golden.cpp, rows from QX_GOLDEN_FROM on).
    if let Ok(gp) = std::env::var("QX_GOLDEN") {
        let from: usize = std::env::var("QX_GOLDEN_FROM").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let raw = std::fs::read(&gp)?;
        let nv = want[0].len();
        let row = |p: usize| -> Option<Vec<f32>> {
            let off = (p.checked_sub(from)?) * nv * 4;
            (off + nv * 4 <= raw.len()).then(|| raw[off..off + nv * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let kl = |p: &[f64], q: &[f64]| -> f64 { p.iter().zip(q).map(|(a, b)| a.exp() * (a - b)).sum() };
        let (mut n_cmp, mut kd, mut kp, mut td, mut tp) = (0usize, 0f64, 0f64, 0usize, 0usize);
        for (i, (w, gl)) in want.iter().zip(&got).enumerate() {
            let Some(gold) = row(n - 1 + i) else { continue };
            let (lg, lw, lp) = (log_softmax(&gold), log_softmax(w), log_softmax(gl));
            kd += kl(&lg, &lw);
            kp += kl(&lg, &lp);
            td += (argmax(&gold) == argmax(w)) as usize;
            tp += (argmax(&gold) == argmax(gl)) as usize;
            n_cmp += 1;
        }
        println!("vs llama.cpp over {n_cmp} positions: decode-only KL mean {:.2e} top-1 {td}/{n_cmp}; prefill+decode KL mean {:.2e} top-1 {tp}/{n_cmp}", kd / n_cmp.max(1) as f64, kp / n_cmp.max(1) as f64);
    }
    let (mut agree, mut kmax, mut ksum) = (0, 0f64, 0f64);
    for (i, (w, gl)) in want.iter().zip(&got).enumerate() {
        let (lw, lg) = (log_softmax(w), log_softmax(gl));
        let kl: f64 = lw.iter().zip(&lg).map(|(p, q)| p.exp() * (p - q)).sum();
        let same = argmax(w) == argmax(gl);
        agree += same as usize;
        kmax = kmax.max(kl);
        ksum += kl;
        if std::env::var("QX_QUIET").is_err() {
            println!("pos {}: KL {:.2e} top1 {} ({} vs {})", n - 1 + i, kl, if same { "same" } else { "DIFF" }, argmax(w), argmax(gl));
        }
    }
    println!("top-1 agreement {}/{}, KL mean {:.2e} max {:.2e}", agree, want.len(), ksum / want.len() as f64, kmax);
    Ok(())
}
