//! `qx-ds-check MODEL.gguf TOKENS N`: runs the DeepSeek-V4.1 GPU decode path and the CPU reference on the same
//! tokens and compares every named activation and the logits, position by position. The reference rounds the
//! experts' inputs as the pool does (`QX_REF_COMPAT=experts`), so the two compute the same thing up to float
//! rounding; `QX_EXACT_EXPERTS=1` runs the GPU path's experts in f32 too.
use qx_engine::dsv41::DsEngine;
use qx_pack::gguf::Gguf;
use std::collections::HashMap;

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let n: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let d: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    n / d.max(1e-30)
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|v| *v as f64 - z).collect()
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if x[best].is_nan() || v > x[best] {
            best = i;
        }
    }
    best
}

fn main() -> anyhow::Result<()> {
    qx_engine::tune_host_allocator();
    if std::env::var("QX_REF_COMPAT").is_err() {
        // before any thread exists: the reference reads it once
        unsafe { std::env::set_var("QX_REF_COMPAT", "experts") };
    }
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let toks: Vec<u32> = std::fs::read(&a[2])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n: usize = a[3].parse()?;
    let toks = &toks[..n.min(toks.len())];
    let verbose = std::env::var("QX_DS_VERBOSE").is_ok();
    let model = qx_ref::dsv41::Model::load(&g)?;
    let mut st = model.new_state();
    // QX_DS_PACK=PACK.qxp with QX_DS_VRAM / QX_DS_RAM (GiB) runs the tiered path off the pack (QX_DS_PROFILE: ranking)
    // QX_DS_MAIN_CPU: pin the engine thread (its sync spins) off the pool's cores
    if let Some(cpu) = std::env::var("QX_DS_MAIN_CPU").ok().and_then(|v| v.parse().ok()) {
        qx_engine::dsv41::pin_thread(cpu);
    }
    let pack = std::env::var("QX_DS_PACK").ok().map(|p| Gguf::open(std::path::Path::new(&p))).transpose()?;
    let max_ctx = (toks.len() + 64).next_multiple_of(256);
    let mut eng = match &pack {
        Some(pk) => {
            let gib = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            let prof = std::env::var("QX_DS_PROFILE").unwrap_or_else(|_| "golden/profile-dsv41.txt".into());
            DsEngine::load_tiered(pk, max_ctx, gib("QX_DS_VRAM", 20.0), gib("QX_DS_RAM", 60.0), Some(std::path::Path::new(&prof)))?
        }
        None => DsEngine::load(&g, max_ctx)?,
    };
    eng.exact_experts = std::env::var("QX_EXACT_EXPERTS").is_ok_and(|v| v == "1");
    // QX_DS_BENCH=N: no reference; run the prompt tokens, then N greedy tokens, and report the time per token.
    // N = 0: the forced tokens only (a text-independent benchmark), timing every forward.
    if let Some(n_gen) = std::env::var("QX_DS_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        let mut tok = toks[0];
        let mut out_toks = Vec::new();
        let (mut t_all, mut t_exp, mut t_eng) = (0f64, 0f64, 0f64);
        let mut per_tok: Vec<f64> = Vec::new();
        let mut phases: Vec<[f64; 5]> = Vec::new();
        let mut nan_seen = false;
        let total = toks.len() + n_gen;
        // QX_DS_DUMP_LOGITS=path: every position's logits (f32 LE rows) for comparisons across modes
        let mut dump_file = std::env::var("QX_DS_DUMP_LOGITS").ok().map(|f| std::io::BufWriter::new(std::fs::File::create(f).expect("logits dump")));
        let mut dump_row = |f: &mut Option<std::io::BufWriter<std::fs::File>>, v: &[f32]| {
            if let Some(f) = f {
                use std::io::Write;
                f.write_all(bytemuck::cast_slice(v)).expect("logits dump");
            }
        };
        // QX_DS_WINDOW=T (forced mode): the tokens in windows of T through forward_window
        let window = std::env::var("QX_DS_WINDOW").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
        if n_gen == 0 && window > 1 {
            // QX_DS_TAIL=N: the last N tokens one at a time (decode at the long context the windows built)
            let tail = std::env::var("QX_DS_TAIL").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(toks.len() - 1);
            let mut tail_times: Vec<f64> = Vec::new();
            let mut tail_phase = [0f64; 5];
            let mut tail_engram = 0f64;
            let mut p = 0;
            while p < toks.len() {
                let cap = if p >= toks.len() - tail { 1 } else { window.min(toks.len() - tail - p) };
                let w = &toks[p..(p + cap).min(toks.len())];
                let logits = eng.forward_window(w, None);
                let nv = logits.len() / w.len();
                for (i, row) in logits.chunks(nv).enumerate() {
                    dump_row(&mut dump_file, row);
                    if !nan_seen && row.iter().any(|v| v.is_nan()) {
                        eprintln!("first NaN logits at position {}", p + i);
                        nan_seen = true;
                    }
                    out_toks.push(argmax(row) as u32);
                }
                t_all += eng.t_forward;
                t_eng += eng.t_engram;
                per_tok.extend(std::iter::repeat(eng.t_forward / w.len() as f64).take(w.len()));
                if w.len() == 1 && tail > 0 {
                    if tail_times.len() < 4 {
                        eprintln!("  tail step {} at {}: {:.1} ms (uploads {:.2}, launch/replay {:.2}, wait {:.2}, flush {:.2}, download {:.2})", tail_times.len(), p, eng.t_forward * 1e3, eng.t_phase[0] * 1e3, eng.t_phase[1] * 1e3, eng.t_phase[2] * 1e3, eng.t_phase[3] * 1e3, eng.t_phase[4] * 1e3);
                    }
                    tail_times.push(eng.t_forward);
                    for (i, v) in eng.t_phase.iter().enumerate() {
                        tail_phase[i] += v;
                    }
                    tail_engram += eng.t_engram;
                }
                p += w.len();
                if (p / window) % 4 == 0 && w.len() > 1 || p == toks.len() {
                    let tiers = eng.tier_stats().unwrap_or_default();
                    eprintln!("pos {}: {:.0} ms per window of {} ({:.1} ms per token){tiers}", p - 1, eng.t_forward * 1e3, w.len(), eng.t_forward * 1e3 / w.len() as f64);
                }
            }
            let n = per_tok.len();
            let last = &per_tok[n.saturating_sub(32)..];
            let last_ms = last.iter().sum::<f64>() * 1e3 / last.len().max(1) as f64;
            println!("windows of {window}: {} tokens, {:.1} ms per token ({:.1} ms engram gather), {:.1} tok/s; last {}: {:.1} ms per token, {:.1} tok/s; tokens {:?}", n, t_all * 1e3 / n as f64, t_eng * 1e3 / n as f64, n as f64 / t_all, last.len(), last_ms, 1e3 / last_ms, &out_toks[..out_toks.len().min(24)]);
            if !tail_times.is_empty() {
                let m = tail_times.iter().sum::<f64>() * 1e3 / tail_times.len() as f64;
                println!("decode tail of {} single steps at position {}+: {:.1} ms per token ({:.1} tok/s)", tail_times.len(), toks.len() - tail_times.len(), m, 1e3 / m);
                let k = tail_times.len() as f64 / 1e3;
                println!("  phases per step: engram {:.2} ms, uploads {:.2}, launch/replay {:.2}, device wait {:.2}, history flush {:.2}, logits download {:.2}", tail_engram / k, tail_phase[0] / k, tail_phase[1] / k, tail_phase[2] / k, tail_phase[3] / k, tail_phase[4] / k);
            }
            return Ok(());
        }
        // QX_DS_REWIND=P / QX_DS_PARK=P: keep every row of the straight pass, then rewind (or park and unpark) slot 0
        // to P and re-run positions P..: every row must come back bit for bit (the replay-based rewind, 2026-11-06)
        let rewind_to = std::env::var("QX_DS_REWIND").ok().and_then(|v| v.parse::<usize>().ok());
        let park_to = std::env::var("QX_DS_PARK").ok().and_then(|v| v.parse::<usize>().ok());
        let mut kept: Vec<Vec<f32>> = Vec::new();
        for p in 0..total {
            // QX_DS_NANCHECK=1: run every forward with activation dumps and name the first NaN activation
            let nancheck = std::env::var("QX_DS_NANCHECK").is_ok_and(|v| v == "1");
            let mut first_nan: Option<String> = None;
            let mut cb = |name: &str, v: Vec<f32>| {
                if first_nan.is_none() && v.iter().any(|x| !x.is_finite()) {
                    let bad = v.iter().filter(|x| !x.is_finite()).count();
                    first_nan = Some(format!("{name} ({bad} of {} not finite)", v.len()));
                }
            };
            let logits = eng.forward(tok, if nancheck { Some(&mut cb) } else { None });
            dump_row(&mut dump_file, &logits);
            if (rewind_to.is_some() || park_to.is_some()) && p < toks.len() {
                kept.push(logits.clone());
            }
            if let Some(n) = first_nan {
                eprintln!("position {p}: first non-finite activation {n}");
            }
            if !nan_seen && logits.iter().any(|v| v.is_nan()) {
                eprintln!("first NaN logits at position {p}");
                nan_seen = true;
            }
            let next = argmax(&logits) as u32;
            if p >= toks.len() || n_gen == 0 {
                t_all += eng.t_forward;
                t_exp += eng.t_experts;
                t_eng += eng.t_engram;
                per_tok.push(eng.t_forward);
                phases.push(eng.t_phase);
                out_toks.push(next);
            }
            tok = if p + 1 < toks.len() { toks[p + 1] } else { next };
            if p % 8 == 7 || p + 1 == total {
                let tiers = eng.tier_stats().unwrap_or_default();
                eprintln!("pos {p}: {:.0} ms ({:.0} ms experts){tiers}", eng.t_forward * 1e3, eng.t_experts * 1e3);
            }
        }
        for (what, target) in [("rewind", rewind_to), ("park/unpark", park_to)] {
            let Some(pp) = target else { continue };
            assert!(pp < toks.len(), "{what} target inside the prompt");
            let t0 = std::time::Instant::now();
            if what == "rewind" {
                eng.rewind(0, pp);
            } else {
                let parked = eng.park(0);
                eng.unpark(0, parked, pp);
            }
            let t_restore = t0.elapsed().as_secs_f64();
            let mut same = 0usize;
            let mut first_bad: Option<usize> = None;
            for p in pp..toks.len() {
                let logits = eng.forward(toks[p], None);
                if logits == kept[p] {
                    same += 1;
                } else if first_bad.is_none() {
                    first_bad = Some(p);
                }
            }
            println!("{what} to {pp} then positions {pp}..{}: bit-identical rows: {same} / {} (restore {:.2}s{})", toks.len(), toks.len() - pp, t_restore,
                first_bad.map(|b| format!(", first differing row {b}")).unwrap_or_default());
        }
        let last = &per_tok[per_tok.len().saturating_sub(32)..];
        let last_ms = last.iter().sum::<f64>() * 1e3 / last.len().max(1) as f64;
        let n_gen = n_gen.max(per_tok.len());
        if let Some(ph) = phases.get(phases.len().saturating_sub(32)..) {
            let k = ph.len().max(1) as f64 / 1e3;
            let sum = |i: usize| ph.iter().map(|v| v[i]).sum::<f64>() / k;
            eprintln!("  phases per step (last {}): uploads {:.2} ms, launch/replay {:.2}, device wait {:.2}, history flush {:.2}, logits download {:.2}", ph.len(), sum(0), sum(1), sum(2), sum(3), sum(4));
        }
        println!("generated {} tokens: {:.1} ms per token ({:.1} ms experts, {:.1} ms engram gather), {:.1} tok/s; last {}: {:.1} ms per token, {:.1} tok/s; tokens {:?}", n_gen, t_all * 1e3 / n_gen as f64, t_exp * 1e3 / n_gen as f64, t_eng * 1e3 / n_gen as f64, n_gen as f64 / t_all, last.len(), last_ms, 1e3 / last_ms, &out_toks[..out_toks.len().min(24)]);
        return Ok(());
    }
    let (mut agree, mut kld_sum) = (0usize, 0f64);
    for (p, &t) in toks.iter().enumerate() {
        let mut reference: HashMap<String, Vec<f32>> = HashMap::new();
        let ref_logits = model.forward(t, &mut st, &mut |name, v| {
            reference.entry(name.to_string()).or_insert_with(|| v.to_vec());
        });
        let mut report: Vec<(String, f32)> = Vec::new();
        let trace = std::env::var("QX_DS_TRACE").is_ok();
        let mut cb = |name: &str, v: Vec<f32>| {
            let nan = v.iter().filter(|x| !x.is_finite()).count();
            if let Some(r) = reference.get(name) {
                if r.len() == v.len() {
                    let e = rel(&v, r);
                    if trace {
                        // for the experts' inputs: how many Q8_K values the two sides round differently
                        let flips = if name.starts_with("ffn_norm-") {
                            let (mut a, mut b) = (Vec::new(), Vec::new());
                            qx_cpu::act::quantize_q8_k(&v, &mut a);
                            qx_cpu::act::quantize_q8_k(r, &mut b);
                            let n = a.iter().zip(&b).map(|(x, y)| x.qs.iter().zip(&y.qs).filter(|(p, q)| p != q).count()).sum::<usize>();
                            format!(", {n} Q8_K flips")
                        } else {
                            String::new()
                        };
                        eprintln!("act {name}: rel err {e:.2e}, {nan} non-finite{flips}, gpu[0..3] {:?} ref[0..3] {:?}", &v[..3.min(v.len())], &r[..3.min(r.len())]);
                    }
                    report.push((name.to_string(), e));
                } else {
                    if trace {
                        eprintln!("act {name}: len {} vs ref {}", v.len(), r.len());
                    }
                    report.push((format!("{name} (len {} vs {})", v.len(), r.len()), f32::NAN));
                }
            } else if trace {
                eprintln!("act {name}: no reference ({} values, {nan} non-finite)", v.len());
            }
        };
        let logits = eng.forward(t, Some(&mut cb));
        let lp = log_softmax(&ref_logits);
        let lq = log_softmax(&logits);
        let kld: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
        kld_sum += kld;
        let same = argmax(&ref_logits) == argmax(&logits);
        agree += same as usize;
        if p == 0 || verbose {
            for (name, e) in &report {
                println!("pos {p}  {name:28} rel err {e:.2e}");
            }
        } else {
            let worst = report.iter().cloned().filter(|r| !r.1.is_nan()).fold(("".into(), 0f32), |a, b| if b.1 > a.1 { b } else { a });
            let first_bad = report.iter().find(|r| r.1 > 1e-3).map(|r| format!("{} {:.1e}", r.0, r.1)).unwrap_or_default();
            println!("pos {p}  {} activations, worst {} {:.2e}, first above 1e-3: {}", report.len(), worst.0, worst.1, first_bad);
        }
        let tiers = eng.tier_stats().unwrap_or_default();
        println!("pos {p:4} tok {t:6}  KLD {kld:.6}  top1 {} (ref {}, gpu {})  {:.0} ms ({:.0} experts){tiers}", if same { "same" } else { "DIFF" }, argmax(&ref_logits), argmax(&logits), eng.t_forward * 1e3, eng.t_experts * 1e3);
    }
    println!("positions {}: top-1 agreement {agree}/{}, KLD mean {:.6}", toks.len(), toks.len(), kld_sum / toks.len() as f64);
    Ok(())
}
