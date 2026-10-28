//! `qx-ref check MODEL GOLDEN_DIR PROMPT_INDEX [N_POSITIONS]`: run the reference on a golden prompt and compare
//! its activations (first positions) and logits (every position) with llama.cpp's.
use anyhow::Result;
use qx_pack::gguf::Gguf;
use qx_ref::model::Model;
use std::collections::HashMap;

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_default();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

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
    x.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 || a[1] != "check" {
        eprintln!("usage: qx-ref check MODEL GOLDEN_DIR PROMPT_INDEX [N_POSITIONS]");
        std::process::exit(2);
    }
    let g = Gguf::open(std::path::Path::new(&a[2]))?;
    let m = Model::load(&g)?;
    let dir = &a[3];
    let k: usize = a[4].parse()?;
    let toks: Vec<u32> = std::fs::read(format!("{dir}/p{k}.tokens"))?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(toks.len()).min(toks.len());
    let nv = m.cfg.n_vocab;
    let logits = read_f32(&format!("{dir}/p{k}.logits"));
    // The golden run may record logits only from some position on: they are the last positions.
    let from = toks.len() - logits.len() / nv;
    // Golden activations: (pos, name) -> files in capture order.
    let mut golden: HashMap<(usize, String), Vec<String>> = HashMap::new();
    if let Ok(idx) = std::fs::read_to_string(format!("{dir}/p{k}.acts/index.tsv")) {
        for line in idx.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            golden.entry((f[0].parse()?, f[2].to_string())).or_default().push(format!("{dir}/p{k}.acts/{}", f[7]));
        }
    }
    let mut st = m.new_state();
    let (mut agree, mut kld_sum, mut kld_max) = (0usize, 0f64, 0f64);
    for (p, &t) in toks.iter().take(n).enumerate() {
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut report: Vec<(String, f32)> = Vec::new();
        let out = m.forward(t, &mut st, &mut |name, v| {
            let c = seen.entry(name.to_string()).or_insert(0);
            if let Some(files) = golden.get(&(p, name.to_string())) {
                if let Some(f) = files.get(*c) {
                    let gv = read_f32(f);
                    if gv.len() == v.len() {
                        report.push((format!("{name}#{c}"), rel(v, &gv)));
                    } else {
                        report.push((format!("{name}#{c} (len {} vs {})", v.len(), gv.len()), f32::NAN));
                    }
                }
            }
            *c += 1;
        });
        if p == 0 || std::env::var("QX_REPORT_POS").ok().and_then(|s| s.parse::<usize>().ok()) == Some(p) {
            for (name, e) in &report {
                println!("pos 0  {name:28} rel err {e:.2e}");
            }
        } else if !report.is_empty() {
            let worst = report.iter().cloned().fold(("".into(), 0f32), |a, b| if b.1 > a.1 { b } else { a });
            println!("pos {p}  {} activations, worst {} {:.2e}", report.len(), worst.0, worst.1);
        }
        if p >= from {
            let gl = &logits[(p - from) * nv..(p - from + 1) * nv];
            let lp = log_softmax(gl);
            let lq = log_softmax(&out);
            let kld: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
            kld_sum += kld;
            kld_max = kld_max.max(kld);
            let same = argmax(gl) == argmax(&out);
            agree += same as usize;
            println!("pos {p:4} tok {t:6}  KLD {kld:.5}  top1 {}", if same { "same" } else { "DIFF" });
        }
    }
    let scored = n.saturating_sub(from);
    println!("positions {scored} (from {from}): top-1 agreement {agree}/{scored}, KLD mean {:.5}, max {:.5}", kld_sum / scored.max(1) as f64, kld_max);
    Ok(())
}
