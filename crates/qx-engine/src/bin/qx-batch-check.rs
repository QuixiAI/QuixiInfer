//! `qx-batch-check TOKENS_FILE B [LEN] [N_GEN]`: B sequence slots, each with its own text of LEN forced tokens
//! (default 80, under the prefill threshold so every token runs in windows). Each slot runs alone in windows of 8,
//! then all slots run together in mixed windows (slot i takes 1 + i % 3 tokens per window, up to TMAX in all):
//! every token's logits must equal its solo row bit for bit. Then N_GEN tokens (default 24) are generated
//! per slot, greedy for even slots and sampled with a per-slot seed for odd ones, alone and together: the
//! tokens must be identical. That holds when one product kernel serves every window (`QX_TMV=0` or `1`); by
//! default windows of several sequences above 4 tokens round differently, and the check wants their logits close
//! (mean KL under 1e-1; measured about 5e-2) instead.
use anyhow::Result;
use qx_engine::engine::{Engine, Sampling, TMAX};
use qx_pack::gguf::Gguf;

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let model = std::env::var("QX_MODEL").unwrap_or_else(|_| qx_engine::default_model_path());
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let b: usize = a[2].parse()?;
    let len: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(80);
    let n_gen: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(24);
    let budget = qx_gpu::mem::mem_info().0;
    let order = qx_engine::routing::default_order(&g, 48, 512);
    let mut eng = Engine::load_slots(&g, (len + n_gen + 64).next_multiple_of(256), budget, &order, b)?;
    eng.warm_up();
    let n = eng.cfg.n_vocab;
    let text = |i: usize| toks[(i * 977) % (toks.len() - len)..][..len].to_vec();
    let greedy = Sampling { temperature: 0.0, ..Default::default() };

    // 1. logits: alone, then together
    let mut solo: Vec<Vec<f32>> = Vec::new();
    for i in 0..b {
        let p0 = eng.begin_seq(i, &text(i), &greedy);
        assert_eq!(p0, 0, "LEN must stay under the prefill threshold");
        let mut rows = Vec::new();
        while eng.slot_pos(i) < len {
            let w = (len - eng.slot_pos(i)).min(8);
            eng.step_segments(&[(i, w)]);
            rows.extend(eng.window_logits(w));
        }
        solo.push(rows);
    }
    let mut together: Vec<Vec<f32>> = vec![Vec::new(); b];
    for i in 0..b {
        eng.begin_seq(i, &text(i), &greedy);
    }
    let mut windows = 0;
    loop {
        let mut segs = Vec::new();
        let mut used = 0;
        for i in 0..b {
            let left = len - eng.slot_pos(i);
            let w = (1 + i % 3).min(left).min(TMAX - used);
            if w > 0 {
                segs.push((i, w));
                used += w;
            }
        }
        if segs.is_empty() {
            break;
        }
        eng.step_segments(&segs);
        windows += 1;
        let l = eng.window_logits(used);
        let mut r = 0;
        for &(i, w) in &segs {
            together[i].extend_from_slice(&l[r * n..(r + w) * n]);
            r += w;
        }
    }
    let (mut differ, mut kl_sum, mut kl_max, mut top1) = (0, 0f64, 0f64, 0);
    let lsm = |x: &[f32]| {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln() + m;
        x.iter().map(|v| *v as f64 - z).collect::<Vec<f64>>()
    };
    let argmax = |x: &[f32]| x.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    for i in 0..b {
        for p in 0..len {
            let (a, c) = (&solo[i][p * n..(p + 1) * n], &together[i][p * n..(p + 1) * n]);
            if a.iter().zip(c).any(|(x, y)| x.to_bits() != y.to_bits()) {
                differ += 1;
            }
            let (la, lc) = (lsm(a), lsm(c));
            let kl: f64 = la.iter().zip(&lc).map(|(x, y)| x.exp() * (x - y)).sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            top1 += (argmax(a) == argmax(c)) as usize;
        }
    }
    let rows = b * len;
    let exact = std::env::var("QX_TMV").is_ok();
    println!("logits: {b} slots x {len} positions in {windows} mixed windows: {differ} rows differ, KL mean {:.2e} max {:.2e}, top-1 {top1}/{rows}", kl_sum / rows as f64, kl_max);

    // 2. generation: alone, then together
    let sp = |i: usize| if i % 2 == 0 { greedy } else { Sampling { temperature: 0.8, top_p: 0.95, top_k: 40, seed: 1000 + i as u64, ..Default::default() } };
    let prompt = |i: usize| text(i)[..len / 2].to_vec();
    let run = |eng: &mut Engine, slots: &[usize]| -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); b];
        for &i in slots {
            eng.begin_seq(i, &prompt(i), &sp(i));
        }
        loop {
            let mut segs: Vec<(usize, usize)> = Vec::new();
            let mut used = 0;
            for &i in slots.iter().filter(|&&i| out[i].len() < n_gen) {
                let w = if eng.slot_pos(i) < len / 2 { (len / 2 - eng.slot_pos(i)).min(4) } else { 1 }.min(TMAX - used);
                if w > 0 {
                    segs.push((i, w));
                    used += w;
                }
            }
            if segs.is_empty() {
                break;
            }
            let smp = eng.step_segments(&segs);
            let mut r = 0;
            for &(i, w) in &segs {
                r += w;
                if eng.slot_pos(i) >= len / 2 {
                    out[i].push(smp[r - 1]);
                }
            }
        }
        out
    };
    let mut alone = vec![Vec::new(); b];
    for i in 0..b {
        alone[i] = run(&mut eng, &[i])[i].clone();
    }
    let all: Vec<usize> = (0..b).collect();
    let both = run(&mut eng, &all);
    let same = (0..b).filter(|&i| alone[i] == both[i]).count();
    println!("generation: {same} of {b} slots identical");
    // the default mixes bf16 (qx_tmv, several sequences, more than 4 tokens) and f32 products: about 5e-2 KL to
    // solo, the same as either path's distance from the exact reference
    let ok = if exact { differ == 0 && same == b } else { kl_sum / (rows as f64) < 1e-1 };
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
