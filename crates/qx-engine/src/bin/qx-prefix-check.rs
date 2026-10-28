//! `qx-prefix-check TOKENS_FILE [N_PREFIX] [N_REST] [N_GEN]`: a prompt of N_PREFIX + N_REST tokens run straight
//! (prefill of the prefix, windows for the rest, N_GEN greedy tokens) against: prefill of the prefix, a
//! snapshot, 12 generated tokens, a reset, a restore of the snapshot, then the same rest and generation. The
//! logits of the rest and the generated tokens must be identical bit for bit.
use anyhow::Result;
use qx_engine::engine::{Engine, Sampling, WMAX};
use qx_pack::gguf::Gguf;

fn main() -> Result<()> {
    qx_gpu::init();
    unsafe { libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0) };
    let a: Vec<String> = std::env::args().collect();
    let model = std::env::var("QX_MODEL").unwrap_or_else(|_| qx_engine::default_model_path());
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let np: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(300);
    let nr: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(40);
    let ng: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(24);
    let order = qx_engine::routing::default_order(&g, 48, 512);
    let mut eng = Engine::load_slots(&g, (np + nr + ng + 64).next_multiple_of(256), qx_gpu::mem::mem_info().0, &order, 2)?;
    eng.warm_up();
    let n = eng.cfg.n_vocab;
    let greedy = Sampling { temperature: 0.0, ..Default::default() };
    let prompt = &toks[..np + nr];
    // the rest of the prompt in windows, then generation: logits rows and tokens
    let run_rest = |eng: &mut Engine, slot: usize| -> (Vec<f32>, Vec<u32>) {
        let mut rows = Vec::new();
        while eng.slot_pos(slot) < np + nr - 1 {
            let w = (np + nr - 1 - eng.slot_pos(slot)).min(WMAX);
            eng.step_segments(&[(slot, w)]);
            rows.extend(eng.window_logits(w));
        }
        let mut out = Vec::new();
        for _ in 0..ng {
            out.push(eng.step_segments(&[(slot, 1)])[0]);
        }
        (rows, out)
    };
    // straight: prefill the prefix (begin_seq prefills all but the last token of what it is given)
    eprintln!("phase: straight");
    eng.begin_seq_deferred(0, prompt, &greedy, true);
    eng.prefill_more(0, np);
    assert_eq!(eng.slot_pos(0), np, "the prefix was not prefilled");
    let (rows_a, out_a) = run_rest(&mut eng, 0);
    eprintln!("phase: snapshot");
    // with a snapshot and restore
    eng.begin_seq_deferred(1, prompt, &greedy, true);
    eng.prefill_more(1, np);
    eng.prefix_snapshot(1, 1);
    eprintln!("phase: perturb");
    for _ in 0..12 {
        eng.step_segments(&[(1, 1)]);
    }
    eprintln!("phase: restore");
    let (e, p) = eng.prefix_lookup(prompt, 1).expect("no cached prefix");
    assert_eq!(p, np);
    eng.prefix_restore(1, e, prompt, &greedy);
    eprintln!("phase: rest");
    let (rows_b, out_b) = run_rest(&mut eng, 1);
    let differ = rows_a.chunks(n).zip(rows_b.chunks(n)).filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.to_bits() != q.to_bits())).count();
    println!("restored prefix of {np}: {differ} of {} logits rows differ; generated tokens identical: {}", rows_a.len() / n, out_a == out_b);
    let (entries, hits, restored, snaps, ev) = eng.prefix_stats();
    println!("cache: {entries} entries, {hits} hits, {restored} tokens restored, {snaps} snapshots, {ev} evictions");
    let ok = differ == 0 && out_a == out_b;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
