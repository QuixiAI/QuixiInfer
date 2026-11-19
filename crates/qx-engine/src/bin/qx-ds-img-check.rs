//! `qx-ds-img-check MODEL.gguf TOKENS BLOCK.f32 [ROWS]`: image positions in the engine against the reference's
//! rules, numerically (QX_DS_PACK as for qx-ds-check; BLOCK.f32 from `qx-vision-check`; QX_DS_CHUNK and
//! QX_DS_PREFILL_EXACT=1 for the chunk comparison).
//!
//! The sequence is 12 text tokens, the first ROWS (default 40) rows of the block, 12 text tokens. As single steps
//! with the activations dumped, per position: the stream initialisation (`hc_init`) must be the block's row for an
//! image position and not for a text one; the Engram gates must be 0 at image positions and not elsewhere; the
//! router's picks must be the top k of sqrt(softplus(logits)) + bias with `exp_probs_b_vl` at image positions and
//! `exp_probs_b` elsewhere (and the count of image positions where the other bias would have picked differently
//! is printed: the check has teeth only if it is not 0). Then the same sequence in windows of 8 and as one exact
//! chunk: the logits after the last token must be the single steps' bit for bit.
use qx_engine::dsv41::DsEngine;
use qx_pack::gguf::Gguf;

fn f32s(g: &Gguf, name: &str) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|| panic!("no tensor {name}"));
    g.data(t).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn topk(logits: &[f32], bias: &[f32], k: usize) -> Vec<i32> {
    let mut s: Vec<(f32, i32)> = logits.iter().zip(bias).enumerate().map(|(e, (&x, &b))| ((if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt() + b, e as i32)).collect();
    s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    s[..k].iter().map(|x| x.1).collect()
}

fn main() -> anyhow::Result<()> {
    qx_engine::tune_host_allocator();
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let text: Vec<u32> = std::fs::read(&a[2])?.chunks_exact(4).take(24).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let block: Vec<f32> = std::fs::read(&a[3])?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let rows: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(40);
    qx_engine::dsv41::pin_engine_early();
    let pk = Gguf::open(std::path::Path::new(&std::env::var("QX_DS_PACK")?))?;
    let gib = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let prof = std::env::var("QX_DS_PROFILE").unwrap_or_else(|_| "golden/profile-dsv41.txt".into());
    let mut eng = DsEngine::load_tiered(&pk, 8192, gib("QX_DS_VRAM", 20.0), gib("QX_DS_RAM", 60.0), Some(std::path::Path::new(&prof)))?;
    let _ = g;
    let (e, hc, k, ne, nl, nv) = (eng.cfg.n_embd, eng.cfg.hc, eng.cfg.n_expert_used, eng.cfg.n_expert, eng.cfg.n_layer, eng.cfg.n_vocab);
    assert!(block.len() >= rows * e, "the block has {} rows", block.len() / e);
    let ids = eng.image_ids(&[(0x1234, &block[..rows * e])], &|_, _| false, &[]).map_err(anyhow::Error::msg)?.remove(0);
    let seq: Vec<u32> = text[..12].iter().copied().chain(ids.iter().copied()).chain(text[12..].iter().copied()).collect();
    let is_img = |p: usize| p >= 12 && p < 12 + rows;
    let bias: Vec<Vec<f32>> = (0..nl).map(|l| f32s(&pk, &format!("blk.{l}.exp_probs_b.bias"))).collect();
    let bias_vl: Vec<Vec<f32>> = (0..nl).map(|l| f32s(&pk, &format!("blk.{l}.exp_probs_b_vl.bias"))).collect();

    // single steps
    let (mut init_bad, mut init_text_same, mut gate_bad, mut gates_seen, mut text_gate_zero) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut pick_ok, mut pick_n, mut pick_other_differs, mut text_ok, mut text_n) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut step_logits: Vec<Vec<f32>> = Vec::new();
    for (p, &t) in seq.iter().enumerate() {
        let mut lg: std::collections::HashMap<usize, Vec<f32>> = Default::default();
        let mut cb = |name: &str, v: Vec<f32>| {
            if name == "hc_init" {
                if is_img(p) {
                    let want = &block[(p - 12) * e..(p - 12 + 1) * e];
                    init_bad += (0..hc).any(|c| v[c * e..(c + 1) * e].iter().zip(want).any(|(a, b)| a.to_bits() != b.to_bits())) as usize;
                } else {
                    init_text_same += (v[..e] == block[..e]) as usize;
                }
            } else if let Some(l) = name.strip_prefix("engram_gate-") {
                let _ = l;
                gates_seen += 1;
                let zero = v[..hc].iter().all(|&x| x == 0.0);
                if is_img(p) { gate_bad += !zero as usize } else { text_gate_zero += zero as usize }
            } else if let Some(l) = name.strip_prefix("ffn_moe_logits-") {
                lg.insert(l.parse().unwrap(), v);
            } else if let Some(l) = name.strip_prefix("ffn_moe_ids-") {
                let l: usize = l.parse().unwrap();
                let got: Vec<i32> = v[..k].iter().map(|x| x.to_bits() as i32).collect();
                let x = &lg[&l][..ne];
                let (with_vl, with_text) = (topk(x, &bias_vl[l], k), topk(x, &bias[l], k));
                if is_img(p) {
                    pick_n += 1;
                    pick_ok += (got == with_vl) as usize;
                    pick_other_differs += (with_vl != with_text) as usize;
                } else {
                    text_n += 1;
                    text_ok += (got == with_text) as usize;
                }
            }
        };
        step_logits.push(eng.forward(t, Some(&mut cb)));
    }
    println!("image positions {rows}, text positions 24, layers {nl}");
    println!("i1. embedding of an image position is its block row (every stream, bit for bit): {} / {rows}; text positions equal to the block's first row: {init_text_same}", rows - init_bad);
    println!("i2. Engram gates at image positions all zero: {} wrong of {} position-layers; text position-layers with all-zero gates: {text_gate_zero}", gate_bad, gates_seen * rows / seq.len());
    println!("i3. router picks at image positions = top {k} with the image bias: {pick_ok} / {pick_n} (the text bias would pick differently at {pick_other_differs}); at text positions = top {k} with the text bias: {text_ok} / {text_n}");
    let last = &step_logits[seq.len() - 1];
    // windows of 8
    eng.reset(0);
    let mut out = Vec::new();
    for w in seq.chunks(8) {
        out = eng.forward_window(w, None);
    }
    let same_w = out[out.len() - nv..].iter().zip(last).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("i4. windows of 8 vs single steps, logits after the last token: {}", if same_w { "bit-identical" } else { "DIFFERENT" });
    let mut ok = init_bad == 0 && init_text_same == 0 && gate_bad == 0 && pick_ok == pick_n && text_ok == text_n && pick_other_differs > 0 && same_w;
    if std::env::var("QX_DS_CHUNK").is_ok_and(|v| !v.is_empty() && v != "0") {
        eng.reset(0);
        let out = eng.forward_chunk(0, &seq);
        let same_c = out[out.len() - nv..].iter().zip(last).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("i5. one chunk (QX_DS_PREFILL_EXACT={}) vs single steps, logits after the last token: {}", std::env::var("QX_DS_PREFILL_EXACT").unwrap_or_default(), if same_c { "bit-identical" } else { "DIFFERENT" });
        ok = ok && same_c;
    }
    println!("{}", if ok { "ok" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 })
}
