//! `qx-ds-multi PACK TOKENS N`: batch invariance across sequences, the property the server relies on and that
//! `tools/ds-checks.sh` (one sequence per window) does not exercise. Sequence A (slot 0) is decoded alone in single
//! steps; then A is decoded again beside sequences B and C (slots 1 and 2, other stretches of the same token file),
//! in windows that mix them the way the server's scheduler does: A's one token, B's 0-5 prompt tokens, C's one token,
//! the composition changing every step. A's logits must be bit-identical between the two runs. The tiers adapt as in
//! the server (QX_DS_ADAPT unset), with the VRAM/RAM budgets of `QX_DS_VRAM` / `QX_DS_RAM` (GiB, default 20 / 60).
//! Prints the number of identical rows and the first differing position; exit 1 on any difference.
//! Added 2026-11-08 after one late token flipped in the server under three concurrent streams.
use anyhow::Result;
use qx_engine::dsv41::DsEngine;
use qx_pack::gguf::Gguf;

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "usage: qx-ds-multi PACK TOKENS N");
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let toks: Vec<u32> = std::fs::read(&a[2])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let n: usize = a[3].parse()?;
    anyhow::ensure!(toks.len() >= 3 * n + 600, "the token file is too short for three stretches of {n}");
    let sa = &toks[..n];
    let sb = &toks[n + 200..];
    let sc = &toks[2 * n + 400..3 * n + 400];
    qx_engine::dsv41::pin_engine_early();
    let gib = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let prof = std::env::var("QX_DS_PROFILE").unwrap_or_else(|_| "golden/profile-dsv41.txt".into());
    let max_ctx = (6 * n + 256).next_multiple_of(256);
    let mut eng = DsEngine::load_tiered_slots(&g, max_ctx, gib("QX_DS_VRAM", 20.0), gib("QX_DS_RAM", 60.0), Some(std::path::Path::new(&prof)), 3)?;
    // run 1: A alone, single steps
    for s in 0..3 {
        eng.reset(s);
    }
    let mut alone: Vec<Vec<u32>> = Vec::with_capacity(n);
    for &t in sa {
        let l = eng.forward_segments(&[(0, t)], None);
        alone.push(l.iter().map(|v| v.to_bits()).collect());
    }
    // run 2: A beside B (0-5 prompt tokens a step, cycling) and C (one token a step), fresh slots
    for s in 0..3 {
        eng.reset(s);
    }
    let mut bi = 0usize;
    let mut same = 0usize;
    let mut first: Option<usize> = None;
    for (i, &t) in sa.iter().enumerate() {
        let kb = [0usize, 1, 3, 5, 2, 4][i % 6];
        let mut segs: Vec<(usize, u32)> = Vec::new();
        // the scheduler's order: decoders first, then prompt tokens
        segs.push((0, t));
        segs.push((2, sc[i]));
        for j in 0..kb {
            segs.push((1, sb[bi + j]));
        }
        bi += kb;
        let l = eng.forward_segments(&segs, None);
        let nv = l.len() / segs.len();
        let row: Vec<u32> = l[..nv].iter().map(|v| v.to_bits()).collect();
        if row == alone[i] {
            same += 1;
        } else if first.is_none() {
            first = Some(i);
        }
    }
    println!("multi-sequence windows vs A alone: bit-identical rows: {same} / {n}{}", first.map(|p| format!(" (first difference at position {p})")).unwrap_or_default());
    std::process::exit(if same == n { 0 } else { 1 });
}
