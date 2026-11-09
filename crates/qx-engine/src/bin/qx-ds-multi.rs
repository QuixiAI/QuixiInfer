//! `qx-ds-multi PACK TOKENS N`: batch invariance across sequences, the property the server relies on and that
//! `tools/ds-checks.sh` (one sequence per window) does not exercise. Sequence A (slot 0) is decoded alone in single
//! steps; then A is decoded again beside sequences B and C (slots 1 and 2, other stretches of the same token file),
//! in windows that mix them the way the server's scheduler does: A's one token, B's 0-5 prompt tokens, C's one token,
//! the composition changing every step. A's logits must be bit-identical between the two runs. The tiers adapt as in
//! the server (QX_DS_ADAPT unset), with the VRAM/RAM budgets of `QX_DS_VRAM` / `QX_DS_RAM` (GiB, default 20 / 60).
//! Prints the number of identical rows and the first differing position; exit 1 on any difference.
//! Added 2026-11-08 after one late token flipped in the server under three concurrent streams.
//! QX_DS_MULTI_WIDTHS=2,3,4,6,8,12,16,24,32,48,64,96,128 exercises wider prompt mixtures.
//! QX_DS_MULTI_ROTATE=1 moves A among the first and last two rows (preserving each slot's token order).
//! QX_DS_MULTI_CHECK_ALL=1 compares B's prompt rows and C's decode rows against their own single-step references too.
//! QX_DS_MULTI_WARM=1 warms every requested width before the references, as the server does at startup.
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
    anyhow::ensure!(n > 0, "N must be positive");
    let widths: Vec<usize> = std::env::var("QX_DS_MULTI_WIDTHS").unwrap_or_else(|_| "2,3,5,7,4,6".into())
        .split(',').map(str::parse).collect::<std::result::Result<_, _>>()?;
    anyhow::ensure!(!widths.is_empty() && widths.iter().all(|&w| (2..=qx_engine::dsv41::DS_TMAX).contains(&w)), "widths must be in 2..=128");
    let rotate = std::env::var("QX_DS_MULTI_ROTATE").is_ok_and(|v| v == "1");
    let check_all = std::env::var("QX_DS_MULTI_CHECK_ALL").is_ok_and(|v| v == "1");
    let warm = std::env::var("QX_DS_MULTI_WARM").is_ok_and(|v| v == "1");
    let b_tokens: usize = (0..n).map(|i| widths[i % widths.len()] - 2).sum();
    let required = (3 * n + 400).max(n + 200 + b_tokens);
    anyhow::ensure!(toks.len() >= required, "the token file needs {required} tokens for this schedule (has {})", toks.len());
    let sa = &toks[..n];
    let sb = &toks[n + 200..n + 200 + b_tokens];
    let sc = &toks[2 * n + 400..3 * n + 400];
    qx_engine::dsv41::pin_engine_early();
    let gib = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let prof = std::env::var("QX_DS_PROFILE").unwrap_or_else(|_| "golden/profile-dsv41.txt".into());
    let max_ctx = (6 * n).max(b_tokens).max(if warm { widths.iter().sum::<usize>() + 3 } else { 0 }).saturating_add(256).next_multiple_of(256);
    eprintln!("multi: n={n}, widths={widths:?}, rotate={rotate}, check_all={check_all}, warm={warm}, B tokens={b_tokens}, ctx={max_ctx}");
    let mut eng = DsEngine::load_tiered_slots(&g, max_ctx, gib("QX_DS_VRAM", 20.0), gib("QX_DS_RAM", 60.0), Some(std::path::Path::new(&prof)), 3)?;
    if warm {
        let zero = vec![0u32; qx_engine::dsv41::DS_TMAX];
        let _ = eng.forward_window(&zero[..2], None);
        let _ = eng.forward_window(&zero[..1], None);
        for &w in &widths {
            let _ = eng.forward_window(&zero[..w], None);
        }
    }
    // run 1: single-step references (A, or every sequence when checking all rows)
    for s in 0..3 {
        eng.reset(s);
    }
    let mut alone: [Vec<Vec<u32>>; 3] = std::array::from_fn(|_| Vec::new());
    for (slot, sequence) in [sa, sb, sc].into_iter().enumerate() {
        if slot != 0 && !check_all {
            continue;
        }
        for &t in sequence {
            let l = eng.forward_segments(&[(slot, t)], None);
            alone[slot].push(l.iter().map(|v| v.to_bits()).collect());
        }
    }
    // run 2: A beside B (0-5 prompt tokens a step, cycling) and C (one token a step), fresh slots
    for s in 0..3 {
        eng.reset(s);
    }
    let mut bi = 0usize;
    let mut same = [0usize; 3];
    let mut compared = [0usize; 3];
    let mut first: Option<(usize, usize)> = None;
    for (i, &t) in sa.iter().enumerate() {
        let kb = widths[i % widths.len()] - 2;
        let mut segs: Vec<(usize, u32)> = Vec::new();
        // the scheduler's order: decoders first, then prompt tokens
        segs.push((0, t));
        segs.push((2, sc[i]));
        for j in 0..kb {
            segs.push((1, sb[bi + j]));
        }
        bi += kb;
        if rotate {
            // Rotation moves A and C around B's contiguous prompt, without changing B's token order.
            let shift = i % 3;
            segs.rotate_left(shift.min(2));
        }
        let l = eng.forward_segments(&segs, None);
        let nv = l.len() / segs.len();
        for (r, &(slot, _)) in segs.iter().enumerate() {
            if slot != 0 && !check_all {
                continue;
            }
            let pos = compared[slot];
            let row: Vec<u32> = l[r * nv..(r + 1) * nv].iter().map(|v| v.to_bits()).collect();
            if row == alone[slot][pos] {
                same[slot] += 1;
            } else if first.is_none() {
                first = Some((slot, pos));
                let j = row.iter().zip(&alone[slot][pos]).position(|(a, b)| a != b).unwrap();
                eprintln!("first mismatch: slot={slot}, position={pos}, width={}, row={r}, vocab={j}, alone={:08x}, mixed={:08x}", segs.len(), alone[slot][pos][j], row[j]);
            }
            compared[slot] += 1;
        }
    }
    println!("multi-sequence windows vs A alone: bit-identical rows: {} / {n}", same[0]);
    if check_all {
        for slot in 1..3 {
            println!("multi-sequence slot {slot} vs alone: bit-identical rows: {} / {}", same[slot], compared[slot]);
        }
    }
    // Stop and join the asynchronous driver while its CUDA buffers and context are still alive.
    // process::exit skipped this and could race CUDA shutdown with a final tier promotion.
    drop(eng.driver.take());
    anyhow::ensure!(first.is_none(), "mixed-sequence logits differ first at {first:?}");
    Ok(())
}
