//! `qx-ds-replay PACK TRACE CTX SLOTS`: re-executes an engine schedule recorded with `QX_DS_SCHED_TRACE` (the server's
//! exact sequence of resets, rewinds, windows and chunks, warm-up included) and compares each forward's per-row logits
//! hashes with the recorded ones. A replay that matches everywhere means the recorded run's bits are a function of
//! its schedule alone (so a wrong token in it is a deterministic, bisectable defect); a mismatch means the recorded
//! run depended on something outside the schedule (timing, the tiers' state).
//! `QX_DS_REPLAY_SPLIT=1`: every window is replayed as single-token steps in the same order instead; the hashes
//! must still match (windows are bit-identical to single steps), and the first mismatch names the window whose
//! rows differ from single steps. Budgets: `QX_DS_VRAM` (21) / `QX_DS_RAM` (95), as the server's.
use anyhow::Result;
use qx_engine::dsv41::{row_hashes, DsEngine};
use qx_pack::gguf::Gguf;

fn main() -> Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "usage: qx-ds-replay PACK TRACE CTX SLOTS");
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let trace = std::fs::read_to_string(&a[2])?;
    let (ctx, slots): (usize, usize) = (a[3].parse()?, a[4].parse()?);
    let split = std::env::var("QX_DS_REPLAY_SPLIT").is_ok_and(|v| v == "1");
    qx_engine::dsv41::pin_engine_early();
    let gib = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let prof = std::env::var("QX_DS_PROFILE").unwrap_or_else(|_| "golden/profile-dsv41.txt".into());
    let mut eng = DsEngine::load_tiered_slots(&g, ctx, gib("QX_DS_VRAM", 21.0), gib("QX_DS_RAM", 95.0), Some(std::path::Path::new(&prof)), slots)?;
    let mut last: Option<(usize, String, String)> = None; // (line, the call, the replay's hashes)
    let (mut calls, mut rows, mut bad_calls, mut bad_rows) = (0usize, 0usize, 0usize, 0usize);
    let mut first: Option<String> = None;
    for (ln, line) in trace.lines().enumerate() {
        let (op, rest) = line.split_once(' ').unwrap_or((line, ""));
        match op {
            "R" => eng.reset(rest.trim().parse()?),
            "W" => {
                let (s, p) = rest.split_once(' ').unwrap();
                eng.rewind(s.parse()?, p.parse()?);
            }
            "L" => eng.chunk_release(),
            "F" => {
                let (kv, segs) = rest.split_once(' ').unwrap();
                let kv = kv == "1";
                let segs: Vec<(usize, u32)> = segs.split(',').map(|x| { let (s, t) = x.split_once(':').unwrap(); (s.parse().unwrap(), t.parse().unwrap()) }).collect();
                let h = if split && !kv {
                    segs.iter().map(|&seg| row_hashes(&eng.forward_segments_opt(&[seg], None, false), 1)).collect::<Vec<_>>().join(" ")
                } else {
                    row_hashes(&eng.forward_segments_opt(&segs, None, kv), segs.len())
                };
                last = Some((ln + 1, format!("F width {} kv {}", segs.len(), kv as u8), h));
            }
            "C" => {
                let mut it = rest.splitn(3, ' ');
                let (slot, kv, toks) = (it.next().unwrap().parse()?, it.next().unwrap() == "1", it.next().unwrap());
                let toks: Vec<u32> = toks.split(',').map(|t| t.parse().unwrap()).collect();
                let h = row_hashes(&eng.forward_chunk_opt(slot, &toks, kv), 1);
                last = Some((ln + 1, format!("C slot {slot} rows {} kv {}", toks.len(), kv as u8), h));
            }
            "K" => {
                let mut it = rest.splitn(3, ' ');
                let (slot, keep_from, toks): (usize, usize, &str) = (it.next().unwrap().parse()?, it.next().unwrap().parse()?, it.next().unwrap());
                let toks: Vec<u32> = toks.split(',').map(|t| t.parse().unwrap()).collect();
                let h = row_hashes(&eng.forward_chunk_keep(slot, &toks, keep_from), 1);
                last = Some((ln + 1, format!("K slot {slot} rows {} full stack from {keep_from}", toks.len()), h));
            }
            "H" => {
                let Some((l0, call, mine)) = last.take() else { continue };
                calls += 1;
                let (rec, mine): (Vec<&str>, Vec<&str>) = (rest.split_whitespace().collect(), mine.split_whitespace().collect());
                rows += rec.len();
                let diff: Vec<usize> = (0..rec.len().max(mine.len())).filter(|&i| rec.get(i) != mine.get(i)).collect();
                if !diff.is_empty() {
                    bad_calls += 1;
                    bad_rows += diff.len();
                    if first.is_none() {
                        first = Some(format!("trace line {l0}: {call}: rows {diff:?} of {} differ", rec.len()));
                    }
                }
            }
            _ => {}
        }
    }
    println!("replay{}: {calls} forwards, {rows} rows; mismatching forwards {bad_calls}, rows {bad_rows}{}", if split { " (split into single steps)" } else { "" }, first.map(|f| format!("; first: {f}")).unwrap_or_default());
    drop(eng);
    std::process::exit(if bad_calls == 0 { 0 } else { 1 });
}
