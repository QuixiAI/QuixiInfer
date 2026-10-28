//! `qx-spec-check TOKENS_FILE N_PROMPT N_NEW [K]`: plain decoding against speculative decoding with a prompt-lookup
//! drafter (up to K drafts, default 4), greedy and sampled (fixed seed): the tokens must be identical.
use anyhow::Result;
use qx_engine::engine::{Engine, Sampling};
use qx_pack::gguf::Gguf;

/// The tokens that followed the latest earlier occurrence of the context's last 3 (else 2) tokens.
fn lookup(ctx: &[u32], k: usize) -> Vec<u32> {
    for n in [3usize, 2] {
        if ctx.len() <= n {
            continue;
        }
        let key = &ctx[ctx.len() - n..];
        for i in (0..ctx.len() - n).rev() {
            if &ctx[i..i + n] == key {
                let e = (i + n + k).min(ctx.len());
                if e > i + n {
                    return ctx[i + n..e].to_vec();
                }
            }
        }
    }
    Vec::new()
}

fn main() -> Result<()> {
    qx_gpu::init();
    // let a debugger attach (yama ptrace_scope 1 otherwise allows only ancestors)
    unsafe { libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0) };
    let a: Vec<String> = std::env::args().collect();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model))?;
    let toks: Vec<u32> = std::fs::read(&a[1])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let np: usize = a[2].parse()?;
    let nn: usize = a[3].parse()?;
    let k: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(4);
    let prompt = &toks[..np];
    let (free, _) = qx_gpu::mem::mem_info();
    let budget = free.saturating_sub((6.2 * (1u64 << 30) as f64) as usize);
    let order: Vec<(usize, usize)> = match std::env::var("QX_PROFILE") {
        Ok(p) => std::fs::read_to_string(p)?.lines().map(|l| {
            let v: Vec<usize> = l.split_whitespace().take(2).map(|x| x.parse().unwrap()).collect();
            (v[0], v[1])
        }).collect(),
        Err(_) => (0..512).flat_map(|e| (0..48).map(move |l| (l, e))).collect(),
    };
    let mut eng = Engine::load(&g, (np + nn + 64).next_multiple_of(256), budget, &order)?;
    eng.warm_up();
    eprintln!("VRAM free after load: {:.2} GiB", qx_gpu::mem::mem_info().0 as f64 / 1073741824.0);
    let mut all_ok = true;
    for (name, sp) in [("greedy", Sampling { temperature: 0.0, ..Default::default() }), ("sampled", Sampling { temperature: 0.8, top_p: 0.95, top_k: 40, seed: 1234, ..Default::default() })] {
        let t0 = std::time::Instant::now();
        let mut plain = Vec::new();
        let spec_only = std::env::var("QX_SPEC_ONLY").is_ok();
        let hr = |a: (u64, u64, u64), b: (u64, u64, u64)| format!("{:.1}% hits, {} swaps", 100.0 * (b.0 - a.0) as f64 / (b.1 - a.1).max(1) as f64, b.2 - a.2);
        let c0 = eng.cache_stats();
        if !spec_only {
            eng.spec = false;
            eng.stream(prompt, nn, &sp, &[], |t| {
                plain.push(t);
                true
            });
            eng.reset();
            eng.spec = true;
        }
        let tp = t0.elapsed().as_secs_f64();
        let c1 = eng.cache_stats();
        let t0 = std::time::Instant::now();
        let mut lk = |c: &[u32], k: usize| lookup(c, k);
        let use_mtp = std::env::var("QX_DRAFT").map(|v| v != "lookup").unwrap_or(true);
        let (spec, st) = eng.generate_spec(prompt, nn, &sp, k, &[], if use_mtp { None } else { Some(&mut lk) });
        let ts = t0.elapsed().as_secs_f64();
        let c2 = eng.cache_stats();
        eprintln!("  cache: plain {}, spec {}", hr(c0, c1), hr(c1, c2));
        eng.reset();
        let same = spec_only || plain == spec;
        all_ok &= same;
        let first_diff = plain.iter().zip(&spec).position(|(a, b)| a != b);
        println!("{name}: {} tokens, identical: {same}{}; plain {tp:.2}s, spec {ts:.2}s ({} windows, {} drafts, {} kept, {:.2} tokens/window)", plain.len(), first_diff.map(|i| format!(" (first difference at {i})")).unwrap_or_default(), st[0], st[1], st[2], spec.len() as f64 / st[0].max(1) as f64);
    }
    println!("{}", if all_ok { "PASS" } else { "FAIL" });
    Ok(())
}
