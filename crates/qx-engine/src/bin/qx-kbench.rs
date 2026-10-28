//! Kernel microbenchmarks on real weights: effective bandwidth of each launch shape the engine uses.
//! `qx-kbench [FILTER]`.
use cubecl::prelude::CubeDim;
use qx_engine::ops::{MGemv, Seg};
use qx_gpu::kernel::modules;
use qx_gpu::mem::DeviceBuf;
use qx_gpu::Gpu;
use qx_pack::gguf::Gguf;
use qx_ref::weights::W;

fn time<F: FnMut()>(gpu: &Gpu, reps: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    gpu.sync();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        f();
    }
    gpu.sync();
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    qx_gpu::init();
    let filter = std::env::args().nth(1);
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model)).unwrap();
    let gpu = Gpu::new();
    let up = |n: &str| {
        let w = W::new(&g, g.tensor(n).unwrap());
        (DeviceBuf::from_slice(w.data), w)
    };
    let x = DeviceBuf::from_slice(&(0..16 * 16384).map(|i| (i as f32 * 0.37).sin()).collect::<Vec<f32>>());
    // QX_KB_T: tokens per launch for the group timings
    let kt: usize = std::env::var("QX_KB_T").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let ys: Vec<DeviceBuf> = (0..8).map(|_| DeviceBuf::new(16 * 300_000 * 4)).collect();
    let groups: Vec<(&str, Vec<&str>)> = vec![
        ("gdn in (Q6_K 10240+6144, F32 48+48)", vec!["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"]),
        ("hc down + inject (Q8_0 320x10240, F32 4x10240)", vec!["blk.0.hc_attn_down.weight", "blk.0.hc_attn_inject.weight"]),
        ("hc up (Q8_0 10240x320)", vec!["blk.0.hc_attn_up.weight"]),
        ("moe in (F32 512, Q6_K 640+640, F32 1)", vec!["blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_gate_inp_shexp.weight"]),
        ("ssm_out (Q6_K 2560x6144)", vec!["blk.0.ssm_out.weight"]),
        ("attn in (Q6_K 12288+512+512, BF16 128+512)", vec!["blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight", "blk.3.indexer.k_proj.weight", "blk.3.indexer.q_proj.weight"]),
        ("head (Q6_K 248320x2560)", vec!["output.weight"]),
    ];
    // QX_KB_T tokens through the GDN prefill recurrence, token by token and chunked: time and agreement
    if filter.as_deref().is_some_and(|f| f == "gdn") {
        let (kh, vh, d, t) = (16usize, 48usize, 128usize, kt.max(64));
        let ch = 2 * kh * d + vh * d;
        let mut cv: Vec<f32> = (0..t * ch).map(|i| ((i as f32) * 0.000731).sin()).collect();
        // q and k rows unit length, as the conv's l2 norm leaves them
        for tt in 0..t {
            for hh in 0..2 * kh {
                let r = &mut cv[tt * ch + hh * d..tt * ch + hh * d + d];
                let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
                r.iter_mut().for_each(|x| *x /= n);
            }
        }
        let dcv = DeviceBuf::from_slice(&cv);
        let beta = DeviceBuf::from_slice(&(0..t * vh).map(|i| ((i as f32) * 0.37).sin()).collect::<Vec<f32>>());
        let alpha = DeviceBuf::from_slice(&(0..t * vh).map(|i| ((i as f32) * 0.11).cos() - 2.0).collect::<Vec<f32>>());
        let dtb = DeviceBuf::from_slice(&vec![0.5f32; vh]);
        let a = DeviceBuf::from_slice(&vec![-0.3f32; vh]);
        let s0: Vec<f32> = (0..vh * d * d).map(|i| ((i as f32) * 0.0013).sin() * 0.05).collect();
        let mut outs = Vec::new();
        for (name, threads) in [("qxp_gdn_recur", 128u32), ("qxp_gdn_chunk", 256)] {
            let st = DeviceBuf::from_slice(&s0);
            let o = DeviceBuf::new(t * vh * d * 4);
            let info = [dcv.ptr, st.ptr, o.ptr, beta.ptr, alpha.ptr, dtb.ptr, a.ptr, kh as u64, d as u64, t as u64, ch as u64, vh as u64];
            gpu.run(modules::PREFILL.kernel(name, CubeDim::new_1d(threads)), (vh as u32, 4, 1), &info);
            gpu.sync();
            outs.push((o.download::<f32>(t * vh * d), st.download::<f32>(vh * d * d)));
            let dt = time(&gpu, 3, || gpu.run(modules::PREFILL.kernel(name, CubeDim::new_1d(threads)), (vh as u32, 4, 1), &info));
            println!("| {name} T={t} | {:.2} ms | {:.2} us/token |", dt * 1e3, dt * 1e6 / t as f64);
        }
        for (scan, splits) in [("qxp_gdn_scan", 4u32)] {
            let st = DeviceBuf::from_slice(&s0);
            let o = DeviceBuf::new(t * vh * d * 4);
            let (u, w, p, gg) = (DeviceBuf::new(t * vh * d * 4), DeviceBuf::new(t * vh * d * 4), DeviceBuf::new(t.div_ceil(32) * vh * 1024 * 4), DeviceBuf::new(t * vh * 4));
            let run = || {
                gpu.run(modules::PREFILL.kernel("qxp_gdn_prep", CubeDim::new_1d(256)), (t.div_ceil(32) as u32, vh as u32, 1), &[dcv.ptr, u.ptr, w.ptr, p.ptr, gg.ptr, beta.ptr, alpha.ptr, dtb.ptr, a.ptr, kh as u64, d as u64, t as u64, ch as u64, vh as u64]);
                gpu.run(modules::PREFILL.kernel(scan, CubeDim::new_1d(256)), (vh as u32, splits, 1), &[dcv.ptr, st.ptr, o.ptr, u.ptr, w.ptr, p.ptr, gg.ptr, kh as u64, d as u64, t as u64, ch as u64, vh as u64]);
            };
            run();
            gpu.sync();
            outs.push((o.download::<f32>(t * vh * d), st.download::<f32>(vh * d * d)));
            st.upload(&s0);
            let dt = time(&gpu, 3, run);
            println!("| qxp_gdn_prep + {scan} T={t} | {:.2} ms | {:.2} us/token |", dt * 1e3, dt * 1e6 / t as f64);
        }
        let rel = |a: &[f32], b: &[f32]| (a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
        println!("chunked vs token by token: outputs rel {:.2e}, state rel {:.2e}", rel(&outs[0].0, &outs[1].0), rel(&outs[0].1, &outs[1].1));
        for k in 2..outs.len() {
            let per = vh * d;
            if let Some(tt) = (0..t).find(|&tt| rel(&outs[0].0[tt * per..(tt + 1) * per], &outs[k].0[tt * per..(tt + 1) * per]) > 1e-4) {
                let bad: Vec<usize> = (0..per).filter(|&q| (outs[0].0[tt * per + q] - outs[k].0[tt * per + q]).abs() > 1e-3 * outs[0].0[tt * per + q].abs().max(1e-3)).take(8).collect();
                println!("two-pass ({k}): first bad token {tt}, (head, column) {:?}", bad.iter().map(|q| (q / d, q % d)).collect::<Vec<_>>());
            }
            println!("two-pass ({k}) vs token by token: outputs rel {:.2e}, state rel {:.2e}", rel(&outs[0].0, &outs[k].0), rel(&outs[0].1, &outs[k].1));
        }
        return;
    }
    println!("| launch | us | GB/s |\n|---|---|---|");
    if filter.as_deref().is_some_and(|f| f == "gemm") {
        let cases = [("blk.0.attn_qkv.weight", true), ("blk.0.hc_attn_down.weight", true), ("blk.0.hc_attn_up.weight", true), ("blk.0.ffn_gate_inp.weight", false), ("output.weight", true)];
        println!("| GEMM | T | rel err vs GEMV | ms | TFLOPS |\n|---|---|---|---|---|");
        for (name, planar) in cases {
            let w = W::new(&g, g.tensor(name).unwrap());
            let rows = w.rows.min(16384);
            let data = &w.data[..rows * w.row_bytes()];
            let (bytes, qx, stride) = if planar { qx_engine::ops::to_planar(w.ty, data, rows, w.cols) } else { (data.to_vec(), qx_engine::ops::qx_type(w.ty), w.row_bytes()) };
            let dw = DeviceBuf::from_slice(&bytes);
            for t in [256usize, 2048, 8192] {
                let xs: Vec<f32> = (0..t * w.cols).map(|i| ((i as f32) * 0.000317).sin() * (1.0 + (i % 13) as f32 * 0.1)).collect();
                let dx = DeviceBuf::from_slice(&xs);
                let dy = DeviceBuf::new(t * rows * 4);
                let kname = if std::env::var("QX_GEMM2").is_ok_and(|v| v == "0") { "qx_gemm" } else { "qx_gemm2" };
                let run = || gpu.run(modules::GEMM.kernel(kname, CubeDim::new_1d(256)), ((rows as u32).div_ceil(64), (t as u32).div_ceil(128), 1), &[dx.ptr, dw.ptr, dy.ptr, t as u64, rows as u64, w.cols as u64, qx, stride as u64, rows as u64, 0]);
                let dt = time(&gpu, 5, run);
                let y: Vec<f32> = dy.download(t * rows);
                // reference: per-token GEMV (f32) on a few tokens
                let seg = MGemv::new(&[Seg { w: dw.ptr, y: ys[0].ptr, rows, cols: w.cols, qx, stride }]);
                let mut num = 0f64;
                let mut den = 0f64;
                for tt in [0usize, t / 3, t - 1] {
                    let xt = DeviceBuf::from_slice(&xs[tt * w.cols..(tt + 1) * w.cols]);
                    seg.run(&gpu, xt.ptr);
                    gpu.sync();
                    let r: Vec<f32> = ys[0].download(rows);
                    for n in 0..rows {
                        num += ((y[tt * rows + n] - r[n]) as f64).powi(2);
                        den += (r[n] as f64).powi(2);
                    }
                }
                println!("| {name} ({rows}x{}) | {t} | {:.1e} | {:.3} | {:.1} |", w.cols, (num / den).sqrt(), dt * 1e3, 2.0 * (t * rows * w.cols) as f64 / dt / 1e12);
            }
        }
        return;
    }
    if filter.as_deref().is_none_or(|f| "probe".contains(f)) {
        let n = 1usize << 30;
        let src = DeviceBuf::new(n);
        let out = DeviceBuf::new(4);
        for blocks in [680u32, 1360, 2720, 5440] {
            let t = time(&gpu, 20, || gpu.run(modules::SMOKE.kernel("qx_read_probe", CubeDim::new_1d(256)), (blocks, 1, 1), &[src.ptr, (n / 16) as u64, out.ptr]));
            println!("| read probe 1 GiB, {blocks} blocks | {:.0} | {:.0} |", t * 1e6, n as f64 / t / 1e9);
        }
    }
    // Out of L2: the same launch shape over every GDN layer's weights in turn.
    if filter.as_deref().is_none_or(|f| "dram".contains(f)) {
        let gdn: Vec<usize> = (0..48).filter(|l| (l + 1) % 4 != 0).collect();
        let mut all = Vec::new();
        let mut bytes = 0usize;
        let mut keep: Vec<Vec<(DeviceBuf, W, u64, usize)>> = Vec::new();
        for &l in &gdn {
            let names = [format!("blk.{l}.attn_qkv.weight"), format!("blk.{l}.attn_gate.weight"), format!("blk.{l}.ssm_beta.weight"), format!("blk.{l}.ssm_alpha.weight")];
            let planar = std::env::var("QX_PLANAR").is_ok_and(|v| v == "1");
            let ws: Vec<(DeviceBuf, W, u64, usize)> = names
                .iter()
                .map(|t| {
                    let w = W::new(&g, g.tensor(t).unwrap());
                    let (bts, qx, st) = if planar { qx_engine::ops::to_planar(w.ty, w.data, w.rows, w.cols) } else { (w.data.to_vec(), qx_engine::ops::qx_type(w.ty), 0) };
                    (DeviceBuf::from_slice(&bts), w, qx, st)
                })
                .collect();
            bytes += ws.iter().map(|(_, w, _, _)| w.data.len()).sum::<usize>();
            let segs: Vec<Seg> = ws.iter().zip(&ys).map(|((b, w, qx, st), y)| Seg { w: b.ptr, y: y.ptr, rows: w.rows, cols: w.cols, qx: *qx, stride: *st }).collect();
            all.push(MGemv::new(&segs));
            keep.push(ws);
        }
        let t = time(&gpu, 20, || {
            for m in &all {
                m.run(&gpu, x.ptr);
            }
        });
        println!("| gdn in x36 layers (DRAM) | {:.1} per layer | {:.0} |", t * 1e6 / 36.0, bytes as f64 / t / 1e9);
        // The same with the next layer's weights prefetched into L2 before each layer's launch, and a small
        // dependent kernel in between, as in a real layer.
        let ranges: Vec<DeviceBuf> = keep.iter().map(|ws| DeviceBuf::from_slice(&ws.iter().flat_map(|(b, w, _, _)| [b.ptr, w.data.len() as u64]).collect::<Vec<u64>>())).collect();
        let pf = |i: usize| gpu.run(modules::SMOKE.kernel("qx_prefetch_l2", CubeDim::new_1d(256)), (64, 1, 1), &[ranges[i].ptr, 4]);
        let small = || gpu.run(modules::SMOKE.kernel("qx_add_scalar", CubeDim::new_1d(256)), (1, 1, 1), &[ys[7].ptr, 0, 0]);
        let t_plain = time(&gpu, 20, || {
            for m in &all {
                for _ in 0..10 { small(); }
                m.run(&gpu, x.ptr);
            }
        });
        let t_pf = time(&gpu, 20, || {
            for (i, m) in all.iter().enumerate() {
                if i + 1 < all.len() { pf(i + 1); }
                for _ in 0..10 { small(); }
                m.run(&gpu, x.ptr);
            }
        });
        println!("| gdn in + 10 small kernels, plain | {:.1} per layer | |", t_plain * 1e6 / 36.0);
        println!("| gdn in + 10 small kernels, next layer prefetched | {:.1} per layer | |", t_pf * 1e6 / 36.0);
    }
    for (name, ts) in groups {
        if filter.as_deref().is_some_and(|f| !name.contains(f)) {
            continue;
        }
        // as the engine holds them: Q6_K and Q8_0 in the planar layout
        let ws: Vec<(DeviceBuf, W, u64, usize)> = ts
            .iter()
            .map(|t| {
                let w = W::new(&g, g.tensor(t).unwrap());
                let (bts, qx, st) = qx_engine::ops::to_planar(w.ty, w.data, w.rows, w.cols);
                (DeviceBuf::from_slice(&bts), w, qx, st)
            })
            .collect();
        let bytes: usize = ws.iter().map(|(_, w, _, _)| w.data.len()).sum();
        let segs: Vec<Seg> = ws.iter().zip(&ys).map(|((b, w, qx, st), y)| Seg { w: b.ptr, y: y.ptr, rows: w.rows, cols: w.cols, qx: *qx, stride: *st }).collect();
        let mg = MGemv::new(&segs);
        let t = time(&gpu, 200, || mg.run_t(&gpu, x.ptr, 0, 0, kt, 16384));
        println!("| {name} | {:.1} | {:.0} |", t * 1e6, bytes as f64 / t / 1e9);
    }
    // Resident experts of one layer: 10 selected, all resident.
    if filter.as_deref().is_none_or(|f| "experts".contains(f)) {
        let l = 0;
        let (gt, gb) = { let w = W::new(&g, g.tensor(&format!("blk.{l}.ffn_gate_exps.weight")).unwrap()); (w.ty, w.row_bytes()) };
        let dw = W::new(&g, g.tensor(&format!("blk.{l}.ffn_down_exps.weight")).unwrap());
        let gw = W::new(&g, g.tensor(&format!("blk.{l}.ffn_gate_exps.weight")).unwrap());
        let uw = W::new(&g, g.tensor(&format!("blk.{l}.ffn_up_exps.weight")).unwrap());
        let (ff, e) = (640usize, 2560usize);
        let gbytes = ff * gb;
        let dbytes = e * dw.row_bytes();
        let rec = 2 * gbytes + dbytes;
        let arena = DeviceBuf::new(10 * rec);
        let mut slots = Vec::new();
        for k in 0..10 {
            let ex = 37 * k + 5;
            let mut v = Vec::new();
            v.extend_from_slice(&gw.data[ex * gbytes..(ex + 1) * gbytes]);
            v.extend_from_slice(&uw.data[ex * gbytes..(ex + 1) * gbytes]);
            v.extend_from_slice(&dw.data[ex * dbytes..(ex + 1) * dbytes]);
            arena.upload_at(k * rec, &v);
            slots.push(arena.ptr + (k * rec) as u64);
        }
        let ds = DeviceBuf::from_slice(&slots);
        let wts = DeviceBuf::from_slice(&[0.1f32; 10]);
        let gu = DeviceBuf::new(10 * 2 * ff * 4);
        let yk = DeviceBuf::new(10 * e * 4);
        let entry = if gt == qx_pack::ggml::GgmlType::IQ3_S { "qx_moe_gateup_iq3_s" } else { "qx_moe_gateup_iq2_s" };
        let t1 = time(&gpu, 200, || gpu.run(modules::MOE.kernel(entry, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), 10, 1), &[ds.ptr, 0, 0, x.ptr, gu.ptr, ff as u64, e as u64, gb as u64]));
        let t2 = time(&gpu, 200, || gpu.run(modules::MOE.kernel("qx_moe_down_iq4_nl", CubeDim::new_1d(256)), ((e as u32).div_ceil(8), 10, 1), &[ds.ptr, 0, 0, wts.ptr, gu.ptr, yk.ptr, ff as u64, e as u64, (2 * gbytes) as u64, dw.row_bytes() as u64]));
        println!("| experts gate+up x10 f32 ({gt:?}) | {:.1} | {:.0} |", t1 * 1e6, (20 * gbytes) as f64 / t1 / 1e9);
        println!("| experts down x10 f32 (IQ4_NL) | {:.1} | {:.0} |", t2 * 1e6, (10 * dbytes) as f64 / t2 / 1e9);
        let xq = DeviceBuf::new(e);
        let xd = DeviceBuf::new(e / 256 * 4);
        let hq = DeviceBuf::new(10 * ff);
        let hd = DeviceBuf::new(10 * ff / 32 * 4);
        let t0 = time(&gpu, 200, || gpu.run(modules::MOE.kernel("qx_quant_q8k", CubeDim::new_1d(256)), ((e / 256) as u32, 1, 1), &[x.ptr, xq.ptr, xd.ptr]));
        let qentry = if gt == qx_pack::ggml::GgmlType::IQ3_S { "qx_moe_gateup_q8k_iq3_s" } else { "qx_moe_gateup_q8k_iq2_s" };
        let t3 = time(&gpu, 200, || gpu.run(modules::MOE.kernel(qentry, CubeDim::new_1d(256)), (((2 * ff) as u32).div_ceil(8), 10, 1), &[ds.ptr, xq.ptr, xd.ptr, 0, gu.ptr, ff as u64, e as u64, gb as u64]));
        let t4 = time(&gpu, 200, || gpu.run(modules::MOE.kernel("qx_moe_hq", CubeDim::new_1d(ff as u32)), (10, 1, 1), &[ds.ptr, gu.ptr, hq.ptr, hd.ptr, ff as u64]));
        let t5 = time(&gpu, 200, || gpu.run(modules::MOE.kernel("qx_moe_down_q80_iq4_nl", CubeDim::new_1d(256)), ((e as u32).div_ceil(16), 10, 1), &[ds.ptr, hq.ptr, hd.ptr, 0, yk.ptr, ff as u64, e as u64, (2 * gbytes) as u64, dw.row_bytes() as u64]));
        println!("| quant q8k | {:.1} | |", t0 * 1e6);
        println!("| experts gate+up x10 int8 ({gt:?}) | {:.1} | {:.0} |", t3 * 1e6, (20 * gbytes) as f64 / t3 / 1e9);
        println!("| experts hq x10 | {:.1} | |", t4 * 1e6);
        println!("| experts down x10 int8 (IQ4_NL) | {:.1} | {:.0} |", t5 * 1e6, (10 * dbytes) as f64 / t5 / 1e9);
    }
}
