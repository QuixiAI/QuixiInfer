//! `qx-tmv-check`: the tensor-core products (`qx_tmv`) against the f32 ones (`qx_mgemv`) on real weights of every
//! dense type and input mode, for 1 to 16 tokens: relative error (bf16 rounding) must stay under 1e-2.
use qx_engine::ops::{MGemv, Seg};
use qx_gpu::mem::DeviceBuf;
use qx_pack::gguf::Gguf;
use qx_ref::weights::W;

fn main() {
    qx_gpu::init();
    let model = qx_engine::default_model_path();
    let g = Gguf::open(std::path::Path::new(&model)).unwrap();
    let gpu = qx_gpu::Gpu::new();
    // (tensors in one group, planar): types Q6_K / Q8_0 planar, F32, BF16, many and few rows, K 320 .. 10240
    let groups: Vec<(Vec<&str>, bool)> = vec![
        (vec!["blk.0.attn_qkv.weight", "blk.0.ssm_beta.weight"], true),
        (vec!["blk.0.hc_attn_down.weight", "blk.0.hc_attn_inject.weight"], true),
        (vec!["blk.0.hc_attn_up.weight"], true),
        (vec!["blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight"], true),
        (vec!["blk.3.indexer.q_proj.weight"], true),
        (vec!["blk.0.ffn_down_shexp.weight"], true),
    ];
    let mut worst = 0f64;
    for (names, planar) in groups {
        let ws: Vec<(DeviceBuf, W, u64, usize)> = names
            .iter()
            .map(|n| {
                let w = W::new(&g, g.tensor(n).unwrap());
                let (b, qx, st) = if planar { qx_engine::ops::to_planar(w.ty, w.data, w.rows, w.cols) } else { (w.data.to_vec(), qx_engine::ops::qx_type(w.ty), w.row_bytes()) };
                (DeviceBuf::from_slice(&b), w, qx, st)
            })
            .collect();
        let k = ws[0].1.cols;
        let ya: Vec<DeviceBuf> = ws.iter().map(|(_, w, _, _)| DeviceBuf::new(32 * w.rows * 4)).collect();
        let yb: Vec<DeviceBuf> = ws.iter().map(|(_, w, _, _)| DeviceBuf::new(32 * w.rows * 4)).collect();
        let seg = |ys: &[DeviceBuf]| MGemv::new(&ws.iter().zip(ys).map(|((b, w, qx, st), y)| Seg { w: b.ptr, y: y.ptr, rows: w.rows, cols: w.cols, qx: *qx, stride: *st }).collect::<Vec<_>>());
        let (ma, mb) = (seg(&ya), seg(&yb));
        let xs: Vec<f32> = (0..32 * k).map(|i| ((i as f32) * 0.0137).sin() * (1.0 + (i % 7) as f32 * 0.3)).collect();
        let x = DeviceBuf::from_slice(&xs);
        let mut gworst = vec![0f64; ws.len()];
        let x2 = DeviceBuf::from_slice(&xs.iter().map(|v| v.cos()).collect::<Vec<f32>>());
        for mode in 0..3u64 {
            let arg = match mode {
                1 => (0.5f32).to_bits() as u64,
                2 => x2.ptr,
                _ => 0,
            };
            for t in [1usize, 2, 7, 8, 9, 16, 17, 24, 32] {
                // reference: f32 mgemv, chunks of 8
                ma.run_t_with(&gpu, x.ptr, mode, arg, t, k, false);
                mb.run_t_with(&gpu, x.ptr, mode, arg, t, k, true);
                gpu.sync();
                let mut err = 0f64;
                for (i, (_, w, _, _)) in ws.iter().enumerate() {
                    let a: Vec<f32> = ya[i].download(t * w.rows);
                    let b: Vec<f32> = yb[i].download(t * w.rows);
                    let (mut num, mut den) = (0f64, 0f64);
                    for (p, q) in a.iter().zip(&b) {
                        num += ((p - q) as f64).powi(2);
                        den += (*p as f64).powi(2);
                    }
                    let e = (num / den.max(1e-30)).sqrt();
                    gworst[i] = gworst[i].max(e);
                    err = err.max(e);
                }
                worst = worst.max(err);
                if err > 1e-2 {
                    let a: Vec<f32> = ya[0].download(64);
                    let b: Vec<f32> = yb[0].download(64);
                    println!("{:?} mode {mode} T {t}: rel err {err:.2e}; f32 {:?} tmv {:?} | rows 16.. f32 {:?} tmv {:?}", names, &a[..3], &b[..3], &a[16..19], &b[16..19]);
                }
            }
        }
        println!("{:?} (K {k}): worst rel err {:?}", names, gworst.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>());
    }
    println!("worst rel err {worst:.2e}: {}", if worst < 1e-2 { "PASS" } else { "FAIL" });
}
