//! GPU kernels against the CPU reference on real model tensors.
//! `cargo test -p qx-engine --release --test parity [-- FILTER]`; needs the GPU and `QX_MODEL` (shard 1 of the GGUF).
use qx_engine::ops;
use qx_gpu::mem::DeviceBuf;
use qx_gpu::Gpu;
use qx_pack::gguf::Gguf;
use qx_ref::weights::W;

fn model() -> Gguf {
    let p = qx_engine::default_model_path();
    Gguf::open(std::path::Path::new(&p)).expect("open model")
}

const SAMPLES: &[&str] = &[
    "blk.0.ssm_alpha.weight",       // F32
    "blk.3.indexer.q_proj.weight",  // BF16
    "blk.0.ffn_down_shexp.weight",  // Q8_0, 640 cols
    "blk.0.hc_attn_up.weight",      // Q8_0, 320 cols
    "token_embd.weight",            // Q6_K
    "blk.0.ffn_down_exps.weight",   // IQ4_NL, 640 cols
    "blk.2.ffn_up_exps.weight",     // IQ3_S
    "blk.0.ffn_gate_exps.weight",   // IQ2_S
];

fn dequant(gpu: &Gpu, g: &Gguf) -> Result<(), String> {
    for name in SAMPLES {
        let t = g.tensor(name).unwrap();
        let w = W::new(g, t);
        let rows = w.rows.min(700);
        let dev = DeviceBuf::from_slice(&w.data[..rows * w.row_bytes()]);
        let out = DeviceBuf::new(rows * w.cols * 4);
        ops::dequant(gpu, t.ty, dev.ptr, out.ptr, rows, w.cols, 0, 0);
        gpu.sync();
        let got: Vec<f32> = out.download(rows * w.cols);
        let mut want = vec![0f32; w.cols];
        for r in 0..rows {
            w.row(r, &mut want);
            for i in 0..w.cols {
                if got[r * w.cols + i].to_bits() != want[i].to_bits() {
                    return Err(format!("{name} {:?} row {r} col {i}: gpu {} cpu {}", t.ty, got[r * w.cols + i], want[i]));
                }
            }
        }
        println!("  dequant {:7?} {name}: {rows} rows bit-exact", t.ty);
    }
    Ok(())
}

fn gemv(gpu: &Gpu, g: &Gguf) -> Result<(), String> {
    for name in SAMPLES {
        let t = g.tensor(name).unwrap();
        let w = W::new(g, t);
        let rows = w.rows.min(4096);
        let x: Vec<f32> = (0..w.cols).map(|i| ((i as f32 * 0.731).sin() + 0.1 * (i % 7) as f32) * 0.5).collect();
        let dev = DeviceBuf::from_slice(&w.data[..rows * w.row_bytes()]);
        let dx = DeviceBuf::from_slice(&x);
        let dy = DeviceBuf::new(rows * 4);
        ops::gemv(gpu, t.ty, dev.ptr, dx.ptr, dy.ptr, rows, w.cols, ops::GEMV_STORE);
        gpu.sync();
        let got: Vec<f32> = dy.download(rows);
        let mut want = vec![0f32; rows];
        w.matvec_rows(0, &x, &mut want);
        let num: f64 = got.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
        let rel = num / den;
        println!("  gemv    {:7?} {name}: {rows} x {} rel err {rel:.2e}", t.ty, w.cols);
        if rel > 1e-5 {
            return Err(format!("{name}: rel err {rel:e}"));
        }
    }
    Ok(())
}

type Test = fn(&Gpu, &Gguf) -> Result<(), String>;

fn main() {
    qx_gpu::init();
    let gpu = Gpu::new();
    let g = model();
    let tests: &[(&str, Test)] = &[("dequant", dequant), ("gemv", gemv)];
    let filter = std::env::args().skip(1).find(|a| !a.starts_with('-'));
    let mut failed = 0;
    for (name, t) in tests {
        if filter.as_deref().is_some_and(|f| !name.contains(f)) {
            continue;
        }
        match t(&gpu, &g) {
            Ok(()) => println!("test {name} ... ok"),
            Err(e) => {
                failed += 1;
                println!("test {name} ... FAILED: {e}");
            }
        }
    }
    std::process::exit(if failed > 0 { 1 } else { 0 });
}
