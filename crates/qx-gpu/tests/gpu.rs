//! GPU tests. Run with `cargo test -p qx-gpu --release --test gpu`. Needs the RTX 5090.
use qx_gpu::kernel::modules::SMOKE;
use qx_gpu::{reference as r, Gpu};
use cubecl::prelude::CubeDim;

type Test = fn(&Gpu) -> Result<(), String>;

fn add_scalar(gpu: &Gpu) -> Result<(), String> {
    let n = 3000usize;
    let x = gpu.upload(&vec![1f32; n]);
    gpu.launch(SMOKE.kernel("qx_add_scalar", CubeDim::new_1d(256)), ((n as u32).div_ceil(256), 1, 1), &[&x], &[n as u64, 4]);
    let v: Vec<f32> = gpu.download(&x);
    if v.iter().all(|&e| e == 5.0) { Ok(()) } else { Err(format!("x[0]={}", v[0])) }
}

fn convert_roundtrip(gpu: &Gpu) -> Result<(), String> {
    // Every E4M3 code, plus values around them, plus out-of-range values.
    let mut xs: Vec<f32> = (0u8..=255).filter(|&c| c & 0x7f != 0x7f).map(r::e4m3_to_f32).collect();
    for i in 0..4000 {
        let t = (i as f32 * 0.37).sin() * 2f32.powf((i % 40) as f32 * 0.25 - 6.0) * 3.0;
        xs.push(t);
    }
    xs.extend([500.0, -1000.0, 1e-9, -3.3e-3, 447.0, 449.0]);
    let n = xs.len();
    let x = gpu.upload(&xs);
    let out = gpu.alloc(n * 3 * 4);
    gpu.launch(SMOKE.kernel("qx_convert_roundtrip", CubeDim::new_1d(128)), ((n as u32).div_ceil(128), 1, 1), &[&x, &out], &[n as u64]);
    let o: Vec<f32> = gpu.download(&out);
    for (i, &v) in xs.iter().enumerate() {
        let want = [r::e4m3_to_f32(r::f32_to_e4m3(v)), r::round_bf16(v), r::round_f16(v)];
        for k in 0..3 {
            let got = o[3 * i + k];
            if got.to_bits() != want[k].to_bits() && !(got == 0.0 && want[k] == 0.0) {
                return Err(format!("x={v:e} conversion {k}: got {got:e}, want {:e}", want[k]));
            }
        }
    }
    Ok(())
}

fn math_bitwise(gpu: &Gpu) -> Result<(), String> {
    let xs: Vec<f32> = (0..200_000).map(|i| -90.0 + i as f32 * 0.00091).collect();
    let n = xs.len();
    let x = gpu.upload(&xs);
    let out = gpu.alloc(2 * n * 4);
    gpu.launch(SMOKE.kernel("qx_math_check", CubeDim::new_1d(256)), ((n as u32).div_ceil(256), 1, 1), &[&x, &out], &[n as u64]);
    let o: Vec<f32> = gpu.download(&out);
    for (i, &v) in xs.iter().enumerate() {
        let (e, s) = (qx_cpu::math::qx_expf(v), qx_cpu::math::qx_silu_mul(v, 1.5));
        if o[i].to_bits() != e.to_bits() || o[n + i].to_bits() != s.to_bits() {
            return Err(format!("x={v}: gpu exp {} silu {}, cpu exp {e} silu {s}", o[i], o[n + i]));
        }
    }
    Ok(())
}

fn main() {
    qx_gpu::init();
    let gpu = Gpu::new();
    let tests: &[(&str, Test)] = &[("add_scalar", add_scalar), ("convert_roundtrip", convert_roundtrip), ("math_bitwise", math_bitwise)];
    let filter = std::env::args().skip(1).find(|a| !a.starts_with('-'));
    let mut failed = 0;
    for (name, t) in tests {
        if filter.as_deref().is_some_and(|f| !name.contains(f)) {
            continue;
        }
        match t(&gpu) {
            Ok(()) => println!("test {name} ... ok"),
            Err(e) => {
                failed += 1;
                println!("test {name} ... FAILED: {e}");
            }
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
