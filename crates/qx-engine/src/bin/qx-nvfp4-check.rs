//! The GPU's NVFP4 expert kernels against the CPU pool on random expert records: the Q8 paths (what decode
//! runs) must be bit-equal between the tiers, the f32 paths close to an f64 reference.
use cubecl::prelude::CubeDim;
use qx_cpu::act::quantize_q8_k;
use qx_cpu::pool::{Expert, GateFmt, Pool};
use qx_gpu::kernel::modules::MOE;
use qx_gpu::mem::DeviceBuf;
use qx_gpu::Gpu;

const E: usize = 2560;
const FF: usize = 640;
const NE: usize = 6;
const K: usize = 4;
const T: usize = 3;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn main() {
    qx_gpu::init();
    let gpu = Gpu::new();
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let gate_row = E / 64 * 36;
    let down_row = FF / 64 * 36;
    let (gb, db) = (FF * gate_row, E * down_row);
    let tail = 2 * gb + db;
    let rec = (tail + 16).next_multiple_of(4096);
    let mut records = vec![0u8; rec * NE];
    let mut scales = vec![[1f32; 3]; NE];
    for e in 0..NE {
        let r = &mut records[e * rec..(e + 1) * rec];
        for b in r[..tail].chunks_exact_mut(36) {
            for (i, v) in b.iter_mut().enumerate() {
                *v = if i < 4 { 0x20 + (rng(&mut s) % 0x30) as u8 } else { rng(&mut s) as u8 };
            }
        }
        for p in 0..3 {
            scales[e][p] = 0.5 + (rng(&mut s) % 1000) as f32 * 1e-3;
            r[tail + 4 * p..tail + 4 * p + 4].copy_from_slice(&scales[e][p].to_le_bytes());
        }
    }
    let drec = DeviceBuf::from_slice(&records);
    let a = T * K;
    let ids: Vec<usize> = (0..a).map(|_| rng(&mut s) as usize % NE).collect();
    let slots: Vec<u64> = ids.iter().map(|&e| drec.ptr + (e * rec) as u64).collect();
    let dslots = DeviceBuf::from_slice(&slots);
    let xs: Vec<Vec<f32>> = (0..T).map(|_| (0..E).map(|_| ((rng(&mut s) % 20001) as f32 - 10000.0) * 1e-4).collect()).collect();
    let mut xq = Vec::new();
    let mut xd = Vec::new();
    for x in &xs {
        let mut q = Vec::new();
        quantize_q8_k(x, &mut q);
        for b in &q {
            xq.extend_from_slice(&b.qs);
            xd.push(b.d);
        }
    }
    let dxq = DeviceBuf::from_slice(&xq);
    let dxd = DeviceBuf::from_slice(&xd);
    let dx = DeviceBuf::from_slice(&xs.concat());
    let gu = DeviceBuf::new(a * 2 * FF * 4);
    let yk = DeviceBuf::new(a * E * 4);
    // the Q8 decode path
    gpu.run(MOE.kernel("qx_moe_gateup_q8k_nvfp4", CubeDim::new_1d(256)), (((2 * FF) as u32).div_ceil(8), a as u32, 1), &[dslots.ptr, dxq.ptr, dxd.ptr, K as u64, gu.ptr, FF as u64, E as u64, gate_row as u64, tail as u64]);
    gpu.run(MOE.kernel("qx_moe_down_q80_nvfp4", CubeDim::new_1d(256)), ((E as u32).div_ceil(16), a as u32, 1), &[dslots.ptr, gu.ptr, 0, 0, yk.ptr, FF as u64, E as u64, (2 * gb) as u64, down_row as u64, tail as u64]);
    gpu.sync();
    let y_gpu: Vec<f32> = yk.download(a * E);
    // the CPU pool
    let pool = Pool::new(&[0, 1, 2, 3]);
    let experts: Vec<Expert> = ids
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let r = records[e * rec..].as_ptr();
            Expert { gate: r, up: unsafe { r.add(gb) }, down: unsafe { r.add(2 * gb) }, out: i, tok: i / K, scale: scales[e] }
        })
        .collect();
    let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
    let mut y_cpu = vec![0f32; a * E];
    pool.run_multi(GateFmt::Nvfp4, &xr, FF, gate_row, down_row, &experts, &mut y_cpu);
    let diff = y_gpu.iter().zip(&y_cpu).filter(|(g, c)| g.to_bits() != c.to_bits()).count();
    let maxrel = y_gpu.iter().zip(&y_cpu).map(|(g, c)| ((g - c).abs() / c.abs().max(1e-6)) as f64).fold(0.0, f64::max);
    println!("q8 path: {diff} of {} outputs differ between GPU and CPU pool (max rel {maxrel:.2e})", a * E);
    // f64 reference with exact f32 inputs
    let mut wrow = vec![0f32; E];
    let mut hrow = vec![0f32; FF];
    let mut y_ref = vec![0f64; a * E];
    for (i, &e) in ids.iter().enumerate() {
        let r = &records[e * rec..(e + 1) * rec];
        let x = &xs[i / K];
        let mut h = vec![0f64; FF];
        for f in 0..FF {
            qx_quant::dequant_nvfp4(&r[f * gate_row..(f + 1) * gate_row], &mut wrow);
            let g: f64 = wrow.iter().zip(x).map(|(w, x)| *w as f64 * *x as f64).sum::<f64>() * scales[e][0] as f64;
            qx_quant::dequant_nvfp4(&r[gb + f * gate_row..gb + (f + 1) * gate_row], &mut wrow);
            let u: f64 = wrow.iter().zip(x).map(|(w, x)| *w as f64 * *x as f64).sum::<f64>() * scales[e][1] as f64;
            h[f] = g / (1.0 + (-g).exp()) * u;
        }
        for n in 0..E {
            qx_quant::dequant_nvfp4(&r[2 * gb + n * down_row..2 * gb + (n + 1) * down_row], &mut hrow);
            y_ref[i * E + n] = hrow.iter().zip(&h).map(|(w, h)| *w as f64 * h).sum::<f64>() * scales[e][2] as f64;
        }
    }
    let rel = |y: &[f32]| {
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in y.iter().zip(&y_ref) {
            num += (*a as f64 - b).powi(2);
            den += b.powi(2);
        }
        (num / den).sqrt()
    };
    println!("q8 path vs f64 reference: rel err {:.2e}", rel(&y_gpu));
    // the f32 exact path
    gpu.run(MOE.kernel("qx_moe_gateup_nvfp4", CubeDim::new_1d(256)), (((2 * FF) as u32).div_ceil(8), a as u32, 1), &[dslots.ptr, K as u64, 0, dx.ptr, gu.ptr, FF as u64, E as u64, gate_row as u64, tail as u64]);
    gpu.run(MOE.kernel("qx_moe_down_nvfp4", CubeDim::new_1d(256)), ((E as u32).div_ceil(8), a as u32, 1), &[dslots.ptr, 0, 0, 0, gu.ptr, yk.ptr, FF as u64, E as u64, (2 * gb) as u64, down_row as u64, tail as u64]);
    gpu.sync();
    let y_f32: Vec<f32> = yk.download(a * E);
    println!("f32 path vs f64 reference: rel err {:.2e}", rel(&y_f32));
    assert!(diff == 0, "the Q8 tiers are not bit-equal");
    assert!(rel(&y_f32) < 1e-5 && rel(&y_gpu) < 3e-2);
    println!("ok");
}
