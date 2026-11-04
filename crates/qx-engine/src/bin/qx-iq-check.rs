//! The GPU's IQ / K-quant expert kernels with Q8_K activations against the CPU pool on random expert records of
//! DeepSeek-V4.1-Flash's shapes (5120 in, 2304 wide, clamped SwiGLU): every (gate, down) format pair the model
//! uses must be bit-equal between the tiers, and close to an f64 reference.
use cubecl::prelude::CubeDim;
use qx_cpu::act::quantize_q8_k;
use qx_cpu::pool::{Expert, ExpertFmt, Pool, WType};
use qx_gpu::kernel::modules::MOE;
use qx_gpu::mem::DeviceBuf;
use qx_gpu::Gpu;
use qx_pack::ggml::GgmlType;

const E: usize = 5120;
const FF: usize = 2304;
const NE: usize = 4;
const K: usize = 6;
const T: usize = 2;
const CLAMP: f32 = 10.0;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn suffix(t: GgmlType) -> &'static str {
    match t {
        GgmlType::IQ2_S => "iq2_s",
        GgmlType::IQ2_XXS => "iq2_xxs",
        GgmlType::IQ2_XS => "iq2_xs",
        GgmlType::IQ3_XXS => "iq3_xxs",
        GgmlType::IQ1_S => "iq1_s",
        GgmlType::Q2_K => "q2_k",
        _ => unreachable!(),
    }
}

/// Random blocks with a sane f16 scale where the format keeps it.
fn blocks(ty: GgmlType, n_elem: usize, s: &mut u64) -> Vec<u8> {
    let (be, bb) = ty.block();
    let mut v: Vec<u8> = (0..n_elem / be * bb).map(|_| rng(s) as u8).collect();
    for b in v.chunks_exact_mut(bb) {
        let offs: &[usize] = if ty == GgmlType::Q2_K { &[80, 82] } else { &[0] };
        for &o in offs {
            let d = half::f16::from_f32(((rng(s) % 1000) as f32 + 1.0) * 1e-4);
            b[o..o + 2].copy_from_slice(&d.to_le_bytes());
        }
    }
    v
}

fn main() {
    qx_gpu::init();
    let gpu = Gpu::new();
    let pool = Pool::new(&[0, 1, 2, 3]);
    let pairs = [
        (GgmlType::IQ1_S, GgmlType::Q2_K),
        (GgmlType::IQ1_S, GgmlType::IQ2_XS),
        (GgmlType::IQ1_S, GgmlType::IQ2_XXS),
        (GgmlType::IQ2_XXS, GgmlType::IQ2_S),
        (GgmlType::IQ2_S, GgmlType::IQ3_XXS),
    ];
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut all_ok = true;
    for (gt, dt) in pairs {
        let gate_row = gt.bytes(E);
        let down_row = dt.bytes(FF);
        let (gb, db) = (FF * gate_row, E * down_row);
        let rec = (2 * gb + db).next_multiple_of(4096);
        let mut records = vec![0u8; rec * NE];
        for e in 0..NE {
            let r = &mut records[e * rec..(e + 1) * rec];
            r[..gb].copy_from_slice(&blocks(gt, FF * E, &mut s));
            r[gb..2 * gb].copy_from_slice(&blocks(gt, FF * E, &mut s));
            r[2 * gb..2 * gb + db].copy_from_slice(&blocks(dt, E * FF, &mut s));
        }
        let drec = DeviceBuf::from_slice(&records);
        let a = T * K;
        let ids: Vec<usize> = (0..a).map(|_| rng(&mut s) as usize % NE).collect();
        let slots: Vec<u64> = ids.iter().map(|&e| drec.ptr + (e * rec) as u64).collect();
        let dslots = DeviceBuf::from_slice(&slots);
        let xs: Vec<Vec<f32>> = (0..T).map(|_| (0..E).map(|_| ((rng(&mut s) % 20001) as f32 - 10000.0) * 1e-3).collect()).collect();
        let (mut xq, mut xd) = (Vec::new(), Vec::new());
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
        let gu = DeviceBuf::new(a * 2 * FF * 4);
        let yk = DeviceBuf::new(a * E * 4);
        // kernel entries are 'static names: these few leak on purpose
        let kg: &'static str = Box::leak(format!("qx_moe_gateup_q8k_{}_w20", suffix(gt)).into_boxed_str());
        let kd: &'static str = Box::leak(format!("qx_moe_down_q8k_{}", suffix(dt)).into_boxed_str());
        gpu.run(MOE.kernel(kg, CubeDim::new_1d(256)), (((2 * FF) as u32).div_ceil(8), a as u32, 1), &[dslots.ptr, dxq.ptr, dxd.ptr, K as u64, gu.ptr, FF as u64, E as u64, gate_row as u64]);
        gpu.run(MOE.kernel(kd, CubeDim::new_1d(256)), ((E as u32).div_ceil(16), a as u32, 1), &[dslots.ptr, gu.ptr, CLAMP.to_bits() as u64, 0, yk.ptr, FF as u64, E as u64, (2 * gb) as u64, down_row as u64]);
        gpu.sync();
        let y_gpu: Vec<f32> = yk.download(a * E);
        // the CPU pool
        let experts: Vec<Expert> = ids
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let r = records[e * rec..].as_ptr();
                Expert { gate: r, up: unsafe { r.add(gb) }, down: unsafe { r.add(2 * gb) }, out: i, tok: i / K, scale: [1.0; 3] }
            })
            .collect();
        let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
        let mut y_cpu = vec![0f32; a * E];
        let fmt = ExpertFmt { gate: qx_engine::experts::wtype(gt), down: qx_engine::experts::wtype(dt), clamp: CLAMP };
        assert!(fmt.down != WType::Iq4Nl);
        pool.run_multi(fmt, &xr, FF, gate_row, down_row, &experts, &mut y_cpu);
        let diff = y_gpu.iter().zip(&y_cpu).filter(|(g, c)| g.to_bits() != c.to_bits()).count();
        // f64 reference on the same Q8_K-rounded activations (x and h), so only the arithmetic is compared
        let rounded = |v: &[f32]| -> Vec<f32> {
            let mut q = Vec::new();
            quantize_q8_k(v, &mut q);
            q.iter().flat_map(|b| b.qs.iter().map(move |&s| s as f32 * b.d)).collect()
        };
        let mut wrow = vec![0f32; E];
        let mut hrow = vec![0f32; FF];
        let (mut num, mut den) = (0f64, 0f64);
        for (i, &e) in ids.iter().enumerate() {
            let r = &records[e * rec..(e + 1) * rec];
            let x = rounded(&xs[i / K]);
            let mut h = vec![0f32; FF];
            for f in 0..FF {
                qx_pack::dequant(gt, &r[f * gate_row..(f + 1) * gate_row], &mut wrow);
                let g: f64 = wrow.iter().zip(&x).map(|(w, x)| *w as f64 * *x as f64).sum::<f64>();
                qx_pack::dequant(gt, &r[gb + f * gate_row..gb + (f + 1) * gate_row], &mut wrow);
                let u: f64 = wrow.iter().zip(&x).map(|(w, x)| *w as f64 * *x as f64).sum::<f64>();
                let (g, u) = (g.min(CLAMP as f64), u.clamp(-(CLAMP as f64), CLAMP as f64));
                h[f] = (g / (1.0 + (-g).exp()) * u) as f32;
            }
            let h = rounded(&h);
            for n in 0..E {
                qx_pack::dequant(dt, &r[2 * gb + n * down_row..2 * gb + (n + 1) * down_row], &mut hrow);
                let yr: f64 = hrow.iter().zip(&h).map(|(w, h)| *w as f64 * *h as f64).sum();
                num += (y_gpu[i * E + n] as f64 - yr).powi(2);
                den += yr.powi(2);
            }
        }
        let rel = (num / den).sqrt();
        // 1e-3: the reference rounds h from f64, the tiers from f32, so a Q8_K step can differ on a few values
        let ok = diff == 0 && rel < 1e-3;
        all_ok &= ok;
        println!("gate {gt:?} down {dt:?}: {diff} of {} outputs differ between GPU and CPU pool; rel err vs f64 {rel:.2e} {}", a * E, if ok { "ok" } else { "FAIL" });
    }
    assert!(all_ok, "mismatch");
    println!("ok");
}
