//! `qx-vision-gpu-check MMPROJ.gguf IMAGE...`: the GPU vision tower against the CPU one on each image: times, and
//! the difference of the two blocks.
fn main() -> anyhow::Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let gpu = qx_gpu::runtime::Gpu::new();
    // (CubeCL's stream must exist before the raw launcher is asked for functions)
    let h = gpu.upload(&[0u32; 4]);
    let _: Vec<u32> = gpu.download(&h);
    let path = std::path::Path::new(&a[1]);
    let v = qx_vision::Vision::load(path)?;
    let t0 = std::time::Instant::now();
    let tower = qx_engine::vision_gpu::GpuTower::new(path, &v, 0)?;
    println!("GPU tower ready in {:.2} s, {:.0} MB of device memory", t0.elapsed().as_secs_f64(), tower.device_bytes() as f64 / 1048576.0);
    let mut ok = true;
    for f in &a[2..] {
        let im = v.preprocess(&std::fs::read(f)?)?;
        let t1 = std::time::Instant::now();
        let (g, gh, gw) = tower.encode(&v, &im)?;
        let tg = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let (g2, _, _) = tower.encode(&v, &im)?;
        let tg2 = t2.elapsed().as_secs_f64();
        let t3 = std::time::Instant::now();
        let (c, _, _) = v.encode(&im);
        let tc = t3.elapsed().as_secs_f64();
        let e = v.out_dim;
        let (mut maxd, mut sumd, mut sumc, mut mincos) = (0f32, 0f64, 0f64, 1f64);
        for (rg, rc) in g.chunks_exact(e).zip(c.chunks_exact(e)) {
            let (mut dot, mut ng, mut nc) = (0f64, 0f64, 0f64);
            for (x, y) in rg.iter().zip(rc) {
                maxd = maxd.max((x - y).abs());
                sumd += (x - y).abs() as f64;
                sumc += y.abs() as f64;
                dot += *x as f64 * *y as f64;
                ng += *x as f64 * *x as f64;
                nc += *y as f64 * *y as f64;
            }
            mincos = mincos.min(dot / (ng.sqrt() * nc.sqrt() + 1e-30));
        }
        let mut coss: Vec<(f64, usize, f64)> = g.chunks_exact(e).zip(c.chunks_exact(e)).enumerate().map(|(i, (rg, rc))| {
            let dot: f64 = rg.iter().zip(rc).map(|(x, y)| *x as f64 * *y as f64).sum();
            let (ng, nc): (f64, f64) = (rg.iter().map(|x| (*x as f64).powi(2)).sum(), rc.iter().map(|x| (*x as f64).powi(2)).sum());
            (dot / (ng.sqrt() * nc.sqrt() + 1e-30), i, nc.sqrt())
        }).collect();
        coss.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("  rows with cosine under 0.999: {}; under 0.99: {}; the three lowest (cosine, row, CPU norm): {:?}", coss.iter().filter(|x| x.0 < 0.999).count(), coss.iter().filter(|x| x.0 < 0.99).count(), coss.iter().take(3).map(|x| (format!("{:.5}", x.0), x.1, format!("{:.2}", x.2))).collect::<Vec<_>>());
        let same_twice = g.iter().zip(&g2).all(|(x, y)| x.to_bits() == y.to_bits());
        let rel = sumd / sumc;
        println!("{}: {}x{} -> grid {gh}x{gw}, {} rows: GPU {tg:.3} s then {tg2:.3} s, CPU {tc:.2} s; max |d| {maxd:.3e}, mean |d| / mean |x| {rel:.3e}, row cosine min {mincos:.6}; two GPU runs {}", f.rsplit('/').next().unwrap(), im.w, im.h, g.len() / e, if same_twice { "bit-identical" } else { "DIFFERENT" });
        // (f16 activations against the CPU tower's f32: the bound is on the block as a whole and on its worst row)
        ok = ok && g.len() == c.len() && rel < 0.01 && mincos > 0.97 && same_twice && g.iter().all(|x| x.is_finite());
    }
    println!("{}", if ok { "ok" } else { "MISMATCH" });
    std::process::exit(if ok { 0 } else { 1 })
}
