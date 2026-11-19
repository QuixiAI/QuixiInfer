//! `qx-vision-check MMPROJ.gguf IMAGE [OUT_PREFIX]`: fits and encodes one image on the CPU; with OUT_PREFIX writes
//! `OUT_PREFIX.rgb` (the fitted pixels) and `OUT_PREFIX.f32` (the block, little-endian floats) for comparison with
//! another implementation (`tools/vision_ref.py`).
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: qx-vision-check MMPROJ.gguf IMAGE [OUT_PREFIX]");
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let v = qx_vision::Vision::load(std::path::Path::new(&a[1]))?;
    let t1 = std::time::Instant::now();
    let im = v.preprocess(&std::fs::read(&a[2])?)?;
    let t2 = std::time::Instant::now();
    let (block, gh, gw) = v.encode(&im);
    let rows = block.len() / v.out_dim;
    println!("loaded in {:.2} s; fitted to {}x{} in {:.3} s; {} patches -> grid {gh}x{gw}, {rows} rows of {} in {:.2} s", (t1 - t0).as_secs_f64(), im.w, im.h, (t2 - t1).as_secs_f64(), (im.w / v.patch) * (im.h / v.patch), v.out_dim, t2.elapsed().as_secs_f64());
    if let Some(p) = a.get(3) {
        std::fs::write(format!("{p}.rgb"), &im.rgb)?;
        std::fs::write(format!("{p}.f32"), block.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
        println!("wrote {p}.rgb ({}x{}) and {p}.f32", im.w, im.h);
    }
    Ok(())
}
