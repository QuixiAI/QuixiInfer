use anyhow::Result;
use qx_pack::gguf::Gguf;
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("inspect") => {
            let g = Gguf::open(std::path::Path::new(&args[2]))?;
            for (k, v) in &g.metadata {
                if k.starts_with("tokenizer.ggml.") && matches!(v, qx_pack::gguf::Value::Arr(_)) {
                    println!("{k} = <array of {}>", v.as_arr().unwrap().len());
                } else if k == "tokenizer.chat_template" {
                    println!("{k} = <{} chars>", v.as_str().map(|s| s.len()).unwrap_or(0));
                } else {
                    println!("{k} = {}", serde_json::to_string(v)?);
                }
            }
            let mut by: BTreeMap<String, (usize, u64)> = BTreeMap::new();
            for t in &g.tensors {
                let key = t.name.split('.').enumerate().map(|(i, p)| if i == 1 && t.name.starts_with("blk.") { "N" } else { p }).collect::<Vec<_>>().join(".");
                let e = by.entry(key).or_default();
                e.0 += 1;
                e.1 += t.bytes;
            }
            let mut v: Vec<_> = by.into_iter().collect();
            v.sort_by_key(|x| std::cmp::Reverse(x.1 .1));
            for (k, (n, b)) in v {
                println!("{:9.3} GiB x{n:3} {k}", b as f64 / (1u64 << 30) as f64);
            }
            println!("tensors: {}", g.tensors.len());
            Ok(())
        }
        Some("dump") => {
            // qx-pack dump MODEL TENSOR N_ELEMENTS OUT_PREFIX: raw bytes and our f32 decoding of the first N elements.
            let g = Gguf::open(std::path::Path::new(&args[2]))?;
            let t = g.tensor(&args[3]).ok_or_else(|| anyhow::anyhow!("no tensor {}", args[3]))?;
            let n: usize = args[4].parse()?;
            let nb = t.ty.bytes(n);
            let raw = &g.data(t)[..nb];
            let mut out = vec![0f32; n];
            qx_pack::dequant(t.ty, raw, &mut out);
            std::fs::write(format!("{}.raw", args[5]), raw)?;
            std::fs::write(format!("{}.f32", args[5]), out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            println!("{} {:?} {n} elements", t.name, t.ty);
            Ok(())
        }
        Some("pack") => {
            // qx-pack pack MODEL.gguf OUT.qxp [--profile FILE] [--delete-shards]
            let g = Gguf::open(std::path::Path::new(&args[2]))?;
            let out = std::path::Path::new(&args[3]);
            let flag = |f: &str| args.iter().position(|a| a == f);
            let ne = g.tensors.iter().find(|t| t.name.ends_with("ffn_gate_exps.weight")).map_or(0, |t| *t.shape.last().unwrap() as usize);
            let text = match flag("--profile") {
                Some(i) => std::fs::read_to_string(&args[i + 1])?,
                None => qx_pack::pack::CALIB_ROUTING.to_string(),
            };
            let routing: Vec<u32> = text
                .lines()
                .filter_map(|l| {
                    let mut v = l.split_whitespace().map(|x| x.parse::<u32>());
                    Some(v.next()?.ok()? * ne as u32 + v.next()?.ok()?)
                })
                .collect();
            qx_pack::pack::write(&g, out, (!routing.is_empty()).then_some(&routing[..]))?;
            qx_pack::pack::verify(out, Some(&g))?;
            if flag("--delete-shards").is_some() {
                for s in &g.shards {
                    std::fs::remove_file(&s.path)?;
                    eprintln!("deleted {}", s.path.display());
                }
            }
            Ok(())
        }
        // scale-q8 FILE TENSOR FACTOR: multiplies every Q8_0 block scale of TENSOR by FACTOR, in place (a converter
        // that dropped a per-tensor scale, as llama.cpp's did for the FP8 PLE table)
        Some("scale-q8") => {
            use rayon::prelude::*;
            use std::os::unix::fs::FileExt;
            let g = Gguf::open(std::path::Path::new(&args[2]))?;
            let t = g.tensor(&args[3]).ok_or_else(|| anyhow::anyhow!("no tensor {}", args[3]))?;
            let factor: f32 = args[4].parse()?;
            anyhow::ensure!(t.ty == qx_pack::ggml::GgmlType::Q8_0, "{} is {:?}, not Q8_0", t.name, t.ty);
            let shard = &g.shards[t.shard];
            let (off, len) = (shard.data_start + t.offset, t.bytes as usize);
            anyhow::ensure!(len % 34 == 0);
            let f = std::fs::OpenOptions::new().read(true).write(true).open(&shard.path)?;
            let chunk = 34 * (1usize << 21);
            let n = len.div_ceil(chunk);
            let done = std::sync::atomic::AtomicUsize::new(0);
            (0..n).into_par_iter().try_for_each(|i| -> Result<()> {
                let (o, l) = (i * chunk, chunk.min(len - i * chunk));
                let mut b = vec![0u8; l];
                f.read_exact_at(&mut b, off + o as u64)?;
                for blk in b.chunks_exact_mut(34) {
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32() * factor;
                    blk[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                }
                f.write_all_at(&b, off + o as u64)?;
                let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if k % 64 == 0 {
                    eprintln!("  {k}/{n} chunks");
                }
                Ok(())
            })?;
            f.sync_all()?;
            println!("scaled {} Q8_0 blocks of {} by {factor}", len / 34, t.name);
            Ok(())
        }
        Some("verify") => {
            // qx-pack verify PACK.qxp [MODEL.gguf]: checksums, and with the source a byte-for-byte round trip
            let src = args.get(3).map(|p| Gguf::open(std::path::Path::new(p))).transpose()?;
            qx_pack::pack::verify(std::path::Path::new(&args[2]), src.as_ref())?;
            println!("ok");
            Ok(())
        }
        _ => {
            eprintln!("usage: qx-pack inspect MODEL-00001-of-0000N.gguf | dump MODEL TENSOR N OUT_PREFIX");
            eprintln!("       qx-pack pack MODEL-00001-of-0000N.gguf OUT.qxp [--profile FILE] [--delete-shards]");
            eprintln!("       qx-pack verify PACK.qxp [MODEL-00001-of-0000N.gguf]");
            eprintln!("       qx-pack scale-q8 MODEL.gguf TENSOR FACTOR   (multiply a Q8_0 tensor's block scales in place)");
            std::process::exit(2)
        }
    }
}
