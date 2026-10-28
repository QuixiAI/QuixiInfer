//! `qx-tok check TEXT_FILE TOKENS_FILE`: our tokenization of the text against llama.cpp's token ids.
use anyhow::Result;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = std::env::var("QX_MODEL").unwrap_or_else(|_| format!("{}/models/Qwen3.8-Flash-Next-GGUF/NVFP4/Qwen3.8-Flash-Next-NVFP4.gguf", std::env::var("HOME").unwrap()));
    let g = qx_pack::gguf::Gguf::open(std::path::Path::new(&model))?;
    let t0 = std::time::Instant::now();
    let tok = qx_tok::Tok::from_gguf(&g)?;
    eprintln!("tokenizer built in {:.2}s", t0.elapsed().as_secs_f64());
    let text = std::fs::read_to_string(&a[2])?;
    let want: Vec<u32> = std::fs::read(&a[3])?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let got = tok.encode(&text)?;
    let first = got.iter().zip(&want).position(|(x, y)| x != y);
    match first {
        None if got.len() == want.len() => println!("identical: {} tokens", got.len()),
        _ => {
            let p = first.unwrap_or(got.len().min(want.len()));
            println!("differ at {p}: ours {:?} vs llama.cpp {:?} (lengths {} vs {})", &got[p..(p + 5).min(got.len())], &want[p..(p + 5).min(want.len())], got.len(), want.len());
        }
    }
    let round = tok.decode(&got)?;
    println!("decode round trip: {}", if round == text { "exact" } else { "DIFFERS" });
    Ok(())
}
