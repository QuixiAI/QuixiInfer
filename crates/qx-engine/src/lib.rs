//! The QuixiInfer inference engine.
pub mod dsv41;
pub mod engine;
pub mod experts;
pub mod host_experts;
pub mod kvpages;
pub mod mailbox;
pub mod ple_io;
pub mod routing;
pub mod tiers;
pub mod ops;

/// The model to load when none is named: the pack if it has been made, else the GGUF.
/// Pins glibc malloc's mmap and trim thresholds. Call first in `main`, before any thread exists.
///
/// glibc raises its mmap threshold dynamically (up to 32 MiB) whenever a chunk above it is freed, after which
/// allocations of that size come from the per-thread arenas instead of fresh mappings, and freed arena memory is
/// only returned to the kernel from the top of a heap. Under the serving load (a 1.5 MB logits download per step,
/// a 517 KB logits copy per request, the pool's per-expert scratch on 16 threads) that pattern grew the server's
/// anonymous memory at about 330 MB/s, as 64 MiB arena heaps (62 of them at 5 GB), until the box ran out: 22.7 GB in
/// 75 s on 2026-11-06, intermittently from run to run. With a fixed 128 KiB threshold every mid-sized buffer is
/// its own mapping and goes back on free; the same load then stays at 1.8 GB. The forced bench never showed it
/// (one slot, one thread allocating per step), so the engine's own data structures are not the cause.
pub fn tune_host_allocator() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: mallopt only sets allocator parameters; called before any other thread allocates.
    unsafe {
        libc::mallopt(libc::M_MMAP_THRESHOLD, 128 * 1024);
        libc::mallopt(libc::M_TRIM_THRESHOLD, 256 * 1024);
    }
}

pub fn default_model_path() -> String {
    if let Ok(m) = std::env::var("QX_MODEL") {
        return m;
    }
    let dir = format!("{}/models/Qwen3.8-Flash-Next-GGUF", std::env::var("HOME").unwrap_or_default());
    // the NVFP4 model (pack, then GGUF), else the IQ3_XXS one it replaced
    for c in ["Qwen3.8-Flash-Next-NVFP4.qxp", "NVFP4/Qwen3.8-Flash-Next-NVFP4.gguf", "Qwen3.8-Flash-Next-UD-IQ3_XXS.qxp", "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"] {
        let p = format!("{dir}/{c}");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    format!("{dir}/Qwen3.8-Flash-Next-NVFP4.qxp")
}
