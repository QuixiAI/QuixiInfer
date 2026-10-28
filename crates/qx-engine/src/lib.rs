//! The QuixiInfer inference engine.
pub mod engine;
pub mod experts;
pub mod host_experts;
pub mod kvpages;
pub mod mailbox;
pub mod ple_io;
pub mod routing;
pub mod ops;

/// The model to load when none is named: the pack if it has been made, else the GGUF.
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
