#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compiler;
mod compute;
mod device;
mod runtime;

pub use compiler::{CudaCompilationOptions, CudaCompiler, CudaRepresentation};
pub use device::*;
pub use runtime::*;

pub mod install {
    use std::path::PathBuf;

    pub fn include_path() -> PathBuf {
        let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
        path.push("include");
        path
    }

    pub fn cccl_include_path() -> PathBuf {
        let mut path = include_path();
        path.push("cccl");
        path
    }

    pub fn cuda_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("CUDA_PATH") {
            return Some(PathBuf::from(path));
        }

        #[cfg(target_os = "linux")]
        {
            // If it is installed as part of the distribution
            return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/usr/local/cuda"))
            } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/opt/cuda"))
            } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
                // Maybe the compiler was installed within the user path.
                Some(PathBuf::from("/usr"))
            } else {
                None
            };
        }

        #[cfg(target_os = "windows")]
        {
            return Some(PathBuf::from(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
            ));
        }

        #[allow(unreachable_code)]
        None
    }
}

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;

    pub use half::bf16;
    pub use half::f16;

    // The complex types are the C++ backend's: the LLVM backend does not lower them, so it does
    // not advertise them (see `restrict_features`) and the tests for them are not generated.
    // `testgen_complex_validation` runs either way -- it asserts that an unadvertised complex
    // operation is refused, which is what the LLVM backend must do and what the C++ backend
    // never reaches.
    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);

    cubecl_core::testgen_launch_dynamic_count!();
    cubecl_std::testgen!();

    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);

    cubecl_std::testgen_quantized_view!(f16);

    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_core!();
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_compare!();
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_math!();
    cubecl_core::testgen_complex_validation!();
    cubecl_core::testgen_profiling!();
    cubecl_core::testgen_profiling_nested!();
}

/// QuixiInfer: every CUDA stream this backend created, as raw `CUstream` handles (usize), in creation order. The
/// engine's raw launcher (`qx_gpu::raw`) enqueues on the first one, the client's default stream, so its launches
/// stay ordered with CubeCL's.
pub static QX_STREAMS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// QuixiInfer: the modules loaded from precompiled CUDA C++ sources, as (FNV-1a of the source text, raw `CUmodule`
/// handle as usize). A precompiled source holds many entry points; upstream compiles and loads the whole text once
/// per kernel id (0.75-1.3 s each for the largest module, 2026-11-12), so the context keeps one module per source
/// and takes each entry point from it. `qx_gpu::raw` reads and fills the same table. `QX_MODULE_SHARE=0` restores
/// one module per kernel id.
pub static QX_MODULES: std::sync::Mutex<Vec<(u64, usize)>> = std::sync::Mutex::new(Vec::new());

/// FNV-1a of `s`, the hash `qx-gpu`'s build script keys its modules with.
pub fn qx_fnv1a(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}

/// Whether precompiled sources share one module per source text.
pub fn qx_module_share() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| !std::env::var("QX_MODULE_SHARE").is_ok_and(|v| v == "0"))
}

/// The PTX of `source` compiled with `options`, from `~/.cache/quixiinfer/ptx` (`QX_PTX_CACHE=0` disables it), or
/// `None`. The key covers the source text, every compiler option (target, include paths) and NVRTC's version. A
/// caller stores an entry only after the driver loaded it, and forgets one the driver refuses.
pub fn qx_ptx_cached(source: &str, options: &[&str]) -> Option<Vec<std::ffi::c_char>> {
    let bytes = std::fs::read(qx_ptx_path(source, options)?).ok()?;
    // a PTX text ends with its NUL; anything else is a torn or foreign file
    if bytes.len() < 64 || *bytes.last()? != 0 || bytes[..bytes.len() - 1].contains(&0) {
        return None;
    }
    Some(bytes.into_iter().map(|b| b as std::ffi::c_char).collect())
}

/// Stores `ptx` for [`qx_ptx_cached`] (written beside and renamed, so a reader never sees half a file).
pub fn qx_ptx_store(source: &str, options: &[&str], ptx: &[std::ffi::c_char]) {
    let Some(path) = qx_ptx_path(source, options) else { return };
    let Some(dir) = path.parent() else { return };
    let bytes: Vec<u8> = ptx.iter().map(|&c| c as u8).collect();
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let _ = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&tmp, &bytes)).and_then(|_| std::fs::rename(&tmp, &path));
}

/// Drops the cached PTX of `source` (the driver refused it).
pub fn qx_ptx_forget(source: &str, options: &[&str]) {
    if let Some(path) = qx_ptx_path(source, options) {
        let _ = std::fs::remove_file(path);
    }
}

fn qx_ptx_path(source: &str, options: &[&str]) -> Option<std::path::PathBuf> {
    if std::env::var("QX_PTX_CACHE").is_ok_and(|v| v == "0") {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    // the NVRTC that loads is chosen by the dynamic linker, not by an option: its version is part of the key
    let (mut major, mut minor) = (0, 0);
    // SAFETY: two valid out-pointers.
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    let opts = options.iter().fold(qx_fnv1a(format!("nvrtc {major}.{minor}").as_bytes()), |h, o| qx_fnv1a(o.as_bytes()) ^ h.rotate_left(17));
    Some(std::path::PathBuf::from(home).join(format!(".cache/quixiinfer/ptx/{:016x}-{:016x}.ptx", qx_fnv1a(source.as_bytes()), opts)))
}
