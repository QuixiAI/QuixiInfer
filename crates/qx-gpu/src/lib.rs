//! QuixiInfer's GPU layer: QuixiInfer-owned CUDA C++ kernels launched through CubeCL's CUDA runtime.

pub mod kernel;
pub mod mem;
pub mod reference;
pub mod runtime;

pub use runtime::Gpu;
/// Directory holding the NVRTC whose PTX the installed driver accepts.
///
/// The driver supports CUDA 13.2, so PTX from 13.3's NVRTC is refused; 13.1's is accepted.
pub const DEFAULT_NVRTC_DIR: &str = "/usr/local/cuda-13.1/lib64";

/// The toolkit root CubeCL hands NVRTC as its include path; it must match the NVRTC above.
pub const DEFAULT_CUDA_ROOT: &str = "/usr/local/cuda-13.1";

/// Puts the NVRTC directory first on `LD_LIBRARY_PATH` and sets `CUDA_PATH` to the matching toolkit,
/// re-executing the process if needed.
///
/// cudarc opens NVRTC by bare name and NVRTC opens its builtins by soname, so the dynamic linker
/// decides which toolkit's NVRTC loads, and it reads `LD_LIBRARY_PATH` only at process start.
/// Call this first in `main`, before any thread starts. `QX_NVRTC_DIR` overrides the directory.
pub fn init() {
    use std::os::unix::process::CommandExt;
    let dir = std::env::var("QX_NVRTC_DIR").unwrap_or_else(|_| DEFAULT_NVRTC_DIR.to_string());
    let root = std::env::var("QX_CUDA_ROOT").unwrap_or_else(|_| DEFAULT_CUDA_ROOT.to_string());
    let cur = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if cur.split(':').next() == Some(dir.as_str()) && std::env::var("CUDA_PATH").ok().as_deref() == Some(root.as_str()) {
        return;
    }
    let new = if cur.is_empty() { dir } else { format!("{dir}:{cur}") };
    let exe = std::env::current_exe().expect("current_exe");
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", new)
        .env("CUDA_PATH", root)
        .exec();
    panic!("qx-gpu: re-exec with LD_LIBRARY_PATH failed: {err}");
}
