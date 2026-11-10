//! cuBLAS on CubeCL's stream: half-precision tensor-core GEMMs with f32 accumulation for prompt processing
//! (the 2026-11-10 contract: prefill KL-bounded against the exact path; Strata's dense products are these calls).
use cudarc::cublas::sys;
use std::ffi::c_void;
use std::sync::Mutex;

struct Handle(sys::cublasHandle_t, bool);
// SAFETY: used from the engine thread only; the mutex serializes any other caller.
unsafe impl Send for Handle {}

static HANDLE: Mutex<Option<Handle>> = Mutex::new(None);

/// The handle, created on first use; its stream is CubeCL's first one, set as soon as that exists (`need_stream`:
/// it must exist by now).
fn with<R>(need_stream: bool, f: impl FnOnce(sys::cublasHandle_t) -> R) -> R {
    let mut g = HANDLE.lock().unwrap();
    if g.is_none() {
        crate::mem::bind();
        let mut h: sys::cublasHandle_t = std::ptr::null_mut();
        unsafe { sys::cublasCreate_v2(&mut h).result().expect("cublasCreate") };
        *g = Some(Handle(h, false));
    }
    let hd = g.as_mut().unwrap();
    if !hd.1 {
        let streams = cubecl::cuda::QX_STREAMS.lock().unwrap();
        if let Some(&st) = streams.first() {
            unsafe { sys::cublasSetStream_v2(hd.0, st as sys::cudaStream_t).result().expect("cublasSetStream") };
            hd.1 = true;
        } else {
            assert!(!need_stream, "CubeCL has created no stream yet");
        }
    }
    f(hd.0)
}

/// Creates the cuBLAS handle now (at load, before the expert tier is sized, so its memory is accounted for).
pub fn init() {
    with(false, |_| ());
}

/// Y[t][n] = sum_k W[n][k] X[t][k] for t < `t`, n < `n`: `w16` holds N rows of K halves, `x16` T rows of K halves,
/// `y` f32 rows `ldc` apart. Queued on CubeCL's stream, behind what was launched on it.
pub fn gemm_hss(w16: u64, k: usize, n: usize, x16: u64, t: usize, y: u64, ldc: usize) {
    let (alpha, beta) = (1f32, 0f32);
    with(true, |h| unsafe {
        // column-major: A' = K x N (W's rows as columns), B' = K x T, C' = N x T = A'^T B'
        sys::cublasGemmEx(
            h, sys::cublasOperation_t::CUBLAS_OP_T, sys::cublasOperation_t::CUBLAS_OP_N, n as i32, t as i32, k as i32,
            &alpha as *const f32 as *const c_void, w16 as *const c_void, sys::cudaDataType_t::CUDA_R_16F, k as i32,
            x16 as *const c_void, sys::cudaDataType_t::CUDA_R_16F, k as i32, &beta as *const f32 as *const c_void,
            y as *mut c_void, sys::cudaDataType_t::CUDA_R_32F, ldc as i32, sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .result()
        .expect("cublasGemmEx (f16 x f16 -> f32)");
    })
}
