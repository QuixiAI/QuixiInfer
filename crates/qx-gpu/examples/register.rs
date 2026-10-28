//! Can a read-only file mapping be pinned for DMA? Registers one tensor's range of the GGUF mapping and
//! times a pinned H2D copy from it.
fn main() {
    use cudarc::driver::{result, sys};
    qx_gpu::mem::bind();
    let path = std::env::args().nth(1).unwrap();
    let off: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let len: usize = std::env::args().nth(3).unwrap().parse().unwrap();
    let f = std::fs::File::open(&path).unwrap();
    let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let page = 4096;
    let start = off / page * page;
    let end = (off + len).div_ceil(page) * page;
    let p = unsafe { map.as_ptr().add(start) } as *mut std::ffi::c_void;
    let t = std::time::Instant::now();
    let r = unsafe { sys::cuMemHostRegister_v2(p, end - start, sys::CU_MEMHOSTREGISTER_READ_ONLY | sys::CU_MEMHOSTREGISTER_PORTABLE) };
    println!("cuMemHostRegister({:.1} MiB, READ_ONLY): {:?} in {:.2}s", (end - start) as f64 / 1048576.0, r, t.elapsed().as_secs_f64());
    if r != sys::CUresult::CUDA_SUCCESS {
        return;
    }
    let dev = unsafe { result::malloc_sync(64 << 20) }.unwrap();
    let stream = unsafe { result::stream::create(result::stream::StreamKind::NonBlocking) }.unwrap();
    let n = (64usize << 20).min(end - start);
    let t = std::time::Instant::now();
    for _ in 0..20 {
        unsafe { sys::cuMemcpyHtoDAsync_v2(dev, p as *const _, n, stream).result().unwrap() };
    }
    unsafe { result::stream::synchronize(stream).unwrap() };
    println!("pinned H2D from the mapping: {:.1} GB/s", 20.0 * n as f64 / t.elapsed().as_secs_f64() / 1e9);
    unsafe { sys::cuMemHostUnregister(p).result().unwrap() };
}
