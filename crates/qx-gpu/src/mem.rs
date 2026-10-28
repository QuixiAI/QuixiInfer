//! Device and host memory QuixiInfer owns, allocated through the CUDA driver in the primary context that
//! CubeCL also uses. Kernels receive these addresses in their info words.
//!
//! Copies here are synchronous. A copy into memory a queued kernel reads must be issued after
//! `Gpu::sync`, and so must a copy out of memory a queued kernel writes.
//!
//! A copy from pageable host memory returns once the bytes are staged, before the DMA lands, and CubeCL's
//! stream is not ordered after the legacy stream the copy runs on, so `upload` waits for the whole context.
use cudarc::driver::{result, sys};
use std::sync::OnceLock;

static CTX: OnceLock<std::sync::Arc<cudarc::driver::CudaContext>> = OnceLock::new();

/// Binds the primary context to the calling thread. Every thread that calls the driver directly needs it.
pub fn bind() {
    let ctx = CTX.get_or_init(|| cudarc::driver::CudaContext::new(0).expect("CUDA primary context"));
    ctx.bind_to_thread().expect("bind CUDA context");
}

pub struct DeviceBuf {
    pub ptr: u64,
    pub bytes: usize,
    /// False for a view into memory another buffer owns (`view`): dropping it frees nothing.
    owned: bool,
}

impl DeviceBuf {
    pub fn new(bytes: usize) -> Self {
        bind();
        let ptr = unsafe { result::malloc_sync(bytes.max(256)) }.expect("cuMemAlloc");
        unsafe { result::memset_d8_sync(ptr, 0, bytes.max(256)) }.expect("cuMemsetD8");
        Self { ptr, bytes, owned: true }
    }

    /// `bytes` of device memory at `ptr` that another buffer owns and outlives this one.
    pub fn view(ptr: u64, bytes: usize) -> Self {
        Self { ptr, bytes, owned: false }
    }

    pub fn from_slice<T: bytemuck::Pod>(data: &[T]) -> Self {
        let b = Self::new(std::mem::size_of_val(data));
        b.upload(data);
        b
    }

    pub fn upload<T: bytemuck::Pod>(&self, data: &[T]) {
        self.upload_at(0, data);
    }

    pub fn upload_at<T: bytemuck::Pod>(&self, offset: usize, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        assert!(offset + bytes.len() <= self.bytes);
        bind();
        unsafe { result::memcpy_htod_sync(self.ptr + offset as u64, bytes) }.expect("cuMemcpyHtoD");
        result::ctx::synchronize().expect("cuCtxSynchronize");
    }

    pub fn download<T: bytemuck::Pod + Default + Clone>(&self, n: usize) -> Vec<T> {
        self.download_at(0, n)
    }

    pub fn download_at<T: bytemuck::Pod + Default + Clone>(&self, offset: usize, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        assert!(offset + std::mem::size_of_val(&v[..]) <= self.bytes);
        bind();
        unsafe { result::memcpy_dtoh_sync(bytemuck::cast_slice_mut::<T, u8>(&mut v[..]), self.ptr + offset as u64) }.expect("cuMemcpyDtoH");
        v
    }

    pub fn zero(&self) {
        bind();
        unsafe { result::memset_d8_sync(self.ptr, 0, self.bytes.max(256)) }.expect("cuMemsetD8");
    }

    /// Zeroes `len` bytes at `offset`.
    pub fn zero_range(&self, offset: usize, len: usize) {
        assert!(offset + len <= self.bytes);
        bind();
        unsafe { result::memset_d8_sync(self.ptr + offset as u64, 0, len) }.expect("cuMemsetD8");
    }

    /// The address `off` bytes in.
    pub fn at(&self, off: usize) -> u64 {
        assert!(off <= self.bytes);
        self.ptr + off as u64
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        bind();
        unsafe {
            let _ = result::free_sync(self.ptr);
        }
    }
}

/// Free and total device memory, in bytes.
pub fn mem_info() -> (usize, usize) {
    bind();
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }.result().expect("cuMemGetInfo");
    (free, total)
}

/// Pinned host memory mapped into the device address space: kernels read and write it over PCIe through
/// `dptr`, the host through `ptr`. Used for the small mailboxes the GPU and the CPU workers exchange.
pub struct HostMapped {
    pub ptr: *mut u8,
    pub dptr: u64,
    pub bytes: usize,
}

unsafe impl Send for HostMapped {}
unsafe impl Sync for HostMapped {}

impl HostMapped {
    pub fn new(bytes: usize) -> Self {
        bind();
        let bytes = bytes.max(4096);
        unsafe {
            let ptr = result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE).expect("cuMemHostAlloc") as *mut u8;
            std::ptr::write_bytes(ptr, 0, bytes);
            let mut dptr: sys::CUdeviceptr = 0;
            sys::cuMemHostGetDevicePointer_v2(&mut dptr, ptr as *mut _, 0).result().expect("cuMemHostGetDevicePointer");
            Self { ptr, dptr, bytes }
        }
    }

    /// The host view of `n` values of `T` at byte offset `off`.
    pub fn slice<T>(&self, off: usize, n: usize) -> &mut [T] {
        assert!(off + n * std::mem::size_of::<T>() <= self.bytes && off % std::mem::align_of::<T>() == 0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut T, n) }
    }

    pub fn atomic_u64(&self, off: usize) -> &std::sync::atomic::AtomicU64 {
        assert!(off + 8 <= self.bytes && off % 8 == 0);
        unsafe { &*(self.ptr.add(off) as *const std::sync::atomic::AtomicU64) }
    }

    pub fn atomic_u32(&self, off: usize) -> &std::sync::atomic::AtomicU32 {
        assert!(off + 4 <= self.bytes && off % 4 == 0);
        unsafe { &*(self.ptr.add(off) as *const std::sync::atomic::AtomicU32) }
    }
}

impl Drop for HostMapped {
    fn drop(&mut self) {
        bind();
        unsafe {
            let _ = result::free_host(self.ptr as *mut _);
        }
    }
}
