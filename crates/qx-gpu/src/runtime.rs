//! A thin handle on the CUDA device: uploads, downloads and kernel launches through CubeCL.
use crate::kernel::CudaSource;
use burn_cubecl::cubecl::prelude::*;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::server::{Handle, KernelArguments, MetadataBindingInfo};
use cubecl_runtime::client::Client;
use cubecl_runtime::runtime::Runtime;

pub struct Gpu {
    pub device: CudaDevice,
    pub client: Client,
}

impl Gpu {
    /// Opens device 0. Call `qx_gpu::init()` first in `main`.
    pub fn new() -> Self {
        // The engine runs kernels that wait in place for a word another stream writes (the experts' staged
        // records). With the default 8 work queues per context, streams share queues, and a waiting kernel can sit
        // in front of the very copy it waits for: with four side streams for the chunk path's experts the first
        // layer hung (2026-11-10); with 32 queues 4 and 8 side streams run. Set before the context exists.
        if std::env::var_os("CUDA_DEVICE_MAX_CONNECTIONS").is_none() {
            // SAFETY: called at start-up, before any other thread of this process reads the environment
            unsafe { std::env::set_var("CUDA_DEVICE_MAX_CONNECTIONS", "32") };
        }
        let device = CudaDevice::default();
        let client = CudaRuntime::client(&device);
        Self { device, client }
    }

    pub fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> Handle {
        self.client.create_from_slice(bytemuck::cast_slice(data))
    }

    pub fn alloc(&self, bytes: usize) -> Handle {
        self.client.empty(bytes.max(4))
    }

    pub fn download<T: bytemuck::Pod>(&self, h: &Handle) -> Vec<T> {
        let bytes = self.client.read_one(h.clone()).expect("read_one");
        bytemuck::pod_collect_to_vec(&bytes)
    }

    /// Launches `k` on a grid of `grid` blocks, with `bufs` as the leading arguments and `info` as
    /// the trailing `const u64*`.
    pub fn launch(&self, k: CudaSource, grid: (u32, u32, u32), bufs: &[&Handle], info: &[u64]) {
        let mut args = KernelArguments::new().with_buffers(bufs.iter().map(|h| (*h).clone().binding()).collect());
        if !info.is_empty() {
            args = args.with_info(MetadataBindingInfo::custom(info.to_vec()));
        }
        self.client.launch(Box::new(k), CubeCount::Static(grid.0, grid.1, grid.2), args);
    }

    /// Launches `k` with no buffer bindings: every address travels in `info`. Inside a chunk pass with
    /// `QX_RAW_LAUNCH=1` the launch goes through the driver API on CubeCL's stream (`crate::raw`).
    pub fn run(&self, k: CudaSource, grid: (u32, u32, u32), info: &[u64]) {
        if crate::raw::RAW.load(std::sync::atomic::Ordering::Relaxed) {
            crate::raw::launch(&k.module, k.entry, k.cube_dim.0.x, grid, info);
            return;
        }
        self.launch(k, grid, &[], info);
    }

    pub fn sync(&self) {
        block_on(self.client.sync()).expect("device sync");
    }
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Polls a future to completion on the current thread; CubeCL's futures need no reactor.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VT)
    }
    static VT: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    // SAFETY: the vtable's functions do nothing and never dereference the data pointer.
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VT)) };
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}
