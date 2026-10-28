//! Phase 1 risk gate (docs/phase1-gate.md): launch QuixiInfer kernels through CubeCL, measure
//! per-launch host overhead, capture and replay a CUDA graph, and write cudarc-allocated memory.
use qx_gpu::kernel::modules::SMOKE;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl_runtime::runtime::Runtime;
use cubecl::server::{KernelArguments, MetadataBindingInfo};
use std::time::Instant;



fn main() {
    qx_gpu::init();
    let device = CudaDevice::default();
    let client = CudaRuntime::client(&device);
    let n: usize = 1024;
    let x = client.create_from_slice(bytemuck::cast_slice(&vec![0f32; n]));

    let launch = |add: u64| {
        client.launch(
            Box::new(SMOKE.kernel("qx_add_scalar", CubeDim::new_1d(256))),
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            KernelArguments::new()
                .with_buffers(vec![x.clone().binding()])
                .with_info(MetadataBindingInfo::custom(vec![n as u64, add])),
        )
    };
    launch(2);
    let out = client.read_one(x.clone()).unwrap();
    let v: &[f32] = bytemuck::cast_slice(&out);
    println!("after one launch: x[0]={} x[{}]={}", v[0], n - 1, v[n - 1]);
    assert!(v.iter().all(|&e| e == 2.0));

    for iters in [1000usize, 20000] {
        let t = Instant::now();
        for _ in 0..iters {
            launch(1);
        }
        let host = t.elapsed();
        futures_lite_block(client.sync());
        let total = t.elapsed();
        println!(
            "{iters} launches: host {:.2} us/launch, wall incl. sync {:.2} us/launch",
            host.as_secs_f64() * 1e6 / iters as f64,
            total.as_secs_f64() * 1e6 / iters as f64
        );
    }
    let out = client.read_one(x.clone()).unwrap();
    let v: &[f32] = bytemuck::cast_slice(&out);
    println!("final x[0]={} (expect {})", v[0], 2 + 21000);

    // Graph capture: record 1000 launches once, replay them.
    client.graph_prepare().expect("graph_prepare");
    for _ in 0..1000 { launch(1); } // warmup inside prepare
    client.start_capture().expect("start_capture");
    for _ in 0..1000 { launch(1); }
    let graph = client.stop_capture().expect("stop_capture");
    futures_lite_block(client.sync()).ok();
    let before: f32 = bytemuck::cast_slice::<u8, f32>(&client.read_one(x.clone()).unwrap())[0];
    let reps = 50;
    let t = Instant::now();
    for _ in 0..reps { unsafe { graph.replay().expect("replay") }; }
    let host = t.elapsed();
    futures_lite_block(client.sync()).ok();
    let total = t.elapsed();
    let after: f32 = bytemuck::cast_slice::<u8, f32>(&client.read_one(x.clone()).unwrap())[0];
    println!("graph of 1000 kernels x {reps} replays: host {:.1} us/replay, wall {:.3} us/kernel; x[0] {} -> {} (expect +{})",
        host.as_secs_f64()*1e6/reps as f64, total.as_secs_f64()*1e6/(reps*1000) as f64, before, after, reps*1000);
    interop(&client);
}


/// Gate item 2: kernels launched through CubeCL write memory that cudarc allocated in the same
/// (primary) context: mapped pinned host memory, and a VMM-reserved device range.
fn interop(client: &cubecl_runtime::client::Client) {
    use cudarc::driver::{result, sys};
    let ctx = cudarc::driver::CudaContext::new(0).expect("cudarc primary context");
    ctx.bind_to_thread().unwrap();
    let n: usize = 4096;
    let launch_to = |ptr: u64, add: u64| {
        client.launch(
            Box::new(SMOKE.kernel("qx_write_ptr", CubeDim::new_1d(256))),
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            KernelArguments::new().with_info(MetadataBindingInfo::custom(vec![ptr, n as u64, add])),
        );
        futures_lite_block(client.sync()).expect("sync");
    };
    // Mapped pinned host memory.
    unsafe {
        let host = result::malloc_host(n * 4, sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE).expect("cuMemHostAlloc") as *mut f32;
        let mut dptr: sys::CUdeviceptr = 0;
        sys::cuMemHostGetDevicePointer_v2(&mut dptr, host as *mut _, 0).result().expect("cuMemHostGetDevicePointer");
        launch_to(dptr, 7);
        let v = std::slice::from_raw_parts(host, n);
        let ok = v.iter().enumerate().all(|(i, &x)| x == i as f32 * 0.5 + 7.0);
        println!("mapped pinned host memory written by a CubeCL launch: {}", if ok { "ok" } else { "MISMATCH" });
        result::free_host(host as *mut _).unwrap();
    }
    // VMM: reserve a range, back it with physical memory, map it.
    unsafe {
        let mut prop: sys::CUmemAllocationProp = std::mem::zeroed();
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        let mut gran: usize = 0;
        sys::cuMemGetAllocationGranularity(&mut gran, &prop, sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM).result().unwrap();
        let size = gran;
        let mut va: sys::CUdeviceptr = 0;
        sys::cuMemAddressReserve(&mut va, size * 4, gran, 0, 0).result().expect("reserve");
        let mut h: sys::CUmemGenericAllocationHandle = 0;
        sys::cuMemCreate(&mut h, size, &prop, 0).result().expect("cuMemCreate");
        sys::cuMemMap(va, size, 0, h, 0).result().expect("cuMemMap");
        let mut access: sys::CUmemAccessDesc = std::mem::zeroed();
        access.location = prop.location;
        access.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        sys::cuMemSetAccess(va, size, &access, 1).result().expect("cuMemSetAccess");
        launch_to(va, 3);
        let mut back = vec![0f32; n];
        sys::cuMemcpyDtoH_v2(back.as_mut_ptr() as *mut _, va, n * 4).result().unwrap();
        let ok = back.iter().enumerate().all(|(i, &x)| x == i as f32 * 0.5 + 3.0);
        println!("VMM-mapped device range ({} KiB granularity) written by a CubeCL launch: {}", gran / 1024, if ok { "ok" } else { "MISMATCH" });
        sys::cuMemUnmap(va, size).result().unwrap();
        sys::cuMemRelease(h).result().unwrap();
        sys::cuMemAddressFree(va, size * 4).result().unwrap();
    }
}

fn futures_lite_block<F: std::future::Future>(f: F) -> F::Output {
    // Minimal executor: CubeCL's sync future completes once polled to readiness.
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker { RawWaker::new(p, &VT) }
    static VT: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VT)) };
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) { return v; }
        std::thread::yield_now();
    }
}