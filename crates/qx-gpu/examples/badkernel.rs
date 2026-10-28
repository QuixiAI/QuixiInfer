//! Does a kernel that fails to compile make any noise? CubeCL attaches launch failures to the buffers a launch
//! binds; QuixiInfer's launches bind none (raw addresses in the info words), so this checks what `flush` and
//! `sync` report.
use burn_cubecl::cubecl::prelude::CubeDim;
use qx_gpu::kernel::Module;
use qx_gpu::Gpu;

fn main() {
    qx_gpu::init();
    let gpu = Gpu::new();
    let m = Module { name: "bogus", source: "typedef unsigned long long u64;\nextern \"C\" __global__ void bad(const u64* info) { undefined_function(info); }\n", source_hash: 0xbad, entries: &["bad"] };
    gpu.run(m.kernel("bad", CubeDim::new_1d(32)), (1, 1, 1), &[0]);
    println!("flush: {:?}", gpu.client.flush().err());
    println!("sync: {:?}", burn_cubecl::cubecl::future::block_on(gpu.client.sync()).err());
    let ok = Module { name: "fine", source: "typedef unsigned long long u64;\nextern \"C\" __global__ void fine(const u64* info) { }\n", source_hash: 0xf1e, entries: &["fine"] };
    gpu.run(ok.kernel("fine", CubeDim::new_1d(32)), (1, 1, 1), &[0]);
    println!("flush after a good launch: {:?}", gpu.client.flush().err());
}
