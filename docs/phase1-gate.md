# Phase 1 risk gate: Burn/CubeCL as the GPU runtime

Run 2026-10-24 on the RTX 5090, driver 595.99 (CUDA 13.2), burn 0.22.1, cubecl 0.11.3.
Program: `crates/qx-gpu/examples/gate.rs`. Result: **pass**; Burn/CubeCL stays.

| Gate item | Result |
|---|---|
| A QuixiInfer CUDA C++ kernel compiled by NVRTC and launched through CubeCL | Pass, correct output |
| Kernels write memory cudarc allocated: mapped pinned host memory | Pass |
| Kernels write memory cudarc allocated: VMM-reserved, mapped device range | Pass (2 MiB granularity) |
| Capture 1000 launches into a CUDA graph and replay | Pass, exact result after 50 replays |
| Host overhead per captured step under 50 us | Pass: 2.6 to 3.4 us per replay of a 1000-kernel graph |

| Measurement | Value |
|---|---|
| Host cost per eager launch through CubeCL | 1.8 to 2.1 us |
| GPU time per tiny kernel inside a graph | 0.79 us |

## What it took

- **C++ backend.** CubeCL's CUDA runtime defaults to its LLVM backend, which refuses source kernels. Enable
  cubecl's `cuda-cpp` feature, and tag sources with language `"cuda"`.
- **Entry point name.** Burn's `SourceKernel` names every entry `main`, which CUDA forbids on a `__global__`
  function. `qx_gpu::kernel::CudaSource` implements `CubeKernel` directly with a named entry point.
- **Arguments.** A launch passes its buffers in order, then a device pointer to the `u64` info words. Raw device
  addresses (VMM arenas, mapped host memory) travel in the info words, so CubeCL's allocator never has to own them.
- **Context.** CubeCL uses the device's primary context, the same one cudarc retains, so pointers are shared.
- **NVRTC version.** The driver refuses PTX ISA 9.3 from CUDA 13.3. NVRTC 13.1 (`apt install cuda-nvrtc-13-1`)
  emits PTX it accepts. cudarc loads NVRTC by bare name and NVRTC loads its builtins by soname, so
  `qx_gpu::init()` re-executes the process with `/usr/local/cuda-13.1/lib64` first on `LD_LIBRARY_PATH`.
- **nvcc.** CUDA 13.1's nvcc cannot parse this system's glibc headers (conflicting `rsqrt` declarations), so
  `build.rs` checks kernels with nvcc 13.3. Cubins from 13.3 load on the 13.2 driver; its PTX does not.
