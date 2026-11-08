// The per-record host-to-device landing time: how long one expert record (default 9.5 MB) takes to reach VRAM by
// (a) cudaMemcpyAsync from pinned memory on one copy stream, (b) the same spread over 4 streams, (c) a gather kernel
// reading GPU-mapped pinned memory with int4 loads (Big-MoE-Small-Box lands an 8-9 MB record in 0.36 ms that way on
// PCIe 4.0; our staged record measured 0.7 ms through the driver, 2026-11-06). Latency per record (one at a time,
// event-timed) and throughput with 24 records in flight. No model, seconds to run:
//   /usr/local/cuda-13.3/bin/nvcc -arch=sm_120a -O3 -o ~/qx-bins/pcie_bench kernels/bench/pcie_bench.cu && ~/qx-bins/pcie_bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

__global__ void gather(const int4* __restrict__ src, int4* __restrict__ dst, size_t n4) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}

int main(int argc, char** argv) {
    const size_t rec = argc > 1 ? (size_t)atof(argv[1]) * 1000000 : 9500000;
    const int nrec = 24;
    void* pinned; CK(cudaHostAlloc(&pinned, rec * nrec, cudaHostAllocMapped));
    std::memset(pinned, 1, rec * nrec);
    void* mapped; CK(cudaHostGetDevicePointer(&mapped, pinned, 0));
    void* dev; CK(cudaMalloc(&dev, rec * nrec));
    cudaStream_t s[4]; for (auto& x : s) CK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time = [&](const char* name, auto f) {
        f(); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0, s[0])); for (int i = 0; i < 3; ++i) CK(cudaStreamWaitEvent(s[i + 1], e0, 0));
        const int iters = 10;
        for (int i = 0; i < iters; ++i) f();
        for (int i = 1; i < 4; ++i) { cudaEvent_t t; CK(cudaEventCreateWithFlags(&t, cudaEventDisableTiming)); CK(cudaEventRecord(t, s[i])); CK(cudaStreamWaitEvent(s[0], t, 0)); }
        CK(cudaEventRecord(e1, s[0])); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
        return ms;
    };
    std::printf("record %.1f MB, %d records\n", rec / 1e6, nrec);
    // one record, one copy at a time (latency)
    float ms = time("one record memcpyAsync", [&] { CK(cudaMemcpyAsync(dev, pinned, rec, cudaMemcpyHostToDevice, s[0])); });
    std::printf("%-44s %7.3f ms  %6.1f GB/s\n", "1 record, cudaMemcpyAsync pinned (latency)", ms, rec / ms / 1e6);
    ms = time("24 records, one stream", [&] { for (int r = 0; r < nrec; ++r) CK(cudaMemcpyAsync((char*)dev + r * rec, (char*)pinned + r * rec, rec, cudaMemcpyHostToDevice, s[0])); });
    std::printf("%-44s %7.3f ms per record  %6.1f GB/s\n", "24 records, cudaMemcpyAsync, 1 stream", ms / nrec, rec * nrec / ms / 1e6);
    ms = time("24 records, 4 streams", [&] { for (int r = 0; r < nrec; ++r) CK(cudaMemcpyAsync((char*)dev + r * rec, (char*)pinned + r * rec, rec, cudaMemcpyHostToDevice, s[r % 4])); });
    std::printf("%-44s %7.3f ms per record  %6.1f GB/s\n", "24 records, cudaMemcpyAsync, 4 streams", ms / nrec, rec * nrec / ms / 1e6);
    for (int blocks : {48, 170, 680}) {
        ms = time("gather one", [&] { gather<<<blocks, 256, 0, s[0]>>>((const int4*)mapped, (int4*)dev, rec / 16); });
        std::printf("1 record, gather kernel from mapped, %4d blocks %7.3f ms  %6.1f GB/s\n", blocks, ms, rec / ms / 1e6);
    }
    ms = time("gather 24", [&] { for (int r = 0; r < nrec; ++r) gather<<<170, 256, 0, s[r % 4]>>>((const int4*)((char*)mapped + r * rec), (int4*)((char*)dev + r * rec), rec / 16); });
    std::printf("%-44s %7.3f ms per record  %6.1f GB/s\n", "24 records, gather kernels, 4 streams", ms / nrec, rec * nrec / ms / 1e6);
    // a kernel reading a mapped record in place, once (the GEMV-like pattern: every byte read once, no store)
    return 0;
}
