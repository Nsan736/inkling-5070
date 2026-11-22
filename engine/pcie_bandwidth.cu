// inkling2_pcie_bandwidth: host -> card copies from pinned and from pageable memory, by the size of one copy, as
// the prompt path stages experts (stage(): copies of a few MB from pinned buffers).
// usage: inkling2_pcie_bandwidth [seconds of load at the end, for nvidia-smi to see the link under it]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const int hold = argc > 1 ? std::atoi(argv[1]) : 0;
    const size_t total = size_t(1) << 30;   // 1 GiB a measurement
    void* dev = nullptr;
    void* pin = nullptr;
    ck(cudaMalloc(&dev, total), "cudaMalloc");
    ck(cudaHostAlloc(&pin, total, cudaHostAllocDefault), "cudaHostAlloc");
    std::vector<unsigned char> page(total, 1);
    std::memset(pin, 1, total);
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (size_t chunk : {size_t(256) << 10, size_t(1) << 20, size_t(4) << 20, size_t(16) << 20, size_t(64) << 20}) {
        for (int kind = 0; kind < 2; ++kind) {
            const unsigned char* src = kind == 0 ? (const unsigned char*) pin : page.data();
            float best = 1e30f;
            for (int r = 0; r < 4; ++r) {
                cudaEventRecord(a, st);
                for (size_t o = 0; o < total; o += chunk)
                    ck(cudaMemcpyAsync((char*) dev + o, src + o, chunk, cudaMemcpyHostToDevice, st), "copy");
                cudaEventRecord(b, st);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                if (r) best = std::min(best, ms);
            }
            std::printf("host -> card, %-8s copies of %6zu KiB: %6.1f GB/s\n", kind == 0 ? "pinned" : "pageable",
                        chunk >> 10, total / (best * 1e-3) / 1e9);
        }
    }
    if (hold > 0) {   // keep the link busy for nvidia-smi
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(hold);
        while (std::chrono::steady_clock::now() < end) {
            ck(cudaMemcpyAsync(dev, pin, total, cudaMemcpyHostToDevice, st), "copy");
            cudaStreamSynchronize(st);
        }
    }
    return 0;
}
