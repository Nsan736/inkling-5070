// inkling2_cpu_bandwidth: what the CPU side of the engine can reach, apart from the engine.
//   1. memory: N threads read a 4 GiB buffer with AVX2 loads (the PC's read bandwidth in practice)
//   2. NVFP4: N threads run ggml-cpu's vec_dot (NVFP4 x Q8_0) over 4 GiB of expert rows, as the engine's CPU experts do
//      (rows of 2304 values, a Q8_0 activation each), so the rows come from RAM and not from the caches
// usage: inkling2_cpu_bandwidth [threads...]   (default 4 8 12 16 20)
#include "ggml.h"
#include "ggml-cpu.h"
#include "cpu_nvfp4.hpp"

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <class F, class R> double best_seconds(int threads, F&& f, R&& reset, int reps = 5) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        reset();
        std::vector<std::thread> th;
        const auto t0 = Clock::now();
        for (int t = 0; t < threads; ++t) th.emplace_back([&, t] { f(t); });
        for (auto& x : th) x.join();
        best = std::min(best, std::chrono::duration<double>(Clock::now() - t0).count());
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> counts;
    for (int i = 1; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
    if (counts.empty()) counts = {4, 8, 12, 16, 20};
    ggml_cpu_init();

    const size_t bytes = size_t(4) << 30;
    uint8_t* buf = (uint8_t*) _aligned_malloc(bytes, 64);
    for (size_t i = 0; i < bytes; i += 4096) buf[i] = (uint8_t) i;   // touch every page
    std::memset(buf, 0x11, bytes);

    // 1. reads
    std::atomic<uint64_t> sink{0};
    for (int n : counts) {
        const double s = best_seconds(n, [&](int t) {
            const size_t per = bytes / n, a = per * t;
            __m256i acc = _mm256_setzero_si256();
            for (size_t i = a; i + 128 <= a + per; i += 128) {
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i*) (buf + i)));
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i*) (buf + i + 32)));
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i*) (buf + i + 64)));
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i*) (buf + i + 96)));
            }
            sink += (uint64_t) _mm256_extract_epi64(acc, 0);
        }, [] {});
        std::printf("memory read   %2d threads: %6.1f GB/s\n", n, bytes / s / 1e9);
    }

    // 2. NVFP4 rows x a Q8_0 activation (ggml-cpu, as cpu_experts.cpp calls it)
    const int n_embd = 2304;
    const auto* tr = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const size_t row = ggml_row_size(GGML_TYPE_NVFP4, n_embd);
    const size_t rows = bytes / row;
    std::vector<float> x(n_embd);
    for (int i = 0; i < n_embd; ++i) x[i] = std::sin(0.01f * (float) i);
    const auto vt = tr->vec_dot_type;
    std::vector<uint8_t> xq(ggml_row_size(vt, n_embd));
    ggml_get_type_traits_cpu(vt)->from_float(x.data(), xq.data(), n_embd);
    // valid NVFP4 blocks: a pattern of codes with small scales (the bytes' meaning does not change the time)
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* p = buf + r * row;
        for (size_t b = 0; b < row; ++b) p[b] = (uint8_t) (b * 37 + r);
    }
    for (int n : counts) {
        std::atomic<int> next{0};
        const double s = best_seconds(n, [&](int) {
            float acc = 0.f, v = 0.f;
            for (;;) {   // tasks of 32 rows, as cpu_experts.cpp hands them out
                const size_t r0 = (size_t) next.fetch_add(32);
                if (r0 >= rows) break;
                for (size_t r = r0; r < std::min(rows, r0 + 32); ++r) {
                    tr->vec_dot(n_embd, &v, 0, buf + r * row, 0, xq.data(), 0, 1);
                    acc += v;
                }
            }
            sink += (uint64_t) (acc != 0.f);
        }, [&] { next = 0; });
        std::printf("NVFP4 vec_dot %2d threads: %6.1f GB/s of weights (%.2f ms for a 4.48 MB expert)\n", n,
                    rows * row / s / 1e9, 4.478e6 / (rows * row / s) * 1e3);
    }
    // 3. the same over 24 MB read again and again: rows that are in the L3 cache (the 265K's is 36 MB)
    const size_t hot_rows = (size_t(24) << 20) / row;
    for (int n : counts) {
        std::atomic<long long> next{0};
        const long long total = (long long) hot_rows * 40;
        const double s = best_seconds(n, [&](int) {
            float acc = 0.f, v = 0.f;
            for (;;) {
                const long long r0 = next.fetch_add(32);
                if (r0 >= total) break;
                for (long long r = r0; r < std::min(total, r0 + 32); ++r) {
                    tr->vec_dot(n_embd, &v, 0, buf + (size_t) (r % (long long) hot_rows) * row, 0, xq.data(), 0, 1);
                    acc += v;
                }
            }
            sink += (uint64_t) (acc != 0.f);
        }, [&] { next = 0; });
        std::printf("NVFP4 vec_dot %2d threads, rows in L3: %6.1f GB/s of weights\n", n, total * row / s / 1e9);
    }
    // 4. this fork's AVX-VNNI kernel (cpu_nvfp4.hpp): the same sums as ggml's, checked bit for bit, and its speed
    {
        namespace nf = strata::poly::nvfp4cpu;
        std::vector<int32_t> ys(n_embd / 4);
        nf::nvfp4_ysum4((const nf::BlockQ8_0*) xq.data(), n_embd, ys.data());
        size_t differ = 0;
        for (size_t r = 0; r < 4096; ++r) {
            float a = 0.f;
            tr->vec_dot(n_embd, &a, 0, buf + r * row, 0, xq.data(), 0, 1);
            const float b = nf::nvfp4_dot(n_embd, (const nf::BlockNvfp4*) (buf + r * row), (const nf::BlockQ8_0*) xq.data(),
                                          ys.data());
            differ += std::memcmp(&a, &b, 4) != 0;
        }
        std::printf("VNNI kernel vs ggml: %zu of 4096 rows differ in any bit\n", differ);
        for (int pass = 0; pass < 2; ++pass)
            for (int n : counts) {
                std::atomic<long long> next{0};
                const long long total = pass == 0 ? (long long) rows : (long long) hot_rows * 40;
                const double s = best_seconds(n, [&](int) {
                    float acc = 0.f;
                    for (;;) {
                        const long long r0 = next.fetch_add(32);
                        if (r0 >= total) break;
                        for (long long r = r0; r < std::min(total, r0 + 32); ++r) {
                            const size_t rr = pass == 0 ? (size_t) r : (size_t) (r % (long long) hot_rows);
                            acc += nf::nvfp4_dot(n_embd, (const nf::BlockNvfp4*) (buf + rr * row),
                                                 (const nf::BlockQ8_0*) xq.data(), ys.data());
                        }
                    }
                    sink += (uint64_t) (acc != 0.f);
                }, [&] { next = 0; });
                std::printf("VNNI kernel   %2d threads%s: %6.1f GB/s of weights\n", n, pass ? ", rows in L3" : "",
                            total * row / s / 1e9);
            }
    }
    // 5. two rows at once, with and without a software prefetch ahead of each row
    {
        namespace nf = strata::poly::nvfp4cpu;
        std::vector<int32_t> ys(n_embd / 4);
        nf::nvfp4_ysum4((const nf::BlockQ8_0*) xq.data(), n_embd, ys.data());
        size_t differ = 0;
        for (size_t r = 0; r + 1 < 4096; r += 2) {
            float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
            tr->vec_dot(n_embd, &a0, 0, buf + r * row, 0, xq.data(), 0, 1);
            tr->vec_dot(n_embd, &a1, 0, buf + (r + 1) * row, 0, xq.data(), 0, 1);
            nf::nvfp4_dot2(n_embd, (const nf::BlockNvfp4*) (buf + r * row), (const nf::BlockNvfp4*) (buf + (r + 1) * row),
                           (const nf::BlockQ8_0*) xq.data(), ys.data(), &b0, &b1, 512);
            differ += (std::memcmp(&a0, &b0, 4) != 0) + (std::memcmp(&a1, &b1, 4) != 0);
        }
        std::printf("two-row kernel vs ggml: %zu of 4096 rows differ in any bit\n", differ);
        for (int pf : {2048, 3072, 4096, 8192})
            for (int n : counts) {
                std::atomic<long long> next{0};
                const long long total = (long long) rows & ~1LL;
                const double s = best_seconds(n, [&](int) {
                    float acc = 0.f, v0 = 0.f, v1 = 0.f;
                    for (;;) {
                        const long long r0 = next.fetch_add(32);
                        if (r0 >= total) break;
                        for (long long r = r0; r + 1 < std::min(total, r0 + 32); r += 2) {
                            nf::nvfp4_dot2(n_embd, (const nf::BlockNvfp4*) (buf + (size_t) r * row),
                                           (const nf::BlockNvfp4*) (buf + (size_t) (r + 1) * row),
                                           (const nf::BlockQ8_0*) xq.data(), ys.data(), &v0, &v1, pf);
                            acc += v0 + v1;
                        }
                    }
                    sink += (uint64_t) (acc != 0.f);
                }, [&] { next = 0; });
                std::printf("two-row kernel %2d threads, prefetch %4d B: %6.1f GB/s of weights\n", n, pf,
                            total * row / s / 1e9);
            }
    }
    std::printf("(sink %llu)\n", (unsigned long long) sink.load());
    _aligned_free(buf);
    return 0;
}
