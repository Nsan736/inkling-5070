// inkling2_cpu_calls: the engine's CPU experts (cpu_experts(), its Workers) called as a written token calls them,
// without the card, to tell the calls' own speed from what the rest of the engine does to it (development tool).
// A pool of NVFP4 experts of Inkling 2 Mini's shape (gate and up 1152 x 2304, down 2304 x 1152) larger than the
// caches; each call takes `k` experts at random from it for one token, as a visit does.
// usage: inkling2_cpu_calls [threads (12)] [pool GiB (6)] [calls (2000)] [gap_us (0)]
#include "cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::poly;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 12;
    const double gib = argc > 2 ? std::atof(argv[2]) : 6.0;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 2000;
    const int gap_us = argc > 4 ? std::atoi(argv[4]) : 0;
    ggml_cpu_init();
    const int n_embd = 2304, n_ff = 1152, t = 40;   // GGML_TYPE_NVFP4
    const size_t gu_row = ggml_row_size((ggml_type) t, n_embd), d_row = ggml_row_size((ggml_type) t, n_ff);
    const size_t one = 2 * (size_t) n_ff * gu_row + (size_t) n_embd * d_row;
    const size_t n_exp = (size_t) (gib * (1ull << 30) / (double) one);
    std::printf("%zu experts of %.2f MB (%.1f GiB), %d threads\n", n_exp, one / 1e6, n_exp * one / (double) (1ull << 30),
                threads);
    std::vector<uint8_t> pool(n_exp * one);
    for (size_t i = 0; i < pool.size(); ++i) pool[i] = (uint8_t) (i * 37 + (i >> 12));   // every page touched
    std::vector<float> x(n_embd), out(n_embd);
    for (int i = 0; i < n_embd; ++i) x[i] = 0.01f * std::sin(0.1f * (float) i);
    Workers w(threads);
    Situ situ{4.f, 25.f};
    std::mt19937 rng(1);
    std::string err;
    if (!cpu_expert_types_ok(t, t, n_embd, n_ff, err)) return std::printf("%s\n", err.c_str()), 1;
    for (int k : {1, 2, 3, 4, 8}) {
        std::vector<double> us;
        for (int c = 0; c < calls / 5; ++c) {
            CpuExpert e[8];
            for (int j = 0; j < k; ++j) {
                const size_t idx = rng() % n_exp;
                uint8_t* p = pool.data() + idx * one;
                e[j] = {p, p + (size_t) n_ff * gu_row, p + 2 * (size_t) n_ff * gu_row, 0.125f, 0};
            }
            const auto t0 = Clock::now();
            cpu_experts(w, e, k, 1, t, t, t, n_embd, n_ff, x.data(), 7.f, false, out.data(), nullptr, nullptr, nullptr,
                        &situ, false);
            us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
            if (gap_us > 0) {   // the card's part of a visit: the workers spin through it as in the engine
                const auto g0 = Clock::now();
                while (std::chrono::duration<double, std::micro>(Clock::now() - g0).count() < gap_us) {}
            }
        }
        std::sort(us.begin(), us.end());
        const double med = us[us.size() / 2];
        std::printf("%d experts a call: median %.0f us, p90 %.0f us -> %.1f GB/s (median)\n", k, med,
                    us[us.size() * 9 / 10], k * one / med / 1e3);
    }
    return 0;
}
