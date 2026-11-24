// inkling2 fork (development only): the time of a prompt batch's full-layer attention over a long cache, for
// profiling the tensor-core path (inkling_attend()) apart from the engine.  The caches hold random codes; only the
// time is looked at.
//
// usage: inkling2_attn_bench [pos] [rows] [kv] [reps]
//   pos: the batch's first position (default 253952: the last 8K of 256K), rows: the batch (default 2048),
//   kv: fp16 | q8_0 | q8v4 (default q8_0), reps: timed calls (default 5).  INKLING2_ATTN_F16 picks the kernel as in
//   the engine.
//        inkling2_attn_bench check: --kv k8v4's store and attention (split, unsplit and tensor-core paths, rotated
//   and not) against the same math on the host in double precision over the cache's own codes.
#include "strata/poly/inkling_kernels.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::poly;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t) i * 2654435761u ^ seed;
        x ^= x >> 13, x *= 0x5bd1e995u, x ^= x >> 15;
        p[i] = (uint8_t) x;
    }
}

__global__ void fill_halves(uint16_t* p, size_t n, float v) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        p[i] = __half_as_ushort(__float2half(v * (1.0f + 0.25f * (float) ((i * 7) % 5) / 5.0f)));
}

template <class T> T* dalloc(size_t n) {
    T* d = nullptr;
    ck(cudaMalloc(&d, n * sizeof(T)), "malloc");
    return d;
}

// the card's tensor-core rate: m16n8k16 steps on registers, f32 or f16 sums (F16), 8 chains a warp
template <bool F16>
__global__ void mma_peak(float* out, int iters) {
    uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u}, b[2] = {0x00010001u, 0x00010001u};
    float c[8][4] = {};
    uint32_t h[8][2] = {};
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if constexpr (F16)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(h[k][0]), "+r"(h[k][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            else
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                             "{%0,%1,%2,%3};\n"
                             : "+f"(c[k][0]), "+f"(c[k][1]), "+f"(c[k][2]), "+f"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
    float s = 0;
    for (int k = 0; k < 8; ++k) s += c[k][0] + c[k][3] + (float) (h[k][0] & 1u);
    if (s == 12345.f) out[0] = s;
}

int peak() {
    float* o = dalloc<float>(1);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int iters = 20000, blocks = sms * 8, threads = 128;
    for (int f16 = 0; f16 < 2; ++f16) {
        auto go = [&] {
            if (f16) mma_peak<true><<<blocks, threads>>>(o, iters);
            else mma_peak<false><<<blocks, threads>>>(o, iters);
        };
        go();
        ck(cudaDeviceSynchronize(), "peak");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0), cudaEventCreate(&e1);
        cudaEventRecord(e0);
        go();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "peak");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double flops = 2.0 * 16 * 8 * 16 * 8 * (double) iters * blocks * (threads / 32);
        std::printf("m16n8k16 %s sums: %.1f TFLOPS\n", f16 ? "f16" : "f32", flops / ms / 1e9);
    }
    return 0;
}

}  // namespace

int check();

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "check") return check();
    if (argc > 1 && std::string(argv[1]) == "peak") return peak();
    const long long pos = argc > 1 ? std::atoll(argv[1]) : 253952;
    const int rows = argc > 2 ? std::atoi(argv[2]) : 2048;
    const std::string kv = argc > 3 ? argv[3] : "q8_0";
    const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
    constexpr int kDim = 128, kHeads = 24, kKv = 8, kRel = 16, kExtent = 8192;
    const size_t keys = (size_t) pos + rows, width = (size_t) kKv * kDim, vals = keys * width;

    uint16_t *kh = nullptr, *vh = nullptr;
    Kvq8 kq{}, vq{};
    if (kv == "fp16") {
        kh = dalloc<uint16_t>(vals), vh = dalloc<uint16_t>(vals);
        fill_halves<<<1024, 256>>>(kh, vals, 0.05f);
        fill_halves<<<1024, 256>>>(vh, vals, 0.05f);
    } else {
        int8_t* kc = dalloc<int8_t>(vals);
        uint16_t* ks = dalloc<uint16_t>(vals / 32);
        fill_bytes<<<1024, 256>>>((uint8_t*) kc, vals, 1u);
        fill_halves<<<1024, 256>>>(ks, vals / 32, 0.0004f);
        kq = {kc, ks};
        const bool v4 = kv == "q8v4";
        const size_t vbytes = v4 ? vals / 2 : vals;
        int8_t* vc = dalloc<int8_t>(vbytes);
        uint16_t* vs = dalloc<uint16_t>(vals / 32);
        fill_bytes<<<1024, 256>>>((uint8_t*) vc, vbytes, 2u);
        fill_halves<<<1024, 256>>>(vs, vals / 32, v4 ? 0.006f : 0.0004f);
        vq = {vc, vs};
        if (v4) vq.bits = 4, vq.wht = true;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hq((size_t) rows * kHeads * kDim), hr((size_t) rows * kHeads * kRel), hs((size_t) rows * kHeads),
        ht((size_t) kExtent * kRel);
    for (float& x : hq) x = nd(rng);
    for (float& x : hr) x = nd(rng);
    for (float& x : hs) x = nd(rng);
    for (float& x : ht) x = 0.1f * nd(rng);
    float *q = dalloc<float>(hq.size()), *r = dalloc<float>(hr.size()), *s = dalloc<float>(hs.size()),
          *t = dalloc<float>(ht.size()), *out = dalloc<float>(hq.size());
    ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(r, hr.data(), hr.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(s, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(t, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice), "up");
    const InklingRel rel{r, t, kRel, kExtent};
    const float scale = 1.f / std::sqrt((float) kDim);
    // up to 16 tokens: the split path (a token's, as the engine calls it: with its work and a room)
    const size_t wn = rows <= kInklingSplitRows ? inkling_attend_work_floats(kHeads, kDim) : 0;
    float* work = wn ? dalloc<float>(wn) : nullptr;
    auto call = [&] {
        inkling_attend(q, kh, vh, kq, vq, rel, s, pos, rows, out, kDim, kHeads, kKv, scale, 0, 0, work, wn, nullptr,
                       wn ? pos + rows : 0);
    };
    call();
    ck(cudaDeviceSynchronize(), "warm");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) call();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    std::vector<float> ho(hq.size());
    ck(cudaMemcpy(ho.data(), out, ho.size() * 4, cudaMemcpyDeviceToHost), "down");
    uint64_t sum = 1469598103934665603ull;   // the whole output's bits (FNV-1a), to compare kernels to the bit
    for (float x : ho) {
        uint32_t b;
        std::memcpy(&b, &x, 4);
        sum = (sum ^ b) * 1099511628211ull;
    }
    std::printf("pos %lld rows %d kv %s: %.2f ms a call (%.1f tok/s for this layer alone), bits %016llx\n", pos,
                rows, kv.c_str(), ms / reps, rows * 1000.0 / (ms / reps), (unsigned long long) sum);
    return 0;
}

// ---- check -------------------------------------------------------------------------------------------------------
namespace {

float h2f(uint16_t h) {
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
}

template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return v;
}

template <class T> T* up(const std::vector<T>& v) {
    T* d = dalloc<T>(v.size());
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
    return d;
}

// the inverse of the store's rotation of a 32-value block: H y / sqrt(32), then the signs
void wht_back_host(double* x) {
    for (int h = 1; h < 32; h <<= 1)
        for (int i = 0; i < 32; ++i)
            if (!(i & h)) {
                const double a = x[i], b = x[i + h];
                x[i] = a + b, x[i + h] = a - b;
            }
    for (int i = 0; i < 32; ++i) x[i] *= 0.17677669529663687 * (((0x9e3779b9u >> i) & 1u) ? -1.0 : 1.0);
}

}  // namespace

int check() {
    constexpr int kDim = 128, kHeads = 24, kKv = 8, kExtent = 64;
    const int N = 1500, T = 40, W = kKv * kDim;   // keys cached; the last T of them are the queries' tokens
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float sd) {
        std::vector<float> v(n);
        for (float& x : v) x = sd * nd(rng);
        return v;
    };
    // outliers in a few channels of V, as real values have, so that the rotation has something to spread
    auto kraw = randv(size_t(N) * W, 1.0f), vraw = randv(size_t(N) * W, 1.0f);
    for (int t = 0; t < N; ++t)
        for (int c = 5; c < W; c += 97) vraw[size_t(t) * W + c] *= 8.0f;
    float *dk = up(kraw), *dv = up(vraw), *kout = dalloc<float>(kraw.size()), *vout = dalloc<float>(vraw.size());
    const std::vector<float> zeros(size_t(4) * W, 0.0f);
    float *kst = up(zeros), *vst = up(zeros), *w0 = up(zeros);
    int failures = 0;
    for (int wht = 0; wht < 2; ++wht) {
        int8_t* kc = dalloc<int8_t>(size_t(N) * W);
        uint16_t* ks = dalloc<uint16_t>(size_t(N) * W / 32);
        int8_t* vc = dalloc<int8_t>(size_t(N) * W / 2);
        uint16_t* vs = dalloc<uint16_t>(size_t(N) * W / 32);
        InklingKvDst dst;
        dst.kc = kc, dst.ks = ks, dst.vc = vc, dst.vs = vs, dst.v4 = true, dst.v_wht = wht != 0;
        // a prompt batch's store (more than 16 tokens), then 8 tokens' (the one-launch kernel)
        inkling_kv_store(dk, dv, kst, vst, w0, w0, nullptr, nullptr, kKv, kDim, N - 8, 1e9f, 1e-6f, dst, 0, kout,
                         vout, nullptr);
        inkling_kv_store(dk + size_t(N - 8) * W, dv + size_t(N - 8) * W, kst, vst, w0, w0, nullptr, nullptr, kKv,
                         kDim, 8, 1e9f, 1e-6f, dst, N - 8, kout + size_t(N - 8) * W, vout + size_t(N - 8) * W,
                         nullptr);
        ck(cudaDeviceSynchronize(), "store");
        const auto hv = down(vout, vraw.size());
        const auto hkc = down(kc, size_t(N) * W), hvc = down(vc, size_t(N) * W / 2);
        const auto hks = down(ks, size_t(N) * W / 32), hvs = down(vs, size_t(N) * W / 32);
        // the cache as values: K's codes and V's (turned back from the rotation)
        std::vector<double> K(size_t(N) * W), V(size_t(N) * W);
        double err = 0, ref = 0;
        for (size_t b = 0; b < size_t(N) * W / 32; ++b) {
            double x[32];
            for (int j = 0; j < 32; ++j) {
                const size_t i = b * 32 + j;
                K[i] = hkc[i] * (double) h2f(hks[b]);
                const unsigned byte = (uint8_t) hvc[i / 2];
                x[j] = (int((i & 1) ? byte >> 4 : byte & 15u) - 8) * (double) h2f(hvs[b]);
            }
            if (wht) wht_back_host(x);
            for (int j = 0; j < 32; ++j) {
                const size_t i = b * 32 + j;
                V[i] = x[j];
                err += (x[j] - hv[i]) * (x[j] - hv[i]);
                ref += (double) hv[i] * hv[i];
            }
        }
        std::printf("k8v4%s: the values' 4-bit error %.4f of their rms\n", wht ? "" : "-plain", std::sqrt(err / ref));
        for (int nrel : {16, 8}) {
            auto q = randv(size_t(T) * kHeads * kDim, 1.0f), r = randv(size_t(T) * kHeads * nrel, 1.0f),
                 sk = randv(size_t(T) * kHeads, 1.0f), tb = randv(size_t(kExtent) * nrel, 0.3f);
            float *dq = up(q), *dr = up(r), *ds = up(sk), *dt = up(tb), *out = dalloc<float>(q.size());
            const InklingRel rel{dr, dt, nrel, kExtent};
            const float scale = 1.f / std::sqrt((float) kDim);
            Kvq8 kq{kc, ks}, vq{vc, vs};
            vq.bits = 4, vq.wht = wht != 0;
            const long long p0 = N - T;
            std::vector<double> want(q.size());
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < kHeads; ++h) {
                    const int g = h / 3;
                    const long long p = p0 + t;
                    std::vector<double> sc((size_t) p + 1);
                    double mx = sk[size_t(t) * kHeads + h];
                    for (long long j = 0; j <= p; ++j) {
                        double d = 0;
                        for (int i = 0; i < kDim; ++i)
                            d += (double) q[(size_t(t) * kHeads + h) * kDim + i] * K[(size_t(j) * kKv + g) * kDim + i];
                        d *= scale;
                        const long long dist = p - j;
                        if (dist < kExtent)
                            for (int k = 0; k < nrel; ++k)
                                d += (double) tb[size_t(dist) * nrel + k] * r[(size_t(t) * kHeads + h) * nrel + k];
                        sc[(size_t) j] = d;
                        mx = std::max(mx, d);
                    }
                    double den = std::exp(sk[size_t(t) * kHeads + h] - mx);
                    std::vector<double> o(kDim, 0.0);
                    for (long long j = 0; j <= p; ++j) {
                        const double e = std::exp(sc[(size_t) j] - mx);
                        den += e;
                        for (int i = 0; i < kDim; ++i) o[i] += e * V[(size_t(j) * kKv + g) * kDim + i];
                    }
                    for (int i = 0; i < kDim; ++i) want[(size_t(t) * kHeads + h) * kDim + i] = o[i] / den;
                }
            auto compare = [&](const char* what, double tol) {
                const auto got = down(out, q.size());
                double worst = 0, rms = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    worst = std::max(worst, std::fabs(got[i] - want[i]));
                    rms += want[i] * want[i];
                }
                rms = std::sqrt(rms / got.size());
                double top = 1.0;   // the tensor cores: inkling_kernels_test's bound, tol of the largest (at least 1)
                for (double w : want) top = std::max(top, std::fabs(w));
                const bool ok = tol >= 1e-3 ? worst <= tol * top : worst <= tol * rms;
                failures += !ok;
                std::printf("  %-44s largest difference %.2e (%.2e of the rms)  %s\n", what, worst, worst / rms,
                            ok ? "ok" : "FAIL");
            };
            if (nrel == 16) {
                ck(cudaMemset(out, 0, q.size() * 4), "zero");
                inkling_attend(dq, nullptr, nullptr, kq, vq, rel, ds, p0, T, out, kDim, kHeads, kKv, scale, 0, 0,
                               nullptr, 0, nullptr);
                compare("tensor cores, 40 tokens", 7.5e-3);
                const size_t wn = inkling_attend_work_floats(kHeads, kDim);
                float* work = dalloc<float>(wn);
                for (long long room : {(long long) N, 64LL}) {
                    ck(cudaMemset(out, 0, q.size() * 4), "zero");
                    for (int t = 0; t < T; t += 8) {
                        const size_t off = size_t(t) * kHeads;
                        inkling_attend(dq + off * kDim, nullptr, nullptr, kq, vq,
                                       InklingRel{dr + off * nrel, dt, nrel, kExtent}, ds + off, p0 + t, 8,
                                       out + off * kDim, kDim, kHeads, kKv, scale, 0, 0, work, wn, nullptr,
                                       room == 64 ? 0 : room);
                    }
                    compare(room == 64 ? "split, 8 tokens a call (room pos + 8)" : "split, 8 tokens a call (room 1500)",
                            1e-4);
                }
                cudaFree(work);
            } else {
                ck(cudaMemset(out, 0, q.size() * 4), "zero");
                inkling_attend(dq, nullptr, nullptr, kq, vq, rel, ds, p0, T, out, kDim, kHeads, kKv, scale, 0, 0,
                               nullptr, 0, nullptr);
                compare("unsplit (8 relative weights), 40 tokens", 1e-4);
            }
            cudaFree(dq), cudaFree(dr), cudaFree(ds), cudaFree(dt), cudaFree(out);
        }
        cudaFree(kc), cudaFree(ks), cudaFree(vc), cudaFree(vs);
    }
    std::printf(failures ? "%d FAILED\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
