// inkling2 fork, INKLING2_SITU_MMQ=1 (opt-in): a prompt batch's SiTU of the experts' gate and up rows straight into
// the down products' MMQ Q8_1 rows, one launch where there were two (situ_rows(), then Mmq::quantize()), without the
// float rows between them in VRAM.  The same bits as the two:
//   - The quantization is llama.cpp's quantize_mmq_q8_1 (ggml-cuda/quantize.cu), the same code, compiled as
//     ink_mmq compiles it, but for -use_fast_math's function mapping: ftz, approximate division and square root
//     (this file's options in CMakeLists.txt).
//   - SiTU is situ_one() of kernels_inkling.cu, which the engine compiles with IEEE division: here the divisions
//     are __fdiv_rn().  tanhf() and expf() are the library's in both (no fast math mapping).  Only a value that is
//     subnormal on the way could differ (ftz here, not there); INKLING2_SITU_MMQ=2 compares the two (situ_mmq_check()).
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include "strata/poly/kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::poly {

namespace {

__device__ __forceinline__ float situ_sigmoid(float x) { return __fdiv_rn(1.0f, 1.0f + expf(-x)); }

__device__ __forceinline__ float situ_value(float g, float u, float beta, float lb) {
    const float act = tanhf(g * __fdiv_rn(1.0f, beta)) * beta * situ_sigmoid(g);
    const float up = lb > 0.0f ? tanhf(u * __fdiv_rn(1.0f, lb)) * lb : u;
    return act * up;
}

__device__ __forceinline__ float2 situ_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

// quantize_mmq_q8_1<ds_layout, false> (ids null, one channel), its x a row of SiTU values computed here
template <mmq_q8_1_ds_layout ds_layout>
__global__ void situ_mmq_q8_1(const float* __restrict__ gate_up, void* __restrict__ vy, const int64_t ne00,
                              const int64_t ne0, const int ne1, float beta, float lb, ExpertScales s) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    ggml_cuda_pdl_sync();

    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs = i0 % QK8_1_MMQ;

    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < ne00) {
        const float* row = gate_up + (size_t) blockIdx.x * 2 * ne00;
        const float2 sc = situ_scales(s, (int) blockIdx.x);
        const float4 g = *(const float4*) (row + i0), u = *(const float4*) (row + ne00 + i0);
        xi.x = situ_value(g.x * sc.x, u.x * sc.y, beta, lb);
        xi.y = situ_value(g.y * sc.x, u.y * sc.y, beta, lb);
        xi.z = situ_value(g.z * sc.x, u.z * sc.y, beta, lb);
        xi.w = situ_value(g.w * sc.x, u.w * sc.y, beta, lb);
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    const int64_t ib = k_block * ne1 + blockIdx.x;
    char4* yqs4 = (char4*) y[ib].qs;
    yqs4[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d;
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) y[ib].ds4[iqs / 32] = make_half2(d, sum);
        else y[ib].d4[iqs / 32] = d;
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

}  // namespace

// rows of n gate then n up values (gate_up, 2n floats a row) into xq as Mmq::quantize(SiTU rows, nullptr, xq, type,
// n, n, rows) writes them.  False, with nothing launched, where that quantization is not Q8_1's (FP4 rows).
bool situ_mmq(const float* gate_up, void* xq, int type, int n, int rows, float beta, float lb,
              const ExpertScales* scales, cudaStream_t stream) {
    if (rows <= 0) return true;
    if (n % 4) return false;
    const int64_t ne0 = pad512(n);
    const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 blocks((unsigned) rows, (unsigned) block_num_y, 1), threads(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const ExpertScales s = scales ? *scales : ExpertScales{};
    switch (mmq_get_q8_1_ds_layout((ggml_type) type)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            situ_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4><<<blocks, threads, 0, stream>>>(gate_up, xq, n, ne0, rows, beta, lb, s);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            situ_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4><<<blocks, threads, 0, stream>>>(gate_up, xq, n, ne0, rows, beta, lb, s);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            situ_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6><<<blocks, threads, 0, stream>>>(gate_up, xq, n, ne0, rows, beta, lb, s);
            break;
        default:
            return false;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("situ_mmq: ") + cudaGetErrorString(e));
    return true;
}

// INKLING2_SITU_MMQ=2: the blocks situ_mmq() wrote to `fused` against the two launches' in `ref` (on the device),
// after the stream is done.  Says the first calls' results and every mismatch, and counts them.
void situ_mmq_check(const void* ref, const void* fused, int type, int n, int rows, cudaStream_t stream) {
    static long long calls = 0, bad = 0;
    const size_t bytes = (size_t) rows * (size_t) (pad512(n) / QK8_1_MMQ) * sizeof(block_q8_1_mmq);
    if (cudaStreamSynchronize(stream) != cudaSuccess) throw std::runtime_error("situ_mmq_check: the stream failed");
    std::vector<unsigned char> a(bytes), b(bytes);
    cudaMemcpy(a.data(), ref, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), fused, bytes, cudaMemcpyDeviceToHost);
    size_t diff = 0, first = bytes;
    for (size_t i = 0; i < bytes; ++i)
        if (a[i] != b[i]) {
            ++diff;
            if (first == bytes) first = i;
        }
    ++calls;
    if (diff) ++bad;
    if (diff || calls <= 3 || calls % 500 == 0)
        std::fprintf(stderr, "strata-inkling2: INKLING2_SITU_MMQ=2: call %lld (type %d, %d x %d): %zu of %zu bytes differ"
                             "%s; %lld of %lld calls differed so far\n", calls, type, rows, n, diff, bytes,
                     diff ? (" (first at " + std::to_string(first) + ")").c_str() : "", bad, calls);
}

}  // namespace strata::poly
