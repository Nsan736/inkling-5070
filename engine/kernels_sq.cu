// inkling2 fork: kernels for .strata files (Strata's SQ builds of Inkling 2 Mini).
//   mul_cols   each output row of a product times its own scale (a Q8_ROW matrix read as Q8_0 blocks of scale 1)
//   sq_gemv    y = W x for a matrix of S_K blocks (strata/sq/sq_types.hpp), x in floats: every weight decoded
//              exactly, D * (s / 255) * (q - HALF), and summed in floats (the output matrix, S6_K in SQ-10 / SQ-15)
#include "strata/poly/kernels.hpp"
#include "strata/sq/sq_types.hpp"
#include "kernels_sq_strata.cuh"
#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace strata::poly {

namespace {

cudaStream_t cs(void* s) { return static_cast<cudaStream_t>(s); }

__global__ void mul_cols_kernel(float* __restrict__ y, const float* __restrict__ s, int n_out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    const float f = s[i];
    for (int t = 0; t < n; ++t) y[(size_t) t * n_out + i] *= f;
}

// The code of value k (0..63) of 64-value group g of a block's payload pl, as Strata's unpacking
// (ink_hybrid.cuh dot_sk_sub64_h) lays it out.
template <int C>
__device__ __forceinline__ int sk_code(const uint8_t* __restrict__ pl, int g, int k) {
    if constexpr (C == sq::S2_K) {
        const uint8_t b = pl[g * 16 + (k >> 2)];
        return (b >> (2 * (k & 3))) & 3;
    } else if constexpr (C == sq::S3_K) {
        const int i = k >> 3, r = k & 7;
        const uint8_t b = pl[g * 16 + 2 * i + (r >> 2)], h = pl[64 + g * 8 + i];
        return ((b >> (2 * (r & 3))) & 3) | (((h >> r) & 1) << 2);
    } else if constexpr (C == sq::S4_K) {
        const uint8_t b = pl[g * 32 + (k >> 1)];
        return (k & 1) ? (b >> 4) : (b & 15);
    } else if constexpr (C == sq::S5_K) {
        const int i = k >> 3, r = k & 7;
        const uint8_t b = pl[g * 32 + 4 * i + (r >> 1)], h = pl[128 + g * 8 + i];
        return ((r & 1) ? (b >> 4) : (b & 15)) | (((h >> r) & 1) << 4);
    } else if constexpr (C == sq::S6_K) {
        const int i = k >> 2, r = k & 3;
        const uint8_t b = pl[g * 32 + 2 * i + (r >> 1)], h = pl[128 + g * 16 + i];
        return ((r & 1) ? (b >> 4) : (b & 15)) | (((h >> (2 * r)) & 3) << 4);
    } else {
        return (int) (int8_t) pl[g * 64 + k];
    }
}

constexpr int kRowsPerBlock = 8;   // a warp a row
constexpr int kTokens = 8;         // tokens a launch

template <int C>
__global__ void __launch_bounds__(256) sq_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out, int n) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.y;
    if (row >= n_out) return;
    constexpr int bb = sq::sk_block_bytes(C);
    constexpr float half = sq::sk_half(C);
    const int lane = threadIdx.x, nb = n_in / 256;
    const uint8_t* wr = w + (size_t) row * nb * bb;
    float acc[kTokens] = {};
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = wr + (size_t) b * bb;
        const float D = __half2float(*reinterpret_cast<const __half*>(blk));
        for (int e = lane; e < 256; e += 32) {
            const int g = e >> 6, k = e & 63;
            const float v = D * ((float) blk[2 + g] * (1.0f / 255.0f)) * ((float) sk_code<C>(blk + 6, g, k) - half);
            const int col = b * 256 + e;
            for (int t = 0; t < n; ++t) acc[t] = fmaf(v, x[(size_t) t * n_in + col], acc[t]);
        }
    }
    for (int t = 0; t < n; ++t) {
        float s = acc[t];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) y[(size_t) t * n_out + row] = s;
    }
}

template <int C>
void sq_gemv_launch(const uint8_t* w, const float* x, float* y, int n_in, int n_out, int n, void* stream) {
    const dim3 block(32, kRowsPerBlock), grid((unsigned) ((n_out + kRowsPerBlock - 1) / kRowsPerBlock));
    for (int at = 0; at < n; at += kTokens)
        sq_gemv_kernel<C><<<grid, block, 0, cs(stream)>>>(w, x + (size_t) at * n_in, y + (size_t) at * n_out, n_in,
                                                          n_out, n - at < kTokens ? n - at : kTokens);
}


// ---- a few tokens' experts in VRAM (moe(): the cached experts of a .strata layer) ----

__device__ __forceinline__ float e4m3(uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v;
    if (e == 15 && m == 7) v = 0.0f;
    else if (e == 0) v = (float) m * (1.0f / 512.0f);
    else v = (1.0f + (float) m * 0.125f) * ldexpf(1.0f, e - 7);
    return s ? -v : v;
}

__device__ __forceinline__ float e2m1(int n) {
    const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (n & 8) ? -t[n & 7] : t[n & 7];
}

// the dot product of an S_K row of 9 blocks with x (2304 floats): a lane's share, summed after
template <int C>
__device__ __forceinline__ float sk_row_dot(const uint8_t* __restrict__ row, const float* __restrict__ x, int lane) {
    constexpr int bb = sq::sk_block_bytes(C);
    constexpr float half = sq::sk_half(C);
    float acc = 0.0f;
    for (int b = 0; b < 9; ++b) {
        const uint8_t* blk = row + b * bb;
        const float D = __half2float(*reinterpret_cast<const __half*>(blk));
        for (int e = lane; e < 256; e += 32) {
            const int g = e >> 6, k = e & 63;
            acc = fmaf(D * ((float) blk[2 + g] * (1.0f / 255.0f)) * ((float) sk_code<C>(blk + 6, g, k) - half),
                       x[b * 256 + e], acc);
        }
    }
    return acc;
}

__device__ __forceinline__ float sf4_row_dot(const uint8_t* __restrict__ nib, const uint8_t* __restrict__ sc,
                                             const float* __restrict__ x, int cols, int lane) {
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        const uint8_t b = nib[c >> 1];
        acc = fmaf(e2m1((c & 1) ? (b >> 4) : (b & 15)) * e4m3(sc[c >> 4]), x[c], acc);
    }
    return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// act[j][i] = SiTU(gate_i x, up_i x) (/ sq_i for S_K) for job j's expert and token
template <int C>
__global__ void __launch_bounds__(256) sq_gu_kernel(SqJobs jobs, const float* __restrict__ x, size_t x_stride,
                                                     float* __restrict__ act, float beta, float lb) {
    const int i = blockIdx.x * 8 + threadIdx.y, j = blockIdx.y, lane = threadIdx.x;
    if (i >= kSqFf) return;
    const uint8_t* body = jobs.body[j];
    const float* xr = x + (size_t) jobs.row[j] * x_stride;
    float g, u, inv = 1.0f;
    if constexpr (C == sq::SF4) {
        const float* hdr = reinterpret_cast<const float*>(body);
        const float outer = hdr[0] / hdr[1];
        const uint8_t* nib = body + 16;
        const uint8_t* sc = nib + (size_t) (2 * kSqFf) * (kSqEmbd / 2);
        g = warp_sum(sf4_row_dot(nib + (size_t) (2 * i) * (kSqEmbd / 2), sc + (size_t) (2 * i) * (kSqEmbd / 16), xr,
                                 kSqEmbd, lane)) * outer;
        u = warp_sum(sf4_row_dot(nib + (size_t) (2 * i + 1) * (kSqEmbd / 2),
                                 sc + (size_t) (2 * i + 1) * (kSqEmbd / 16), xr, kSqEmbd, lane)) * outer;
    } else {
        constexpr size_t row_b = 9 * (size_t) sq::sk_block_bytes(C);
        const uint8_t* gr = body + kSqFf * 2 + (size_t) (2 * i) * row_b;
        g = warp_sum(sk_row_dot<C>(gr, xr, lane));
        u = warp_sum(sk_row_dot<C>(gr + row_b, xr, lane));
        inv = 1.0f / __half2float(reinterpret_cast<const __half*>(body)[i]);
    }
    if (lane == 0) {
        const float sig = 1.0f / (1.0f + expf(-g));
        act[(size_t) j * kSqFf + i] = (beta * tanhf(g / beta) * sig) * (lb * tanhf(u / lb)) * inv;
    }
}

// y[j][r] = down_r act[j] for job j's expert
template <int C>
__global__ void __launch_bounds__(256) sq_dn_kernel(SqJobs jobs, const float* __restrict__ act, float* __restrict__ y,
                                                     size_t y_stride) {
    const int r = blockIdx.x * 8 + threadIdx.y, j = blockIdx.y, lane = threadIdx.x;
    if (r >= kSqEmbd) return;
    const uint8_t* dn = jobs.body[j] + jobs.dn_off[j];
    const float* a = act + (size_t) j * kSqFf;
    float s = 0.0f;
    if constexpr (C == sq::SF4) {
        const float* hdr = reinterpret_cast<const float*>(jobs.body[j]);
        s = warp_sum(sf4_row_dot(dn + (size_t) r * (kSqFf / 2),
                                 dn + (size_t) kSqEmbd * (kSqFf / 2) + (size_t) r * (kSqFf / 16), a, kSqFf, lane)) *
            (hdr[2] / hdr[3]);
    } else {
        // rows 2p and 2p + 1 share the 9 blocks of pair p: row 2p is the pair's values [0, 1152), row 2p + 1 the rest
        constexpr int bb = sq::sk_block_bytes(C);
        constexpr float half = sq::sk_half(C);
        const uint8_t* pr = dn + (size_t) (r >> 1) * 9 * bb;
        const int base = (r & 1) * kSqFf;
        float acc = 0.0f;
        for (int c = lane; c < kSqFf; c += 32) {
            const int pe = base + c, b = pe >> 8, e = pe & 255, g = e >> 6, k = e & 63;
            const uint8_t* blk = pr + b * bb;
            const float D = __half2float(*reinterpret_cast<const __half*>(blk));
            acc = fmaf(D * ((float) blk[2 + g] * (1.0f / 255.0f)) * ((float) sk_code<C>(blk + 6, g, k) - half), a[c],
                       acc);
        }
        s = warp_sum(acc);
    }
    if (lane == 0) y[(size_t) j * y_stride + r] = s;
}


// ---- S_K experts by Strata's integer sums (kernels_sq_strata.cuh): x and h in Q8_1_G64 blocks, dp4a ----

// act[j][feat] = SiTU(gate x, up x) / sq for job j: 32 features a block, 4 lanes (a 64-value group each) a feature,
// as Strata's k_ink_moe_gu_sk for one token
template <int C>
__global__ void __launch_bounds__(128) sq_gu_dp_kernel(SqJobs jobs, const ink::InkQ8_1_G64* __restrict__ xq,
                                                        float* __restrict__ act) {
    const int j = blockIdx.y, tid = threadIdx.x, sub = tid >> 2, g = tid & 3, feat = blockIdx.x * 32 + sub;
    const uint8_t* body = jobs.body[j];
    constexpr size_t row_b = (size_t) (ink::kInkHidden / 256) * ink::ink_sb_bytes(C);
    const uint8_t* gr = body + ink::kInkInter * 2 + (size_t) (2 * feat) * row_b;
    __shared__ alignas(16) ink::InkQ8_1_G64 s_xq[ink::kInkHidden / 64];
    constexpr int N4 = (int) ((ink::kInkHidden / 64) * sizeof(ink::InkQ8_1_G64) / 16);
    const uint4* src = reinterpret_cast<const uint4*>(xq + (size_t) jobs.row[j] * (ink::kInkHidden / 64));
    for (int i = tid; i < N4; i += 128) reinterpret_cast<uint4*>(s_xq)[i] = src[i];
    __syncthreads();
    if (feat >= ink::kInkInter) return;
    const ink::InkQ8_1_G64* const xp[1] = {s_xq};
    float ag[1] = {}, au[1] = {};
    ink::sk_gu_fused_4lane<C, 1>(gr, gr + row_b, xp, 1, g, ag, au);
    if (g == 0) {
        const float inv_sq = 1.0f / ink::f16_to_f32(ink::ld16_u(body + feat * 2));
        act[(size_t) j * ink::kInkInter + feat] = ink::situ_f32(ag[0], au[0]) * inv_sq;
    }
}

// y[j][2p], y[j][2p + 1] = down rows 2p, 2p + 1 of job j's expert times its h, as Strata's k_ink_moe_dn_sk (without
// the weight: the engine's add_scaled() weighs and sums)
template <int C>
__global__ void __launch_bounds__(128) sq_dn_dp_kernel(SqJobs jobs, const ink::InkQ8_1_G64* __restrict__ hq,
                                                        float* __restrict__ y, size_t y_stride) {
    const int j = blockIdx.y, tid = threadIdx.x, sub = tid >> 2, g = tid & 3, pair_r = blockIdx.x * 32 + sub;
    constexpr size_t pair_b = 9ull * ink::ink_sb_bytes(C);
    const uint8_t* pr = jobs.body[j] + jobs.dn_off[j] + (size_t) pair_r * pair_b;
    __shared__ alignas(16) ink::InkQ8_1_G64 s_hq[ink::kInkInter / 64];
    constexpr int N4 = (int) ((ink::kInkInter / 64) * sizeof(ink::InkQ8_1_G64) / 16);
    const uint4* src = reinterpret_cast<const uint4*>(hq + (size_t) j * (ink::kInkInter / 64));
    for (int i = tid; i < N4; i += 128) reinterpret_cast<uint4*>(s_hq)[i] = src[i];
    __syncthreads();
    if (pair_r >= ink::kInkHidden / 2) return;
    const ink::InkQ8_1_G64* const hp[1] = {s_hq};
    float a0[1] = {}, a1[1] = {};
    ink::sk_dn_paired2_4lane<C, 1>(pr, hp, 1, g, a0, a1);
    if (g == 0) {
        y[(size_t) j * y_stride + 2 * pair_r] = a0[0];
        y[(size_t) j * y_stride + 2 * pair_r + 1] = a1[0];
    }
}

// The same for jobs of several S_K codes in one launch (each job's code its own: SqJobs::gu / dn), so that a
// token's cached experts are one launch whatever their formats
template <int C>
__device__ __forceinline__ void sq_gu_one(const uint8_t* body, int feat, int g, const ink::InkQ8_1_G64* const (&xp)[1],
                                          float (&ag)[1], float (&au)[1]) {
    constexpr size_t row_b = (size_t) (ink::kInkHidden / 256) * ink::ink_sb_bytes(C);
    const uint8_t* gr = body + ink::kInkInter * 2 + (size_t) (2 * feat) * row_b;
    ink::sk_gu_fused_4lane<C, 1>(gr, gr + row_b, xp, 1, g, ag, au);
}

template <int C>
__device__ __forceinline__ void sq_dn_one(const uint8_t* body, uint32_t dn_off, int pair_r, int g,
                                          const ink::InkQ8_1_G64* const (&hp)[1], float (&a0)[1], float (&a1)[1]) {
    constexpr size_t pair_b = 9ull * ink::ink_sb_bytes(C);
    ink::sk_dn_paired2_4lane<C, 1>(body + dn_off + (size_t) pair_r * pair_b, hp, 1, g, a0, a1);
}

__global__ void __launch_bounds__(128) sq_gu_dp_any_kernel(SqJobs jobs, const ink::InkQ8_1_G64* __restrict__ xq,
                                                           float* __restrict__ act) {
    const int j = blockIdx.y, tid = threadIdx.x, sub = tid >> 2, g = tid & 3, feat = blockIdx.x * 32 + sub;
    const uint8_t* body = jobs.body[j];
    __shared__ alignas(16) ink::InkQ8_1_G64 s_xq[ink::kInkHidden / 64];
    constexpr int N4 = (int) ((ink::kInkHidden / 64) * sizeof(ink::InkQ8_1_G64) / 16);
    const uint4* src = reinterpret_cast<const uint4*>(xq + (size_t) jobs.row[j] * (ink::kInkHidden / 64));
    for (int i = tid; i < N4; i += 128) reinterpret_cast<uint4*>(s_xq)[i] = src[i];
    __syncthreads();
    if (feat >= ink::kInkInter) return;
    const ink::InkQ8_1_G64* const xp[1] = {s_xq};
    float ag[1] = {}, au[1] = {};
    switch (jobs.gu[j]) {
    case sq::S2_K: sq_gu_one<sq::S2_K>(body, feat, g, xp, ag, au); break;
    case sq::S3_K: sq_gu_one<sq::S3_K>(body, feat, g, xp, ag, au); break;
    case sq::S4_K: sq_gu_one<sq::S4_K>(body, feat, g, xp, ag, au); break;
    case sq::S5_K: sq_gu_one<sq::S5_K>(body, feat, g, xp, ag, au); break;
    case sq::S6_K: sq_gu_one<sq::S6_K>(body, feat, g, xp, ag, au); break;
    default: sq_gu_one<sq::S8_K>(body, feat, g, xp, ag, au); break;
    }
    if (g == 0) {
        const float inv_sq = 1.0f / ink::f16_to_f32(ink::ld16_u(body + feat * 2));
        act[(size_t) j * ink::kInkInter + feat] = ink::situ_f32(ag[0], au[0]) * inv_sq;
    }
}

__global__ void __launch_bounds__(128) sq_dn_dp_any_kernel(SqJobs jobs, const ink::InkQ8_1_G64* __restrict__ hq,
                                                           float* __restrict__ y, size_t y_stride) {
    const int j = blockIdx.y, tid = threadIdx.x, sub = tid >> 2, g = tid & 3, pair_r = blockIdx.x * 32 + sub;
    __shared__ alignas(16) ink::InkQ8_1_G64 s_hq[ink::kInkInter / 64];
    constexpr int N4 = (int) ((ink::kInkInter / 64) * sizeof(ink::InkQ8_1_G64) / 16);
    const uint4* src = reinterpret_cast<const uint4*>(hq + (size_t) j * (ink::kInkInter / 64));
    for (int i = tid; i < N4; i += 128) reinterpret_cast<uint4*>(s_hq)[i] = src[i];
    __syncthreads();
    if (pair_r >= ink::kInkHidden / 2) return;
    const ink::InkQ8_1_G64* const hp[1] = {s_hq};
    float a0[1] = {}, a1[1] = {};
    const uint8_t* body = jobs.body[j];
    const uint32_t off = jobs.dn_off[j];
    switch (jobs.dn[j]) {
    case sq::S2_K: sq_dn_one<sq::S2_K>(body, off, pair_r, g, hp, a0, a1); break;
    case sq::S3_K: sq_dn_one<sq::S3_K>(body, off, pair_r, g, hp, a0, a1); break;
    case sq::S4_K: sq_dn_one<sq::S4_K>(body, off, pair_r, g, hp, a0, a1); break;
    case sq::S5_K: sq_dn_one<sq::S5_K>(body, off, pair_r, g, hp, a0, a1); break;
    case sq::S6_K: sq_dn_one<sq::S6_K>(body, off, pair_r, g, hp, a0, a1); break;
    default: sq_dn_one<sq::S8_K>(body, off, pair_r, g, hp, a0, a1); break;
    }
    if (g == 0) {
        y[(size_t) j * y_stride + 2 * pair_r] = a0[0];
        y[(size_t) j * y_stride + 2 * pair_r + 1] = a1[0];
    }
}

// SF4 gate|up and down for one token a job, as Strata's k_ink_moe_gu_sf4_mx8 / k_ink_moe_dn_sf4_mx8 run M = 1
// (their n_e == 1 path: x and h rounded to halves, 16 products a half2 sum): 16 features or row pairs a block,
// 8 lanes each
__global__ void __launch_bounds__(128) sq_gu_sf4_h_kernel(SqJobs jobs, const float* __restrict__ x, size_t x_stride,
                                                          float* __restrict__ act) {
    const int j = blockIdx.y, tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, feat = blockIdx.x * 16 + sub;
    const uint8_t* body = jobs.body[j];
    __shared__ alignas(16) __half2 s_x[ink::kInkHidden / 2];
    const float2* src2 = reinterpret_cast<const float2*>(x + (size_t) jobs.row[j] * x_stride);
    for (int i = tid; i < ink::kInkHidden / 2; i += 128) {
        const float2 f = src2[i];
        s_x[i] = __floats2half2_rn(f.x, f.y);
    }
    __syncthreads();
    if (feat >= ink::kInkInter) return;
    const float2 h01 = *reinterpret_cast<const float2*>(body);
    const float w_outer = h01.x / h01.y;
    constexpr size_t W_P_ROW = ink::kInkHidden / 2, W_S_ROW = ink::kInkHidden / 16;
    const uint8_t* base_p = body + 16;
    const uint8_t* base_s = base_p + (size_t) (2 * ink::kInkInter) * W_P_ROW;
    const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t) (2 * feat) * W_P_ROW);
    const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t) (2 * feat + 1) * W_P_ROW);
    const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t) (2 * feat) * W_S_ROW);
    const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t) (2 * feat + 1) * W_S_ROW);
    float ag0 = 0.0f, au0 = 0.0f;
    for (int i = lg; i < 72; i += 8) {
        const uint4 gv = gp16[i], uv = up16[i];
        const float2 sg = ink::e4m3x2_to_f32x2(gs2[i]);
        const float2 su = ink::e4m3x2_to_f32x2(us2[i]);
        __half2 gw[8], uw[8];
        const __half2* xh0 = s_x + i * 16;
        ink::sf4_decode16_h2(gv.x, gv.y, gw);
        ink::sf4_decode16_h2(uv.x, uv.y, uw);
        ag0 = fmaf(ink::dot16_h2_2acc(gw, xh0), sg.x, ag0);
        au0 = fmaf(ink::dot16_h2_2acc(uw, xh0), su.x, au0);
        ink::sf4_decode16_h2(gv.z, gv.w, gw);
        ink::sf4_decode16_h2(uv.z, uv.w, uw);
        ag0 = fmaf(ink::dot16_h2_2acc(gw, xh0 + 8), sg.y, ag0);
        au0 = fmaf(ink::dot16_h2_2acc(uw, xh0 + 8), su.y, au0);
    }
    ag0 *= w_outer;
    au0 *= w_outer;
#pragma unroll
    for (int o = 4; o >= 1; o >>= 1) {
        ag0 += __shfl_xor_sync(0xFFFFFFFFu, ag0, o);
        au0 += __shfl_xor_sync(0xFFFFFFFFu, au0, o);
    }
    if (lg == 0) act[(size_t) j * ink::kInkInter + feat] = ink::situ_f32(ag0, au0);
}

__global__ void __launch_bounds__(128) sq_dn_sf4_h_kernel(SqJobs jobs, const float* __restrict__ act,
                                                          float* __restrict__ y, size_t y_stride) {
    const int j = blockIdx.y, tid = threadIdx.x, sub = tid >> 3, lg = tid & 7, pair_r = blockIdx.x * 16 + sub;
    const uint8_t* body = jobs.body[j];
    __shared__ alignas(16) __half2 s_h[ink::kInkInter / 2];
    const float2* src2 = reinterpret_cast<const float2*>(act + (size_t) j * ink::kInkInter);
    for (int i = tid; i < ink::kInkInter / 2; i += 128) {
        const float2 f = src2[i];
        s_h[i] = __floats2half2_rn(f.x, f.y);
    }
    __syncthreads();
    if (pair_r >= ink::kInkHidden / 2) return;
    const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
    const float w_outer = h23.x / h23.y;
    constexpr size_t W_P_ROW = ink::kInkInter / 2, W_S_ROW = ink::kInkInter / 16;
    const uint8_t* base_p = body + jobs.dn_off[j];
    const uint8_t* base_s = base_p + (size_t) ink::kInkHidden * W_P_ROW;
    const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t) (2 * pair_r) * W_P_ROW);
    const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t) (2 * pair_r + 1) * W_P_ROW);
    const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t) (2 * pair_r) * W_S_ROW);
    const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t) (2 * pair_r + 1) * W_S_ROW);
    float y0 = 0.0f, y1 = 0.0f;
    for (int i = lg; i < 36; i += 8) {
        const uint4 v0 = p0[i], v1 = p1[i];
        const float2 sc0 = ink::e4m3x2_to_f32x2(s0[i]);
        const float2 sc1 = ink::e4m3x2_to_f32x2(s1[i]);
        __half2 w0[8], w1[8];
        const __half2* hh = s_h + i * 16;
        ink::sf4_decode16_h2(v0.x, v0.y, w0);
        ink::sf4_decode16_h2(v1.x, v1.y, w1);
        y0 = fmaf(ink::dot16_h2_2acc(w0, hh), sc0.x, y0);
        y1 = fmaf(ink::dot16_h2_2acc(w1, hh), sc1.x, y1);
        ink::sf4_decode16_h2(v0.z, v0.w, w0);
        ink::sf4_decode16_h2(v1.z, v1.w, w1);
        y0 = fmaf(ink::dot16_h2_2acc(w0, hh + 8), sc0.y, y0);
        y1 = fmaf(ink::dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
    }
    y0 *= w_outer;
    y1 *= w_outer;
#pragma unroll
    for (int o = 4; o >= 1; o >>= 1) {
        y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
        y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
    }
    if (lg == 0) {
        y[(size_t) j * y_stride + 2 * pair_r] = y0;
        y[(size_t) j * y_stride + 2 * pair_r + 1] = y1;
    }
}

}  // namespace

void sq_experts_sf4_h(const SqJobs& jobs, int n, const float* x, size_t x_stride, float* act, float* y,
                      size_t y_stride, void* stream) {
    if (n <= 0) return;
    sq_gu_sf4_h_kernel<<<dim3((unsigned) (kSqFf / 16), (unsigned) n), 128, 0, cs(stream)>>>(jobs, x, x_stride, act);
    sq_dn_sf4_h_kernel<<<dim3((unsigned) (kSqEmbd / 2 / 16), (unsigned) n), 128, 0, cs(stream)>>>(jobs, act, y, y_stride);
}

void sq_experts_gu_dp_any(const SqJobs& jobs, int n, const void* xq, float* act, void* stream) {
    if (n <= 0) return;
    const dim3 grid((unsigned) ((kSqFf + 31) / 32), (unsigned) n);
    sq_gu_dp_any_kernel<<<grid, 128, 0, cs(stream)>>>(jobs, (const ink::InkQ8_1_G64*) xq, act);
}

void sq_experts_dn_dp_any(const SqJobs& jobs, int n, const void* hq, float* y, size_t y_stride, void* stream) {
    if (n <= 0) return;
    const dim3 grid((unsigned) ((kSqEmbd / 2 + 31) / 32), (unsigned) n);
    sq_dn_dp_any_kernel<<<grid, 128, 0, cs(stream)>>>(jobs, (const ink::InkQ8_1_G64*) hq, y, y_stride);
}

void sq_quantize_g64(const float* x, void* q, int rows, int cols, void* stream) {
    if (rows <= 0) return;
    const int groups = rows * (cols / 64);
    ink::k_ink_quant_q8_1<<<(groups + 7) / 8, 256, 0, cs(stream)>>>(x, (ink::InkQ8_1_G64*) q, rows, cols);
}

void sq_experts_gu_dp(int code, const SqJobs& jobs, int n, const void* xq, float* act, void* stream) {
    if (n <= 0) return;
    const dim3 grid((unsigned) ((kSqFf + 31) / 32), (unsigned) n);
    const auto* x = (const ink::InkQ8_1_G64*) xq;
    switch (code) {
#define SQ_GU(C) \
    case C: sq_gu_dp_kernel<C><<<grid, 128, 0, cs(stream)>>>(jobs, x, act); return;
        SQ_GU(sq::S2_K) SQ_GU(sq::S3_K) SQ_GU(sq::S4_K) SQ_GU(sq::S5_K) SQ_GU(sq::S6_K) SQ_GU(sq::S8_K)
#undef SQ_GU
    default: throw std::invalid_argument("sq_experts_gu_dp: not an S_K code");
    }
}

void sq_experts_dn_dp(int code, const SqJobs& jobs, int n, const void* hq, float* y, size_t y_stride, void* stream) {
    if (n <= 0) return;
    const dim3 grid((unsigned) ((kSqEmbd / 2 + 31) / 32), (unsigned) n);
    const auto* h = (const ink::InkQ8_1_G64*) hq;
    switch (code) {
#define SQ_DN(C) \
    case C: sq_dn_dp_kernel<C><<<grid, 128, 0, cs(stream)>>>(jobs, h, y, y_stride); return;
        SQ_DN(sq::S2_K) SQ_DN(sq::S3_K) SQ_DN(sq::S4_K) SQ_DN(sq::S5_K) SQ_DN(sq::S6_K) SQ_DN(sq::S8_K)
#undef SQ_DN
    default: throw std::invalid_argument("sq_experts_dn_dp: not an S_K code");
    }
}

namespace {
}  // namespace

void sq_experts_gu(int code, const SqJobs& jobs, int n, const float* x, size_t x_stride, float* act, float beta,
                   float lb, void* stream) {
    if (n <= 0) return;
    if (n > kSqJobs) throw std::invalid_argument("sq_experts_gu: more than kSqJobs experts");
    const dim3 block(32, 8), grid((unsigned) ((kSqFf + 7) / 8), (unsigned) n);
    switch (code) {
#define SQ_GU(C)                                                                                     \
    case C: sq_gu_kernel<C><<<grid, block, 0, cs(stream)>>>(jobs, x, x_stride, act, beta, lb); return;
        SQ_GU(sq::S2_K) SQ_GU(sq::S3_K) SQ_GU(sq::S4_K) SQ_GU(sq::S5_K) SQ_GU(sq::S6_K) SQ_GU(sq::S8_K) SQ_GU(sq::SF4)
#undef SQ_GU
    default: throw std::invalid_argument("sq_experts_gu: not an expert code");
    }
}

void sq_experts_dn(int code, const SqJobs& jobs, int n, const float* act, float* y, size_t y_stride, void* stream) {
    if (n <= 0) return;
    if (n > kSqJobs) throw std::invalid_argument("sq_experts_dn: more than kSqJobs experts");
    const dim3 block(32, 8), grid((unsigned) ((kSqEmbd + 7) / 8), (unsigned) n);
    switch (code) {
#define SQ_DN(C) \
    case C: sq_dn_kernel<C><<<grid, block, 0, cs(stream)>>>(jobs, act, y, y_stride); return;
        SQ_DN(sq::S2_K) SQ_DN(sq::S3_K) SQ_DN(sq::S4_K) SQ_DN(sq::S5_K) SQ_DN(sq::S6_K) SQ_DN(sq::S8_K) SQ_DN(sq::SF4)
#undef SQ_DN
    default: throw std::invalid_argument("sq_experts_dn: not an expert code");
    }
}

void mul_cols(float* y, const float* s, int n_out, int n, void* stream) {
    if (n_out <= 0 || n <= 0) return;
    mul_cols_kernel<<<(n_out + 255) / 256, 256, 0, cs(stream)>>>(y, s, n_out, n);
}

void sq_gemv(int code, const uint8_t* w, const float* x, float* y, int n_in, int n_out, int n, void* stream) {
    if (n_in % 256) throw std::invalid_argument("sq_gemv: a row is not whole 256-value blocks");
    switch (code) {
    case sq::S2_K: return sq_gemv_launch<sq::S2_K>(w, x, y, n_in, n_out, n, stream);
    case sq::S3_K: return sq_gemv_launch<sq::S3_K>(w, x, y, n_in, n_out, n, stream);
    case sq::S4_K: return sq_gemv_launch<sq::S4_K>(w, x, y, n_in, n_out, n, stream);
    case sq::S5_K: return sq_gemv_launch<sq::S5_K>(w, x, y, n_in, n_out, n, stream);
    case sq::S6_K: return sq_gemv_launch<sq::S6_K>(w, x, y, n_in, n_out, n, stream);
    case sq::S8_K: return sq_gemv_launch<sq::S8_K>(w, x, y, n_in, n_out, n, stream);
    default: throw std::invalid_argument("sq_gemv: not an S_K code");
    }
}

// ---- a prompt batch's experts by Strata's tensor-core kernels (kernels_sq_strata.cuh) ----

static_assert(sizeof(SqTcExpert) == sizeof(ink::InkExpertDev), "SqTcExpert is Strata's InkExpertDev");

namespace {

ink::InkMoeTables tables(const SqTcTables& t) {
    ink::InkMoeTables tb;
    tb.cnt = t.cnt;
    tb.off = t.off;
    tb.active = t.active;
    tb.nact = t.nact;
    tb.pairs = t.pairs;
    tb.tok_idx = t.tok_idx;
    tb.n_tiles = t.n_tiles;
    tb.tiles = reinterpret_cast<int2*>(t.tiles);
    return tb;
}

// out[t] += the pairs of token t whose expert is in this round (tier 0), in the order of its places
__global__ void sq_rows_add_kernel(const float* __restrict__ y, const int32_t* __restrict__ ids,
                                   const ink::InkExpertDev* __restrict__ ex, float* __restrict__ out, int used) {
    const int t = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= ink::kInkHidden) return;
    float s = 0.f;
    for (int k = 0; k < used; ++k) {
        const int i = t * used + k;
        if (ex[ids[i]].tier == 0) s += y[(size_t) i * ink::kInkHidden + d];
    }
    out[(size_t) t * ink::kInkHidden + d] += s;
}

}  // namespace

void sq_rows_round(const SqTcTables& t, int n_tiles, unsigned kinds, const float* x, int m, void* xq, float* h,
                   void* hq, int pairs, float* y, void* stream) {
    if (n_tiles <= 0) return;
    const cudaStream_t st = cs(stream);
    const auto* ex = (const ink::InkExpertDev*) t.experts;
    const ink::InkMoeTables tb = tables(t);
    const unsigned grid = (unsigned) std::min(std::max(1, n_tiles * 18), 2048);
    auto* xq8 = (ink::InkQ8_1_G64*) xq;
    auto* hq8 = (ink::InkQ8_1_G64*) hq;
    if (kinds & kSqTcGuSf4)
        ink::k_ink_moe_sf4_mx8_fp16tc<true><<<grid, 256, 0, st>>>(nullptr, ex, tb, x, nullptr, h);
    if (kinds & kSqTcGuSk) {
        const int nb = m * (ink::kInkHidden / 64);
        ink::k_ink_quant_q8_1<<<(nb + 7) / 8, 256, 0, st>>>(x, xq8, m, ink::kInkHidden);
        ink::k_ink_moe_sk_tc<true, 16><<<grid, 256, 0, st>>>(nullptr, ex, tb, xq8, nullptr, h);
    }
    if (kinds & kSqTcDnSf4)
        ink::k_ink_moe_sf4_mx8_fp16tc<false><<<grid, 256, 0, st>>>(nullptr, ex, tb, h, t.w, y);
    if (kinds & kSqTcDnSk) {
        const int nb = pairs * (ink::kInkInter / 64);
        ink::k_ink_quant_q8_1<<<(nb + 7) / 8, 256, 0, st>>>(h, hq8, pairs, ink::kInkInter);
        ink::k_ink_moe_sk_tc<false, 16><<<grid, 256, 0, st>>>(nullptr, ex, tb, hq8, t.w, y);
    }
}

void sq_rows_add(const float* y, const int32_t* ids, const void* experts, float* out, int m, int used, void* stream) {
    if (m <= 0) return;
    const dim3 grid((unsigned) ((ink::kInkHidden + 255) / 256), (unsigned) m);
    sq_rows_add_kernel<<<grid, 256, 0, cs(stream)>>>(y, ids, (const ink::InkExpertDev*) experts, out, used);
}

}  // namespace strata::poly
