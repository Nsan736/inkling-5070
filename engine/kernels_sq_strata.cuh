// inkling2 fork: Strata v0.2.19's device helpers for .strata S_K experts, copied unchanged from
// engine/src/inkling2/cuda/ink_cuda.cuh and ink_moe_cuda.cuh (MIT, Copyright (c) 2026 Niko1221 and the Strata
// contributors): the Q8_1_G64 activation blocks and their quantizer, the S_K unpacking into int8x4 words, the dp4a
// sums of a 64-value group, and the 4-lane gate|up and paired-down dot products of its decode kernels
// (k_ink_moe_gu_sk / k_ink_moe_dn_sk), and its prompt batches' tensor-core kernels
// (k_ink_moe_sk_tc, k_ink_moe_sf4_mx8_fp16tc).  kernels_sq.cu launches them for this engine's jobs.
// Made by extracting those definitions whole (the inkling-mini-strata workspace's scratch script).
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>

namespace strata::ink {

constexpr int kInkHidden = 2304;
constexpr int kInkInter = 1152;
constexpr float kInkSituBeta = 4.0f;
constexpr float kInkSituLb = 25.0f;

enum InkFmt : int {
    FMT_S1_25 = 0,
    FMT_S1_5  = 1,
    FMT_S2_K  = 2,
    FMT_S3_K  = 3,
    FMT_S4_K  = 4,
    FMT_S5_K  = 5,
    FMT_S6_K  = 6,
    FMT_S8_K  = 8,
    FMT_SF4   = 15,
    FMT_MX8   = 16,
    FMT_Q8ROW = 100,
    FMT_F32   = 101,
    FMT_BF16  = 102,
    FMT_F16   = 103,
};

__host__ __device__ constexpr int ink_sb_bytes(int fmt) {
    switch (fmt) {
        case FMT_S1_25: return 48;
        case FMT_S1_5:  return 58;
        case FMT_S2_K:  return 70;
        case FMT_S3_K:  return 102;
        case FMT_S4_K:  return 134;
        case FMT_S5_K:  return 166;
        case FMT_S6_K:  return 198;
        case FMT_S8_K:  return 262;
        default:        return 0;
    }
}

__host__ __device__ constexpr float ink_sk_half(int fmt) {
    switch (fmt) {
        case FMT_S2_K: return 1.5f;
        case FMT_S3_K: return 3.5f;
        case FMT_S4_K: return 7.5f;
        case FMT_S5_K: return 15.5f;
        case FMT_S6_K: return 31.5f;
        default:       return 0.0f;
    }
}

__device__ __forceinline__ float f16_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float situ_f32(float g, float u) {
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float ga = kInkSituBeta * tanhf(g * (1.0f / kInkSituBeta)) * sig;
    const float ua = kInkSituLb * tanhf(u * (1.0f / kInkSituLb));
    return ga * ua;
}

__device__ __forceinline__ int32_t dp4a_i8(uint32_t a, uint32_t b, int32_t c) {
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 610) || (__CUDA_ARCH__ >= 750))
    return __dp4a((int)a, (int)b, c);
#else
    const int32_t a0 = (int32_t)(int8_t)(a),        b0 = (int32_t)(int8_t)(b);
    const int32_t a1 = (int32_t)(int8_t)(a >> 8),   b1 = (int32_t)(int8_t)(b >> 8);
    const int32_t a2 = (int32_t)(int8_t)(a >> 16),  b2 = (int32_t)(int8_t)(b >> 16);
    const int32_t a3 = (int32_t)(int8_t)(a >> 24),  b3 = (int32_t)(int8_t)(b >> 24);
    return c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

__device__ __forceinline__ uint16_t ld16_u(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}

__device__ __forceinline__ uint32_t ld32_u(const uint8_t* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t)p16[0] | ((uint32_t)p16[1] << 16);
}

struct alignas(16) InkQ8_1_G64 {
    int8_t qs[64];
    float  d;
    float  s;  // d * sum(qs)
};

__device__ __constant__ int8_t kS15TritLut[256][5] = {
#define _T5(b) { \
    (int8_t)(((b) % 3) - 1), \
    (int8_t)((((b) / 3) % 3) - 1), \
    (int8_t)((((b) / 9) % 3) - 1), \
    (int8_t)((((b) / 27) % 3) - 1), \
    (int8_t)((((b) / 81) % 3) - 1) }
#define _T5_8(x) _T5(x), _T5((x)+1), _T5((x)+2), _T5((x)+3), _T5((x)+4), _T5((x)+5), _T5((x)+6), _T5((x)+7)
#define _T5_32(x) _T5_8(x), _T5_8((x)+8), _T5_8((x)+16), _T5_8((x)+24)
    _T5_32(0), _T5_32(32), _T5_32(64), _T5_32(96), _T5_32(128), _T5_32(160), _T5_32(192), _T5_32(224)
#undef _T5_32
#undef _T5_8
#undef _T5
};

__global__ void __launch_bounds__(256) k_ink_quant_q8_1(
    const float* __restrict__ x,
    InkQ8_1_G64* __restrict__ out,
    int rows,
    int cols) {
    const int ng = cols / 64;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int total = rows * ng;
    if (w >= total) return;
    const float* xb = x + (size_t)w * 64;
    const float v0 = xb[2 * lane];
    const float v1 = xb[2 * lane + 1];
    float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
    const float d = am * (1.0f / 127.0f);
    const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
    int q0 = max(-127, min(127, __float2int_rn(v0 * inv)));
    int q1 = max(-127, min(127, __float2int_rn(v1 * inv)));
    int s = q0 + q1;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    InkQ8_1_G64& dst = out[w];
    dst.qs[2 * lane]     = (int8_t)q0;
    dst.qs[2 * lane + 1] = (int8_t)q1;
    if (lane == 0) {
        dst.d = d;
        dst.s = d * (float)s;
    }
}

template <int FMT>
__device__ __forceinline__ void sk_unpack64_g_to_u32(const uint8_t* pl, int g, uint32_t (&w)[16]) {
    if constexpr (FMT == FMT_S2_K) {
        const uint8_t* p = pl + g * 16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint32_t b0 = q & 0xFFu, b1 = (q >> 8) & 0xFFu, b2 = (q >> 16) & 0xFFu, b3 = q >> 24;
            w[4*i + 0] = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
            w[4*i + 1] = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
            w[4*i + 2] = (b2 & 3u) | (((b2 >> 2) & 3u) << 8) | (((b2 >> 4) & 3u) << 16) | (((b2 >> 6) & 3u) << 24);
            w[4*i + 3] = (b3 & 3u) | (((b3 >> 2) & 3u) << 8) | (((b3 >> 4) & 3u) << 16) | (((b3 >> 6) & 3u) << 24);
        }
    } else if constexpr (FMT == FMT_S3_K) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint16_t h = ld16_u(hp + i * 2);
            const uint32_t b0 = q & 0xFFu, b1 = (q >> 8) & 0xFFu, b2 = (q >> 16) & 0xFFu, b3 = q >> 24;
            const uint32_t h0 = h & 0xFFu, h1 = h >> 8;
            w[4*i + 0] = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24)
                       | ((h0 & 1u) << 2) | (((h0 >> 1) & 1u) << 10) | (((h0 >> 2) & 1u) << 18) | (((h0 >> 3) & 1u) << 26);
            w[4*i + 1] = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24)
                       | (((h0 >> 4) & 1u) << 2) | (((h0 >> 5) & 1u) << 10) | (((h0 >> 6) & 1u) << 18) | (((h0 >> 7) & 1u) << 26);
            w[4*i + 2] = (b2 & 3u) | (((b2 >> 2) & 3u) << 8) | (((b2 >> 4) & 3u) << 16) | (((b2 >> 6) & 3u) << 24)
                       | ((h1 & 1u) << 2) | (((h1 >> 1) & 1u) << 10) | (((h1 >> 2) & 1u) << 18) | (((h1 >> 3) & 1u) << 26);
            w[4*i + 3] = (b3 & 3u) | (((b3 >> 2) & 3u) << 8) | (((b3 >> 4) & 3u) << 16) | (((b3 >> 6) & 3u) << 24)
                       | (((h1 >> 4) & 1u) << 2) | (((h1 >> 5) & 1u) << 10) | (((h1 >> 6) & 1u) << 18) | (((h1 >> 7) & 1u) << 26);
        }
    } else if constexpr (FMT == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            w[2*i + 0] = (q & 0x0Fu) | (((q >> 4) & 0x0Fu) << 8) | (((q >> 8) & 0x0Fu) << 16) | (((q >> 12) & 0x0Fu) << 24);
            w[2*i + 1] = ((q >> 16) & 0x0Fu) | (((q >> 20) & 0x0Fu) << 8) | (((q >> 24) & 0x0Fu) << 16) | (((q >> 28) & 0x0Fu) << 24);
        }
    } else if constexpr (FMT == FMT_S5_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint8_t hb = hp[i];
            w[2*i + 0] = (q & 0x0Fu) | (((q >> 4) & 0x0Fu) << 8) | (((q >> 8) & 0x0Fu) << 16) | (((q >> 12) & 0x0Fu) << 24)
                       | ((hb & 1u) << 4) | (((hb >> 1) & 1u) << 12) | (((hb >> 2) & 1u) << 20) | (((hb >> 3) & 1u) << 28);
            w[2*i + 1] = ((q >> 16) & 0x0Fu) | (((q >> 20) & 0x0Fu) << 8) | (((q >> 24) & 0x0Fu) << 16) | (((q >> 28) & 0x0Fu) << 24)
                       | (((hb >> 4) & 1u) << 4) | (((hb >> 5) & 1u) << 12) | (((hb >> 6) & 1u) << 20) | (((hb >> 7) & 1u) << 28);
        }
    } else if constexpr (FMT == FMT_S6_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = ld32_u(p + i * 4);
            const uint16_t h16 = ld16_u(hp + i * 2);
            w[2*i + 0] = (q & 0x0Fu) | (((q >> 4) & 0x0Fu) << 8) | (((q >> 8) & 0x0Fu) << 16) | (((q >> 12) & 0x0Fu) << 24)
                       | ((h16 & 3u) << 4) | (((h16 >> 2) & 3u) << 12) | (((h16 >> 4) & 3u) << 20) | (((h16 >> 6) & 3u) << 28);
            w[2*i + 1] = ((q >> 16) & 0x0Fu) | (((q >> 20) & 0x0Fu) << 8) | (((q >> 24) & 0x0Fu) << 16) | (((q >> 28) & 0x0Fu) << 24)
                       | (((h16 >> 8) & 3u) << 4) | (((h16 >> 10) & 3u) << 12) | (((h16 >> 12) & 3u) << 20) | (((h16 >> 14) & 3u) << 28);
        }
    } else if constexpr (FMT == FMT_S8_K) {
        const uint8_t* p = pl + g * 64;
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = ld32_u(p + i * 4);
    } else if constexpr (FMT == FMT_S1_5) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int b0 = g * 64 + i * 4;
            uint32_t t0 = (uint8_t)kS15TritLut[pl[(b0 + 0) / 5]][(b0 + 0) % 5];
            uint32_t t1 = (uint8_t)kS15TritLut[pl[(b0 + 1) / 5]][(b0 + 1) % 5];
            uint32_t t2 = (uint8_t)kS15TritLut[pl[(b0 + 2) / 5]][(b0 + 2) % 5];
            uint32_t t3 = (uint8_t)kS15TritLut[pl[(b0 + 3) / 5]][(b0 + 3) % 5];
            w[i] = t0 | (t1 << 8) | (t2 << 16) | (t3 << 24);
        }
    }
}

__device__ __forceinline__ int32_t dp64_i8_smem(const uint32_t (&w)[16], const InkQ8_1_G64& xb) {
    const uint4* x4 = reinterpret_cast<const uint4*>(xb.qs);
    const uint4 v0 = x4[0], v1 = x4[1], v2 = x4[2], v3 = x4[3];
    int32_t s0 = dp4a_i8(w[0],  v0.x, 0);
    int32_t s1 = dp4a_i8(w[1],  v0.y, 0);
    s0 = dp4a_i8(w[2],  v0.z, s0); s1 = dp4a_i8(w[3],  v0.w, s1);
    s0 = dp4a_i8(w[4],  v1.x, s0); s1 = dp4a_i8(w[5],  v1.y, s1);
    s0 = dp4a_i8(w[6],  v1.z, s0); s1 = dp4a_i8(w[7],  v1.w, s1);
    s0 = dp4a_i8(w[8],  v2.x, s0); s1 = dp4a_i8(w[9],  v2.y, s1);
    s0 = dp4a_i8(w[10], v2.z, s0); s1 = dp4a_i8(w[11], v2.w, s1);
    s0 = dp4a_i8(w[12], v3.x, s0); s1 = dp4a_i8(w[13], v3.y, s1);
    s0 = dp4a_i8(w[14], v3.z, s0); s1 = dp4a_i8(w[15], v3.w, s1);
    return s0 + s1;
}

template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_gu_fused_4lane(
    const uint8_t* gr, const uint8_t* ur,
    const InkQ8_1_G64* const (&xp)[MAX_M],
    int n_e, int g,
    float (&ag)[MAX_M], float (&au)[MAX_M]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    constexpr bool HAS_HALF = (FMT != FMT_S8_K && FMT != FMT_S1_5);
    if (n_e == 1) {
        float g0 = 0.0f, u0 = 0.0f;
        const InkQ8_1_G64* x0 = xp[0];
        for (int s = 0; s < 9; ++s) {
            const uint8_t* bg = gr + s * SB;
            const uint8_t* bu = ur + s * SB;
            const float eg = f16_to_f32(ld16_u(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
            const float eu = f16_to_f32(ld16_u(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
            const InkQ8_1_G64& xb = x0[s * 4 + g];
            uint32_t wg[16], wu[16];
            sk_unpack64_g_to_u32<FMT>(bg + 6, g, wg);
            sk_unpack64_g_to_u32<FMT>(bu + 6, g, wu);
            const int32_t ig = dp64_i8_smem(wg, xb);
            const int32_t iu = dp64_i8_smem(wu, xb);
            if constexpr (HAS_HALF) {
                g0 = fmaf(eg, xb.d * (float)ig - HALF * xb.s, g0);
                u0 = fmaf(eu, xb.d * (float)iu - HALF * xb.s, u0);
            } else {
                g0 = fmaf(eg, xb.d * (float)ig, g0);
                u0 = fmaf(eu, xb.d * (float)iu, u0);
            }
        }
        g0 += __shfl_xor_sync(0xFFFFFFFFu, g0, 2); g0 += __shfl_xor_sync(0xFFFFFFFFu, g0, 1);
        u0 += __shfl_xor_sync(0xFFFFFFFFu, u0, 2); u0 += __shfl_xor_sync(0xFFFFFFFFu, u0, 1);
        ag[0] = g0; au[0] = u0;
    } else {
        for (int s = 0; s < 9; ++s) {
            const uint8_t* bg = gr + s * SB;
            const uint8_t* bu = ur + s * SB;
            const float eg = f16_to_f32(ld16_u(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
            const float eu = f16_to_f32(ld16_u(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
            uint32_t wg[16], wu[16];
            sk_unpack64_g_to_u32<FMT>(bg + 6, g, wg);
            sk_unpack64_g_to_u32<FMT>(bu + 6, g, wu);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = xp[m][s * 4 + g];
                    const int32_t ig = dp64_i8_smem(wg, xb);
                    const int32_t iu = dp64_i8_smem(wu, xb);
                    if constexpr (HAS_HALF) {
                        ag[m] = fmaf(eg, xb.d * (float)ig - HALF * xb.s, ag[m]);
                        au[m] = fmaf(eu, xb.d * (float)iu - HALF * xb.s, au[m]);
                    } else {
                        ag[m] = fmaf(eg, xb.d * (float)ig, ag[m]);
                        au[m] = fmaf(eu, xb.d * (float)iu, au[m]);
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                ag[m] += __shfl_xor_sync(0xFFFFFFFFu, ag[m], 2); ag[m] += __shfl_xor_sync(0xFFFFFFFFu, ag[m], 1);
                au[m] += __shfl_xor_sync(0xFFFFFFFFu, au[m], 2); au[m] += __shfl_xor_sync(0xFFFFFFFFu, au[m], 1);
            }
        }
    }
}

template <int FMT, int MAX_M>
__device__ __forceinline__ void sk_dn_paired2_4lane(
    const uint8_t* pr,
    const InkQ8_1_G64* const (&hp)[MAX_M],
    int n_e, int g,
    float (&a0)[MAX_M], float (&a1)[MAX_M]) {
    constexpr int SB = ink_sb_bytes(FMT);
    constexpr float HALF = ink_sk_half(FMT);
    constexpr bool HAS_HALF = (FMT != FMT_S8_K && FMT != FMT_S1_5);
    if (n_e == 1) {
        float y0 = 0.0f, y1 = 0.0f;
        const InkQ8_1_G64* h0 = hp[0];
        for (int s = 0; s < 9; ++s) {
            const uint8_t* blk = pr + s * SB;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const int sub_g = s * 4 + g;
            const bool is_r1 = (sub_g >= 18);
            const InkQ8_1_G64& xb = h0[is_r1 ? (sub_g - 18) : sub_g];
            uint32_t w[16];
            sk_unpack64_g_to_u32<FMT>(blk + 6, g, w);
            const int32_t isum = dp64_i8_smem(w, xb);
            const float term = HAS_HALF ? (eff * (xb.d * (float)isum - HALF * xb.s)) : (eff * xb.d * (float)isum);
            if (is_r1) y1 += term; else y0 += term;
        }
        y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, 2); y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, 1);
        y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, 2); y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, 1);
        a0[0] = y0; a1[0] = y1;
    } else {
        for (int s = 0; s < 9; ++s) {
            const uint8_t* blk = pr + s * SB;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const int sub_g = s * 4 + g;
            const bool is_r1 = (sub_g >= 18);
            const int xg = is_r1 ? (sub_g - 18) : sub_g;
            uint32_t w[16];
            sk_unpack64_g_to_u32<FMT>(blk + 6, g, w);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    const InkQ8_1_G64& xb = hp[m][xg];
                    const int32_t isum = dp64_i8_smem(w, xb);
                    const float term = HAS_HALF ? (eff * (xb.d * (float)isum - HALF * xb.s)) : (eff * xb.d * (float)isum);
                    if (is_r1) a1[m] += term; else a0[m] += term;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_e) {
                a0[m] += __shfl_xor_sync(0xFFFFFFFFu, a0[m], 2); a0[m] += __shfl_xor_sync(0xFFFFFFFFu, a0[m], 1);
                a1[m] += __shfl_xor_sync(0xFFFFFFFFu, a1[m], 2); a1[m] += __shfl_xor_sync(0xFFFFFFFFu, a1[m], 1);
            }
        }
    }
}

struct InkExpertDev {
    uint64_t body_off;  // offset from Section 4 base (4 KiB aligned)
    uint32_t dn_off;    // byte offset of w2 inside expert body
    uint8_t  gu_code;   // InkFmt code id (15=SF4, 16=MX8, 1..8=S_K)
    uint8_t  dn_code;
    uint16_t tier;      // 0 = GPU VRAM, 1 = pinned host, 2 = CPU hybrid (skip on GPU)
};

struct InkMoeTables {
    int32_t* cnt;       // [128]
    int32_t* off;       // [129]
    int32_t* active;    // [128]
    int32_t* nact;      // [1]
    int32_t* pairs;     // [P] packed (tok * 8 + slot)
    int32_t* tok_idx;   // [P] source token index for each sorted pair
    int32_t* n_tiles;   // [1] total 64-row tiles across all active experts
    int2*    tiles;     // [max_tiles] {.x = expert_id, .y = pair_row0}
};

__device__ __forceinline__ float e2m1_nibble_to_f32(uint32_t nib) {
    const uint32_t mag = nib & 7u;
    // mag: 0->0.0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
    // Pack magnitudes as scaled half-integers (x2): {0, 1, 2, 3, 4, 6, 8, 12} -> 4 bits per entry in 0xC8643210u!
    const uint32_t x2 = (0xC8643210u >> (mag * 4u)) & 0xFu;
    float v = 0.5f * (float)x2;
    return (nib & 8u) ? -v : v;
}

__device__ __forceinline__ float2 e2m1x2_byte_to_f32x2(uint8_t byte) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)byte, __NV_E2M1);
    return __half22float2(__half2(hr));
#else
    return make_float2(e2m1_nibble_to_f32(byte & 0xFu), e2m1_nibble_to_f32((byte >> 4) & 0xFu));
#endif
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
    return __half2float(__half(hr));
#else
    // Branchless bit-math (~4 ALU ops, ZERO divergent branches on Volta/Turing/Ampere/Pascal!)
    const uint32_t sgn = ((uint32_t)c & 0x80u) << 24;
    const uint32_t mag = (uint32_t)c & 0x7Fu;
    const uint32_t exp = mag >> 3;
    const uint32_t man = mag & 7u;
    const float norm = __uint_as_float(sgn | ((exp + 120u) << 23) | (man << 20));
    const float subn = __uint_as_float(sgn | __float_as_uint((float)man * (1.0f / 512.0f)));
    float res = (exp == 0u) ? subn : norm;
    return (mag == 0x7Fu) ? 0.0f : res;
#endif
}

__device__ __forceinline__ float2 e4m3x2_to_f32x2(uint16_t pair) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)pair, __NV_E4M3);
    return __half22float2(__half2(hr));
#else
    return make_float2(e4m3_to_f32((uint8_t)(pair & 0xFFu)), e4m3_to_f32((uint8_t)(pair >> 8)));
#endif
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
    if (e == 255u) return 0.0f;
    if (e == 0u) return 5.877471754111438e-39f;
    return __uint_as_float((uint32_t)e << 23);
}

__device__ __forceinline__ void mma_m16n8k32_s8(int32_t (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

template <int FMT>
__device__ __forceinline__ void sk_unpack8_lane8(const uint8_t* pl, int g, int lane_in_g, uint32_t& w0, uint32_t& w1) {
    const int in_g = lane_in_g * 8;
    if constexpr (FMT == FMT_S8_K) {
        w0 = ld32_u(pl + g * 64 + in_g);
        w1 = ld32_u(pl + g * 64 + in_g + 4);
    } else if constexpr (FMT == FMT_S2_K) {
        const uint8_t b0 = pl[g * 16 + lane_in_g * 2 + 0];
        const uint8_t b1 = pl[g * 16 + lane_in_g * 2 + 1];
        w0 = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
        w1 = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
    } else if constexpr (FMT == FMT_S3_K) {
        const uint8_t b0 = pl[g * 16 + lane_in_g * 2 + 0];
        const uint8_t b1 = pl[g * 16 + lane_in_g * 2 + 1];
        const uint8_t hb = pl[64 + g * 8 + lane_in_g];
        w0 = (b0 & 3u) | (((b0 >> 2) & 3u) << 8) | (((b0 >> 4) & 3u) << 16) | (((b0 >> 6) & 3u) << 24);
        w1 = (b1 & 3u) | (((b1 >> 2) & 3u) << 8) | (((b1 >> 4) & 3u) << 16) | (((b1 >> 6) & 3u) << 24);
        w0 |= ((hb & 1u) << 2) | (((hb >> 1) & 1u) << 10) | (((hb >> 2) & 1u) << 18) | (((hb >> 3) & 1u) << 26);
        w1 |= (((hb >> 4) & 1u) << 2) | (((hb >> 5) & 1u) << 10) | (((hb >> 6) & 1u) << 18) | (((hb >> 7) & 1u) << 26);
    } else if constexpr (FMT == FMT_S4_K) {
        const uint32_t p4 = ld32_u(pl + g * 32 + lane_in_g * 4);
        w0 = (p4 & 0x0Fu) | (((p4 >> 4) & 0x0Fu) << 8) | (((p4 >> 8) & 0x0Fu) << 16) | (((p4 >> 12) & 0x0Fu) << 24);
        w1 = ((p4 >> 16) & 0x0Fu) | (((p4 >> 20) & 0x0Fu) << 8) | (((p4 >> 24) & 0x0Fu) << 16) | (((p4 >> 28) & 0x0Fu) << 24);
    } else if constexpr (FMT == FMT_S5_K) {
        const uint32_t p4 = ld32_u(pl + g * 32 + lane_in_g * 4);
        const uint8_t hb = pl[128 + g * 8 + lane_in_g];
        w0 = (p4 & 0x0Fu) | (((p4 >> 4) & 0x0Fu) << 8) | (((p4 >> 8) & 0x0Fu) << 16) | (((p4 >> 12) & 0x0Fu) << 24);
        w1 = ((p4 >> 16) & 0x0Fu) | (((p4 >> 20) & 0x0Fu) << 8) | (((p4 >> 24) & 0x0Fu) << 16) | (((p4 >> 28) & 0x0Fu) << 24);
        w0 |= ((hb & 1u) << 4) | (((hb >> 1) & 1u) << 12) | (((hb >> 2) & 1u) << 20) | (((hb >> 3) & 1u) << 28);
        w1 |= (((hb >> 4) & 1u) << 4) | (((hb >> 5) & 1u) << 12) | (((hb >> 6) & 1u) << 20) | (((hb >> 7) & 1u) << 28);
    } else if constexpr (FMT == FMT_S6_K) {
        const uint32_t p4 = ld32_u(pl + g * 32 + lane_in_g * 4);
        const uint16_t h16 = ld16_u(pl + 128 + g * 16 + lane_in_g * 2);
        w0 = (p4 & 0x0Fu) | (((p4 >> 4) & 0x0Fu) << 8) | (((p4 >> 8) & 0x0Fu) << 16) | (((p4 >> 12) & 0x0Fu) << 24);
        w1 = ((p4 >> 16) & 0x0Fu) | (((p4 >> 20) & 0x0Fu) << 8) | (((p4 >> 24) & 0x0Fu) << 16) | (((p4 >> 28) & 0x0Fu) << 24);
        w0 |= ((h16 & 3u) << 4) | (((h16 >> 2) & 3u) << 12) | (((h16 >> 4) & 3u) << 20) | (((h16 >> 6) & 3u) << 28);
        w1 |= (((h16 >> 8) & 3u) << 4) | (((h16 >> 10) & 3u) << 12) | (((h16 >> 12) & 3u) << 20) | (((h16 >> 14) & 3u) << 28);
    } else if constexpr (FMT == FMT_S1_5) {
        uint8_t b[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int idx = g * 64 + in_g + k;
            b[k] = (uint8_t)kS15TritLut[pl[idx / 5]][idx % 5];
        }
        w0 = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
        w1 = (uint32_t)b[4] | ((uint32_t)b[5] << 8) | ((uint32_t)b[6] << 16) | ((uint32_t)b[7] << 24);
    } else {
        w0 = w1 = 0u;
    }
}

__device__ __forceinline__ void sf4_decode16_h2(uint32_t w0, uint32_t w1, __half2 (&w)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    auto c2 = [](uint8_t b) -> __half2 {
        __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
        return *reinterpret_cast<const __half2*>(&r);
    };
#else
    auto c2 = [](uint8_t b) -> __half2 {
        float2 f = e2m1x2_byte_to_f32x2(b);
        return __floats2half2_rn(f.x, f.y);
    };
#endif
    w[0] = c2((uint8_t)(w0 & 0xFFu));
    w[1] = c2((uint8_t)((w0 >> 8) & 0xFFu));
    w[2] = c2((uint8_t)((w0 >> 16) & 0xFFu));
    w[3] = c2((uint8_t)(w0 >> 24));
    w[4] = c2((uint8_t)(w1 & 0xFFu));
    w[5] = c2((uint8_t)((w1 >> 8) & 0xFFu));
    w[6] = c2((uint8_t)((w1 >> 16) & 0xFFu));
    w[7] = c2((uint8_t)(w1 >> 24));
}

__device__ __forceinline__ void mx8_decode16_h2(const uint4& wv, __half2 (&w)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    auto c2 = [](uint16_t u) -> __half2 {
        __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)u, __NV_E4M3);
        return *reinterpret_cast<const __half2*>(&r);
    };
#else
    auto c2 = [](uint16_t u) -> __half2 {
        float2 f = e4m3x2_to_f32x2(u);
        return __floats2half2_rn(f.x, f.y);
    };
#endif
    w[0] = c2((uint16_t)(wv.x & 0xFFFFu)); w[1] = c2((uint16_t)(wv.x >> 16));
    w[2] = c2((uint16_t)(wv.y & 0xFFFFu)); w[3] = c2((uint16_t)(wv.y >> 16));
    w[4] = c2((uint16_t)(wv.z & 0xFFFFu)); w[5] = c2((uint16_t)(wv.z >> 16));
    w[6] = c2((uint16_t)(wv.w & 0xFFFFu)); w[7] = c2((uint16_t)(wv.w >> 16));
}

__device__ __forceinline__ float dot16_h2_2acc(const __half2 (&w)[8], const __half2* xh2) {
    // 2x 128-bit LDS.128 loads + 2 independent accumulators -> zero pipeline stall!
    const uint4 v0 = *reinterpret_cast<const uint4*>(xh2);
    const uint4 v1 = *reinterpret_cast<const uint4*>(xh2 + 4);
    const __half2* x0 = reinterpret_cast<const __half2*>(&v0);
    const __half2* x1 = reinterpret_cast<const __half2*>(&v1);
    __half2 a0 = __hmul2(w[0], x0[0]);
    __half2 a1 = __hmul2(w[1], x0[1]);
    a0 = __hfma2(w[2], x0[2], a0);
    a1 = __hfma2(w[3], x0[3], a1);
    a0 = __hfma2(w[4], x1[0], a0);
    a1 = __hfma2(w[5], x1[1], a1);
    a0 = __hfma2(w[6], x1[2], a0);
    a1 = __hfma2(w[7], x1[3], a1);
    const float2 f = __half22float2(__hadd2(a0, a1));
    return f.x + f.y;
}

template <bool GU, int TR, int FMT>
__device__ __forceinline__ void ink_moe_sk_tc_kloop(
    const uint8_t* __restrict__ body,
    uint64_t dn_off,
    int fb,
    int nrows,
    int tid, int wf, int nb0, int g, int tig,
    bool on0, bool on1,
    const InkQ8_1_G64* __restrict__ act_sk,
    const int32_t* __restrict__ s_row,
    int8_t (&s_wq)[128][80],
    int8_t (&s_aq)[TR][80],
    float (&s_wsc)[128],
    float (&s_ad)[TR],
    float (&s_as)[TR],
    float (&acc)[(128 / (8 / (TR / 16))) / 16][2][4]) {
    constexpr int kWR      = 128;
    constexpr int NG       = GU ? (kInkHidden / 64) : (kInkInter / 64);
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;
    constexpr int SB       = ink_sb_bytes(FMT);
    constexpr float HALF   = ink_sk_half(FMT);
    constexpr bool HAS_HALF = (FMT != FMT_S8_K && FMT != FMT_S1_5);

    const uint8_t* gu_base = body + kInkInter * 2;
    constexpr size_t gu_row_b  = 9ull * (size_t)SB;
    constexpr size_t dn_pair_b = 9ull * (size_t)SB;
    const int r_in_w = tid >> 3;  // 0..31 rows per wave
    const int lg     = tid & 7;   // 0..7 coalesced lanes per row (8 weights/lane = 64 weights/row!)

    for (int xg = 0; xg < NG; ++xg) {
        // 1. ALL 256 threads unpack 128 weight rows x 64 weights in 4 coalesced waves of 32 rows!
#pragma unroll
        for (int wave = 0; wave < 4; ++wave) {
            const int row_s = wave * 32 + r_in_w;
            const int phys_row = GU ? (2 * (fb * 64 + 8 * (row_s >> 4) + (row_s & 7)) + ((row_s >> 3) & 1))
                                    : (fb * kWR + row_s);
            const uint8_t* blk;
            int sg;
            if constexpr (GU) {
                blk = gu_base + (size_t)phys_row * gu_row_b + (size_t)(xg >> 2) * SB;
                sg  = xg & 3;
            } else {
                const int pair_r = phys_row >> 1;
                const int sub_glob = xg + (phys_row & 1) * 18;
                blk = body + dn_off + (size_t)pair_r * dn_pair_b + (size_t)(sub_glob >> 2) * SB;
                sg  = sub_glob & 3;
            }
            uint32_t w0, w1;
            sk_unpack8_lane8<FMT>(blk + 6, sg, lg, w0, w1);
            uint32_t* d4 = reinterpret_cast<uint32_t*>(s_wq[row_s]);
            d4[2 * lg + 0] = w0;
            d4[2 * lg + 1] = w1;
            if (lg == 0) s_wsc[row_s] = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + sg] * (1.0f / 255.0f));
        }
        // 2. Load TR activation blocks (2 threads per row -> <= 128 threads active)
        if (tid < nrows * 2) {
            const int r = tid >> 1;
            const int h = tid & 1;
            const InkQ8_1_G64& xb = act_sk[(size_t)s_row[r] * NG + xg];
            const uint4* src = reinterpret_cast<const uint4*>(xb.qs + h * 32);
            uint4* dst = reinterpret_cast<uint4*>(s_aq[r] + h * 32);
            dst[0] = src[0]; dst[1] = src[1];
            if (h == 0) { s_ad[r] = xb.d; s_as[r] = xb.s; }
        }
        __syncthreads();

        // 3. Run 2x mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (k=0..31 and k=32..63)!
        if (on0) {
            const uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 4 * tig);
            const uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 16 + 4 * tig);
            const uint32_t b0_2 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 32 + 4 * tig);
            const uint32_t b0_3 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + g] + 48 + 4 * tig);
            uint32_t b1_0 = 0, b1_1 = 0, b1_2 = 0, b1_3 = 0;
            if (on1) {
                b1_0 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 4 * tig);
                b1_1 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 16 + 4 * tig);
                b1_2 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 32 + 4 * tig);
                b1_3 = *reinterpret_cast<const uint32_t*>(s_aq[nb0 + 8 + g] + 48 + 4 * tig);
            }
            const float xd00 = s_ad[nb0 + 2 * tig + 0], xs00 = s_as[nb0 + 2 * tig + 0];
            const float xd01 = s_ad[nb0 + 2 * tig + 1], xs01 = s_as[nb0 + 2 * tig + 1];
            const float xd10 = on1 ? s_ad[nb0 + 8 + 2 * tig + 0] : 0.0f;
            const float xs10 = on1 ? s_as[nb0 + 8 + 2 * tig + 0] : 0.0f;
            const float xd11 = on1 ? s_ad[nb0 + 8 + 2 * tig + 1] : 0.0f;
            const float xs11 = on1 ? s_as[nb0 + 8 + 2 * tig + 1] : 0.0f;

#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int wr0 = W_PER_WF * wf + 16 * i + g;
                const int wr1 = wr0 + 8;
                const float eff0 = s_wsc[wr0];
                const float eff1 = s_wsc[wr1];

                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 4 * tig);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 4 * tig);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 16 + 4 * tig);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 16 + 4 * tig);
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 32 + 4 * tig);
                const uint32_t a5 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 32 + 4 * tig);
                const uint32_t a6 = *reinterpret_cast<const uint32_t*>(s_wq[wr0] + 48 + 4 * tig);
                const uint32_t a7 = *reinterpret_cast<const uint32_t*>(s_wq[wr1] + 48 + 4 * tig);

                int32_t cv[4] = {};
                mma_m16n8k32_s8(cv, a0, a1, a2, a3, b0_0, b0_1);
                mma_m16n8k32_s8(cv, a4, a5, a6, a7, b0_2, b0_3);
                if constexpr (HAS_HALF) {
                    acc[i][0][0] = fmaf(eff0, xd00 * (float)cv[0] - HALF * xs00, acc[i][0][0]);
                    acc[i][0][1] = fmaf(eff0, xd01 * (float)cv[1] - HALF * xs01, acc[i][0][1]);
                    acc[i][0][2] = fmaf(eff1, xd00 * (float)cv[2] - HALF * xs00, acc[i][0][2]);
                    acc[i][0][3] = fmaf(eff1, xd01 * (float)cv[3] - HALF * xs01, acc[i][0][3]);
                } else {
                    acc[i][0][0] = fmaf(eff0, xd00 * (float)cv[0], acc[i][0][0]);
                    acc[i][0][1] = fmaf(eff0, xd01 * (float)cv[1], acc[i][0][1]);
                    acc[i][0][2] = fmaf(eff1, xd00 * (float)cv[2], acc[i][0][2]);
                    acc[i][0][3] = fmaf(eff1, xd01 * (float)cv[3], acc[i][0][3]);
                }
                if (on1) {
                    int32_t dv[4] = {};
                    mma_m16n8k32_s8(dv, a0, a1, a2, a3, b1_0, b1_1);
                    mma_m16n8k32_s8(dv, a4, a5, a6, a7, b1_2, b1_3);
                    if constexpr (HAS_HALF) {
                        acc[i][1][0] = fmaf(eff0, xd10 * (float)dv[0] - HALF * xs10, acc[i][1][0]);
                        acc[i][1][1] = fmaf(eff0, xd11 * (float)dv[1] - HALF * xs11, acc[i][1][1]);
                        acc[i][1][2] = fmaf(eff1, xd10 * (float)dv[2] - HALF * xs10, acc[i][1][2]);
                        acc[i][1][3] = fmaf(eff1, xd11 * (float)dv[3] - HALF * xs11, acc[i][1][3]);
                    } else {
                        acc[i][1][0] = fmaf(eff0, xd10 * (float)dv[0], acc[i][1][0]);
                        acc[i][1][1] = fmaf(eff0, xd11 * (float)dv[1], acc[i][1][1]);
                        acc[i][1][2] = fmaf(eff1, xd10 * (float)dv[2], acc[i][1][2]);
                        acc[i][1][3] = fmaf(eff1, xd11 * (float)dv[3], acc[i][1][3]);
                    }
                }
            }
        }
        if (xg + 1 < NG) __syncthreads();
    }
}

template <bool GU, int TR>
__global__ void __launch_bounds__(256) k_ink_moe_sk_tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const InkQ8_1_G64* __restrict__ act_sk,   // GU: [M, 36]; DN: [P, 18]
    const float* __restrict__ topk_w,         // DN only
    float* __restrict__ out_buf) {            // GU: h_f32 [P, 1152]; DN: y_pair [P, 2304]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kWR      = 128;
    constexpr int ROWS_N   = GU ? (2 * kInkInter) : kInkHidden;  // 2304
    constexpr int NFB      = ROWS_N / kWR;                       // 18 blocks of 128 rows
    constexpr int N_WN     = TR / 16;                            // 1 (TR=16) or 4 (TR=64)
    constexpr int N_WF     = 8 / N_WN;                           // 8 (TR=16) or 2 (TR=64)
    constexpr int W_PER_WF = kWR / N_WF;                         // 16 or 64
    constexpr int NI       = W_PER_WF / 16;                      // 1 or 4
    constexpr int WSTR     = 80;
    constexpr int ASTR     = 80;

    __shared__ alignas(16) int8_t  s_wq[kWR][WSTR];
    __shared__ alignas(16) int8_t  s_aq[TR][ASTR];
    __shared__ float               s_wsc[kWR];
    __shared__ float               s_ad[TR];
    __shared__ float               s_as[TR];
    __shared__ int32_t             s_row[TR];
    __shared__ float               s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        const int fmt = GU ? (int)ed.gu_code : (int)ed.dn_code;
        if (ed.tier > 1 || fmt == FMT_SF4 || fmt == FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                s_ad[tid] = 0.0f; s_as[tid] = 0.0f;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                uint32_t* z = reinterpret_cast<uint32_t*>(s_aq[tid]);
                for (int k = 0; k < 16; ++k) z[k] = 0u;
            }
        }
        __syncthreads();

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        switch (fmt) {
            case FMT_S2_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S2_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S3_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S3_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S4_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S4_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S5_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S5_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S6_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S6_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S8_K: ink_moe_sk_tc_kloop<GU, TR, FMT_S8_K>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            case FMT_S1_5: ink_moe_sk_tc_kloop<GU, TR, FMT_S1_5>(body, ed.dn_off, fb, nrows, tid, wf, nb0, g, tig, on0, on1, act_sk, s_row, s_wq, s_aq, s_wsc, s_ad, s_as, acc); break;
            default: break;
        }

        if (on0) {
            if constexpr (GU) {
                const int fbase = fb * 64 + (W_PER_WF / 2) * wf;
                float inv_sq[NI];
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    const int feat = fbase + 8 * i + g;
                    inv_sq[i] = 1.0f / f16_to_f32(ld16_u(body + feat * 2));
                }
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            float* dst = out_buf + (size_t)(row0 + r) * kInkInter;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[fbase + 8 * i + g] = situ_f32(acc[i][n][c], acc[i][n][2 + c]) * inv_sq[i];
                            }
                        }
                    }
                }
            } else {
                const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            const float sc = s_rw[r];
                            float* dst = out_buf + (size_t)tb.pairs[row0 + r] * kInkHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_sk; (void)topk_w; (void)out_buf;
#endif
}

template <bool GU>
__global__ void __launch_bounds__(256) k_ink_moe_sf4_mx8_fp16tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const float* __restrict__ act_f32,        // GU: x_f32 [M, 2304]; DN: h_f32 [P, 1152]
    const float* __restrict__ topk_w,         // DN only
    float* __restrict__ out_buf) {            // GU: h_f32 [P, 1152]; DN: y_pair [P, 2304]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kWR    = 128;
    constexpr int TR     = 16;
    constexpr int K      = GU ? kInkHidden : kInkInter;       // 2304 or 1152
    constexpr int ROWS_N = GU ? (2 * kInkInter) : kInkHidden; // 2304
    constexpr int NFB    = ROWS_N / kWR;                      // 18 blocks of 128 rows
    constexpr int N32    = K / 32;                            // 72 (GU) or 36 (DN)

    __shared__ alignas(16) __half s_w[kWR][40];   // 10,240 B (stride 80B = 16B aligned, bank-conflict free!)
    __shared__ alignas(16) __half s_a[TR][40];    //  1,280 B
    __shared__ float              s_wsc[kWR][2];  //  1,024 B (exact scale per K=16 block!)
    __shared__ int32_t            s_row[TR];      //     64 B
    __shared__ float              s_rw[TR];       //     64 B

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..7 (each warp owns 16 weight rows: wf = warp)
    const int lane = tid & 31;
    const int g    = lane >> 2;  // 0..7
    const int tig  = lane & 3;   // 0..3

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        const int fmt = GU ? (int)ed.gu_code : (int)ed.dn_code;
        if (ed.tier > 1 || (fmt != FMT_SF4 && fmt != FMT_MX8)) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float w_outer = 1.0f;
        if (fmt == FMT_SF4) {
            float hdr[4];
            memcpy(hdr, body, 16);
            w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
        }

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                uint32_t* z = reinterpret_cast<uint32_t*>(s_a[tid]);
                for (int k = 0; k < 16; ++k) z[k] = 0u;
            }
        }
        __syncthreads();

        float acc[2][4] = {};
        const bool on1 = 8 < nrows;
        const int wr0 = 16 * warp + g;
        const int wr1 = wr0 + 8;

        auto phys_row_fn = [&](int r) -> int {
            return GU ? (2 * (fb * 64 + 8 * (r >> 4) + (r & 7)) + ((r >> 3) & 1))
                      : (fb * kWR + r);
        };

        if (fmt == FMT_SF4) {
            constexpr size_t W_P_ROW = K / 2;
            constexpr size_t W_S_ROW = K / 16;
            const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            const int pr_t = (tid < kWR) ? phys_row_fn(tid) : 0;
            const uint4* wp16 = reinterpret_cast<const uint4*>(base_p + (size_t)pr_t * W_P_ROW);
            const uint16_t* ws2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)pr_t * W_S_ROW);

            for (int k32 = 0; k32 < N32; ++k32) {
                if (tid < kWR) {
                    const uint4 v = wp16[k32];
                    const uint16_t sp = ws2[k32];
                    s_wsc[tid][0] = e4m3_to_f32((uint8_t)(sp & 0xFFu)) * w_outer;
                    s_wsc[tid][1] = e4m3_to_f32((uint8_t)(sp >> 8))    * w_outer;
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v.x, v.y, w0);
                    sf4_decode16_h2(v.z, v.w, w1);
                    __half2* dw = reinterpret_cast<__half2*>(s_w[tid]);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { dw[j] = w0[j]; dw[8 + j] = w1[j]; }
                } else {
                    const int atid = tid - kWR;  // 0..127 -> 8 threads per activation row (16 rows x 8 = 128!)
                    const int r = atid >> 3;
                    const int j = atid & 7;      // 4 floats = 2 half2s
                    if (r < nrows) {
                        const float4 fv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k32 * 32 + j * 4);
                        __half2* da = reinterpret_cast<__half2*>(s_a[r]);
                        da[2 * j + 0] = __floats2half2_rn(fv.x, fv.y);
                        da[2 * j + 1] = __floats2half2_rn(fv.z, fv.w);
                    }
                }
                __syncthreads();

#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int ko = sub * 16;
                    const float sc0 = s_wsc[wr0][sub];
                    const float sc1 = s_wsc[wr1][sub];
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 2 * tig);
                    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 2 * tig);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 8 + 2 * tig);
                    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 8 + 2 * tig);
                    const uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 2 * tig);
                    const uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 8 + 2 * tig);
                    float cv[4] = {};
                    mma_m16n8k16_f16(cv, a0, a1, a2, a3, b0_0, b0_1);
                    acc[0][0] = fmaf(sc0, cv[0], acc[0][0]);
                    acc[0][1] = fmaf(sc0, cv[1], acc[0][1]);
                    acc[0][2] = fmaf(sc1, cv[2], acc[0][2]);
                    acc[0][3] = fmaf(sc1, cv[3], acc[0][3]);
                    if (on1) {
                        const uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 2 * tig);
                        const uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 8 + 2 * tig);
                        float dv[4] = {};
                        mma_m16n8k16_f16(dv, a0, a1, a2, a3, b1_0, b1_1);
                        acc[1][0] = fmaf(sc0, dv[0], acc[1][0]);
                        acc[1][1] = fmaf(sc0, dv[1], acc[1][1]);
                        acc[1][2] = fmaf(sc1, dv[2], acc[1][2]);
                        acc[1][3] = fmaf(sc1, dv[3], acc[1][3]);
                    }
                }
                if (k32 + 1 < N32) __syncthreads();
            }
        } else {
            constexpr size_t W_P_ROW = K;
            constexpr size_t W_S_ROW = K / 32;
            const uint8_t* base_p = GU ? body : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            const int r_w = tid >> 1;
            const int h_w = tid & 1;
            const int pr_t = phys_row_fn(r_w);
            const uint4* wp16 = reinterpret_cast<const uint4*>(base_p + (size_t)pr_t * W_P_ROW);
            const uint8_t* ws = base_s + (size_t)pr_t * W_S_ROW;

            for (int k32 = 0; k32 < N32; ++k32) {
                __half2 wh[8];
                mx8_decode16_h2(wp16[k32 * 2 + h_w], wh);
                __half2* dw = reinterpret_cast<__half2*>(s_w[r_w]) + h_w * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) dw[j] = wh[j];
                if (h_w == 0) {
                    const float sc = ue8m0_to_f32(ws[k32]);
                    s_wsc[r_w][0] = sc;
                }
                if (tid < nrows * 8) {
                    const int r = tid >> 3;
                    const int j = tid & 7;
                    const float4 fv = *reinterpret_cast<const float4*>(act_f32 + (size_t)s_row[r] * K + k32 * 32 + j * 4);
                    __half2* da = reinterpret_cast<__half2*>(s_a[r]);
                    da[2 * j + 0] = __floats2half2_rn(fv.x, fv.y);
                    da[2 * j + 1] = __floats2half2_rn(fv.z, fv.w);
                }
                __syncthreads();

                const float sc0 = s_wsc[wr0][0];
                const float sc1 = s_wsc[wr1][0];
                float cv[4] = {}, dv[4] = {};
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int ko = sub * 16;
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 2 * tig);
                    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 2 * tig);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(s_w[wr0] + ko + 8 + 2 * tig);
                    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(s_w[wr1] + ko + 8 + 2 * tig);
                    const uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 2 * tig);
                    const uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(s_a[g] + ko + 8 + 2 * tig);
                    mma_m16n8k16_f16(cv, a0, a1, a2, a3, b0_0, b0_1);
                    if (on1) {
                        const uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 2 * tig);
                        const uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(s_a[8 + g] + ko + 8 + 2 * tig);
                        mma_m16n8k16_f16(dv, a0, a1, a2, a3, b1_0, b1_1);
                    }
                }
                acc[0][0] = fmaf(sc0, cv[0], acc[0][0]);
                acc[0][1] = fmaf(sc0, cv[1], acc[0][1]);
                acc[0][2] = fmaf(sc1, cv[2], acc[0][2]);
                acc[0][3] = fmaf(sc1, cv[3], acc[0][3]);
                if (on1) {
                    acc[1][0] = fmaf(sc0, dv[0], acc[1][0]);
                    acc[1][1] = fmaf(sc0, dv[1], acc[1][1]);
                    acc[1][2] = fmaf(sc1, dv[2], acc[1][2]);
                    acc[1][3] = fmaf(sc1, dv[3], acc[1][3]);
                }
                if (k32 + 1 < N32) __syncthreads();
            }
        }

        if constexpr (GU) {
            const int feat = fb * 64 + 8 * warp + g;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        out_buf[(size_t)(row0 + r) * kInkInter + feat] = situ_f32(acc[n][c], acc[n][2 + c]);
                    }
                }
            }
        } else {
            const int rbase = fb * kWR + 16 * warp;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        float* dst = out_buf + (size_t)tb.pairs[row0 + r] * kInkHidden;
                        dst[rbase + g + 0] = acc[n][c] * sc;
                        dst[rbase + g + 8] = acc[n][2 + c] * sc;
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act_f32; (void)topk_w; (void)out_buf;
#endif
}

}  // namespace strata::ink
