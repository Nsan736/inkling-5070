// cpu_nvfp4.hpp - inkling2 fork: NVFP4 x Q8_0 dot products for the CPU experts with AVX-VNNI (vpdpbusd).
//
// The same integer sums as ggml-cpu's AVX2 ggml_vec_dot_nvfp4_q8_0 (arch/x86/quants.c): each 4 products of a block
// added as int32, then a float vector of them times the sub-block's scale (UE4M3) times the activation block's
// (fp16), accumulated with FMA in the same lanes and order.  Only the int8 products are done differently: the E2M1
// values are taken as unsigned (value + 12, 0 .. 24) so that one vpdpbusd does what _mm256_sign_epi8 x 2,
// _mm256_maddubs_epi16 and _mm256_madd_epi16 did, and 12 * (the activation's sum of each 4 values) - kept once per
// activation row for every expert row - is taken off again.  The integers are exact either way, so the sums are.
//
// Activation side (once per token): Q8_0 blocks of 32 (ggml's block_q8_0) and nvfp4_ysum4(): an int32 per 4 values.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::poly::nvfp4cpu {

struct BlockNvfp4 {
    uint8_t d[4];    // UE4M3 scales, one per 16 values
    uint8_t qs[32];  // E2M1 codes, two per byte
};
struct BlockQ8_0 {
    uint16_t d;      // fp16
    int8_t qs[32];
};
static_assert(sizeof(BlockNvfp4) == 36 && sizeof(BlockQ8_0) == 34, "block layouts");

inline float fp16_to_f32(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

/// ggml's UE4M3 decoding (ggml-impl.h ggml_ue4m3_to_fp32), as a table
struct Ue4m3Table {
    float v[256];
    Ue4m3Table() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7f) {
                v[x] = 0.f;
                continue;
            }
            const int exp = (x >> 3) & 0xF, man = x & 0x7;
            float raw;
            if (exp == 0) raw = (float) man * (1.0f / 8.0f) * (1.0f / 64.0f);   // 2^(1-7) * man/8
            else {
                const uint32_t bits = (uint32_t) (exp + 120) << 23 | (uint32_t) man << 20;
                std::memcpy(&raw, &bits, 4);
            }
            v[x] = raw * 0.5f;
        }
    }
};
inline const Ue4m3Table& ue4m3() {
    static const Ue4m3Table t;
    return t;
}

/// The activation's int32 sums of each 4 values (n / 4 of them), for the offset that vpdpbusd's unsigned side adds.
inline void nvfp4_ysum4(const BlockQ8_0* y, int n, int32_t* out) {
    const __m256i ones = _mm256_set1_epi8(1);
    for (int b = 0; b < n / 32; ++b) {
        const __m256i q = _mm256_loadu_si256((const __m256i*) y[b].qs);
        // sign(q) * 1 per byte pair would lose the sign: dpbusd(ones (unsigned), q (signed)) sums 4 signed bytes
        _mm256_storeu_si256((__m256i*) (out + b * 8), _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones, q));
    }
}

/// Row x (n values, n / 64 NVFP4 blocks) . activation y (Q8_0) with ysum4 from nvfp4_ysum4().
inline float nvfp4_dot(int n, const BlockNvfp4* x, const BlockQ8_0* y, const int32_t* ysum4) {
    const Ue4m3Table& t = ue4m3();
    // the E2M1 values + 12, so that they are unsigned: kvalues_fp4 {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12} + 12
    const __m256i lut = _mm256_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0,
                                         12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i twelve = _mm256_set1_epi32(12);
    __m256 acc = _mm256_setzero_ps();
    const int nb = n / 64;
    for (int ib = 0; ib < nb; ++ib) {
        const __m256i q = _mm256_loadu_si256((const __m256i*) x[ib].qs);   // lane 0: sub-blocks 0, 1; lane 1: 2, 3
        const __m256i lo = _mm256_and_si256(q, m4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
        // ggml's order: a sub-block's 8 low nibbles, then its 8 high ones (the values 0..7 and 8..15 of it)
        const __m256i a = _mm256_unpacklo_epi64(lo, hi);   // sub-blocks 0 | 2
        const __m256i b = _mm256_unpackhi_epi64(lo, hi);   // sub-blocks 1 | 3
        const __m256i x01 = _mm256_shuffle_epi8(lut, _mm256_permute2x128_si256(a, b, 0x20));   // 0 | 1
        const __m256i x23 = _mm256_shuffle_epi8(lut, _mm256_permute2x128_si256(a, b, 0x31));   // 2 | 3
        const __m256i y01 = _mm256_loadu_si256((const __m256i*) y[2 * ib].qs);
        const __m256i y23 = _mm256_loadu_si256((const __m256i*) y[2 * ib + 1].qs);
        __m256i p01 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), x01, y01);
        __m256i p23 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), x23, y23);
        p01 = _mm256_sub_epi32(p01, _mm256_mullo_epi32(twelve, _mm256_loadu_si256((const __m256i*) (ysum4 + 16 * ib))));
        p23 = _mm256_sub_epi32(p23,
                               _mm256_mullo_epi32(twelve, _mm256_loadu_si256((const __m256i*) (ysum4 + 16 * ib + 8))));
        const float dy0 = fp16_to_f32(y[2 * ib].d), dy1 = fp16_to_f32(y[2 * ib + 1].d);
        const float s0 = t.v[x[ib].d[0]] * dy0, s1 = t.v[x[ib].d[1]] * dy0;
        const float s2 = t.v[x[ib].d[2]] * dy1, s3 = t.v[x[ib].d[3]] * dy1;
        acc = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0)), _mm256_cvtepi32_ps(p01), acc);
        acc = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2)), _mm256_cvtepi32_ps(p23), acc);
    }
    // ggml's hsum_float_8
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

/// Rows x0 and x1 against the same activation: each row's sums exactly as nvfp4_dot()'s (the same lanes, the same
/// order), the activation's loads shared, and the two rows' loads in flight together.  pf: bytes ahead of each row
/// to prefetch (0: none).
inline void nvfp4_dot2(int n, const BlockNvfp4* x0, const BlockNvfp4* x1, const BlockQ8_0* y, const int32_t* ysum4,
                       float* out0, float* out1, int pf = 0) {
    const Ue4m3Table& t = ue4m3();
    const __m256i lut = _mm256_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0,
                                         12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m256i twelve = _mm256_set1_epi32(12);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    const int nb = n / 64;
    const auto prods = [&](const BlockNvfp4& b, __m256i y01, __m256i y23, __m256i c01, __m256i c23, __m256i& p01,
                           __m256i& p23) {
        const __m256i q = _mm256_loadu_si256((const __m256i*) b.qs);
        const __m256i lo = _mm256_and_si256(q, m4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
        const __m256i a = _mm256_unpacklo_epi64(lo, hi);
        const __m256i bb = _mm256_unpackhi_epi64(lo, hi);
        p01 = _mm256_sub_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(),
                                                       _mm256_shuffle_epi8(lut, _mm256_permute2x128_si256(a, bb, 0x20)),
                                                       y01),
                               c01);
        p23 = _mm256_sub_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(),
                                                       _mm256_shuffle_epi8(lut, _mm256_permute2x128_si256(a, bb, 0x31)),
                                                       y23),
                               c23);
    };
    for (int ib = 0; ib < nb; ++ib) {
        if (pf) {
            _mm_prefetch((const char*) (x0 + ib) + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib) + pf, _MM_HINT_T0);
        }
        const __m256i y01 = _mm256_loadu_si256((const __m256i*) y[2 * ib].qs);
        const __m256i y23 = _mm256_loadu_si256((const __m256i*) y[2 * ib + 1].qs);
        const __m256i c01 = _mm256_mullo_epi32(twelve, _mm256_loadu_si256((const __m256i*) (ysum4 + 16 * ib)));
        const __m256i c23 = _mm256_mullo_epi32(twelve, _mm256_loadu_si256((const __m256i*) (ysum4 + 16 * ib + 8)));
        const float dy0 = fp16_to_f32(y[2 * ib].d), dy1 = fp16_to_f32(y[2 * ib + 1].d);
        __m256i p01, p23, r01, r23;
        prods(x0[ib], y01, y23, c01, c23, p01, p23);
        prods(x1[ib], y01, y23, c01, c23, r01, r23);
        acc0 = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(t.v[x0[ib].d[1]] * dy0), _mm_set1_ps(t.v[x0[ib].d[0]] * dy0)),
                               _mm256_cvtepi32_ps(p01), acc0);
        acc0 = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(t.v[x0[ib].d[3]] * dy1), _mm_set1_ps(t.v[x0[ib].d[2]] * dy1)),
                               _mm256_cvtepi32_ps(p23), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(t.v[x1[ib].d[1]] * dy0), _mm_set1_ps(t.v[x1[ib].d[0]] * dy0)),
                               _mm256_cvtepi32_ps(r01), acc1);
        acc1 = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(t.v[x1[ib].d[3]] * dy1), _mm_set1_ps(t.v[x1[ib].d[2]] * dy1)),
                               _mm256_cvtepi32_ps(r23), acc1);
    }
    const auto hsum = [](__m256 v) {
        __m128 r = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        return _mm_cvtss_f32(r);
    };
    *out0 = hsum(acc0);
    *out1 = hsum(acc1);
}

}  // namespace strata::poly::nvfp4cpu
