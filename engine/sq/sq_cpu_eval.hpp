// inkling2 fork: the CPU's arithmetic for .strata experts (S_K and SF4), Strata v0.2.19's host evaluator of
// engine/src/inkling2/cuda/ink_hybrid.cuh (struct InkHostExpertMeta and namespace host_eval, MIT, Copyright (c) 2026
// Niko1221 and the Strata contributors), copied unchanged below the definitions it takes from ink_cuda.cuh.  Its
// integer sums are the GPU kernels'; cpu_sq.cpp parallelizes it over the engine's workers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define INK_HOST_AVX2 1
#endif

namespace strata::ink {

constexpr int kInkHidden = 2304;
constexpr int kInkInter = 1152;
constexpr float kInkSituBeta = 4.0f;
constexpr float kInkSituLb = 25.0f;

enum InkFmt : int {
    FMT_S1_25 = 0,
    FMT_S1_5 = 1,
    FMT_S2_K = 2,
    FMT_S3_K = 3,
    FMT_S4_K = 4,
    FMT_S5_K = 5,
    FMT_S6_K = 6,
    FMT_S8_K = 8,
    FMT_SF4 = 15,
    FMT_MX8 = 16,
};

constexpr int ink_sb_bytes(int fmt) {
    switch (fmt) {
    case FMT_S1_25: return 48;
    case FMT_S1_5: return 58;
    case FMT_S2_K: return 70;
    case FMT_S3_K: return 102;
    case FMT_S4_K: return 134;
    case FMT_S5_K: return 166;
    case FMT_S6_K: return 198;
    case FMT_S8_K: return 262;
    default: return 0;
    }
}

struct alignas(16) InkQ8_1_G64 {
    int8_t qs[64];
    float d;
    float s;
};

// ---- from ink_hybrid.cuh ----
struct InkHostExpertMeta {
    const uint8_t* body = nullptr;  // pinned host RAM body pointer
    uint32_t body_bytes = 0;
    uint32_t dn_off     = 0;
    uint8_t  gu_code    = 0;
    uint8_t  dn_code    = 0;
};

namespace host_eval {

inline float f16_to_f32_h(uint16_t h) noexcept {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {
            uint32_t mm = m, ee = 127 - 14;
            while ((mm & 0x0400u) == 0) { mm <<= 1; --ee; }
            f = s | (ee << 23) | ((mm & 0x03FFu) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline const float* e4m3_lut_h() noexcept {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int c = 0; c < 256; ++c) {
            const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
            float v;
            if (e == 15 && m == 7) v = 0.0f;
            else if (e == 0) v = (float)m * (1.0f / 512.0f);
            else v = (1.0f + (float)m * 0.125f) * std::ldexp(1.0f, e - 7);
            t[c] = s ? -v : v;
        }
        init = true;
    }
    return t;
}

inline const float* ue8m0_lut_h() noexcept {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int e = 0; e < 256; ++e) {
            if (e == 255) t[e] = 0.0f;
            else if (e == 0) t[e] = 5.877471754111438e-39f;
            else {
                uint32_t u = (uint32_t)e << 23;
                std::memcpy(&t[e], &u, 4);
            }
        }
        init = true;
    }
    return t;
}

#if defined(INK_HOST_AVX2)
inline float avx2_hsum_ps(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline int32_t avx2_hsum_epi32(__m256i v) noexcept {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline int32_t avx2_dot_u8_i8_64(__m256i u0, __m256i u1, const int8_t* qs) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
    const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u0, q0), ones);
    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, q1), ones);
    return avx2_hsum_epi32(_mm256_add_epi32(p0, p1));
}
#endif

inline float dot_sf4_row_h(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    const int nb = cols >> 4;
#if defined(INK_HOST_AVX2)
    // 2x E2M1 integer LUT -> _mm_shuffle_epi8 unpacks 16 weights/block in 1 cycle + 8-wide AVX2 FMA!
    const __m128i lut2x  = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + b * 8));
        const __m128i lo  = _mm_and_si128(raw, mask0f);
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
        const __m128i nib = _mm_unpacklo_epi8(lo, hi);
        const __m128i w8  = _mm_shuffle_epi8(lut2x, nib);
        const __m256 wf0  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
        const __m256 wf1  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w8, 8)));
        const float* xb   = x + b * 16;
        const __m256 prod = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(xb), _mm256_mul_ps(wf1, _mm256_loadu_ps(xb + 8)));
        vsum = _mm256_fmadd_ps(prod, _mm256_set1_ps(0.5f * e4[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum) * w_outer;
#elif defined(INK_HOST_NEON)
    static const int8_t kLut2x[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t lut2x = vld1q_s8(kLut2x);
    const uint8x8_t m0f   = vdup_n_u8(0x0F);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        const uint8x8_t raw = vld1_u8(p + b * 8);
        const uint8x8_t lo  = vand_u8(raw, m0f);
        const uint8x8_t hi  = vshr_n_u8(raw, 4);
        const uint8x8x2_t zp = vzip_u8(lo, hi);
        const uint8x16_t nib = vcombine_u8(zp.val[0], zp.val[1]);
        const int8x16_t w8 = vqtbl1q_s8(lut2x, nib);
        const int16x8_t w16_0 = vmovl_s8(vget_low_s8(w8));
        const int16x8_t w16_1 = vmovl_s8(vget_high_s8(w8));
        const float* xb = x + b * 16;
        float32x4_t acc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_0))),  vld1q_f32(xb + 0));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_0))), vld1q_f32(xb + 4));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_1))),  vld1q_f32(xb + 8));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_1))), vld1q_f32(xb + 12));
        vsum = vfmaq_n_f32(vsum, acc, 0.5f * e4[sc[b]]);
    }
    return vaddvq_f32(vsum) * w_outer;
#else
    static constexpr float kE2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* pb = p + b * 8;
        const float* xb = x + b * 16;
        float bs = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = pb[i];
            bs += kE2M1[byte & 0xFu] * xb[2 * i] + kE2M1[byte >> 4] * xb[2 * i + 1];
        }
        sum += bs * e4[sc[b]];
    }
    return sum * w_outer;
#endif
}

inline float dot_mx8_row_h(const uint8_t* w, const uint8_t* sc, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    const float* u8 = ue8m0_lut_h();
    const int nb = cols >> 5;
#if defined(INK_HOST_AVX2)
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < 32; k += 8) {
            const __m256 wf = _mm256_setr_ps(
                e4[wb[k + 0]], e4[wb[k + 1]], e4[wb[k + 2]], e4[wb[k + 3]],
                e4[wb[k + 4]], e4[wb[k + 5]], e4[wb[k + 6]], e4[wb[k + 7]]);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xb + k), acc);
        }
        vsum = _mm256_fmadd_ps(acc, _mm256_set1_ps(u8[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum);
#elif defined(INK_HOST_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 32; k += 4) {
            const float tmp[4] = {e4[wb[k + 0]], e4[wb[k + 1]], e4[wb[k + 2]], e4[wb[k + 3]]};
            acc = vfmaq_f32(acc, vld1q_f32(tmp), vld1q_f32(xb + k));
        }
        vsum = vfmaq_n_f32(vsum, acc, u8[sc[b]]);
    }
    return vaddvq_f32(vsum);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        float bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += e4[wb[i]] * xb[i];
        sum += bs * u8[sc[b]];
    }
    return sum;
#endif
}

inline void quant_q8_1_h(const float* x, InkQ8_1_G64* out, int cols) noexcept {
    const int ng = cols >> 6;
    for (int g = 0; g < ng; ++g) {
        const float* xg = x + g * 64;
        float am = 0.0f;
        for (int i = 0; i < 64; ++i) am = std::max(am, std::fabs(xg[i]));
        const float d = am * (1.0f / 127.0f);
        const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
        int s = 0;
        for (int i = 0; i < 64; ++i) {
            int q = std::max(-127, std::min(127, (int)std::lrintf(xg[i] * inv)));
            out[g].qs[i] = (int8_t)q;
            s += q;
        }
        out[g].d = d;
        out[g].s = d * (float)s;
    }
}

#if defined(INK_HOST_AVX2)
// inkling2 fork: S2_K / S3_K's unpacking by AVX2 (the original's byte loop below, kept for the test):
// value 4 m + j of a group is bits 2 j.. of its byte m, and S3_K's value 8 i + k takes bit k of its byte i of the
// high plane as 4.  The same 64 bytes as the loop.
inline void sk_unpack2_avx2(const uint8_t* p, __m256i& u0, __m256i& u1) noexcept {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i v0 = _mm_and_si128(b, m3);
    const __m128i v1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i v2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i v3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a_lo = _mm_unpacklo_epi8(v0, v1), b_lo = _mm_unpacklo_epi8(v2, v3);
    const __m128i a_hi = _mm_unpackhi_epi8(v0, v1), b_hi = _mm_unpackhi_epi8(v2, v3);
    u0 = _mm256_set_m128i(_mm_unpackhi_epi16(a_lo, b_lo), _mm_unpacklo_epi16(a_lo, b_lo));
    u1 = _mm256_set_m128i(_mm_unpackhi_epi16(a_hi, b_hi), _mm_unpacklo_epi16(a_hi, b_hi));
}

inline void sk_high4_avx2(const uint8_t* hp, __m256i& u0, __m256i& u1) noexcept {
    const __m256i h = _mm256_broadcastsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(hp)));
    const __m256i bits = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i four = _mm256_set1_epi8(4);
    const __m256i i0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i i1 = _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7);
    const __m256i r0 = _mm256_and_si256(_mm256_shuffle_epi8(h, i0), bits);
    const __m256i r1 = _mm256_and_si256(_mm256_shuffle_epi8(h, i1), bits);
    u0 = _mm256_or_si256(u0, _mm256_and_si256(_mm256_cmpeq_epi8(r0, bits), four));
    u1 = _mm256_or_si256(u1, _mm256_and_si256(_mm256_cmpeq_epi8(r1, bits), four));
}

#endif

inline float dot_sk_sub64_h(const uint8_t* pl, int fmt, int g, const InkQ8_1_G64& xb, float eff) noexcept {
    const int8_t* qs = xb.qs;
    int32_t isum = 0;
#if defined(INK_HOST_AVX2)
    if (fmt == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
        const __m128i lo0 = _mm_and_si128(v0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m0f);
        const __m128i lo1 = _mm_and_si128(v1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m0f);
        const __m256i u0 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
        const __m256i u1 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
        isum = avx2_dot_u8_i8_64(u0, u1, qs);
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    } else if (fmt == FMT_S8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i a0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p0), _mm256_sign_epi8(q0, p0)), ones);
        const __m256i a1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p1), _mm256_sign_epi8(q1, p1)), ones);
        isum = avx2_hsum_epi32(_mm256_add_epi32(a0, a1));
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S2_K || fmt == FMT_S3_K || fmt == FMT_S5_K || fmt == FMT_S6_K) {
        alignas(32) uint8_t u[64];
#if !defined(INK_SK_SCALAR_UNPACK)
        if (fmt == FMT_S2_K || fmt == FMT_S3_K) {
            __m256i u0, u1;
            sk_unpack2_avx2(pl + g * 16, u0, u1);
            if (fmt == FMT_S3_K) sk_high4_avx2(pl + 64 + g * 8, u0, u1);
            isum = avx2_dot_u8_i8_64(u0, u1, qs);
            return eff * (xb.d * (float)isum - (fmt == FMT_S2_K ? 1.5f : 3.5f) * xb.s);
        }
#endif
        if (fmt == FMT_S2_K) {
            const uint8_t* p = pl + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t b = p[i];
                u[4 * i + 0] = (uint8_t)(b & 3u);
                u[4 * i + 1] = (uint8_t)((b >> 2) & 3u);
                u[4 * i + 2] = (uint8_t)((b >> 4) & 3u);
                u[4 * i + 3] = (uint8_t)(b >> 6);
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 1.5f * xb.s);
        } else if (fmt == FMT_S3_K) {
            const uint8_t* p = pl + g * 16;
            const uint8_t* hp = pl + 64 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
                u[8 * i + 0] = (uint8_t)((b0 & 3u) | ((h & 1u) << 2));
                u[8 * i + 1] = (uint8_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2));
                u[8 * i + 2] = (uint8_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2));
                u[8 * i + 3] = (uint8_t)((b0 >> 6) | (((h >> 3) & 1u) << 2));
                u[8 * i + 4] = (uint8_t)((b1 & 3u) | (((h >> 4) & 1u) << 2));
                u[8 * i + 5] = (uint8_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2));
                u[8 * i + 6] = (uint8_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2));
                u[8 * i + 7] = (uint8_t)((b1 >> 6) | (((h >> 7) & 1u) << 2));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 3.5f * xb.s);
        } else if (fmt == FMT_S5_K) {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t h = hp[i];
                for (int k = 0; k < 4; ++k) {
                    const uint8_t b = p[4 * i + k];
                    u[8 * i + 2 * k + 0] = (uint8_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4));
                    u[8 * i + 2 * k + 1] = (uint8_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4));
                }
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 15.5f * xb.s);
        } else {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t h = hp[i], b0 = p[2 * i], b1 = p[2 * i + 1];
                u[4 * i + 0] = (uint8_t)((b0 & 0xFu) | ((h & 3u) << 4));
                u[4 * i + 1] = (uint8_t)((b0 >> 4) | (((h >> 2) & 3u) << 4));
                u[4 * i + 2] = (uint8_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4));
                u[4 * i + 3] = (uint8_t)((b1 >> 4) | ((h >> 6) << 4));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (xb.d * (float)isum - 31.5f * xb.s);
        }
    }
#elif defined(INK_HOST_NEON)
    auto neon_dot_s8_64 = [&](const int8_t* w8) -> int32_t {
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < 64; i += 16) {
            const int8x16_t a = vld1q_s8(w8 + i);
            const int8x16_t b = vld1q_s8(qs + i);
            const int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
            const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
            acc = vpadalq_s16(acc, p0);
            acc = vpadalq_s16(acc, p1);
        }
        return vaddvq_s32(acc);
    };
    if (fmt == FMT_S8_K) {
        isum = neon_dot_s8_64(reinterpret_cast<const int8_t*>(pl + g * 64));
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S4_K) {
        alignas(16) int8_t u[64];
        const uint8_t* p = pl + g * 32;
        const uint8x16_t m0f = vdupq_n_u8(0x0F);
        for (int k = 0; k < 2; ++k) {
            const uint8x16_t raw = vld1q_u8(p + k * 16);
            const uint8x16_t lo  = vandq_u8(raw, m0f);
            const uint8x16_t hi  = vshrq_n_u8(raw, 4);
            const uint8x16x2_t zp = vzipq_u8(lo, hi);
            vst1q_s8(u + k * 32 +  0, vreinterpretq_s8_u8(zp.val[0]));
            vst1q_s8(u + k * 32 + 16, vreinterpretq_s8_u8(zp.val[1]));
        }
        isum = neon_dot_s8_64(u);
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    }
#endif
    if (fmt == FMT_S2_K) {
        const uint8_t* p = pl + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 3u) * qs[4 * i + 0] + (int32_t)((b >> 2) & 3u) * qs[4 * i + 1]
                  + (int32_t)((b >> 4) & 3u) * qs[4 * i + 2] + (int32_t)(b >> 6) * qs[4 * i + 3];
        }
        return eff * (xb.d * (float)isum - 1.5f * xb.s);
    } else if (fmt == FMT_S3_K) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
            isum += (int32_t)((b0 & 3u) | ((h & 1u) << 2)) * qs[8 * i + 0]
                  + (int32_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2)) * qs[8 * i + 1]
                  + (int32_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2)) * qs[8 * i + 2]
                  + (int32_t)((b0 >> 6) | (((h >> 3) & 1u) << 2)) * qs[8 * i + 3]
                  + (int32_t)((b1 & 3u) | (((h >> 4) & 1u) << 2)) * qs[8 * i + 4]
                  + (int32_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2)) * qs[8 * i + 5]
                  + (int32_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2)) * qs[8 * i + 6]
                  + (int32_t)((b1 >> 6) | (((h >> 7) & 1u) << 2)) * qs[8 * i + 7];
        }
        return eff * (xb.d * (float)isum - 3.5f * xb.s);
    } else if (fmt == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
        for (int i = 0; i < 32; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 0xFu) * qs[2 * i] + (int32_t)(b >> 4) * qs[2 * i + 1];
        }
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    } else if (fmt == FMT_S5_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t h = hp[i];
            for (int k = 0; k < 4; ++k) {
                const uint8_t b = p[4 * i + k];
                isum += (int32_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4)) * qs[8 * i + 2 * k]
                      + (int32_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4)) * qs[8 * i + 2 * k + 1];
            }
        }
        return eff * (xb.d * (float)isum - 15.5f * xb.s);
    } else if (fmt == FMT_S6_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t h = hp[i];
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1];
            isum += (int32_t)((b0 & 0xFu) | ((h & 3u) << 4)) * qs[4 * i + 0]
                  + (int32_t)((b0 >> 4) | (((h >> 2) & 3u) << 4)) * qs[4 * i + 1]
                  + (int32_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4)) * qs[4 * i + 2]
                  + (int32_t)((b1 >> 4) | ((h >> 6) << 4)) * qs[4 * i + 3];
        }
        return eff * (xb.d * (float)isum - 31.5f * xb.s);
    } else if (fmt == FMT_S8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        for (int i = 0; i < 64; ++i) isum += (int32_t)p[i] * qs[i];
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S1_5) {
        for (int i = 0; i < 64; ++i) {
            const int idx = g * 64 + i;
            int b = pl[idx / 5];
            int k = idx % 5;
            for (int r = 0; r < k; ++r) b /= 3;
            int tr = (b % 3) - 1;
            isum += tr * qs[i];
        }
        return eff * xb.d * (float)isum;
    }
    return 0.0f;
}

inline void eval_expert_f32_h(const InkHostExpertMeta& em, const float* x, float* y_out) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const uint8_t* body = em.body;
    if (!body) {
        std::memset(y_out, 0, (size_t)D * sizeof(float));
        return;
    }
    float act[I];

    // 1. Gate + Up + SITU(4.0, 25.0)
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = 0; j < I; ++j) {
            const float g = dot_sf4_row_h(bp + (size_t)(2 * j) * (D / 2), bs + (size_t)(2 * j) * (D / 16), w_outer, x, D);
            const float u = dot_sf4_row_h(bp + (size_t)(2 * j + 1) * (D / 2), bs + (size_t)(2 * j + 1) * (D / 16), w_outer, x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = 0; j < I; ++j) {
            const float g = dot_mx8_row_h(bp + (size_t)(2 * j) * D, bs + (size_t)(2 * j) * (D / 32), x, D);
            const float u = dot_mx8_row_h(bp + (size_t)(2 * j + 1) * D, bs + (size_t)(2 * j + 1) * (D / 32), x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else {
        InkQ8_1_G64 xq[D / 64];
        quant_q8_1_h(x, xq, D);
        const int sb = ink_sb_bytes(em.gu_code);
        const size_t row_b = 9ull * sb;
        const uint8_t* gu_base = body + I * 2;
        for (int j = 0; j < I; ++j) {
            const uint8_t* gr = gu_base + (size_t)(2 * j) * row_b;
            const uint8_t* ur = gr + row_b;
            float g = 0.0f, u = 0.0f;
            for (int s = 0; s < 9; ++s) {
                const uint8_t* bg = gr + s * sb;
                const uint8_t* bu = ur + s * sb;
                uint16_t dg, du;
                std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
                const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
                for (int sg = 0; sg < 4; ++sg) {
                    g += dot_sk_sub64_h(bg + 6, em.gu_code, sg, xq[s * 4 + sg], Dg * ((float)bg[2 + sg] * (1.0f / 255.0f)));
                    u += dot_sk_sub64_h(bu + 6, em.gu_code, sg, xq[s * 4 + sg], Du * ((float)bu[2 + sg] * (1.0f / 255.0f)));
                }
            }
            uint16_t sq_bits;
            std::memcpy(&sq_bits, body + j * 2, 2);
            const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb)) * inv_sq;
        }
    }

    // 2. Down projection [D, I]
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = 0; r < D; ++r) {
            y_out[r] = dot_sf4_row_h(bp + (size_t)r * (I / 2), bs + (size_t)r * (I / 16), w_outer, act, I);
        }
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = 0; r < D; ++r) {
            y_out[r] = dot_mx8_row_h(bp + (size_t)r * I, bs + (size_t)r * (I / 32), act, I);
        }
    } else {
        InkQ8_1_G64 hq[I / 64];
        quant_q8_1_h(act, hq, I);
        const int sb = ink_sb_bytes(em.dn_code);
        const size_t pair_b = 9ull * sb;
        const uint8_t* dn_base = body + em.dn_off;
        for (int pr = 0; pr < D / 2; ++pr) {
            const uint8_t* blk0 = dn_base + (size_t)pr * pair_b;
            float y0 = 0.0f, y1 = 0.0f;
            for (int s = 0; s < 9; ++s) {
                const uint8_t* blk = blk0 + s * sb;
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const float Dsc = f16_to_f32_h(d16);
                for (int sg = 0; sg < 4; ++sg) {
                    const int sub_g = s * 4 + sg;
                    const float eff = Dsc * ((float)blk[2 + sg] * (1.0f / 255.0f));
                    if (sub_g < 18) y0 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g], eff);
                    else            y1 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g - 18], eff);
                }
            }
            y_out[2 * pr]     = y0;
            y_out[2 * pr + 1] = y1;
        }
    }
}

}  // namespace host_eval

}  // namespace strata::ink
