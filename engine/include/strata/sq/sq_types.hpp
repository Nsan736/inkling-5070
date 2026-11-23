// inkling2 fork: the expert and matrix formats of Strata's .strata v1.1 files that this engine runs (SQ-10, SQ-15).
// From Strata v0.2.19's include/strata/quant/sk_codecs.hpp (MIT, Copyright (c) 2026 Niko1221 and the Strata
// contributors):
//   S_K (S2..S6, S8)  256 values a block: fp16 D, four u8 sub-scales s (64 values each), then the codes q packed;
//                     a value is D * (s / 255) * (q - HALF), HALF = 2^(bits-1) - 0.5 (S8: signed q, no HALF)
//   SF4              NVFP4: E2M1 nibbles (low = even column), an F8_E4M3 scale every 16 values, and an f32 outer
//                     scale amax / denom for gate|up and one for down
// An S_K expert's body: fp16 sq[1152] (h is divided by it before down), gate|up as 2 * 1152 rows of 9 blocks (row 2j
// gate j, row 2j + 1 up j), then from dn_off down as 1152 pairs of rows, each pair 2 * 1152 values in 9 blocks.
// An SF4 expert's: f32 {amax13, denom13, amax2, denom2}, gate|up's nibbles [2304][1152] and scales [2304][144],
// then from dn_off down's nibbles [2304][576] and scales [2304][72].
#pragma once

#include <cstdint>
#include <string>

#if defined(__CUDACC__)
#define SQ_HD __host__ __device__
#else
#define SQ_HD
#endif

namespace strata::sq {

enum Code : uint8_t { S2_K = 2, S3_K = 3, S4_K = 4, S5_K = 5, S6_K = 6, S8_K = 8, SF4 = 15, MX8 = 16 };

constexpr uint32_t kTypeBase = 200;   ///< a matrix of an S_K code in the engine's tensors: type 200 + code

SQ_HD constexpr bool is_sk(int c) { return c == S2_K || c == S3_K || c == S4_K || c == S5_K || c == S6_K || c == S8_K; }
SQ_HD constexpr bool is_sq_type(uint32_t t) { return t >= kTypeBase && t < kTypeBase + 32 && is_sk((int) (t - kTypeBase)); }

/// Bytes of a 256-value block.
SQ_HD constexpr int sk_block_bytes(int c) {
    return c == S2_K ? 70 : c == S3_K ? 102 : c == S4_K ? 134 : c == S5_K ? 166 : c == S6_K ? 198 : c == S8_K ? 262 : 0;
}
/// HALF, the codes' midpoint (S8: 0)
SQ_HD constexpr float sk_half(int c) {
    return c == S2_K ? 1.5f : c == S3_K ? 3.5f : c == S4_K ? 7.5f : c == S5_K ? 15.5f : c == S6_K ? 31.5f : 0.0f;
}

inline int code_of(const std::string& f) {
    if (f == "S2_K") return S2_K;
    if (f == "S3_K") return S3_K;
    if (f == "S4_K") return S4_K;
    if (f == "S5_K") return S5_K;
    if (f == "S6_K") return S6_K;
    if (f == "S8_K") return S8_K;
    if (f == "SF4") return SF4;
    if (f == "MX8") return MX8;
    return 0;
}

inline const char* code_name(int c) {
    switch (c) {
    case S2_K: return "S2_K";
    case S3_K: return "S3_K";
    case S4_K: return "S4_K";
    case S5_K: return "S5_K";
    case S6_K: return "S6_K";
    case S8_K: return "S8_K";
    case SF4: return "SF4";
    case MX8: return "MX8";
    default: return "?";
    }
}

/// The experts this engine runs: gate|up and down both S_K, or both SF4.
SQ_HD constexpr bool expert_codes_ok(int gu, int dn) { return (is_sk(gu) && is_sk(dn)) || (gu == SF4 && dn == SF4); }

}  // namespace strata::sq
