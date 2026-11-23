// inkling2 fork: a few tokens' .strata experts (S_K, SF4) on the CPU, as cpu_experts() does a GGUF's.
#pragma once

#include "../cpu_experts.hpp"

#include <cstdint>
#include <functional>

namespace strata::poly {

struct SqCpuExpert {
    const uint8_t* body = nullptr;
    uint32_t dn_off = 0;
    uint8_t gu = 0, dn = 0;
    float weight = 0.f;   ///< the router's weight of it for its token
    int row = 0;          ///< its token's row of x and out; the entries are in token order
};

/// out[row] = sum of weight * expert(x[row]) over each row's entries, in their order (n_embd 2304, n_ff 1152).
/// meanwhile: one task run beside the experts (the card's launches), as cpu_experts() takes it.
void cpu_experts_sq(Workers& workers, const SqCpuExpert* experts, int n, int rows, const float* x, float* out,
                    const std::function<void()>* meanwhile);

}  // namespace strata::poly
