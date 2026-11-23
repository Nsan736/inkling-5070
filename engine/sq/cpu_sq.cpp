// inkling2 fork: .strata experts on the CPU.  Each expert's gate|up rows, its activation and its down rows are
// Strata's host evaluator (sq_cpu_eval.hpp, host_eval::eval_expert_f32_h) term for term: the same quantization of x
// and of h, the same integer sums a 64-value group, the same float order within a row.  What this file adds is the
// division of that work over the workers (rows of gate|up, then rows of down) and each token's sum of its experts in
// their order.
#include "cpu_sq.hpp"

#include "sq_cpu_eval.hpp"
#include "strata/sq/sq_types.hpp"

#include <vector>

namespace strata::poly {

namespace {

using namespace strata::ink;
using namespace strata::ink::host_eval;

constexpr int D = kInkHidden, I = kInkInter;
constexpr int kGuRows = 64;     // gate|up rows (pairs j) a task
constexpr int kDnPairs = 16;    // down pairs of output rows a task

// act[j] for j in [j0, j1) of one expert, as eval_expert_f32_h's step 1
void gate_up(const SqCpuExpert& e, const float* x, const InkQ8_1_G64* xq, float* act, int j0, int j1) {
    const uint8_t* body = e.body;
    if (e.gu == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t) (2 * I) * (D / 2);
        for (int j = j0; j < j1; ++j) {
            const float g = dot_sf4_row_h(bp + (size_t) (2 * j) * (D / 2), bs + (size_t) (2 * j) * (D / 16), w_outer, x, D);
            const float u =
                dot_sf4_row_h(bp + (size_t) (2 * j + 1) * (D / 2), bs + (size_t) (2 * j + 1) * (D / 16), w_outer, x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
        return;
    }
    const int sb = ink_sb_bytes(e.gu);
    const size_t row_b = 9ull * sb;
    const uint8_t* gu_base = body + I * 2;
    for (int j = j0; j < j1; ++j) {
        const uint8_t* gr = gu_base + (size_t) (2 * j) * row_b;
        const uint8_t* ur = gr + row_b;
        float g = 0.0f, u = 0.0f;
        for (int s = 0; s < 9; ++s) {
            const uint8_t* bg = gr + s * sb;
            const uint8_t* bu = ur + s * sb;
            uint16_t dg, du;
            std::memcpy(&dg, bg, 2);
            std::memcpy(&du, bu, 2);
            const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
            for (int sg = 0; sg < 4; ++sg) {
                g += dot_sk_sub64_h(bg + 6, e.gu, sg, xq[s * 4 + sg], Dg * ((float) bg[2 + sg] * (1.0f / 255.0f)));
                u += dot_sk_sub64_h(bu + 6, e.gu, sg, xq[s * 4 + sg], Du * ((float) bu[2 + sg] * (1.0f / 255.0f)));
            }
        }
        uint16_t sq_bits;
        std::memcpy(&sq_bits, body + j * 2, 2);
        const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
        const float sig = 1.0f / (1.0f + std::exp(-g));
        act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb)) * inv_sq;
    }
}

// y[r] for the output rows 2 * pr and 2 * pr + 1, pr in [p0, p1), as eval_expert_f32_h's step 2
void down(const SqCpuExpert& e, const float* act, const InkQ8_1_G64* hq, float* y, int p0, int p1) {
    const uint8_t* body = e.body;
    if (e.dn == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + e.dn_off;
        const uint8_t* bs = bp + (size_t) D * (I / 2);
        for (int r = 2 * p0; r < 2 * p1; ++r)
            y[r] = dot_sf4_row_h(bp + (size_t) r * (I / 2), bs + (size_t) r * (I / 16), w_outer, act, I);
        return;
    }
    const int sb = ink_sb_bytes(e.dn);
    const size_t pair_b = 9ull * sb;
    const uint8_t* dn_base = body + e.dn_off;
    for (int pr = p0; pr < p1; ++pr) {
        const uint8_t* blk0 = dn_base + (size_t) pr * pair_b;
        float y0 = 0.0f, y1 = 0.0f;
        for (int s = 0; s < 9; ++s) {
            const uint8_t* blk = blk0 + s * sb;
            uint16_t d16;
            std::memcpy(&d16, blk, 2);
            const float Dsc = f16_to_f32_h(d16);
            for (int sg = 0; sg < 4; ++sg) {
                const int sub_g = s * 4 + sg;
                const float eff = Dsc * ((float) blk[2 + sg] * (1.0f / 255.0f));
                if (sub_g < 18) y0 += dot_sk_sub64_h(blk + 6, e.dn, sg, hq[sub_g], eff);
                else y1 += dot_sk_sub64_h(blk + 6, e.dn, sg, hq[sub_g - 18], eff);
            }
        }
        y[2 * pr] = y0;
        y[2 * pr + 1] = y1;
    }
}

}  // namespace

void cpu_experts_sq(Workers& workers, const SqCpuExpert* experts, int n, int rows, const float* x, float* out,
                    const std::function<void()>* meanwhile) {
    static thread_local std::vector<InkQ8_1_G64> xq, hq;
    static thread_local std::vector<float> act, y;
    xq.resize((size_t) rows * (D / 64));
    hq.resize((size_t) n * (I / 64));
    act.resize((size_t) n * I);
    y.resize((size_t) n * D);
    InkQ8_1_G64* xqp = xq.data();
    InkQ8_1_G64* hqp = hq.data();
    float* actp = act.data();
    float* yp = y.data();
    for (int r = 0; r < rows; ++r) quant_q8_1_h(x + (size_t) r * D, xqp + (size_t) r * (D / 64), D);

    const int first = meanwhile ? 1 : 0;
    const int gu_tasks = I / kGuRows;
    workers.run(first + n * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int j = t / gu_tasks, c = t % gu_tasks;
        const SqCpuExpert& e = experts[j];
        gate_up(e, x + (size_t) e.row * D, xqp + (size_t) e.row * (D / 64), actp + (size_t) j * I, c * kGuRows,
                (c + 1) * kGuRows);
    });
    workers.run(n, [&](int j) {
        if (strata::sq::is_sk(experts[j].dn)) quant_q8_1_h(actp + (size_t) j * I, hqp + (size_t) j * (I / 64), I);
    });
    const int dn_tasks = (D / 2) / kDnPairs;
    workers.run(dn_tasks, [&](int t) {
        const int p0 = t * kDnPairs, p1 = p0 + kDnPairs;
        for (int j = 0; j < n; ++j)
            down(experts[j], actp + (size_t) j * I, hqp + (size_t) j * (I / 64), yp + (size_t) j * D, p0, p1);
        // each token's experts in their order
        int j = 0;
        for (int row = 0; row < rows; ++row) {
            float* o = out + (size_t) row * D;
            for (int r = 2 * p0; r < 2 * p1; ++r) o[r] = 0.f;
            for (; j < n && experts[j].row == row; ++j)
                for (int r = 2 * p0; r < 2 * p1; ++r) o[r] += experts[j].weight * yp[(size_t) j * D + r];
        }
    });
}

}  // namespace strata::poly
