// inkling2 fork, INKLING2_SQ_W4A4 (opt-in): a .strata prompt batch's SF4 (NVFP4) experts by Strata's sm_120a
// kernels (ink_moe_fp4_sm120.cu, MIT): the activations rounded to FP4 (W4A4), x and h, as Strata runs them on RTX
// 50.  Off by default: the prompt batches' SF4 experts take fp16 activations (kernels_sq.cu's sq_rows_round()).
#include "ink_moe_fp4_sm120.cuh"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

static_assert(sizeof(SqTcExpert) == sizeof(ink_fp4::InkExpertDev), "SqTcExpert is Strata's InkExpertDev");

bool sq_w4a4_available() { return ink_fp4::sm120::available(); }

void sq_rows_w4a4(const SqTcTables& t, int n_tiles, const float* x, int m, void* xa, void* ha, float* y,
                  void* stream) {
    if (n_tiles <= 0) return;
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    ink_fp4::InkMoeTables tb;
    tb.cnt = t.cnt;
    tb.off = t.off;
    tb.active = t.active;
    tb.nact = t.nact;
    tb.pairs = t.pairs;
    tb.tok_idx = t.tok_idx;
    tb.n_tiles = t.n_tiles;
    tb.tiles = reinterpret_cast<int2*>(t.tiles);
    const auto* ex = static_cast<const ink_fp4::InkExpertDev*>(t.experts);
    ink_fp4::sm120::quantize_act_fp4(x, m, kSqEmbd, static_cast<uint8_t*>(xa), st);
    ink_fp4::sm120::moe_sf4_gu_tc(nullptr, ex, tb, static_cast<const uint8_t*>(xa), static_cast<uint8_t*>(ha),
                                  n_tiles, 16, st);
    ink_fp4::sm120::moe_sf4_dn_tc(nullptr, ex, tb, static_cast<const uint8_t*>(ha), t.w, y, n_tiles, 16, st);
}

}  // namespace strata::poly
