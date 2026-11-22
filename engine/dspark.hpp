// dspark.hpp - Inkling 2 Mini's DSpark block drafter (the official checkpoint's mtp.safetensors, BF16) for speculative
// decoding in strata-inkling2.
//
// Ported from Strata pull request #6506's ik2 drafter (engine/src/ik2/ik2_draft.cu, MIT License, Copyright (c) 2026
// Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors), itself after SGLang's
// inkling2_mtp.py / dspark.py.  Changes here, for one 12 GB card: the drafter's context caches are fp16 (the window
// layers a ring of their window); its attention is one launch for a group's 24 query rows that reads each cached row
// once (split over the keys, then merged); the base logits and the Markov head's fold go through the target's own
// output head (Engine::lin), so no fp16 copy of that head is kept.  What the drafter proposes only decides how many
// tokens a verify step keeps: the target checks every one, so none of this changes the answers.
//
//   * Taps: the target's stream at the start of visits 5, 23 and 41 and its final normed hidden, a row per position.
//     For each committed position each draft layer mixes them (softmax(fuse_weight[layer])), RMS-norms them (its
//     dflare_norm) and projects K / V (k_proj_ctx, v_proj_ctx; K's per-head norm k_norm_ctx and NeoX RoPE, theta 1e6)
//     into its context cache.  The last committed position's mix (fuse_weight[3], the Markov head's dflare_norm) is
//     f_rnn.
//   * A block of 8 rows at positions P .. P+7: the anchor (the last token the target chose, embedded and normed as the
//     target embeds it) then 7 mask_embed rows, through 3 dense layers (per-head q / k norm, RoPE, SiLU MLP), the
//     rows seeing each other and the context (layers 0 and 2: the 1023 positions before P); rows 1 .. 7 of the final
//     norm through the target's head are the base logits.
//   * The Markov head drafts the 7 tokens one by one: a GRU-like cell over [state; markov_w1[prev]; hidden; f_rnn]
//     gives a latent; the token is argmax(base + fold_w . latent + fold_b), fold_w = (head . hidden_out)^T.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::poly::dspark {

struct Drafter;

constexpr int kBlock = 8;   ///< the anchor and the 7 drafted tokens
constexpr int kTaps = 4;

/// Loads mtp.safetensors onto the current device.  C: the target's width; ctx: the positions the caches hold.
Drafter* load(const std::string& path, int C, int ctx, void* stream);
void destroy(Drafter* d);
/// VRAM it holds, bytes.
size_t bytes(const Drafter* d);
int markov_rank(const Drafter* d);

/// The Markov head's fold: hidden_out's columns (R rows of C floats, device) for the caller to put through the
/// target's head, and its bias (C floats); then the products (R rows of n_rows floats, and n_rows floats) back.
const float* fold_inputs(Drafter* d);   ///< R + 1 rows of C floats: hidden_out[:, r] for r < R, then the bias
void fold(Drafter* d, const float* products, int n_rows);   ///< products: R + 1 rows of n_rows floats (device)

/// The context caches for positions pos0 .. pos0 + m - 1 from the taps: taps[k] + i * C is row i of tap k (device).
void write_context(Drafter* d, const float* const taps[kTaps], int m, int64_t pos0);
/// f_rnn from row `row` of the taps.
void set_f_rnn(Drafter* d, const float* const taps[kTaps], int row);

/// The block's layers for the anchor at position P: anchor_h, its normed embedding (device, C floats).  Returns
/// rows 1 .. 7 of the block's final norm (device, 7 rows of C floats) for the caller's base logits.
const float* block(Drafter* d, int anchor, const float* anchor_h, int64_t P);
/// The Markov head over the base logits (device, 7 rows of n_rows floats; the argmax looks at the first n_vocab):
/// the 7 drafted tokens into out (host).
void markov(Drafter* d, const float* base, int n_rows, int n_vocab, int32_t* out);

}  // namespace strata::poly::dspark
