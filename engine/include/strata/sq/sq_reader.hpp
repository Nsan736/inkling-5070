// inkling2 fork: Strata's .strata v1.1 files of Inkling 2 Mini (Strata v0.2.19's SQ-10 and SQ-15 builds), read as the
// GGUF this engine knows.  The layout is Strata's (engine/src/artifact/strata_reader.cpp and
// engine/src/inkling2/ink_manifest.h, MIT, Copyright (c) 2026 Niko1221 and the Strata contributors); this file reads it
// on its own.
//
//   header    the first 4 KiB: "STRATA\0\1", then 6 sections of {offset, length, sha256, id, flags} from 0x10
//   S0        the manifest: zstd-compressed JSON (the model's config, the trunk's tensors with their formats)
//   S1        the trunk (and the DSpark drafter, not read here), each tensor at its manifest offset
//   S3        an expert table of 32-byte entries, layer * 128 + rank, and the order the experts go to VRAM
//   S4        the experts' bodies, each one block of its formats
//
// The trunk's tensors get the names, shapes (GGUF order) and types of Unsloth's GGUF of the same model, so the loader
// takes them as it takes that file's (tools/strata_map.py matched them by value):
//   F32, BF16   in place in the mapped file; two are changed as llama.cpp's converter changes them (conversion/
//               inkling2.py): a global layer's static convolution taps (flipped and transposed) and the relative
//               table (transposed), into memory of their own
//   Q8_ROW      int8 rows with one f32 scale each: rewritten into Q8_0 blocks whose fp16 scale is 1.0, so every code
//               keeps its value, beside a tensor NAME.rscale of the rows' scales that the engine multiplies each
//               product by (Dev::rscale) - the same value as code times scale, without a rounding of the scale
//   S2_K..S8_K  Strata's own blocks (sq_types.hpp), in place, as type 200 + code (the output matrix: S6_K)
// The routed experts are not tensors here: Parsed::experts has each one's formats and body, which the engine runs
// with its SQ kernels.
#pragma once

#include "sq_json.hpp"
#include "sq_types.hpp"

#include <zstd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::sq {

struct Expert {
    const uint8_t* body = nullptr;   ///< in the mapped file
    uint32_t bytes = 0;
    uint32_t dn_off = 0;             ///< from the body's start to its down projection
    uint8_t gu = 0, dn = 0;          ///< SkCode of gate|up and of down
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;     ///< GGUF order
    uint32_t type = 0;
    const uint8_t* ptr = nullptr;
    uint64_t bytes = 0;
};

struct Parsed {
    std::string build;               ///< "SQ-10", "SQ-15", ...
    int n_layer = 0, n_expert = 0, n_embd = 0, n_ff = 0;
    std::vector<Tensor> tensors;
    std::vector<std::unique_ptr<uint8_t[]>> owned;
    std::vector<Expert> experts;     ///< layer * n_expert + expert id
    std::vector<uint16_t> load_order;   ///< the file's VRAM order of the experts (table indices), or empty
};

inline bool is_strata(const uint8_t* base, size_t size) {
    return size >= 4096 && std::memcmp(base, "STRATA\0\1", 8) == 0;
}

namespace detail {

inline std::string layer_name(const std::string& rest, bool& special) {
    struct Rule {
        const char* sq;
        const char* gguf;
    };
    static const Rule rules[] = {
        {"attn.attn_sink_proj.weight", "attn_sink_proj.weight"}, {"attn.gate_attn.weight", "attn_gate.weight"},
        {"attn.k_sconv.weight", "attn_k_conv.weight"},           {"attn.v_sconv.weight", "attn_v_conv.weight"},
        {"attn.k_sconv.weight_proj", "attn_k_conv_proj.weight"}, {"attn.v_sconv.weight_proj", "attn_v_conv_proj.weight"},
        {"attn.q_l2_clip.scale", "attn_q_clip.weight"},          {"attn.rel_l2_clip.scale", "attn_rel_clip.weight"},
        {"attn.rel_logits_proj.proj", "attn_rel_bias.weight"},  {"attn.wk_dv.weight", "attn_k.weight"},
        {"attn.wv_dv.weight", "attn_v.weight"},                  {"attn.wo_ud.weight", "attn_output.weight"},
        {"attn.wq_du.weight", "attn_q.weight"},                  {"attn.wr_du.weight", "attn_rel.weight"},
        {"attn_norm.weight", "attn_norm.weight"},                {"attn_sconv.weight", "attn_post_conv.weight"},
        {"k_input_norm.weight", "attn_k_inp_norm.weight"},       {"q_input_norm.weight", "attn_q_inp_norm.weight"},
        {"v_input_norm.weight", "attn_v_inp_norm.weight"},       {"mlp.gate.bias", "exp_probs_b.bias"},
        {"mlp.gate.weight", "ffn_gate_inp.weight"},              {"mlp_norm.weight", "ffn_norm.weight"},
        {"mlp_sconv.weight", "ffn_post_conv.weight"},
        {"controllers.attn.down_weight", "loop_attn_down.weight"},
        {"controllers.attn.forget_weight", "loop_attn_forget.weight"},
        {"controllers.attn.gain_weight", "loop_attn_gain.weight"},
        {"controllers.k.down_weight", "loop_k_down.weight"},     {"controllers.k.forget_weight", "loop_k_forget.weight"},
        {"controllers.k.gain_weight", "loop_k_gain.weight"},     {"controllers.v.down_weight", "loop_v_down.weight"},
        {"controllers.v.forget_weight", "loop_v_forget.weight"}, {"controllers.v.gain_weight", "loop_v_gain.weight"},
        {"controllers.mlp.down_weight", "loop_mlp_down.weight"},
        {"controllers.mlp.forget_weight", "loop_mlp_forget.weight"},
        {"controllers.mlp.gain_weight", "loop_mlp_gain.weight"},
    };
    special = false;
    if (rest == "attn.k_sconv.static_w" || rest == "attn.v_sconv.static_w") {
        special = true;   // flip(0).T, as llama.cpp's converter does
        return rest[5] == 'k' ? "attn_k_conv.weight" : "attn_v_conv.weight";
    }
    for (const Rule& r : rules)
        if (rest == r.sq) return r.gguf;
    return {};
}

/// The GGUF name of a trunk tensor, or empty for one this engine does not take.  transform: 0 none, 1 transpose,
/// 2 flip the first axis and transpose.
inline std::string gguf_name(const std::string& n, int& transform) {
    transform = 0;
    static const std::pair<const char*, const char*> top[] = {
        {"model.llm.embed.weight", "token_embd.weight"},
        {"model.llm.embed_norm.weight", "token_embd_norm.weight"},
        {"model.llm.norm.weight", "output_norm.weight"},
        {"model.llm.unembed.weight", "output.weight"},
        {"model.llm.attn_res.output_query", "output_res_query.weight"},
        {"model.llm.middle_cycle_cross_pass_queries", "loop_res_query.weight"},
        {"model.llm.middle_cycle_cross_pass_kv_queries", "loop_kv_res_query.weight"},
    };
    for (const auto& t : top)
        if (n == t.first) return t.second;
    const auto index = [&](const char* prefix, int& at) -> int {
        const size_t len = std::strlen(prefix);
        if (n.compare(0, len, prefix) != 0) return -1;
        size_t end = len;
        while (end < n.size() && n[end] >= '0' && n[end] <= '9') ++end;
        if (end == len) return -1;
        at = (int) end;
        return std::stoi(n.substr(len, end - len));
    };
    int at = 0;
    // attention-residual queries: a layer's attention's (from layer 1) and its MLP's
    if (int i = index("model.llm.attn_res.attn_queries.", at); i >= 0 && at == (int) n.size())
        return "blk." + std::to_string(i + 1) + ".attn_res_query.weight";
    if (int i = index("model.llm.attn_res.mlp_queries.", at); i >= 0 && at == (int) n.size())
        return "blk." + std::to_string(i) + ".ffn_res_query.weight";
    if (int L = index("model.llm.layers.", at); L >= 0 && at < (int) n.size() && n[(size_t) at] == '.') {
        bool special = false;
        const std::string g = layer_name(n.substr((size_t) at + 1), special);
        if (g.empty()) return {};
        if (special) transform = 2;
        else if (g == "attn_rel_bias.weight") transform = 1;
        return "blk." + std::to_string(L) + "." + g;
    }
    return {};
}

inline uint16_t f16_one() { return 0x3C00; }

}  // namespace detail

/// Reads the mapped .strata file at base.  Throws on anything it does not take.
inline Parsed parse(const uint8_t* base, size_t size) {
    if (!is_strata(base, size)) throw std::runtime_error("not a .strata file");
    struct Sec {
        uint64_t off, len;
        uint32_t id, flags;
    } sec[6];
    for (int i = 0; i < 6; ++i) {
        std::memcpy(&sec[i].off, base + 0x10 + i * 64, 8);
        std::memcpy(&sec[i].len, base + 0x10 + i * 64 + 8, 8);
        std::memcpy(&sec[i].id, base + 0x10 + i * 64 + 48, 4);
        std::memcpy(&sec[i].flags, base + 0x10 + i * 64 + 52, 4);
        if (sec[i].off + sec[i].len > size) throw std::runtime_error(".strata: section " + std::to_string(i) + " past EOF");
    }
    // S0: the manifest
    std::string json;
    {
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        ZSTD_inBuffer in{base + sec[0].off, (size_t) sec[0].len, 0};
        std::vector<char> chunk(size_t(1) << 20);
        while (in.pos < in.size) {
            ZSTD_outBuffer out{chunk.data(), chunk.size(), 0};
            const size_t r = ZSTD_decompressStream(dctx, &out, &in);
            if (ZSTD_isError(r)) {
                ZSTD_freeDCtx(dctx);
                throw std::runtime_error(std::string(".strata manifest: ") + ZSTD_getErrorName(r));
            }
            json.append(chunk.data(), out.pos);
            if (r == 0 && out.pos == 0) break;
        }
        ZSTD_freeDCtx(dctx);
    }
    const ink2::JVal man = ink2::json_parse(json);
    Parsed p;
    if (man.at("format").str() != "strata-v1.1") throw std::runtime_error(".strata: format " + man.at("format").str());
    if (man.at("arch").str() != "inkling2_loop") throw std::runtime_error(".strata: arch " + man.at("arch").str());
    p.build = man.at("build").str();
    const ink2::JVal& tc = man.at("config").at("text_config");
    p.n_layer = (int) tc.at("num_hidden_layers").i();
    p.n_expert = (int) tc.at("n_routed_experts").i();
    p.n_embd = (int) tc.at("hidden_size").i();
    p.n_ff = (int) tc.at("intermediate_size").i();
    if (p.n_embd != 2304 || p.n_ff != 1152 || p.n_layer != 24 || p.n_expert != 128)
        throw std::runtime_error(".strata: not Inkling 2 Mini's shape");

    // S1: the trunk
    const uint8_t* s1 = base + sec[1].off;
    for (const ink2::JVal& t : man.at("trunk_tensors").a) {
        const std::string sqname = t.at("name").str(), fmt = t.at("format").str();
        std::vector<uint64_t> shape;
        for (const ink2::JVal& d : t.at("shape").a) shape.push_back((uint64_t) d.i());
        const uint64_t off = (uint64_t) t.at("offset").i(), bytes = (uint64_t) t.at("bytes").i();
        if (off + bytes > sec[1].len) throw std::runtime_error(".strata: " + sqname + " is outside its section");
        int transform = 0;
        const std::string name = detail::gguf_name(sqname, transform);
        if (name.empty()) throw std::runtime_error(".strata: no GGUF name for " + sqname);
        const uint8_t* src = s1 + off;
        // the GGUF shape: the axes reversed, a convolution's [channels, 1, taps] as [taps, channels]
        std::vector<uint64_t> g;
        for (size_t i = shape.size(); i-- > 0;)
            if (!(shape.size() == 3 && i == 1 && shape[1] == 1)) g.push_back(shape[i]);
        Tensor out;
        out.name = name;
        if (fmt == "F32" || fmt == "BF16") {
            out.type = fmt == "F32" ? 0u : 30u;
            out.shape = g;
            out.ptr = src;
            out.bytes = bytes;
            if (transform) {
                if (fmt != "F32" || shape.size() != 2) throw std::runtime_error(".strata: " + sqname + " to transpose");
                const uint64_t R = shape[0], C = shape[1];
                auto buf = std::make_unique<uint8_t[]>(bytes);
                float* dst = (float*) buf.get();
                const float* a = (const float*) src;
                for (uint64_t r = 0; r < R; ++r)
                    for (uint64_t c = 0; c < C; ++c)
                        dst[c * R + r] = a[(transform == 2 ? R - 1 - r : r) * C + c];
                out.shape = {R, C};   // GGUF order of the transposed [C, R]
                out.ptr = buf.get();
                p.owned.push_back(std::move(buf));
            }
        } else if (fmt == "Q8_ROW") {
            if (shape.size() != 2 || shape[1] % 32) throw std::runtime_error(".strata: " + sqname + " is not [rows, k*32]");
            const uint64_t R = shape[0], K = shape[1];
            if (bytes < R * 4 + R * K) throw std::runtime_error(".strata: " + sqname + " is short");
            const uint64_t qbytes = R * (K / 32) * 34;
            auto buf = std::make_unique<uint8_t[]>(qbytes);
            const int8_t* q = (const int8_t*) (src + R * 4);
            uint8_t* d = buf.get();
            const uint16_t one = detail::f16_one();
            for (uint64_t r = 0; r < R; ++r)
                for (uint64_t b = 0; b < K / 32; ++b) {
                    std::memcpy(d, &one, 2);
                    std::memcpy(d + 2, q + r * K + b * 32, 32);
                    d += 34;
                }
            out.type = 8;   // GGML_TYPE_Q8_0
            out.shape = {K, R};
            out.ptr = buf.get();
            out.bytes = qbytes;
            p.owned.push_back(std::move(buf));
            Tensor rs;
            rs.name = name.substr(0, name.size() - std::strlen(".weight")) + ".rscale";
            rs.type = 0;
            rs.shape = {R};
            rs.ptr = src;   // the f32 row scales come first in the tensor
            rs.bytes = R * 4;
            p.tensors.push_back(std::move(rs));
        } else {
            const int code = code_of(fmt);
            if (code <= 0 || !is_sk(code) || shape.size() != 2 || shape[1] % 256)
                throw std::runtime_error(".strata: " + sqname + " in " + fmt + " is not taken");
            out.type = kTypeBase + (uint32_t) code;
            out.shape = {shape[1], shape[0]};
            out.ptr = src;
            out.bytes = bytes;
            if (bytes < shape[0] * (shape[1] / 256) * (uint64_t) sk_block_bytes(code))
                throw std::runtime_error(".strata: " + sqname + " is short");
        }
        p.tensors.push_back(std::move(out));
    }

    // S3 and S4: the experts
    const bool ordered = (sec[3].flags & 1u) != 0;
    const size_t entry = ordered ? 34 : 32, n = (size_t) (sec[3].len / entry);
    if (n != (size_t) p.n_layer * (size_t) p.n_expert) throw std::runtime_error(".strata: not one entry an expert");
    p.experts.assign(n, Expert{});
    const uint8_t* s3 = base + sec[3].off;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = s3 + i * 32;
        uint16_t id;
        uint64_t off;
        uint32_t bytes, dn;
        std::memcpy(&id, e, 2);
        std::memcpy(&off, e + 8, 8);
        std::memcpy(&bytes, e + 16, 4);
        std::memcpy(&dn, e + 20, 4);
        const size_t layer = i / (size_t) p.n_expert;
        if (id >= p.n_expert || off + bytes > sec[4].len || dn >= bytes)
            throw std::runtime_error(".strata: bad expert entry " + std::to_string(i));
        Expert& x = p.experts[layer * (size_t) p.n_expert + id];
        x.body = base + sec[4].off + off;
        x.bytes = bytes;
        x.dn_off = dn;
        x.gu = e[2];
        x.dn = e[3];
        if (!expert_codes_ok(x.gu, x.dn)) throw std::runtime_error(".strata: expert formats " + std::to_string(x.gu) +
                                                                    "+" + std::to_string(x.dn) + " are not taken");
    }
    if (ordered) {
        p.load_order.resize(n);
        std::memcpy(p.load_order.data(), s3 + n * 32, n * 2);
    }
    return p;
}

}  // namespace strata::sq
