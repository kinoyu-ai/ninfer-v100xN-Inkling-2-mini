#pragma once

#include "artifact/materializer.h"
#include "core/tensor.h"
#include "targets/inkling2_mini/impl/config.h"
#include <ninfer/targets/inkling2_mini/frontend.h>

#include <array>
#include <cstdint>

namespace ninfer::targets::inkling2_mini::detail {

struct ControllerWeights {
    Tensor down;        // BF16 [4608, 64]
    Tensor gain_forget; // BF16 [64, 4608]
};

struct LayerWeights {
    bool is_local = true;
    bool is_core  = false;
    bool is_nvfp4 = true;

    Tensor attn_res_attn_query; // FP32 [2304] (local, ip>0) or [2304, 3] (full)
    Tensor attn_res_mlp_query;  // FP32 [2304]

    Tensor attn_norm;    // BF16 [2304] (local)
    Tensor q_input_norm; // BF16 [2304] (full)
    Tensor k_input_norm; // BF16 [2304] (full)
    Tensor v_input_norm; // BF16 [2304] (full)
    Tensor mlp_norm;     // BF16 [2304]

    Tensor q_gate_r_sink;   // BF16 [2304, 6552/tp]
    Tensor k_proj;          // BF16 [2304, 1024/tp]
    Tensor v_proj;          // BF16 [2304, 1024/tp]
    Tensor o_proj;          // BF16 [3072/tp, 2304]
    Tensor q_clip_scale;    // BF16 [128, 24/tp]
    Tensor rel_clip_scale;  // BF16 [16, 24/tp]
    Tensor rel_logits_proj; // BF16 [1024, 16]

    // SConv weights
    Tensor k_sconv;      // BF16 [4, 1024/tp] (static for local, static_w for full)
    Tensor v_sconv;      // BF16 [4, 1024/tp]
    Tensor k_sconv_proj; // BF16 [2304, 32/tp] (full only)
    Tensor v_sconv_proj; // BF16 [2304, 32/tp] (full only)
    Tensor attn_sconv;   // BF16 [4, 2304]
    Tensor mlp_sconv;    // BF16 [4, 2304]

    // Core cross-pass controllers (ip 6..17)
    ControllerWeights ctrl_attn;
    ControllerWeights ctrl_k;
    ControllerWeights ctrl_v;
    ControllerWeights ctrl_mlp;

    // Sparse MoE
    Tensor router_weight; // BF16 [2304, 128]
    Tensor router_bias;   // FP32 [128]

    // Expert weight raw device pointers (num_slabs = 2/tp)
    const void* w13_codes  = nullptr;
    const void* w13_scales = nullptr;
    const float* w13_amax  = nullptr; // NVFP4 only
    const float* w13_denom = nullptr; // NVFP4 only
    const void* w2_codes   = nullptr;
    const void* w2_scales  = nullptr;
    const float* w2_amax   = nullptr; // NVFP4 only
    const float* w2_denom  = nullptr; // NVFP4 only
};

struct MtpLayerWeights {
    Tensor dflare_norm;         // BF16 [2304]
    Tensor input_norm;          // BF16 [2304]
    Tensor post_attention_norm; // BF16 [2304]
    Tensor qkv;                 // BF16 [2304, 5120/tp]
    Tensor kv_ctx;              // BF16 [2304, 2048/tp]
    Tensor q_norm;              // BF16 [128]
    Tensor k_norm;              // BF16 [128]
    Tensor k_norm_ctx;          // BF16 [128]
    Tensor o_proj;              // BF16 [3072/tp, 2304]
    Tensor gate_up;             // BF16 [2304, 18432/tp]
    Tensor down_proj;           // BF16 [9216/tp, 2304]
};

struct MtpModelView {
    bool present = false;
    Tensor fuse_weight; // BF16 [4, 4]
    Tensor mask_embed;  // BF16 [2304]
    Tensor final_norm;  // BF16 [2304]
    std::array<MtpLayerWeights, kMtpNumLayers> layers{};

    Tensor markov_dflare_norm;       // BF16 [2304]
    Tensor markov_w1;                // BF16 [256, 201024]
    Tensor markov_rnn_proj_weight;   // BF16 [5120, 768]
    Tensor markov_rnn_proj_bias;     // BF16 [768]
    Tensor markov_hidden_out_weight; // BF16 [256, 2304]
    Tensor markov_hidden_out_bias;   // BF16 [2304]
};

struct RankModelView {
    int rank                = 0;
    int tp                  = 1;
    std::int32_t vocab_off  = 0;
    std::int32_t vocab_rows = kPaddedVocabSize;
    std::int32_t valid_rows = kUnpaddedVocab;
    std::int32_t num_q_heads  = kNumQHeads;
    std::int32_t num_kv_heads = kNumKVHeads;
    std::int32_t moe_slabs    = 2;

    Tensor embed;                // BF16 [2304, vocab_rows]
    Tensor embed_norm;           // BF16 [2304]
    Tensor final_norm;           // BF16 [2304]
    Tensor output_head;          // BF16 [2304, vocab_rows]
    Tensor attn_res_out_query;   // FP32 [2304]
    Tensor cross_pass_queries;   // FP32 [2304, 24]
    Tensor cross_pass_kv_queries;// FP32 [2304, 24]

    // Pre-rearranged [2304, 4] FP32 queries per core layer (ip 6..17 -> idx 0..11)
    // constructed on GPU from cross_pass_queries & cross_pass_kv_queries at Program init!

    std::array<LayerWeights, kNumPhysLayers> layers{};
    MtpModelView mtp;
};

struct LoadedModelData {
    artifact::MaterializedArtifact materialized;
    FrontendResources frontend;
    int tp                  = 1;
    bool mtp_enabled        = false;
    std::uint32_t draft_tokens = 0;
    std::array<RankModelView, 2> ranks{};

    [[nodiscard]] const RankModelView& view(int rank) const {
        return ranks[static_cast<std::size_t>(rank)];
    }
};

} // namespace ninfer::targets::inkling2_mini::detail
