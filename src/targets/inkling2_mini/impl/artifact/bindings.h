#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "targets/inkling2_mini/impl/artifact/model_view.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::inkling2_mini::detail {

struct LayerHandles {
    artifact::ObjectHandle attn_res_attn_query;
    artifact::ObjectHandle attn_res_mlp_query;

    artifact::ObjectHandle attn_norm;
    artifact::ObjectHandle q_input_norm;
    artifact::ObjectHandle k_input_norm;
    artifact::ObjectHandle v_input_norm;
    artifact::ObjectHandle mlp_norm;

    artifact::ObjectHandle q_gate_r_sink;
    artifact::ObjectHandle k_proj;
    artifact::ObjectHandle v_proj;
    artifact::ObjectHandle o_proj;
    artifact::ObjectHandle q_clip_scale;
    artifact::ObjectHandle rel_clip_scale;
    artifact::ObjectHandle rel_logits_proj;

    artifact::ObjectHandle k_sconv;
    artifact::ObjectHandle v_sconv;
    artifact::ObjectHandle k_sconv_proj;
    artifact::ObjectHandle v_sconv_proj;
    artifact::ObjectHandle attn_sconv;
    artifact::ObjectHandle mlp_sconv;

    struct CtrlHandles {
        artifact::ObjectHandle down;
        artifact::ObjectHandle gain_forget;
    };
    CtrlHandles ctrl_attn;
    CtrlHandles ctrl_k;
    CtrlHandles ctrl_v;
    CtrlHandles ctrl_mlp;

    artifact::ObjectHandle router_weight;
    artifact::ObjectHandle router_bias;
    artifact::ObjectHandle w13_codes;
    artifact::ObjectHandle w13_scales;
    artifact::ObjectHandle w13_amax;
    artifact::ObjectHandle w13_denom;
    artifact::ObjectHandle w2_codes;
    artifact::ObjectHandle w2_scales;
    artifact::ObjectHandle w2_amax;
    artifact::ObjectHandle w2_denom;
};

struct MtpHandles {
    bool bound = false;
    artifact::ObjectHandle fuse_weight;
    artifact::ObjectHandle mask_embed;
    artifact::ObjectHandle final_norm;

    struct Layer {
        artifact::ObjectHandle dflare_norm;
        artifact::ObjectHandle input_norm;
        artifact::ObjectHandle post_attention_norm;
        artifact::ObjectHandle qkv;
        artifact::ObjectHandle kv_ctx;
        artifact::ObjectHandle q_norm;
        artifact::ObjectHandle k_norm;
        artifact::ObjectHandle k_norm_ctx;
        artifact::ObjectHandle o_proj;
        artifact::ObjectHandle gate_up;
        artifact::ObjectHandle down_proj;
    };
    std::array<Layer, kMtpNumLayers> layers{};

    artifact::ObjectHandle markov_dflare_norm;
    artifact::ObjectHandle markov_w1;
    artifact::ObjectHandle markov_rnn_proj_weight;
    artifact::ObjectHandle markov_rnn_proj_bias;
    artifact::ObjectHandle markov_hidden_out_weight;
    artifact::ObjectHandle markov_hidden_out_bias;
};

struct ArtifactBindings {
    artifact::ObjectHandle res_tokenizer_json;
    artifact::ObjectHandle res_tokenizer_config_json;
    artifact::ObjectHandle res_chat_template_jinja;
    artifact::ObjectHandle res_generation_config_json;

    artifact::ObjectHandle embed;
    artifact::ObjectHandle embed_norm;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle attn_res_out_query;
    artifact::ObjectHandle cross_pass_queries;
    artifact::ObjectHandle cross_pass_kv_queries;

    std::array<LayerHandles, kNumPhysLayers> layers{};
    MtpHandles mtp;
};

struct ArtifactLoadPlan {
    ArtifactBindings bindings;
    artifact::MaterializationPlan materialization;
    int tp                  = 1;
    bool mtp_enabled        = false;
    std::uint32_t draft_tokens = 0;
};

[[nodiscard]] ArtifactLoadPlan bind_artifact(artifact::Binder& binder, const EngineOptions& options);

[[nodiscard]] LoadedModelData materialize_model(const ArtifactBindings& bindings,
                                                artifact::MaterializedArtifact&& materialized,
                                                int tp, bool mtp_enabled,
                                                std::uint32_t draft_tokens);

} // namespace ninfer::targets::inkling2_mini::detail
