#include "targets/inkling2_mini/impl/artifact/bindings.h"

#include "artifact/typed_binding.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::inkling2_mini::detail {
namespace {

using artifact::NumericFormat;
using artifact::ShardAxis;
using artifact::ShardPlacement;
using artifact::SliceRange;

ShardPlacement split_rows_half(std::uint64_t rows) {
    ShardPlacement p;
    p.axis             = ShardAxis::Rows;
    const std::uint64_t half = rows / 2;
    p.device_ranges[0] = {{0, half}};
    p.device_ranges[1] = {{half, rows - half}};
    return p;
}

ShardPlacement split_cols_half(std::uint64_t cols) {
    ShardPlacement p;
    p.axis             = ShardAxis::Columns;
    const std::uint64_t half = cols / 2;
    p.device_ranges[0] = {{0, half}};
    p.device_ranges[1] = {{half, cols - half}};
    return p;
}

// Resolves TP2 shard geometry by exact artifact tensor name.
ShardPlacement resolve_tp2_shard(std::string_view name) {
    if (name == "text/embed" || name == "text/output_head") {
        return split_rows_half(kPaddedVocabSize); // 100512 / rank
    }
    if (name.ends_with("/attn/q_gate_r_sink")) {
        // Packed [Q(3072), gate(3072), R(384), sink(24)] -> split each block in half!
        ShardPlacement p;
        p.axis             = ShardAxis::Rows;
        p.device_ranges[0] = {{0, 1536}, {3072, 1536}, {6144, 192}, {6528, 12}};
        p.device_ranges[1] = {{1536, 1536}, {4608, 1536}, {6336, 192}, {6540, 12}};
        return p;
    }
    if (name.ends_with("/attn/k") || name.ends_with("/attn/v") ||
        name.ends_with("/attn/k_sconv") || name.ends_with("/attn/v_sconv") ||
        name.ends_with("/attn/k_sconv_static") || name.ends_with("/attn/v_sconv_static")) {
        return split_rows_half(1024);
    }
    if (name.ends_with("/attn/k_sconv_proj") || name.ends_with("/attn/v_sconv_proj")) {
        return split_rows_half(32); // 8 KV heads * 4 lags -> 4 KV heads * 4 lags = 16 / rank
    }
    if (name.ends_with("/attn/q_clip_scale") || name.ends_with("/attn/rel_clip_scale")) {
        return split_rows_half(24); // 12 Q heads / rank
    }
    if (name.ends_with("/attn/o")) { return split_cols_half(3072); }

    // Expert banks: axis-0 is TP2 slab index [2, 128, ...] -> single contiguous slice per GPU!
    if (name.ends_with("/mlp/experts/w13_codes") || name.ends_with("/mlp/experts/w13_scales") ||
        name.ends_with("/mlp/experts/w2_codes") || name.ends_with("/mlp/experts/w2_scales")) {
        return split_rows_half(2);
    }

    // DSpark MTP layers
    if (name.starts_with("mtp/")) {
        if (name.ends_with("/attention/qkv")) {
            // [Q(3072), K(1024), V(1024)] = 5120
            ShardPlacement p;
            p.axis             = ShardAxis::Rows;
            p.device_ranges[0] = {{0, 1536}, {3072, 512}, {4096, 512}};
            p.device_ranges[1] = {{1536, 1536}, {3584, 512}, {4608, 512}};
            return p;
        }
        if (name.ends_with("/attention/kv_ctx")) {
            // [K(1024), V(1024)] = 2048
            ShardPlacement p;
            p.axis             = ShardAxis::Rows;
            p.device_ranges[0] = {{0, 512}, {1024, 512}};
            p.device_ranges[1] = {{512, 512}, {1536, 512}};
            return p;
        }
        if (name.ends_with("/attention/o")) { return split_cols_half(3072); }
        if (name.ends_with("/mlp/gate_up")) {
            // [gate(9216), up(9216)] = 18432
            ShardPlacement p;
            p.axis             = ShardAxis::Rows;
            p.device_ranges[0] = {{0, 4608}, {9216, 4608}};
            p.device_ranges[1] = {{4608, 4608}, {13824, 4608}};
            return p;
        }
        if (name.ends_with("/mlp/down")) { return split_cols_half(9216); }
    }

    return ShardPlacement{}; // Replicated
}

std::string take_host_string(artifact::MaterializedArtifact& m, artifact::ObjectHandle h) {
    auto bytes = m.take_host_bytes(h);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, const EngineOptions& options) {
    if (options.tp != 1 && options.tp != 2) {
        throw std::invalid_argument("inkling2-mini supports --tp 1 or --tp 2");
    }
    if (options.pp != 1) {
        throw std::invalid_argument("inkling2-mini does not support pipeline parallelism (--pp 2)");
    }
    if (options.enable_vision) {
        throw std::invalid_argument("inkling2-mini does not yet support --vision");
    }
    if (options.storage_device >= 0) {
        throw std::invalid_argument("inkling2-mini does not support --storage-device");
    }
    if (options.rope_mode != RopeMode::Native) {
        throw std::invalid_argument("inkling2-mini uses relative attention bias and does not use --rope yarn");
    }

    const bool want_mtp = (options.speculative.backend != SpeculativeBackend::None);
    std::uint32_t draft_tokens = 0;
    if (want_mtp) {
        draft_tokens = options.speculative.draft_tokens;
        if (draft_tokens == 0 || draft_tokens > 7) {
            throw std::invalid_argument(
                "inkling2-mini DSpark speculative decoding supports --draft-tokens in [1, 7]");
        }
        if (!binder.has_tensor("mtp/fuse_weight")) {
            throw std::invalid_argument(
                "speculative decoding requested, but this inkling2-mini .ninfer artifact does not "
                "include the DSpark MTP package");
        }
    }

    if (options.tp == 2) { binder.set_shard_resolver(resolve_tp2_shard); }

    ArtifactBindings b;
    b.res_tokenizer_json         = artifact::bind_raw_resource(binder, "frontend/tokenizer.json");
    b.res_tokenizer_config_json  = artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json");
    b.res_chat_template_jinja    = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja");
    b.res_generation_config_json = artifact::bind_raw_resource(binder, "frontend/generation_config.json");

    b.embed                 = artifact::bind_device_tensor(binder, "text/embed", NumericFormat::BF16, {kPaddedVocabSize, kHiddenSize});
    b.embed_norm            = artifact::bind_device_tensor(binder, "text/embed_norm", NumericFormat::BF16, {kHiddenSize});
    b.final_norm            = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {kHiddenSize});
    b.output_head           = artifact::bind_device_tensor(binder, "text/output_head", NumericFormat::BF16, {kPaddedVocabSize, kHiddenSize});
    b.attn_res_out_query    = artifact::bind_device_tensor(binder, "text/attn_res/output_query", NumericFormat::FP32, {kHiddenSize});
    b.cross_pass_queries    = artifact::bind_device_tensor(binder, "text/loop/cross_pass_queries", NumericFormat::FP32, {24, kHiddenSize});
    b.cross_pass_kv_queries = artifact::bind_device_tensor(binder, "text/loop/cross_pass_kv_queries", NumericFormat::FP32, {24, kHiddenSize});

    for (std::int32_t ip = 0; ip < kNumPhysLayers; ++ip) {
        const std::string lp = "text/layers/" + std::to_string(ip) + "/";
        auto& lh             = b.layers[static_cast<std::size_t>(ip)];
        const bool is_local  = is_local_phys_layer(ip);

        if (ip > 0) {
            if (is_local) {
                lh.attn_res_attn_query = artifact::bind_device_tensor(
                    binder, lp + "attn_res/attn_query", NumericFormat::FP32, {kHiddenSize});
            } else {
                lh.attn_res_attn_query = artifact::bind_device_tensor(
                    binder, lp + "attn_res/attn_query", NumericFormat::FP32, {3, kHiddenSize});
            }
        }
        lh.attn_res_mlp_query = artifact::bind_device_tensor(binder, lp + "attn_res/mlp_query",
                                                             NumericFormat::FP32, {kHiddenSize});

        if (is_local) {
            lh.attn_norm = artifact::bind_device_tensor(binder, lp + "attn_norm",
                                                        NumericFormat::BF16, {kHiddenSize});
        } else {
            lh.q_input_norm = artifact::bind_device_tensor(binder, lp + "q_input_norm",
                                                           NumericFormat::BF16, {kHiddenSize});
            lh.k_input_norm = artifact::bind_device_tensor(binder, lp + "k_input_norm",
                                                           NumericFormat::BF16, {kHiddenSize});
            lh.v_input_norm = artifact::bind_device_tensor(binder, lp + "v_input_norm",
                                                           NumericFormat::BF16, {kHiddenSize});
        }
        lh.mlp_norm = artifact::bind_device_tensor(binder, lp + "mlp_norm", NumericFormat::BF16,
                                                   {kHiddenSize});

        lh.q_gate_r_sink   = artifact::bind_device_tensor(binder, lp + "attn/q_gate_r_sink", NumericFormat::BF16, {6552, kHiddenSize});
        lh.k_proj          = artifact::bind_device_tensor(binder, lp + "attn/k", NumericFormat::BF16, {1024, kHiddenSize});
        lh.v_proj          = artifact::bind_device_tensor(binder, lp + "attn/v", NumericFormat::BF16, {1024, kHiddenSize});
        lh.o_proj          = artifact::bind_device_tensor(binder, lp + "attn/o", NumericFormat::BF16, {kHiddenSize, 3072});
        lh.q_clip_scale    = artifact::bind_device_tensor(binder, lp + "attn/q_clip_scale", NumericFormat::BF16, {kNumQHeads, kHeadDim});
        lh.rel_clip_scale  = artifact::bind_device_tensor(binder, lp + "attn/rel_clip_scale", NumericFormat::BF16, {kNumQHeads, kDRel});
        lh.rel_logits_proj = artifact::bind_device_tensor(binder, lp + "attn/rel_logits_proj", NumericFormat::BF16, {kDRel, kRelExtent});

        if (is_local) {
            lh.k_sconv = artifact::bind_device_tensor(binder, lp + "attn/k_sconv", NumericFormat::BF16, {1024, 4});
            lh.v_sconv = artifact::bind_device_tensor(binder, lp + "attn/v_sconv", NumericFormat::BF16, {1024, 4});
        } else {
            lh.k_sconv      = artifact::bind_device_tensor(binder, lp + "attn/k_sconv_static", NumericFormat::BF16, {1024, 4});
            lh.v_sconv      = artifact::bind_device_tensor(binder, lp + "attn/v_sconv_static", NumericFormat::BF16, {1024, 4});
            lh.k_sconv_proj = artifact::bind_device_tensor(binder, lp + "attn/k_sconv_proj", NumericFormat::BF16, {32, kHiddenSize});
            lh.v_sconv_proj = artifact::bind_device_tensor(binder, lp + "attn/v_sconv_proj", NumericFormat::BF16, {32, kHiddenSize});
        }
        lh.attn_sconv = artifact::bind_device_tensor(binder, lp + "attn_sconv", NumericFormat::BF16, {kHiddenSize, 4});
        lh.mlp_sconv  = artifact::bind_device_tensor(binder, lp + "mlp_sconv", NumericFormat::BF16, {kHiddenSize, 4});

        if (is_core_phys_layer(ip)) {
            auto bind_ctrl = [&](const char* name, LayerHandles::CtrlHandles& ch) {
                ch.down        = artifact::bind_device_tensor(binder, lp + "controllers/" + name + "/down",
                                                       NumericFormat::BF16, {kGateRank, 2 * kHiddenSize});
                ch.gain_forget = artifact::bind_device_tensor(binder, lp + "controllers/" + name + "/gain_forget",
                                                              NumericFormat::BF16, {2 * kHiddenSize, kGateRank});
            };
            bind_ctrl("attn", lh.ctrl_attn);
            bind_ctrl("k", lh.ctrl_k);
            bind_ctrl("v", lh.ctrl_v);
            bind_ctrl("mlp", lh.ctrl_mlp);
        }

        lh.router_weight = artifact::bind_device_tensor(binder, lp + "mlp/router_weight", NumericFormat::BF16, {kNumExperts, kHiddenSize});
        lh.router_bias   = artifact::bind_device_tensor(binder, lp + "mlp/router_bias", NumericFormat::FP32, {kNumExperts});

        if (is_nvfp4_phys_layer(ip)) {
            lh.w13_codes  = artifact::bind_device_tensor(binder, lp + "mlp/experts/w13_codes", NumericFormat::U8, {2, kNumExperts, 1152, 1152});
            lh.w13_scales = artifact::bind_device_tensor(binder, lp + "mlp/experts/w13_scales", NumericFormat::FP8_E4M3FN, {2, kNumExperts, 1152, 144});
            lh.w13_amax   = artifact::bind_device_tensor(binder, lp + "mlp/experts/w13_amax", NumericFormat::FP32, {kNumExperts});
            lh.w13_denom  = artifact::bind_device_tensor(binder, lp + "mlp/experts/w13_denom", NumericFormat::FP32, {1});
            lh.w2_codes   = artifact::bind_device_tensor(binder, lp + "mlp/experts/w2_codes", NumericFormat::U8, {2, kNumExperts, kHiddenSize, 288});
            lh.w2_scales  = artifact::bind_device_tensor(binder, lp + "mlp/experts/w2_scales", NumericFormat::FP8_E4M3FN, {2, kNumExperts, kHiddenSize, 36});
            lh.w2_amax    = artifact::bind_device_tensor(binder, lp + "mlp/experts/w2_amax", NumericFormat::FP32, {kNumExperts});
            lh.w2_denom   = artifact::bind_device_tensor(binder, lp + "mlp/experts/w2_denom", NumericFormat::FP32, {1});
        } else {
            lh.w13_codes  = artifact::bind_device_tensor(binder, lp + "mlp/experts/w13_codes", NumericFormat::FP8_E4M3FN, {2, kNumExperts, 1152, kHiddenSize});
            lh.w13_scales = artifact::bind_device_tensor(binder, lp + "mlp/experts/w13_scales", NumericFormat::U8, {2, kNumExperts, 1152, 72});
            lh.w2_codes   = artifact::bind_device_tensor(binder, lp + "mlp/experts/w2_codes", NumericFormat::FP8_E4M3FN, {2, kNumExperts, kHiddenSize, 576});
            lh.w2_scales  = artifact::bind_device_tensor(binder, lp + "mlp/experts/w2_scales", NumericFormat::U8, {2, kNumExperts, kHiddenSize, 18});
        }
    }

    if (want_mtp) {
        b.mtp.bound       = true;
        b.mtp.fuse_weight = artifact::bind_device_tensor(binder, "mtp/fuse_weight", NumericFormat::BF16, {4, 4});
        b.mtp.mask_embed  = artifact::bind_device_tensor(binder, "mtp/mask_embed", NumericFormat::BF16, {kHiddenSize});
        b.mtp.final_norm  = artifact::bind_device_tensor(binder, "mtp/norm", NumericFormat::BF16, {kHiddenSize});
        for (std::int32_t li = 0; li < kMtpNumLayers; ++li) {
            const std::string mp = "mtp/layers/" + std::to_string(li) + "/";
            auto& ml             = b.mtp.layers[static_cast<std::size_t>(li)];
            ml.dflare_norm         = artifact::bind_device_tensor(binder, mp + "dflare_norm", NumericFormat::BF16, {kHiddenSize});
            ml.input_norm          = artifact::bind_device_tensor(binder, mp + "input_norm", NumericFormat::BF16, {kHiddenSize});
            ml.post_attention_norm = artifact::bind_device_tensor(binder, mp + "post_attention_norm", NumericFormat::BF16, {kHiddenSize});
            ml.qkv                 = artifact::bind_device_tensor(binder, mp + "attention/qkv", NumericFormat::BF16, {5120, kHiddenSize});
            ml.kv_ctx              = artifact::bind_device_tensor(binder, mp + "attention/kv_ctx", NumericFormat::BF16, {2048, kHiddenSize});
            ml.q_norm              = artifact::bind_device_tensor(binder, mp + "attention/q_norm", NumericFormat::BF16, {kHeadDim});
            ml.k_norm              = artifact::bind_device_tensor(binder, mp + "attention/k_norm", NumericFormat::BF16, {kHeadDim});
            ml.k_norm_ctx          = artifact::bind_device_tensor(binder, mp + "attention/k_norm_ctx", NumericFormat::BF16, {kHeadDim});
            ml.o_proj              = artifact::bind_device_tensor(binder, mp + "attention/o", NumericFormat::BF16, {kHiddenSize, 3072});
            ml.gate_up             = artifact::bind_device_tensor(binder, mp + "mlp/gate_up", NumericFormat::BF16, {2 * kMtpInterSize, kHiddenSize});
            ml.down_proj           = artifact::bind_device_tensor(binder, mp + "mlp/down", NumericFormat::BF16, {kHiddenSize, kMtpInterSize});
        }
        b.mtp.markov_dflare_norm       = artifact::bind_device_tensor(binder, "mtp/markov/dflare_norm", NumericFormat::BF16, {kHiddenSize});
        b.mtp.markov_w1                = artifact::bind_device_tensor(binder, "mtp/markov/w1", NumericFormat::BF16, {kPaddedVocabSize, kMtpMarkovRank});
        b.mtp.markov_rnn_proj_weight   = artifact::bind_device_tensor(binder, "mtp/markov/rnn_proj_weight", NumericFormat::BF16, {3 * kMtpMarkovRank, 2 * kMtpMarkovRank + 2 * kHiddenSize});
        b.mtp.markov_rnn_proj_bias     = artifact::bind_device_tensor(binder, "mtp/markov/rnn_proj_bias", NumericFormat::BF16, {3 * kMtpMarkovRank});
        b.mtp.markov_hidden_out_weight = artifact::bind_device_tensor(binder, "mtp/markov/hidden_out_weight", NumericFormat::BF16, {kHiddenSize, kMtpMarkovRank});
        b.mtp.markov_hidden_out_bias   = artifact::bind_device_tensor(binder, "mtp/markov/hidden_out_bias", NumericFormat::BF16, {kHiddenSize});
    } else {
        binder.validate_unconsumed_matching("mtp/");
    }

    return ArtifactLoadPlan{
        .bindings        = b,
        .materialization = binder.finish(),
        .tp              = options.tp,
        .mtp_enabled     = want_mtp,
        .draft_tokens    = draft_tokens,
    };
}

LoadedModelData materialize_model(const ArtifactBindings& b,
                                  artifact::MaterializedArtifact&& materialized, int tp,
                                  bool mtp_enabled, std::uint32_t draft_tokens) {
    LoadedModelData out;
    out.frontend.tokenizer_json         = take_host_string(materialized, b.res_tokenizer_json);
    out.frontend.tokenizer_config_json  = take_host_string(materialized, b.res_tokenizer_config_json);
    out.frontend.chat_template_jinja    = take_host_string(materialized, b.res_chat_template_jinja);
    out.frontend.generation_config_json = take_host_string(materialized, b.res_generation_config_json);
    out.tp                              = tp;
    out.mtp_enabled                     = mtp_enabled;
    out.draft_tokens                    = draft_tokens;

    const std::int32_t vocab_rows   = kPaddedVocabSize / tp;
    const std::int32_t q_heads_loc  = kNumQHeads / tp;
    const std::int32_t kv_heads_loc = kNumKVHeads / tp;
    const std::int32_t slabs_loc    = 2 / tp;

    for (int rank = 0; rank < tp; ++rank) {
        auto& rv        = out.ranks[static_cast<std::size_t>(rank)];
        rv.rank         = rank;
        rv.tp           = tp;
        rv.vocab_off    = rank * vocab_rows;
        rv.vocab_rows   = vocab_rows;
        rv.valid_rows   = std::max<std::int32_t>(
            0, std::min<std::int32_t>(vocab_rows, kUnpaddedVocab - rv.vocab_off));
        rv.num_q_heads  = q_heads_loc;
        rv.num_kv_heads = kv_heads_loc;
        rv.moe_slabs    = slabs_loc;

        auto mt = [&](artifact::ObjectHandle h, NumericFormat fmt,
                      std::initializer_list<std::int32_t> shape) {
            return artifact::materialized_tensor(materialized, h, fmt, shape, rank);
        };

        rv.embed                 = mt(b.embed, NumericFormat::BF16, {kHiddenSize, vocab_rows});
        rv.embed_norm            = mt(b.embed_norm, NumericFormat::BF16, {kHiddenSize});
        rv.final_norm            = mt(b.final_norm, NumericFormat::BF16, {kHiddenSize});
        rv.output_head           = mt(b.output_head, NumericFormat::BF16, {kHiddenSize, vocab_rows});
        rv.attn_res_out_query    = mt(b.attn_res_out_query, NumericFormat::FP32, {kHiddenSize});
        rv.cross_pass_queries    = mt(b.cross_pass_queries, NumericFormat::FP32, {kHiddenSize, 24});
        rv.cross_pass_kv_queries = mt(b.cross_pass_kv_queries, NumericFormat::FP32, {kHiddenSize, 24});

        for (std::int32_t ip = 0; ip < kNumPhysLayers; ++ip) {
            const auto& lh = b.layers[static_cast<std::size_t>(ip)];
            auto& lw       = rv.layers[static_cast<std::size_t>(ip)];
            lw.is_local    = is_local_phys_layer(ip);
            lw.is_core     = is_core_phys_layer(ip);
            lw.is_nvfp4    = is_nvfp4_phys_layer(ip);

            if (ip > 0) {
                if (lw.is_local) {
                    lw.attn_res_attn_query = mt(lh.attn_res_attn_query, NumericFormat::FP32, {kHiddenSize});
                } else {
                    lw.attn_res_attn_query = mt(lh.attn_res_attn_query, NumericFormat::FP32, {kHiddenSize, 3});
                }
            }
            lw.attn_res_mlp_query = mt(lh.attn_res_mlp_query, NumericFormat::FP32, {kHiddenSize});

            if (lw.is_local) {
                lw.attn_norm = mt(lh.attn_norm, NumericFormat::BF16, {kHiddenSize});
            } else {
                lw.q_input_norm = mt(lh.q_input_norm, NumericFormat::BF16, {kHiddenSize});
                lw.k_input_norm = mt(lh.k_input_norm, NumericFormat::BF16, {kHiddenSize});
                lw.v_input_norm = mt(lh.v_input_norm, NumericFormat::BF16, {kHiddenSize});
            }
            lw.mlp_norm = mt(lh.mlp_norm, NumericFormat::BF16, {kHiddenSize});

            lw.q_gate_r_sink   = mt(lh.q_gate_r_sink, NumericFormat::BF16, {kHiddenSize, 6552 / tp});
            lw.k_proj          = mt(lh.k_proj, NumericFormat::BF16, {kHiddenSize, 1024 / tp});
            lw.v_proj          = mt(lh.v_proj, NumericFormat::BF16, {kHiddenSize, 1024 / tp});
            lw.o_proj          = mt(lh.o_proj, NumericFormat::BF16, {3072 / tp, kHiddenSize});
            lw.q_clip_scale    = mt(lh.q_clip_scale, NumericFormat::BF16, {kHeadDim, q_heads_loc});
            lw.rel_clip_scale  = mt(lh.rel_clip_scale, NumericFormat::BF16, {kDRel, q_heads_loc});
            lw.rel_logits_proj = mt(lh.rel_logits_proj, NumericFormat::BF16, {kRelExtent, kDRel});

            lw.k_sconv = mt(lh.k_sconv, NumericFormat::BF16, {4, 1024 / tp});
            lw.v_sconv = mt(lh.v_sconv, NumericFormat::BF16, {4, 1024 / tp});
            if (!lw.is_local) {
                lw.k_sconv_proj = mt(lh.k_sconv_proj, NumericFormat::BF16, {kHiddenSize, 32 / tp});
                lw.v_sconv_proj = mt(lh.v_sconv_proj, NumericFormat::BF16, {kHiddenSize, 32 / tp});
            }
            lw.attn_sconv = mt(lh.attn_sconv, NumericFormat::BF16, {4, kHiddenSize});
            lw.mlp_sconv  = mt(lh.mlp_sconv, NumericFormat::BF16, {4, kHiddenSize});

            if (lw.is_core) {
                auto mat_ctrl = [&](const LayerHandles::CtrlHandles& ch, ControllerWeights& cw) {
                    cw.down        = mt(ch.down, NumericFormat::BF16, {2 * kHiddenSize, kGateRank});
                    cw.gain_forget = mt(ch.gain_forget, NumericFormat::BF16, {kGateRank, 2 * kHiddenSize});
                };
                mat_ctrl(lh.ctrl_attn, lw.ctrl_attn);
                mat_ctrl(lh.ctrl_k, lw.ctrl_k);
                mat_ctrl(lh.ctrl_v, lw.ctrl_v);
                mat_ctrl(lh.ctrl_mlp, lw.ctrl_mlp);
            }

            lw.router_weight = mt(lh.router_weight, NumericFormat::BF16, {kHiddenSize, kNumExperts});
            lw.router_bias   = mt(lh.router_bias, NumericFormat::FP32, {kNumExperts});

            if (lw.is_nvfp4) {
                artifact::require_placement_bytes(
                    materialized, lh.w13_codes, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * 1152 * 1152);
                artifact::require_placement_bytes(
                    materialized, lh.w13_scales, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * 1152 * 144);
                artifact::require_placement_bytes(
                    materialized, lh.w2_codes, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * kHiddenSize * 288);
                artifact::require_placement_bytes(
                    materialized, lh.w2_scales, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * kHiddenSize * 36);
                lw.w13_codes  = materialized.device_data(lh.w13_codes, rank);
                lw.w13_scales = materialized.device_data(lh.w13_scales, rank);
                lw.w13_amax   = static_cast<const float*>(materialized.device_data(lh.w13_amax, rank));
                lw.w13_denom  = static_cast<const float*>(materialized.device_data(lh.w13_denom, rank));
                lw.w2_codes   = materialized.device_data(lh.w2_codes, rank);
                lw.w2_scales  = materialized.device_data(lh.w2_scales, rank);
                lw.w2_amax    = static_cast<const float*>(materialized.device_data(lh.w2_amax, rank));
                lw.w2_denom   = static_cast<const float*>(materialized.device_data(lh.w2_denom, rank));
            } else {
                artifact::require_placement_bytes(
                    materialized, lh.w13_codes, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * 1152 * kHiddenSize);
                artifact::require_placement_bytes(
                    materialized, lh.w13_scales, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * 1152 * 72);
                artifact::require_placement_bytes(
                    materialized, lh.w2_codes, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * kHiddenSize * 576);
                artifact::require_placement_bytes(
                    materialized, lh.w2_scales, rank,
                    static_cast<std::uint64_t>(slabs_loc) * kNumExperts * kHiddenSize * 18);
                lw.w13_codes  = materialized.device_data(lh.w13_codes, rank);
                lw.w13_scales = materialized.device_data(lh.w13_scales, rank);
                lw.w2_codes   = materialized.device_data(lh.w2_codes, rank);
                lw.w2_scales  = materialized.device_data(lh.w2_scales, rank);
            }
        }

        if (mtp_enabled && b.mtp.bound) {
            rv.mtp.present     = true;
            rv.mtp.fuse_weight = mt(b.mtp.fuse_weight, NumericFormat::BF16, {4, 4});
            rv.mtp.mask_embed  = mt(b.mtp.mask_embed, NumericFormat::BF16, {kHiddenSize});
            rv.mtp.final_norm  = mt(b.mtp.final_norm, NumericFormat::BF16, {kHiddenSize});
            for (std::int32_t li = 0; li < kMtpNumLayers; ++li) {
                const auto& ml = b.mtp.layers[static_cast<std::size_t>(li)];
                auto& mw       = rv.mtp.layers[static_cast<std::size_t>(li)];
                mw.dflare_norm         = mt(ml.dflare_norm, NumericFormat::BF16, {kHiddenSize});
                mw.input_norm          = mt(ml.input_norm, NumericFormat::BF16, {kHiddenSize});
                mw.post_attention_norm = mt(ml.post_attention_norm, NumericFormat::BF16, {kHiddenSize});
                mw.qkv                 = mt(ml.qkv, NumericFormat::BF16, {kHiddenSize, 5120 / tp});
                mw.kv_ctx              = mt(ml.kv_ctx, NumericFormat::BF16, {kHiddenSize, 2048 / tp});
                mw.q_norm              = mt(ml.q_norm, NumericFormat::BF16, {kHeadDim});
                mw.k_norm              = mt(ml.k_norm, NumericFormat::BF16, {kHeadDim});
                mw.k_norm_ctx          = mt(ml.k_norm_ctx, NumericFormat::BF16, {kHeadDim});
                mw.o_proj              = mt(ml.o_proj, NumericFormat::BF16, {3072 / tp, kHiddenSize});
                mw.gate_up             = mt(ml.gate_up, NumericFormat::BF16, {kHiddenSize, (2 * kMtpInterSize) / tp});
                mw.down_proj           = mt(ml.down_proj, NumericFormat::BF16, {kMtpInterSize / tp, kHiddenSize});
            }
            rv.mtp.markov_dflare_norm       = mt(b.mtp.markov_dflare_norm, NumericFormat::BF16, {kHiddenSize});
            rv.mtp.markov_w1                = mt(b.mtp.markov_w1, NumericFormat::BF16, {kMtpMarkovRank, kPaddedVocabSize});
            rv.mtp.markov_rnn_proj_weight   = mt(b.mtp.markov_rnn_proj_weight, NumericFormat::BF16, {2 * kMtpMarkovRank + 2 * kHiddenSize, 3 * kMtpMarkovRank});
            rv.mtp.markov_rnn_proj_bias     = mt(b.mtp.markov_rnn_proj_bias, NumericFormat::BF16, {3 * kMtpMarkovRank});
            rv.mtp.markov_hidden_out_weight = mt(b.mtp.markov_hidden_out_weight, NumericFormat::BF16, {kMtpMarkovRank, kHiddenSize});
            rv.mtp.markov_hidden_out_bias   = mt(b.mtp.markov_hidden_out_bias, NumericFormat::BF16, {kHiddenSize});
        }
    }

    out.materialized = std::move(materialized);
    return out;
}

} // namespace ninfer::targets::inkling2_mini::detail
