#include <ninfer/targets/inkling2_mini/runtime.h>

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/inkling2.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/sampling.h"
#include "targets/inkling2_mini/impl/artifact/model_view.h"
#include "targets/inkling2_mini/impl/config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::inkling2_mini {
namespace {

using Clock = std::chrono::steady_clock;
using namespace detail;

constexpr std::size_t kGraphAllowanceBytes = 64ULL << 20;

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void ensure_kv_tokens(PagedKVAllocation& alloc, std::uint32_t tokens, cudaStream_t stream) {
    const std::uint32_t needed = pages_for_tokens(tokens);
    if (needed > alloc.mapped_page_count()) { alloc.materialize_pages(needed, stream); }
}

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

struct SequenceLayoutSpec {
    int tp                         = 1;
    std::uint32_t max_context      = 2048;
    std::uint32_t max_concurrency  = 1;
    std::uint32_t prefill_chunk    = 1024;
    std::uint32_t main_page_groups = 32;
    bool mtp_enabled               = false;
    std::uint32_t draft_tokens     = 0;
    bool use_cuda_graph            = true;
    SpeculativeBackend spec_backend = SpeculativeBackend::None;
};

struct SequenceRegions {
    PagedKVPoolLayout main_kv;
    std::array<TensorRegion, kNumLocalLogicalLayers> swa_k{};
    std::array<TensorRegion, kNumLocalLogicalLayers> swa_v{};
    TensorRegion sconv_state;
    TensorRegion sconv_verify;
    TensorRegion cross_pass_4q;
    TensorRegion token_counts;

    // Optional DSpark MTP persistent regions
    PagedKVPoolLayout mtp_global_kv;
    std::array<TensorRegion, 2> mtp_swa_k{};
    std::array<TensorRegion, 2> mtp_swa_v{};
    TensorRegion mtp_fold_w;
    TensorRegion mtp_fold_b;
    TensorRegion mtp_f_rnn;

    // Fixed-address round metadata buffers (CUDA Graph capturable)
    TensorRegion round_token_ids;
    TensorRegion round_positions;
    TensorRegion round_tok_lanes;
    TensorRegion round_batch_lanes;
    TensorRegion round_ctx_lens;
    TensorRegion round_acc_counts;
    TensorRegion round_anchor_ids;
    TensorRegion round_draft_out; // I32 [max_concurrency, 7]
    TensorRegion round_winner_ids;
    TensorRegion round_local_max_val;
    TensorRegion round_local_max_id;
    TensorRegion round_gather_vals;
    TensorRegion round_gather_ids;
    TensorRegion round_sampled_ids;
    TensorRegion round_sample_pos;
    TensorRegion round_sampling_cfg;
    TensorRegion round_mtp_src_idx;
    TensorRegion round_mtp_last_src;
    TensorRegion round_mtp_last_dst;

    std::size_t lane_sconv_floats = 0;
    std::size_t total_bytes       = 0;
};

struct WorkspaceRegions {
    TensorRegion cur;
    TensorRegion prelude;
    TensorRegion ffn_inp;
    std::array<TensorRegion, 4> blocks{};
    TensorRegion bank0;
    TensorRegion bank_out;
    std::array<TensorRegion, 4> msgs{};
    TensorRegion q_in;
    TensorRegion k_in;
    TensorRegion v_in;
    TensorRegion mlp_in;
    TensorRegion attn_out;
    TensorRegion moe_out;
    TensorRegion ar_staging;
    TensorRegion final_h;

    TensorRegion q_gate_r_sink;
    TensorRegion k_buf;
    TensorRegion v_buf;
    TensorRegion k_dyn;
    TensorRegion v_dyn;
    TensorRegion q_clip;
    TensorRegion rel_bias_f32;
    TensorRegion attn_val;
    TensorRegion split_scratch;

    TensorRegion ctrl_z;
    TensorRegion ctrl_h;
    TensorRegion ctrl_gf;

    // MoE scratch
    TensorRegion moe_router_logits;
    TensorRegion moe_topk_ids;
    TensorRegion moe_topk_weights;
    TensorRegion moe_expert_counts;
    TensorRegion moe_expert_offsets;
    TensorRegion moe_sorted_pairs;
    TensorRegion moe_active_experts;
    TensorRegion moe_num_active;
    TensorRegion moe_inter_bf16;
    TensorRegion moe_accum_f32;
    TensorRegion moe_router_done;

    // Output head / sampling
    TensorRegion logits_shard_f32;
    TensorRegion logits_shard_bf16;
    TensorRegion logits_full_bf16;

    // DSpark MTP scratch
    TensorRegion mtp_target_caps; // [H, 4, max_T]
    TensorRegion mtp_ctx_h;       // [H, max_T]
    TensorRegion mtp_kv_ctx;      // [2 * kv_dim, max_T]
    TensorRegion mtp_h;           // [H, max_B * 8]
    TensorRegion mtp_norm;        // [H, max_B * 8]
    TensorRegion mtp_qkv;         // [5120/tp, max_B * 8]
    TensorRegion mtp_q;           // [128, n_q, max_B * 8]
    TensorRegion mtp_kb;          // [128, n_kv, max_B * 8]
    TensorRegion mtp_vb;          // [128, n_kv, max_B * 8]
    TensorRegion mtp_attn_val;    // [q_dim, max_B * 8]
    TensorRegion mtp_attn_out;    // [H, max_B * 8]
    TensorRegion mtp_gate_up;     // [18432/tp, max_B * 8]
    TensorRegion mtp_mlp_act;     // [9216/tp, max_B * 8]
    TensorRegion mtp_mlp_out;     // [H, max_B * 8]
    TensorRegion mtp_draft_h;     // [H, max_B * 7]
    TensorRegion mtp_base_logits; // BF16 [vocab_rows, max_B * 7]
    TensorRegion mtp_rnn_state;   // BF16 [256, max_B]
    TensorRegion mtp_rnn_z;       // BF16 [5120, max_B]
    TensorRegion mtp_rnn_proj;    // BF16 [768, max_B]
    TensorRegion mtp_rnn_latent;  // BF16 [256, max_B]
    TensorRegion mtp_anchor_emb;  // BF16 [H, max_B]

    std::size_t total_bytes = 0;
};

SequenceRegions build_sequence_regions(const SequenceLayoutSpec& spec) {
    LayoutBuilder b;
    SequenceRegions r;
    const std::int32_t num_kv = kNumKVHeads / spec.tp;
    const std::int32_t kv_dim = num_kv * kHeadDim;
    const std::int32_t B      = static_cast<std::int32_t>(spec.max_concurrency);
    const std::int32_t max_T =
        std::max<std::int32_t>(static_cast<std::int32_t>(spec.prefill_chunk), B * kMtpMaxBlockSize);
    const std::int32_t max_dec_T = B * (spec.mtp_enabled ? kMtpMaxBlockSize : 1);
    const std::uint32_t logical_pages = pages_for_tokens(spec.max_context);

    PagedKVPoolSpec main_spec;
    main_spec.page_group_count      = spec.main_page_groups;
    main_spec.logical_page_capacity = logical_pages;
    main_spec.table_rows            = B;
    main_spec.plane_order           = PagedKVPlaneOrder::PageMajor;
    main_spec.planes.reserve(kNumGlobalLogicalLayers * 2);
    for (int i = 0; i < kNumGlobalLogicalLayers * 2; ++i) {
        main_spec.planes.push_back(PagedKVPlaneSpec{
            .dtype          = DType::BF16,
            .leading_extent = kHeadDim,
            .head_extent    = num_kv,
            .alignment      = 256,
        });
    }
    r.main_kv = plan_paged_kv_pool(b, main_spec);

    for (int i = 0; i < kNumLocalLogicalLayers; ++i) {
        r.swa_k[static_cast<std::size_t>(i)] =
            b.add_tensor(DType::BF16, {kHeadDim, num_kv, kSwaRingCap, B}, 256, "swa_k");
        r.swa_v[static_cast<std::size_t>(i)] =
            b.add_tensor(DType::BF16, {kHeadDim, num_kv, kSwaRingCap, B}, 256, "swa_v");
    }

    r.lane_sconv_floats = static_cast<std::size_t>(kNumLogicalLayers) * 3ULL *
                          static_cast<std::size_t>(2 * kv_dim + 2 * kHiddenSize);
    r.sconv_state = b.add_tensor(
        DType::FP32, {static_cast<std::int32_t>(r.lane_sconv_floats), B}, 256, "sconv_state");
    r.cross_pass_4q = b.add_tensor(DType::FP32, {kHiddenSize, 4, kCoreLen}, 256, "cross_pass_4q");
    r.token_counts  = b.add_tensor(DType::I32, {kUnpaddedVocab, B}, 256, "token_counts");

    if (spec.mtp_enabled) {
        r.sconv_verify = b.add_tensor(
            DType::FP32,
            {static_cast<std::int32_t>(r.lane_sconv_floats), kMtpMaxBlockSize, B}, 256,
            "sconv_verify");

        PagedKVPoolSpec mtp_spec;
        mtp_spec.page_group_count      = spec.main_page_groups;
        mtp_spec.logical_page_capacity = logical_pages;
        mtp_spec.table_rows            = B;
        mtp_spec.plane_order           = PagedKVPlaneOrder::PageMajor;
        for (int i = 0; i < 2; ++i) {
            mtp_spec.planes.push_back(PagedKVPlaneSpec{
                .dtype          = DType::BF16,
                .leading_extent = kHeadDim,
                .head_extent    = num_kv,
                .alignment      = 256,
            });
        }
        r.mtp_global_kv = plan_paged_kv_pool(b, mtp_spec);
        for (int i = 0; i < 2; ++i) {
            r.mtp_swa_k[static_cast<std::size_t>(i)] =
                b.add_tensor(DType::BF16, {kHeadDim, num_kv, kSwaRingCap, B}, 256, "mtp_swa_k");
            r.mtp_swa_v[static_cast<std::size_t>(i)] =
                b.add_tensor(DType::BF16, {kHeadDim, num_kv, kSwaRingCap, B}, 256, "mtp_swa_v");
        }
        const std::int32_t vrows = kPaddedVocabSize / spec.tp;
        r.mtp_fold_w = b.add_tensor(DType::BF16, {kMtpMarkovRank, vrows}, 256, "mtp_fold_w");
        r.mtp_fold_b = b.add_tensor(DType::FP32, {vrows}, 256, "mtp_fold_b");
        r.mtp_f_rnn  = b.add_tensor(DType::BF16, {kHiddenSize, B}, 256, "mtp_f_rnn");
    }

    r.round_token_ids    = b.add_tensor(DType::I32, {max_T}, 256, "round_token_ids");
    r.round_positions    = b.add_tensor(DType::I32, {max_T}, 256, "round_positions");
    r.round_tok_lanes    = b.add_tensor(DType::I32, {max_T}, 256, "round_tok_lanes");
    r.round_batch_lanes  = b.add_tensor(DType::I32, {B}, 256, "round_batch_lanes");
    r.round_ctx_lens     = b.add_tensor(DType::I32, {B}, 256, "round_ctx_lens");
    r.round_acc_counts   = b.add_tensor(DType::I32, {B}, 256, "round_acc_counts");
    r.round_anchor_ids   = b.add_tensor(DType::I32, {B}, 256, "round_anchor_ids");
    r.round_draft_out    = b.add_tensor(DType::I32, {B, 7}, 256, "round_draft_out");
    r.round_winner_ids   = b.add_tensor(DType::I32, {max_dec_T}, 256, "round_winner_ids");
    r.round_local_max_val = b.add_tensor(DType::FP32, {1, max_dec_T}, 256, "round_local_max_val");
    r.round_local_max_id = b.add_tensor(DType::I32, {1, max_dec_T}, 256, "round_local_max_id");
    r.round_gather_vals  = b.add_tensor(DType::FP32, {2, max_dec_T}, 256, "round_gather_vals");
    r.round_gather_ids   = b.add_tensor(DType::I32, {2, max_dec_T}, 256, "round_gather_ids");
    r.round_sampled_ids  = b.add_tensor(DType::I32, {B}, 256, "round_sampled_ids");
    r.round_sample_pos   = b.add_tensor(DType::I32, {B}, 256, "round_sample_pos");
    r.round_sampling_cfg = b.add_tensor(
        DType::U8, {static_cast<std::int32_t>(sizeof(ops::SamplingConfig)) * B}, 256,
        "round_sampling_cfg");
    r.round_mtp_src_idx  = b.add_tensor(DType::I32, {max_dec_T}, 256, "round_mtp_src_idx");
    r.round_mtp_last_src = b.add_tensor(DType::I32, {B}, 256, "round_mtp_last_src");
    r.round_mtp_last_dst = b.add_tensor(DType::I32, {B}, 256, "round_mtp_last_dst");

    r.total_bytes = b.finish(256, "inkling2_seq");
    return r;
}

WorkspaceRegions build_workspace_regions(const SequenceLayoutSpec& spec) {
    LayoutBuilder b;
    WorkspaceRegions w;
    const std::int32_t num_q  = kNumQHeads / spec.tp;
    const std::int32_t num_kv = kNumKVHeads / spec.tp;
    const std::int32_t q_dim  = num_q * kHeadDim;
    const std::int32_t kv_dim = num_kv * kHeadDim;
    const std::int32_t qgrs   = 6552 / spec.tp;
    const std::int32_t slabs  = 2 / spec.tp;
    const std::int32_t vrows  = kPaddedVocabSize / spec.tp;
    const std::int32_t B      = static_cast<std::int32_t>(spec.max_concurrency);
    const std::int32_t max_T =
        std::max<std::int32_t>(static_cast<std::int32_t>(spec.prefill_chunk), B * kMtpMaxBlockSize);
    const std::int32_t max_dec_T = B * (spec.mtp_enabled ? kMtpMaxBlockSize : 1);

    w.cur        = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "cur");
    w.prelude    = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "prelude");
    w.ffn_inp    = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "ffn_inp");
    for (int i = 0; i < 4; ++i) {
        w.blocks[static_cast<std::size_t>(i)] =
            b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "block");
    }
    w.bank0    = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "bank0");
    w.bank_out = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "bank_out");
    for (int i = 0; i < 4; ++i) {
        w.msgs[static_cast<std::size_t>(i)] =
            b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "msg");
    }
    w.q_in       = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "q_in");
    w.k_in       = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "k_in");
    w.v_in       = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "v_in");
    w.mlp_in     = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "mlp_in");
    w.attn_out   = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "attn_out");
    w.moe_out    = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "moe_out");
    w.ar_staging = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "ar_staging");
    w.final_h    = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "final_h");

    w.q_gate_r_sink = b.add_tensor(DType::BF16, {qgrs, max_T}, 256, "qgrs");
    w.k_buf         = b.add_tensor(DType::BF16, {kv_dim, max_T}, 256, "k_buf");
    w.v_buf         = b.add_tensor(DType::BF16, {kv_dim, max_T}, 256, "v_buf");
    w.k_dyn         = b.add_tensor(DType::BF16, {4 * num_kv, max_T}, 256, "k_dyn");
    w.v_dyn         = b.add_tensor(DType::BF16, {4 * num_kv, max_T}, 256, "v_dyn");
    w.q_clip        = b.add_tensor(DType::BF16, {kHeadDim, num_q, max_T}, 256, "q_clip");
    w.rel_bias_f32  = b.add_tensor(DType::FP32, {kRelExtent, num_q, max_T}, 256, "rel_bias");
    w.attn_val      = b.add_tensor(DType::BF16, {q_dim, max_T}, 256, "attn_val");
    w.split_scratch =
        b.add_tensor(DType::FP32, {kHeadDim + 2, 16, num_q, max_dec_T}, 256, "split_scratch");

    w.ctrl_z  = b.add_tensor(DType::BF16, {2 * kHiddenSize, 3 * max_T}, 256, "ctrl_z");
    w.ctrl_h  = b.add_tensor(DType::BF16, {kGateRank, 3 * max_T}, 256, "ctrl_h");
    w.ctrl_gf = b.add_tensor(DType::BF16, {2 * kHiddenSize, 3 * max_T}, 256, "ctrl_gf");

    w.moe_router_logits  = b.add_tensor(DType::FP32, {kNumExperts, max_T}, 256, "moe_rlog");
    w.moe_topk_ids       = b.add_tensor(DType::I32, {kTopK, max_T}, 256, "moe_tids");
    w.moe_topk_weights   = b.add_tensor(DType::FP32, {kTopK, max_T}, 256, "moe_tw");
    w.moe_expert_counts  = b.add_tensor(DType::I32, {kNumExperts}, 256, "moe_ecnt");
    w.moe_expert_offsets = b.add_tensor(DType::I32, {kNumExperts + 1}, 256, "moe_eoff");
    w.moe_sorted_pairs   = b.add_tensor(DType::I32, {max_T * kTopK}, 256, "moe_spairs");
    w.moe_active_experts = b.add_tensor(DType::I32, {kNumExperts}, 256, "moe_aexp");
    w.moe_num_active     = b.add_tensor(DType::I32, {1}, 256, "moe_nact");
    w.moe_inter_bf16 =
        b.add_tensor(DType::BF16, {slabs * kMoeInterPerSlab, max_T * kTopK}, 256, "moe_inter");
    w.moe_accum_f32   = b.add_tensor(DType::FP32, {kHiddenSize, max_T}, 256, "moe_accum");
    w.moe_router_done = b.add_tensor(DType::I32, {1}, 256, "moe_rdone");

    w.logits_shard_f32  = b.add_tensor(DType::FP32, {vrows, max_dec_T}, 256, "logits_shard_f32");
    w.logits_shard_bf16 = b.add_tensor(DType::BF16, {vrows, B}, 256, "logits_shard_bf16");
    w.logits_full_bf16  = b.add_tensor(DType::BF16, {kPaddedVocabSize, B}, 256, "logits_full_bf16");

    if (spec.mtp_enabled) {
        const std::int32_t blk_T = B * kMtpMaxBlockSize;
        const std::int32_t drf_T = B * (kMtpMaxBlockSize - 1);
        w.mtp_target_caps = b.add_tensor(DType::BF16, {kHiddenSize, 4, max_T}, 256, "mtp_caps");
        w.mtp_ctx_h       = b.add_tensor(DType::BF16, {kHiddenSize, max_T}, 256, "mtp_ctx_h");
        w.mtp_kv_ctx      = b.add_tensor(DType::BF16, {2 * kv_dim, max_T}, 256, "mtp_kv_ctx");
        w.mtp_h           = b.add_tensor(DType::BF16, {kHiddenSize, blk_T}, 256, "mtp_h");
        w.mtp_norm        = b.add_tensor(DType::BF16, {kHiddenSize, blk_T}, 256, "mtp_norm");
        w.mtp_qkv         = b.add_tensor(DType::BF16, {5120 / spec.tp, blk_T}, 256, "mtp_qkv");
        w.mtp_q           = b.add_tensor(DType::BF16, {kHeadDim, num_q, blk_T}, 256, "mtp_q");
        w.mtp_kb          = b.add_tensor(DType::BF16, {kHeadDim, num_kv, blk_T}, 256, "mtp_kb");
        w.mtp_vb          = b.add_tensor(DType::BF16, {kHeadDim, num_kv, blk_T}, 256, "mtp_vb");
        w.mtp_attn_val    = b.add_tensor(DType::BF16, {q_dim, blk_T}, 256, "mtp_attn_val");
        w.mtp_attn_out    = b.add_tensor(DType::BF16, {kHiddenSize, blk_T}, 256, "mtp_attn_out");
        w.mtp_gate_up =
            b.add_tensor(DType::BF16, {(2 * kMtpInterSize) / spec.tp, blk_T}, 256, "mtp_gu");
        w.mtp_mlp_act =
            b.add_tensor(DType::BF16, {kMtpInterSize / spec.tp, blk_T}, 256, "mtp_act");
        w.mtp_mlp_out     = b.add_tensor(DType::BF16, {kHiddenSize, blk_T}, 256, "mtp_mlp_out");
        w.mtp_draft_h     = b.add_tensor(DType::BF16, {kHiddenSize, drf_T}, 256, "mtp_draft_h");
        w.mtp_base_logits = b.add_tensor(DType::BF16, {vrows, drf_T}, 256, "mtp_base_logits");
        w.mtp_rnn_state   = b.add_tensor(DType::BF16, {kMtpMarkovRank, B}, 256, "mtp_rnn_state");
        w.mtp_rnn_z       = b.add_tensor(DType::BF16, {2 * kMtpMarkovRank + 2 * kHiddenSize, B},
                                         256, "mtp_rnn_z");
        w.mtp_rnn_proj    = b.add_tensor(DType::BF16, {3 * kMtpMarkovRank, B}, 256, "mtp_rnn_proj");
        w.mtp_rnn_latent  = b.add_tensor(DType::BF16, {kMtpMarkovRank, B}, 256, "mtp_rnn_lat");
        w.mtp_anchor_emb  = b.add_tensor(DType::BF16, {kHiddenSize, B}, 256, "mtp_anchor_emb");
    }

    w.total_bytes = b.finish(256, "inkling2_ws");
    return w;
}

std::size_t sampling_ws_bytes(std::uint32_t max_concurrency) {
    return ops::sampling_workspace_capacity_bytes(
        kUnpaddedVocab, 1, static_cast<std::int32_t>(max_concurrency));
}

} // namespace

namespace detail {

struct SequencePlanImpl {
    SequenceLayoutSpec spec;
    SequenceRegions seq_regions;
    WorkspaceRegions ws_regions;
    std::size_t sample_ws_bytes = 0;
    std::size_t reservation_bytes = 0;
};

struct SequencePlannerImpl {
    SequenceLayoutSpec base_spec;
    runtime::SequenceCapacityCurve curve;
};

struct RequestBasePlanImpl {
    runtime::RequestPlanSummary summary;
    runtime::ResolvedExecutionOptions options;
};

struct RequestPlanImpl {
    runtime::RequestPlanSummary summary;
    runtime::ResolvedExecutionOptions options;
    std::uint32_t target_lane = 0;
};

// ---------------------------------------------------------------------------
// Per-rank runtime storage
// ---------------------------------------------------------------------------

struct RankState {
    int rank = 0;
    DeviceContext* dev = nullptr;
    const RankModelView* model = nullptr;

    std::unique_ptr<DeviceArena> seq_arena;
    std::unique_ptr<DeviceArena> ws_arena;
    std::unique_ptr<WorkspaceArena> sample_ws;
    std::unique_ptr<PagedKVPool> main_kv;
    std::unique_ptr<PagedKVPool> mtp_global_kv;

    std::array<Tensor, kNumLocalLogicalLayers> swa_k{};
    std::array<Tensor, kNumLocalLogicalLayers> swa_v{};
    Tensor sconv_state;
    Tensor sconv_verify;
    Tensor cross_pass_4q;
    Tensor token_counts;

    std::array<Tensor, 2> mtp_swa_k{};
    std::array<Tensor, 2> mtp_swa_v{};
    Tensor mtp_fold_w;
    Tensor mtp_fold_b;
    Tensor mtp_f_rnn;

    Tensor round_token_ids;
    Tensor round_positions;
    Tensor round_tok_lanes;
    Tensor round_batch_lanes;
    Tensor round_ctx_lens;
    Tensor round_acc_counts;
    Tensor round_anchor_ids;
    Tensor round_draft_out;
    Tensor round_winner_ids;
    Tensor round_local_max_val;
    Tensor round_local_max_id;
    Tensor round_gather_vals;
    Tensor round_gather_ids;
    Tensor round_sampled_ids;
    Tensor round_sample_pos;
    Tensor round_sampling_cfg;
    Tensor round_mtp_src_idx;
    Tensor round_mtp_last_src;
    Tensor round_mtp_last_dst;

    // Workspace bound tensors
    Tensor cur, prelude, ffn_inp, bank0, bank_out, q_in, k_in, v_in, mlp_in, attn_out, moe_out,
        ar_staging, final_h;
    std::array<Tensor, 4> blocks{};
    std::array<Tensor, 4> msgs{};
    Tensor q_gate_r_sink, k_buf, v_buf, k_dyn, v_dyn, q_clip, rel_bias_f32, attn_val, split_scratch;
    Tensor ctrl_z, ctrl_h, ctrl_gf;
    ops::Inkling2MoeScratch moe_sc{};
    Tensor moe_router_logits, moe_inter_bf16;
    Tensor logits_shard_f32, logits_shard_bf16, logits_full_bf16;

    // MTP workspace
    Tensor mtp_target_caps, mtp_ctx_h, mtp_kv_ctx, mtp_h, mtp_norm, mtp_qkv, mtp_q, mtp_kb, mtp_vb,
        mtp_attn_val, mtp_attn_out, mtp_gate_up, mtp_mlp_act, mtp_mlp_out, mtp_draft_h,
        mtp_base_logits, mtp_rnn_state, mtp_rnn_z, mtp_rnn_proj, mtp_rnn_latent, mtp_anchor_emb;
};

struct GraphSlot {
    DecodeGraphDefinition def;
    DecodeGraphExecutable exec;
    bool captured = false;
};

class ProgramImpl {
public:
    ProgramImpl(const LoadedModelData& model, SequencePlanImpl plan, ExecutionContext& execution)
        : model_(model), plan_(std::move(plan)), ec_(execution), tp_(plan_.spec.tp),
          max_concurrency_(plan_.spec.max_concurrency), max_context_(plan_.spec.max_context),
          prefill_chunk_(plan_.spec.prefill_chunk), mtp_enabled_(plan_.spec.mtp_enabled),
          draft_tokens_(plan_.spec.draft_tokens), use_cuda_graph_(plan_.spec.use_cuda_graph) {
        if (ec_.device_count != tp_) {
            throw std::invalid_argument("inkling2 ProgramImpl: ExecutionContext width != tp");
        }
        if (tp_ == 2) {
            const bool p2p = ops::enable_peer_access(ec_);
            peer_events_.emplace(ec_, p2p);
            peer_bridge_.emplace(ec_.dev[0]->device, ec_.dev[1]->device);
        }

        for (int r = 0; r < tp_; ++r) {
            auto& rs = ranks_[static_cast<std::size_t>(r)];
            rs.rank  = r;
            rs.dev   = &(*ec_.dev[static_cast<std::size_t>(r)]);
            rs.model = &model_.view(r);
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            ops::inkling2_ensure_argmax_scratch(rs.dev->device, 32768);
            ops::inkling2_ensure_moe_scratch(rs.dev->device,
                                             std::max<std::size_t>(prefill_chunk_, 256));

            rs.seq_arena = std::make_unique<DeviceArena>(plan_.seq_regions.total_bytes);
            rs.ws_arena  = std::make_unique<DeviceArena>(plan_.ws_regions.total_bytes);
            if (r == 0 && plan_.sample_ws_bytes > 0) {
                rs.sample_ws = std::make_unique<WorkspaceArena>(plan_.sample_ws_bytes);
            }
            const DeviceSpan s_span = rs.seq_arena->alloc_bytes(plan_.seq_regions.total_bytes, 256);
            const DeviceSpan w_span = rs.ws_arena->alloc_bytes(plan_.ws_regions.total_bytes, 256);

            const auto& sr = plan_.seq_regions;
            rs.main_kv     = std::make_unique<PagedKVPool>(s_span, sr.main_kv);
            for (int i = 0; i < kNumLocalLogicalLayers; ++i) {
                rs.swa_k[static_cast<std::size_t>(i)] = sr.swa_k[static_cast<std::size_t>(i)].bind(s_span);
                rs.swa_v[static_cast<std::size_t>(i)] = sr.swa_v[static_cast<std::size_t>(i)].bind(s_span);
            }
            rs.sconv_state   = sr.sconv_state.bind(s_span);
            rs.cross_pass_4q = sr.cross_pass_4q.bind(s_span);
            rs.token_counts  = sr.token_counts.bind(s_span);

            rs.round_token_ids    = sr.round_token_ids.bind(s_span);
            rs.round_positions    = sr.round_positions.bind(s_span);
            rs.round_tok_lanes    = sr.round_tok_lanes.bind(s_span);
            rs.round_batch_lanes  = sr.round_batch_lanes.bind(s_span);
            rs.round_ctx_lens     = sr.round_ctx_lens.bind(s_span);
            rs.round_acc_counts   = sr.round_acc_counts.bind(s_span);
            rs.round_anchor_ids   = sr.round_anchor_ids.bind(s_span);
            rs.round_draft_out    = sr.round_draft_out.bind(s_span);
            rs.round_winner_ids   = sr.round_winner_ids.bind(s_span);
            rs.round_local_max_val = sr.round_local_max_val.bind(s_span);
            rs.round_local_max_id = sr.round_local_max_id.bind(s_span);
            rs.round_gather_vals  = sr.round_gather_vals.bind(s_span);
            rs.round_gather_ids   = sr.round_gather_ids.bind(s_span);
            rs.round_sampled_ids  = sr.round_sampled_ids.bind(s_span);
            rs.round_sample_pos   = sr.round_sample_pos.bind(s_span);
            rs.round_sampling_cfg = sr.round_sampling_cfg.bind(s_span);
            rs.round_mtp_src_idx  = sr.round_mtp_src_idx.bind(s_span);
            rs.round_mtp_last_src = sr.round_mtp_last_src.bind(s_span);
            rs.round_mtp_last_dst = sr.round_mtp_last_dst.bind(s_span);

            const auto& wr = plan_.ws_regions;
            rs.cur        = wr.cur.bind(w_span);
            rs.prelude    = wr.prelude.bind(w_span);
            rs.ffn_inp    = wr.ffn_inp.bind(w_span);
            for (int i = 0; i < 4; ++i) {
                rs.blocks[static_cast<std::size_t>(i)] = wr.blocks[static_cast<std::size_t>(i)].bind(w_span);
                rs.msgs[static_cast<std::size_t>(i)]   = wr.msgs[static_cast<std::size_t>(i)].bind(w_span);
            }
            rs.bank0         = wr.bank0.bind(w_span);
            rs.bank_out      = wr.bank_out.bind(w_span);
            rs.q_in          = wr.q_in.bind(w_span);
            rs.k_in          = wr.k_in.bind(w_span);
            rs.v_in          = wr.v_in.bind(w_span);
            rs.mlp_in        = wr.mlp_in.bind(w_span);
            rs.attn_out      = wr.attn_out.bind(w_span);
            rs.moe_out       = wr.moe_out.bind(w_span);
            rs.ar_staging    = wr.ar_staging.bind(w_span);
            rs.final_h       = wr.final_h.bind(w_span);
            rs.q_gate_r_sink = wr.q_gate_r_sink.bind(w_span);
            rs.k_buf         = wr.k_buf.bind(w_span);
            rs.v_buf         = wr.v_buf.bind(w_span);
            rs.k_dyn         = wr.k_dyn.bind(w_span);
            rs.v_dyn         = wr.v_dyn.bind(w_span);
            rs.q_clip        = wr.q_clip.bind(w_span);
            rs.rel_bias_f32  = wr.rel_bias_f32.bind(w_span);
            rs.attn_val      = wr.attn_val.bind(w_span);
            rs.split_scratch = wr.split_scratch.bind(w_span);
            rs.ctrl_z        = wr.ctrl_z.bind(w_span);
            rs.ctrl_h        = wr.ctrl_h.bind(w_span);
            rs.ctrl_gf       = wr.ctrl_gf.bind(w_span);

            rs.moe_router_logits       = wr.moe_router_logits.bind(w_span);
            rs.moe_inter_bf16          = wr.moe_inter_bf16.bind(w_span);
            rs.moe_sc.router_logits    = static_cast<float*>(rs.moe_router_logits.data);
            rs.moe_sc.topk_ids         = static_cast<std::int32_t*>(wr.moe_topk_ids.bind(w_span).data);
            rs.moe_sc.topk_weights     = static_cast<float*>(wr.moe_topk_weights.bind(w_span).data);
            rs.moe_sc.expert_counts    = static_cast<std::int32_t*>(wr.moe_expert_counts.bind(w_span).data);
            rs.moe_sc.expert_offsets   = static_cast<std::int32_t*>(wr.moe_expert_offsets.bind(w_span).data);
            rs.moe_sc.sorted_pairs     = static_cast<std::int32_t*>(wr.moe_sorted_pairs.bind(w_span).data);
            rs.moe_sc.active_experts   = static_cast<std::int32_t*>(wr.moe_active_experts.bind(w_span).data);
            rs.moe_sc.num_active       = static_cast<std::int32_t*>(wr.moe_num_active.bind(w_span).data);
            rs.moe_sc.intermediate_bf16 = rs.moe_inter_bf16.data;
            rs.moe_sc.accum_f32        = static_cast<float*>(wr.moe_accum_f32.bind(w_span).data);
            rs.moe_sc.router_done      = static_cast<std::int32_t*>(wr.moe_router_done.bind(w_span).data);
            CUDA_CHECK(cudaMemsetAsync(rs.moe_sc.router_done, 0, sizeof(std::int32_t), rs.dev->stream));

            rs.logits_shard_f32  = wr.logits_shard_f32.bind(w_span);
            rs.logits_shard_bf16 = wr.logits_shard_bf16.bind(w_span);
            rs.logits_full_bf16  = wr.logits_full_bf16.bind(w_span);

            // Build pre-rearranged cross-pass query blocks [4, 2304] per core layer (idx 0..11)
            const std::size_t q_bytes = static_cast<std::size_t>(kHiddenSize) * sizeof(float);
            auto* dst_base            = static_cast<std::byte*>(rs.cross_pass_4q.data);
            const auto* qc_base       = static_cast<const std::byte*>(rs.model->cross_pass_queries.data);
            const auto* qkv_base      = static_cast<const std::byte*>(rs.model->cross_pass_kv_queries.data);
            for (int idx = 0; idx < kCoreLen; ++idx) {
                std::byte* blk = dst_base + static_cast<std::size_t>(idx) * 4 * q_bytes;
                CUDA_CHECK(cudaMemcpyAsync(blk + 0 * q_bytes, qc_base + (2 * idx) * q_bytes,
                                           q_bytes, cudaMemcpyDeviceToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(blk + 1 * q_bytes, qkv_base + (2 * idx) * q_bytes,
                                           q_bytes, cudaMemcpyDeviceToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(blk + 2 * q_bytes, qkv_base + (2 * idx + 1) * q_bytes,
                                           q_bytes, cudaMemcpyDeviceToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(blk + 3 * q_bytes, qc_base + (2 * idx + 1) * q_bytes,
                                           q_bytes, cudaMemcpyDeviceToDevice, rs.dev->stream));
            }

            if (mtp_enabled_) {
                rs.sconv_verify  = sr.sconv_verify.bind(s_span);
                rs.mtp_global_kv = std::make_unique<PagedKVPool>(s_span, sr.mtp_global_kv);
                for (int i = 0; i < 2; ++i) {
                    rs.mtp_swa_k[static_cast<std::size_t>(i)] =
                        sr.mtp_swa_k[static_cast<std::size_t>(i)].bind(s_span);
                    rs.mtp_swa_v[static_cast<std::size_t>(i)] =
                        sr.mtp_swa_v[static_cast<std::size_t>(i)].bind(s_span);
                }
                rs.mtp_fold_w = sr.mtp_fold_w.bind(s_span);
                rs.mtp_fold_b = sr.mtp_fold_b.bind(s_span);
                rs.mtp_f_rnn  = sr.mtp_f_rnn.bind(s_span);

                rs.mtp_target_caps = wr.mtp_target_caps.bind(w_span);
                rs.mtp_ctx_h       = wr.mtp_ctx_h.bind(w_span);
                rs.mtp_kv_ctx      = wr.mtp_kv_ctx.bind(w_span);
                rs.mtp_h           = wr.mtp_h.bind(w_span);
                rs.mtp_norm        = wr.mtp_norm.bind(w_span);
                rs.mtp_qkv         = wr.mtp_qkv.bind(w_span);
                rs.mtp_q           = wr.mtp_q.bind(w_span);
                rs.mtp_kb          = wr.mtp_kb.bind(w_span);
                rs.mtp_vb          = wr.mtp_vb.bind(w_span);
                rs.mtp_attn_val    = wr.mtp_attn_val.bind(w_span);
                rs.mtp_attn_out    = wr.mtp_attn_out.bind(w_span);
                rs.mtp_gate_up     = wr.mtp_gate_up.bind(w_span);
                rs.mtp_mlp_act     = wr.mtp_mlp_act.bind(w_span);
                rs.mtp_mlp_out     = wr.mtp_mlp_out.bind(w_span);
                rs.mtp_draft_h     = wr.mtp_draft_h.bind(w_span);
                rs.mtp_base_logits = wr.mtp_base_logits.bind(w_span);
                rs.mtp_rnn_state   = wr.mtp_rnn_state.bind(w_span);
                rs.mtp_rnn_z       = wr.mtp_rnn_z.bind(w_span);
                rs.mtp_rnn_proj    = wr.mtp_rnn_proj.bind(w_span);
                rs.mtp_rnn_latent  = wr.mtp_rnn_latent.bind(w_span);
                rs.mtp_anchor_emb  = wr.mtp_anchor_emb.bind(w_span);

                // Precompute folded Markov head on GPU once:
                // fold_w [256, vocab_rows] = output_head [2304, vocab_rows] @ hidden_out_weight^T
                // fold_b [vocab_rows]      = output_head [2304, vocab_rows] @ hidden_out_bias
                Tensor wt_tmp(rs.ar_staging.data, DType::BF16, {kHiddenSize, kMtpMarkovRank});
                ops::inkling2_bf16_transpose_2d(rs.model->mtp.markov_hidden_out_weight, wt_tmp,
                                                rs.dev->stream);
                ops::inkling2_bf16_linear(rs.model->output_head, wt_tmp, rs.mtp_fold_w,
                                          rs.dev->stream);
                Tensor b_in(rs.model->mtp.markov_hidden_out_bias.data, DType::BF16,
                            {kHiddenSize, 1});
                Tensor b_out(rs.mtp_fold_b.data, DType::FP32, {rs.model->vocab_rows, 1});
                ops::inkling2_bf16_linear(b_in, rs.model->output_head, b_out, rs.dev->stream);
            }
            rs.dev->synchronize();
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
    }

    ~ProgramImpl() noexcept {
        for (int r = tp_ - 1; r >= 0; --r) {
            try {
                ranks_[static_cast<std::size_t>(r)].dev->synchronize();
            } catch (...) {}
        }
    }

    // ---- Admission / Request Planning ----
    runtime::AdmissionResources admission_capacity() const noexcept {
        return runtime::AdmissionResources{
            .active_lanes     = max_concurrency_,
            .main_kv_pages    = ranks_[0].main_kv->page_group_count(),
            .backend_kv_pages = 0,
        };
    }

    RequestBasePlan plan_request_base(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
        const auto& data = PreparedPromptAccess::view(prompt);
        const auto prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
        if (prompt_tokens == 0 || prompt_tokens >= max_context_) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                               "prompt length exceeds inkling2-mini max_context");
        }
        const std::uint32_t remaining = max_context_ - prompt_tokens;
        const bool user_limited =
            options.requested_output_tokens != 0 && options.requested_output_tokens <= remaining;
        const std::uint32_t effective_out =
            user_limited ? options.requested_output_tokens : remaining;
        const FinishReason limit_reason =
            user_limited ? FinishReason::OutputLimit : FinishReason::ContextCapacity;

        const std::uint32_t spec_headroom = mtp_enabled_ ? draft_tokens_ : 0U;
        const std::uint32_t total_tokens =
            std::min(max_context_, prompt_tokens + effective_out + spec_headroom);
        const std::uint32_t needed_pages = pages_for_tokens(total_tokens);
        const std::uint32_t chunks =
            (prompt_tokens + prefill_chunk_ - 1) / prefill_chunk_;

        auto impl     = std::make_unique<RequestBasePlanImpl>();
        impl->options = options;
        impl->summary = runtime::RequestPlanSummary{
            .prompt_tokens           = prompt_tokens,
            .reusable_prompt_tokens  = 0,
            .requested_output_tokens = options.requested_output_tokens,
            .effective_output_tokens = effective_out,
            .effective_limit_reason  = limit_reason,
            .transient_bytes         = 0,
            .transient_alignment     = 1,
            .admission =
                runtime::AdmissionResources{
                    .active_lanes     = 1,
                    .main_kv_pages    = needed_pages,
                    .backend_kv_pages = 0,
                },
            .service_work_quanta = static_cast<std::uint64_t>(chunks) +
                                   static_cast<std::uint64_t>(effective_out > 0 ? effective_out - 1 : 0),
        };
        return RequestBasePlan(std::move(impl));
    }

    RequestPlan plan_request_for_lane(std::uint32_t lane, const PreparedPrompt& /*prompt*/,
                                      const RequestBasePlan& base) {
        auto impl         = std::make_unique<RequestPlanImpl>();
        impl->summary     = base.impl_->summary;
        impl->options     = base.impl_->options;
        impl->target_lane = lane;
        return RequestPlan(std::move(impl));
    }

    bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
        if (lane >= max_concurrency_ || lanes_[lane].active) { return false; }
        const std::uint32_t pages = plan.impl_->summary.admission.main_kv_pages;
        for (int r = 0; r < tp_; ++r) {
            if (!ranks_[static_cast<std::size_t>(r)].main_kv->can_reserve(pages)) { return false; }
            if (mtp_enabled_ &&
                !ranks_[static_cast<std::size_t>(r)].mtp_global_kv->can_reserve(pages)) {
                return false;
            }
        }
        return true;
    }

    bool can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                const RequestPlan& plan) const noexcept {
        return can_admit_lane(lane, plan);
    }

    // ---- Prefill ----
    runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                                  RequestPlan&& plan,
                                                  runtime::TransientRegion /*transient*/) {
        auto& st = lanes_[lane];
        clear_lane(lane);
        st.active            = true;
        st.prefilling        = true;
        st.prompt_tokens     = PreparedPromptAccess::take(std::move(prompt)).token_ids;
        st.options           = plan.impl_->options;
        st.reserved_pages    = plan.impl_->summary.admission.main_kv_pages;
        st.prefill_cursor    = 0;
        st.seq_len           = 0;
        st.timings           = {};
        st.spec_stats        = {};
        st.spec_stats.backend      = mtp_enabled_ ? plan_.spec.spec_backend : SpeculativeBackend::None;
        st.spec_stats.enabled      = mtp_enabled_;
        st.spec_stats.draft_window = mtp_enabled_ ? draft_tokens_ : 0;
        if (mtp_enabled_) { st.spec_stats.accepted_per_position.assign(draft_tokens_, 0); }

        // Reserve & bind Paged KV allocations on every rank, zero sconv states & token counts
        for (int r = 0; r < tp_; ++r) {
            auto& rs = ranks_[static_cast<std::size_t>(r)];
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            st.kv_alloc[static_cast<std::size_t>(r)] = rs.main_kv->reserve(st.reserved_pages);
            st.kv_alloc[static_cast<std::size_t>(r)].bind_row(static_cast<std::int32_t>(lane),
                                                              rs.dev->stream);
            if (mtp_enabled_) {
                st.mtp_kv_alloc[static_cast<std::size_t>(r)] =
                    rs.mtp_global_kv->reserve(st.reserved_pages);
                st.mtp_kv_alloc[static_cast<std::size_t>(r)].bind_row(
                    static_cast<std::int32_t>(lane), rs.dev->stream);
            }
            const std::size_t lane_bytes = plan_.seq_regions.lane_sconv_floats * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(
                static_cast<std::byte*>(rs.sconv_state.data) + lane * lane_bytes, 0, lane_bytes,
                rs.dev->stream));
            CUDA_CHECK(cudaMemsetAsync(
                static_cast<std::int32_t*>(rs.token_counts.data) +
                    static_cast<std::size_t>(lane) * kUnpaddedVocab,
                0, static_cast<std::size_t>(kUnpaddedVocab) * sizeof(std::int32_t), rs.dev->stream));
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
        return run_prefill_chunk(lane);
    }

    runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane) {
        if (!lanes_[lane].active || !lanes_[lane].prefilling) {
            throw std::logic_error("inkling2 advance_prefill_lane: lane not prefilling");
        }
        return run_prefill_chunk(lane);
    }

    void resolve_prefill_lane(std::uint32_t lane, bool terminal) {
        auto& st = lanes_[lane];
        if (!st.active) { throw std::logic_error("inkling2 resolve_prefill_lane: lane inactive"); }
        if (terminal) {
            clear_lane(lane);
            return;
        }
        // If DSpark is enabled and request is greedy, draft the initial block anchored at first_token
        if (mtp_enabled_ && st.options.sampling.temperature <= 0.0f) {
            const std::uint32_t single_lane = lane;
            run_dspark_draft_batch(std::span<const std::uint32_t>(&single_lane, 1));
            st.has_draft = true;
        }
    }

    // ---- Batched Decode (Normal + DSpark Speculative) ----
    runtime::BatchedGeneratedRound decode_batch(std::span<const std::uint32_t> lanes,
                                                std::span<const runtime::RoundBudget> budgets) {
        const auto t0      = Clock::now();
        const int B        = static_cast<int>(lanes.size());
        if (B <= 0 || B > static_cast<int>(max_concurrency_)) {
            throw std::invalid_argument("inkling2 decode_batch: invalid batch size");
        }

        // Determine if entire batch qualifies for DSpark speculative verification
        bool all_greedy = true;
        bool can_spec   = mtp_enabled_;
        for (int b = 0; b < B; ++b) {
            const auto& st = lanes_[lanes[static_cast<std::size_t>(b)]];
            if (!st.active || st.prefilling) {
                throw std::logic_error("inkling2 decode_batch: lane not decode-ready");
            }
            if (st.options.sampling.temperature > 0.0f) {
                all_greedy = false;
                can_spec   = false;
            }
            if (!st.has_draft || budgets[static_cast<std::size_t>(b)].generated_tokens_remaining < 2 ||
                st.seq_len + draft_tokens_ + 1 > max_context_) {
                can_spec = false;
            }
        }

        pending_lanes_.assign(lanes.begin(), lanes.end());
        pending_was_spec_ = can_spec;
        pending_emitted_counts_.assign(static_cast<std::size_t>(B), 1);

        if (can_spec) {
            const int V = static_cast<int>(draft_tokens_) + 1; // verify_len (e.g. 8)
            pending_tokens_.assign(static_cast<std::size_t>(B * V), 0);

            // Ensure KV pages materialized for [seq_len .. seq_len + V) on every rank
            std::vector<std::int32_t> h_ids(static_cast<std::size_t>(B * V));
            std::vector<std::int32_t> h_pos(static_cast<std::size_t>(B * V));
            std::vector<std::int32_t> h_tok_lanes(static_cast<std::size_t>(B * V));
            std::vector<std::int32_t> h_batch_lanes(static_cast<std::size_t>(B));

            for (int b = 0; b < B; ++b) {
                const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
                auto& st              = lanes_[l];
                h_batch_lanes[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(l);
                h_ids[static_cast<std::size_t>(b * V + 0)] = st.last_token;
                for (int d = 0; d < static_cast<int>(draft_tokens_); ++d) {
                    h_ids[static_cast<std::size_t>(b * V + 1 + d)] =
                        st.draft_tokens[static_cast<std::size_t>(d)];
                }
                for (int v = 0; v < V; ++v) {
                    h_pos[static_cast<std::size_t>(b * V + v)] =
                        static_cast<std::int32_t>(st.seq_len) + v;
                    h_tok_lanes[static_cast<std::size_t>(b * V + v)] = static_cast<std::int32_t>(l);
                }
                for (int r = 0; r < tp_; ++r) {
                    auto& rs = ranks_[static_cast<std::size_t>(r)];
                    CUDA_CHECK(cudaSetDevice(rs.dev->device));
                    ensure_kv_tokens(st.kv_alloc[static_cast<std::size_t>(r)],
                                     st.seq_len + static_cast<std::uint32_t>(V), rs.dev->stream);
                }
            }
            CUDA_CHECK(cudaSetDevice(ec_.primary().device));

            // Upload round metadata to all ranks
            for (int r = 0; r < tp_; ++r) {
                auto& rs = ranks_[static_cast<std::size_t>(r)];
                CUDA_CHECK(cudaSetDevice(rs.dev->device));
                CUDA_CHECK(cudaMemcpyAsync(rs.round_token_ids.data, h_ids.data(),
                                           h_ids.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(rs.round_positions.data, h_pos.data(),
                                           h_pos.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(rs.round_tok_lanes.data, h_tok_lanes.data(),
                                           h_tok_lanes.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(rs.round_batch_lanes.data, h_batch_lanes.data(),
                                           h_batch_lanes.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, rs.dev->stream));
            }
            CUDA_CHECK(cudaSetDevice(ec_.primary().device));

            // Run target verify forward (captured CUDA Graph per batch size B!)
            run_verify_step_maybe_graph(B, V);

            // Download B * V predicted target tokens from rank 0
            std::vector<std::int32_t> target_preds(static_cast<std::size_t>(B * V));
            CUDA_CHECK(cudaMemcpyAsync(target_preds.data(), ranks_[0].round_winner_ids.data,
                                       target_preds.size() * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, ranks_[0].dev->stream));
            ranks_[0].dev->synchronize();
            if (!use_cuda_graph_ && tp_ == 2) { ranks_[1].dev->synchronize(); }

            // Verify drafts against target greedy predictions
            for (int b = 0; b < B; ++b) {
                const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
                auto& st              = lanes_[l];
                const int max_emit =
                    std::min<int>(V, static_cast<int>(budgets[static_cast<std::size_t>(b)]
                                                          .generated_tokens_remaining));
                int acc = 0;
                for (int d = 0; d < static_cast<int>(draft_tokens_) && (acc + 1) < max_emit; ++d) {
                    const TokenId pred = target_preds[static_cast<std::size_t>(b * V + d)];
                    if (pred == st.draft_tokens[static_cast<std::size_t>(d)]) {
                        pending_tokens_[static_cast<std::size_t>(b * V + acc)] = pred;
                        ++acc;
                        ++st.spec_stats.accepted_per_position[static_cast<std::size_t>(d)];
                    } else {
                        break;
                    }
                }
                // Bonus / correction token at index `acc`
                pending_tokens_[static_cast<std::size_t>(b * V + acc)] =
                    target_preds[static_cast<std::size_t>(b * V + acc)];
                const int emitted = acc + 1;
                pending_emitted_counts_[static_cast<std::size_t>(b)] = emitted;
                ++st.spec_stats.rounds;
                st.spec_stats.drafted_tokens += draft_tokens_;
                st.spec_stats.accepted_tokens += static_cast<std::uint64_t>(acc);
            }

            const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
            for (const std::uint32_t l : lanes) { lanes_[l].timings.decode_seconds += dt; }

            return runtime::BatchedGeneratedRound{
                .tokens     = std::span<const TokenId>(pending_tokens_.data(), pending_tokens_.size()),
                .row_counts = std::span<const std::int32_t>(pending_emitted_counts_.data(),
                                                            pending_emitted_counts_.size()),
                .row_stride = static_cast<std::uint32_t>(V),
            };
        }

        // ---- Ordinary 1-token batched decode ----
        pending_tokens_.assign(static_cast<std::size_t>(B), 0);
        std::vector<std::int32_t> h_ids(static_cast<std::size_t>(B));
        std::vector<std::int32_t> h_pos(static_cast<std::size_t>(B));
        std::vector<std::int32_t> h_lanes(static_cast<std::size_t>(B));

        for (int b = 0; b < B; ++b) {
            const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
            auto& st              = lanes_[l];
            if (mtp_enabled_) { ++st.spec_stats.fallback_steps; }
            h_ids[static_cast<std::size_t>(b)]   = st.last_token;
            h_pos[static_cast<std::size_t>(b)]   = static_cast<std::int32_t>(st.seq_len);
            h_lanes[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(l);
            for (int r = 0; r < tp_; ++r) {
                auto& rs = ranks_[static_cast<std::size_t>(r)];
                CUDA_CHECK(cudaSetDevice(rs.dev->device));
                ensure_kv_tokens(st.kv_alloc[static_cast<std::size_t>(r)], st.seq_len + 1,
                                 rs.dev->stream);
            }
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));

        for (int r = 0; r < tp_; ++r) {
            auto& rs = ranks_[static_cast<std::size_t>(r)];
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_token_ids.data, h_ids.data(),
                                       h_ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       rs.dev->stream));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_positions.data, h_pos.data(),
                                       h_pos.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       rs.dev->stream));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_tok_lanes.data, h_lanes.data(),
                                       h_lanes.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, rs.dev->stream));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_batch_lanes.data, h_lanes.data(),
                                       h_lanes.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, rs.dev->stream));
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));

        if (all_greedy && !logits_capture_enabled_) {
            run_decode_greedy_maybe_graph(B);
            CUDA_CHECK(cudaMemcpyAsync(pending_tokens_.data(), ranks_[0].round_winner_ids.data,
                                       static_cast<std::size_t>(B) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, ranks_[0].dev->stream));
            ranks_[0].dev->synchronize();
            if (!use_cuda_graph_ && tp_ == 2) { ranks_[1].dev->synchronize(); }
        } else {
            // Eager forward + stochastic sampling / logit capture
            run_forward_body(ForwardMode::Decode, B, 1, 0, 0, mtp_enabled_);
            sample_or_capture_logits(lanes, ops::kSamplePurposeDecode, pending_tokens_);
        }

        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        for (const std::uint32_t l : lanes) { lanes_[l].timings.decode_seconds += dt; }

        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(pending_tokens_.data(), pending_tokens_.size()),
            .row_counts = {},
            .row_stride = 1,
        };
    }

    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled) {
        const auto t0 = Clock::now();
        const int B   = static_cast<int>(lanes.size());
        const int V   = pending_was_spec_ ? (static_cast<int>(draft_tokens_) + 1) : 1;

        if (pending_was_spec_) {
            // 1. Commit SConv verify states for all B lanes in ONE launch per rank!
            std::vector<std::int32_t> h_batch_lanes(static_cast<std::size_t>(B));
            std::vector<std::int32_t> h_acc(static_cast<std::size_t>(B));
            for (int b = 0; b < B; ++b) {
                h_batch_lanes[static_cast<std::size_t>(b)] =
                    static_cast<std::int32_t>(lanes[static_cast<std::size_t>(b)]);
                h_acc[static_cast<std::size_t>(b)] =
                    (cancelled[static_cast<std::size_t>(b)] != 0 ||
                     terminal[static_cast<std::size_t>(b)] != 0)
                        ? 0
                        : static_cast<std::int32_t>(accepted_tokens[static_cast<std::size_t>(b)]);
            }
            for (int r = 0; r < tp_; ++r) {
                auto& rs = ranks_[static_cast<std::size_t>(r)];
                CUDA_CHECK(cudaSetDevice(rs.dev->device));
                CUDA_CHECK(cudaMemcpyAsync(rs.round_batch_lanes.data, h_batch_lanes.data(),
                                           h_batch_lanes.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, rs.dev->stream));
                CUDA_CHECK(cudaMemcpyAsync(rs.round_acc_counts.data, h_acc.data(),
                                           h_acc.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, rs.dev->stream));
                ops::inkling2_sconv_commit_verify(
                    static_cast<float*>(rs.sconv_state.data),
                    static_cast<const float*>(rs.sconv_verify.data),
                    plan_.seq_regions.lane_sconv_floats, kMtpMaxBlockSize,
                    static_cast<const std::int32_t*>(rs.round_batch_lanes.data),
                    static_cast<const std::int32_t*>(rs.round_acc_counts.data), B, rs.dev->stream);
            }
            CUDA_CHECK(cudaSetDevice(ec_.primary().device));

            // 2. Append accepted target captures to MTP context KV cache + update f_rnn in ONE batched GPU pass!
            std::vector<std::uint32_t> continuing_lanes;
            std::vector<int> cont_cap_offsets;
            std::vector<int> cont_counts;
            std::vector<std::uint32_t> cont_start_pos;
            continuing_lanes.reserve(static_cast<std::size_t>(B));
            for (int b = 0; b < B; ++b) {
                const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
                if (cancelled[static_cast<std::size_t>(b)] != 0 ||
                    terminal[static_cast<std::size_t>(b)] != 0) {
                    clear_lane(l);
                    continue;
                }
                auto& st                = lanes_[l];
                const std::uint32_t acc = accepted_tokens[static_cast<std::size_t>(b)];
                continuing_lanes.push_back(l);
                cont_cap_offsets.push_back(b * V);
                cont_counts.push_back(static_cast<int>(acc));
                cont_start_pos.push_back(st.seq_len);
                st.seq_len += acc;
                st.last_token = pending_tokens_[static_cast<std::size_t>(b * V + static_cast<int>(acc) - 1)];
                st.has_draft  = false;
            }
            if (!continuing_lanes.empty()) {
                append_mtp_ctx_from_caps_batch(continuing_lanes, cont_cap_offsets, cont_counts,
                                               cont_start_pos);
            }

            // 3. Draft next blocks for all continuing lanes in ONE batched DSpark call (Graph captured!)
            if (!continuing_lanes.empty()) {
                run_dspark_draft_batch(continuing_lanes);
                for (const std::uint32_t l : continuing_lanes) { lanes_[l].has_draft = true; }
            }
        } else {
            std::vector<std::uint32_t> mtp_lanes;
            std::vector<int> mtp_offsets;
            std::vector<int> mtp_counts;
            std::vector<std::uint32_t> mtp_starts;
            for (int b = 0; b < B; ++b) {
                const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
                if (cancelled[static_cast<std::size_t>(b)] != 0 ||
                    terminal[static_cast<std::size_t>(b)] != 0) {
                    clear_lane(l);
                    continue;
                }
                auto& st = lanes_[l];
                if (mtp_enabled_) {
                    mtp_lanes.push_back(l);
                    mtp_offsets.push_back(b);
                    mtp_counts.push_back(1);
                    mtp_starts.push_back(st.seq_len);
                }
                st.seq_len += 1;
                st.last_token = pending_tokens_[static_cast<std::size_t>(b)];
                st.has_draft  = false;
            }
            if (mtp_enabled_ && !mtp_lanes.empty()) {
                append_mtp_ctx_from_caps_batch(mtp_lanes, mtp_offsets, mtp_counts, mtp_starts);
            }
        }

        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        for (const std::uint32_t l : lanes) {
            if (lanes_[l].active) { lanes_[l].timings.decode_seconds += dt; }
        }
    }

    void abort_lane(std::uint32_t lane) noexcept { clear_lane(lane); }
    bool has_retained_lane(std::uint32_t) const noexcept { return false; }
    void evict_retained_lane(std::uint32_t) {}

    GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept {
        return lane < max_concurrency_ ? lanes_[lane].timings : GenerationTimings{};
    }

    const SpeculativeStats& speculative_stats_lane(std::uint32_t lane) const noexcept {
        return lanes_[lane].spec_stats;
    }

    MemorySummary memory_summary() const noexcept {
        MemorySummary out;
        out.device                   = ec_.primary().device;
        out.max_context              = max_context_;
        out.effective_max_context    = kMaxContextNative;
        out.kv_capacity              = plan_.spec.main_page_groups * static_cast<std::uint32_t>(kPagedKVPageSize);
        out.kv_cache                 = KvCacheStorage::BFloat16;
        out.sequence.capacity_bytes  = plan_.seq_regions.total_bytes;
        out.sequence.used_bytes      = plan_.seq_regions.total_bytes;
        out.sequence.peak_used_bytes = plan_.seq_regions.total_bytes;
        out.workspace.capacity_bytes = plan_.ws_regions.total_bytes;
        out.workspace.peak_used_bytes = plan_.ws_regions.total_bytes;
        out.kv_payload_bytes         = plan_.seq_regions.main_kv.payload_bytes();
        out.cuda_graph_allowance_bytes = use_cuda_graph_ ? kGraphAllowanceBytes : 0;
        out.cuda_graph_node_count    = last_graph_nodes_;
        return out;
    }

    void reset_memory_peaks() noexcept {}

    std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return captured_logits_bf16_;
    }

    void enable_logits_capture(bool enabled) {
        logits_capture_enabled_ = enabled;
        if (!enabled) { captured_logits_bf16_.clear(); }
    }

    void enable_peer_egress_check(bool) noexcept {}
    std::uint64_t peer_egress_check_rounds() const noexcept { return 0; }
    std::uint64_t peer_egress_check_mismatches() const noexcept { return 0; }

private:
    enum class ForwardMode { Prefill, Decode, Verify };

    struct LaneState {
        bool active             = false;
        bool prefilling         = false;
        bool has_draft          = false;
        std::vector<TokenId> prompt_tokens;
        runtime::ResolvedExecutionOptions options;
        std::uint32_t reserved_pages = 0;
        std::uint32_t prefill_cursor = 0;
        std::uint32_t seq_len        = 0;
        TokenId last_token           = 0;
        std::array<TokenId, 7> draft_tokens{};
        std::array<PagedKVAllocation, 2> kv_alloc{};
        std::array<PagedKVAllocation, 2> mtp_kv_alloc{};
        GenerationTimings timings;
        SpeculativeStats spec_stats;
    };

    void clear_lane(std::uint32_t lane) noexcept {
        if (lane >= max_concurrency_) { return; }
        auto& st = lanes_[lane];
        for (int r = 0; r < tp_; ++r) {
            st.kv_alloc[static_cast<std::size_t>(r)].release();
            st.mtp_kv_alloc[static_cast<std::size_t>(r)].release();
        }
        st.active         = false;
        st.prefilling     = false;
        st.has_draft      = false;
        st.prompt_tokens.clear();
        st.prefill_cursor = 0;
        st.seq_len        = 0;
    }

    ops::Inkling2KvTarget make_kv_target(const RankState& rs, int il) const noexcept {
        ops::Inkling2KvTarget kv{};
        const int gidx = global_layer_index(il);
        if (gidx >= 0) {
            kv.is_local          = false;
            kv.paged_k           = rs.main_kv->plane(static_cast<std::size_t>(gidx * 2 + 0)).data;
            kv.paged_v           = rs.main_kv->plane(static_cast<std::size_t>(gidx * 2 + 1)).data;
            kv.block_tables      = static_cast<const std::int32_t*>(rs.main_kv->block_tables().data);
            kv.max_logical_pages = static_cast<std::int32_t>(rs.main_kv->logical_page_capacity());
        } else {
            const int lidx = local_layer_index(il);
            kv.is_local    = true;
            kv.ring_k      = rs.swa_k[static_cast<std::size_t>(lidx)].data;
            kv.ring_v      = rs.swa_v[static_cast<std::size_t>(lidx)].data;
            kv.ring_cap    = kSwaRingCap;
        }
        return kv;
    }

    ops::Inkling2KvTarget make_mtp_kv_target(const RankState& rs, int mtp_layer) const noexcept {
        ops::Inkling2KvTarget kv{};
        if (mtp_layer == 1) {
            kv.is_local          = false;
            kv.paged_k           = rs.mtp_global_kv->plane(0).data;
            kv.paged_v           = rs.mtp_global_kv->plane(1).data;
            kv.block_tables      = static_cast<const std::int32_t*>(rs.mtp_global_kv->block_tables().data);
            kv.max_logical_pages = static_cast<std::int32_t>(rs.mtp_global_kv->logical_page_capacity());
        } else {
            const int lidx = (mtp_layer == 0) ? 0 : 1;
            kv.is_local    = true;
            kv.ring_k      = rs.mtp_swa_k[static_cast<std::size_t>(lidx)].data;
            kv.ring_v      = rs.mtp_swa_v[static_cast<std::size_t>(lidx)].data;
            kv.ring_cap    = kSwaRingCap;
        }
        return kv;
    }

    void allreduce_bf16(std::array<Tensor, 2>& bufs, std::array<Tensor, 2>& staging) {
        if (tp_ == 1) { return; }
        ops::allreduce_sum(std::span<const Tensor>(bufs.data(), 2),
                           std::span<const Tensor>(staging.data(), 2), ec_, *peer_events_);
    }

    // ---- Complete 48-logical-layer forward pass (Prefill / Decode / Verify) ----
    // Capturable cleanly into a single cross-device CUDA Graph when mode is Decode or Verify!
    void run_forward_body(ForwardMode mode, int batch_size, int q_per_lane, int prefill_lane,
                          int prefill_start_pos, bool capture_mtp) {
        const int T = (mode == ForwardMode::Prefill) ? q_per_lane : (batch_size * q_per_lane);
        const std::size_t lane_sconv_stride = plan_.seq_regions.lane_sconv_floats;

        auto slice_t = [&](const Tensor& base, std::int32_t ne0) {
            return Tensor(base.data, base.dtype, {ne0, T});
        };

        std::array<Tensor, 2> cur_pair{}, ar_pair{}, attn_pair{}, moe_pair{};
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cur_pair[r]    = slice_t(rs.cur, kHiddenSize);
            ar_pair[r]     = slice_t(rs.ar_staging, kHiddenSize);
            attn_pair[r]   = slice_t(rs.attn_out, kHiddenSize);
            moe_pair[r]    = slice_t(rs.moe_out, kHiddenSize);
        }

        // 1. Embedding gather + TP2 AllReduce + embed_norm
        for (int r = 0; r < tp_; ++r) {
            auto& rs = ranks_[static_cast<std::size_t>(r)];
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            Tensor tids(rs.round_token_ids.data, DType::I32, {T});
            ops::inkling2_embed_gather_shard(tids, rs.model->embed, rs.model->vocab_off,
                                             rs.model->vocab_rows, cur_pair[r], rs.dev->stream);
        }
        allreduce_bf16(cur_pair, ar_pair);
        for (int r = 0; r < tp_; ++r) {
            auto& rs = ranks_[static_cast<std::size_t>(r)];
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            ops::inkling2_rmsnorm(cur_pair[r], &rs.model->embed_norm, cur_pair[r], kRmsNormEps,
                                  rs.dev->stream);
        }

        // Track block residual stack (same logical count across both ranks)
        int num_blocks = 0;

        auto run_logical_layer = [&](int il, int ip, bool has_messages, int core_idx) {
            // Optional MTP capture BEFORE logical layers 5, 23, 41
            if (capture_mtp && (il == 5 || il == 23 || il == 41)) {
                const int slot = (il == 5) ? 0 : (il == 23 ? 1 : 2);
                for (int r = 0; r < tp_; ++r) {
                    auto& rs = ranks_[static_cast<std::size_t>(r)];
                    CUDA_CHECK(cudaSetDevice(rs.dev->device));
                    Tensor caps(rs.mtp_target_caps.data, DType::BF16, {kHiddenSize, 4, T});
                    ops::inkling2_mtp_capture_slot(cur_pair[r], caps, slot, rs.dev->stream);
                }
            }

            const bool is_local = is_local_phys_layer(ip);
            const std::size_t kv_dim_loc =
                static_cast<std::size_t>(ranks_[0].model->num_kv_heads) * kHeadDim;
            const std::size_t layer_sconv_off =
                static_cast<std::size_t>(il) * 3ULL * (2 * kv_dim_loc + 2 * kHiddenSize);
            const std::size_t k_off    = layer_sconv_off;
            const std::size_t v_off    = k_off + 3 * kv_dim_loc;
            const std::size_t attn_off = v_off + 3 * kv_dim_loc;
            const std::size_t mlp_off  = attn_off + 3 * kHiddenSize;

            // --- Part A: Pre-Attention (AttnRes, Cross-Pass Control, Input Norm, Projections, SConv, Attention, O-proj) ---
            for (int r = 0; r < tp_; ++r) {
                auto& rs     = ranks_[static_cast<std::size_t>(r)];
                const auto& lw = rs.model->layers[static_cast<std::size_t>(ip)];
                cudaStream_t s = rs.dev->stream;
                CUDA_CHECK(cudaSetDevice(rs.dev->device));

                Tensor q_in_t   = slice_t(rs.q_in, kHiddenSize);
                Tensor k_in_t   = slice_t(rs.k_in, kHiddenSize);
                Tensor v_in_t   = slice_t(rs.v_in, kHiddenSize);

                // Block AttnRes before attention + optional cross-pass messages & loop control + input RMSNorm
                const void* srcs[5]{};
                for (int bi = 0; bi < num_blocks; ++bi) {
                    srcs[bi] = rs.blocks[static_cast<std::size_t>(bi)].data;
                }
                srcs[num_blocks] = rs.cur.data;
                const int n_srcs = num_blocks + 1;

                const Tensor* k_src_proj = &q_in_t;
                const Tensor* v_src_proj = &q_in_t;

                if (ip == 0) {
                    ops::inkling2_rmsnorm(cur_pair[r], &lw.attn_norm, q_in_t, kRmsNormEps, s);
                } else if (!has_messages) {
                    if (is_local) {
                        const void* nw[1] = {lw.attn_norm.data};
                        void* outs[1]     = {q_in_t.data};
                        ops::inkling2_attn_res_mix(srcs, n_srcs, lw.attn_res_attn_query, nw, outs,
                                                   1, kHiddenSize, T, kRmsNormEps, s);
                    } else {
                        const void* nw[3] = {lw.q_input_norm.data, lw.k_input_norm.data,
                                             lw.v_input_norm.data};
                        void* outs[3]     = {q_in_t.data, k_in_t.data, v_in_t.data};
                        ops::inkling2_attn_res_mix(srcs, n_srcs, lw.attn_res_attn_query, nw, outs,
                                                   3, kHiddenSize, T, kRmsNormEps, s);
                        k_src_proj = &k_in_t;
                        v_src_proj = &v_in_t;
                    }
                } else {
                    // Pass 1 or 2: Paired Cross-Pass Messages + Pre-Attn AttnRes in ONE launch -> batched Q/K/V loop controllers!
                    const void* bank_srcs[2] = {rs.bank0.data, rs.bank_out.data};
                    void* msg_outs[4]        = {rs.msgs[0].data, rs.msgs[1].data, rs.msgs[2].data,
                                                rs.msgs[3].data};
                    Tensor cp_q(static_cast<std::byte*>(rs.cross_pass_4q.data) +
                                    static_cast<std::size_t>(core_idx) * 4 * kHiddenSize * sizeof(float),
                                DType::FP32, {kHiddenSize, 4});
                    Tensor msg0      = slice_t(rs.msgs[0], kHiddenSize);
                    Tensor msg1      = slice_t(rs.msgs[1], kHiddenSize);
                    Tensor msg2      = slice_t(rs.msgs[2], kHiddenSize);
                    Tensor ctrl_z_3t(rs.ctrl_z.data, DType::BF16, {2 * kHiddenSize, 3 * T});
                    Tensor ctrl_h_3t(rs.ctrl_h.data, DType::BF16, {kGateRank, 3 * T});
                    Tensor ctrl_gf_3t(rs.ctrl_gf.data, DType::BF16, {2 * kHiddenSize, 3 * T});
                    if (is_local) {
                        void* outs[1] = {rs.mlp_in.data}; // temp unnormed mixed
                        ops::inkling2_attn_res_mix_pair(bank_srcs, 2, cp_q, msg_outs, 4,
                                                        srcs, n_srcs, lw.attn_res_attn_query,
                                                        nullptr, outs, 1, kHiddenSize, T,
                                                        kRmsNormEps, s);
                        Tensor mixed = slice_t(rs.mlp_in, kHiddenSize);
                        ops::inkling2_loop_control_qkv(
                            mixed, mixed, mixed, msg0, msg1, msg2,
                            lw.ctrl_attn.down, lw.ctrl_k.down, lw.ctrl_v.down,
                            lw.ctrl_attn.gain_forget, lw.ctrl_k.gain_forget, lw.ctrl_v.gain_forget,
                            &lw.attn_norm, &lw.attn_norm, &lw.attn_norm,
                            ctrl_z_3t, ctrl_h_3t, ctrl_gf_3t, q_in_t, k_in_t, v_in_t, kRmsNormEps, s);
                    } else {
                        void* outs[3] = {rs.q_in.data, rs.k_in.data, rs.v_in.data};
                        ops::inkling2_attn_res_mix_pair(bank_srcs, 2, cp_q, msg_outs, 4,
                                                        srcs, n_srcs, lw.attn_res_attn_query,
                                                        nullptr, outs, 3, kHiddenSize, T,
                                                        kRmsNormEps, s);
                        Tensor tmp_q = slice_t(rs.mlp_in, kHiddenSize);
                        Tensor tmp_k = slice_t(rs.attn_out, kHiddenSize);
                        Tensor tmp_v = slice_t(rs.moe_out, kHiddenSize);
                        ops::inkling2_loop_control_qkv(
                            q_in_t, k_in_t, v_in_t, msg0, msg1, msg2,
                            lw.ctrl_attn.down, lw.ctrl_k.down, lw.ctrl_v.down,
                            lw.ctrl_attn.gain_forget, lw.ctrl_k.gain_forget, lw.ctrl_v.gain_forget,
                            &lw.q_input_norm, &lw.k_input_norm, &lw.v_input_norm,
                            ctrl_z_3t, ctrl_h_3t, ctrl_gf_3t, tmp_q, tmp_k, tmp_v, kRmsNormEps, s);
                        q_in_t = tmp_q;
                        k_in_t = tmp_k;
                        v_in_t = tmp_v;
                    }
                    k_src_proj = &k_in_t;
                    v_src_proj = &v_in_t;
                }

                // Projections: Q/gate/R/sink + K + V fused into ONE launch (zero tail wave!)
                const std::int32_t qgrs_n = 6552 / tp_;
                const std::int32_t kv_n   = rs.model->num_kv_heads * kHeadDim;
                const std::int32_t q_n    = rs.model->num_q_heads * kHeadDim;
                Tensor qgrs_t = slice_t(rs.q_gate_r_sink, qgrs_n);
                Tensor k_t    = slice_t(rs.k_buf, kv_n);
                Tensor v_t    = slice_t(rs.v_buf, kv_n);
                ops::inkling2_bf16_linear_triple(q_in_t, lw.q_gate_r_sink, qgrs_t,
                                                 *k_src_proj, lw.k_proj, k_t,
                                                 *v_src_proj, lw.v_proj, v_t, s);

                // K/V causal short convolutions (W = 4) — fused K+V pair kernels!
                float* sconv_base  = static_cast<float*>(rs.sconv_state.data);
                float* verify_base = mtp_enabled_ ? static_cast<float*>(rs.sconv_verify.data) : nullptr;
                const auto* batch_lanes = static_cast<const std::int32_t*>(rs.round_batch_lanes.data);
                const auto* tok_lanes   = static_cast<const std::int32_t*>(rs.round_tok_lanes.data);
                const auto* positions   = static_cast<const std::int32_t*>(rs.round_positions.data);

                if (is_local) {
                    if (mode == ForwardMode::Prefill) {
                        float* lane_st = sconv_base + static_cast<std::size_t>(prefill_lane) * lane_sconv_stride;
                        ops::inkling2_sconv_static_prefill(k_t, lw.k_sconv, lane_st + k_off, k_t, nullptr, s);
                        ops::inkling2_sconv_static_prefill(v_t, lw.v_sconv, lane_st + v_off, v_t, nullptr, s);
                    } else if (mode == ForwardMode::Decode) {
                        ops::inkling2_sconv_kv_static_decode(k_t, lw.k_sconv, sconv_base + k_off,
                                                             v_t, lw.v_sconv, sconv_base + v_off,
                                                             batch_lanes, batch_size, k_t, v_t, s,
                                                             lane_sconv_stride);
                    } else {
                        ops::inkling2_sconv_kv_static_verify(
                            k_t, lw.k_sconv, sconv_base + k_off, verify_base + k_off,
                            v_t, lw.v_sconv, sconv_base + v_off, verify_base + v_off,
                            batch_lanes, batch_size, q_per_lane, k_t, v_t, s,
                            lane_sconv_stride, kMtpMaxBlockSize);
                    }
                } else {
                    Tensor k_dyn_t = slice_t(rs.k_dyn, 4 * rs.model->num_kv_heads);
                    Tensor v_dyn_t = slice_t(rs.v_dyn, 4 * rs.model->num_kv_heads);
                    ops::inkling2_bf16_linear_pair(*k_src_proj, lw.k_sconv_proj, k_dyn_t,
                                                   *v_src_proj, lw.v_sconv_proj, v_dyn_t, s);
                    if (mode == ForwardMode::Prefill) {
                        float* lane_st = sconv_base + static_cast<std::size_t>(prefill_lane) * lane_sconv_stride;
                        ops::inkling2_sconv_dynamic_prefill(k_t, k_dyn_t, lw.k_sconv,
                                                            rs.model->num_kv_heads, lane_st + k_off,
                                                            k_t, s);
                        ops::inkling2_sconv_dynamic_prefill(v_t, v_dyn_t, lw.v_sconv,
                                                            rs.model->num_kv_heads, lane_st + v_off,
                                                            v_t, s);
                    } else if (mode == ForwardMode::Decode) {
                        ops::inkling2_sconv_kv_dynamic_decode(
                            k_t, k_dyn_t, lw.k_sconv, sconv_base + k_off,
                            v_t, v_dyn_t, lw.v_sconv, sconv_base + v_off,
                            rs.model->num_kv_heads, batch_lanes, batch_size, k_t, v_t, s,
                            lane_sconv_stride);
                    } else {
                        ops::inkling2_sconv_kv_dynamic_verify(
                            k_t, k_dyn_t, lw.k_sconv, sconv_base + k_off, verify_base + k_off,
                            v_t, v_dyn_t, lw.v_sconv, sconv_base + v_off, verify_base + v_off,
                            rs.model->num_kv_heads, batch_lanes, batch_size, q_per_lane, k_t, v_t,
                            s, lane_sconv_stride, kMtpMaxBlockSize);
                    }
                }

                // Attention prologue: Q/R L2-clip (+tau), K L2-clip + V RMSNorm -> KV cache, R@proj -> rel_bias
                const ops::Inkling2KvTarget kv = make_kv_target(rs, il);
                Tensor q_clip_t(rs.q_clip.data, DType::BF16, {kHeadDim, rs.model->num_q_heads, T});
                Tensor rb_t(rs.rel_bias_f32.data, DType::FP32,
                            {kRelExtent, rs.model->num_q_heads, T});
                ops::inkling2_attn_prologue(qgrs_t, k_t, v_t, lw.q_clip_scale, lw.rel_clip_scale,
                                            lw.rel_logits_proj, positions, tok_lanes, T,
                                            rs.model->num_q_heads, rs.model->num_kv_heads, kv,
                                            is_local, kQkClipThreshold, kRelClipThreshold,
                                            kLogScalingFloor, kLogScalingAlpha, kRmsNormEps,
                                            q_clip_t, rb_t, s);

                // Attention kernel + fused sigmoid(gate)
                Tensor aval_t = slice_t(rs.attn_val, q_n);
                if (mode == ForwardMode::Prefill) {
                    ops::inkling2_attn_prefill(q_clip_t, qgrs_t, rb_t, kv, prefill_lane,
                                               prefill_start_pos, T, rs.model->num_q_heads,
                                               rs.model->num_kv_heads, kSwaWindow, aval_t, s);
                } else {
                    ops::inkling2_attn_decode_verify(
                        q_clip_t, qgrs_t, rb_t, kv, batch_lanes, positions, batch_size, q_per_lane,
                        rs.model->num_q_heads, rs.model->num_kv_heads, kSwaWindow,
                        static_cast<float*>(rs.split_scratch.data), 16, aval_t, s);
                }

                // O projection (row-parallel)
                ops::inkling2_bf16_linear(aval_t, lw.o_proj, attn_pair[r], s);
            }

            // TP2 AllReduce + attn_sconv + block copy / residual (FUSED into ONE kernel on TP2 direct-P2P Decode!)
            const bool is_block_boundary = (ip % kAttnResBlockSize == 0);
            const bool can_fuse_tp2_ar_sconv =
                (mode == ForwardMode::Decode && tp_ == 2 && peer_events_.has_value() &&
                 peer_events_->packet_storage(0) != nullptr &&
                 attn_pair[0].bytes() <= peer_events_->packet_capacity_bytes() &&
                 (attn_pair[0].bytes() % 16 == 0));

            if (can_fuse_tp2_ar_sconv) {
                for (int r = 0; r < 2; ++r) {
                    auto& rs       = ranks_[static_cast<std::size_t>(r)];
                    const auto& lw = rs.model->layers[static_cast<std::size_t>(ip)];
                    CUDA_CHECK(cudaSetDevice(rs.dev->device));
                    Tensor ffn_t   = slice_t(rs.ffn_inp, kHiddenSize);
                    Tensor blk_dst = slice_t(rs.blocks[static_cast<std::size_t>(num_blocks)], kHiddenSize);
                    const Tensor* res_add = is_block_boundary ? nullptr : &cur_pair[r];
                    const Tensor* blk_src = is_block_boundary ? &cur_pair[r] : nullptr;
                    Tensor* blk_dst_ptr   = is_block_boundary ? &blk_dst : nullptr;
                    ops::inkling2_tp2_packet_allreduce_sconv_decode(
                        attn_pair[r], peer_events_->packet_storage(r),
                        peer_events_->packet_storage(1 - r), peer_events_->packet_capacity_bytes(),
                        lw.attn_sconv, static_cast<float*>(rs.sconv_state.data) + attn_off,
                        static_cast<const std::int32_t*>(rs.round_batch_lanes.data), batch_size,
                        lane_sconv_stride, ffn_t, res_add, blk_src, blk_dst_ptr, rs.dev->stream);
                }
            } else {
                allreduce_bf16(attn_pair, ar_pair);
            }

            // --- Part B: (unfused attn_sconv if needed) + Pre-MoE AttnRes/Control + Router + MoE ---
            for (int r = 0; r < tp_; ++r) {
                auto& rs     = ranks_[static_cast<std::size_t>(r)];
                const auto& lw = rs.model->layers[static_cast<std::size_t>(ip)];
                cudaStream_t s = rs.dev->stream;
                CUDA_CHECK(cudaSetDevice(rs.dev->device));

                Tensor ffn_t  = slice_t(rs.ffn_inp, kHiddenSize);
                Tensor mlp_t  = slice_t(rs.mlp_in, kHiddenSize);
                float* sconv_base  = static_cast<float*>(rs.sconv_state.data);
                float* verify_base = mtp_enabled_ ? static_cast<float*>(rs.sconv_verify.data) : nullptr;
                const auto* batch_lanes = static_cast<const std::int32_t*>(rs.round_batch_lanes.data);

                if (!can_fuse_tp2_ar_sconv) {
                    if (is_block_boundary) {
                        // Save cur (= inpSA) into blocks[num_blocks]; ffn_inp = attn_sconv(attn_out)
                        CUDA_CHECK(cudaMemcpyAsync(rs.blocks[static_cast<std::size_t>(num_blocks)].data,
                                                   rs.cur.data,
                                                   static_cast<std::size_t>(T) * kHiddenSize * 2,
                                                   cudaMemcpyDeviceToDevice, s));
                    }
                    const Tensor* res_add = is_block_boundary ? nullptr : &cur_pair[r];
                    if (mode == ForwardMode::Prefill) {
                        float* lane_st = sconv_base + static_cast<std::size_t>(prefill_lane) * lane_sconv_stride;
                        ops::inkling2_sconv_static_prefill(attn_pair[r], lw.attn_sconv,
                                                           lane_st + attn_off, ffn_t, res_add, s);
                    } else if (mode == ForwardMode::Decode) {
                        ops::inkling2_sconv_static_decode(attn_pair[r], lw.attn_sconv,
                                                          sconv_base + attn_off, batch_lanes,
                                                          batch_size, ffn_t, res_add, s,
                                                          lane_sconv_stride);
                    } else {
                        ops::inkling2_sconv_static_verify(attn_pair[r], lw.attn_sconv,
                                                          sconv_base + attn_off, verify_base + attn_off,
                                                          batch_lanes, batch_size, q_per_lane, ffn_t,
                                                          res_add, s, lane_sconv_stride,
                                                          kMtpMaxBlockSize);
                    }
                }

                const int cur_blocks = num_blocks + (is_block_boundary ? 1 : 0);
                const void* m_srcs[5]{};
                for (int bi = 0; bi < cur_blocks; ++bi) {
                    m_srcs[bi] = rs.blocks[static_cast<std::size_t>(bi)].data;
                }
                m_srcs[cur_blocks] = rs.ffn_inp.data;

                if (!has_messages) {
                    const void* nw[1] = {lw.mlp_norm.data};
                    void* outs[1]     = {mlp_t.data};
                    ops::inkling2_attn_res_mix(m_srcs, cur_blocks + 1, lw.attn_res_mlp_query, nw,
                                               outs, 1, kHiddenSize, T, kRmsNormEps, s);
                } else {
                    void* outs[1] = {rs.q_in.data}; // temp unnormed mlp_in
                    ops::inkling2_attn_res_mix(m_srcs, cur_blocks + 1, lw.attn_res_mlp_query,
                                               nullptr, outs, 1, kHiddenSize, T, kRmsNormEps, s);
                    Tensor q_tmp    = slice_t(rs.q_in, kHiddenSize);
                    Tensor msg3     = slice_t(rs.msgs[3], kHiddenSize);
                    Tensor ctrl_z_t = slice_t(rs.ctrl_z, 2 * kHiddenSize);
                    Tensor ctrl_h_t = slice_t(rs.ctrl_h, kGateRank);
                    Tensor ctrl_gf_t = slice_t(rs.ctrl_gf, 2 * kHiddenSize);
                    ops::inkling2_loop_control(q_tmp, msg3, lw.ctrl_mlp.down,
                                               lw.ctrl_mlp.gain_forget, &lw.mlp_norm, ctrl_z_t,
                                               ctrl_h_t, ctrl_gf_t, mlp_t, kRmsNormEps, s);
                }

                // Fused Router (GEMV + sigmoid + bias top-8 + grouping in ONE launch via Last-CTA sync!)
                ops::inkling2_moe_router_fused(mlp_t, lw.router_weight, lw.router_bias,
                                               kRouteScale, rs.moe_sc, s);

                // Sparse MoE (NVFP4 layers 0..19, MXFP8 layers 20..23)
                if (lw.is_nvfp4) {
                    ops::inkling2_moe_nvfp4_forward(mlp_t, lw.w13_codes, lw.w13_scales, lw.w13_amax,
                                                    lw.w13_denom, lw.w2_codes, lw.w2_scales,
                                                    lw.w2_amax, lw.w2_denom, rs.model->moe_slabs,
                                                    kMlpGateSoftcap, kMlpUpSoftcap, rs.moe_sc,
                                                    moe_pair[r], s);
                } else {
                    ops::inkling2_moe_mxfp8_forward(mlp_t, lw.w13_codes, lw.w13_scales, lw.w2_codes,
                                                    lw.w2_scales, rs.model->moe_slabs,
                                                    kMlpGateSoftcap, kMlpUpSoftcap, rs.moe_sc,
                                                    moe_pair[r], s);
                }
            }
            if (is_block_boundary) { ++num_blocks; }

            // TP2 AllReduce + mlp_sconv + fused residual add with ffn_inp -> cur (FUSED on TP2 direct-P2P Decode!)
            if (can_fuse_tp2_ar_sconv) {
                for (int r = 0; r < 2; ++r) {
                    auto& rs       = ranks_[static_cast<std::size_t>(r)];
                    const auto& lw = rs.model->layers[static_cast<std::size_t>(ip)];
                    CUDA_CHECK(cudaSetDevice(rs.dev->device));
                    Tensor ffn_t = slice_t(rs.ffn_inp, kHiddenSize);
                    ops::inkling2_tp2_packet_allreduce_sconv_decode(
                        moe_pair[r], peer_events_->packet_storage(r),
                        peer_events_->packet_storage(1 - r), peer_events_->packet_capacity_bytes(),
                        lw.mlp_sconv, static_cast<float*>(rs.sconv_state.data) + mlp_off,
                        static_cast<const std::int32_t*>(rs.round_batch_lanes.data), batch_size,
                        lane_sconv_stride, cur_pair[r], &ffn_t, nullptr, nullptr, rs.dev->stream);
                }
            } else {
                allreduce_bf16(moe_pair, ar_pair);

                // --- Part C: mlp_sconv + fused residual add with ffn_inp -> cur ---
                for (int r = 0; r < tp_; ++r) {
                    auto& rs     = ranks_[static_cast<std::size_t>(r)];
                    const auto& lw = rs.model->layers[static_cast<std::size_t>(ip)];
                    cudaStream_t s = rs.dev->stream;
                    CUDA_CHECK(cudaSetDevice(rs.dev->device));

                    Tensor ffn_t = slice_t(rs.ffn_inp, kHiddenSize);
                    float* sconv_base  = static_cast<float*>(rs.sconv_state.data);
                    float* verify_base = mtp_enabled_ ? static_cast<float*>(rs.sconv_verify.data) : nullptr;
                    const auto* batch_lanes = static_cast<const std::int32_t*>(rs.round_batch_lanes.data);

                    if (mode == ForwardMode::Prefill) {
                        float* lane_st = sconv_base + static_cast<std::size_t>(prefill_lane) * lane_sconv_stride;
                        ops::inkling2_sconv_static_prefill(moe_pair[r], lw.mlp_sconv, lane_st + mlp_off,
                                                           cur_pair[r], &ffn_t, s);
                    } else if (mode == ForwardMode::Decode) {
                        ops::inkling2_sconv_static_decode(moe_pair[r], lw.mlp_sconv,
                                                          sconv_base + mlp_off, batch_lanes, batch_size,
                                                          cur_pair[r], &ffn_t, s, lane_sconv_stride);
                    } else {
                        ops::inkling2_sconv_static_verify(moe_pair[r], lw.mlp_sconv,
                                                          sconv_base + mlp_off, verify_base + mlp_off,
                                                          batch_lanes, batch_size, q_per_lane,
                                                          cur_pair[r], &ffn_t, s, lane_sconv_stride,
                                                          kMtpMaxBlockSize);
                    }
                }
            }
        };

        // ---- Execute 48 logical layers ----
        int il = 0;
        // 1. Prelude: ip = 0..5
        for (int ip = 0; ip < kCoreStart; ++ip) { run_logical_layer(il++, ip, false, 0); }

        // Save prelude state (cur -> prelude; blocks[0] stays invariant across passes!)
        for (int r = 0; r < tp_; ++r) {
            auto& rs = ranks_[static_cast<std::size_t>(r)];
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            CUDA_CHECK(cudaMemcpyAsync(rs.prelude.data, rs.cur.data,
                                       static_cast<std::size_t>(T) * kHiddenSize * 2,
                                       cudaMemcpyDeviceToDevice, rs.dev->stream));
        }

        // 2. Looped Core: 3 passes over ip = 6..17
        for (int pass = 0; pass < kNumPasses; ++pass) {
            if (pass > 0) {
                // bank = [blocks[2]], bank_out = cur; reset cur = prelude, num_blocks = 1
                for (int r = 0; r < tp_; ++r) {
                    auto& rs = ranks_[static_cast<std::size_t>(r)];
                    CUDA_CHECK(cudaSetDevice(rs.dev->device));
                    const std::size_t bytes = static_cast<std::size_t>(T) * kHiddenSize * 2;
                    CUDA_CHECK(cudaMemcpyAsync(rs.bank0.data, rs.blocks[2].data, bytes,
                                               cudaMemcpyDeviceToDevice, rs.dev->stream));
                    CUDA_CHECK(cudaMemcpyAsync(rs.bank_out.data, rs.cur.data, bytes,
                                               cudaMemcpyDeviceToDevice, rs.dev->stream));
                    CUDA_CHECK(cudaMemcpyAsync(rs.cur.data, rs.prelude.data, bytes,
                                               cudaMemcpyDeviceToDevice, rs.dev->stream));
                }
                num_blocks = 1;
            }
            for (int ip = kCoreStart; ip < kCoreEnd; ++ip) {
                run_logical_layer(il++, ip, pass > 0, ip - kCoreStart);
            }
        }

        // 3. Epilogue: ip = 18..23
        for (int ip = kCoreEnd; ip < kNumPhysLayers; ++ip) { run_logical_layer(il++, ip, false, 0); }

        // 4. Final Output AttnRes + Final RMSNorm (fused!) + optional MTP capture slot 3 (layer 48)
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));

            const void* srcs[5] = {rs.blocks[0].data, rs.blocks[1].data, rs.blocks[2].data,
                                   rs.blocks[3].data, rs.cur.data};
            const void* nw[1]   = {rs.model->final_norm.data};
            void* outs[1]       = {rs.final_h.data};
            ops::inkling2_attn_res_mix(srcs, 5, rs.model->attn_res_out_query, nw, outs, 1,
                                       kHiddenSize, T, kRmsNormEps, s);

            if (capture_mtp) {
                Tensor fh(rs.final_h.data, DType::BF16, {kHiddenSize, T});
                Tensor caps(rs.mtp_target_caps.data, DType::BF16, {kHiddenSize, 4, T});
                ops::inkling2_mtp_capture_slot(fh, caps, 3, s);
            }
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
    }

    // ---- Output Head + Greedy Argmax (replicated to round_winner_ids on rank 0/1) ----
    void run_unembed_greedy(int num_rows, int src_row_offset) {
        std::array<Tensor, 2> vals{}, ids{}, outs{};
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            Tensor h_in(static_cast<std::byte*>(rs.final_h.data) +
                            static_cast<std::size_t>(src_row_offset) * kHiddenSize * 2,
                        DType::BF16, {kHiddenSize, num_rows});
            vals[r] = Tensor(rs.round_local_max_val.data, DType::FP32, {1, num_rows});
            ids[r]  = Tensor(rs.round_local_max_id.data, DType::I32, {1, num_rows});
            outs[r] = Tensor(rs.round_winner_ids.data, DType::I32, {num_rows});

            // Fused unembed + exact monotonic softcap argmax (zero FP32 logits written for ALL T!)
            ops::inkling2_unembed_project(h_in, rs.model->output_head, kFinalLogitSoftcap,
                                          rs.model->valid_rows, nullptr, &vals[r], &ids[r], s);
            if (tp_ == 1) {
                CUDA_CHECK(cudaMemcpyAsync(outs[0].data, ids[0].data,
                                           static_cast<std::size_t>(num_rows) * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }
        if (tp_ == 2) {
            Tensor gv(ranks_[0].round_gather_vals.data, DType::FP32, {2, num_rows});
            Tensor gi(ranks_[0].round_gather_ids.data, DType::I32, {2, num_rows});
            ops::allreduce_argmax(std::span<const Tensor>(vals.data(), 2),
                                  std::span<const Tensor>(ids.data(), 2),
                                  std::span<const Tensor>(outs.data(), 2), gv, gi,
                                  ranks_[0].model->vocab_rows, ec_, *peer_events_);
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
    }

    // ---- Output Head + Full BF16 Logits + Stochastic Sampling / Logit Capture ----
    void sample_or_capture_logits(std::span<const std::uint32_t> lanes, std::int32_t purpose,
                                  std::vector<TokenId>& out_tokens, int src_row_offset = 0) {
        const int B = static_cast<int>(lanes.size());
        std::array<Tensor, 2> bf16_shards{};
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            Tensor h_in(static_cast<std::byte*>(rs.final_h.data) +
                            static_cast<std::size_t>(src_row_offset) * kHiddenSize * 2,
                        DType::BF16, {kHiddenSize, B});
            Tensor l_f32(rs.logits_shard_f32.data, DType::FP32, {rs.model->vocab_rows, B});
            bf16_shards[r] = Tensor(rs.logits_shard_bf16.data, DType::BF16,
                                    {rs.model->vocab_rows, B});
            ops::inkling2_unembed_project(h_in, rs.model->output_head, kFinalLogitSoftcap,
                                          rs.model->valid_rows, &l_f32, nullptr, nullptr, s);
            ops::cast_fp32_to_bf16(l_f32, bf16_shards[r], s);
        }

        Tensor full_logits(ranks_[0].logits_full_bf16.data, DType::BF16, {kPaddedVocabSize, B});
        if (tp_ == 2) {
            ops::gather_columns_rank0(full_logits, std::span<const Tensor>(bf16_shards.data(), 2),
                                      ec_, *peer_events_);
        } else {
            CUDA_CHECK(cudaSetDevice(ranks_[0].dev->device));
            CUDA_CHECK(cudaMemcpyAsync(full_logits.data, bf16_shards[0].data, full_logits.bytes(),
                                       cudaMemcpyDeviceToDevice, ranks_[0].dev->stream));
        }

        CUDA_CHECK(cudaSetDevice(ranks_[0].dev->device));
        cudaStream_t s0 = ranks_[0].dev->stream;

        if (logits_capture_enabled_ && B == 1) {
            captured_logits_bf16_.resize(kUnpaddedVocab);
            CUDA_CHECK(cudaMemcpyAsync(captured_logits_bf16_.data(), full_logits.data,
                                       kUnpaddedVocab * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, s0));
        }

        std::vector<ops::SamplingConfig> cfgs(static_cast<std::size_t>(B));
        std::vector<std::int32_t> sample_pos(static_cast<std::size_t>(B));
        for (int b = 0; b < B; ++b) {
            const auto& st = lanes_[lanes[static_cast<std::size_t>(b)]];
            const auto& sp = st.options.sampling;
            cfgs[static_cast<std::size_t>(b)] = ops::SamplingConfig{
                .temperature       = sp.temperature,
                .top_k             = sp.top_k,
                .top_p             = sp.top_p,
                .min_p             = sp.min_p,
                .presence_penalty  = sp.presence_penalty,
                .frequency_penalty = sp.frequency_penalty,
                .seed              = sp.seed,
                .token_counts      = static_cast<std::int32_t*>(ranks_[0].token_counts.data) +
                                static_cast<std::size_t>(lanes[static_cast<std::size_t>(b)]) *
                                    kUnpaddedVocab,
            };
            sample_pos[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(st.seq_len);
        }

        CUDA_CHECK(cudaMemcpyAsync(ranks_[0].round_sampling_cfg.data, cfgs.data(),
                                   cfgs.size() * sizeof(ops::SamplingConfig),
                                   cudaMemcpyHostToDevice, s0));
        CUDA_CHECK(cudaMemcpyAsync(ranks_[0].round_sample_pos.data, sample_pos.data(),
                                   sample_pos.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, s0));

        Tensor sampled_t(ranks_[0].round_sampled_ids.data, DType::I32, {B});
        Tensor spos_t(ranks_[0].round_sample_pos.data, DType::I32, {B});
        ranks_[0].sample_ws->reset();
        ops::sample(full_logits, sampled_t, kUnpaddedVocab,
                    static_cast<const ops::SamplingConfig*>(ranks_[0].round_sampling_cfg.data),
                    spos_t, purpose, *ranks_[0].sample_ws, s0);

        out_tokens.resize(static_cast<std::size_t>(B));
        CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), sampled_t.data,
                                   static_cast<std::size_t>(B) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, s0));
        ranks_[0].dev->synchronize();
    }

    // ---- Append `count` target captures starting at `cap_offset` in mtp_target_caps into lane's MTP KV ----
    void append_mtp_ctx_from_caps(std::uint32_t lane, int cap_offset, int count,
                                  std::uint32_t start_pos) {
        if (!mtp_enabled_ || count <= 0) { return; }
        std::vector<std::int32_t> h_pos(static_cast<std::size_t>(count));
        std::vector<std::int32_t> h_lanes(static_cast<std::size_t>(count),
                                          static_cast<std::int32_t>(lane));
        for (int i = 0; i < count; ++i) {
            h_pos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(start_pos) + i;
        }
        auto& st = lanes_[lane];
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            ensure_kv_tokens(st.mtp_kv_alloc[static_cast<std::size_t>(r)],
                             start_pos + static_cast<std::uint32_t>(count), s);
            CUDA_CHECK(cudaMemcpyAsync(rs.round_positions.data, h_pos.data(),
                                       h_pos.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_tok_lanes.data, h_lanes.data(),
                                       h_lanes.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));

            Tensor caps_slice(static_cast<std::byte*>(rs.mtp_target_caps.data) +
                                  static_cast<std::size_t>(cap_offset) * 4 * kHiddenSize * 2,
                              DType::BF16, {kHiddenSize, 4, count});
            Tensor ctx_h(rs.mtp_ctx_h.data, DType::BF16, {kHiddenSize, count});
            Tensor kv_ctx(rs.mtp_kv_ctx.data, DType::BF16,
                          {2 * rs.model->num_kv_heads * kHeadDim, count});

            for (int li = 0; li < kMtpNumLayers; ++li) {
                const auto& ml = rs.model->mtp.layers[static_cast<std::size_t>(li)];
                ops::inkling2_mtp_mix_rmsnorm(caps_slice, rs.model->mtp.fuse_weight, li,
                                              ml.dflare_norm, kRmsNormEps, ctx_h, s);
                ops::inkling2_bf16_linear(ctx_h, ml.kv_ctx, kv_ctx, s);
                ops::inkling2_mtp_ctx_kv_write(
                    kv_ctx, ml.k_norm_ctx, static_cast<const std::int32_t*>(rs.round_positions.data),
                    static_cast<const std::int32_t*>(rs.round_tok_lanes.data), count,
                    rs.model->num_kv_heads, kMtpRopeTheta, kRmsNormEps, make_mtp_kv_target(rs, li),
                    s);
            }

            // Update lane's f_rnn from the last accepted token's capture!
            Tensor last_cap(static_cast<std::byte*>(rs.mtp_target_caps.data) +
                                static_cast<std::size_t>(cap_offset + count - 1) * 4 * kHiddenSize * 2,
                            DType::BF16, {kHiddenSize, 4, 1});
            Tensor frnn_dst(static_cast<std::byte*>(rs.mtp_f_rnn.data) +
                                static_cast<std::size_t>(lane) * kHiddenSize * 2,
                            DType::BF16, {kHiddenSize, 1});
            ops::inkling2_mtp_mix_rmsnorm(last_cap, rs.model->mtp.fuse_weight, 3,
                                          rs.model->mtp.markov_dflare_norm, kRmsNormEps, frnn_dst,
                                          s);
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
    }

    // ---- Batched append of accepted target captures for multiple continuing lanes in ONE GPU pass ----
    void append_mtp_ctx_from_caps_batch(const std::vector<std::uint32_t>& lanes,
                                        const std::vector<int>& cap_offsets,
                                        const std::vector<int>& counts,
                                        const std::vector<std::uint32_t>& start_positions) {
        if (!mtp_enabled_ || lanes.empty()) { return; }
        const int num_lanes = static_cast<int>(lanes.size());
        int total_count     = 0;
        for (int i = 0; i < num_lanes; ++i) { total_count += counts[static_cast<std::size_t>(i)]; }
        if (total_count <= 0) { return; }

        std::vector<std::int32_t> h_src_idx;
        std::vector<std::int32_t> h_pos;
        std::vector<std::int32_t> h_tok_lanes;
        std::vector<std::int32_t> h_last_src;
        std::vector<std::int32_t> h_last_dst;
        h_src_idx.reserve(static_cast<std::size_t>(total_count));
        h_pos.reserve(static_cast<std::size_t>(total_count));
        h_tok_lanes.reserve(static_cast<std::size_t>(total_count));
        h_last_src.reserve(static_cast<std::size_t>(num_lanes));
        h_last_dst.reserve(static_cast<std::size_t>(num_lanes));

        for (int b = 0; b < num_lanes; ++b) {
            const std::uint32_t l   = lanes[static_cast<std::size_t>(b)];
            const int cap_off       = cap_offsets[static_cast<std::size_t>(b)];
            const int cnt           = counts[static_cast<std::size_t>(b)];
            const std::uint32_t sp  = start_positions[static_cast<std::size_t>(b)];
            if (cnt <= 0) { continue; }
            for (int i = 0; i < cnt; ++i) {
                h_src_idx.push_back(cap_off + i);
                h_pos.push_back(static_cast<std::int32_t>(sp) + i);
                h_tok_lanes.push_back(static_cast<std::int32_t>(l));
            }
            h_last_src.push_back(cap_off + cnt - 1);
            h_last_dst.push_back(static_cast<std::int32_t>(l));
        }

        const int active_lanes = static_cast<int>(h_last_src.size());
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));

            for (int b = 0; b < num_lanes; ++b) {
                const std::uint32_t l  = lanes[static_cast<std::size_t>(b)];
                const int cnt          = counts[static_cast<std::size_t>(b)];
                const std::uint32_t sp = start_positions[static_cast<std::size_t>(b)];
                if (cnt > 0) {
                    ensure_kv_tokens(lanes_[l].mtp_kv_alloc[static_cast<std::size_t>(r)],
                                     sp + static_cast<std::uint32_t>(cnt), s);
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(rs.round_mtp_src_idx.data, h_src_idx.data(),
                                       h_src_idx.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_positions.data, h_pos.data(),
                                       h_pos.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_tok_lanes.data, h_tok_lanes.data(),
                                       h_tok_lanes.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_mtp_last_src.data, h_last_src.data(),
                                       h_last_src.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_mtp_last_dst.data, h_last_dst.data(),
                                       h_last_dst.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));

            Tensor ctx_h(rs.mtp_ctx_h.data, DType::BF16, {kHiddenSize, total_count});
            Tensor kv_ctx(rs.mtp_kv_ctx.data, DType::BF16,
                          {2 * rs.model->num_kv_heads * kHeadDim, total_count});

            for (int li = 0; li < kMtpNumLayers; ++li) {
                const auto& ml = rs.model->mtp.layers[static_cast<std::size_t>(li)];
                ops::inkling2_mtp_mix_rmsnorm_indexed(
                    rs.mtp_target_caps,
                    static_cast<const std::int32_t*>(rs.round_mtp_src_idx.data), nullptr,
                    total_count, rs.model->mtp.fuse_weight, li, ml.dflare_norm, kRmsNormEps,
                    ctx_h, s);
                ops::inkling2_bf16_linear(ctx_h, ml.kv_ctx, kv_ctx, s);
                ops::inkling2_mtp_ctx_kv_write(
                    kv_ctx, ml.k_norm_ctx, static_cast<const std::int32_t*>(rs.round_positions.data),
                    static_cast<const std::int32_t*>(rs.round_tok_lanes.data), total_count,
                    rs.model->num_kv_heads, kMtpRopeTheta, kRmsNormEps, make_mtp_kv_target(rs, li),
                    s);
            }

            ops::inkling2_mtp_mix_rmsnorm_indexed(
                rs.mtp_target_caps,
                static_cast<const std::int32_t*>(rs.round_mtp_last_src.data),
                static_cast<const std::int32_t*>(rs.round_mtp_last_dst.data), active_lanes,
                rs.model->mtp.fuse_weight, 3, rs.model->mtp.markov_dflare_norm, kRmsNormEps,
                rs.mtp_f_rnn, s);
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
    }

    // ---- DSpark Draft Block (3-layer non-causal MTP + D folded Markov steps) ----
    // Fully CUDA-Graph-capturable per batch size B!
    void run_dspark_draft_body(int B) {
        const int D     = static_cast<int>(draft_tokens_);
        const int blk   = D + 1; // block_size = 1 anchor + D drafts (e.g. 8)
        const int blk_T = B * blk;
        const int drf_T = B * D;

        std::array<Tensor, 2> anchor_pair{}, ar_pair{}, attn_pair{}, mlp_pair{};
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            anchor_pair[r] = Tensor(rs.mtp_anchor_emb.data, DType::BF16, {kHiddenSize, B});
            ar_pair[r]     = Tensor(rs.ar_staging.data, DType::BF16, {kHiddenSize, blk_T});
            attn_pair[r]   = Tensor(rs.mtp_attn_out.data, DType::BF16, {kHiddenSize, blk_T});
            mlp_pair[r]    = Tensor(rs.mtp_mlp_out.data, DType::BF16, {kHiddenSize, blk_T});
        }
        std::array<Tensor, 2> ar_b_pair{};
        for (int r = 0; r < tp_; ++r) {
            ar_b_pair[r] = Tensor(ranks_[static_cast<std::size_t>(r)].ar_staging.data, DType::BF16,
                                  {kHiddenSize, B});
        }

        // 1. Anchor embedding + AllReduce + embed_norm -> build block [anchor, mask, ..., mask]
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            Tensor aids(rs.round_anchor_ids.data, DType::I32, {B});
            ops::inkling2_embed_gather_shard(aids, rs.model->embed, rs.model->vocab_off,
                                             rs.model->vocab_rows, anchor_pair[r], s);
        }
        allreduce_bf16(anchor_pair, ar_b_pair);
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            ops::inkling2_rmsnorm(anchor_pair[r], &rs.model->embed_norm, anchor_pair[r],
                                  kRmsNormEps, s);
            Tensor mtp_h_t(rs.mtp_h.data, DType::BF16, {kHiddenSize, blk_T});
            ops::inkling2_mtp_build_block_embed(anchor_pair[r], rs.model->mtp.mask_embed, B, blk,
                                                mtp_h_t, s);
        }

        // 2. Three DSpark MTP transformer layers
        for (int li = 0; li < kMtpNumLayers; ++li) {
            for (int r = 0; r < tp_; ++r) {
                auto& rs       = ranks_[static_cast<std::size_t>(r)];
                const auto& ml = rs.model->mtp.layers[static_cast<std::size_t>(li)];
                cudaStream_t s = rs.dev->stream;
                CUDA_CHECK(cudaSetDevice(rs.dev->device));

                Tensor h_t(rs.mtp_h.data, DType::BF16, {kHiddenSize, blk_T});
                Tensor norm_t(rs.mtp_norm.data, DType::BF16, {kHiddenSize, blk_T});
                Tensor qkv_t(rs.mtp_qkv.data, DType::BF16, {5120 / tp_, blk_T});
                Tensor q_t(rs.mtp_q.data, DType::BF16, {kHeadDim, rs.model->num_q_heads, blk_T});
                Tensor kb_t(rs.mtp_kb.data, DType::BF16, {kHeadDim, rs.model->num_kv_heads, blk_T});
                Tensor vb_t(rs.mtp_vb.data, DType::BF16, {kHeadDim, rs.model->num_kv_heads, blk_T});
                Tensor aval_t(rs.mtp_attn_val.data, DType::BF16,
                              {rs.model->num_q_heads * kHeadDim, blk_T});

                ops::inkling2_rmsnorm(h_t, &ml.input_norm, norm_t, kRmsNormEps, s);
                ops::inkling2_bf16_linear(norm_t, ml.qkv, qkv_t, s);
                ops::inkling2_mtp_block_qkv_rope(
                    qkv_t, ml.q_norm, ml.k_norm,
                    static_cast<const std::int32_t*>(rs.round_positions.data), blk_T,
                    rs.model->num_q_heads, rs.model->num_kv_heads, kMtpRopeTheta, kRmsNormEps, q_t,
                    kb_t, vb_t, s);
                ops::inkling2_mtp_block_attention(
                    q_t, kb_t, vb_t, make_mtp_kv_target(rs, li),
                    static_cast<const std::int32_t*>(rs.round_batch_lanes.data),
                    static_cast<const std::int32_t*>(rs.round_ctx_lens.data), B, blk,
                    rs.model->num_q_heads, rs.model->num_kv_heads, kSwaWindow, aval_t, s);
                ops::inkling2_bf16_linear(aval_t, ml.o_proj, attn_pair[r], s);
            }
            allreduce_bf16(attn_pair, ar_pair);

            for (int r = 0; r < tp_; ++r) {
                auto& rs       = ranks_[static_cast<std::size_t>(r)];
                const auto& ml = rs.model->mtp.layers[static_cast<std::size_t>(li)];
                cudaStream_t s = rs.dev->stream;
                CUDA_CHECK(cudaSetDevice(rs.dev->device));

                Tensor h_t(rs.mtp_h.data, DType::BF16, {kHiddenSize, blk_T});
                Tensor norm_t(rs.mtp_norm.data, DType::BF16, {kHiddenSize, blk_T});
                Tensor gu_t(rs.mtp_gate_up.data, DType::BF16, {(2 * kMtpInterSize) / tp_, blk_T});
                Tensor act_t(rs.mtp_mlp_act.data, DType::BF16, {kMtpInterSize / tp_, blk_T});

                ops::residual_add(attn_pair[r], h_t, s);
                ops::inkling2_rmsnorm(h_t, &ml.post_attention_norm, norm_t, kRmsNormEps, s);
                ops::inkling2_bf16_linear(norm_t, ml.gate_up, gu_t, s);
                ops::inkling2_silu_mul_packed(gu_t, act_t, s);
                ops::inkling2_bf16_linear(act_t, ml.down_proj, mlp_pair[r], s);
            }
            allreduce_bf16(mlp_pair, ar_pair);

            for (int r = 0; r < tp_; ++r) {
                auto& rs = ranks_[static_cast<std::size_t>(r)];
                CUDA_CHECK(cudaSetDevice(rs.dev->device));
                Tensor h_t(rs.mtp_h.data, DType::BF16, {kHiddenSize, blk_T});
                ops::residual_add(mlp_pair[r], h_t, rs.dev->stream);
            }
        }

        // 3. Final MTP norm + batched base_logits = h_draft[1..D] @ output_head^T + zero RNN state
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));

            Tensor h_t(rs.mtp_h.data, DType::BF16, {kHiddenSize, blk_T});
            Tensor draft_h(rs.mtp_draft_h.data, DType::BF16, {kHiddenSize, drf_T});
            Tensor base_log(rs.mtp_base_logits.data, DType::BF16, {rs.model->vocab_rows, drf_T});

            ops::inkling2_rmsnorm(h_t, &rs.model->mtp.final_norm, h_t, kRmsNormEps, s);
            ops::inkling2_mtp_extract_draft_hiddens(h_t, B, blk, draft_h, s);
            ops::inkling2_bf16_linear(draft_h, rs.model->output_head, base_log, s);
            CUDA_CHECK(cudaMemsetAsync(rs.mtp_rnn_state.data, 0,
                                       static_cast<std::size_t>(B) * kMtpMarkovRank * 2, s));
        }

        // 4. Autoregressive Folded Markov RNN (D steps: 0 .. D-1)
        // Note: `mtp_f_rnn` is indexed by `lane`! Copy compact batch f_rnn into `mtp_anchor_emb` (reuse [H, B]!)
        // Actually: for B==1 or arbitrary lanes, let's copy per-batch f_rnn on host before launching the graph!
        // Here `mtp_anchor_emb` is overwritten in step 1, so we use `cur` [H, B] (already free!) to hold compact f_rnn!
        for (int step = 0; step < D; ++step) {
            std::array<Tensor, 2> vals{}, ids{}, outs{};
            for (int r = 0; r < tp_; ++r) {
                auto& rs       = ranks_[static_cast<std::size_t>(r)];
                cudaStream_t s = rs.dev->stream;
                CUDA_CHECK(cudaSetDevice(rs.dev->device));

                Tensor st_t(rs.mtp_rnn_state.data, DType::BF16, {kMtpMarkovRank, B});
                Tensor prev_t((step == 0) ? rs.round_anchor_ids.data
                                          : static_cast<void*>(
                                                static_cast<std::int32_t*>(rs.round_draft_out.data) +
                                                static_cast<std::size_t>(step - 1) * B),
                              DType::I32, {B});
                Tensor h_blk(rs.mtp_h.data, DType::BF16, {kHiddenSize, blk_T});
                Tensor frnn_b(rs.cur.data, DType::BF16, {kHiddenSize, B});
                Tensor z_t(rs.mtp_rnn_z.data, DType::BF16, {2 * kMtpMarkovRank + 2 * kHiddenSize, B});
                Tensor proj_t(rs.mtp_rnn_proj.data, DType::BF16, {3 * kMtpMarkovRank, B});
                Tensor lat_t(rs.mtp_rnn_latent.data, DType::BF16, {kMtpMarkovRank, B});

                ops::inkling2_mtp_markov_rnn_prep(st_t, prev_t, rs.model->mtp.markov_w1, h_blk,
                                                  step + 1, blk, frnn_b, z_t, s);
                ops::inkling2_bf16_linear(z_t, rs.model->mtp.markov_rnn_proj_weight, proj_t, s,
                                          &rs.model->mtp.markov_rnn_proj_bias);
                ops::inkling2_mtp_markov_rnn_gates(proj_t, st_t, lat_t, s);

                // Strided Markov logits argmax directly on mtp_base_logits [vocab_rows, B * D] -> ZERO D2D copies!
                Tensor base_all(rs.mtp_base_logits.data, DType::BF16,
                                {rs.model->vocab_rows, drf_T});

                vals[r] = Tensor(rs.round_local_max_val.data, DType::FP32, {1, B});
                ids[r]  = Tensor(rs.round_local_max_id.data, DType::I32, {1, B});
                outs[r] = Tensor(static_cast<std::int32_t*>(rs.round_draft_out.data) +
                                     static_cast<std::size_t>(step) * B,
                                 DType::I32, {B});

                ops::inkling2_markov_logits_argmax_strided(base_all, lat_t, rs.mtp_fold_w,
                                                           rs.mtp_fold_b, rs.model->valid_rows,
                                                           D, step, vals[r], ids[r], s);
                if (tp_ == 1) {
                    CUDA_CHECK(cudaMemcpyAsync(outs[0].data, ids[0].data,
                                               static_cast<std::size_t>(B) * sizeof(std::int32_t),
                                               cudaMemcpyDeviceToDevice, s));
                }
            }
            if (tp_ == 2) {
                Tensor gv(ranks_[0].round_gather_vals.data, DType::FP32, {2, B});
                Tensor gi(ranks_[0].round_gather_ids.data, DType::I32, {2, B});
                ops::allreduce_argmax(std::span<const Tensor>(vals.data(), 2),
                                      std::span<const Tensor>(ids.data(), 2),
                                      std::span<const Tensor>(outs.data(), 2), gv, gi,
                                      ranks_[0].model->vocab_rows, ec_, *peer_events_);
            }
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));
    }

    void run_dspark_draft_batch(std::span<const std::uint32_t> lanes) {
        const int B   = static_cast<int>(lanes.size());
        const int D   = static_cast<int>(draft_tokens_);
        const int blk = D + 1;

        std::vector<std::int32_t> h_anchors(static_cast<std::size_t>(B));
        std::vector<std::int32_t> h_lanes(static_cast<std::size_t>(B));
        std::vector<std::int32_t> h_ctx_lens(static_cast<std::size_t>(B));
        std::vector<std::int32_t> h_blk_pos(static_cast<std::size_t>(B * blk));

        for (int b = 0; b < B; ++b) {
            const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
            const auto& st        = lanes_[l];
            h_anchors[static_cast<std::size_t>(b)]  = st.last_token;
            h_lanes[static_cast<std::size_t>(b)]    = static_cast<std::int32_t>(l);
            h_ctx_lens[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(st.seq_len);
            for (int s = 0; s < blk; ++s) {
                h_blk_pos[static_cast<std::size_t>(b * blk + s)] =
                    static_cast<std::int32_t>(st.seq_len) + s;
            }
        }

        // Stage inputs + compact f_rnn[b] -> cur[b] on every rank
        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_anchor_ids.data, h_anchors.data(),
                                       h_anchors.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_batch_lanes.data, h_lanes.data(),
                                       h_lanes.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_ctx_lens.data, h_ctx_lens.data(),
                                       h_ctx_lens.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_positions.data, h_blk_pos.data(),
                                       h_blk_pos.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, s));
            for (int b = 0; b < B; ++b) {
                const std::uint32_t l = lanes[static_cast<std::size_t>(b)];
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::byte*>(rs.cur.data) + static_cast<std::size_t>(b) * kHiddenSize * 2,
                    static_cast<const std::byte*>(rs.mtp_f_rnn.data) +
                        static_cast<std::size_t>(l) * kHiddenSize * 2,
                    kHiddenSize * 2, cudaMemcpyDeviceToDevice, s));
            }
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));

        run_graph_or_eager(draft_graphs_[static_cast<std::size_t>(B - 1)],
                           [&] { run_dspark_draft_body(B); });

        // Copy D * B drafted token IDs back to host
        std::vector<std::int32_t> h_drafts(static_cast<std::size_t>(D * B));
        CUDA_CHECK(cudaMemcpyAsync(h_drafts.data(), ranks_[0].round_draft_out.data,
                                   h_drafts.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   ranks_[0].dev->stream));
        ranks_[0].dev->synchronize();
        if (!use_cuda_graph_ && tp_ == 2) { ranks_[1].dev->synchronize(); }

        for (int b = 0; b < B; ++b) {
            auto& st = lanes_[lanes[static_cast<std::size_t>(b)]];
            for (int step = 0; step < D; ++step) {
                st.draft_tokens[static_cast<std::size_t>(step)] =
                    h_drafts[static_cast<std::size_t>(step * B + b)];
            }
        }
    }

    // ---- CUDA Graph Capture / Replay Helper (handles both TP1 & TP2 single-graph capture!) ----
    void run_graph_or_eager(GraphSlot& slot, const std::function<void()>& body) {
        if (!use_cuda_graph_) {
            body();
            return;
        }
        if (!slot.captured) {
            // Synchronize all ranks before capture
            for (int r = 0; r < tp_; ++r) { ranks_[static_cast<std::size_t>(r)].dev->synchronize(); }
            CUDA_CHECK(cudaSetDevice(ranks_[0].dev->device));
            if (tp_ == 2) {
                DecodeGraphPeerCapture peer{.bridge = &*peer_bridge_,
                                            .stream = ranks_[1].dev->stream};
                slot.def.capture(ranks_[0].dev->stream, body,
                                 std::span<const DecodeGraphPeerCapture>(&peer, 1));
            } else {
                slot.def.capture(ranks_[0].dev->stream, body);
            }
            slot.exec.instantiate(slot.def);
            slot.exec.upload(ranks_[0].dev->stream);
            last_graph_nodes_ = slot.def.node_count();
            slot.captured     = true;
        }
        CUDA_CHECK(cudaSetDevice(ranks_[0].dev->device));
        if (tp_ == 2) {
            peer_bridge_->gate_launch(ranks_[1].dev->stream, ranks_[0].dev->stream);
        }
        slot.exec.launch(ranks_[0].dev->stream);
    }

    void run_decode_greedy_maybe_graph(int B) {
        run_graph_or_eager(decode_graphs_[static_cast<std::size_t>(B - 1)], [&] {
            run_forward_body(ForwardMode::Decode, B, 1, 0, 0, mtp_enabled_);
            run_unembed_greedy(B, 0);
        });
    }

    void run_verify_step_maybe_graph(int B, int V) {
        run_graph_or_eager(verify_graphs_[static_cast<std::size_t>(B - 1)], [&] {
            run_forward_body(ForwardMode::Verify, B, V, 0, 0, true);
            run_unembed_greedy(B * V, 0);
        });
    }

    // ---- Chunked Prefill Step ----
    runtime::PrefillStepResult run_prefill_chunk(std::uint32_t lane) {
        const auto t0 = Clock::now();
        auto& st      = lanes_[lane];
        const std::uint32_t total = static_cast<std::uint32_t>(st.prompt_tokens.size());
        const std::uint32_t start = st.prefill_cursor;
        const std::uint32_t count = std::min(prefill_chunk_, total - start);
        const bool complete       = (start + count == total);

        std::vector<std::int32_t> h_ids(count);
        std::vector<std::int32_t> h_pos(count);
        std::vector<std::int32_t> h_lanes(count, static_cast<std::int32_t>(lane));
        for (std::uint32_t i = 0; i < count; ++i) {
            h_ids[i] = st.prompt_tokens[start + i];
            h_pos[i] = static_cast<std::int32_t>(start + i);
        }

        for (int r = 0; r < tp_; ++r) {
            auto& rs       = ranks_[static_cast<std::size_t>(r)];
            cudaStream_t s = rs.dev->stream;
            CUDA_CHECK(cudaSetDevice(rs.dev->device));
            ensure_kv_tokens(st.kv_alloc[static_cast<std::size_t>(r)], start + count, s);
            CUDA_CHECK(cudaMemcpyAsync(rs.round_token_ids.data, h_ids.data(),
                                       count * sizeof(std::int32_t), cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_positions.data, h_pos.data(),
                                       count * sizeof(std::int32_t), cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(rs.round_tok_lanes.data, h_lanes.data(),
                                       count * sizeof(std::int32_t), cudaMemcpyHostToDevice, s));
        }
        CUDA_CHECK(cudaSetDevice(ec_.primary().device));

        run_forward_body(ForwardMode::Prefill, 1, static_cast<int>(count),
                         static_cast<int>(lane), static_cast<int>(start), mtp_enabled_);

        if (mtp_enabled_) {
            append_mtp_ctx_from_caps(lane, 0, static_cast<int>(count), start);
        }

        st.prefill_cursor = start + count;
        st.seq_len        = start + count;

        runtime::PrefillStepResult res;
        res.processed_prompt_tokens = count;
        res.complete                = complete;
        res.summary = runtime::BeginSummary{
            .prompt_tokens        = total,
            .reused_prompt_tokens = 0,
            .prefix_reuse_path    = PrefixReusePath::FullReset,
        };

        if (complete) {
            st.prefilling = false;
            if (st.options.sampling.temperature <= 0.0f && !logits_capture_enabled_) {
                run_unembed_greedy(1, static_cast<int>(count) - 1);
                CUDA_CHECK(cudaMemcpyAsync(&st.last_token, ranks_[0].round_winner_ids.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                           ranks_[0].dev->stream));
                for (int r = 0; r < tp_; ++r) { ranks_[static_cast<std::size_t>(r)].dev->synchronize(); }
            } else {
                std::vector<TokenId> tok_vec;
                const std::uint32_t l_arr[1] = {lane};
                sample_or_capture_logits(std::span<const std::uint32_t>(l_arr, 1),
                                         ops::kSamplePurposePrefill, tok_vec,
                                         static_cast<int>(count) - 1);
                st.last_token = tok_vec[0];
            }
            res.round.tokens = std::span<const TokenId>(&st.last_token, 1);
        } else {
            // Synchronize so the next chunk/decode sees finished state
            for (int r = 0; r < tp_; ++r) { ranks_[static_cast<std::size_t>(r)].dev->synchronize(); }
        }

        st.timings.prefill_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
        return res;
    }

    const LoadedModelData& model_;
    SequencePlanImpl plan_;
    ExecutionContext& ec_;
    int tp_                       = 1;
    std::uint32_t max_concurrency_ = 1;
    std::uint32_t max_context_    = 2048;
    std::uint32_t prefill_chunk_  = 1024;
    bool mtp_enabled_             = false;
    std::uint32_t draft_tokens_   = 0;
    bool use_cuda_graph_          = true;

    std::optional<ops::PeerEvents> peer_events_;
    std::optional<DecodeGraphPeerBridge> peer_bridge_;
    std::array<RankState, 2> ranks_{};
    std::array<LaneState, kMaximumConcurrency> lanes_{};

    std::array<GraphSlot, kMaximumConcurrency> decode_graphs_{};
    std::array<GraphSlot, kMaximumConcurrency> verify_graphs_{};
    std::array<GraphSlot, kMaximumConcurrency> draft_graphs_{};
    std::size_t last_graph_nodes_ = 0;

    std::vector<std::uint32_t> pending_lanes_;
    std::vector<TokenId> pending_tokens_;
    std::vector<std::int32_t> pending_emitted_counts_;
    bool pending_was_spec_ = false;

    bool logits_capture_enabled_ = false;
    std::vector<std::uint16_t> captured_logits_bf16_;
};

} // namespace detail

// ---------------------------------------------------------------------------
// Public wrapper forwarding
// ---------------------------------------------------------------------------

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
SequencePlan::SequencePlan(SequencePlan&&) noexcept            = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;
SequencePlan::~SequencePlan()                                  = default;

std::uint32_t SequencePlan::capacity() const noexcept { return impl_->spec.max_context; }
std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_->spec.main_page_groups * static_cast<std::uint32_t>(kPagedKVPageSize);
}
std::uint32_t SequencePlan::max_concurrency() const noexcept { return impl_->spec.max_concurrency; }
std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_->reservation_bytes;
}
std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_->ws_regions.total_bytes;
}
std::size_t SequencePlan::request_transient_capacity_bytes() const noexcept { return 0; }

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}
SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept            = default;
SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;
SequencePlanner::~SequencePlanner()                                     = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    return impl_->curve;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    auto out              = std::make_unique<detail::SequencePlanImpl>();
    out->spec             = impl_->base_spec;
    out->spec.main_page_groups = main_page_groups;
    out->seq_regions      = build_sequence_regions(out->spec);
    out->ws_regions       = build_workspace_regions(out->spec);
    out->sample_ws_bytes  = sampling_ws_bytes(out->spec.max_concurrency);
    out->reservation_bytes = impl_->curve.reservation_bytes(main_page_groups);
    return SequencePlan(std::move(out));
}

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;
const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    return impl_->summary;
}

RequestPlan::RequestPlan(std::unique_ptr<detail::RequestPlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
RequestPlan::RequestPlan(RequestPlan&&) noexcept            = default;
RequestPlan& RequestPlan::operator=(RequestPlan&&) noexcept = default;
RequestPlan::~RequestPlan()                                 = default;
const runtime::RequestPlanSummary& RequestPlan::summary() const noexcept { return impl_->summary; }

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() noexcept = default;

double Program::prepare_media(PreparedPrompt&, const PreparationControl&) const { return 0.0; }
RequestBasePlan Program::plan_request_base(const PreparedPrompt& prompt,
                                           const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request_base(prompt, options);
}
RequestPlan Program::plan_request_for_lane(std::uint32_t lane, const PreparedPrompt& prompt,
                                           const RequestBasePlan& base) {
    return impl_->plan_request_for_lane(lane, prompt, base);
}
bool Program::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    return impl_->can_admit_lane(lane, plan);
}
bool Program::can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                     const RequestPlan& plan) const noexcept {
    return impl_->can_admit_lane_after_retained_eviction(lane, plan);
}
runtime::AdmissionResources Program::admission_capacity() const noexcept {
    return impl_->admission_capacity();
}
runtime::PrefillStepResult Program::start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                                       RequestPlan&& plan,
                                                       runtime::TransientRegion transient) {
    return impl_->start_prefill_lane(lane, std::move(prompt), std::move(plan), transient);
}
runtime::PrefillStepResult Program::advance_prefill_lane(std::uint32_t lane) {
    return impl_->advance_prefill_lane(lane);
}
runtime::BatchedGeneratedRound Program::decode_batch(std::span<const std::uint32_t> lanes,
                                                     std::span<const runtime::RoundBudget> budgets) {
    return impl_->decode_batch(lanes, budgets);
}
void Program::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    impl_->resolve_prefill_lane(lane, terminal);
}
void Program::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                    std::span<const std::uint32_t> accepted_tokens,
                                    std::span<const std::uint8_t> terminal,
                                    std::span<const std::uint8_t> cancelled) {
    impl_->resolve_pending_batch(lanes, accepted_tokens, terminal, cancelled);
}
void Program::abort_lane(std::uint32_t lane) noexcept { impl_->abort_lane(lane); }
bool Program::has_retained_lane(std::uint32_t lane) const noexcept {
    return impl_->has_retained_lane(lane);
}
void Program::evict_retained_lane(std::uint32_t lane) { impl_->evict_retained_lane(lane); }
GenerationTimings Program::generation_timings_lane(std::uint32_t lane) const noexcept {
    return impl_->generation_timings_lane(lane);
}
const SpeculativeStats& Program::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return impl_->speculative_stats_lane(lane);
}
MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }
void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }
std::span<const std::uint16_t> Program::last_round_logits_bf16() const noexcept {
    return impl_->last_round_logits_bf16();
}
void Program::enable_logits_capture(bool enabled) { impl_->enable_logits_capture(enabled); }
void Program::enable_peer_egress_check(bool enabled) noexcept {
    impl_->enable_peer_egress_check(enabled);
}
std::uint64_t Program::peer_egress_check_rounds() const noexcept {
    return impl_->peer_egress_check_rounds();
}
std::uint64_t Program::peer_egress_check_mismatches() const noexcept {
    return impl_->peer_egress_check_mismatches();
}

SequencePlanner make_sequence_planner(DeviceContext& /*device*/, const EngineOptions& options) {
    if (options.max_context > static_cast<std::uint32_t>(kMaxContextNative)) {
        throw std::invalid_argument("inkling2-mini max_context exceeds 262144");
    }
    const bool mtp = (options.speculative.backend != SpeculativeBackend::None);
    SequenceLayoutSpec spec;
    spec.tp              = options.tp;
    spec.max_context     = options.max_context;
    spec.max_concurrency = options.max_concurrency;
    spec.prefill_chunk   = options.prefill_chunk;
    spec.mtp_enabled     = mtp;
    spec.draft_tokens    = mtp ? options.speculative.draft_tokens : 0;
    spec.use_cuda_graph  = options.use_cuda_graph;
    spec.spec_backend    = options.speculative.backend;

    const std::uint32_t min_pages = pages_for_tokens(options.max_context);
    const std::uint32_t max_pages =
        std::max<std::uint32_t>(min_pages, min_pages * options.max_concurrency);

    spec.main_page_groups     = min_pages;
    const auto seq_min        = build_sequence_regions(spec);
    const auto ws             = build_workspace_regions(spec);
    const std::size_t sws     = sampling_ws_bytes(spec.max_concurrency);
    const std::size_t graph_b = spec.use_cuda_graph ? kGraphAllowanceBytes : 0;
    const std::size_t min_res = seq_min.total_bytes + ws.total_bytes + sws + graph_b;

    spec.main_page_groups     = min_pages + 1;
    const auto seq_plus1      = build_sequence_regions(spec);
    const std::size_t stride  = seq_plus1.total_bytes - seq_min.total_bytes;

    auto impl       = std::make_unique<detail::SequencePlannerImpl>();
    impl->base_spec = spec;
    impl->curve     = runtime::SequenceCapacityCurve{
        .main_page_tokens                   = static_cast<std::uint32_t>(kPagedKVPageSize),
        .minimum_main_page_groups           = min_pages,
        .maximum_main_page_groups           = max_pages,
        .minimum_device_reservation_bytes   = min_res,
        .bytes_per_additional_main_page_group = stride,
    };
    return SequencePlanner(std::move(impl));
}

std::unique_ptr<Program> create_program(const detail::LoadedModelData& model, SequencePlan&& plan,
                                        ExecutionContext& execution) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("inkling2 SequencePlan is empty"); }
    auto impl = std::make_unique<detail::ProgramImpl>(model, std::move(*plan.impl_), execution);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::targets::inkling2_mini
