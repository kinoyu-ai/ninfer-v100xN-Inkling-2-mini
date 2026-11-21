#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops {

// ---------------------------------------------------------------------------
// 1. Generalized Volta BF16 Linear & Output Head
// ---------------------------------------------------------------------------

// y[T, N] = x[T, K] @ w[N, K]^T (optional bias[N], optional residual add: y = x@w^T + residual).
// x: BF16 [K, T] (NInfer row-major: ne[0]=K, ne[1]=T), w: BF16 [K, N], out: BF16 or FP32 [N, T].
void inkling2_bf16_linear(const Tensor& x, const Tensor& w, Tensor& out, cudaStream_t stream,
                          const Tensor* bias = nullptr, const Tensor* residual = nullptr);

// Paired projection: runs (x0 @ w0^T -> out0) and (x1 @ w1^T -> out1) with shared launch when small.
void inkling2_bf16_linear_pair(const Tensor& x0, const Tensor& w0, Tensor& out0,
                               const Tensor& x1, const Tensor& w1, Tensor& out1,
                               cudaStream_t stream);

// Triple projection (e.g. q_gate_r_sink + k_proj + v_proj in ONE launch -> zero tail wave & 96 fewer launches!).
void inkling2_bf16_linear_triple(const Tensor& x0, const Tensor& w0, Tensor& out0,
                                 const Tensor& x1, const Tensor& w1, Tensor& out1,
                                 const Tensor& x2, const Tensor& w2, Tensor& out2,
                                 cudaStream_t stream);

// Output head + optional softcap + optional vocab-limit masking + optional local argmax.
// w: BF16 [K, N_shard], x: BF16 [K, T].
// If logits_out != nullptr: writes FP32 [N_shard, T] (softcapped, masked beyond valid_shard_cols).
// If max_vals != nullptr && max_ids != nullptr: writes per-row local argmax over valid_shard_cols.
void inkling2_unembed_project(const Tensor& x, const Tensor& w, float softcap,
                              std::int32_t valid_shard_cols, Tensor* logits_out,
                              Tensor* max_vals, Tensor* max_ids, cudaStream_t stream);

// Pre-warm per-device internal argmax reduction scratch before CUDA Graph capture.
void inkling2_ensure_argmax_scratch(int device, std::size_t min_blocks = 32768);

// Fused DSpark Markov logits step: logits_i = base_logits_i + latent @ fold_w^T + fold_b,
// then local argmax over [0, valid_shard_cols).
// base_logits: BF16/FP32 [N_shard, B], latent: BF16/FP32 [256, B],
// fold_w: BF16 [256, N_shard], fold_b: FP32 [N_shard].
void inkling2_markov_logits_argmax(const Tensor& base_logits, const Tensor& latent,
                                   const Tensor& fold_w, const Tensor& fold_b,
                                   std::int32_t valid_shard_cols, Tensor& max_vals,
                                   Tensor& max_ids, cudaStream_t stream);

// Strided variant reading base_logits row (b * base_row_stride + base_row_offset) directly (zero D2D copies!).
void inkling2_markov_logits_argmax_strided(const Tensor& base_logits, const Tensor& latent,
                                           const Tensor& fold_w, const Tensor& fold_b,
                                           std::int32_t valid_shard_cols,
                                           std::int32_t base_row_stride,
                                           std::int32_t base_row_offset, Tensor& max_vals,
                                           Tensor& max_ids, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 2. Norms, Attention Residuals, Cross-Pass Loop Controllers
// ---------------------------------------------------------------------------

// Plain RMSNorm (NO +1 weight offset): out = (x / sqrt(mean(x^2) + eps)) * weight (weight optional).
void inkling2_rmsnorm(const Tensor& x, const Tensor* weight, Tensor& out, float eps,
                      cudaStream_t stream);

// Embedding gather + plain RMSNorm (vocab-sharded TP aware: out-of-shard tokens write 0 for AllReduce).
void inkling2_embed_gather_shard(const Tensor& token_ids, const Tensor& table,
                                 std::int32_t vocab_offset, std::int32_t vocab_count,
                                 Tensor& out, cudaStream_t stream);

// Block Attention Residuals over S in [1, 5] sources [H, T] (BF16) and Q in [1, 4] queries [H, Q] (FP32).
// Computes v_s = rms_norm_no_gamma(sources[s], eps), scores = q @ v^T / sqrt(H), out[q] = softmax(scores) @ v.
// If post_norm_w[q] != nullptr, fuses plain RMSNorm(out[q], post_norm_w[q], eps) directly into BF16!
void inkling2_attn_res_mix(const void* const* source_ptrs, std::int32_t num_sources,
                           const Tensor& queries_f32, const void* const* post_norm_w_ptrs,
                           void* const* out_ptrs, std::int32_t num_queries, std::int32_t hidden,
                           std::int32_t tokens, float eps, cudaStream_t stream);

// Paired AttnRes launch (runs cross-pass messages S=2,Q=4 AND pre-attn block AttnRes S=1..4,Q=1/3 in ONE launch!).
void inkling2_attn_res_mix_pair(const void* const* src0, std::int32_t num_src0,
                                const Tensor& q0_f32, void* const* out0, std::int32_t num_q0,
                                const void* const* src1, std::int32_t num_src1,
                                const Tensor& q1_f32, const void* const* norm1, void* const* out1,
                                std::int32_t num_q1, std::int32_t hidden, std::int32_t tokens,
                                float eps, cudaStream_t stream);

// Cross-Pass Loop Controller:
//   z = concat(rms_no_gamma(cur), rms_no_gamma(msg)) [4608, T]
//   h = z @ down^T [64, T]
//   [g_raw, f_raw] = h @ gain_forget^T [4608, T]
//   out = (1 - tanh(f_raw)) * cur + tanh(g_raw) * msg
// If post_norm_w != nullptr, fuses plain RMSNorm(out, post_norm_w, eps) directly!
void inkling2_loop_control(const Tensor& cur, const Tensor& msg, const Tensor& down_w,
                           const Tensor& gain_forget_w, const Tensor* post_norm_w,
                           Tensor& scratch_z, Tensor& scratch_h, Tensor& scratch_gf,
                           Tensor& out, float eps, cudaStream_t stream);

// Batched 3-controller launch (Q, K, V controllers simultaneously in grid.z=3 -> 192 fewer launches/step!).
void inkling2_loop_control_qkv(const Tensor& cur0, const Tensor& cur1, const Tensor& cur2,
                               const Tensor& msg0, const Tensor& msg1, const Tensor& msg2,
                               const Tensor& down0, const Tensor& down1, const Tensor& down2,
                               const Tensor& gf0, const Tensor& gf1, const Tensor& gf2,
                               const Tensor* norm0, const Tensor* norm1, const Tensor* norm2,
                               Tensor& scratch_z, Tensor& scratch_h, Tensor& scratch_gf,
                               Tensor& out0, Tensor& out1, Tensor& out2,
                               float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 3. Causal Short Convolutions (W = 4, Static & Dynamic, FP32 State)
// ---------------------------------------------------------------------------

// State layout per lane: [3, C] FP32 (slot 0 = t-3, slot 1 = t-2, slot 2 = t-1).
// Full state buffer for all lanes: [3, C, max_lanes] -> lane stride = 3 * C floats.
// Verify scratch state per lane: [3, C, max_verify_tokens] -> saves intermediate post-token states
// so `inkling2_sconv_commit_verify` can atomically commit after verification without clobbering.

// Static causal sconv (W=4, weight [4, C] BF16 where weight[c*4 + 3] is lag 0, scale 1/sqrt(5)).
// Optional residual_add: out = sconv(x) + residual_add (for fused attn_sconv / mlp_sconv residual).
void inkling2_sconv_static_prefill(const Tensor& x, const Tensor& weight, float* state_lane,
                                   Tensor& out, const Tensor* residual_add, cudaStream_t stream);

void inkling2_sconv_static_decode(const Tensor& x, const Tensor& weight, float* state_pool,
                                  const std::int32_t* lane_ids, std::int32_t batch_size,
                                  Tensor& out, const Tensor* residual_add, cudaStream_t stream,
                                  std::size_t state_lane_stride = 0);

// Verify: `batch_size` lanes, `verify_len` tokens per lane (x is [C, batch_size * verify_len]).
// Does NOT modify state_pool; writes candidate post-token states into verify_states_pool.
// When state_lane_stride == 0, defaults to 3 * C; when max_verify_len == 0, defaults to verify_len.
// In unified flat layout, state_pool = state_base + conv_off, verify_states_pool = verify_base + conv_off,
// state_lane_stride = lane_state_floats, max_verify_len = pool's max_verify_len.
void inkling2_sconv_static_verify(const Tensor& x, const Tensor& weight, const float* state_pool,
                                  float* verify_states_pool, const std::int32_t* lane_ids,
                                  std::int32_t batch_size, std::int32_t verify_len,
                                  Tensor& out, const Tensor* residual_add, cudaStream_t stream,
                                  std::size_t state_lane_stride = 0,
                                  std::int32_t max_verify_len   = 0);

// Dynamic causal sconv (W=4, static_w [4, C] BF16 where static_w[c*4 + lag] is backward distance lag,
// dyn_w [4, H_kv, T] BF16, scale 1/sqrt(9)).
void inkling2_sconv_dynamic_prefill(const Tensor& x, const Tensor& dyn_w, const Tensor& static_w,
                                    std::int32_t num_kv_heads, float* state_lane, Tensor& out,
                                    cudaStream_t stream);

void inkling2_sconv_dynamic_decode(const Tensor& x, const Tensor& dyn_w, const Tensor& static_w,
                                   std::int32_t num_kv_heads, float* state_pool,
                                   const std::int32_t* lane_ids, std::int32_t batch_size,
                                   Tensor& out, cudaStream_t stream,
                                   std::size_t state_lane_stride = 0);

void inkling2_sconv_dynamic_verify(const Tensor& x, const Tensor& dyn_w, const Tensor& static_w,
                                   std::int32_t num_kv_heads, const float* state_pool,
                                   float* verify_states_pool, const std::int32_t* lane_ids,
                                   std::int32_t batch_size, std::int32_t verify_len,
                                   Tensor& out, cudaStream_t stream,
                                   std::size_t state_lane_stride = 0,
                                   std::int32_t max_verify_len   = 0);

// Commit accepted verify sconv states across all 48 logical layers (4 convs/layer) in one launch!
// state_base: contiguous pool of all 48 logical layers' sconv states per lane.
// verify_base: contiguous pool of [verify_len] snapshots per lane.
// accepted_counts[b] in [1, verify_len] (0 means no commit / cancelled).
void inkling2_sconv_commit_verify(float* state_base, const float* verify_base,
                                  std::size_t lane_state_floats, std::int32_t max_verify_len,
                                  const std::int32_t* lane_ids, const std::int32_t* accepted_counts,
                                  std::int32_t batch_size, cudaStream_t stream);

// Fused K+V static/dynamic SConv pair (1 launch instead of 2 -> 48 fewer launches/step!).
void inkling2_sconv_kv_static_decode(const Tensor& k, const Tensor& wk, float* state_k,
                                     const Tensor& v, const Tensor& wv, float* state_v,
                                     const std::int32_t* lane_ids, std::int32_t batch_size,
                                     Tensor& out_k, Tensor& out_v, cudaStream_t stream,
                                     std::size_t state_lane_stride);

void inkling2_sconv_kv_static_verify(const Tensor& k, const Tensor& wk, const float* state_k,
                                     float* verify_k, const Tensor& v, const Tensor& wv,
                                     const float* state_v, float* verify_v,
                                     const std::int32_t* lane_ids, std::int32_t batch_size,
                                     std::int32_t verify_len, Tensor& out_k, Tensor& out_v,
                                     cudaStream_t stream, std::size_t state_lane_stride,
                                     std::int32_t max_verify_len);

void inkling2_sconv_kv_dynamic_decode(const Tensor& k, const Tensor& dyn_k, const Tensor& wk,
                                      float* state_k, const Tensor& v, const Tensor& dyn_v,
                                      const Tensor& wv, float* state_v, std::int32_t num_kv_heads,
                                      const std::int32_t* lane_ids, std::int32_t batch_size,
                                      Tensor& out_k, Tensor& out_v, cudaStream_t stream,
                                      std::size_t state_lane_stride);

void inkling2_sconv_kv_dynamic_verify(const Tensor& k, const Tensor& dyn_k, const Tensor& wk,
                                      const float* state_k, float* verify_k, const Tensor& v,
                                      const Tensor& dyn_v, const Tensor& wv, const float* state_v,
                                      float* verify_v, std::int32_t num_kv_heads,
                                      const std::int32_t* lane_ids, std::int32_t batch_size,
                                      std::int32_t verify_len, Tensor& out_k, Tensor& out_v,
                                      cudaStream_t stream, std::size_t state_lane_stride,
                                      std::int32_t max_verify_len);

// Fused TP2 direct-P2P packet AllReduce + static SConv (+ optional residual_add & optional block_copy).
// Participates in the exact same PeerEvents two-epoch packet protocol (blockDim=64, 8 BF16/thread).
void inkling2_tp2_packet_allreduce_sconv_decode(
    const Tensor& partial_in, void* local_storage, void* peer_storage, std::size_t epoch_bytes,
    const Tensor& sconv_weight, float* state_pool, const std::int32_t* lane_ids,
    std::int32_t batch_size, std::size_t state_lane_stride, Tensor& out,
    const Tensor* residual_add, const Tensor* block_copy_src, Tensor* block_copy_dst,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 4. Inkling-2 Attention (L2Clip + Tau + RelBias + KV Append + SWA/Full Sink GQA)
// ---------------------------------------------------------------------------

struct Inkling2KvTarget {
    bool is_local = true;
    // For local (cyclic ring): k_buf/v_buf are BF16 [head_dim, num_kv_heads, ring_cap, max_lanes]
    void* ring_k = nullptr;
    void* ring_v = nullptr;
    std::int32_t ring_cap = 0;
    // For global (paged BF16): k_pages/v_pages are [head_dim, page_size(64), num_kv_heads, num_pages]
    void* paged_k = nullptr;
    void* paged_v = nullptr;
    const std::int32_t* block_tables = nullptr; // [max_logical_pages, max_lanes]
    std::int32_t max_logical_pages = 0;
};

// Fused prologue:
//   - Extracts Q, R from packed q_gate_r_sink [q_dim + q_dim + r_dim + n_q, T]
//   - Applies q_l2_clip (+ tau if !is_local) -> q_out [head_dim, n_q, T]
//   - Applies k_l2_clip -> writes directly to KV cache at (lane_ids[t], positions[t])
//   - Applies v_unweighted_rmsnorm -> writes directly to KV cache at (lane_ids[t], positions[t])
//   - Applies rel_l2_clip (+ tau if !is_local) -> computes rel_bias [1024, n_q, T] (FP32) = R @ proj
void inkling2_attn_prologue(const Tensor& q_gate_r_sink, const Tensor& k_in, const Tensor& v_in,
                            const Tensor& q_clip_scale, const Tensor& rel_clip_scale,
                            const Tensor& rel_logits_proj, const std::int32_t* positions,
                            const std::int32_t* lane_ids, std::int32_t tokens,
                            std::int32_t num_q_heads, std::int32_t num_kv_heads,
                            const Inkling2KvTarget& kv, bool is_local, float qk_clip_threshold,
                            float rel_clip_threshold, float log_floor, float log_alpha, float eps,
                            Tensor& q_out, Tensor& rel_bias_out, cudaStream_t stream);

// Decode / Verify attention (batched across `batch_size` lanes, `q_per_lane` in [1..8]).
// Reads Q [128, n_q, T], rel_bias [1024, n_q, T] (FP32), gate & sinks from packed q_gate_r_sink,
// attends to cyclic SWA-1024 or paged global BF16 KV, applies sink absorption + sigmoid(gate),
// writes BF16 out [n_q * 128, T].
// split_scratch: FP32 scratch buffer of size >= batch_size * q_per_lane * n_q * max_splits * (128 + 2).
void inkling2_attn_decode_verify(const Tensor& q, const Tensor& q_gate_r_sink,
                                 const Tensor& rel_bias_f32, const Inkling2KvTarget& kv,
                                 const std::int32_t* lane_ids, const std::int32_t* q_positions,
                                 std::int32_t batch_size, std::int32_t q_per_lane,
                                 std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                 std::int32_t swa_window, float* split_scratch,
                                 std::int32_t max_splits, Tensor& out, cudaStream_t stream);

// Prefill attention (single lane, `tokens` contiguous queries starting at `start_pos`).
// Same exact relative-bias, sink absorption, SWA-1024 / global causal semantics + sigmoid(gate).
void inkling2_attn_prefill(const Tensor& q, const Tensor& q_gate_r_sink,
                           const Tensor& rel_bias_f32, const Inkling2KvTarget& kv,
                           std::int32_t lane, std::int32_t start_pos, std::int32_t tokens,
                           std::int32_t num_q_heads, std::int32_t num_kv_heads,
                           std::int32_t swa_window, Tensor& out, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 5. Sparse MoE (Router + NVFP4 / MXFP8 Experts with Softcapped SwiGLU)
// ---------------------------------------------------------------------------

struct Inkling2MoeScratch {
    float* router_logits;       // [128, max_T]
    std::int32_t* topk_ids;     // [8, max_T]
    float* topk_weights;        // [8, max_T]
    std::int32_t* expert_counts;  // [128]
    std::int32_t* expert_offsets; // [129]
    std::int32_t* sorted_pairs;   // [max_T * 8] (encodes (tok << 3) | slot)
    std::int32_t* active_experts; // [128]
    std::int32_t* num_active;     // [1]
    void* intermediate_bf16;      // BF16 [inter_local, max_T * 8]
    float* accum_f32;             // FP32 [2304, max_T]
    std::int32_t* router_done;    // I32 [1] pre-zeroed counter for Last-CTA fused router (optional)
};

// Pre-warm per-device MoE FP16 staging scratch before CUDA Graph capture.
void inkling2_ensure_moe_scratch(int device, std::size_t max_tokens = 1024);

// Router: sigmoid(logits) + bias top-8 + scaled renorm (route_scale = 8.0), plus GPU grouping for T > 1.
void inkling2_moe_route(const Tensor& router_logits_f32, const Tensor& router_bias_f32,
                        float route_scale, Inkling2MoeScratch& scratch, std::int32_t tokens,
                        cudaStream_t stream);

// Fused Router GEMV + sigmoid + bias top-8 (+ grouping for T <= 8) in ONE launch via Last-CTA sync!
void inkling2_moe_router_fused(const Tensor& x, const Tensor& router_weight,
                               const Tensor& router_bias_f32, float route_scale,
                               Inkling2MoeScratch& scratch, cudaStream_t stream);

// NVFP4 MoE (layers 0..19): num_slabs = 1 (TP2, inter_local=576) or 2 (TP1, inter_local=1152).
// w13_codes: U8 [num_slabs, 128, 1152, 1152], w13_scales: FP8_E4M3 [num_slabs, 128, 1152, 144]
// w2_codes:  U8 [num_slabs, 128, 2304, 288],  w2_scales:  FP8_E4M3 [num_slabs, 128, 2304, 36]
void inkling2_moe_nvfp4_forward(const Tensor& x, const void* w13_codes, const void* w13_scales,
                                const float* w13_amax, const float* w13_denom,
                                const void* w2_codes, const void* w2_scales,
                                const float* w2_amax, const float* w2_denom,
                                std::int32_t num_slabs, float gate_cap, float up_cap,
                                Inkling2MoeScratch& scratch, Tensor& out, cudaStream_t stream);

// MXFP8 MoE (layers 20..23): num_slabs = 1 (TP2) or 2 (TP1).
// w13_codes: FP8_E4M3 [num_slabs, 128, 1152, 2304], w13_scales: U8 [num_slabs, 128, 1152, 72]
// w2_codes:  FP8_E4M3 [num_slabs, 128, 2304, 576],  w2_scales:  U8 [num_slabs, 128, 2304, 18]
void inkling2_moe_mxfp8_forward(const Tensor& x, const void* w13_codes, const void* w13_scales,
                                const void* w2_codes, const void* w2_scales,
                                std::int32_t num_slabs, float gate_cap, float up_cap,
                                Inkling2MoeScratch& scratch, Tensor& out, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 6. DSpark MTP Drafter Ops
// ---------------------------------------------------------------------------

// mix_rmsnorm: mixes 4 target hidden states [H, 4, T] with softmax(fuse_weight)[row], then RMSNorm.
void inkling2_mtp_mix_rmsnorm(const Tensor& target_hiddens, const Tensor& fuse_weight,
                              std::int32_t row, const Tensor& norm_weight, float eps,
                              Tensor& out, cudaStream_t stream);

// Indexed variant: reads target_hiddens[:, :, src_indices[t]] -> writes out[:, dst_indices[t]] (or [t] if null).
void inkling2_mtp_mix_rmsnorm_indexed(const Tensor& target_hiddens, const std::int32_t* src_indices,
                                      const std::int32_t* dst_indices, std::int32_t count,
                                      const Tensor& fuse_weight, std::int32_t row,
                                      const Tensor& norm_weight, float eps, Tensor& out,
                                      cudaStream_t stream);

// Context KV norm + NeoX RoPE + write into MTP KV cache (cyclic SWA-1024 for layers 0,2; paged for layer 1).
void inkling2_mtp_ctx_kv_write(const Tensor& kv_ctx, const Tensor& k_norm_ctx_w,
                               const std::int32_t* positions, const std::int32_t* lane_ids,
                               std::int32_t tokens, std::int32_t num_kv_heads, float rope_theta,
                               float eps, const Inkling2KvTarget& kv, cudaStream_t stream);

// Block Q/K norm + NeoX RoPE (in-place / split from packed qkv [q+k+v, T]).
void inkling2_mtp_block_qkv_rope(const Tensor& qkv, const Tensor& q_norm_w, const Tensor& k_norm_w,
                                 const std::int32_t* positions, std::int32_t tokens,
                                 std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                 float rope_theta, float eps, Tensor& q_out, Tensor& k_out,
                                 Tensor& v_out, cudaStream_t stream);

// Non-causal block attention over [ctx_kv_window, block_kv] for `batch_size` lanes, `block_size=8`.
// `ctx_lens[b]` is number of context KV tokens before the block's anchor position.
void inkling2_mtp_block_attention(const Tensor& q, const Tensor& k_block, const Tensor& v_block,
                                  const Inkling2KvTarget& kv, const std::int32_t* lane_ids,
                                  const std::int32_t* ctx_lens, std::int32_t batch_size,
                                  std::int32_t block_size, std::int32_t num_q_heads,
                                  std::int32_t num_kv_heads, std::int32_t swa_window,
                                  Tensor& out, cudaStream_t stream);

// Standard SiLU(gate) * up on packed gate_up [2*inter, T] (split halves: gate = [0..inter), up = [inter..2*inter)).
void inkling2_silu_mul_packed(const Tensor& gate_up, Tensor& out, cudaStream_t stream);

// Markov RNN input gather & gate update:
//   z = [state, markov_w1[prev_tok], hidden_i, f_rnn] -> proj = z @ rnn_w^T + rnn_b
//   gate = sigmoid(g_raw), state' = gate * state + (1 - gate) * tanh(c_raw), latent = tanh(o_raw).
void inkling2_mtp_markov_rnn_prep(const Tensor& state, const Tensor& prev_tokens,
                                  const Tensor& markov_w1, const Tensor& hidden_block,
                                  std::int32_t block_pos, std::int32_t block_size,
                                  const Tensor& f_rnn, Tensor& z_out, cudaStream_t stream);

void inkling2_mtp_markov_rnn_gates(const Tensor& proj, Tensor& state, Tensor& latent_out,
                                   cudaStream_t stream);

// Assemble draft block embeddings: slot 0 = anchor_embed[b], slots 1..block_size-1 = mask_embed.
void inkling2_mtp_build_block_embed(const Tensor& anchor_embed, const Tensor& mask_embed,
                                    std::int32_t batch_size, std::int32_t block_size,
                                    Tensor& out, cudaStream_t stream);

// Copy h [H, T] into target_caps [H, 4, T] at slot in {0, 1, 2, 3}.
void inkling2_mtp_capture_slot(const Tensor& h, Tensor& target_caps, std::int32_t slot,
                               cudaStream_t stream);

// One-time startup transpose of BF16 [C, R] -> BF16 [R, C] (for precomputing folded Markov head).
void inkling2_bf16_transpose_2d(const Tensor& src, Tensor& dst, cudaStream_t stream);

// Pack slice of BF16 [H, B * block_size] at positions 1..block_size-1 into [H, B * (block_size - 1)].
void inkling2_mtp_extract_draft_hiddens(const Tensor& hidden_block, std::int32_t batch_size,
                                        std::int32_t block_size, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
