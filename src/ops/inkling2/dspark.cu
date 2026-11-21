#include "ninfer/ops/inkling2.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim  = 128;
constexpr int kPageSize = 64;
constexpr float kAttnScale = 0.08838834764831844f; // 1 / sqrt(128)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum_256(float v) {
    __shared__ float smem[8];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    v              = warp_sum(v);
    if (lane == 0) { smem[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        v = (lane < 8) ? smem[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) { smem[0] = v; }
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ __nv_bfloat16* kv_slot_ptr(const Inkling2KvTarget& kv, bool is_v,
                                                      int lane, int kv_head, int num_kv_heads,
                                                      int pos) {
    if (kv.is_local) {
        const int slot = pos % kv.ring_cap;
        auto* base     = static_cast<__nv_bfloat16*>(is_v ? kv.ring_v : kv.ring_k);
        const std::int64_t idx =
            ((static_cast<std::int64_t>(lane) * kv.ring_cap + slot) * num_kv_heads + kv_head) *
            kHeadDim;
        return base + idx;
    }
    const int logical_page = pos / kPageSize;
    const int page_off     = pos % kPageSize;
    const int phys_page =
        __ldg(kv.block_tables + static_cast<std::int64_t>(lane) * kv.max_logical_pages +
              logical_page);
    auto* base = static_cast<__nv_bfloat16*>(is_v ? kv.paged_v : kv.paged_k);
    const std::int64_t idx =
        ((static_cast<std::int64_t>(phys_page) * num_kv_heads + kv_head) * kPageSize + page_off) *
        kHeadDim;
    return base + idx;
}

__device__ __forceinline__ float4 load_bf16x4(const __nv_bfloat16* p) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
    const auto* b   = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 f0 = __bfloat1622float2(b[0]);
    const float2 f1 = __bfloat1622float2(b[1]);
    return make_float4(f0.x, f0.y, f1.x, f1.y);
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float a, float b, float c, float d) {
    uint2 out;
    auto* dst = reinterpret_cast<__nv_bfloat162*>(&out);
    dst[0]    = __floats2bfloat162_rn(a, b);
    dst[1]    = __floats2bfloat162_rn(c, d);
    *reinterpret_cast<uint2*>(p) = out;
}

// 1. mix_rmsnorm: softmax(fuse_weight[row, 0..3]) weighted sum of 4 target hiddens + plain RMSNorm
// Fast specialized H==2304 path (576 threads = 18 warps, 4 BF16/thread) + optional src/dst indices!
__global__ void mtp_mix_rmsnorm_h2304_kernel(const __nv_bfloat16* __restrict__ targets,
                                             const std::int32_t* __restrict__ src_indices,
                                             const std::int32_t* __restrict__ dst_indices,
                                             const __nv_bfloat16* __restrict__ fuse_weight,
                                             int row_idx, const __nv_bfloat16* __restrict__ norm_w,
                                             float eps, __nv_bfloat16* __restrict__ out,
                                             int count) {
    __shared__ float s_warp[18];
    const int t = blockIdx.x;
    if (t >= count) { return; }
    const int tid  = threadIdx.x; // 0..575
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 4;

    float fw[4];
    float mx = -1e30f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        fw[s] = __bfloat162float(__ldg(fuse_weight + row_idx * 4 + s));
        mx    = fmaxf(mx, fw[s]);
    }
    float denom = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        fw[s] = expf(fw[s] - mx);
        denom += fw[s];
    }
    const float inv_d = 1.0f / denom;
#pragma unroll
    for (int s = 0; s < 4; ++s) { fw[s] *= inv_d; }

    const int src_t            = (src_indices != nullptr) ? src_indices[t] : t;
    const int dst_t            = (dst_indices != nullptr) ? dst_indices[t] : t;
    const __nv_bfloat16* tbase = targets + static_cast<std::int64_t>(src_t) * 4 * 2304 + idx;

    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float4 v = load_bf16x4(tbase + s * 2304);
        a              = fmaf(fw[s], v.x, a);
        b              = fmaf(fw[s], v.y, b);
        c              = fmaf(fw[s], v.z, c);
        d              = fmaf(fw[s], v.w, d);
    }
    float sq = warp_sum(a * a + b * b + c * c + d * d);
    if (lane == 0) { s_warp[warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < 18) ? s_warp[lane] : 0.0f;
        sum       = warp_sum(sum);
        if (lane == 0) { s_warp[0] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    const float inv = s_warp[0];
    const float4 nw = load_bf16x4(norm_w + idx);
    store_bf16x4(out + static_cast<std::int64_t>(dst_t) * 2304 + idx,
                 a * inv * nw.x, b * inv * nw.y, c * inv * nw.z, d * inv * nw.w);
}

__global__ void mtp_mix_rmsnorm_kernel(const __nv_bfloat16* __restrict__ targets,
                                       const std::int32_t* __restrict__ src_indices,
                                       const std::int32_t* __restrict__ dst_indices,
                                       const __nv_bfloat16* __restrict__ fuse_weight, int row_idx,
                                       const __nv_bfloat16* __restrict__ norm_w, float eps,
                                       __nv_bfloat16* __restrict__ out, int hidden, int tokens) {
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= tokens) { return; }

    float fw[4];
    float mx = -1e30f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        fw[s] = __bfloat162float(__ldg(fuse_weight + row_idx * 4 + s));
        mx    = fmaxf(mx, fw[s]);
    }
    float denom = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        fw[s] = expf(fw[s] - mx);
        denom += fw[s];
    }
    const float inv_d = 1.0f / denom;
#pragma unroll
    for (int s = 0; s < 4; ++s) { fw[s] *= inv_d; }

    const int src_t            = (src_indices != nullptr) ? src_indices[t] : t;
    const int dst_t            = (dst_indices != nullptr) ? dst_indices[t] : t;
    const __nv_bfloat16* tbase = targets + static_cast<std::int64_t>(src_t) * 4 * hidden;
    constexpr int kMaxE        = 16;
    const int n_e              = (hidden + 255) / 256;
    float vals[kMaxE];
    float sq = 0.0f;

    for (int e = 0; e < n_e; ++e) {
        const int i = tid + e * 256;
        if (i < hidden) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                acc = fmaf(fw[s], __bfloat162float(__ldg(tbase + s * hidden + i)), acc);
            }
            vals[e] = acc;
            sq      = fmaf(acc, acc, sq);
        }
    }
    const float inv_rms = rsqrtf(block_sum_256(sq) / static_cast<float>(hidden) + eps);
    __nv_bfloat16* orow = out + static_cast<std::int64_t>(dst_t) * hidden;
    for (int e = 0; e < n_e; ++e) {
        const int i = tid + e * 256;
        if (i < hidden) {
            orow[i] = __float2bfloat16_rn(vals[e] * inv_rms * __bfloat162float(__ldg(norm_w + i)));
        }
    }
}

// Per-head RMSNorm + NeoX RoPE (split half: x[0..63], x[64..127])
__device__ __forceinline__ void apply_head_rmsnorm_neox_rope(const __nv_bfloat16* src,
                                                             const __nv_bfloat16* norm_w,
                                                             __nv_bfloat16* dst, int pos,
                                                             float rope_theta, float eps) {
    // 64 threads per head (each thread handles pair (d, d+64) for d in 0..63)
    const int d    = threadIdx.x & 63;
    const float x0 = __bfloat162float(src[d]);
    const float x1 = __bfloat162float(src[d + 64]);

    float sq = x0 * x0 + x1 * x1;
    // Reduce across 2 warps (64 threads)
    __shared__ float s_sq[4];
    const int half_warp_id = threadIdx.x >> 6; // which head in the block
    const int warp_in_head = (threadIdx.x >> 5) & 1;
    sq                     = warp_sum(sq);
    if ((threadIdx.x & 31) == 0) { s_sq[half_warp_id * 2 + warp_in_head] = sq; }
    __syncthreads();
    const float total_sq = s_sq[half_warp_id * 2] + s_sq[half_warp_id * 2 + 1];
    const float inv_rms  = rsqrtf(total_sq / static_cast<float>(kHeadDim) + eps);

    // Match BF16 norm output before RoPE
    const float n0 = __bfloat162float(
        __float2bfloat16_rn(x0 * inv_rms * __bfloat162float(__ldg(norm_w + d))));
    const float n1 = __bfloat162float(
        __float2bfloat16_rn(x1 * inv_rms * __bfloat162float(__ldg(norm_w + d + 64))));

    const float inv_freq = powf(rope_theta, -static_cast<float>(d) / 64.0f);
    const float angle    = static_cast<float>(pos) * inv_freq;
    float s, c;
    __sincosf(angle, &s, &c);
    dst[d]      = __float2bfloat16_rn(n0 * c - n1 * s);
    dst[d + 64] = __float2bfloat16_rn(n1 * c + n0 * s);
}

// 2. Context KV norm + NeoX RoPE + direct write into MTP KV cache
// grid = (num_kv_heads, tokens), block = 64
__global__ void mtp_ctx_kv_write_kernel(const __nv_bfloat16* __restrict__ kv_ctx,
                                        const __nv_bfloat16* __restrict__ k_norm_w,
                                        const std::int32_t* __restrict__ positions,
                                        const std::int32_t* __restrict__ lane_ids,
                                        Inkling2KvTarget kv, int num_kv_heads, float rope_theta,
                                        float eps) {
    const int h       = blockIdx.x;
    const int t       = blockIdx.y;
    const int d       = threadIdx.x; // 0..63
    const int kv_dim  = num_kv_heads * kHeadDim;
    const int lane_id = (lane_ids != nullptr) ? lane_ids[t] : 0;
    const int pos     = positions[t];

    const __nv_bfloat16* k_src = kv_ctx + static_cast<std::int64_t>(t) * (2 * kv_dim) + h * kHeadDim;
    const __nv_bfloat16* v_src = k_src + kv_dim;

    __nv_bfloat16* k_dst = kv_slot_ptr(kv, false, lane_id, h, num_kv_heads, pos);
    __nv_bfloat16* v_dst = kv_slot_ptr(kv, true, lane_id, h, num_kv_heads, pos);

    apply_head_rmsnorm_neox_rope(k_src, k_norm_w, k_dst, pos, rope_theta, eps);
    v_dst[d]      = v_src[d];
    v_dst[d + 64] = v_src[d + 64];
}

// 3. Block QKV split + Q/K norm + NeoX RoPE
// grid = (num_q_heads + num_kv_heads, tokens), block = 64
__global__ void mtp_block_qkv_rope_kernel(const __nv_bfloat16* __restrict__ qkv,
                                          const __nv_bfloat16* __restrict__ q_norm_w,
                                          const __nv_bfloat16* __restrict__ k_norm_w,
                                          const std::int32_t* __restrict__ positions,
                                          int num_q_heads, int num_kv_heads, float rope_theta,
                                          float eps, __nv_bfloat16* __restrict__ q_out,
                                          __nv_bfloat16* __restrict__ k_out,
                                          __nv_bfloat16* __restrict__ v_out) {
    const int h_idx  = blockIdx.x;
    const int t      = blockIdx.y;
    const int d      = threadIdx.x; // 0..63
    const int q_dim  = num_q_heads * kHeadDim;
    const int kv_dim = num_kv_heads * kHeadDim;
    const int stride = q_dim + 2 * kv_dim;
    const int pos    = positions[t];

    const __nv_bfloat16* row = qkv + static_cast<std::int64_t>(t) * stride;
    if (h_idx < num_q_heads) {
        const __nv_bfloat16* src = row + h_idx * kHeadDim;
        __nv_bfloat16* dst       = q_out + (static_cast<std::int64_t>(t) * num_q_heads + h_idx) * kHeadDim;
        apply_head_rmsnorm_neox_rope(src, q_norm_w, dst, pos, rope_theta, eps);
    } else {
        const int kvh            = h_idx - num_q_heads;
        const __nv_bfloat16* ks  = row + q_dim + kvh * kHeadDim;
        const __nv_bfloat16* vs  = row + q_dim + kv_dim + kvh * kHeadDim;
        __nv_bfloat16* kd        = k_out + (static_cast<std::int64_t>(t) * num_kv_heads + kvh) * kHeadDim;
        __nv_bfloat16* vd        = v_out + (static_cast<std::int64_t>(t) * num_kv_heads + kvh) * kHeadDim;
        apply_head_rmsnorm_neox_rope(ks, k_norm_w, kd, pos, rope_theta, eps);
        vd[d]      = vs[d];
        vd[d + 64] = vs[d + 64];
    }
}

// 4. Non-Causal Block Attention over [ctx_kv_window, block_kv]
// grid = (num_q_heads, batch_size), block = 256 (8 warps = 1 warp per block query 0..7!)
__global__ void mtp_block_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                           const __nv_bfloat16* __restrict__ k_block,
                                           const __nv_bfloat16* __restrict__ v_block,
                                           Inkling2KvTarget kv,
                                           const std::int32_t* __restrict__ lane_ids,
                                           const std::int32_t* __restrict__ ctx_lens,
                                           int batch_size, int block_size, int num_q_heads,
                                           int num_kv_heads, int swa_window,
                                           __nv_bfloat16* __restrict__ out) {
    const int qh      = blockIdx.x;
    const int b       = blockIdx.y;
    const int warp    = threadIdx.x >> 5; // 0..7 (matches block_size=8!)
    const int wlane   = threadIdx.x & 31;
    const int d_base  = wlane * 4;
    if (b >= batch_size || warp >= block_size) { return; }

    const int lane    = lane_ids[b];
    const int ctx_len = ctx_lens[b];
    const int kv_h    = qh / (num_q_heads / num_kv_heads);
    const int tok     = b * block_size + warp;

    const __nv_bfloat16* qp = q + (static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + d_base;
    const uint2 qraw        = __ldg(reinterpret_cast<const uint2*>(qp));
    const auto* qb          = reinterpret_cast<const __nv_bfloat162*>(&qraw);
    const float2 qf0        = __bfloat1622float2(qb[0]);
    const float2 qf1        = __bfloat1622float2(qb[1]);
    const float q0          = qf0.x * kAttnScale;
    const float q1          = qf0.y * kAttnScale;
    const float q2          = qf1.x * kAttnScale;
    const float q3          = qf1.y * kAttnScale;

    float m      = -std::numeric_limits<float>::infinity();
    float l      = 0.0f;
    float acc[4] = {};

    const int k_beg = kv.is_local ? max(0, ctx_len - swa_window) : 0;
    // Attend over context KV [k_beg, ctx_len)
    for (int kpos = k_beg; kpos < ctx_len; ++kpos) {
        const __nv_bfloat16* kp = kv_slot_ptr(kv, false, lane, kv_h, num_kv_heads, kpos) + d_base;
        const __nv_bfloat16* vp = kv_slot_ptr(kv, true, lane, kv_h, num_kv_heads, kpos) + d_base;
        const uint2 kraw        = __ldg(reinterpret_cast<const uint2*>(kp));
        const uint2 vraw        = __ldg(reinterpret_cast<const uint2*>(vp));
        const auto* kb          = reinterpret_cast<const __nv_bfloat162*>(&kraw);
        const auto* vb          = reinterpret_cast<const __nv_bfloat162*>(&vraw);
        const float2 kf0        = __bfloat1622float2(kb[0]);
        const float2 kf1        = __bfloat1622float2(kb[1]);
        const float2 vf0        = __bfloat1622float2(vb[0]);
        const float2 vf1        = __bfloat1622float2(vb[1]);

        float dot = q0 * kf0.x + q1 * kf0.y + q2 * kf1.x + q3 * kf1.y;
        dot       = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);

        const float new_m = fmaxf(m, dot);
        const float alpha = (m == -std::numeric_limits<float>::infinity()) ? 0.0f : expf(m - new_m);
        const float p     = expf(dot - new_m);
        l                 = l * alpha + p;
        acc[0]            = fmaf(p, vf0.x, acc[0] * alpha);
        acc[1]            = fmaf(p, vf0.y, acc[1] * alpha);
        acc[2]            = fmaf(p, vf1.x, acc[2] * alpha);
        acc[3]            = fmaf(p, vf1.y, acc[3] * alpha);
        m                 = new_m;
    }

    // Attend non-causally over all `block_size` block keys
    for (int bi = 0; bi < block_size; ++bi) {
        const int btok = b * block_size + bi;
        const __nv_bfloat16* kp =
            k_block + (static_cast<std::int64_t>(btok) * num_kv_heads + kv_h) * kHeadDim + d_base;
        const __nv_bfloat16* vp =
            v_block + (static_cast<std::int64_t>(btok) * num_kv_heads + kv_h) * kHeadDim + d_base;
        const uint2 kraw = __ldg(reinterpret_cast<const uint2*>(kp));
        const uint2 vraw = __ldg(reinterpret_cast<const uint2*>(vp));
        const auto* kb   = reinterpret_cast<const __nv_bfloat162*>(&kraw);
        const auto* vb   = reinterpret_cast<const __nv_bfloat162*>(&vraw);
        const float2 kf0 = __bfloat1622float2(kb[0]);
        const float2 kf1 = __bfloat1622float2(kb[1]);
        const float2 vf0 = __bfloat1622float2(vb[0]);
        const float2 vf1 = __bfloat1622float2(vb[1]);

        float dot = q0 * kf0.x + q1 * kf0.y + q2 * kf1.x + q3 * kf1.y;
        dot       = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);

        const float new_m = fmaxf(m, dot);
        const float alpha = (m == -std::numeric_limits<float>::infinity()) ? 0.0f : expf(m - new_m);
        const float p     = expf(dot - new_m);
        l                 = l * alpha + p;
        acc[0]            = fmaf(p, vf0.x, acc[0] * alpha);
        acc[1]            = fmaf(p, vf0.y, acc[1] * alpha);
        acc[2]            = fmaf(p, vf1.x, acc[2] * alpha);
        acc[3]            = fmaf(p, vf1.y, acc[3] * alpha);
        m                 = new_m;
    }

    const float inv_l = 1.0f / l;
    __nv_bfloat16* op = out + (static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + d_base;
    op[0]             = __float2bfloat16_rn(acc[0] * inv_l);
    op[1]             = __float2bfloat16_rn(acc[1] * inv_l);
    op[2]             = __float2bfloat16_rn(acc[2] * inv_l);
    op[3]             = __float2bfloat16_rn(acc[3] * inv_l);
}

// 5. Packed SiLU(gate) * up (vectorized uint2 = 4 BF16/thread when inter % 4 == 0!)
__global__ void silu_mul_packed_vec4_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                            __nv_bfloat16* __restrict__ out, int inter, int tokens) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int t   = blockIdx.y;
    if (idx >= inter || t >= tokens) { return; }
    const __nv_bfloat16* row = gate_up + static_cast<std::int64_t>(t) * (2 * inter);
    const float4 g           = load_bf16x4(row + idx);
    const float4 u           = load_bf16x4(row + inter + idx);
    store_bf16x4(out + static_cast<std::int64_t>(t) * inter + idx,
                 (g.x / (1.0f + expf(-g.x))) * u.x, (g.y / (1.0f + expf(-g.y))) * u.y,
                 (g.z / (1.0f + expf(-g.z))) * u.z, (g.w / (1.0f + expf(-g.w))) * u.w);
}

__global__ void silu_mul_packed_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                       __nv_bfloat16* __restrict__ out, int inter, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= inter || t >= tokens) { return; }
    const __nv_bfloat16* row = gate_up + static_cast<std::int64_t>(t) * (2 * inter);
    const float g            = __bfloat162float(row[i]);
    const float u            = __bfloat162float(row[inter + i]);
    const float y            = (g / (1.0f + expf(-g))) * u;
    out[static_cast<std::int64_t>(t) * inter + i] = __float2bfloat16_rn(y);
}

// 6. Markov RNN prep & gates
__global__ void markov_rnn_prep_kernel(const __nv_bfloat16* __restrict__ state,
                                       const std::int32_t* __restrict__ prev_tokens,
                                       const __nv_bfloat16* __restrict__ markov_w1,
                                       const __nv_bfloat16* __restrict__ hidden_block,
                                       int block_pos, int block_size,
                                       const __nv_bfloat16* __restrict__ f_rnn,
                                       __nv_bfloat16* __restrict__ z_out, int rank, int hidden,
                                       int batch) {
    const int b     = blockIdx.y;
    const int width = 2 * rank + 2 * hidden;
    const int i     = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= width) { return; }

    __nv_bfloat16* dst = z_out + static_cast<std::int64_t>(b) * width;
    if (i < rank) {
        dst[i] = state[static_cast<std::int64_t>(b) * rank + i];
    } else if (i < 2 * rank) {
        const int tok = prev_tokens[b];
        dst[i]        = __ldg(markov_w1 + static_cast<std::int64_t>(tok) * rank + (i - rank));
    } else if (i < 2 * rank + hidden) {
        const int h_idx = b * block_size + block_pos;
        dst[i]          = hidden_block[static_cast<std::int64_t>(h_idx) * hidden + (i - 2 * rank)];
    } else {
        dst[i] = f_rnn[static_cast<std::int64_t>(b) * hidden + (i - 2 * rank - hidden)];
    }
}

__global__ void markov_rnn_gates_kernel(const __nv_bfloat16* __restrict__ proj,
                                        __nv_bfloat16* __restrict__ state,
                                        __nv_bfloat16* __restrict__ latent, int rank, int batch) {
    const int i = threadIdx.x;
    const int b = blockIdx.x;
    if (i >= rank || b >= batch) { return; }

    const __nv_bfloat16* p = proj + static_cast<std::int64_t>(b) * (3 * rank);
    const float g_raw      = __bfloat162float(p[i]);
    const float c_raw      = __bfloat162float(p[rank + i]);
    const float o_raw      = __bfloat162float(p[2 * rank + i]);
    const float st         = __bfloat162float(state[static_cast<std::int64_t>(b) * rank + i]);

    const float gate       = 1.0f / (1.0f + expf(-g_raw));
    const float new_st     = gate * st + (1.0f - gate) * tanhf(c_raw);
    state[static_cast<std::int64_t>(b) * rank + i]  = __float2bfloat16_rn(new_st);
    latent[static_cast<std::int64_t>(b) * rank + i] = __float2bfloat16_rn(tanhf(o_raw));
}

__global__ void mtp_build_block_embed_kernel(const __nv_bfloat16* __restrict__ anchor_embed,
                                             const __nv_bfloat16* __restrict__ mask_embed,
                                             __nv_bfloat16* __restrict__ out, int hidden,
                                             int batch_size, int block_size) {
    const int i    = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.y; // 0 .. block_size - 1
    const int b    = blockIdx.z; // 0 .. batch_size - 1
    if (i >= hidden || b >= batch_size) { return; }
    const __nv_bfloat16 val =
        (slot == 0) ? anchor_embed[static_cast<std::int64_t>(b) * hidden + i] : __ldg(mask_embed + i);
    out[static_cast<std::int64_t>(b * block_size + slot) * hidden + i] = val;
}

__global__ void mtp_capture_slot_kernel(const __nv_bfloat16* __restrict__ h,
                                        __nv_bfloat16* __restrict__ caps, int slot, int hidden,
                                        int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= hidden || t >= tokens) { return; }
    caps[(static_cast<std::int64_t>(t) * 4 + slot) * hidden + i] =
        h[static_cast<std::int64_t>(t) * hidden + i];
}

__global__ void bf16_transpose_2d_kernel(const __nv_bfloat16* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst, int cols, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < cols && r < rows) {
        dst[static_cast<std::int64_t>(c) * rows + r] = src[static_cast<std::int64_t>(r) * cols + c];
    }
}

__global__ void mtp_extract_draft_hiddens_kernel(const __nv_bfloat16* __restrict__ in_blk,
                                                 __nv_bfloat16* __restrict__ out, int hidden,
                                                 int batch_size, int block_size) {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int pos = blockIdx.y; // 0 .. block_size - 2 (maps to source slot pos + 1)
    const int b   = blockIdx.z;
    const int d   = block_size - 1;
    if (i >= hidden || pos >= d || b >= batch_size) { return; }
    out[static_cast<std::int64_t>(b * d + pos) * hidden + i] =
        in_blk[static_cast<std::int64_t>(b * block_size + (pos + 1)) * hidden + i];
}

} // namespace

void inkling2_mtp_mix_rmsnorm(const Tensor& target_hiddens, const Tensor& fuse_weight,
                              std::int32_t row, const Tensor& norm_weight, float eps,
                              Tensor& out, cudaStream_t stream) {
    inkling2_mtp_mix_rmsnorm_indexed(target_hiddens, nullptr, nullptr, out.ne[1], fuse_weight, row,
                                     norm_weight, eps, out, stream);
}

void inkling2_mtp_mix_rmsnorm_indexed(const Tensor& target_hiddens, const std::int32_t* src_indices,
                                      const std::int32_t* dst_indices, std::int32_t count,
                                      const Tensor& fuse_weight, std::int32_t row,
                                      const Tensor& norm_weight, float eps, Tensor& out,
                                      cudaStream_t stream) {
    if (count <= 0) { return; }
    const int hidden = norm_weight.ne[0];
    if (hidden == 2304) {
        mtp_mix_rmsnorm_h2304_kernel<<<count, 576, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(target_hiddens.data), src_indices, dst_indices,
            static_cast<const __nv_bfloat16*>(fuse_weight.data), row,
            static_cast<const __nv_bfloat16*>(norm_weight.data), eps,
            static_cast<__nv_bfloat16*>(out.data), count);
    } else {
        mtp_mix_rmsnorm_kernel<<<count, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(target_hiddens.data), src_indices, dst_indices,
            static_cast<const __nv_bfloat16*>(fuse_weight.data), row,
            static_cast<const __nv_bfloat16*>(norm_weight.data), eps,
            static_cast<__nv_bfloat16*>(out.data), hidden, count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_ctx_kv_write(const Tensor& kv_ctx, const Tensor& k_norm_ctx_w,
                               const std::int32_t* positions, const std::int32_t* lane_ids,
                               std::int32_t tokens, std::int32_t num_kv_heads, float rope_theta,
                               float eps, const Inkling2KvTarget& kv, cudaStream_t stream) {
    mtp_ctx_kv_write_kernel<<<dim3(num_kv_heads, tokens), 64, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv_ctx.data),
        static_cast<const __nv_bfloat16*>(k_norm_ctx_w.data), positions, lane_ids, kv, num_kv_heads,
        rope_theta, eps);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_block_qkv_rope(const Tensor& qkv, const Tensor& q_norm_w, const Tensor& k_norm_w,
                                 const std::int32_t* positions, std::int32_t tokens,
                                 std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                 float rope_theta, float eps, Tensor& q_out, Tensor& k_out,
                                 Tensor& v_out, cudaStream_t stream) {
    mtp_block_qkv_rope_kernel<<<dim3(num_q_heads + num_kv_heads, tokens), 64, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv.data),
        static_cast<const __nv_bfloat16*>(q_norm_w.data),
        static_cast<const __nv_bfloat16*>(k_norm_w.data), positions, num_q_heads, num_kv_heads,
        rope_theta, eps, static_cast<__nv_bfloat16*>(q_out.data),
        static_cast<__nv_bfloat16*>(k_out.data), static_cast<__nv_bfloat16*>(v_out.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_block_attention(const Tensor& q, const Tensor& k_block, const Tensor& v_block,
                                  const Inkling2KvTarget& kv, const std::int32_t* lane_ids,
                                  const std::int32_t* ctx_lens, std::int32_t batch_size,
                                  std::int32_t block_size, std::int32_t num_q_heads,
                                  std::int32_t num_kv_heads, std::int32_t swa_window,
                                  Tensor& out, cudaStream_t stream) {
    if (block_size > 8) {
        throw std::invalid_argument("inkling2_mtp_block_attention: block_size > 8 unsupported");
    }
    mtp_block_attention_kernel<<<dim3(num_q_heads, batch_size), block_size * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k_block.data),
        static_cast<const __nv_bfloat16*>(v_block.data), kv, lane_ids, ctx_lens, batch_size,
        block_size, num_q_heads, num_kv_heads, swa_window, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_silu_mul_packed(const Tensor& gate_up, Tensor& out, cudaStream_t stream) {
    const int inter  = out.ne[0];
    const int tokens = out.ne[1];
    if (inter % 4 == 0) {
        const int inter4 = inter >> 2;
        silu_mul_packed_vec4_kernel<<<dim3((inter4 + 255) / 256, tokens), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate_up.data), static_cast<__nv_bfloat16*>(out.data),
            inter, tokens);
    } else {
        silu_mul_packed_kernel<<<dim3((inter + 255) / 256, tokens), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate_up.data), static_cast<__nv_bfloat16*>(out.data),
            inter, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_markov_rnn_prep(const Tensor& state, const Tensor& prev_tokens,
                                  const Tensor& markov_w1, const Tensor& hidden_block,
                                  std::int32_t block_pos, std::int32_t block_size,
                                  const Tensor& f_rnn, Tensor& z_out, cudaStream_t stream) {
    const int rank   = state.ne[0];
    const int batch  = state.ne[1];
    const int hidden = f_rnn.ne[0];
    const int width  = 2 * rank + 2 * hidden;
    markov_rnn_prep_kernel<<<dim3((width + 255) / 256, batch), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(state.data),
        static_cast<const std::int32_t*>(prev_tokens.data),
        static_cast<const __nv_bfloat16*>(markov_w1.data),
        static_cast<const __nv_bfloat16*>(hidden_block.data), block_pos, block_size,
        static_cast<const __nv_bfloat16*>(f_rnn.data), static_cast<__nv_bfloat16*>(z_out.data),
        rank, hidden, batch);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_markov_rnn_gates(const Tensor& proj, Tensor& state, Tensor& latent_out,
                                   cudaStream_t stream) {
    const int rank  = state.ne[0];
    const int batch = state.ne[1];
    markov_rnn_gates_kernel<<<batch, rank, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(proj.data), static_cast<__nv_bfloat16*>(state.data),
        static_cast<__nv_bfloat16*>(latent_out.data), rank, batch);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_build_block_embed(const Tensor& anchor_embed, const Tensor& mask_embed,
                                    std::int32_t batch_size, std::int32_t block_size,
                                    Tensor& out, cudaStream_t stream) {
    const int hidden = mask_embed.ne[0];
    mtp_build_block_embed_kernel<<<dim3((hidden + 255) / 256, block_size, batch_size), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(anchor_embed.data),
        static_cast<const __nv_bfloat16*>(mask_embed.data), static_cast<__nv_bfloat16*>(out.data),
        hidden, batch_size, block_size);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_capture_slot(const Tensor& h, Tensor& target_caps, std::int32_t slot,
                               cudaStream_t stream) {
    const int hidden = h.ne[0];
    const int tokens = h.ne[1];
    mtp_capture_slot_kernel<<<dim3((hidden + 255) / 256, tokens), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(h.data),
        static_cast<__nv_bfloat16*>(target_caps.data), slot, hidden, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_bf16_transpose_2d(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    const int cols = src.ne[0];
    const int rows = src.ne[1];
    const dim3 block(16, 16);
    const dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    bf16_transpose_2d_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src.data), static_cast<__nv_bfloat16*>(dst.data), cols,
        rows);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_mtp_extract_draft_hiddens(const Tensor& hidden_block, std::int32_t batch_size,
                                        std::int32_t block_size, Tensor& out, cudaStream_t stream) {
    const int hidden = hidden_block.ne[0];
    const int d      = block_size - 1;
    mtp_extract_draft_hiddens_kernel<<<dim3((hidden + 255) / 256, d, batch_size), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_block.data),
        static_cast<__nv_bfloat16*>(out.data), hidden, batch_size, block_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
