#include "ninfer/ops/inkling2.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
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

__device__ __forceinline__ void load_bf16x8(const __nv_bfloat16* p, float (&out)[8]) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(p));
    const auto* b   = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(b[j]);
        out[2 * j]     = f.x;
        out[2 * j + 1] = f.y;
    }
}

__device__ __forceinline__ void store_bf16x8(__nv_bfloat16* p, const float (&v)[8]) {
    uint4 out;
    auto* dst = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int j = 0; j < 4; ++j) { dst[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]); }
    *reinterpret_cast<uint4*>(p) = out;
}

__device__ __forceinline__ uint4 ld_global_cg_u4(const void* ptr) {
    uint4 bits;
    asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(ptr));
    return bits;
}

__device__ __forceinline__ float ld_cg_f32_vol(const float* ptr) {
    float val;
    asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

// 1. mix_rmsnorm: softmax(fuse_weight[row, 0..3]) weighted sum of 4 target hiddens + plain RMSNorm
// Fast specialized H==2304 path (288 threads = 9 warps, 8 BF16 = uint4/thread!)
__global__ void mtp_mix_rmsnorm_h2304_kernel(const __nv_bfloat16* __restrict__ targets,
                                             const std::int32_t* __restrict__ src_indices,
                                             const std::int32_t* __restrict__ dst_indices,
                                             const __nv_bfloat16* __restrict__ fuse_weight,
                                             int row_idx, const __nv_bfloat16* __restrict__ norm_w,
                                             float eps, __nv_bfloat16* __restrict__ out,
                                             int count) {
    __shared__ float s_warp[9];
    const int t = blockIdx.x;
    if (t >= count) { return; }
    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;

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

    float acc[8] = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float v[8];
        load_bf16x8(tbase + s * 2304, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[j] = fmaf(fw[s], v[j], acc[j]); }
    }
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) { sq = fmaf(acc[j], acc[j], sq); }
    sq = warp_sum(sq);
    if (lane == 0) { s_warp[warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < 9) ? s_warp[lane] : 0.0f;
        sum       = warp_sum(sum);
        if (lane == 0) { s_warp[0] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    const float inv = s_warp[0];
    float nw[8];
    load_bf16x8(norm_w + idx, nw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j] = acc[j] * inv * nw[j]; }
    store_bf16x8(out + static_cast<std::int64_t>(dst_t) * 2304 + idx, acc);
}

// Batched 3-layer ctx mix_rmsnorm + f_rnn mix_rmsnorm in ONE launch (grid = (max(ctx, frnn), 4), 288t)
__global__ void mtp_mix_rmsnorm_batch3_and_frnn_h2304_kernel(
    const __nv_bfloat16* __restrict__ targets, const std::int32_t* __restrict__ ctx_src_indices,
    int ctx_count, const std::int32_t* __restrict__ frnn_src_indices,
    const std::int32_t* __restrict__ frnn_dst_indices, int frnn_count,
    const __nv_bfloat16* __restrict__ fuse_weight, const __nv_bfloat16* __restrict__ norm0,
    const __nv_bfloat16* __restrict__ norm1, const __nv_bfloat16* __restrict__ norm2,
    const __nv_bfloat16* __restrict__ frnn_norm, float eps, __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1, __nv_bfloat16* __restrict__ out2,
    __nv_bfloat16* __restrict__ out_frnn) {
    __shared__ float s_warp[9];
    const int t       = blockIdx.x;
    const int row_idx = blockIdx.y; // 0, 1, 2 = ctx layers; 3 = f_rnn
    int src_t = 0, dst_t = 0;
    const __nv_bfloat16* norm_w = nullptr;
    __nv_bfloat16* out_p        = nullptr;

    if (row_idx < 3) {
        if (t >= ctx_count) { return; }
        src_t  = (ctx_src_indices != nullptr) ? ctx_src_indices[t] : t;
        dst_t  = t;
        norm_w = (row_idx == 0) ? norm0 : ((row_idx == 1) ? norm1 : norm2);
        out_p  = (row_idx == 0) ? out0 : ((row_idx == 1) ? out1 : out2);
    } else {
        if (t >= frnn_count) { return; }
        src_t  = (frnn_src_indices != nullptr) ? frnn_src_indices[t] : t;
        dst_t  = (frnn_dst_indices != nullptr) ? frnn_dst_indices[t] : t;
        norm_w = frnn_norm;
        out_p  = out_frnn;
    }

    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;

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

    const __nv_bfloat16* tbase = targets + static_cast<std::int64_t>(src_t) * 4 * 2304 + idx;
    float acc[8]               = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float v[8];
        load_bf16x8(tbase + s * 2304, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[j] = fmaf(fw[s], v[j], acc[j]); }
    }
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) { sq = fmaf(acc[j], acc[j], sq); }
    sq = warp_sum(sq);
    if (lane == 0) { s_warp[warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < 9) ? s_warp[lane] : 0.0f;
        sum       = warp_sum(sum);
        if (lane == 0) { s_warp[0] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    const float inv = s_warp[0];
    float nw[8];
    load_bf16x8(norm_w + idx, nw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc[j] = acc[j] * inv * nw[j]; }
    store_bf16x8(out_p + static_cast<std::int64_t>(dst_t) * 2304 + idx, acc);
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

// Batched 3-layer Context KV write in ONE launch (grid = (num_kv_heads, tokens, 3), block = 64)
__global__ void mtp_ctx_kv_write_3_kernel(
    const __nv_bfloat16* __restrict__ kv_ctx0, const __nv_bfloat16* __restrict__ k_norm0,
    Inkling2KvTarget kv0, const __nv_bfloat16* __restrict__ kv_ctx1,
    const __nv_bfloat16* __restrict__ k_norm1, Inkling2KvTarget kv1,
    const __nv_bfloat16* __restrict__ kv_ctx2, const __nv_bfloat16* __restrict__ k_norm2,
    Inkling2KvTarget kv2, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ lane_ids, int num_kv_heads, float rope_theta, float eps) {
    const int h       = blockIdx.x;
    const int t       = blockIdx.y;
    const int li      = blockIdx.z; // 0, 1, 2
    const int d       = threadIdx.x; // 0..63
    const int kv_dim  = num_kv_heads * kHeadDim;
    const int lane_id = (lane_ids != nullptr) ? lane_ids[t] : 0;
    const int pos     = positions[t];

    const __nv_bfloat16* kv_ctx   = (li == 0) ? kv_ctx0 : ((li == 1) ? kv_ctx1 : kv_ctx2);
    const __nv_bfloat16* k_norm_w = (li == 0) ? k_norm0 : ((li == 1) ? k_norm1 : k_norm2);
    const Inkling2KvTarget kv     = (li == 0) ? kv0 : ((li == 1) ? kv1 : kv2);

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

// 4. Adaptive Single-Launch Last-CTA Q-Head Split-KV Non-Causal Block Attention over [ctx, block]
//    grid = (num_q_heads, batch_size * block_size, num_splits), block = 128 (4 warps)
//    - Incremental ring slot (ZERO % ring_cap inside inner loop) + cached global page lookup!
//    - Short context or split_scratch==nullptr: eff_splits=1 (skips scratch & atomics!).
//    - Long context: Warp-0 SMEM Last-CTA reduction (saturates all 80 SMs even at 4K+!).
template <int kWarps>
__global__ __launch_bounds__(kWarps * 32, 8) void mtp_block_attention_qhead_adaptive_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_block,
    const __nv_bfloat16* __restrict__ v_block, Inkling2KvTarget kv,
    const std::int32_t* __restrict__ lane_ids, const std::int32_t* __restrict__ ctx_lens,
    int block_size, int num_q_heads, int num_kv_heads, int swa_window, int num_splits,
    int min_split_k, float* __restrict__ split_scratch, std::int32_t* __restrict__ split_done,
    __nv_bfloat16* __restrict__ out) {
    constexpr int kSpStride = kHeadDim + 4; // 132 floats = 528B (16-byte aligned)

    const int qh       = blockIdx.x;
    const int tok      = blockIdx.y; // 0 .. batch_size * block_size - 1
    const int split_id = blockIdx.z;

    const int b       = tok / block_size;
    const int lane    = lane_ids[b];
    const int ctx_len = ctx_lens[b];
    const int ctx_beg = kv.is_local ? max(0, ctx_len - swa_window) : 0;
    const int ctx_k   = max(0, ctx_len - ctx_beg);
    const int total_k = ctx_k + block_size;

    const int eff_splits =
        (num_splits <= 1) ? 1 : max(1, min(num_splits, (total_k + min_split_k - 1) / min_split_k));
    if (split_id >= eff_splits) { return; }

    const int chunk  = (total_k + eff_splits - 1) / eff_splits;
    const int ki_beg = split_id * chunk;
    const int ki_end = min(total_k, ki_beg + chunk);
    const int kv_h   = qh / (num_q_heads / num_kv_heads);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int wlane  = tid & 31;
    const int d_base = wlane * 4;

    const __nv_bfloat16* qp =
        q + (static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + d_base;
    const uint2 qraw = __ldg(reinterpret_cast<const uint2*>(qp));
    const auto* qb   = reinterpret_cast<const __nv_bfloat162*>(&qraw);
    const float2 qf0 = __bfloat1622float2(qb[0]);
    const float2 qf1 = __bfloat1622float2(qb[1]);
    const float q0   = qf0.x * kAttnScale;
    const float q1   = qf0.y * kAttnScale;
    const float q2   = qf1.x * kAttnScale;
    const float q3   = qf1.y * kAttnScale;

    float m  = -std::numeric_limits<float>::infinity();
    float l  = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    // 1. Context KV range [ki_beg, min(ki_end, ctx_k))
    const int ctx_end_loc = min(ki_end, ctx_k);
    if (ki_beg < ctx_end_loc) {
        const int kpos_beg = ctx_beg + ki_beg;
        const int kpos_end = ctx_beg + ctx_end_loc;
        int kpos           = kpos_beg + warp;
        if (kv.is_local) {
            const auto* rk_base          = static_cast<const __nv_bfloat16*>(kv.ring_k);
            const auto* rv_base          = static_cast<const __nv_bfloat16*>(kv.ring_v);
            const std::int64_t lane_base = static_cast<std::int64_t>(lane) * kv.ring_cap;
            int slot                     = (kpos < kv.ring_cap) ? kpos : (kpos % kv.ring_cap);
            for (; kpos < kpos_end; kpos += kWarps) {
                const std::int64_t off =
                    ((lane_base + slot) * num_kv_heads + kv_h) * kHeadDim + d_base;
                const uint2 kraw = __ldg(reinterpret_cast<const uint2*>(rk_base + off));
                const uint2 vraw = __ldg(reinterpret_cast<const uint2*>(rv_base + off));
                const auto* kb   = reinterpret_cast<const __nv_bfloat162*>(&kraw);
                const auto* vb   = reinterpret_cast<const __nv_bfloat162*>(&vraw);
                const float2 kf0 = __bfloat1622float2(kb[0]);
                const float2 kf1 = __bfloat1622float2(kb[1]);
                const float2 vf0 = __bfloat1622float2(vb[0]);
                const float2 vf1 = __bfloat1622float2(vb[1]);

                float dot = q0 * kf0.x + q1 * kf0.y + q2 * kf1.x + q3 * kf1.y;
                dot       = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);

                const float new_m = fmaxf(m, dot);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                        ? 0.0f
                                        : __expf(m - new_m);
                const float p     = __expf(dot - new_m);
                l                 = l * alpha + p;
                a0                = fmaf(p, vf0.x, a0 * alpha);
                a1                = fmaf(p, vf0.y, a1 * alpha);
                a2                = fmaf(p, vf1.x, a2 * alpha);
                a3                = fmaf(p, vf1.y, a3 * alpha);
                m                 = new_m;

                slot += kWarps;
                if (slot >= kv.ring_cap) { slot -= kv.ring_cap; }
            }
        } else {
            const auto* pk_base       = static_cast<const __nv_bfloat16*>(kv.paged_k);
            const auto* pv_base       = static_cast<const __nv_bfloat16*>(kv.paged_v);
            const std::int32_t* bt    = kv.block_tables + static_cast<std::int64_t>(lane) * kv.max_logical_pages;
            int cached_lpage          = -1;
            std::int64_t page_kv_base = 0;
            for (; kpos < kpos_end; kpos += kWarps) {
                const int lpage = kpos >> 6;
                if (lpage != cached_lpage) {
                    cached_lpage = lpage;
                    const int pp = __ldg(bt + lpage);
                    page_kv_base = (static_cast<std::int64_t>(pp) * num_kv_heads + kv_h) * kPageSize;
                }
                const std::int64_t off = (page_kv_base + (kpos & 63)) * kHeadDim + d_base;
                const uint2 kraw       = __ldg(reinterpret_cast<const uint2*>(pk_base + off));
                const uint2 vraw       = __ldg(reinterpret_cast<const uint2*>(pv_base + off));
                const auto* kb         = reinterpret_cast<const __nv_bfloat162*>(&kraw);
                const auto* vb         = reinterpret_cast<const __nv_bfloat162*>(&vraw);
                const float2 kf0       = __bfloat1622float2(kb[0]);
                const float2 kf1       = __bfloat1622float2(kb[1]);
                const float2 vf0       = __bfloat1622float2(vb[0]);
                const float2 vf1       = __bfloat1622float2(vb[1]);

                float dot = q0 * kf0.x + q1 * kf0.y + q2 * kf1.x + q3 * kf1.y;
                dot       = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);

                const float new_m = fmaxf(m, dot);
                const float alpha = (m == -std::numeric_limits<float>::infinity())
                                        ? 0.0f
                                        : __expf(m - new_m);
                const float p     = __expf(dot - new_m);
                l                 = l * alpha + p;
                a0                = fmaf(p, vf0.x, a0 * alpha);
                a1                = fmaf(p, vf0.y, a1 * alpha);
                a2                = fmaf(p, vf1.x, a2 * alpha);
                a3                = fmaf(p, vf1.y, a3 * alpha);
                m                 = new_m;
            }
        }
    }

    // 2. Non-causal block KV range [max(0, ki_beg - ctx_k), ki_end - ctx_k)
    const int bi_beg = max(0, ki_beg - ctx_k);
    const int bi_end = ki_end - ctx_k;
    for (int bi = bi_beg + warp; bi < bi_end; bi += kWarps) {
        const int btok = b * block_size + bi;
        const std::int64_t boff =
            (static_cast<std::int64_t>(btok) * num_kv_heads + kv_h) * kHeadDim + d_base;
        const uint2 kraw = __ldg(reinterpret_cast<const uint2*>(k_block + boff));
        const uint2 vraw = __ldg(reinterpret_cast<const uint2*>(v_block + boff));
        const auto* kb   = reinterpret_cast<const __nv_bfloat162*>(&kraw);
        const auto* vb   = reinterpret_cast<const __nv_bfloat162*>(&vraw);
        const float2 kf0 = __bfloat1622float2(kb[0]);
        const float2 kf1 = __bfloat1622float2(kb[1]);
        const float2 vf0 = __bfloat1622float2(vb[0]);
        const float2 vf1 = __bfloat1622float2(vb[1]);

        float dot = q0 * kf0.x + q1 * kf0.y + q2 * kf1.x + q3 * kf1.y;
        dot       = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);

        const float new_m = fmaxf(m, dot);
        const float alpha = (m == -std::numeric_limits<float>::infinity())
                                ? 0.0f
                                : __expf(m - new_m);
        const float p     = __expf(dot - new_m);
        l                 = l * alpha + p;
        a0                = fmaf(p, vf0.x, a0 * alpha);
        a1                = fmaf(p, vf0.y, a1 * alpha);
        a2                = fmaf(p, vf1.x, a2 * alpha);
        a3                = fmaf(p, vf1.y, a3 * alpha);
        m                 = new_m;
    }

    __shared__ float s_m[kWarps];
    __shared__ float s_l[kWarps];
    __shared__ float s_o[kWarps][kHeadDim];
    __shared__ float s_sc[32];
    __shared__ float s_inv_l;
    __shared__ int s_is_last;

    if (wlane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    s_o[warp][d_base + 0] = a0;
    s_o[warp][d_base + 1] = a1;
    s_o[warp][d_base + 2] = a2;
    s_o[warp][d_base + 3] = a3;
    __syncthreads();

    if (eff_splits == 1) {
        float gm = -std::numeric_limits<float>::infinity();
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { gm = fmaxf(gm, s_m[w]); }
        float gl = 0.0f;
        float go = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float wm = s_m[w];
            if (wm > -std::numeric_limits<float>::infinity()) {
                const float sc = __expf(wm - gm);
                gl += s_l[w] * sc;
                go += s_o[w][tid] * sc;
            }
        }
        out[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + tid] =
            __float2bfloat16_rn(go / gl);
        return;
    }

    // Multi-split path: write partial (go, gm, gl) to L2-hot split_scratch, Last CTA finishes!
    {
        float gm = -std::numeric_limits<float>::infinity();
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { gm = fmaxf(gm, s_m[w]); }
        float gl = 0.0f;
        float go = 0.0f;
        if (gm > -std::numeric_limits<float>::infinity()) {
#pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                const float wm = s_m[w];
                if (wm > -std::numeric_limits<float>::infinity()) {
                    const float sc = __expf(wm - gm);
                    gl += s_l[w] * sc;
                    go += s_o[w][tid] * sc;
                }
            }
        }
        float* dst = split_scratch +
                     ((static_cast<std::int64_t>(tok) * num_q_heads + qh) * num_splits + split_id) *
                         kSpStride;
        dst[tid] = go;
        if (tid == 0) {
            dst[kHeadDim]     = gm;
            dst[kHeadDim + 1] = gl;
        }
    }

    __threadfence();
    __syncthreads();
    const int done_idx = tok * num_q_heads + qh;
    if (tid == 0) { s_is_last = (atomicAdd(split_done + done_idx, 1) == eff_splits - 1) ? 1 : 0; }
    __syncthreads();
    if (!s_is_last) { return; }
    if (tid == 0) { split_done[done_idx] = 0; }

    const float* base = split_scratch +
                        (static_cast<std::int64_t>(tok) * num_q_heads + qh) * num_splits * kSpStride;
    if (warp == 0) {
        const float my_m = (wlane < eff_splits)
                               ? ld_cg_f32_vol(base + wlane * kSpStride + kHeadDim)
                               : -std::numeric_limits<float>::infinity();
        const float my_l = (wlane < eff_splits)
                               ? ld_cg_f32_vol(base + wlane * kSpStride + kHeadDim + 1)
                               : 0.0f;
        float gm = my_m;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFFu, gm, off));
        }
        const float sc =
            (wlane < eff_splits && my_m > -std::numeric_limits<float>::infinity())
                ? __expf(my_m - gm)
                : 0.0f;
        if (wlane < eff_splits) { s_sc[wlane] = sc; }
        const float gl = warp_sum(my_l * sc);
        if (wlane == 0) { s_inv_l = 1.0f / gl; }
    }
    __syncthreads();

    float go = 0.0f;
    for (int s = 0; s < eff_splits; ++s) {
        const float sc = s_sc[s];
        if (sc > 0.0f) { go = fmaf(ld_cg_f32_vol(base + s * kSpStride + tid), sc, go); }
    }
    out[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + tid] =
        __float2bfloat16_rn(go * s_inv_l);
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

// 7. Fused MTP Final RMSNorm (in-place) + Extract Draft Hiddens + Zero RNN State (1 launch, 288t!)
__global__ void mtp_final_norm_extract_zero_h2304_kernel(
    __nv_bfloat16* __restrict__ hidden_block, const __nv_bfloat16* __restrict__ norm_w, float eps,
    int batch_size, int block_size, __nv_bfloat16* __restrict__ draft_out,
    __nv_bfloat16* __restrict__ rnn_state, int rnn_rank) {
    __shared__ float s_warp[9];
    const int t = blockIdx.x; // 0 .. batch_size * block_size - 1
    if (t >= batch_size * block_size) { return; }

    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;

    const int b    = t / block_size;
    const int slot = t - b * block_size;

    // Slot 0 of each batch lane also zeroes rnn_state[b, 0..rnn_rank-1] in parallel!
    if (slot == 0) {
        const __nv_bfloat16 zero = __float2bfloat16(0.0f);
        for (int r = tid; r < rnn_rank; r += 288) {
            rnn_state[static_cast<std::int64_t>(b) * rnn_rank + r] = zero;
        }
    }

    __nv_bfloat16* row = hidden_block + static_cast<std::int64_t>(t) * 2304 + idx;
    float v[8];
    load_bf16x8(row, v);
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) { sq = fmaf(v[j], v[j], sq); }
    sq = warp_sum(sq);
    if (lane == 0) { s_warp[warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < 9) ? s_warp[lane] : 0.0f;
        sum       = warp_sum(sum);
        if (lane == 0) { s_warp[0] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    const float inv = s_warp[0];

    float nw[8];
    load_bf16x8(norm_w + idx, nw);
    uint4 out_pack;
    auto* dst = reinterpret_cast<__nv_bfloat162*>(&out_pack);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        dst[j] = __floats2bfloat162_rn(v[2 * j] * inv * nw[2 * j],
                                       v[2 * j + 1] * inv * nw[2 * j + 1]);
    }
    *reinterpret_cast<uint4*>(row) = out_pack;

    if (slot >= 1) {
        const int d = block_size - 1;
        __nv_bfloat16* drow =
            draft_out + static_cast<std::int64_t>(b * d + (slot - 1)) * 2304 + idx;
        *reinterpret_cast<uint4*>(drow) = out_pack;
    }
}

// 8. Fused Markov RNN Step (prep + [width -> 3*rank] BF16 Linear + Gates) in ONE launch!
//    grid = (rank, batch), block = 128 (4 warps; 640 uint4s / 128 = 5 clean iters/thread!).
//    Reads virtual z ONCE across all 3 rows (g, c, o); disjoint state_in / state_out = zero races!
__global__ __launch_bounds__(128, 6) void mtp_markov_rnn_step_u4_kernel(
    const __nv_bfloat16* __restrict__ state_in, const std::int32_t* __restrict__ prev_tokens,
    const __nv_bfloat16* __restrict__ markov_w1, const __nv_bfloat16* __restrict__ hidden_block,
    int block_pos, int block_size, const __nv_bfloat16* __restrict__ f_rnn,
    const __nv_bfloat16* __restrict__ proj_w, const __nv_bfloat16* __restrict__ proj_b,
    __nv_bfloat16* __restrict__ state_out, __nv_bfloat16* __restrict__ latent_out, int rank,
    int hidden) {
    __shared__ float s_acc[3][4];

    const int i    = blockIdx.x; // 0..rank-1
    const int b    = blockIdx.y; // 0..batch-1
    const int tid  = threadIdx.x; // 0..127
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int width    = 2 * rank + 2 * hidden;
    const int r8       = rank >> 3;
    const int h8       = hidden >> 3;
    const int w8       = 2 * r8 + 2 * h8;
    const int prev_tok = prev_tokens[b];

    const uint4* z_st = reinterpret_cast<const uint4*>(state_in + static_cast<std::int64_t>(b) * rank);
    const uint4* z_w1 =
        reinterpret_cast<const uint4*>(markov_w1 + static_cast<std::int64_t>(prev_tok) * rank);
    const uint4* z_hb = reinterpret_cast<const uint4*>(
        hidden_block + static_cast<std::int64_t>(b * block_size + block_pos) * hidden);
    const uint4* z_fr = reinterpret_cast<const uint4*>(f_rnn + static_cast<std::int64_t>(b) * hidden);

    const uint4* wg = reinterpret_cast<const uint4*>(proj_w + static_cast<std::int64_t>(i) * width);
    const uint4* wc =
        reinterpret_cast<const uint4*>(proj_w + static_cast<std::int64_t>(rank + i) * width);
    const uint4* wo =
        reinterpret_cast<const uint4*>(proj_w + static_cast<std::int64_t>(2 * rank + i) * width);

    float ag = 0.0f, ac = 0.0f, ao = 0.0f;

    for (int k8 = tid; k8 < w8; k8 += 128) {
        uint4 zv;
        if (k8 < r8) {
            zv = __ldg(z_st + k8);
        } else if (k8 < 2 * r8) {
            zv = __ldg(z_w1 + (k8 - r8));
        } else if (k8 < 2 * r8 + h8) {
            zv = __ldg(z_hb + (k8 - 2 * r8));
        } else {
            zv = __ldg(z_fr + (k8 - 2 * r8 - h8));
        }
        const uint4 gwv = ld_global_cg_u4(wg + k8);
        const uint4 cwv = ld_global_cg_u4(wc + k8);
        const uint4 owv = ld_global_cg_u4(wo + k8);

        const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&zv);
        const auto* gb = reinterpret_cast<const __nv_bfloat162*>(&gwv);
        const auto* cb = reinterpret_cast<const __nv_bfloat162*>(&cwv);
        const auto* ob = reinterpret_cast<const __nv_bfloat162*>(&owv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 xf = __bfloat1622float2(xb[j]);
            const float2 gf = __bfloat1622float2(gb[j]);
            const float2 cf = __bfloat1622float2(cb[j]);
            const float2 of = __bfloat1622float2(ob[j]);
            ag              = fmaf(gf.x, xf.x, fmaf(gf.y, xf.y, ag));
            ac              = fmaf(cf.x, xf.x, fmaf(cf.y, xf.y, ac));
            ao              = fmaf(of.x, xf.x, fmaf(of.y, xf.y, ao));
        }
    }

    ag = warp_sum(ag);
    ac = warp_sum(ac);
    ao = warp_sum(ao);
    if (lane == 0) {
        s_acc[0][warp] = ag;
        s_acc[1][warp] = ac;
        s_acc[2][warp] = ao;
    }
    __syncthreads();

    if (tid == 0) {
        const float sum_g =
            (s_acc[0][0] + s_acc[0][1]) + (s_acc[0][2] + s_acc[0][3]) +
            __bfloat162float(__ldg(proj_b + i));
        const float sum_c =
            (s_acc[1][0] + s_acc[1][1]) + (s_acc[1][2] + s_acc[1][3]) +
            __bfloat162float(__ldg(proj_b + rank + i));
        const float sum_o =
            (s_acc[2][0] + s_acc[2][1]) + (s_acc[2][2] + s_acc[2][3]) +
            __bfloat162float(__ldg(proj_b + 2 * rank + i));

        // Exact BF16 rounding boundary matching unfused linear(BF16) -> markov_rnn_gates!
        const float g_raw = __bfloat162float(__float2bfloat16_rn(sum_g));
        const float c_raw = __bfloat162float(__float2bfloat16_rn(sum_c));
        const float o_raw = __bfloat162float(__float2bfloat16_rn(sum_o));
        const float st    = __bfloat162float(state_in[static_cast<std::int64_t>(b) * rank + i]);

        const float gate   = 1.0f / (1.0f + expf(-g_raw));
        const float new_st = gate * st + (1.0f - gate) * tanhf(c_raw);
        state_out[static_cast<std::int64_t>(b) * rank + i]  = __float2bfloat16_rn(new_st);
        latent_out[static_cast<std::int64_t>(b) * rank + i] = __float2bfloat16_rn(tanhf(o_raw));
    }
}

__global__ void mtp_markov_rnn_step_generic_kernel(
    const __nv_bfloat16* __restrict__ state_in, const std::int32_t* __restrict__ prev_tokens,
    const __nv_bfloat16* __restrict__ markov_w1, const __nv_bfloat16* __restrict__ hidden_block,
    int block_pos, int block_size, const __nv_bfloat16* __restrict__ f_rnn,
    const __nv_bfloat16* __restrict__ proj_w, const __nv_bfloat16* __restrict__ proj_b,
    __nv_bfloat16* __restrict__ state_out, __nv_bfloat16* __restrict__ latent_out, int rank,
    int hidden) {
    __shared__ float s_acc[3][4];
    const int i        = blockIdx.x;
    const int b        = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int width    = 2 * rank + 2 * hidden;
    const int prev_tok = prev_tokens[b];

    const __nv_bfloat16* wg = proj_w + static_cast<std::int64_t>(i) * width;
    const __nv_bfloat16* wc = proj_w + static_cast<std::int64_t>(rank + i) * width;
    const __nv_bfloat16* wo = proj_w + static_cast<std::int64_t>(2 * rank + i) * width;

    float ag = 0.0f, ac = 0.0f, ao = 0.0f;
    for (int k = tid; k < width; k += 128) {
        float x;
        if (k < rank) {
            x = __bfloat162float(state_in[static_cast<std::int64_t>(b) * rank + k]);
        } else if (k < 2 * rank) {
            x = __bfloat162float(__ldg(markov_w1 + static_cast<std::int64_t>(prev_tok) * rank + (k - rank)));
        } else if (k < 2 * rank + hidden) {
            x = __bfloat162float(
                hidden_block[static_cast<std::int64_t>(b * block_size + block_pos) * hidden + (k - 2 * rank)]);
        } else {
            x = __bfloat162float(f_rnn[static_cast<std::int64_t>(b) * hidden + (k - 2 * rank - hidden)]);
        }
        ag = fmaf(__bfloat162float(__ldg(wg + k)), x, ag);
        ac = fmaf(__bfloat162float(__ldg(wc + k)), x, ac);
        ao = fmaf(__bfloat162float(__ldg(wo + k)), x, ao);
    }
    ag = warp_sum(ag);
    ac = warp_sum(ac);
    ao = warp_sum(ao);
    if (lane == 0) {
        s_acc[0][warp] = ag;
        s_acc[1][warp] = ac;
        s_acc[2][warp] = ao;
    }
    __syncthreads();
    if (tid == 0) {
        const float g_raw = __bfloat162float(__float2bfloat16_rn(
            (s_acc[0][0] + s_acc[0][1]) + (s_acc[0][2] + s_acc[0][3]) + __bfloat162float(__ldg(proj_b + i))));
        const float c_raw = __bfloat162float(__float2bfloat16_rn(
            (s_acc[1][0] + s_acc[1][1]) + (s_acc[1][2] + s_acc[1][3]) + __bfloat162float(__ldg(proj_b + rank + i))));
        const float o_raw = __bfloat162float(__float2bfloat16_rn(
            (s_acc[2][0] + s_acc[2][1]) + (s_acc[2][2] + s_acc[2][3]) + __bfloat162float(__ldg(proj_b + 2 * rank + i))));
        const float st     = __bfloat162float(state_in[static_cast<std::int64_t>(b) * rank + i]);
        const float gate   = 1.0f / (1.0f + expf(-g_raw));
        state_out[static_cast<std::int64_t>(b) * rank + i]  = __float2bfloat16_rn(gate * st + (1.0f - gate) * tanhf(c_raw));
        latent_out[static_cast<std::int64_t>(b) * rank + i] = __float2bfloat16_rn(tanhf(o_raw));
    }
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
        mtp_mix_rmsnorm_h2304_kernel<<<count, 288, 0, stream>>>(
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

void inkling2_mtp_mix_rmsnorm_batch3_and_frnn(
    const Tensor& target_hiddens, const std::int32_t* ctx_src_indices, std::int32_t ctx_count,
    const std::int32_t* frnn_src_indices, const std::int32_t* frnn_dst_indices,
    std::int32_t frnn_count, const Tensor& fuse_weight,
    const Tensor& norm0, const Tensor& norm1, const Tensor& norm2, const Tensor& frnn_norm,
    float eps, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out_frnn, cudaStream_t stream) {
    const int max_cnt = std::max(ctx_count, frnn_count);
    if (max_cnt <= 0) { return; }
    const int hidden = norm0.ne[0];
    if (hidden == 2304) {
        mtp_mix_rmsnorm_batch3_and_frnn_h2304_kernel<<<dim3(max_cnt, 4), 288, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(target_hiddens.data), ctx_src_indices, ctx_count,
            frnn_src_indices, frnn_dst_indices, frnn_count,
            static_cast<const __nv_bfloat16*>(fuse_weight.data),
            static_cast<const __nv_bfloat16*>(norm0.data),
            static_cast<const __nv_bfloat16*>(norm1.data),
            static_cast<const __nv_bfloat16*>(norm2.data),
            static_cast<const __nv_bfloat16*>(frnn_norm.data), eps,
            static_cast<__nv_bfloat16*>(out0.data), static_cast<__nv_bfloat16*>(out1.data),
            static_cast<__nv_bfloat16*>(out2.data), static_cast<__nv_bfloat16*>(out_frnn.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    inkling2_mtp_mix_rmsnorm_indexed(target_hiddens, ctx_src_indices, nullptr, ctx_count,
                                     fuse_weight, 0, norm0, eps, out0, stream);
    inkling2_mtp_mix_rmsnorm_indexed(target_hiddens, ctx_src_indices, nullptr, ctx_count,
                                     fuse_weight, 1, norm1, eps, out1, stream);
    inkling2_mtp_mix_rmsnorm_indexed(target_hiddens, ctx_src_indices, nullptr, ctx_count,
                                     fuse_weight, 2, norm2, eps, out2, stream);
    inkling2_mtp_mix_rmsnorm_indexed(target_hiddens, frnn_src_indices, frnn_dst_indices, frnn_count,
                                     fuse_weight, 3, frnn_norm, eps, out_frnn, stream);
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

void inkling2_mtp_ctx_kv_write_3(const Tensor& kv_ctx0, const Tensor& k_norm0, const Inkling2KvTarget& kv0,
                                 const Tensor& kv_ctx1, const Tensor& k_norm1, const Inkling2KvTarget& kv1,
                                 const Tensor& kv_ctx2, const Tensor& k_norm2, const Inkling2KvTarget& kv2,
                                 const std::int32_t* positions, const std::int32_t* lane_ids,
                                 std::int32_t tokens, std::int32_t num_kv_heads, float rope_theta,
                                 float eps, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    mtp_ctx_kv_write_3_kernel<<<dim3(num_kv_heads, tokens, 3), 64, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv_ctx0.data),
        static_cast<const __nv_bfloat16*>(k_norm0.data), kv0,
        static_cast<const __nv_bfloat16*>(kv_ctx1.data),
        static_cast<const __nv_bfloat16*>(k_norm1.data), kv1,
        static_cast<const __nv_bfloat16*>(kv_ctx2.data),
        static_cast<const __nv_bfloat16*>(k_norm2.data), kv2, positions, lane_ids, num_kv_heads,
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
                                  Tensor& out, cudaStream_t stream, float* split_scratch,
                                  std::int32_t max_splits, std::int32_t* split_done) {
    if (block_size <= 0 || block_size > 8) {
        throw std::invalid_argument("inkling2_mtp_block_attention: block_size must be in 1..8");
    }
    const int total_tokens = batch_size * block_size;
    const bool use_splits  = (split_scratch != nullptr && split_done != nullptr && max_splits > 1);
    const int num_splits   = use_splits ? std::min<int>(max_splits, 16) : 1;
    const int min_split_k  = (total_tokens <= 2) ? 64 : ((total_tokens <= 8) ? 128 : 256);

    mtp_block_attention_qhead_adaptive_kernel<4>
        <<<dim3(num_q_heads, total_tokens, num_splits), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k_block.data),
            static_cast<const __nv_bfloat16*>(v_block.data), kv, lane_ids, ctx_lens, block_size,
            num_q_heads, num_kv_heads, swa_window, num_splits, min_split_k, split_scratch,
            split_done, static_cast<__nv_bfloat16*>(out.data));
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

void inkling2_mtp_markov_rnn_step(const Tensor& state_in, const Tensor& prev_tokens,
                                  const Tensor& markov_w1, const Tensor& hidden_block,
                                  std::int32_t block_pos, std::int32_t block_size,
                                  const Tensor& f_rnn, const Tensor& proj_w, const Tensor& proj_b,
                                  Tensor& state_out, Tensor& latent_out, cudaStream_t stream) {
    const int rank   = state_in.ne[0];
    const int batch  = state_in.ne[1];
    const int hidden = f_rnn.ne[0];
    if ((rank % 8 == 0) && (hidden % 8 == 0)) {
        mtp_markov_rnn_step_u4_kernel<<<dim3(rank, batch), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(state_in.data),
            static_cast<const std::int32_t*>(prev_tokens.data),
            static_cast<const __nv_bfloat16*>(markov_w1.data),
            static_cast<const __nv_bfloat16*>(hidden_block.data), block_pos, block_size,
            static_cast<const __nv_bfloat16*>(f_rnn.data),
            static_cast<const __nv_bfloat16*>(proj_w.data),
            static_cast<const __nv_bfloat16*>(proj_b.data),
            static_cast<__nv_bfloat16*>(state_out.data),
            static_cast<__nv_bfloat16*>(latent_out.data), rank, hidden);
    } else {
        mtp_markov_rnn_step_generic_kernel<<<dim3(rank, batch), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(state_in.data),
            static_cast<const std::int32_t*>(prev_tokens.data),
            static_cast<const __nv_bfloat16*>(markov_w1.data),
            static_cast<const __nv_bfloat16*>(hidden_block.data), block_pos, block_size,
            static_cast<const __nv_bfloat16*>(f_rnn.data),
            static_cast<const __nv_bfloat16*>(proj_w.data),
            static_cast<const __nv_bfloat16*>(proj_b.data),
            static_cast<__nv_bfloat16*>(state_out.data),
            static_cast<__nv_bfloat16*>(latent_out.data), rank, hidden);
    }
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

void inkling2_mtp_final_norm_extract_zero(Tensor& hidden_block, const Tensor& final_norm_w,
                                          float eps, std::int32_t batch_size,
                                          std::int32_t block_size, Tensor& draft_out,
                                          Tensor& rnn_state_to_zero, cudaStream_t stream) {
    const int hidden = hidden_block.ne[0];
    const int blk_T  = batch_size * block_size;
    if (hidden == 2304) {
        mtp_final_norm_extract_zero_h2304_kernel<<<blk_T, 288, 0, stream>>>(
            static_cast<__nv_bfloat16*>(hidden_block.data),
            static_cast<const __nv_bfloat16*>(final_norm_w.data), eps, batch_size, block_size,
            static_cast<__nv_bfloat16*>(draft_out.data),
            static_cast<__nv_bfloat16*>(rnn_state_to_zero.data), rnn_state_to_zero.ne[0]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    inkling2_rmsnorm(hidden_block, &final_norm_w, hidden_block, eps, stream);
    inkling2_mtp_extract_draft_hiddens(hidden_block, batch_size, block_size, draft_out, stream);
    CUDA_CHECK(cudaMemsetAsync(rnn_state_to_zero.data, 0, rnn_state_to_zero.bytes(), stream));
}

} // namespace ninfer::ops
