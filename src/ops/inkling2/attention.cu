#include "ninfer/ops/inkling2.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 128;
constexpr int kDRel        = 16;
constexpr int kRelExt      = 1024;
constexpr int kPageSize    = 64;
constexpr float kAttnScale = 0.08838834764831844f; // 1 / sqrt(128)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
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

// ============================================================================
// 1. SINGLE-LAUNCH Fused Attention Prologue (3 kernels -> 1 kernel, saves 96 launches/step!)
//    grid = (num_q_heads + num_kv_heads, tokens), block = 128
//    - blockIdx.x < num_q_heads: Q scaled L2 clip (+tau) -> q_out AND R scaled L2 clip (+tau)
//      -> rel_bias [0 .. min(1024, pos+1)-1] (skips all unused distances > pos!).
//    - blockIdx.x >= num_q_heads: K unscaled L2 clip + V unweighted RMSNorm -> direct KV write!
// ============================================================================

__global__ void attn_prologue_fused_kernel(
    const __nv_bfloat16* __restrict__ qgrs, const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in, const __nv_bfloat16* __restrict__ q_scale,
    const __nv_bfloat16* __restrict__ rel_scale, const __nv_bfloat16* __restrict__ rel_proj,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ lane_ids,
    Inkling2KvTarget kv, int packed_stride, int r_offset, int num_q_heads, int num_kv_heads,
    bool is_local, float qk_thresh, float rel_thresh, float log_floor, float log_alpha, float eps,
    __nv_bfloat16* __restrict__ q_out, float* __restrict__ rel_bias_out) {
    const int blk  = blockIdx.x;
    const int t    = blockIdx.y;
    const int d    = threadIdx.x; // 0..127
    const int warp = d >> 5;
    const int lane = d & 31;
    const int pos  = positions[t];

    if (blk < num_q_heads) {
        __shared__ float s_q[4];
        __shared__ float s_r[kDRel];

        // 1a. Q scaled L2 clip (+ tau if global)
        const __nv_bfloat16* q_src =
            qgrs + static_cast<std::int64_t>(t) * packed_stride + blk * kHeadDim;
        const float qval = __bfloat162float(q_src[d]) *
                           __bfloat162float(__ldg(q_scale + blk * kHeadDim + d));
        float sq = warp_sum(qval * qval);
        if (lane == 0) { s_q[warp] = sq; }

        const float tau =
            is_local ? 1.0f : (1.0f + log_alpha * logf(fmaxf(1.0f, static_cast<float>(pos + 1) / log_floor)));

        // 1b. R scaled L2 clip (+ tau if global) in parallel on first 16 threads!
        if (d < kDRel) {
            const __nv_bfloat16* r_src =
                qgrs + static_cast<std::int64_t>(t) * packed_stride + r_offset + blk * kDRel;
            float rval = __bfloat162float(r_src[d]) *
                         __bfloat162float(__ldg(rel_scale + blk * kDRel + d));
            float rsq = rval * rval;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                rsq += __shfl_xor_sync(0x0000FFFFu, rsq, off);
            }
            float ry = rval * fminf(1.0f, rel_thresh / (sqrtf(rsq) + eps));
            if (!is_local) {
                ry = __bfloat162float(__float2bfloat16_rn(ry)) * tau;
            }
            s_r[d] = __bfloat162float(__float2bfloat16_rn(ry));
        }
        __syncthreads();

        const float total_qsq = s_q[0] + s_q[1] + s_q[2] + s_q[3];
        float qy              = qval * fminf(1.0f, qk_thresh / (sqrtf(total_qsq) + eps));
        if (!is_local) {
            qy = __bfloat162float(__float2bfloat16_rn(qy)) * tau;
        }
        q_out[(static_cast<std::int64_t>(t) * num_q_heads + blk) * kHeadDim + d] =
            __float2bfloat16_rn(qy);

        // 1c. Compute R @ rel_logits_proj ONLY for causally reachable distances [0 .. min(1024, pos+1))!
        const int max_dist = min(kRelExt, pos + 1);
        float r_reg[kDRel];
#pragma unroll
        for (int i = 0; i < kDRel; ++i) { r_reg[i] = s_r[i]; }

        float* out_row =
            rel_bias_out + (static_cast<std::int64_t>(t) * num_q_heads + blk) * kRelExt;

        for (int dist = d; dist < max_dist; dist += 128) {
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < kDRel; ++i) {
                acc = fmaf(r_reg[i], __bfloat162float(__ldg(rel_proj + i * kRelExt + dist)), acc);
            }
            out_row[dist] = acc;
        }
    } else {
        // K clip + V unweighted RMSNorm -> direct KV cache write!
        const int h            = blk - num_q_heads;
        const std::int64_t off = (static_cast<std::int64_t>(t) * num_kv_heads + h) * kHeadDim + d;
        const float kv_val     = __bfloat162float(k_in[off]);
        const float vv_val     = __bfloat162float(v_in[off]);

        __shared__ float sk[4];
        __shared__ float sv[4];
        const float sq_k = warp_sum(kv_val * kv_val);
        const float sq_v = warp_sum(vv_val * vv_val);
        if (lane == 0) {
            sk[warp] = sq_k;
            sv[warp] = sq_v;
        }
        __syncthreads();

        const float k_factor = fminf(1.0f, qk_thresh / (sqrtf(sk[0] + sk[1] + sk[2] + sk[3]) + eps));
        const float v_inv    = rsqrtf((sv[0] + sv[1] + sv[2] + sv[3]) * (1.0f / 128.0f) + eps);

        const int lane_id = (lane_ids != nullptr) ? lane_ids[t] : 0;
        __nv_bfloat16* kd = kv_slot_ptr(kv, false, lane_id, h, num_kv_heads, pos);
        __nv_bfloat16* vd = kv_slot_ptr(kv, true, lane_id, h, num_kv_heads, pos);
        kd[d]             = __float2bfloat16_rn(kv_val * k_factor);
        vd[d]             = __float2bfloat16_rn(vv_val * v_inv);
    }
}

// ============================================================================
// 2. Single-Pass Fused Decode / Verify GQA-3 Attention (NO split_scratch, NO reduce launch!)
//    Used whenever num_splits == 1 (e.g. T >= 4 on SWA or T >= 16 global) -> saves 48 launches!
// ============================================================================

template <int kGqaGroup, int kWarps>
__global__ void attn_decode_verify_single_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ qgrs,
    const float* __restrict__ rel_bias, Inkling2KvTarget kv,
    const std::int32_t* __restrict__ lane_ids, const std::int32_t* __restrict__ q_positions,
    int batch_size, int q_per_lane, int packed_stride, int gate_offset, int sink_offset,
    int num_q_heads, int num_kv_heads, int swa_window, __nv_bfloat16* __restrict__ out) {
    const int kv_h     = blockIdx.x;
    const int tok      = blockIdx.y;
    const int b        = tok / q_per_lane;
    const int lane     = lane_ids[b];
    const int q_pos    = q_positions[tok];
    const int kv_start = kv.is_local ? max(0, q_pos - swa_window + 1) : 0;
    const int kv_end   = q_pos + 1;

    const int warp        = threadIdx.x >> 5;
    const int wlane       = threadIdx.x & 31;
    const int d_base      = wlane * 4;
    const int q_head_base = kv_h * kGqaGroup;

    float q_reg[kGqaGroup][4];
#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        const __nv_bfloat16* qp =
            q + (static_cast<std::int64_t>(tok) * num_q_heads + q_head_base + g) * kHeadDim + d_base;
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(qp));
        const auto* qb  = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 f0 = __bfloat1622float2(qb[0]);
        const float2 f1 = __bfloat1622float2(qb[1]);
        q_reg[g][0]     = f0.x * kAttnScale;
        q_reg[g][1]     = f0.y * kAttnScale;
        q_reg[g][2]     = f1.x * kAttnScale;
        q_reg[g][3]     = f1.y * kAttnScale;
    }

    float m[kGqaGroup];
    float l[kGqaGroup];
    float acc[kGqaGroup][4];
#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        m[g] = -std::numeric_limits<float>::infinity();
        l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { acc[g][j] = 0.0f; }
    }

    for (int kpos = kv_start + warp; kpos < kv_end; kpos += kWarps) {
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

        const int dist = q_pos - kpos;
#pragma unroll
        for (int g = 0; g < kGqaGroup; ++g) {
            float dot = q_reg[g][0] * kf0.x + q_reg[g][1] * kf0.y +
                        q_reg[g][2] * kf1.x + q_reg[g][3] * kf1.y;
            dot = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);
            if (dist < kRelExt) {
                const float* rb =
                    rel_bias +
                    (static_cast<std::int64_t>(tok) * num_q_heads + q_head_base + g) * kRelExt;
                dot += __ldg(rb + dist);
            }
            const float new_m = fmaxf(m[g], dot);
            const float alpha = (m[g] == -std::numeric_limits<float>::infinity())
                                    ? 0.0f
                                    : expf(m[g] - new_m);
            const float p     = expf(dot - new_m);
            l[g]              = l[g] * alpha + p;
            acc[g][0]         = fmaf(p, vf0.x, acc[g][0] * alpha);
            acc[g][1]         = fmaf(p, vf0.y, acc[g][1] * alpha);
            acc[g][2]         = fmaf(p, vf1.x, acc[g][2] * alpha);
            acc[g][3]         = fmaf(p, vf1.y, acc[g][3] * alpha);
            m[g]              = new_m;
        }
    }

    __shared__ float s_m[kGqaGroup][kWarps];
    __shared__ float s_l[kGqaGroup][kWarps];
    __shared__ float s_o[kGqaGroup][kWarps][kHeadDim];

#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        if (wlane == 0) {
            s_m[g][warp] = m[g];
            s_l[g][warp] = l[g];
        }
        s_o[g][warp][d_base + 0] = acc[g][0];
        s_o[g][warp][d_base + 1] = acc[g][1];
        s_o[g][warp][d_base + 2] = acc[g][2];
        s_o[g][warp][d_base + 3] = acc[g][3];
    }
    __syncthreads();

    const int d = threadIdx.x; // 0..127
#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        const int qh   = q_head_base + g;
        const float sink =
            __bfloat162float(__ldg(qgrs + static_cast<std::int64_t>(tok) * packed_stride + sink_offset + qh));
        float gm = sink;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { gm = fmaxf(gm, s_m[g][w]); }
        float gl = expf(sink - gm);
        float go = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float wm = s_m[g][w];
            if (wm > -std::numeric_limits<float>::infinity()) {
                const float sc = expf(wm - gm);
                gl += s_l[g][w] * sc;
                go += s_o[g][w][d] * sc;
            }
        }
        const float gate_raw = __bfloat162float(__ldg(
            qgrs + static_cast<std::int64_t>(tok) * packed_stride + gate_offset + qh * kHeadDim + d));
        const float gate_sig = 1.0f / (1.0f + expf(-gate_raw));
        out[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + d] =
            __float2bfloat16_rn((go / gl) * gate_sig);
    }
}

// ============================================================================
// 3. Split-K Decode / Verify GQA-3 FlashDecoding Kernel (for long global context / low T)
// ============================================================================

template <int kGqaGroup, int kWarps>
__global__ void attn_decode_verify_splitk_kernel(
    const __nv_bfloat16* __restrict__ q, const float* __restrict__ rel_bias,
    Inkling2KvTarget kv, const std::int32_t* __restrict__ lane_ids,
    const std::int32_t* __restrict__ q_positions, int batch_size, int q_per_lane,
    int num_q_heads, int num_kv_heads, int swa_window, int num_splits,
    float* __restrict__ split_scratch) {
    const int kv_h     = blockIdx.x;
    const int tok      = blockIdx.y;
    const int split_id = blockIdx.z;

    const int b        = tok / q_per_lane;
    const int lane     = lane_ids[b];
    const int q_pos    = q_positions[tok];
    const int kv_start = kv.is_local ? max(0, q_pos - swa_window + 1) : 0;
    const int kv_end   = q_pos + 1;
    const int total_k  = kv_end - kv_start;

    const int chunk = (total_k + num_splits - 1) / num_splits;
    const int k_beg = kv_start + split_id * chunk;
    const int k_end = min(kv_end, k_beg + chunk);

    const int warp        = threadIdx.x >> 5;
    const int wlane       = threadIdx.x & 31;
    const int d_base      = wlane * 4;
    const int q_head_base = kv_h * kGqaGroup;

    float q_reg[kGqaGroup][4];
#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        const __nv_bfloat16* qp =
            q + (static_cast<std::int64_t>(tok) * num_q_heads + q_head_base + g) * kHeadDim + d_base;
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(qp));
        const auto* qb  = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 f0 = __bfloat1622float2(qb[0]);
        const float2 f1 = __bfloat1622float2(qb[1]);
        q_reg[g][0]     = f0.x * kAttnScale;
        q_reg[g][1]     = f0.y * kAttnScale;
        q_reg[g][2]     = f1.x * kAttnScale;
        q_reg[g][3]     = f1.y * kAttnScale;
    }

    float m[kGqaGroup];
    float l[kGqaGroup];
    float acc[kGqaGroup][4];
#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        m[g] = -std::numeric_limits<float>::infinity();
        l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { acc[g][j] = 0.0f; }
    }

    for (int kpos = k_beg + warp; kpos < k_end; kpos += kWarps) {
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

        const int dist = q_pos - kpos;
#pragma unroll
        for (int g = 0; g < kGqaGroup; ++g) {
            float dot = q_reg[g][0] * kf0.x + q_reg[g][1] * kf0.y +
                        q_reg[g][2] * kf1.x + q_reg[g][3] * kf1.y;
            dot = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);
            if (dist < kRelExt) {
                const float* rb =
                    rel_bias +
                    (static_cast<std::int64_t>(tok) * num_q_heads + q_head_base + g) * kRelExt;
                dot += __ldg(rb + dist);
            }
            const float new_m = fmaxf(m[g], dot);
            const float alpha = (m[g] == -std::numeric_limits<float>::infinity())
                                    ? 0.0f
                                    : expf(m[g] - new_m);
            const float p     = expf(dot - new_m);
            l[g]              = l[g] * alpha + p;
            acc[g][0]         = fmaf(p, vf0.x, acc[g][0] * alpha);
            acc[g][1]         = fmaf(p, vf0.y, acc[g][1] * alpha);
            acc[g][2]         = fmaf(p, vf1.x, acc[g][2] * alpha);
            acc[g][3]         = fmaf(p, vf1.y, acc[g][3] * alpha);
            m[g]              = new_m;
        }
    }

    __shared__ float s_m[kGqaGroup][kWarps];
    __shared__ float s_l[kGqaGroup][kWarps];
    __shared__ float s_o[kGqaGroup][kWarps][kHeadDim];

#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        if (wlane == 0) {
            s_m[g][warp] = m[g];
            s_l[g][warp] = l[g];
        }
        s_o[g][warp][d_base + 0] = acc[g][0];
        s_o[g][warp][d_base + 1] = acc[g][1];
        s_o[g][warp][d_base + 2] = acc[g][2];
        s_o[g][warp][d_base + 3] = acc[g][3];
    }
    __syncthreads();

    const int d = threadIdx.x;
#pragma unroll
    for (int g = 0; g < kGqaGroup; ++g) {
        const int qh = q_head_base + g;
        float gm     = -std::numeric_limits<float>::infinity();
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { gm = fmaxf(gm, s_m[g][w]); }
        float gl = 0.0f;
        float go = 0.0f;
        if (gm > -std::numeric_limits<float>::infinity()) {
#pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                const float wm = s_m[g][w];
                if (wm > -std::numeric_limits<float>::infinity()) {
                    const float sc = expf(wm - gm);
                    gl += s_l[g][w] * sc;
                    go += s_o[g][w][d] * sc;
                }
            }
        }
        float* dst = split_scratch +
                     ((static_cast<std::int64_t>(tok) * num_q_heads + qh) * num_splits + split_id) *
                         (kHeadDim + 2);
        dst[d] = go;
        if (d == 0) {
            dst[kHeadDim]     = gm;
            dst[kHeadDim + 1] = gl;
        }
    }
}

__global__ void attn_decode_verify_reduce_kernel(const float* __restrict__ split_scratch,
                                                 const __nv_bfloat16* __restrict__ qgrs,
                                                 __nv_bfloat16* __restrict__ out,
                                                 int packed_stride, int gate_offset,
                                                 int sink_offset, int num_q_heads,
                                                 int num_splits) {
    const int qh = blockIdx.x;
    const int t  = blockIdx.y;
    const int d  = threadIdx.x;

    const float* base =
        split_scratch + (static_cast<std::int64_t>(t) * num_q_heads + qh) * num_splits * (kHeadDim + 2);
    const float sink =
        __bfloat162float(__ldg(qgrs + static_cast<std::int64_t>(t) * packed_stride + sink_offset + qh));

    float gm = sink;
    for (int s = 0; s < num_splits; ++s) {
        gm = fmaxf(gm, base[s * (kHeadDim + 2) + kHeadDim]);
    }
    float gl = expf(sink - gm);
    float go = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const float* sp = base + s * (kHeadDim + 2);
        const float sm  = sp[kHeadDim];
        if (sm > -std::numeric_limits<float>::infinity()) {
            const float sc = expf(sm - gm);
            gl += sp[kHeadDim + 1] * sc;
            go += sp[d] * sc;
        }
    }

    const float gate_raw = __bfloat162float(
        __ldg(qgrs + static_cast<std::int64_t>(t) * packed_stride + gate_offset + qh * kHeadDim + d));
    const float gate_sig = 1.0f / (1.0f + expf(-gate_raw));
    out[(static_cast<std::int64_t>(t) * num_q_heads + qh) * kHeadDim + d] =
        __float2bfloat16_rn((go / gl) * gate_sig);
}

// ============================================================================
// 4. Prefill Tiled Causal SWA/Full Relative-Bias Sink Attention
// ============================================================================

__global__ void attn_prefill_kernel(const __nv_bfloat16* __restrict__ q,
                                    const __nv_bfloat16* __restrict__ qgrs,
                                    const float* __restrict__ rel_bias, Inkling2KvTarget kv,
                                    int lane, int start_pos, int tokens, int packed_stride,
                                    int gate_offset, int sink_offset, int num_q_heads,
                                    int num_kv_heads, int swa_window,
                                    __nv_bfloat16* __restrict__ out) {
    constexpr int kWarps = 4;
    constexpr int kBc    = 32;
    __shared__ __nv_bfloat16 smem_k[kBc][kHeadDim];
    __shared__ __nv_bfloat16 smem_v[kBc][kHeadDim];

    const int qh       = blockIdx.x;
    const int q_block  = blockIdx.y * kWarps;
    const int warp     = threadIdx.x >> 5;
    const int wlane    = threadIdx.x & 31;
    const int d_base   = wlane * 4;
    const int gqa_rep  = num_q_heads / num_kv_heads;
    const int kv_h     = qh / gqa_rep;
    const int q_local  = q_block + warp;
    const bool valid_q = q_local < tokens;
    const int q_pos    = start_pos + q_local;

    const int block_q_min = start_pos + q_block;
    const int block_q_max = start_pos + min(tokens - 1, q_block + kWarps - 1);
    const int k_min       = kv.is_local ? max(0, block_q_min - swa_window + 1) : 0;
    const int k_max       = block_q_max + 1;

    float q_reg[4] = {};
    float m        = -std::numeric_limits<float>::infinity();
    float l        = 0.0f;
    float acc[4]   = {};

    if (valid_q) {
        const __nv_bfloat16* qp =
            q + (static_cast<std::int64_t>(q_local) * num_q_heads + qh) * kHeadDim + d_base;
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(qp));
        const auto* qb  = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 f0 = __bfloat1622float2(qb[0]);
        const float2 f1 = __bfloat1622float2(qb[1]);
        q_reg[0]        = f0.x * kAttnScale;
        q_reg[1]        = f0.y * kAttnScale;
        q_reg[2]        = f1.x * kAttnScale;
        q_reg[3]        = f1.y * kAttnScale;

        m = __bfloat162float(__ldg(qgrs + static_cast<std::int64_t>(q_local) * packed_stride +
                                   sink_offset + qh));
        l = 1.0f;
    }

    const float* rb_row =
        valid_q ? (rel_bias + (static_cast<std::int64_t>(q_local) * num_q_heads + qh) * kRelExt)
                : nullptr;

    for (int k_tile = k_min; k_tile < k_max; k_tile += kBc) {
        const int tile_len = min(kBc, k_max - k_tile);
        for (int r = warp; r < tile_len; r += kWarps) {
            const int kpos          = k_tile + r;
            const __nv_bfloat16* kp = kv_slot_ptr(kv, false, lane, kv_h, num_kv_heads, kpos) + d_base;
            const __nv_bfloat16* vp = kv_slot_ptr(kv, true, lane, kv_h, num_kv_heads, kpos) + d_base;
            *reinterpret_cast<uint2*>(&smem_k[r][d_base]) = __ldg(reinterpret_cast<const uint2*>(kp));
            *reinterpret_cast<uint2*>(&smem_v[r][d_base]) = __ldg(reinterpret_cast<const uint2*>(vp));
        }
        __syncthreads();

        if (valid_q) {
            for (int r = 0; r < tile_len; ++r) {
                const int kpos = k_tile + r;
                const int dist = q_pos - kpos;
                if (dist < 0 || (kv.is_local && dist >= swa_window)) { continue; }

                const uint2 kraw = *reinterpret_cast<const uint2*>(&smem_k[r][d_base]);
                const uint2 vraw = *reinterpret_cast<const uint2*>(&smem_v[r][d_base]);
                const auto* kb   = reinterpret_cast<const __nv_bfloat162*>(&kraw);
                const auto* vb   = reinterpret_cast<const __nv_bfloat162*>(&vraw);
                const float2 kf0 = __bfloat1622float2(kb[0]);
                const float2 kf1 = __bfloat1622float2(kb[1]);
                const float2 vf0 = __bfloat1622float2(vb[0]);
                const float2 vf1 = __bfloat1622float2(vb[1]);

                float dot = q_reg[0] * kf0.x + q_reg[1] * kf0.y +
                            q_reg[2] * kf1.x + q_reg[3] * kf1.y;
                dot = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0);
                if (dist < kRelExt) { dot += __ldg(rb_row + dist); }

                const float new_m = fmaxf(m, dot);
                const float alpha = expf(m - new_m);
                const float p     = expf(dot - new_m);
                l                 = l * alpha + p;
                acc[0]            = fmaf(p, vf0.x, acc[0] * alpha);
                acc[1]            = fmaf(p, vf0.y, acc[1] * alpha);
                acc[2]            = fmaf(p, vf1.x, acc[2] * alpha);
                acc[3]            = fmaf(p, vf1.y, acc[3] * alpha);
                m                 = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        const float inv_l = 1.0f / l;
        const __nv_bfloat16* gp =
            qgrs + static_cast<std::int64_t>(q_local) * packed_stride + gate_offset + qh * kHeadDim + d_base;
        const uint2 graw = __ldg(reinterpret_cast<const uint2*>(gp));
        const auto* gb   = reinterpret_cast<const __nv_bfloat162*>(&graw);
        const float2 gf0 = __bfloat1622float2(gb[0]);
        const float2 gf1 = __bfloat1622float2(gb[1]);

        __nv_bfloat16* op =
            out + (static_cast<std::int64_t>(q_local) * num_q_heads + qh) * kHeadDim + d_base;
        op[0] = __float2bfloat16_rn((acc[0] * inv_l) / (1.0f + expf(-gf0.x)));
        op[1] = __float2bfloat16_rn((acc[1] * inv_l) / (1.0f + expf(-gf0.y)));
        op[2] = __float2bfloat16_rn((acc[2] * inv_l) / (1.0f + expf(-gf1.x)));
        op[3] = __float2bfloat16_rn((acc[3] * inv_l) / (1.0f + expf(-gf1.y)));
    }
}

} // namespace

void inkling2_attn_prologue(const Tensor& q_gate_r_sink, const Tensor& k_in, const Tensor& v_in,
                            const Tensor& q_clip_scale, const Tensor& rel_clip_scale,
                            const Tensor& rel_logits_proj, const std::int32_t* positions,
                            const std::int32_t* lane_ids, std::int32_t tokens,
                            std::int32_t num_q_heads, std::int32_t num_kv_heads,
                            const Inkling2KvTarget& kv, bool is_local, float qk_clip_threshold,
                            float rel_clip_threshold, float log_floor, float log_alpha, float eps,
                            Tensor& q_out, Tensor& rel_bias_out, cudaStream_t stream) {
    const int packed_stride = q_gate_r_sink.ne[0];
    const int q_dim         = num_q_heads * kHeadDim;
    const int r_offset      = 2 * q_dim;

    attn_prologue_fused_kernel<<<dim3(num_q_heads + num_kv_heads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_gate_r_sink.data),
        static_cast<const __nv_bfloat16*>(k_in.data),
        static_cast<const __nv_bfloat16*>(v_in.data),
        static_cast<const __nv_bfloat16*>(q_clip_scale.data),
        static_cast<const __nv_bfloat16*>(rel_clip_scale.data),
        static_cast<const __nv_bfloat16*>(rel_logits_proj.data), positions, lane_ids, kv,
        packed_stride, r_offset, num_q_heads, num_kv_heads, is_local, qk_clip_threshold,
        rel_clip_threshold, log_floor, log_alpha, eps, static_cast<__nv_bfloat16*>(q_out.data),
        static_cast<float*>(rel_bias_out.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attn_decode_verify(const Tensor& q, const Tensor& q_gate_r_sink,
                                 const Tensor& rel_bias_f32, const Inkling2KvTarget& kv,
                                 const std::int32_t* lane_ids, const std::int32_t* q_positions,
                                 std::int32_t batch_size, std::int32_t q_per_lane,
                                 std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                 std::int32_t swa_window, float* split_scratch,
                                 std::int32_t max_splits, Tensor& out, cudaStream_t stream) {
    const int total_tokens  = batch_size * q_per_lane;
    const int gqa_group     = num_q_heads / num_kv_heads;
    const int packed_stride = q_gate_r_sink.ne[0];
    const int q_dim         = num_q_heads * kHeadDim;
    const int gate_offset   = q_dim;
    const int sink_offset   = 2 * q_dim + num_q_heads * kDRel;
    if (gqa_group != 3) {
        throw std::invalid_argument("inkling2_attn_decode_verify: expected GQA group size 3");
    }

    // Only skip Split-KV when base_ctas already saturates half the GPU (>= 40 CTAs on 80 SMs)!
    // Low base_ctas (e.g. B=1 V=1..8 -> 4..32 CTAs, B=4 V=1 -> 16 CTAs) MUST Split-KV across SMs!
    const int base_ctas        = num_kv_heads * total_tokens;
    const bool use_single_pass = (max_splits <= 1) || (base_ctas >= 40);
    if (use_single_pass) {
        attn_decode_verify_single_kernel<3, 4>
            <<<dim3(num_kv_heads, total_tokens), 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(q_gate_r_sink.data),
                static_cast<const float*>(rel_bias_f32.data), kv, lane_ids, q_positions, batch_size,
                q_per_lane, packed_stride, gate_offset, sink_offset, num_q_heads, num_kv_heads,
                swa_window, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_splits = std::max(2, std::min(max_splits, (160 + base_ctas - 1) / base_ctas));
    if (kv.is_local) { num_splits = std::min(num_splits, 8); }

    attn_decode_verify_splitk_kernel<3, 4>
        <<<dim3(num_kv_heads, total_tokens, num_splits), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const float*>(rel_bias_f32.data), kv, lane_ids, q_positions, batch_size,
            q_per_lane, num_q_heads, num_kv_heads, swa_window, num_splits, split_scratch);
    CUDA_CHECK(cudaGetLastError());

    attn_decode_verify_reduce_kernel<<<dim3(num_q_heads, total_tokens), kHeadDim, 0, stream>>>(
        split_scratch, static_cast<const __nv_bfloat16*>(q_gate_r_sink.data),
        static_cast<__nv_bfloat16*>(out.data), packed_stride, gate_offset, sink_offset, num_q_heads,
        num_splits);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attn_prefill(const Tensor& q, const Tensor& q_gate_r_sink,
                           const Tensor& rel_bias_f32, const Inkling2KvTarget& kv,
                           std::int32_t lane, std::int32_t start_pos, std::int32_t tokens,
                           std::int32_t num_q_heads, std::int32_t num_kv_heads,
                           std::int32_t swa_window, Tensor& out, cudaStream_t stream) {
    const int packed_stride = q_gate_r_sink.ne[0];
    const int q_dim         = num_q_heads * kHeadDim;
    const int gate_offset   = q_dim;
    const int sink_offset   = 2 * q_dim + num_q_heads * kDRel;
    const dim3 grid(num_q_heads, (tokens + 3) / 4);
    attn_prefill_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(q_gate_r_sink.data),
        static_cast<const float*>(rel_bias_f32.data), kv, lane, start_pos, tokens, packed_stride,
        gate_offset, sink_offset, num_q_heads, num_kv_heads, swa_window,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
