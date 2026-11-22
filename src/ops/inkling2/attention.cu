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
constexpr float kInvSqrt5  = 0.44721359549995793928f;
constexpr float kInvSqrt9  = 0.33333333333333333333f;

__device__ __forceinline__ float ld_cg_f32_vol(const float* ptr) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(ptr));
    return v;
}

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
        const int slot = (kv.ring_cap == 1024) ? (pos & 1023) : (pos % kv.ring_cap);
        auto* base     = static_cast<__nv_bfloat16*>(is_v ? kv.ring_v : kv.ring_k);
        const std::int64_t idx =
            ((static_cast<std::int64_t>(lane) * kv.ring_cap + slot) * num_kv_heads + kv_head) *
            kHeadDim;
        return base + idx;
    }
    const int logical_page = pos >> 6;
    const int page_off     = pos & 63;
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
// 1. SINGLE-LAUNCH Fused Attention Prologue (+ Optional K/V Static/Dynamic SConv!)
//    grid = (num_q_heads + num_kv_heads, tokens), block = 128
//    - blockIdx.x < num_q_heads: Q scaled L2 clip (+tau) -> q_out AND R scaled L2 clip (+tau)
//      -> vectorized float4 rel_bias [0 .. min(1024, pos+1)) (4x fewer loads/stores!).
//    - blockIdx.x >= num_q_heads: optional K/V static/dynamic SConv (Decode or Verify in registers!)
//      + K unscaled L2 clip + V unweighted RMSNorm -> direct KV write!
// ============================================================================

template <int SconvMode> // 0=off, 1=static_decode, 2=static_verify, 3=dynamic_decode, 4=dynamic_verify
__global__ void attn_prologue_fused_kernel(
    const __nv_bfloat16* __restrict__ qgrs, const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in, const __nv_bfloat16* __restrict__ q_scale,
    const __nv_bfloat16* __restrict__ rel_scale, const __nv_bfloat16* __restrict__ rel_proj,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ lane_ids,
    Inkling2KvTarget kv, int packed_stride, int r_offset, int num_q_heads, int num_kv_heads,
    bool is_local, float qk_thresh, float rel_thresh, float log_floor, float log_alpha, float eps,
    __nv_bfloat16* __restrict__ q_out, float* __restrict__ rel_bias_out,
    const __nv_bfloat16* __restrict__ kw, const __nv_bfloat16* __restrict__ vw,
    const __nv_bfloat16* __restrict__ k_dyn, const __nv_bfloat16* __restrict__ v_dyn,
    float* __restrict__ k_state, float* __restrict__ v_state,
    float* __restrict__ k_verify, float* __restrict__ v_verify,
    std::size_t state_lane_stride, int q_per_lane, int max_verify_len) {
    const int blk  = blockIdx.x;
    const int t    = blockIdx.y;
    const int d    = threadIdx.x; // 0..127
    const int warp = d >> 5;
    const int lane = d & 31;

    if (blk < num_q_heads) {
        const int pos = positions[t];
        __shared__ float s_q[4];
        __shared__ float s_r[kDRel];

        const __nv_bfloat16* q_src =
            qgrs + static_cast<std::int64_t>(t) * packed_stride + blk * kHeadDim;
        const float qval = __bfloat162float(q_src[d]) *
                           __bfloat162float(__ldg(q_scale + blk * kHeadDim + d));
        float sq = warp_sum(qval * qval);
        if (lane == 0) { s_q[warp] = sq; }

        const float tau =
            is_local ? 1.0f : (1.0f + log_alpha * logf(fmaxf(1.0f, static_cast<float>(pos + 1) / log_floor)));

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

        // Vectorized 4-distance/thread R @ rel_proj (128 threads x 4 = 512 distances/iter!)
        const int max_dist4 = min(kRelExt >> 2, (pos + 4) >> 2);
        float r_reg[kDRel];
#pragma unroll
        for (int i = 0; i < kDRel; ++i) { r_reg[i] = s_r[i]; }

        float4* out_row4 = reinterpret_cast<float4*>(
            rel_bias_out + (static_cast<std::int64_t>(t) * num_q_heads + blk) * kRelExt);

        for (int d4 = d; d4 < max_dist4; d4 += 128) {
            const int dist = d4 << 2;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int i = 0; i < kDRel; ++i) {
                const uint2 praw = __ldg(reinterpret_cast<const uint2*>(rel_proj + i * kRelExt + dist));
                const auto* pb   = reinterpret_cast<const __nv_bfloat162*>(&praw);
                const float2 f0  = __bfloat1622float2(pb[0]);
                const float2 f1  = __bfloat1622float2(pb[1]);
                const float rv   = r_reg[i];
                a0               = fmaf(rv, f0.x, a0);
                a1               = fmaf(rv, f0.y, a1);
                a2               = fmaf(rv, f1.x, a2);
                a3               = fmaf(rv, f1.y, a3);
            }
            out_row4[d4] = make_float4(a0, a1, a2, a3);
        }
        return;
    }

    // blk >= num_q_heads: K/V SConv (optional) + K L2-clip + V RMSNorm -> direct KV write!
    const int h        = blk - num_q_heads;
    const int channels = num_kv_heads * kHeadDim;
    const int c        = h * kHeadDim + d;
    __shared__ float sk[4];
    __shared__ float sv[4];

    auto clip_norm_write = [&](float kv_val, float vv_val, int lane_id, int pos) {
        const float sq_k = warp_sum(kv_val * kv_val);
        const float sq_v = warp_sum(vv_val * vv_val);
        if (lane == 0) {
            sk[warp] = sq_k;
            sv[warp] = sq_v;
        }
        __syncthreads();
        const float k_factor = fminf(1.0f, qk_thresh / (sqrtf(sk[0] + sk[1] + sk[2] + sk[3]) + eps));
        const float v_inv    = rsqrtf((sv[0] + sv[1] + sv[2] + sv[3]) * (1.0f / 128.0f) + eps);
        __nv_bfloat16* kd    = kv_slot_ptr(kv, false, lane_id, h, num_kv_heads, pos);
        __nv_bfloat16* vd    = kv_slot_ptr(kv, true, lane_id, h, num_kv_heads, pos);
        kd[d]                = __float2bfloat16_rn(kv_val * k_factor);
        vd[d]                = __float2bfloat16_rn(vv_val * v_inv);
    };

    if constexpr (SconvMode == 0) {
        const int pos          = positions[t];
        const int lane_id      = (lane_ids != nullptr) ? lane_ids[t] : 0;
        const std::int64_t off = static_cast<std::int64_t>(t) * channels + c;
        clip_norm_write(__bfloat162float(k_in[off]), __bfloat162float(v_in[off]), lane_id, pos);
    } else if constexpr (SconvMode == 1 || SconvMode == 3) {
        // Static or Dynamic Decode (q_per_lane == 1, in-place state update)
        const int pos     = positions[t];
        const int lane_id = (lane_ids != nullptr) ? lane_ids[t] : 0;
        float* stk        = k_state + static_cast<std::size_t>(lane_id) * state_lane_stride;
        float* stv        = v_state + static_cast<std::size_t>(lane_id) * state_lane_stride;

        const uint2 kwr = __ldg(reinterpret_cast<const uint2*>(kw + c * 4));
        const uint2 vwr = __ldg(reinterpret_cast<const uint2*>(vw + c * 4));
        const auto* kwb = reinterpret_cast<const __nv_bfloat162*>(&kwr);
        const auto* vwb = reinterpret_cast<const __nv_bfloat162*>(&vwr);
        float kw0 = __low2float(kwb[0]), kw1 = __high2float(kwb[0]),
              kw2 = __low2float(kwb[1]), kw3 = __high2float(kwb[1]);
        float vw0 = __low2float(vwb[0]), vw1 = __high2float(vwb[0]),
              vw2 = __low2float(vwb[1]), vw3 = __high2float(vwb[1]);

        const float ks0 = stk[0 * channels + c], ks1 = stk[1 * channels + c], ks2 = stk[2 * channels + c];
        const float vs0 = stv[0 * channels + c], vs1 = stv[1 * channels + c], vs2 = stv[2 * channels + c];
        const std::int64_t off = static_cast<std::int64_t>(t) * channels + c;
        const float xk         = __bfloat162float(k_in[off]);
        const float xv         = __bfloat162float(v_in[off]);

        float yk, yv;
        if constexpr (SconvMode == 1) {
            yk = (kw0 * ks0 + kw1 * ks1 + kw2 * ks2 + kw3 * xk + xk) * kInvSqrt5;
            yv = (vw0 * vs0 + vw1 * vs1 + vw2 * vs2 + vw3 * xv + xv) * kInvSqrt5;
        } else {
            const __nv_bfloat16* dk = k_dyn + (static_cast<std::int64_t>(t) * num_kv_heads + h) * 4;
            const __nv_bfloat16* dv = v_dyn + (static_cast<std::int64_t>(t) * num_kv_heads + h) * 4;
            kw0 += __bfloat162float(__ldg(dk + 0));
            kw1 += __bfloat162float(__ldg(dk + 1));
            kw2 += __bfloat162float(__ldg(dk + 2));
            kw3 += __bfloat162float(__ldg(dk + 3));
            vw0 += __bfloat162float(__ldg(dv + 0));
            vw1 += __bfloat162float(__ldg(dv + 1));
            vw2 += __bfloat162float(__ldg(dv + 2));
            vw3 += __bfloat162float(__ldg(dv + 3));
            yk = (kw0 * xk + kw1 * ks2 + kw2 * ks1 + kw3 * ks0 + xk) * kInvSqrt9;
            yv = (vw0 * xv + vw1 * vs2 + vw2 * vs1 + vw3 * vs0 + xv) * kInvSqrt9;
        }
        stk[0 * channels + c] = ks1;
        stk[1 * channels + c] = ks2;
        stk[2 * channels + c] = xk;
        stv[0 * channels + c] = vs1;
        stv[1 * channels + c] = vs2;
        stv[2 * channels + c] = xv;

        // Exact BF16 boundary match with separate kv_sconv -> attn_prologue!
        const float k_bf16 = __bfloat162float(__float2bfloat16_rn(yk));
        const float v_bf16 = __bfloat162float(__float2bfloat16_rn(yv));
        clip_norm_write(k_bf16, v_bf16, lane_id, pos);
    } else {
        // Static or Dynamic Verify (SconvMode 2 or 4): sequential across v = 0..q_per_lane-1!
        if ((t % q_per_lane) != 0) { return; }
        const int lane_id = (lane_ids != nullptr) ? lane_ids[t] : 0;
        const float* stk  = k_state + static_cast<std::size_t>(lane_id) * state_lane_stride;
        const float* stv  = v_state + static_cast<std::size_t>(lane_id) * state_lane_stride;
        float* vstk_lane  = k_verify + static_cast<std::size_t>(lane_id) *
                                           static_cast<std::size_t>(max_verify_len) * state_lane_stride;
        float* vstv_lane  = v_verify + static_cast<std::size_t>(lane_id) *
                                           static_cast<std::size_t>(max_verify_len) * state_lane_stride;

        const uint2 kwr = __ldg(reinterpret_cast<const uint2*>(kw + c * 4));
        const uint2 vwr = __ldg(reinterpret_cast<const uint2*>(vw + c * 4));
        const auto* kwb = reinterpret_cast<const __nv_bfloat162*>(&kwr);
        const auto* vwb = reinterpret_cast<const __nv_bfloat162*>(&vwr);
        const float skw0 = __low2float(kwb[0]), skw1 = __high2float(kwb[0]),
                    skw2 = __low2float(kwb[1]), skw3 = __high2float(kwb[1]);
        const float svw0 = __low2float(vwb[0]), svw1 = __high2float(vwb[0]),
                    svw2 = __low2float(vwb[1]), svw3 = __high2float(vwb[1]);

        float ks0 = stk[0 * channels + c], ks1 = stk[1 * channels + c], ks2 = stk[2 * channels + c];
        float vs0 = stv[0 * channels + c], vs1 = stv[1 * channels + c], vs2 = stv[2 * channels + c];

        for (int v = 0; v < q_per_lane; ++v) {
            const int tok          = t + v;
            const int pos          = positions[tok];
            const std::int64_t off = static_cast<std::int64_t>(tok) * channels + c;
            const float xk         = __bfloat162float(k_in[off]);
            const float xv         = __bfloat162float(v_in[off]);
            float yk, yv;
            if constexpr (SconvMode == 2) {
                yk = (skw0 * ks0 + skw1 * ks1 + skw2 * ks2 + skw3 * xk + xk) * kInvSqrt5;
                yv = (svw0 * vs0 + svw1 * vs1 + svw2 * vs2 + svw3 * xv + xv) * kInvSqrt5;
            } else {
                const __nv_bfloat16* dk = k_dyn + (static_cast<std::int64_t>(tok) * num_kv_heads + h) * 4;
                const __nv_bfloat16* dv = v_dyn + (static_cast<std::int64_t>(tok) * num_kv_heads + h) * 4;
                const float kw0 = skw0 + __bfloat162float(__ldg(dk + 0));
                const float kw1 = skw1 + __bfloat162float(__ldg(dk + 1));
                const float kw2 = skw2 + __bfloat162float(__ldg(dk + 2));
                const float kw3 = skw3 + __bfloat162float(__ldg(dk + 3));
                const float vw0 = svw0 + __bfloat162float(__ldg(dv + 0));
                const float vw1 = svw1 + __bfloat162float(__ldg(dv + 1));
                const float vw2 = svw2 + __bfloat162float(__ldg(dv + 2));
                const float vw3 = svw3 + __bfloat162float(__ldg(dv + 3));
                yk = (kw0 * xk + kw1 * ks2 + kw2 * ks1 + kw3 * ks0 + xk) * kInvSqrt9;
                yv = (vw0 * xv + vw1 * vs2 + vw2 * vs1 + vw3 * vs0 + xv) * kInvSqrt9;
            }
            ks0 = ks1; ks1 = ks2; ks2 = xk;
            vs0 = vs1; vs1 = vs2; vs2 = xv;

            float* vstk            = vstk_lane + static_cast<std::size_t>(v) * state_lane_stride;
            float* vstv            = vstv_lane + static_cast<std::size_t>(v) * state_lane_stride;
            vstk[0 * channels + c] = ks0;
            vstk[1 * channels + c] = ks1;
            vstk[2 * channels + c] = ks2;
            vstv[0 * channels + c] = vs0;
            vstv[1 * channels + c] = vs1;
            vstv[2 * channels + c] = vs2;

            const float k_bf16 = __bfloat162float(__float2bfloat16_rn(yk));
            const float v_bf16 = __bfloat162float(__float2bfloat16_rn(yv));
            clip_norm_write(k_bf16, v_bf16, lane_id, pos);
            if (v + 1 < q_per_lane) { __syncthreads(); }
        }
    }
}

// ============================================================================
// 2. Adaptive Single-Launch Last-CTA Q-Head Split-KV Decode / Verify Attention
//    - 1 CTA = (q_head, tok, split_id), 128 threads (4 warps), ~30 regs, 2 KB SMEM (16 blks/SM!).
//    - Eliminates 3x GQA serial loop inside warps (V100 6 MB L2 absorbs 3-way KV reuse across
//      concurrent Q-head CTAs -> 2.5x-3.6x faster than GQA-3 serial loop at 322..4096 tok!).
//    - Incremental ring slot (ZERO % 1040 division in inner loop) + page-cached global KV.
//    - Short context (eff_splits == 1): skips split_scratch & atomics completely!
//    - Long context (eff_splits > 1): Warp-0 SMEM Last-CTA reduction (zero redundant expf/ld.cg,
//      ZERO reduce kernel launches across ALL 48 layers!).
// ============================================================================

template <int kWarps>
__global__ __launch_bounds__(kWarps * 32, 8) void attn_qhead_adaptive_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ qgrs,
    const float* __restrict__ rel_bias, Inkling2KvTarget kv,
    const std::int32_t* __restrict__ lane_ids, const std::int32_t* __restrict__ q_positions,
    int q_per_lane, int packed_stride, int gate_offset, int sink_offset, int num_q_heads,
    int num_kv_heads, int swa_window, int num_splits, int min_split_k,
    float* __restrict__ split_scratch, std::int32_t* __restrict__ split_done,
    __nv_bfloat16* __restrict__ out) {
    constexpr int kSpStride = kHeadDim + 4; // 132 floats = 528B (16-byte aligned)

    const int qh       = blockIdx.x;
    const int tok      = blockIdx.y;
    const int split_id = blockIdx.z;

    const int q_pos      = q_positions[tok];
    const int kv_start   = kv.is_local ? max(0, q_pos - swa_window + 1) : 0;
    const int kv_end     = q_pos + 1;
    const int total_k    = kv_end - kv_start;
    const int eff_splits = max(1, min(num_splits, (total_k + min_split_k - 1) / min_split_k));
    if (split_id >= eff_splits) { return; }

    const int b       = tok / q_per_lane;
    const int lane    = lane_ids[b];
    const int chunk   = (total_k + eff_splits - 1) / eff_splits;
    const int k_beg   = kv_start + split_id * chunk;
    const int k_end   = min(kv_end, k_beg + chunk);
    const int kv_h    = qh / (num_q_heads / num_kv_heads);

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

    const float* rb_row =
        rel_bias + (static_cast<std::int64_t>(tok) * num_q_heads + qh) * kRelExt;

    float m  = -std::numeric_limits<float>::infinity();
    float l  = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    int kpos = k_beg + warp;
    if (kv.is_local) {
        const auto* rk_base = static_cast<const __nv_bfloat16*>(kv.ring_k);
        const auto* rv_base = static_cast<const __nv_bfloat16*>(kv.ring_v);
        const std::int64_t lane_base = static_cast<std::int64_t>(lane) * kv.ring_cap;
        int slot = (kpos < kv.ring_cap) ? kpos : (kpos % kv.ring_cap);
        for (; kpos < k_end; kpos += kWarps) {
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
            const int dist = q_pos - kpos;
            if (dist < kRelExt) { dot += __ldg(rb_row + dist); }

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
        for (; kpos < k_end; kpos += kWarps) {
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
            const int dist = q_pos - kpos;
            if (dist < kRelExt) { dot += __ldg(rb_row + dist); }

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
        // Fast short-context single-pass finish (ZERO split_scratch, ZERO atomics!)
        const float sink = __bfloat162float(
            __ldg(qgrs + static_cast<std::int64_t>(tok) * packed_stride + sink_offset + qh));
        float gm = sink;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { gm = fmaxf(gm, s_m[w]); }
        float gl = __expf(sink - gm);
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
        const float gate_raw = __bfloat162float(__ldg(
            qgrs + static_cast<std::int64_t>(tok) * packed_stride + gate_offset +
            qh * kHeadDim + tid));
        const float gate_sig = 1.0f / (1.0f + __expf(-gate_raw));
        out[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + tid] =
            __float2bfloat16_rn((go / gl) * gate_sig);
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

    // Warp 0 loads (m, l) ONCE per split and computes scale factors & inv_l into SMEM!
    const float* base = split_scratch +
                        (static_cast<std::int64_t>(tok) * num_q_heads + qh) * num_splits * kSpStride;
    if (warp == 0) {
        const float sink = __bfloat162float(
            __ldg(qgrs + static_cast<std::int64_t>(tok) * packed_stride + sink_offset + qh));
        const float my_m = (wlane < eff_splits)
                               ? ld_cg_f32_vol(base + wlane * kSpStride + kHeadDim)
                               : -std::numeric_limits<float>::infinity();
        const float my_l = (wlane < eff_splits)
                               ? ld_cg_f32_vol(base + wlane * kSpStride + kHeadDim + 1)
                               : 0.0f;
        float gm = fmaxf(sink, my_m);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFFu, gm, off));
        }
        const float sc =
            (wlane < eff_splits && my_m > -std::numeric_limits<float>::infinity())
                ? __expf(my_m - gm)
                : 0.0f;
        if (wlane < eff_splits) { s_sc[wlane] = sc; }
        const float gl = __expf(sink - gm) + warp_sum(my_l * sc);
        if (wlane == 0) { s_inv_l = 1.0f / gl; }
    }
    __syncthreads();

    float go = 0.0f;
    for (int s = 0; s < eff_splits; ++s) {
        const float sc = s_sc[s];
        if (sc > 0.0f) { go = fmaf(ld_cg_f32_vol(base + s * kSpStride + tid), sc, go); }
    }
    const float gate_raw = __bfloat162float(__ldg(
        qgrs + static_cast<std::int64_t>(tok) * packed_stride + gate_offset +
        qh * kHeadDim + tid));
    const float gate_sig = 1.0f / (1.0f + __expf(-gate_raw));
    out[(static_cast<std::int64_t>(tok) * num_q_heads + qh) * kHeadDim + tid] =
        __float2bfloat16_rn((go * s_inv_l) * gate_sig);
}

// Legacy single-pass kernel (kept for max_splits <= 1 or split_done == nullptr fallback)
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

    const int d = threadIdx.x;
    if (d >= kHeadDim) { return; }
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
                            Tensor& q_out, Tensor& rel_bias_out, cudaStream_t stream,
                            const Inkling2AttnSconvArgs* sconv) {
    const int packed_stride = q_gate_r_sink.ne[0];
    const int q_dim         = num_q_heads * kHeadDim;
    const int r_offset      = 2 * q_dim;
    const dim3 grid(num_q_heads + num_kv_heads, tokens);

    const __nv_bfloat16* kw    = sconv ? static_cast<const __nv_bfloat16*>(sconv->k_weight) : nullptr;
    const __nv_bfloat16* vw    = sconv ? static_cast<const __nv_bfloat16*>(sconv->v_weight) : nullptr;
    const __nv_bfloat16* kdyn  = sconv ? static_cast<const __nv_bfloat16*>(sconv->k_dyn) : nullptr;
    const __nv_bfloat16* vdyn  = sconv ? static_cast<const __nv_bfloat16*>(sconv->v_dyn) : nullptr;
    float* kst                 = sconv ? sconv->k_state_pool : nullptr;
    float* vst                 = sconv ? sconv->v_state_pool : nullptr;
    float* kvst                = sconv ? sconv->k_verify_pool : nullptr;
    float* vvst                = sconv ? sconv->v_verify_pool : nullptr;
    const std::size_t stride   = sconv ? sconv->lane_stride : 0;
    const int qpl              = sconv ? sconv->q_per_lane : 1;
    const int max_v            = sconv ? (sconv->max_verify_len ? sconv->max_verify_len : qpl) : 0;

    auto launch = [&]<int Mode>() {
        attn_prologue_fused_kernel<Mode><<<grid, kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_gate_r_sink.data),
            static_cast<const __nv_bfloat16*>(k_in.data),
            static_cast<const __nv_bfloat16*>(v_in.data),
            static_cast<const __nv_bfloat16*>(q_clip_scale.data),
            static_cast<const __nv_bfloat16*>(rel_clip_scale.data),
            static_cast<const __nv_bfloat16*>(rel_logits_proj.data), positions, lane_ids, kv,
            packed_stride, r_offset, num_q_heads, num_kv_heads, is_local, qk_clip_threshold,
            rel_clip_threshold, log_floor, log_alpha, eps, static_cast<__nv_bfloat16*>(q_out.data),
            static_cast<float*>(rel_bias_out.data), kw, vw, kdyn, vdyn, kst, vst, kvst, vvst,
            stride, qpl, max_v);
    };

    if (sconv == nullptr) {
        launch.template operator()<0>();
    } else if (!sconv->is_dynamic) {
        if (sconv->k_verify_pool == nullptr) {
            launch.template operator()<1>();
        } else {
            launch.template operator()<2>();
        }
    } else {
        if (sconv->k_verify_pool == nullptr) {
            launch.template operator()<3>();
        } else {
            launch.template operator()<4>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attn_decode_verify(const Tensor& q, const Tensor& q_gate_r_sink,
                                 const Tensor& rel_bias_f32, const Inkling2KvTarget& kv,
                                 const std::int32_t* lane_ids, const std::int32_t* q_positions,
                                 std::int32_t batch_size, std::int32_t q_per_lane,
                                 std::int32_t num_q_heads, std::int32_t num_kv_heads,
                                 std::int32_t swa_window, float* split_scratch,
                                 std::int32_t max_splits, Tensor& out, cudaStream_t stream,
                                 std::int32_t* split_done) {
    const int total_tokens  = batch_size * q_per_lane;
    const int gqa_group     = num_q_heads / num_kv_heads;
    const int packed_stride = q_gate_r_sink.ne[0];
    const int q_dim         = num_q_heads * kHeadDim;
    const int gate_offset   = q_dim;
    const int sink_offset   = 2 * q_dim + num_q_heads * kDRel;
    if (gqa_group != 3) {
        throw std::invalid_argument("inkling2_attn_decode_verify: expected GQA group size 3");
    }

    const int base_q_ctas = num_q_heads * total_tokens;
    if (split_done != nullptr && max_splits > 1) {
        const int cap_splits = std::min(max_splits, 32);
        int num_splits       = 1;
        int min_split_k      = 64;
        if (base_q_ctas <= 16) {
            num_splits  = std::min(cap_splits, 16);
            min_split_k = 20;
        } else if (base_q_ctas <= 48) {
            num_splits  = kv.is_local ? std::min(cap_splits, 4) : std::min(cap_splits, 8);
            min_split_k = 64;
        } else {
            num_splits  = kv.is_local ? std::min(cap_splits, 2) : std::min(cap_splits, 4);
            min_split_k = 128;
        }
        attn_qhead_adaptive_kernel<4>
            <<<dim3(num_q_heads, total_tokens, num_splits), 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(q_gate_r_sink.data),
                static_cast<const float*>(rel_bias_f32.data), kv, lane_ids, q_positions, q_per_lane,
                packed_stride, gate_offset, sink_offset, num_q_heads, num_kv_heads, swa_window,
                num_splits, min_split_k, split_scratch, split_done,
                static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int base_ctas      = num_kv_heads * total_tokens;
    const bool use_single_pass = (max_splits <= 1) || (base_ctas >= 40);
    if (use_single_pass) {
        attn_decode_verify_single_kernel<3, 8>
            <<<dim3(num_kv_heads, total_tokens), 256, 0, stream>>>(
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
