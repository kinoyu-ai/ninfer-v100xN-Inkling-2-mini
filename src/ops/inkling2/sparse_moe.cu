#include "ninfer/ops/inkling2.h"

#include "core/device.h"
#include "ops/common/volta_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kNumExperts   = 128;
constexpr int kTopK         = 8;
constexpr int kHidden       = 2304;
constexpr int kInterPerSlab = 576;  // 1152 / 2
constexpr int kW13RowsSlab  = 1152; // 2 * 576 interleaved (gate, up)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ void warp_argmax(float& val, int& idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, val, offset);
        const int oi   = __shfl_down_sync(0xFFFFFFFFu, idx, offset);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
}

__device__ __forceinline__ uint4 ld_cg_u4(const void* ptr) {
    uint4 bits;
    // Non-volatile streaming weight load: lets nvcc overlap multiple HBM2 loads in flight!
    asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(ptr));
    return bits;
}

__device__ __forceinline__ float4 ld_cg_f4_volatile(const void* ptr) {
    float4 val;
    // L2-coherent read for cross-CTA Last-CTA router handoff (bypasses stale SM L1!)
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(ptr));
    return val;
}

__device__ __forceinline__ float softcapped_swiglu(float g, float u, float gate_cap, float up_cap) {
    const float sig = 1.0f / (1.0f + expf(-g));
    const float gc  = gate_cap * tanhf(g / gate_cap);
    const float uc  = up_cap * tanhf(u / up_cap);
    return gc * sig * uc;
}

// ============================================================================
// Exact Volta Shift Decoders
// ============================================================================

__device__ __forceinline__ void nvfp4_decode8_h2(std::uint32_t q, half2 rebias16384,
                                                 half2 (&out)[4]) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x0E000E00u;
    const std::uint32_t v0        = ((q << 12) & kSign) | ((q << 9) & kExpM);
    const std::uint32_t v1        = ((q << 8) & kSign) | ((q << 5) & kExpM);
    const std::uint32_t v2        = ((q << 4) & kSign) | ((q << 1) & kExpM);
    const std::uint32_t v3        = (q & kSign) | ((q >> 3) & kExpM);
    out[0]                        = __hmul2(*reinterpret_cast<const half2*>(&v0), rebias16384);
    out[1]                        = __hmul2(*reinterpret_cast<const half2*>(&v1), rebias16384);
    out[2]                        = __hmul2(*reinterpret_cast<const half2*>(&v2), rebias16384);
    out[3]                        = __hmul2(*reinterpret_cast<const half2*>(&v3), rebias16384);
}

__device__ __forceinline__ void nvfp4_stage8_h2(const half* seq, half2* out) {
    const auto* r    = reinterpret_cast<const std::uint32_t*>(seq);
    std::uint32_t o0 = __byte_perm(r[0], r[2], 0x5410);
    std::uint32_t o1 = __byte_perm(r[0], r[2], 0x7632);
    std::uint32_t o2 = __byte_perm(r[1], r[3], 0x5410);
    std::uint32_t o3 = __byte_perm(r[1], r[3], 0x7632);
    out[0]           = *reinterpret_cast<const half2*>(&o0);
    out[1]           = *reinterpret_cast<const half2*>(&o1);
    out[2]           = *reinterpret_cast<const half2*>(&o2);
    out[3]           = *reinterpret_cast<const half2*>(&o3);
}

__device__ __forceinline__ void bf16x8_to_staged_h2(const uint4& raw, half2 (&out)[4]) {
    const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
    half2 seq[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(xb[i]);
        seq[i]         = __floats2half2_rn(f.x, f.y);
    }
    nvfp4_stage8_h2(reinterpret_cast<const half*>(seq), out);
}

__device__ __forceinline__ void bf16x8_to_h2(const uint4& raw, half2 (&out)[4]) {
    const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(xb[i]);
        out[i]         = __floats2half2_rn(f.x, f.y);
    }
}

__device__ __forceinline__ float decode_e4m3_f32(std::uint8_t b) {
    const std::uint16_t hb = static_cast<std::uint16_t>((b & 0x80u) << 8) |
                             static_cast<std::uint16_t>((b & 0x7Fu) << 7);
    return __half2float(__ushort_as_half(hb)) * 256.0f;
}

__device__ __forceinline__ half2 decode_e4m3_h2_div256(std::uint8_t b) {
    const std::uint16_t hb = static_cast<std::uint16_t>((b & 0x80u) << 8) |
                             static_cast<std::uint16_t>((b & 0x7Fu) << 7);
    return __half2half2(__ushort_as_half(hb));
}

__device__ __forceinline__ void fp8_decode4_h2_div256(std::uint32_t word, half2& lo, half2& hi) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x3F803F80u;
    const std::uint32_t p0        = __byte_perm(word, word, 0x0110);
    const std::uint32_t p1        = __byte_perm(word, word, 0x2332);
    const std::uint32_t v0        = ((p0 << 8) & kSign) | ((p0 << 7) & kExpM);
    const std::uint32_t v1        = ((p1 << 8) & kSign) | ((p1 << 7) & kExpM);
    lo                            = *reinterpret_cast<const half2*>(&v0);
    hi                            = *reinterpret_cast<const half2*>(&v1);
}

__device__ __forceinline__ float decode_ue8m0_f32(std::uint8_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 23);
}

__device__ __forceinline__ float h2_dot16_f32(const half2 (&a0)[4], const half2 (&a1)[4],
                                              const half2 (&b0)[4], const half2 (&b1)[4]) {
    half2 p0 = __hmul2(a0[0], b0[0]);
    half2 p1 = __hmul2(a0[1], b0[1]);
    p0       = __hfma2(a0[2], b0[2], p0);
    p1       = __hfma2(a0[3], b0[3], p1);
    p0       = __hfma2(a1[0], b1[0], p0);
    p1       = __hfma2(a1[1], b1[1], p1);
    p0       = __hfma2(a1[2], b1[2], p0);
    p1       = __hfma2(a1[3], b1[3], p1);
    const float2 f = __half22float2(__hadd2(p0, p1));
    return f.x + f.y;
}

// ============================================================================
// 1. Ultra-Fast Warp-Register Top-8 Router + Fused Grouping (ZERO __syncthreads() for T=1!)
// ============================================================================

// Single warp (32 lanes, 4 experts/lane = 128 experts!) selects Top-8 purely in registers!
__device__ __forceinline__ void warp_select_top8_128(const float* __restrict__ logits_row,
                                                     const float* __restrict__ bias,
                                                     float route_scale,
                                                     std::int32_t* __restrict__ out_ids,
                                                     float* __restrict__ out_weights,
                                                     int* __restrict__ smem_ids_row) {
    const int lane   = threadIdx.x & 31;
    const int base_e = lane * 4;
    const float4 lv  = ld_cg_f4_volatile(logits_row + base_e);
    const float4 bv  = __ldg(reinterpret_cast<const float4*>(bias + base_e));

    float raw[4] = {
        1.0f / (1.0f + expf(-lv.x)),
        1.0f / (1.0f + expf(-lv.y)),
        1.0f / (1.0f + expf(-lv.z)),
        1.0f / (1.0f + expf(-lv.w)),
    };
    float score[4] = {
        raw[0] + bv.x,
        raw[1] + bv.y,
        raw[2] + bv.z,
        raw[3] + bv.w,
    };

    int chosen_id[kTopK];
    float chosen_w[kTopK];
    float wsum = 0.0f;

#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        // Local best across this lane's 4 experts
        float lv_max = score[0];
        int li_max   = 0;
        if (score[1] > lv_max) { lv_max = score[1]; li_max = 1; }
        if (score[2] > lv_max) { lv_max = score[2]; li_max = 2; }
        if (score[3] > lv_max) { lv_max = score[3]; li_max = 3; }

        float v = lv_max;
        int id  = base_e + li_max;
        warp_argmax(v, id);
        const int best_id = __shfl_sync(0xFFFFFFFFu, id, 0);

        float my_raw = 0.0f;
        if ((best_id >> 2) == lane) {
            const int sub = best_id & 3;
            my_raw        = raw[sub];
            score[sub]    = -std::numeric_limits<float>::infinity();
        }
        const float best_raw = __shfl_sync(0xFFFFFFFFu, my_raw, best_id >> 2);
        chosen_id[k]         = best_id;
        chosen_w[k]          = best_raw;
        wsum += best_raw;
    }

    if (lane < kTopK) {
        const float inv   = route_scale / (wsum + 1e-20f);
        out_ids[lane]     = chosen_id[lane];
        out_weights[lane] = chosen_w[lane] * inv;
        if (smem_ids_row != nullptr) { smem_ids_row[lane] = chosen_id[lane]; }
    }
}

// T == 1: 1 warp (32 threads), ZERO shared memory, ZERO __syncthreads()!
__global__ void moe_route_t1_warp_kernel(const float* __restrict__ logits,
                                         const float* __restrict__ bias, float route_scale,
                                         std::int32_t* __restrict__ topk_ids,
                                         float* __restrict__ topk_weights) {
    warp_select_top8_128(logits, bias, route_scale, topk_ids, topk_weights, nullptr);
}

// T in [2..64]: FUSED Top-8 + Expert Grouping in ONE SINGLE KERNEL (4 warps = 128 threads)!
// Eliminates 48 separate grouping launches + topk_ids global round-trip!
__global__ void moe_route_and_group_small_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias, float route_scale, int tokens,
    std::int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights,
    std::int32_t* __restrict__ expert_counts, std::int32_t* __restrict__ expert_offsets,
    std::int32_t* __restrict__ sorted_pairs, std::int32_t* __restrict__ active_experts,
    std::int32_t* __restrict__ num_active) {
    __shared__ int s_ids[512]; // up to 64 * 8
    __shared__ int s_wsum[4];
    __shared__ int s_wact[4];

    const int e    = threadIdx.x; // 0..127
    const int warp = e >> 5;      // 0..3
    const int lane = e & 31;

    for (int t = warp; t < tokens; t += 4) {
        warp_select_top8_128(logits + static_cast<std::int64_t>(t) * kNumExperts, bias, route_scale,
                             topk_ids + t * kTopK, topk_weights + t * kTopK, s_ids + t * kTopK);
    }
    __syncthreads();

    const int total = tokens * kTopK;
    int cnt         = 0;
    for (int i = 0; i < total; ++i) { cnt += (s_ids[i] == e); }
    expert_counts[e] = cnt;

    int off_inc = cnt;
    int act_inc = (cnt > 0) ? 1 : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int o = __shfl_up_sync(0xFFFFFFFFu, off_inc, d);
        const int a = __shfl_up_sync(0xFFFFFFFFu, act_inc, d);
        if (lane >= d) {
            off_inc += o;
            act_inc += a;
        }
    }
    if (lane == 31) {
        s_wsum[warp] = off_inc;
        s_wact[warp] = act_inc;
    }
    __syncthreads();

    int warp_off_base = 0;
    int warp_act_base = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (w < warp) {
            warp_off_base += s_wsum[w];
            warp_act_base += s_wact[w];
        }
    }
    const int my_off = warp_off_base + off_inc - cnt;
    const int my_act = warp_act_base + act_inc - ((cnt > 0) ? 1 : 0);
    expert_offsets[e] = my_off;
    if (cnt > 0) { active_experts[my_act] = e; }
    if (e == 127) {
        expert_offsets[kNumExperts] = warp_off_base + off_inc;
        num_active[0]               = warp_act_base + act_inc;
    }

    if (cnt > 0) {
        int cursor = my_off;
        for (int i = 0; i < total; ++i) {
            if (s_ids[i] == e) { sorted_pairs[cursor++] = i; }
        }
    }
}

// Fused Router GEMV + Sigmoid + Bias Top-8 (+ Grouping for T=2..8) in ONE SINGLE LAUNCH!
// grid = 128, block = WarpsPerRow * 32 (4 warps for T<=2, 8 warps for T=3..8).
// Last CTA via atomicAdd(done_counter, 1) == 127 immediately runs Top-8 + grouping on L2-hot logits!
template <int ActiveT, int WarpsPerRow>
__global__ __launch_bounds__(WarpsPerRow * 32, 2) void moe_router_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ bias, float route_scale, float* __restrict__ router_logits,
    std::int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights,
    std::int32_t* __restrict__ expert_counts, std::int32_t* __restrict__ expert_offsets,
    std::int32_t* __restrict__ sorted_pairs, std::int32_t* __restrict__ active_experts,
    std::int32_t* __restrict__ num_active, std::int32_t* __restrict__ done_counter) {
    __shared__ float smem[ActiveT][WarpsPerRow];
    __shared__ int s_is_last;

    const int row  = blockIdx.x; // 0..127
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const __nv_bfloat16* wrow = w + static_cast<std::int64_t>(row) * kHidden;
    constexpr int kK8         = kHidden / 8; // 288
    float acc[ActiveT]        = {};

    for (int i = tid; i < kK8; i += WarpsPerRow * 32) {
        const uint4 wv = ld_cg_u4(reinterpret_cast<const uint4*>(wrow) + i);
        const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(wb[j]);
            wf[2 * j]      = f.x;
            wf[2 * j + 1]  = f.y;
        }
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(ti) * kHidden) + i);
            const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 xf = __bfloat1622float2(xb[j]);
                a0              = fmaf(wf[2 * j], xf.x, a0);
                a1              = fmaf(wf[2 * j + 1], xf.y, a1);
            }
            acc[ti] += a0 + a1;
        }
    }

#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        const float ws = warp_sum(acc[ti]);
        if (lane == 0) { smem[ti][warp] = ws; }
    }
    __syncthreads();

    if (warp == 0 && lane < ActiveT) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < WarpsPerRow; ++s) { sum += smem[lane][s]; }
        router_logits[static_cast<std::int64_t>(lane) * kNumExperts + row] = sum;
    }
    __threadfence();
    __syncthreads();

    if (tid == 0) { s_is_last = (atomicAdd(done_counter, 1) == (kNumExperts - 1)) ? 1 : 0; }
    __syncthreads();
    if (!s_is_last) { return; }

    if (tid == 0) { done_counter[0] = 0; }

    if constexpr (ActiveT == 1) {
        if (warp == 0) {
            warp_select_top8_128(router_logits, bias, route_scale, topk_ids, topk_weights, nullptr);
        }
        return;
    } else {
        __shared__ int s_ids[ActiveT * kTopK];
        __shared__ int s_wsum[4];
        __shared__ int s_wact[4];
        for (int t = warp; t < ActiveT; t += WarpsPerRow) {
            warp_select_top8_128(router_logits + static_cast<std::int64_t>(t) * kNumExperts, bias,
                                 route_scale, topk_ids + t * kTopK, topk_weights + t * kTopK,
                                 s_ids + t * kTopK);
        }
        __syncthreads();

        // First 128 threads (warps 0..3) perform the 128-expert histogram & prefix sum!
        if (tid < kNumExperts) {
            const int e         = tid;
            constexpr int total = ActiveT * kTopK;
            int cnt             = 0;
#pragma unroll
            for (int i = 0; i < total; ++i) { cnt += (s_ids[i] == e); }
            expert_counts[e] = cnt;

            int off_inc = cnt;
            int act_inc = (cnt > 0) ? 1 : 0;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int o = __shfl_up_sync(0xFFFFFFFFu, off_inc, d);
                const int a = __shfl_up_sync(0xFFFFFFFFu, act_inc, d);
                if (lane >= d) {
                    off_inc += o;
                    act_inc += a;
                }
            }
            if (lane == 31) {
                s_wsum[warp] = off_inc;
                s_wact[warp] = act_inc;
            }
            __syncthreads();

            int warp_off_base = 0;
            int warp_act_base = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (w < warp) {
                    warp_off_base += s_wsum[w];
                    warp_act_base += s_wact[w];
                }
            }
            const int my_off  = warp_off_base + off_inc - cnt;
            const int my_act  = warp_act_base + act_inc - ((cnt > 0) ? 1 : 0);
            expert_offsets[e] = my_off;
            if (cnt > 0) { active_experts[my_act] = e; }
            if (e == 127) {
                expert_offsets[kNumExperts] = warp_off_base + off_inc;
                num_active[0]               = warp_act_base + act_inc;
            }
            if (cnt > 0) {
                int cursor = my_off;
#pragma unroll
                for (int i = 0; i < total; ++i) {
                    if (s_ids[i] == e) { sorted_pairs[cursor++] = i; }
                }
            }
        } else {
            __syncthreads();
        }
    }
}

// Large-T (T > 64, prefill) separate topk + grouping
__global__ void moe_route_topk_large_kernel(const float* __restrict__ logits,
                                            const float* __restrict__ bias, float route_scale,
                                            std::int32_t* __restrict__ topk_ids,
                                            float* __restrict__ topk_weights, int tokens) {
    const int t = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (t >= tokens) { return; }
    warp_select_top8_128(logits + static_cast<std::int64_t>(t) * kNumExperts, bias, route_scale,
                         topk_ids + t * kTopK, topk_weights + t * kTopK, nullptr);
}

__global__ void moe_group_assignments_large_kernel(
    const std::int32_t* __restrict__ topk_ids, int tokens,
    std::int32_t* __restrict__ expert_counts, std::int32_t* __restrict__ expert_offsets,
    std::int32_t* __restrict__ sorted_pairs, std::int32_t* __restrict__ active_experts,
    std::int32_t* __restrict__ num_active) {
    __shared__ int s_wsum[4];
    __shared__ int s_wact[4];
    const int e     = threadIdx.x;
    const int warp  = e >> 5;
    const int lane  = e & 31;
    const int total = tokens * kTopK;

    int cnt = 0;
    for (int i = 0; i < total; ++i) { cnt += (topk_ids[i] == e); }
    expert_counts[e] = cnt;

    int off_inc = cnt;
    int act_inc = (cnt > 0) ? 1 : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int o = __shfl_up_sync(0xFFFFFFFFu, off_inc, d);
        const int a = __shfl_up_sync(0xFFFFFFFFu, act_inc, d);
        if (lane >= d) {
            off_inc += o;
            act_inc += a;
        }
    }
    if (lane == 31) {
        s_wsum[warp] = off_inc;
        s_wact[warp] = act_inc;
    }
    __syncthreads();

    int warp_off_base = 0;
    int warp_act_base = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (w < warp) {
            warp_off_base += s_wsum[w];
            warp_act_base += s_wact[w];
        }
    }
    const int my_off = warp_off_base + off_inc - cnt;
    const int my_act = warp_act_base + act_inc - ((cnt > 0) ? 1 : 0);
    expert_offsets[e] = my_off;
    if (cnt > 0) { active_experts[my_act] = e; }
    if (e == 127) {
        expert_offsets[kNumExperts] = warp_off_base + off_inc;
        num_active[0]               = warp_act_base + act_inc;
    }
    if (cnt > 0) {
        int cursor = my_off;
        for (int i = 0; i < total; ++i) {
            if (topk_ids[i] == e) { sorted_pairs[cursor++] = i; }
        }
    }
}

// ============================================================================
// 2. Ultra-Fast 288-Thread (16 rows x 18 K-lanes) 100%-Coalesced NVFP4 T == 1 Kernels
//    - 18 K-lanes * 16 bytes (uint4) = 288 contiguous bytes per load!
//    - w13: 4 iters * 18 = 72 uint4 = 144 groups = 2304 EXACT, ZERO tail, 16 pairs/CTA!
//    - w2:  1 iter  * 18 = 18 uint4 = 36 groups  = 576  EXACT, ZERO K-loop, 16 rows/CTA!
// ============================================================================

__global__ __launch_bounds__(288, 3) void moe_nvfp4_t1_w13_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom, const std::int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights, int num_slabs, float gate_cap, float up_cap,
    half* __restrict__ inter_out) {
    __shared__ half2 s_x[144][8];
    __shared__ float s_red[2][16][18];

    const int tid = threadIdx.x; // 0..287
    if (tid < 144) {
        const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(x + tid * 16));
        const uint4 x1 = __ldg(reinterpret_cast<const uint4*>(x + tid * 16 + 8));
        half2 xa0[4], xa1[4];
        bf16x8_to_staged_h2(x0, xa0);
        bf16x8_to_staged_h2(x1, xa1);
        *reinterpret_cast<uint4*>(&s_x[tid][0]) = *reinterpret_cast<const uint4*>(xa0);
        *reinterpret_cast<uint4*>(&s_x[tid][4]) = *reinterpret_cast<const uint4*>(xa1);
    }
    __syncthreads();

    const int p_loc       = tid / 18; // 0..15
    const int k_lane      = tid - p_loc * 18; // 0..17
    const int inter_local = num_slabs * kInterPerSlab;
    const int j_global    = blockIdx.x * 16 + p_loc;
    const int slot        = blockIdx.y;
    const int slab        = j_global / kInterPerSlab;
    const int j_local     = j_global - slab * kInterPerSlab;
    const int eid         = topk_ids[slot];

    const std::int64_t row_base =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + 2 * j_local;
    const std::uint8_t* gc_ptr = w13_codes + row_base * 1152;
    const std::uint8_t* uc_ptr = gc_ptr + 1152;
    const std::uint8_t* gs_ptr = w13_scales + row_base * 144;
    const std::uint8_t* us_ptr = gs_ptr + 144;

    const half2 rebias = __float2half2_rn(16384.0f);
    float acc_g        = 0.0f;
    float acc_u        = 0.0f;

#pragma unroll 1
    for (int it = 0; it < 4; ++it) {
        const int q  = it * 18 + k_lane;
        const int g0 = q * 2;
        const int g1 = g0 + 1;

        const uint4 gw         = ld_cg_u4(gc_ptr + q * 16);
        const uint4 uw         = ld_cg_u4(uc_ptr + q * 16);
        const std::uint16_t gs = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + q * 2));
        const std::uint16_t us = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + q * 2));

        half2 xa0[4], xa1[4], d0[4], d1[4];
        *reinterpret_cast<uint4*>(xa0) = *reinterpret_cast<const uint4*>(&s_x[g0][0]);
        *reinterpret_cast<uint4*>(xa1) = *reinterpret_cast<const uint4*>(&s_x[g0][4]);
        nvfp4_decode8_h2(gw.x, rebias, d0);
        nvfp4_decode8_h2(gw.y, rebias, d1);
        acc_g = fmaf(h2_dot16_f32(xa0, xa1, d0, d1), decode_e4m3_f32(gs & 0xFFu), acc_g);
        nvfp4_decode8_h2(uw.x, rebias, d0);
        nvfp4_decode8_h2(uw.y, rebias, d1);
        acc_u = fmaf(h2_dot16_f32(xa0, xa1, d0, d1), decode_e4m3_f32(us & 0xFFu), acc_u);

        *reinterpret_cast<uint4*>(xa0) = *reinterpret_cast<const uint4*>(&s_x[g1][0]);
        *reinterpret_cast<uint4*>(xa1) = *reinterpret_cast<const uint4*>(&s_x[g1][4]);
        nvfp4_decode8_h2(gw.z, rebias, d0);
        nvfp4_decode8_h2(gw.w, rebias, d1);
        acc_g = fmaf(h2_dot16_f32(xa0, xa1, d0, d1), decode_e4m3_f32(gs >> 8), acc_g);
        nvfp4_decode8_h2(uw.z, rebias, d0);
        nvfp4_decode8_h2(uw.w, rebias, d1);
        acc_u = fmaf(h2_dot16_f32(xa0, xa1, d0, d1), decode_e4m3_f32(us >> 8), acc_u);
    }

    s_red[0][p_loc][k_lane] = acc_g;
    s_red[1][p_loc][k_lane] = acc_u;
    __syncthreads();

    if (tid < 16) {
        float gsum = 0.0f, usum = 0.0f;
#pragma unroll
        for (int k = 0; k < 18; ++k) {
            gsum += s_red[0][tid][k];
            usum += s_red[1][tid][k];
        }
        const float outer = __ldg(w13_amax + eid) / __ldg(w13_denom);
        const float y     = softcapped_swiglu(gsum * outer, usum * outer, gate_cap, up_cap) *
                            topk_weights[slot];
        const int jg    = blockIdx.x * 16 + tid;
        const int sl    = jg / kInterPerSlab;
        const int jl    = jg - sl * kInterPerSlab;
        const int blk8  = jl & ~7;
        const int p8    = jl & 7;
        const int dst_j = sl * kInterPerSlab + blk8 + ((p8 & 3) * 2 + (p8 >> 2));
        inter_out[static_cast<std::int64_t>(slot) * inter_local + dst_j] = __float2half_rn(y);
    }
}

__global__ __launch_bounds__(288, 3) void moe_nvfp4_t1_w2_kernel(
    const half* __restrict__ inter_in, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom, const std::int32_t* __restrict__ topk_ids,
    int num_slabs, __nv_bfloat16* __restrict__ out) {
    __shared__ half2 s_inter[8][36][8];
    __shared__ float s_red[16][18];
    __shared__ int s_eid[8];
    __shared__ float s_outer[8];

    const int tid    = threadIdx.x; // 0..287
    const int r_loc  = tid / 18;    // 0..15
    const int k_lane = tid - r_loc * 18; // 0..17
    const int row    = blockIdx.x * 16 + r_loc;

    const float inv_denom = 1.0f / __ldg(w2_denom);
    if (tid < kTopK) {
        const int eid = topk_ids[tid];
        s_eid[tid]    = eid;
        s_outer[tid]  = __ldg(w2_amax + eid) * inv_denom;
    }

    const int inter_local = num_slabs * kInterPerSlab;
    const half2 rebias    = __float2half2_rn(16384.0f);
    float total           = 0.0f;

    const int load_slot = tid / 36; // 0..7
    const int load_g    = tid - load_slot * 36; // 0..35

    for (int slab = 0; slab < num_slabs; ++slab) {
        const half* src_ptr = inter_in + static_cast<std::int64_t>(load_slot) * inter_local +
                              slab * kInterPerSlab + load_g * 16;
        *reinterpret_cast<uint4*>(&s_inter[load_slot][load_g][0]) =
            __ldg(reinterpret_cast<const uint4*>(src_ptr));
        *reinterpret_cast<uint4*>(&s_inter[load_slot][load_g][4]) =
            __ldg(reinterpret_cast<const uint4*>(src_ptr + 8));
        __syncthreads();

        const int g0 = k_lane * 2;
        const int g1 = g0 + 1;
#pragma unroll 1
        for (int slot = 0; slot < kTopK; ++slot) {
            const int eid = s_eid[slot];
            const std::int64_t row_idx =
                (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + row;
            const uint4 ww         = ld_cg_u4(w2_codes + row_idx * 288 + k_lane * 16);
            const std::uint16_t ws = __ldg(reinterpret_cast<const std::uint16_t*>(
                w2_scales + row_idx * 36 + k_lane * 2));

            half2 wd0[4], wd1[4], xa0[4], xa1[4];
            nvfp4_decode8_h2(ww.x, rebias, wd0);
            nvfp4_decode8_h2(ww.y, rebias, wd1);
            *reinterpret_cast<uint4*>(xa0) = *reinterpret_cast<const uint4*>(&s_inter[slot][g0][0]);
            *reinterpret_cast<uint4*>(xa1) = *reinterpret_cast<const uint4*>(&s_inter[slot][g0][4]);
            float acc = h2_dot16_f32(xa0, xa1, wd0, wd1) * decode_e4m3_f32(ws & 0xFFu);

            nvfp4_decode8_h2(ww.z, rebias, wd0);
            nvfp4_decode8_h2(ww.w, rebias, wd1);
            *reinterpret_cast<uint4*>(xa0) = *reinterpret_cast<const uint4*>(&s_inter[slot][g1][0]);
            *reinterpret_cast<uint4*>(xa1) = *reinterpret_cast<const uint4*>(&s_inter[slot][g1][4]);
            acc = fmaf(h2_dot16_f32(xa0, xa1, wd0, wd1), decode_e4m3_f32(ws >> 8), acc);

            total = fmaf(acc, s_outer[slot], total);
        }
        if (num_slabs > 1) { __syncthreads(); }
    }

    s_red[r_loc][k_lane] = total;
    __syncthreads();

    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 18; ++k) { sum += s_red[tid][k]; }
        out[blockIdx.x * 16 + tid] = __float2bfloat16_rn(sum);
    }
}

// ============================================================================
// 3. Ultra-Fast 288-Thread (16 x 18 K-lanes) 100%-Coalesced MXFP8 T == 1 Kernels
// ============================================================================

__global__ __launch_bounds__(288, 2) void moe_mxfp8_t1_w13_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const std::int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights, int num_slabs, float gate_cap, float up_cap,
    half* __restrict__ inter_out) {
    __shared__ half2 s_x[72][16];
    __shared__ float s_red[2][16][18];

    const int tid = threadIdx.x;
    if (tid < 288) {
        const int g  = tid >> 2; // 0..71
        const int q  = tid & 3;  // 0..3
        const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + g * 32 + q * 8));
        half2 xh[4];
        bf16x8_to_h2(xv, xh);
        *reinterpret_cast<uint4*>(&s_x[g][q * 4]) = *reinterpret_cast<const uint4*>(xh);
    }
    __syncthreads();

    const int p_loc       = tid / 18;
    const int k_lane      = tid - p_loc * 18;
    const int inter_local = num_slabs * kInterPerSlab;
    const int j_global    = blockIdx.x * 16 + p_loc;
    const int slot        = blockIdx.y;
    const int slab        = j_global / kInterPerSlab;
    const int j_local     = j_global - slab * kInterPerSlab;
    const int eid         = topk_ids[slot];

    const std::int64_t row_base =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + 2 * j_local;
    const std::uint8_t* gc_ptr = w13_codes + row_base * 2304;
    const std::uint8_t* uc_ptr = gc_ptr + 2304;
    const std::uint8_t* gs_ptr = w13_scales + row_base * 72;
    const std::uint8_t* us_ptr = gs_ptr + 72;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int g    = it * 18 + k_lane;
        const float gs = decode_ue8m0_f32(__ldg(gs_ptr + g)) * 256.0f;
        const float us = decode_ue8m0_f32(__ldg(us_ptr + g)) * 256.0f;
        half2 gp0 = __half2half2(__ushort_as_half(0));
        half2 gp1 = __half2half2(__ushort_as_half(0));
        half2 up0 = __half2half2(__ushort_as_half(0));
        half2 up1 = __half2half2(__ushort_as_half(0));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 gv            = ld_cg_u4(gc_ptr + g * 32 + h * 16);
            const uint4 uv            = ld_cg_u4(uc_ptr + g * 32 + h * 16);
            const std::uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
            const std::uint32_t uw[4] = {uv.x, uv.y, uv.z, uv.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const half2 x0 = s_x[g][h * 8 + w * 2 + 0];
                const half2 x1 = s_x[g][h * 8 + w * 2 + 1];
                half2 lo, hi;
                fp8_decode4_h2_div256(gw[w], lo, hi);
                gp0 = __hfma2(x0, lo, gp0);
                gp1 = __hfma2(x1, hi, gp1);
                fp8_decode4_h2_div256(uw[w], lo, hi);
                up0 = __hfma2(x0, lo, up0);
                up1 = __hfma2(x1, hi, up1);
            }
        }
        const float2 gf = __half22float2(__hadd2(gp0, gp1));
        const float2 uf = __half22float2(__hadd2(up0, up1));
        acc_g           = fmaf(gf.x + gf.y, gs, acc_g);
        acc_u           = fmaf(uf.x + uf.y, us, acc_u);
    }

    s_red[0][p_loc][k_lane] = acc_g;
    s_red[1][p_loc][k_lane] = acc_u;
    __syncthreads();

    if (tid < 16) {
        float gsum = 0.0f, usum = 0.0f;
#pragma unroll
        for (int k = 0; k < 18; ++k) {
            gsum += s_red[0][tid][k];
            usum += s_red[1][tid][k];
        }
        const float y = softcapped_swiglu(gsum, usum, gate_cap, up_cap) * topk_weights[slot];
        inter_out[static_cast<std::int64_t>(slot) * inter_local + blockIdx.x * 16 + tid] =
            __float2half_rn(y);
    }
}

__global__ __launch_bounds__(288, 2) void moe_mxfp8_t1_w2_kernel(
    const half* __restrict__ inter_in, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const std::int32_t* __restrict__ topk_ids,
    int num_slabs, __nv_bfloat16* __restrict__ out) {
    __shared__ half2 s_inter[8][18][16];
    __shared__ float s_red[16][18];
    __shared__ int s_eid[8];

    const int tid    = threadIdx.x;
    const int r_loc  = tid / 18;
    const int k_lane = tid - r_loc * 18;
    const int row    = blockIdx.x * 16 + r_loc;

    if (tid < kTopK) { s_eid[tid] = topk_ids[tid]; }

    const int inter_local = num_slabs * kInterPerSlab;
    float total           = 0.0f;

    for (int slab = 0; slab < num_slabs; ++slab) {
        // 8 * 18 = 144 groups -> first 144 threads load 4 uint4 each (or all 288 load 2 uint4 each!)
        const int idx = tid >> 1; // 0..143
        const int hlf = tid & 1;  // 0..1
        const int slt = idx / 18;
        const int grp = idx - slt * 18;
        const half* sp = inter_in + static_cast<std::int64_t>(slt) * inter_local +
                         slab * kInterPerSlab + grp * 32 + hlf * 16;
        *reinterpret_cast<uint4*>(&s_inter[slt][grp][hlf * 8 + 0]) =
            __ldg(reinterpret_cast<const uint4*>(sp));
        *reinterpret_cast<uint4*>(&s_inter[slt][grp][hlf * 8 + 4]) =
            __ldg(reinterpret_cast<const uint4*>(sp + 8));
        __syncthreads();

#pragma unroll
        for (int slot = 0; slot < kTopK; ++slot) {
            const int eid = s_eid[slot];
            const std::int64_t row_idx =
                (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + row;
            const float ws = decode_ue8m0_f32(__ldg(w2_scales + row_idx * 18 + k_lane)) * 256.0f;
            half2 p0       = __half2half2(__ushort_as_half(0));
            half2 p1       = __half2half2(__ushort_as_half(0));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 wv             = ld_cg_u4(w2_codes + row_idx * 576 + k_lane * 32 + h * 16);
                const std::uint32_t ws4[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    half2 lo, hi;
                    fp8_decode4_h2_div256(ws4[w], lo, hi);
                    p0 = __hfma2(s_inter[slot][k_lane][h * 8 + w * 2 + 0], lo, p0);
                    p1 = __hfma2(s_inter[slot][k_lane][h * 8 + w * 2 + 1], hi, p1);
                }
            }
            const float2 pf = __half22float2(__hadd2(p0, p1));
            total           = fmaf(pf.x + pf.y, ws, total);
        }
        if (num_slabs > 1) { __syncthreads(); }
    }

    s_red[r_loc][k_lane] = total;
    __syncthreads();
    if (tid < 16) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 18; ++k) { sum += s_red[tid][k]; }
        out[blockIdx.x * 16 + tid] = __float2bfloat16_rn(sum);
    }
}

// ============================================================================
// 4. Pre-stage x BF16 -> FP16 (staged (j, j+4) for NVFP4 or plain for MXFP8)
// ============================================================================

template <bool StageNvfp4>
__global__ void prestage_x_fp16_kernel(const __nv_bfloat16* __restrict__ x, half* __restrict__ out,
                                       int tokens) {
    const int g = threadIdx.x; // 0..287
    const int t = blockIdx.x;
    if (t >= tokens) { return; }
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
        x + static_cast<std::int64_t>(t) * kHidden + g * 8));
    half2 h[4];
    if constexpr (StageNvfp4) {
        bf16x8_to_staged_h2(raw, h);
    } else {
        bf16x8_to_h2(raw, h);
    }
    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(t) * kHidden + g * 8) =
        *reinterpret_cast<const uint4*>(h);
}

struct MoeXScratch {
    half* data      = nullptr;
    std::size_t cap = 0;
};

half* get_moe_x_scratch(int device, std::size_t elems) {
    static MoeXScratch sc[8];
    auto& s                = sc[device & 7];
    const std::size_t need = std::max<std::size_t>(elems, 256ULL * kHidden);
    if (s.cap < need) {
        if (s.data) { (void)cudaFree(s.data); }
        CUDA_CHECK(cudaMalloc(&s.data, need * sizeof(half)));
        s.cap = need;
    }
    return s.data;
}

// ============================================================================
// 5. Ultra-Fast ZERO-SPILL Coalesced-Row Grouped MoE for T = 2..32 (NVFP4 & MXFP8)
//    Verified 0 bytes stack/spill via ptxas -v!
//    - #pragma unroll 1 on K-loop + 16-weight streaming decode (<= 56 regs -> 3-4 CTAs/SM!)
//    - w2 uses 2 rows/thread (32 rows/CTA, grid.x = num_slabs * 72 -> 2x MLP, <= 52 regs, 0 spill!)
// ============================================================================

__global__ __launch_bounds__(288, 2) void moe_nvfp4_grouped_w13_coalesced_kernel(
    const half* __restrict__ x_staged, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, const float* __restrict__ topk_weights,
    int num_slabs, float gate_cap, float up_cap, half* __restrict__ inter_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];
    __shared__ float s_w[4];
    __shared__ float s_red[2][16][18];

    const int tid         = threadIdx.x; // 0..287
    const int p_loc       = tid / 18;    // 0..15
    const int k_lane      = tid - p_loc * 18;
    const int inter_local = num_slabs * kInterPerSlab;
    const int j_global    = blockIdx.x * 16 + p_loc;
    const int slab        = j_global / kInterPerSlab;
    const int j_local     = j_global - slab * kInterPerSlab;

    const std::int64_t row_base =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + 2 * j_local;
    const std::uint8_t* gc_ptr = w13_codes + row_base * 1152;
    const std::uint8_t* uc_ptr = gc_ptr + 1152;
    const std::uint8_t* gs_ptr = w13_scales + row_base * 144;
    const std::uint8_t* us_ptr = gs_ptr + 144;

    const half2 rebias = __float2half2_rn(16384.0f);
    const float outer  = __ldg(w13_amax + eid) / __ldg(w13_denom);

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
            s_w[tid]       = topk_weights[pair];
        }
        __syncthreads();

        if (m_cnt == 1) {
            const half* x0 = x_staged + static_cast<std::int64_t>(s_toks[0]) * kHidden;
            float ag = 0.0f, au = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 4; ++it) {
                const int q  = it * 18 + k_lane;
                const int k0 = q * 32;

                const uint4 gw         = ld_cg_u4(gc_ptr + q * 16);
                const uint4 uw         = ld_cg_u4(uc_ptr + q * 16);
                const std::uint16_t gs = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + q * 2));
                const std::uint16_t us = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + q * 2));

                half2 a0[4], a1[4], d0[4], d1[4];
                *reinterpret_cast<uint4*>(a0) = __ldg(reinterpret_cast<const uint4*>(x0 + k0 + 0));
                *reinterpret_cast<uint4*>(a1) = __ldg(reinterpret_cast<const uint4*>(x0 + k0 + 8));
                nvfp4_decode8_h2(gw.x, rebias, d0);
                nvfp4_decode8_h2(gw.y, rebias, d1);
                ag = fmaf(h2_dot16_f32(a0, a1, d0, d1), decode_e4m3_f32(gs & 0xFFu), ag);
                nvfp4_decode8_h2(uw.x, rebias, d0);
                nvfp4_decode8_h2(uw.y, rebias, d1);
                au = fmaf(h2_dot16_f32(a0, a1, d0, d1), decode_e4m3_f32(us & 0xFFu), au);

                *reinterpret_cast<uint4*>(a0) = __ldg(reinterpret_cast<const uint4*>(x0 + k0 + 16));
                *reinterpret_cast<uint4*>(a1) = __ldg(reinterpret_cast<const uint4*>(x0 + k0 + 24));
                nvfp4_decode8_h2(gw.z, rebias, d0);
                nvfp4_decode8_h2(gw.w, rebias, d1);
                ag = fmaf(h2_dot16_f32(a0, a1, d0, d1), decode_e4m3_f32(gs >> 8), ag);
                nvfp4_decode8_h2(uw.z, rebias, d0);
                nvfp4_decode8_h2(uw.w, rebias, d1);
                au = fmaf(h2_dot16_f32(a0, a1, d0, d1), decode_e4m3_f32(us >> 8), au);
            }
            s_red[0][p_loc][k_lane] = ag;
            s_red[1][p_loc][k_lane] = au;
            __syncthreads();
            if (tid < 16) {
                float gsum = 0.0f, usum = 0.0f;
#pragma unroll
                for (int k = 0; k < 18; ++k) {
                    gsum += s_red[0][tid][k];
                    usum += s_red[1][tid][k];
                }
                const float y   = softcapped_swiglu(gsum * outer, usum * outer, gate_cap, up_cap) * s_w[0];
                const int jg    = blockIdx.x * 16 + tid;
                const int sl    = jg / kInterPerSlab;
                const int jl    = jg - sl * kInterPerSlab;
                const int blk8  = jl & ~7;
                const int p8    = jl & 7;
                const int dst_j = sl * kInterPerSlab + blk8 + ((p8 & 3) * 2 + (p8 >> 2));
                inter_out[static_cast<std::int64_t>(s_pairs[0]) * inter_local + dst_j] = __float2half_rn(y);
            }
        } else {
            // m_cnt in 2..4: stream 16 weights at a time (d0, d1 = only 8 regs!) -> ZERO SPILLS!
            float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
            float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

#pragma unroll 1
            for (int it = 0; it < 4; ++it) {
                const int q  = it * 18 + k_lane;
                const int k0 = q * 32;

                const uint4 gw         = ld_cg_u4(gc_ptr + q * 16);
                const uint4 uw         = ld_cg_u4(uc_ptr + q * 16);
                const std::uint16_t gs = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + q * 2));
                const std::uint16_t us = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + q * 2));

                half2 d0[4], d1[4], a0[4], a1[4];

                auto dot16_all = [&](std::uint32_t wlo, std::uint32_t whi, int koff, float sc,
                                     float& acc0, float& acc1, float& acc2, float& acc3) {
                    nvfp4_decode8_h2(wlo, rebias, d0);
                    nvfp4_decode8_h2(whi, rebias, d1);
                    const half* p0 = x_staged + static_cast<std::int64_t>(s_toks[0]) * kHidden + koff;
                    *reinterpret_cast<uint4*>(a0) = __ldg(reinterpret_cast<const uint4*>(p0));
                    *reinterpret_cast<uint4*>(a1) = __ldg(reinterpret_cast<const uint4*>(p0 + 8));
                    acc0 = fmaf(h2_dot16_f32(a0, a1, d0, d1), sc, acc0);

                    const half* p1 = x_staged + static_cast<std::int64_t>(s_toks[1]) * kHidden + koff;
                    *reinterpret_cast<uint4*>(a0) = __ldg(reinterpret_cast<const uint4*>(p1));
                    *reinterpret_cast<uint4*>(a1) = __ldg(reinterpret_cast<const uint4*>(p1 + 8));
                    acc1 = fmaf(h2_dot16_f32(a0, a1, d0, d1), sc, acc1);

                    if (m_cnt > 2) {
                        const half* p2 = x_staged + static_cast<std::int64_t>(s_toks[2]) * kHidden + koff;
                        *reinterpret_cast<uint4*>(a0) = __ldg(reinterpret_cast<const uint4*>(p2));
                        *reinterpret_cast<uint4*>(a1) = __ldg(reinterpret_cast<const uint4*>(p2 + 8));
                        acc2 = fmaf(h2_dot16_f32(a0, a1, d0, d1), sc, acc2);
                    }
                    if (m_cnt > 3) {
                        const half* p3 = x_staged + static_cast<std::int64_t>(s_toks[3]) * kHidden + koff;
                        *reinterpret_cast<uint4*>(a0) = __ldg(reinterpret_cast<const uint4*>(p3));
                        *reinterpret_cast<uint4*>(a1) = __ldg(reinterpret_cast<const uint4*>(p3 + 8));
                        acc3 = fmaf(h2_dot16_f32(a0, a1, d0, d1), sc, acc3);
                    }
                };

                dot16_all(gw.x, gw.y, k0,      decode_e4m3_f32(gs & 0xFFu), ag0, ag1, ag2, ag3);
                dot16_all(gw.z, gw.w, k0 + 16, decode_e4m3_f32(gs >> 8),    ag0, ag1, ag2, ag3);
                dot16_all(uw.x, uw.y, k0,      decode_e4m3_f32(us & 0xFFu), au0, au1, au2, au3);
                dot16_all(uw.z, uw.w, k0 + 16, decode_e4m3_f32(us >> 8),    au0, au1, au2, au3);
            }

            auto finish_m = [&](int m, float ag, float au) {
                s_red[0][p_loc][k_lane] = ag;
                s_red[1][p_loc][k_lane] = au;
                __syncthreads();
                if (tid < 16) {
                    float gsum = 0.0f, usum = 0.0f;
#pragma unroll
                    for (int k = 0; k < 18; ++k) {
                        gsum += s_red[0][tid][k];
                        usum += s_red[1][tid][k];
                    }
                    const float y   = softcapped_swiglu(gsum * outer, usum * outer, gate_cap, up_cap) * s_w[m];
                    const int jg    = blockIdx.x * 16 + tid;
                    const int sl    = jg / kInterPerSlab;
                    const int jl    = jg - sl * kInterPerSlab;
                    const int blk8  = jl & ~7;
                    const int p8    = jl & 7;
                    const int dst_j = sl * kInterPerSlab + blk8 + ((p8 & 3) * 2 + (p8 >> 2));
                    inter_out[static_cast<std::int64_t>(s_pairs[m]) * inter_local + dst_j] = __float2half_rn(y);
                }
                __syncthreads();
            };
            finish_m(0, ag0, au0);
            finish_m(1, ag1, au1);
            if (m_cnt > 2) { finish_m(2, ag2, au2); }
            if (m_cnt > 3) { finish_m(3, ag3, au3); }
        }
        if (m_base + 4 < end) { __syncthreads(); }
    }
}

// 2 rows/thread (32 rows/CTA, grid.x = num_slabs * 72 -> 2x fewer CTAs, 2x MLP, <=52 regs, 0 SPILL!)
__global__ __launch_bounds__(288, 3) void moe_nvfp4_grouped_w2_coalesced_kernel(
    const half* __restrict__ inter_staged, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, int num_slabs, float* __restrict__ accum_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    __shared__ half2 s_in[4][36][8];
    __shared__ float s_red[2][16][18];
    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];

    const int tid           = threadIdx.x; // 0..287
    const int r_loc         = tid / 18;
    const int k_lane        = tid - r_loc * 18;
    const int blocks_per_sl = kHidden / 32; // 72
    const int slab          = static_cast<int>(blockIdx.x) / blocks_per_sl;
    const int bx            = static_cast<int>(blockIdx.x) - slab * blocks_per_sl;
    const int inter_local   = num_slabs * kInterPerSlab;

    // Load BOTH rows concurrently ONCE before token loop -> 2x HBM2 MLP, only 10 regs!
    const float outer          = __ldg(w2_amax + eid) / __ldg(w2_denom);
    const std::int64_t ridx0   = (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + (bx * 32 + r_loc);
    const std::int64_t ridx1   = ridx0 + 16;
    const uint4 ww0            = ld_cg_u4(w2_codes + ridx0 * 288 + k_lane * 16);
    const uint4 ww1            = ld_cg_u4(w2_codes + ridx1 * 288 + k_lane * 16);
    const std::uint16_t ws0    = __ldg(reinterpret_cast<const std::uint16_t*>(w2_scales + ridx0 * 36 + k_lane * 2));
    const std::uint16_t ws1    = __ldg(reinterpret_cast<const std::uint16_t*>(w2_scales + ridx1 * 36 + k_lane * 2));
    const float s0_lo          = decode_e4m3_f32(ws0 & 0xFFu) * outer;
    const float s0_hi          = decode_e4m3_f32(ws0 >> 8) * outer;
    const float s1_lo          = decode_e4m3_f32(ws1 & 0xFFu) * outer;
    const float s1_hi          = decode_e4m3_f32(ws1 >> 8) * outer;

    const half2 rebias = __float2half2_rn(16384.0f);
    const int g0       = k_lane * 2;
    const int g1       = g0 + 1;

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
        }
        __syncthreads();

        if (tid < m_cnt * 36) {
            const int m     = tid / 36;
            const int g     = tid - m * 36;
            const half* src = inter_staged + static_cast<std::int64_t>(s_pairs[m]) * inter_local +
                              slab * kInterPerSlab + g * 16;
            *reinterpret_cast<uint4*>(&s_in[m][g][0]) = __ldg(reinterpret_cast<const uint4*>(src));
            *reinterpret_cast<uint4*>(&s_in[m][g][4]) = __ldg(reinterpret_cast<const uint4*>(src + 8));
        }
        __syncthreads();

        for (int m = 0; m < m_cnt; ++m) {
            half2 a0[4], a1[4], d0[4], d1[4];
            *reinterpret_cast<uint4*>(a0) = *reinterpret_cast<const uint4*>(&s_in[m][g0][0]);
            *reinterpret_cast<uint4*>(a1) = *reinterpret_cast<const uint4*>(&s_in[m][g0][4]);
            nvfp4_decode8_h2(ww0.x, rebias, d0);
            nvfp4_decode8_h2(ww0.y, rebias, d1);
            float acc0 = h2_dot16_f32(a0, a1, d0, d1) * s0_lo;
            nvfp4_decode8_h2(ww1.x, rebias, d0);
            nvfp4_decode8_h2(ww1.y, rebias, d1);
            float acc1 = h2_dot16_f32(a0, a1, d0, d1) * s1_lo;

            *reinterpret_cast<uint4*>(a0) = *reinterpret_cast<const uint4*>(&s_in[m][g1][0]);
            *reinterpret_cast<uint4*>(a1) = *reinterpret_cast<const uint4*>(&s_in[m][g1][4]);
            nvfp4_decode8_h2(ww0.z, rebias, d0);
            nvfp4_decode8_h2(ww0.w, rebias, d1);
            acc0 = fmaf(h2_dot16_f32(a0, a1, d0, d1), s0_hi, acc0);
            nvfp4_decode8_h2(ww1.z, rebias, d0);
            nvfp4_decode8_h2(ww1.w, rebias, d1);
            acc1 = fmaf(h2_dot16_f32(a0, a1, d0, d1), s1_hi, acc1);

            s_red[0][r_loc][k_lane] = acc0;
            s_red[1][r_loc][k_lane] = acc1;
            __syncthreads();
            if (tid < 32) {
                const int rs = tid >> 4;
                const int rl = tid & 15;
                float sum    = 0.0f;
#pragma unroll
                for (int k = 0; k < 18; ++k) { sum += s_red[rs][rl][k]; }
                atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[m]) * kHidden +
                              bx * 32 + rs * 16 + rl,
                          sum);
            }
            if (m + 1 < m_cnt) { __syncthreads(); }
        }
        if (m_base + 4 < end) { __syncthreads(); }
    }
}

// Coalesced-row grouped MXFP8 for T = 2..32 (ZERO-SPILL w13 + 4-rows/thread w2!)
__global__ __launch_bounds__(288, 2) void moe_mxfp8_grouped_w13_coalesced_kernel(
    const half* __restrict__ x_fp16, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, const float* __restrict__ topk_weights,
    int num_slabs, float gate_cap, float up_cap, half* __restrict__ inter_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];
    __shared__ float s_w[4];
    __shared__ float s_red[2][16][18];

    const int tid         = threadIdx.x;
    const int p_loc       = tid / 18;
    const int k_lane      = tid - p_loc * 18;
    const int inter_local = num_slabs * kInterPerSlab;
    const int j_global    = blockIdx.x * 16 + p_loc;
    const int slab        = j_global / kInterPerSlab;
    const int j_local     = j_global - slab * kInterPerSlab;

    const std::int64_t row_base =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + 2 * j_local;
    const std::uint8_t* gc_ptr = w13_codes + row_base * 2304;
    const std::uint8_t* uc_ptr = gc_ptr + 2304;
    const std::uint8_t* gs_ptr = w13_scales + row_base * 72;
    const std::uint8_t* us_ptr = gs_ptr + 72;

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
            s_w[tid]       = topk_weights[pair];
        }
        __syncthreads();

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

#pragma unroll 1
        for (int it = 0; it < 4; ++it) {
            const int g    = it * 18 + k_lane;
            const float gs = decode_ue8m0_f32(__ldg(gs_ptr + g)) * 256.0f;
            const float us = decode_ue8m0_f32(__ldg(us_ptr + g)) * 256.0f;

            auto run_half16 = [&](const std::uint8_t* cptr, int h_off, float sc,
                                  float& acc0, float& acc1, float& acc2, float& acc3) {
                const uint4 v = ld_cg_u4(cptr + g * 32 + h_off);
                half2 d[8];
                fp8_decode4_h2_div256(v.x, d[0], d[1]);
                fp8_decode4_h2_div256(v.y, d[2], d[3]);
                fp8_decode4_h2_div256(v.z, d[4], d[5]);
                fp8_decode4_h2_div256(v.w, d[6], d[7]);
                auto dot16 = [&](int tok_idx, float& acc) {
                    const half* xm = x_fp16 + static_cast<std::int64_t>(tok_idx) * kHidden + g * 32 + h_off;
                    const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(xm));
                    const uint4 x1 = __ldg(reinterpret_cast<const uint4*>(xm + 8));
                    const half2* h0 = reinterpret_cast<const half2*>(&x0);
                    const half2* h1 = reinterpret_cast<const half2*>(&x1);
                    half2 p0 = __hmul2(h0[0], d[0]);
                    half2 p1 = __hmul2(h0[1], d[1]);
                    p0       = __hfma2(h0[2], d[2], p0);
                    p1       = __hfma2(h0[3], d[3], p1);
                    p0       = __hfma2(h1[0], d[4], p0);
                    p1       = __hfma2(h1[1], d[5], p1);
                    p0       = __hfma2(h1[2], d[6], p0);
                    p1       = __hfma2(h1[3], d[7], p1);
                    const float2 pf = __half22float2(__hadd2(p0, p1));
                    acc             = fmaf(pf.x + pf.y, sc, acc);
                };
                dot16(s_toks[0], acc0);
                if (m_cnt > 1) { dot16(s_toks[1], acc1); }
                if (m_cnt > 2) { dot16(s_toks[2], acc2); }
                if (m_cnt > 3) { dot16(s_toks[3], acc3); }
            };
            run_half16(gc_ptr, 0,  gs, ag0, ag1, ag2, ag3);
            run_half16(gc_ptr, 16, gs, ag0, ag1, ag2, ag3);
            run_half16(uc_ptr, 0,  us, au0, au1, au2, au3);
            run_half16(uc_ptr, 16, us, au0, au1, au2, au3);
        }

        auto finish_m = [&](int m, float ag, float au) {
            s_red[0][p_loc][k_lane] = ag;
            s_red[1][p_loc][k_lane] = au;
            __syncthreads();
            if (tid < 16) {
                float gsum = 0.0f, usum = 0.0f;
#pragma unroll
                for (int k = 0; k < 18; ++k) {
                    gsum += s_red[0][tid][k];
                    usum += s_red[1][tid][k];
                }
                const float y = softcapped_swiglu(gsum, usum, gate_cap, up_cap) * s_w[m];
                inter_out[static_cast<std::int64_t>(s_pairs[m]) * inter_local + blockIdx.x * 16 + tid] =
                    __float2half_rn(y);
            }
            __syncthreads();
        };
        finish_m(0, ag0, au0);
        if (m_cnt > 1) { finish_m(1, ag1, au1); }
        if (m_cnt > 2) { finish_m(2, ag2, au2); }
        if (m_cnt > 3) { finish_m(3, ag3, au3); }
    }
}

__global__ __launch_bounds__(288, 3) void moe_mxfp8_grouped_w2_coalesced_kernel(
    const half* __restrict__ inter_fp16, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, int num_slabs, float* __restrict__ accum_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    __shared__ half2 s_in[4][18][16];
    __shared__ float s_red[16][18];
    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];

    const int tid           = threadIdx.x;
    const int r_loc         = tid / 18;
    const int k_lane        = tid - r_loc * 18;
    const int blocks_per_sl = kHidden / 64; // 36
    const int slab          = static_cast<int>(blockIdx.x) / blocks_per_sl;
    const int bx            = static_cast<int>(blockIdx.x) - slab * blocks_per_sl;
    const int inter_local   = num_slabs * kInterPerSlab;

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
        }
        __syncthreads();

        if (tid < m_cnt * 72) {
            const int g_flat = tid >> 2;
            const int q      = tid & 3;
            const int m      = g_flat / 18;
            const int g      = g_flat - m * 18;
            const half* src  = inter_fp16 + static_cast<std::int64_t>(s_pairs[m]) * inter_local +
                               slab * kInterPerSlab + g * 32 + q * 8;
            *reinterpret_cast<uint4*>(&s_in[m][g][q * 4]) =
                __ldg(reinterpret_cast<const uint4*>(src));
        }
        __syncthreads();

#pragma unroll
        for (int rs = 0; rs < 4; ++rs) {
            const int row = bx * 64 + rs * 16 + r_loc;
            const std::int64_t row_idx =
                (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + row;
            const float ws  = decode_ue8m0_f32(__ldg(w2_scales + row_idx * 18 + k_lane)) * 256.0f;
            const uint4 wv0 = ld_cg_u4(w2_codes + row_idx * 576 + k_lane * 32 + 0);
            const uint4 wv1 = ld_cg_u4(w2_codes + row_idx * 576 + k_lane * 32 + 16);
            const std::uint32_t ww[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
            half2 wd[16];
#pragma unroll
            for (int w = 0; w < 8; ++w) { fp8_decode4_h2_div256(ww[w], wd[2 * w], wd[2 * w + 1]); }

            for (int m = 0; m < m_cnt; ++m) {
                half2 p0 = __half2half2(__ushort_as_half(0));
                half2 p1 = __half2half2(__ushort_as_half(0));
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    p0 = __hfma2(s_in[m][k_lane][2 * i], wd[2 * i], p0);
                    p1 = __hfma2(s_in[m][k_lane][2 * i + 1], wd[2 * i + 1], p1);
                }
                const float2 pf      = __half22float2(__hadd2(p0, p1));
                s_red[r_loc][k_lane] = (pf.x + pf.y) * ws;
                __syncthreads();
                if (tid < 16) {
                    float sum = 0.0f;
#pragma unroll
                    for (int k = 0; k < 18; ++k) { sum += s_red[tid][k]; }
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[m]) * kHidden +
                                  bx * 64 + rs * 16 + tid,
                              sum);
                }
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// 6. Prefill (T > 32) Grouped Tensor Core QPN Kernels (high tokens/expert -> Tensor Core wins!)
// ============================================================================

template <int SplitK>
__global__ __launch_bounds__(SplitK * 32, (32 / SplitK) > 0 ? (32 / SplitK) : 1)
void moe_nvfp4_grouped_w13_qpn_kernel(
    const half* __restrict__ x_staged, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, const float* __restrict__ topk_weights,
    int num_slabs, float gate_cap, float up_cap, half* __restrict__ inter_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    const int tiles_per_slab = kW13RowsSlab / 32;
    const int slab           = static_cast<int>(blockIdx.x) / tiles_per_slab;
    const int tile_x         = static_cast<int>(blockIdx.x) % tiles_per_slab;
    const int inter_local    = num_slabs * kInterPerSlab;

    __shared__ float cs[SplitK][8 * 32];
    __shared__ int s_toks[8];
    __shared__ int s_pairs[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col  = tile_x * 32 + qp * 8 + r;

    const std::int64_t row_idx =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + col;
    const std::uint8_t* crow = w13_codes + row_idx * 1152;
    const std::uint8_t* srow = w13_scales + row_idx * 144;

    constexpr int kChunks = 36;
    const int c_beg       = warp * (kChunks / SplitK) + min(warp, kChunks % SplitK);
    const int c_end       = c_beg + (kChunks / SplitK) + (warp < (kChunks % SplitK));
    const half2 rebias    = __float2half2_rn(16384.0f);
    const float outer     = (__ldg(w13_amax + eid) / __ldg(w13_denom)) * 256.0f;

    for (int m_base = start; m_base < end; m_base += 8) {
        const int m_cnt = min(8, end - m_base);
        if (threadIdx.x < 8) {
            if (static_cast<int>(threadIdx.x) < m_cnt) {
                const int pair       = sorted_pairs[m_base + threadIdx.x];
                s_pairs[threadIdx.x] = pair;
                s_toks[threadIdx.x]  = pair / kTopK;
            } else {
                s_pairs[threadIdx.x] = -1;
                s_toks[threadIdx.x]  = -1;
            }
        }
        __syncthreads();

        const int tok_r = s_toks[r];
        const half* x_r = (tok_r >= 0) ? (x_staged + static_cast<std::int64_t>(tok_r) * kHidden) : nullptr;
        float c[2][8]   = {};

        for (int ch = c_beg; ch < c_end; ++ch) {
            const std::uint32_t sc4 = __ldg(reinterpret_cast<const std::uint32_t*>(srow + ch * 4));
            const uint4 cw0         = ld_cg_u4(crow + ch * 32);
            const uint4 cw1         = ld_cg_u4(crow + ch * 32 + 16);
            const std::uint32_t cwords[8] = {cw0.x, cw0.y, cw0.z, cw0.w,
                                             cw1.x, cw1.y, cw1.z, cw1.w};
#pragma unroll
            for (int grp = 0; grp < 4; ++grp) {
                const half2 gs = decode_e4m3_h2_div256(static_cast<std::uint8_t>((sc4 >> (grp * 8)) & 0xFFu));
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int widx = grp * 2 + sub;
                    half2 b4[4];
                    nvfp4_decode8_h2(cwords[widx], rebias, b4);
#pragma unroll
                    for (int j = 0; j < 4; ++j) { b4[j] = __hmul2(b4[j], gs); }
                    uint4 a_raw{0, 0, 0, 0};
                    if (x_r != nullptr) {
                        a_raw = __ldg(reinterpret_cast<const uint4*>(x_r + ch * 64 + widx * 8));
                    }
                    const auto* A = reinterpret_cast<const unsigned*>(&a_raw);
                    const auto* B = reinterpret_cast<const unsigned*>(b4);
                    volta_mma_qp_n(c[0], A[0], A[1], B[0], B[1]);
                    volta_mma_qp_n(c[1], A[2], A[3], B[2], B[3]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][row * 32 + qp * 8 + cl] = c[0][i] + c[1][i];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < 8 * 16; e += SplitK * 32) {
            const int row = e >> 4;
            const int p   = e & 15;
            if (row < m_cnt) {
                float gv = 0.0f, uv = 0.0f;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    gv += cs[s][row * 32 + 2 * p];
                    uv += cs[s][row * 32 + 2 * p + 1];
                }
                const int pair  = s_pairs[row];
                const float y   = softcapped_swiglu(gv * outer, uv * outer, gate_cap, up_cap) *
                                  topk_weights[pair];
                const int j_loc = tile_x * 16 + p;
                const int blk8  = j_loc & ~7;
                const int p8    = j_loc & 7;
                const int dst_j = slab * kInterPerSlab + blk8 + ((p8 & 3) * 2 + (p8 >> 2));
                inter_out[static_cast<std::int64_t>(pair) * inter_local + dst_j] = __float2half_rn(y);
            }
        }
        __syncthreads();
    }
}

template <int SplitK>
__global__ __launch_bounds__(SplitK * 32, (32 / SplitK) > 0 ? (32 / SplitK) : 1)
void moe_nvfp4_grouped_w2_qpn_kernel(
    const half* __restrict__ inter_staged, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, int num_slabs, float* __restrict__ accum_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    const int tiles_per_slab = kHidden / 32;
    const int slab           = static_cast<int>(blockIdx.x) / tiles_per_slab;
    const int tile_x         = static_cast<int>(blockIdx.x) % tiles_per_slab;
    const int inter_local    = num_slabs * kInterPerSlab;

    __shared__ float cs[SplitK][8 * 32];
    __shared__ int s_pairs[8];
    __shared__ int s_toks[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col  = tile_x * 32 + qp * 8 + r;

    const std::int64_t row_idx =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + col;
    const std::uint8_t* crow = w2_codes + row_idx * 288;
    const std::uint8_t* srow = w2_scales + row_idx * 36;

    constexpr int kChunks = 9;
    const int c_beg       = warp * (kChunks / SplitK) + min(warp, kChunks % SplitK);
    const int c_end       = c_beg + (kChunks / SplitK) + (warp < (kChunks % SplitK));
    const half2 rebias    = __float2half2_rn(16384.0f);
    const float outer     = (__ldg(w2_amax + eid) / __ldg(w2_denom)) * 256.0f;

    for (int m_base = start; m_base < end; m_base += 8) {
        const int m_cnt = min(8, end - m_base);
        if (threadIdx.x < 8) {
            if (static_cast<int>(threadIdx.x) < m_cnt) {
                const int pair       = sorted_pairs[m_base + threadIdx.x];
                s_pairs[threadIdx.x] = pair;
                s_toks[threadIdx.x]  = pair / kTopK;
            } else {
                s_pairs[threadIdx.x] = -1;
                s_toks[threadIdx.x]  = -1;
            }
        }
        __syncthreads();

        const int pair_r = s_pairs[r];
        const half* x_r  = (pair_r >= 0) ? (inter_staged + static_cast<std::int64_t>(pair_r) * inter_local +
                                            slab * kInterPerSlab)
                                         : nullptr;
        float c[2][8]    = {};

        for (int ch = c_beg; ch < c_end; ++ch) {
            const std::uint32_t sc4 = __ldg(reinterpret_cast<const std::uint32_t*>(srow + ch * 4));
            const uint4 cw0         = ld_cg_u4(crow + ch * 32);
            const uint4 cw1         = ld_cg_u4(crow + ch * 32 + 16);
            const std::uint32_t cwords[8] = {cw0.x, cw0.y, cw0.z, cw0.w,
                                             cw1.x, cw1.y, cw1.z, cw1.w};
#pragma unroll
            for (int grp = 0; grp < 4; ++grp) {
                const half2 gs = decode_e4m3_h2_div256(static_cast<std::uint8_t>((sc4 >> (grp * 8)) & 0xFFu));
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int widx = grp * 2 + sub;
                    half2 b4[4];
                    nvfp4_decode8_h2(cwords[widx], rebias, b4);
#pragma unroll
                    for (int j = 0; j < 4; ++j) { b4[j] = __hmul2(b4[j], gs); }
                    uint4 a_raw{0, 0, 0, 0};
                    if (x_r != nullptr) {
                        a_raw = __ldg(reinterpret_cast<const uint4*>(x_r + ch * 64 + widx * 8));
                    }
                    const auto* A = reinterpret_cast<const unsigned*>(&a_raw);
                    const auto* B = reinterpret_cast<const unsigned*>(b4);
                    volta_mma_qp_n(c[0], A[0], A[1], B[0], B[1]);
                    volta_mma_qp_n(c[1], A[2], A[3], B[2], B[3]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][row * 32 + qp * 8 + cl] = c[0][i] + c[1][i];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < 8 * 32; e += SplitK * 32) {
            const int row  = e >> 5;
            const int cl   = e & 31;
            const int gcol = tile_x * 32 + cl;
            if (row < m_cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) { sum += cs[s][e]; }
                atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[row]) * kHidden + gcol,
                          sum * outer);
            }
        }
        __syncthreads();
    }
}

template <int SplitK>
__global__ __launch_bounds__(SplitK * 32, (32 / SplitK) > 0 ? (32 / SplitK) : 1)
void moe_mxfp8_grouped_w13_qpn_kernel(
    const half* __restrict__ x_fp16, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, const float* __restrict__ topk_weights,
    int num_slabs, float gate_cap, float up_cap, half* __restrict__ inter_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    const int tiles_per_slab = kW13RowsSlab / 32;
    const int slab           = static_cast<int>(blockIdx.x) / tiles_per_slab;
    const int tile_x         = static_cast<int>(blockIdx.x) % tiles_per_slab;
    const int inter_local    = num_slabs * kInterPerSlab;

    __shared__ float cs[SplitK][8 * 32];
    __shared__ int s_toks[8];
    __shared__ int s_pairs[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col  = tile_x * 32 + qp * 8 + r;

    const std::int64_t row_idx =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + col;
    const std::uint8_t* crow = w13_codes + row_idx * 2304;
    const std::uint8_t* srow = w13_scales + row_idx * 72;

    constexpr int kGroups = 72;
    const int g_beg       = warp * (kGroups / SplitK) + min(warp, kGroups % SplitK);
    const int g_end       = g_beg + (kGroups / SplitK) + (warp < (kGroups % SplitK));

    for (int m_base = start; m_base < end; m_base += 8) {
        const int m_cnt = min(8, end - m_base);
        if (threadIdx.x < 8) {
            if (static_cast<int>(threadIdx.x) < m_cnt) {
                const int pair       = sorted_pairs[m_base + threadIdx.x];
                s_pairs[threadIdx.x] = pair;
                s_toks[threadIdx.x]  = pair / kTopK;
            } else {
                s_pairs[threadIdx.x] = -1;
                s_toks[threadIdx.x]  = -1;
            }
        }
        __syncthreads();

        const int tok_r = s_toks[r];
        const half* x_r = (tok_r >= 0) ? (x_fp16 + static_cast<std::int64_t>(tok_r) * kHidden) : nullptr;
        float c[2][8]   = {};

        for (int grp = g_beg; grp < g_end; ++grp) {
            const half2 gs  = __float2half2_rn(decode_ue8m0_f32(__ldg(srow + grp)));
            const uint4 cw0 = ld_cg_u4(crow + grp * 32);
            const uint4 cw1 = ld_cg_u4(crow + grp * 32 + 16);
            const std::uint32_t words[8] = {cw0.x, cw0.y, cw0.z, cw0.w,
                                            cw1.x, cw1.y, cw1.z, cw1.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                half2 b4[4];
                fp8_decode4_h2_div256(words[2 * u], b4[0], b4[1]);
                fp8_decode4_h2_div256(words[2 * u + 1], b4[2], b4[3]);
#pragma unroll
                for (int j = 0; j < 4; ++j) { b4[j] = __hmul2(b4[j], gs); }
                uint4 a_raw{0, 0, 0, 0};
                if (x_r != nullptr) {
                    a_raw = __ldg(reinterpret_cast<const uint4*>(x_r + grp * 32 + u * 8));
                }
                const auto* A = reinterpret_cast<const unsigned*>(&a_raw);
                const auto* B = reinterpret_cast<const unsigned*>(b4);
                volta_mma_qp_n(c[0], A[0], A[1], B[0], B[1]);
                volta_mma_qp_n(c[1], A[2], A[3], B[2], B[3]);
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][row * 32 + qp * 8 + cl] = c[0][i] + c[1][i];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < 8 * 16; e += SplitK * 32) {
            const int row = e >> 4;
            const int p   = e & 15;
            if (row < m_cnt) {
                float gv = 0.0f, uv = 0.0f;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    gv += cs[s][row * 32 + 2 * p];
                    uv += cs[s][row * 32 + 2 * p + 1];
                }
                const int pair     = s_pairs[row];
                const float y      = softcapped_swiglu(gv * 256.0f, uv * 256.0f, gate_cap, up_cap) *
                                     topk_weights[pair];
                const int j_global = slab * kInterPerSlab + tile_x * 16 + p;
                inter_out[static_cast<std::int64_t>(pair) * inter_local + j_global] = __float2half_rn(y);
            }
        }
        __syncthreads();
    }
}

template <int SplitK>
__global__ __launch_bounds__(SplitK * 32, (32 / SplitK) > 0 ? (32 / SplitK) : 1)
void moe_mxfp8_grouped_w2_qpn_kernel(
    const half* __restrict__ inter_fp16, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, int num_slabs, float* __restrict__ accum_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    const int tiles_per_slab = kHidden / 32;
    const int slab           = static_cast<int>(blockIdx.x) / tiles_per_slab;
    const int tile_x         = static_cast<int>(blockIdx.x) % tiles_per_slab;
    const int inter_local    = num_slabs * kInterPerSlab;

    __shared__ float cs[SplitK][8 * 32];
    __shared__ int s_pairs[8];
    __shared__ int s_toks[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col  = tile_x * 32 + qp * 8 + r;

    const std::int64_t row_idx =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + col;
    const std::uint8_t* crow = w2_codes + row_idx * 576;
    const std::uint8_t* srow = w2_scales + row_idx * 18;

    constexpr int kGroups = 18;
    const int g_beg       = warp * (kGroups / SplitK) + min(warp, kGroups % SplitK);
    const int g_end       = g_beg + (kGroups / SplitK) + (warp < (kGroups % SplitK));

    for (int m_base = start; m_base < end; m_base += 8) {
        const int m_cnt = min(8, end - m_base);
        if (threadIdx.x < 8) {
            if (static_cast<int>(threadIdx.x) < m_cnt) {
                const int pair       = sorted_pairs[m_base + threadIdx.x];
                s_pairs[threadIdx.x] = pair;
                s_toks[threadIdx.x]  = pair / kTopK;
            } else {
                s_pairs[threadIdx.x] = -1;
                s_toks[threadIdx.x]  = -1;
            }
        }
        __syncthreads();

        const int pair_r = s_pairs[r];
        const half* x_r  = (pair_r >= 0) ? (inter_fp16 + static_cast<std::int64_t>(pair_r) * inter_local +
                                            slab * kInterPerSlab)
                                         : nullptr;
        float c[2][8]    = {};

        for (int grp = g_beg; grp < g_end; ++grp) {
            const half2 gs  = __float2half2_rn(decode_ue8m0_f32(__ldg(srow + grp)));
            const uint4 cw0 = ld_cg_u4(crow + grp * 32);
            const uint4 cw1 = ld_cg_u4(crow + grp * 32 + 16);
            const std::uint32_t words[8] = {cw0.x, cw0.y, cw0.z, cw0.w,
                                            cw1.x, cw1.y, cw1.z, cw1.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                half2 b4[4];
                fp8_decode4_h2_div256(words[2 * u], b4[0], b4[1]);
                fp8_decode4_h2_div256(words[2 * u + 1], b4[2], b4[3]);
#pragma unroll
                for (int j = 0; j < 4; ++j) { b4[j] = __hmul2(b4[j], gs); }
                uint4 a_raw{0, 0, 0, 0};
                if (x_r != nullptr) {
                    a_raw = __ldg(reinterpret_cast<const uint4*>(x_r + grp * 32 + u * 8));
                }
                const auto* A = reinterpret_cast<const unsigned*>(&a_raw);
                const auto* B = reinterpret_cast<const unsigned*>(b4);
                volta_mma_qp_n(c[0], A[0], A[1], B[0], B[1]);
                volta_mma_qp_n(c[1], A[2], A[3], B[2], B[3]);
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][row * 32 + qp * 8 + cl] = c[0][i] + c[1][i];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < 8 * 32; e += SplitK * 32) {
            const int row  = e >> 5;
            const int cl   = e & 31;
            const int gcol = tile_x * 32 + cl;
            if (row < m_cnt) {
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) { sum += cs[s][e]; }
                atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[row]) * kHidden + gcol,
                          sum * 256.0f);
            }
        }
        __syncthreads();
    }
}

__global__ void cast_f32_to_bf16_vec4_kernel(const float* __restrict__ src,
                                             __nv_bfloat16* __restrict__ dst, std::size_t n4) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n4) { return; }
    const float4 v = reinterpret_cast<const float4*>(src)[i];
    uint2 out;
    auto* b = reinterpret_cast<__nv_bfloat162*>(&out);
    b[0]    = __floats2bfloat162_rn(v.x, v.y);
    b[1]    = __floats2bfloat162_rn(v.z, v.w);
    reinterpret_cast<uint2*>(dst)[i] = out;
}

} // namespace

void inkling2_ensure_moe_scratch(int device, std::size_t max_tokens) {
    int prev = 0;
    CUDA_CHECK(cudaGetDevice(&prev));
    if (prev != device) { CUDA_CHECK(cudaSetDevice(device)); }
    (void)get_moe_x_scratch(device, max_tokens * kHidden);
    if (prev != device) { CUDA_CHECK(cudaSetDevice(prev)); }
}

void inkling2_moe_route(const Tensor& router_logits_f32, const Tensor& router_bias_f32,
                        float route_scale, Inkling2MoeScratch& scratch, std::int32_t tokens,
                        cudaStream_t stream) {
    if (tokens == 1) {
        moe_route_t1_warp_kernel<<<1, 32, 0, stream>>>(
            static_cast<const float*>(router_logits_f32.data),
            static_cast<const float*>(router_bias_f32.data), route_scale, scratch.topk_ids,
            scratch.topk_weights);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (tokens <= 64) {
        moe_route_and_group_small_kernel<<<1, kNumExperts, 0, stream>>>(
            static_cast<const float*>(router_logits_f32.data),
            static_cast<const float*>(router_bias_f32.data), route_scale, tokens, scratch.topk_ids,
            scratch.topk_weights, scratch.expert_counts, scratch.expert_offsets,
            scratch.sorted_pairs, scratch.active_experts, scratch.num_active);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    moe_route_topk_large_kernel<<<(tokens + 3) / 4, 128, 0, stream>>>(
        static_cast<const float*>(router_logits_f32.data),
        static_cast<const float*>(router_bias_f32.data), route_scale, scratch.topk_ids,
        scratch.topk_weights, tokens);
    CUDA_CHECK(cudaGetLastError());
    moe_group_assignments_large_kernel<<<1, kNumExperts, 0, stream>>>(
        scratch.topk_ids, tokens, scratch.expert_counts, scratch.expert_offsets,
        scratch.sorted_pairs, scratch.active_experts, scratch.num_active);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_moe_router_fused(const Tensor& x, const Tensor& router_weight,
                               const Tensor& router_bias_f32, float route_scale,
                               Inkling2MoeScratch& scratch, cudaStream_t stream) {
    const int k      = x.ne[0];
    const int tokens = x.ne[1];
    const int n      = router_weight.ne[1];
    if (scratch.router_done != nullptr && k == kHidden && n == kNumExperts &&
        x.dtype == DType::BF16 && router_weight.dtype == DType::BF16 &&
        tokens >= 1 && tokens <= 8) {
        const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
        const auto* wp = static_cast<const __nv_bfloat16*>(router_weight.data);
        const auto* bp = static_cast<const float*>(router_bias_f32.data);
        auto launch    = [&]<int ActiveT>() {
            constexpr int kWarps = (ActiveT <= 2) ? 4 : 8;
            moe_router_fused_kernel<ActiveT, kWarps><<<kNumExperts, kWarps * 32, 0, stream>>>(
                xp, wp, bp, route_scale, scratch.router_logits, scratch.topk_ids,
                scratch.topk_weights, scratch.expert_counts, scratch.expert_offsets,
                scratch.sorted_pairs, scratch.active_experts, scratch.num_active,
                scratch.router_done);
        };
        switch (tokens) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 3: launch.template operator()<3>(); break;
        case 4: launch.template operator()<4>(); break;
        case 5: launch.template operator()<5>(); break;
        case 6: launch.template operator()<6>(); break;
        case 7: launch.template operator()<7>(); break;
        case 8: launch.template operator()<8>(); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    Tensor rlog(scratch.router_logits, DType::FP32, {n, tokens});
    inkling2_bf16_linear(x, router_weight, rlog, stream);
    inkling2_moe_route(rlog, router_bias_f32, route_scale, scratch, tokens, stream);
}

void inkling2_moe_nvfp4_forward(const Tensor& x, const void* w13_codes, const void* w13_scales,
                                const float* w13_amax, const float* w13_denom,
                                const void* w2_codes, const void* w2_scales,
                                const float* w2_amax, const float* w2_denom,
                                std::int32_t num_slabs, float gate_cap, float up_cap,
                                Inkling2MoeScratch& scratch, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        const int inter_local = num_slabs * kInterPerSlab;
        const dim3 g1(inter_local / 16, kTopK);
        moe_nvfp4_t1_w13_kernel<<<g1, 288, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), w13_amax, w13_denom, scratch.topk_ids,
            scratch.topk_weights, num_slabs, gate_cap, up_cap,
            static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        moe_nvfp4_t1_w2_kernel<<<kHidden / 16, 288, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), w2_amax, w2_denom, scratch.topk_ids,
            num_slabs, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    half* x_staged = get_moe_x_scratch(dev, static_cast<std::size_t>(tokens) * kHidden);
    prestage_x_fp16_kernel<true><<<tokens, 288, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), x_staged, tokens);
    CUDA_CHECK(cudaGetLastError());

    const int max_active        = std::min(kNumExperts, tokens * kTopK);
    const std::size_t out_elems = static_cast<std::size_t>(tokens) * kHidden;

    if (tokens <= 32) {
        const dim3 g1(num_slabs * (kInterPerSlab / 16), max_active);
        moe_nvfp4_grouped_w13_coalesced_kernel<<<g1, 288, 0, stream>>>(
            x_staged, static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), w13_amax, w13_denom,
            scratch.active_experts, scratch.num_active, scratch.expert_offsets,
            scratch.sorted_pairs, scratch.topk_weights, num_slabs, gate_cap, up_cap,
            static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));

        const dim3 g2(num_slabs * (kHidden / 32), max_active);
        moe_nvfp4_grouped_w2_coalesced_kernel<<<g2, 288, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), w2_amax, w2_denom, scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, num_slabs,
            scratch.accum_f32);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const dim3 g1(num_slabs * (kW13RowsSlab / 32), max_active);
        moe_nvfp4_grouped_w13_qpn_kernel<4><<<g1, 128, 0, stream>>>(
            x_staged, static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), w13_amax, w13_denom,
            scratch.active_experts, scratch.num_active, scratch.expert_offsets,
            scratch.sorted_pairs, scratch.topk_weights, num_slabs, gate_cap, up_cap,
            static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));

        const dim3 g2(num_slabs * (kHidden / 32), max_active);
        moe_nvfp4_grouped_w2_qpn_kernel<3><<<g2, 96, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), w2_amax, w2_denom, scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, num_slabs,
            scratch.accum_f32);
        CUDA_CHECK(cudaGetLastError());
    }

    const std::size_t n4 = out_elems / 4;
    cast_f32_to_bf16_vec4_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
        scratch.accum_f32, static_cast<__nv_bfloat16*>(out.data), n4);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_moe_mxfp8_forward(const Tensor& x, const void* w13_codes, const void* w13_scales,
                                const void* w2_codes, const void* w2_scales,
                                std::int32_t num_slabs, float gate_cap, float up_cap,
                                Inkling2MoeScratch& scratch, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        const int inter_local = num_slabs * kInterPerSlab;
        const dim3 g1(inter_local / 16, kTopK);
        moe_mxfp8_t1_w13_kernel<<<g1, 288, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), scratch.topk_ids, scratch.topk_weights,
            num_slabs, gate_cap, up_cap, static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        moe_mxfp8_t1_w2_kernel<<<kHidden / 16, 288, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), scratch.topk_ids, num_slabs,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    half* x_fp16 = get_moe_x_scratch(dev, static_cast<std::size_t>(tokens) * kHidden);
    prestage_x_fp16_kernel<false><<<tokens, 288, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), x_fp16, tokens);
    CUDA_CHECK(cudaGetLastError());

    const int max_active        = std::min(kNumExperts, tokens * kTopK);
    const std::size_t out_elems = static_cast<std::size_t>(tokens) * kHidden;

    if (tokens <= 32) {
        const dim3 g1(num_slabs * (kInterPerSlab / 16), max_active);
        moe_mxfp8_grouped_w13_coalesced_kernel<<<g1, 288, 0, stream>>>(
            x_fp16, static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, scratch.topk_weights,
            num_slabs, gate_cap, up_cap, static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));

        const dim3 g2(num_slabs * (kHidden / 64), max_active);
        moe_mxfp8_grouped_w2_coalesced_kernel<<<g2, 288, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, num_slabs,
            scratch.accum_f32);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const dim3 g1(num_slabs * (kW13RowsSlab / 32), max_active);
        moe_mxfp8_grouped_w13_qpn_kernel<4><<<g1, 128, 0, stream>>>(
            x_fp16, static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, scratch.topk_weights,
            num_slabs, gate_cap, up_cap, static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));

        const dim3 g2(num_slabs * (kHidden / 32), max_active);
        moe_mxfp8_grouped_w2_qpn_kernel<3><<<g2, 96, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, num_slabs,
            scratch.accum_f32);
        CUDA_CHECK(cudaGetLastError());
    }

    const std::size_t n4 = out_elems / 4;
    cast_f32_to_bf16_vec4_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
        scratch.accum_f32, static_cast<__nv_bfloat16*>(out.data), n4);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
