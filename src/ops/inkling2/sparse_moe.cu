#include "ninfer/ops/inkling2.h"

#include "core/device.h"
#include "ops/common/volta_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
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

__device__ __forceinline__ float4 ld_cg_f4(const void* ptr) {
    float4 val;
    asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        : "l"(ptr));
    return val;
}

__device__ __forceinline__ float ld_cg_f32(const float* ptr) {
    float val;
    asm("ld.global.cg.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ std::uint16_t ld_cg_u16(const void* ptr) {
    std::uint16_t val;
    asm("ld.global.cg.u16 %0, [%1];\n" : "=h"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ void st_cg_f32(float* ptr, float v) {
    asm volatile("st.global.cg.f32 [%0], %1;\n" :: "l"(ptr), "f"(v) : "memory");
}

__device__ __forceinline__ void st_cg_u16(void* ptr, std::uint16_t v) {
    asm volatile("st.global.cg.u16 [%0], %1;\n" :: "l"(ptr), "h"(v) : "memory");
}

__device__ __forceinline__ void st_cg_u4(void* ptr, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
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

__device__ __forceinline__ uint4 pack_bf16x8(const float (&v)[8]) {
    uint4 out;
    auto* dst = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int j = 0; j < 4; ++j) { dst[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]); }
    return out;
}

__device__ __forceinline__ void grid_barrier_64(unsigned int* bar_word) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int delta = (blockIdx.x == 0) ? (0x80000001u - 64u) : 1u;
        unsigned int old, cur;
        asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;\n"
                     : "=r"(old)
                     : "l"(bar_word), "r"(delta)
                     : "memory");
        do {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];\n" : "=r"(cur) : "l"(bar_word) : "memory");
        } while (((cur ^ old) & 0x80000000u) == 0u);
    }
    __syncthreads();
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ float warp_reduce_64_f4(const float* __restrict__ arr64, int lane) {
    float v = 0.0f;
    if (lane < 16) {
        const float4 f = ld_cg_f4(arr64 + lane * 4);
        v              = (f.x + f.y) + (f.z + f.w);
    }
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    return v;
}

int get_moe_device_sm_count() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    static std::atomic<int> s_sm[16]{};
    if (dev >= 0 && dev < 16) {
        const int cached = s_sm[dev].load(std::memory_order_relaxed);
        if (cached > 0) { return cached; }
        int sms = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        s_sm[dev].store(sms, std::memory_order_relaxed);
        return sms;
    }
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return sms;
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

__device__ __forceinline__ float h2_dot16_u4_f32(const uint4& a0_u4, const uint4& a1_u4,
                                                 const half2 (&b0)[4], const half2 (&b1)[4]) {
    const auto* a0 = reinterpret_cast<const half2*>(&a0_u4);
    const auto* a1 = reinterpret_cast<const half2*>(&a1_u4);
    half2 p0       = __hmul2(a0[0], b0[0]);
    half2 p1       = __hmul2(a0[1], b0[1]);
    p0             = __hfma2(a0[2], b0[2], p0);
    p1             = __hfma2(a0[3], b0[3], p1);
    p0             = __hfma2(a1[0], b1[0], p0);
    p1             = __hfma2(a1[1], b1[1], p1);
    p0             = __hfma2(a1[2], b1[2], p0);
    p1             = __hfma2(a1[3], b1[3], p1);
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

    int my_chosen_id = 0;
    float my_chosen_w = 0.0f;
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
            if (sub == 0) {
                my_raw   = raw[0];
                score[0] = -std::numeric_limits<float>::infinity();
            } else if (sub == 1) {
                my_raw   = raw[1];
                score[1] = -std::numeric_limits<float>::infinity();
            } else if (sub == 2) {
                my_raw   = raw[2];
                score[2] = -std::numeric_limits<float>::infinity();
            } else {
                my_raw   = raw[3];
                score[3] = -std::numeric_limits<float>::infinity();
            }
        }
        const float best_raw = __shfl_sync(0xFFFFFFFFu, my_raw, best_id >> 2);
        if (lane == k) {
            my_chosen_id = best_id;
            my_chosen_w  = best_raw;
        }
        wsum += best_raw;
    }

    if (lane < kTopK) {
        const float inv   = route_scale / (wsum + 1e-20f);
        out_ids[lane]     = my_chosen_id;
        out_weights[lane] = my_chosen_w * inv;
        if (smem_ids_row != nullptr) { smem_ids_row[lane] = my_chosen_id; }
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

// ============================================================================
// 1b. 4-Warp (128t) Branchless Bitonic Top-8 of 128 Precomputed (score, raw)
//     Replaces 8-pass serial warp_argmax (cuts T==1 router tail by ~2 us/layer!)
// ============================================================================

__device__ __forceinline__ bool better_expert(float sa, int ia, float sb, int ib) {
    return (sa > sb) || (sa == sb && ia < ib);
}

__device__ __forceinline__ void cmp_exch_expert(float& s, int& id, int mask, bool want_better) {
    const float os        = __shfl_xor_sync(0xFFFFFFFFu, s, mask);
    const int oid         = __shfl_xor_sync(0xFFFFFFFFu, id, mask);
    const bool take_other = want_better ? better_expert(os, oid, s, id) : better_expert(s, id, os, oid);
    if (take_other) {
        s  = os;
        id = oid;
    }
}

__device__ __forceinline__ void bitonic_sort_8(float& s, int& id, bool desc) {
    const int lane = threadIdx.x & 31;
    const bool d2  = ((lane & 2) == 0) ? desc : !desc;
    cmp_exch_expert(s, id, 1, ((lane & 1) == 0) ? d2 : !d2);
    const bool d4 = ((lane & 4) == 0) ? desc : !desc;
    cmp_exch_expert(s, id, 2, ((lane & 2) == 0) ? d4 : !d4);
    cmp_exch_expert(s, id, 1, ((lane & 1) == 0) ? d4 : !d4);
    cmp_exch_expert(s, id, 4, ((lane & 4) == 0) ? desc : !desc);
    cmp_exch_expert(s, id, 2, ((lane & 2) == 0) ? desc : !desc);
    cmp_exch_expert(s, id, 1, ((lane & 1) == 0) ? desc : !desc);
}

__device__ __forceinline__ void bitonic_merge_8(float& s, int& id, bool desc) {
    const int lane = threadIdx.x & 31;
    cmp_exch_expert(s, id, 4, ((lane & 4) == 0) ? desc : !desc);
    cmp_exch_expert(s, id, 2, ((lane & 2) == 0) ? desc : !desc);
    cmp_exch_expert(s, id, 1, ((lane & 1) == 0) ? desc : !desc);
}

__device__ __forceinline__ void warp_bitonic_top8_of_32(float& s, int& id) {
    const int lane = threadIdx.x & 31;
    bitonic_sort_8(s, id, (lane & 8) == 0);
    cmp_exch_expert(s, id, 8, (lane & 8) == 0);
    bitonic_merge_8(s, id, (lane & 16) == 0);
    cmp_exch_expert(s, id, 16, (lane & 16) == 0);
    bitonic_merge_8(s, id, true);
}

// 4-warp (128 threads = 1 expert/thread) exact Top-8 of 128 precomputed (scores, raws).
// Caller MUST call with all threads in the CTA reaching __syncthreads()!
__device__ __forceinline__ void block_bitonic_top8_128(
    const float* __restrict__ scores, const float* __restrict__ raws, float route_scale,
    float* s_cand_s, int* s_cand_id, std::int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < 128) {
        float s = ld_cg_f32(scores + tid);
        int id  = tid;
        warp_bitonic_top8_of_32(s, id);
        if (lane < 8) {
            const int dst  = warp * 8 + ((warp & 1) ? (7 - lane) : lane);
            s_cand_s[dst]  = s;
            s_cand_id[dst] = id;
        }
    }
    __syncthreads();
    if (warp == 0) {
        float s = s_cand_s[lane];
        int id  = s_cand_id[lane];
        cmp_exch_expert(s, id, 8, (lane & 8) == 0);
        bitonic_merge_8(s, id, (lane & 16) == 0);
        cmp_exch_expert(s, id, 16, (lane & 16) == 0);
        bitonic_merge_8(s, id, true);

        const float rw = (lane < 8) ? ld_cg_f32(raws + id) : 0.0f;
        float wsum     = rw;
        wsum += __shfl_xor_sync(0xFFFFFFFFu, wsum, 1, 8);
        wsum += __shfl_xor_sync(0xFFFFFFFFu, wsum, 2, 8);
        wsum += __shfl_xor_sync(0xFFFFFFFFu, wsum, 4, 8);
        if (lane < 8) {
            out_ids[lane]     = id;
            out_weights[lane] = rw * (route_scale / (wsum + 1e-20f));
        }
    }
}

// T == 1 Ultra-Fast Bitonic Router (64 CTAs x 288t = 1 uint4/thread ZERO loop,
// precomputed sigmoid+bias across 64 SMs, single tid==0 fence, 4-warp Bitonic Top-8!)
template <int PrepMode>
__global__ __launch_bounds__(288, 2) void moe_router_t1_bitonic_288t_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ bias, float route_scale, float* __restrict__ router_logits,
    float* __restrict__ scores_buf, float* __restrict__ raws_buf,
    std::int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights,
    std::int32_t* __restrict__ done_counter, half* __restrict__ x_staged) {
    __shared__ float smem[2][9];
    __shared__ float s_cs[32];
    __shared__ int s_cid[32];
    __shared__ int s_is_last;

    const int row0  = blockIdx.x * 2;
    const int tid   = threadIdx.x; // 0..287
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const uint4* w0 = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row0) * kHidden);
    const uint4* w1 = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row0 + 1) * kHidden);
    const uint4 wv0 = ld_cg_u4(w0 + tid);
    const uint4 wv1 = ld_cg_u4(w1 + tid);
    const uint4 xv  = __ldg(reinterpret_cast<const uint4*>(x) + tid);

    float b0 = 0.0f, b1 = 0.0f;
    if (tid == 0) {
        b0 = __ldg(bias + row0);
        b1 = __ldg(bias + row0 + 1);
    }

    const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&wv0);
    const auto* wb1 = reinterpret_cast<const __nv_bfloat162*>(&wv1);
    const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&xv);
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f0 = __bfloat1622float2(wb0[j]);
        const float2 f1 = __bfloat1622float2(wb1[j]);
        const float2 xf = __bfloat1622float2(xb[j]);
        a0              = fmaf(f0.x, xf.x, fmaf(f0.y, xf.y, a0));
        a1              = fmaf(f1.x, xf.x, fmaf(f1.y, xf.y, a1));
    }

    if constexpr (PrepMode != 0) {
        if (blockIdx.x == 0) {
            alignas(16) half2 h[4];
            if constexpr (PrepMode == 1) {
                bf16x8_to_staged_h2(xv, h);
            } else {
                bf16x8_to_h2(xv, h);
            }
            reinterpret_cast<uint4*>(x_staged)[tid] = *reinterpret_cast<const uint4*>(h);
        }
    }

    const float ws0 = warp_sum(a0);
    const float ws1 = warp_sum(a1);
    if (lane == 0) {
        smem[0][warp] = ws0;
        smem[1][warp] = ws1;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = (((smem[0][0] + smem[0][1]) + (smem[0][2] + smem[0][3])) +
                          ((smem[0][4] + smem[0][5]) + (smem[0][6] + smem[0][7]))) +
                         smem[0][8];
        const float s1 = (((smem[1][0] + smem[1][1]) + (smem[1][2] + smem[1][3])) +
                          ((smem[1][4] + smem[1][5]) + (smem[1][6] + smem[1][7]))) +
                         smem[1][8];
        router_logits[row0 + 0] = s0;
        router_logits[row0 + 1] = s1;
        const float r0          = 1.0f / (1.0f + expf(-s0));
        const float r1          = 1.0f / (1.0f + expf(-s1));
        st_cg_f32(&raws_buf[row0 + 0], r0);
        st_cg_f32(&raws_buf[row0 + 1], r1);
        st_cg_f32(&scores_buf[row0 + 0], r0 + b0);
        st_cg_f32(&scores_buf[row0 + 1], r1 + b1);
        __threadfence();
        s_is_last = (atomicAdd(done_counter, 1) == 63) ? 1 : 0;
    }
    __syncthreads();
    if (!s_is_last) { return; }
    if (tid == 0) { done_counter[0] = 0; }
    asm volatile("" ::: "memory");
    block_bitonic_top8_128(scores_buf, raws_buf, route_scale, s_cs, s_cid, topk_ids, topk_weights);
}

// Fused Single-Wave 64-CTA x 288t Router GEMV (1 uint4/thread ZERO loop!) +
// Sigmoid + Bias Top-8 + Bitmask Single-Pass Grouping + Optional Fused x_staged & accum_f32 Zero!
// PrepMode: 0 = router only; 1 = also prestage x (NVFP4 staged) + zero accum_f32; 2 = MXFP8 + zero!
template <int ActiveT, int PrepMode>
__global__ __launch_bounds__(288, 2) void moe_router_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ bias, float route_scale, float* __restrict__ router_logits,
    std::int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights,
    std::int32_t* __restrict__ expert_counts, std::int32_t* __restrict__ expert_offsets,
    std::int32_t* __restrict__ sorted_pairs, std::int32_t* __restrict__ active_experts,
    std::int32_t* __restrict__ num_active, std::int32_t* __restrict__ done_counter,
    half* __restrict__ x_staged, float* __restrict__ accum_f32) {
    constexpr int kWarps = 9;
    __shared__ float smem[2][ActiveT][kWarps];
    __shared__ int s_is_last;

    const int row0 = blockIdx.x * 2; // 0, 2, ..., 126
    const int tid  = threadIdx.x;    // 0..287
    const int warp = tid >> 5;       // 0..8
    const int lane = tid & 31;

    // Piggyback x_staged prestage & accum_f32 zeroing across all 64*288 = 18432 threads!
    if constexpr (PrepMode != 0 && ActiveT >= 2) {
        const int g_tid = blockIdx.x * 288 + tid;
        if (g_tid < ActiveT * (kHidden / 8)) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(x) + g_tid);
            alignas(16) half2 h[4];
            if constexpr (PrepMode == 1) {
                bf16x8_to_staged_h2(raw, h);
            } else {
                bf16x8_to_h2(raw, h);
            }
            reinterpret_cast<uint4*>(x_staged)[g_tid] = *reinterpret_cast<const uint4*>(h);
        }
        if (g_tid < ActiveT * (kHidden / 4)) {
            reinterpret_cast<float4*>(accum_f32)[g_tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const __nv_bfloat16* wrow0 = w + static_cast<std::int64_t>(row0) * kHidden;
    const __nv_bfloat16* wrow1 = wrow0 + kHidden;
    const uint4 wv0            = ld_cg_u4(reinterpret_cast<const uint4*>(wrow0) + tid);
    const uint4 wv1            = ld_cg_u4(reinterpret_cast<const uint4*>(wrow1) + tid);
    const auto* wb0            = reinterpret_cast<const __nv_bfloat162*>(&wv0);
    const auto* wb1            = reinterpret_cast<const __nv_bfloat162*>(&wv1);
    float wf0[8], wf1[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f0 = __bfloat1622float2(wb0[j]);
        const float2 f1 = __bfloat1622float2(wb1[j]);
        wf0[2 * j]      = f0.x;
        wf0[2 * j + 1]  = f0.y;
        wf1[2 * j]      = f1.x;
        wf1[2 * j + 1]  = f1.y;
    }

    float acc0[ActiveT] = {};
    float acc1[ActiveT] = {};
#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                                   x + static_cast<std::int64_t>(ti) * kHidden) + tid);
        const auto* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        float a00 = 0.0f, a01 = 0.0f, a10 = 0.0f, a11 = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 xf = __bfloat1622float2(xb[j]);
            a00             = fmaf(wf0[2 * j], xf.x, a00);
            a01             = fmaf(wf0[2 * j + 1], xf.y, a01);
            a10             = fmaf(wf1[2 * j], xf.x, a10);
            a11             = fmaf(wf1[2 * j + 1], xf.y, a11);
        }
        acc0[ti] = a00 + a01;
        acc1[ti] = a10 + a11;
    }

#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        const float ws0 = warp_sum(acc0[ti]);
        const float ws1 = warp_sum(acc1[ti]);
        if (lane == 0) {
            smem[0][ti][warp] = ws0;
            smem[1][ti][warp] = ws1;
        }
    }
    __syncthreads();

    // Only tid==0 writes all ActiveT pairs (<= 16 floats) -> single-thread __threadfence()!
    if (tid == 0) {
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            const float s0 = (((smem[0][ti][0] + smem[0][ti][1]) + (smem[0][ti][2] + smem[0][ti][3])) +
                              ((smem[0][ti][4] + smem[0][ti][5]) + (smem[0][ti][6] + smem[0][ti][7]))) +
                             smem[0][ti][8];
            const float s1 = (((smem[1][ti][0] + smem[1][ti][1]) + (smem[1][ti][2] + smem[1][ti][3])) +
                              ((smem[1][ti][4] + smem[1][ti][5]) + (smem[1][ti][6] + smem[1][ti][7]))) +
                             smem[1][ti][8];
            const std::int64_t base = static_cast<std::int64_t>(ti) * kNumExperts + row0;
            st_cg_f32(&router_logits[base + 0], s0);
            st_cg_f32(&router_logits[base + 1], s1);
        }
        __threadfence();
        s_is_last = (atomicAdd(done_counter, 1) == 63) ? 1 : 0;
    }
    __syncthreads();
    if (!s_is_last) { return; }

    if (tid == 0) { done_counter[0] = 0; }
    asm volatile("" ::: "memory");

    if constexpr (ActiveT == 1) {
        if (warp == 0) {
            warp_select_top8_128(router_logits, bias, route_scale, topk_ids, topk_weights, nullptr);
        } else if constexpr (PrepMode != 0) {
            // Warps 1..8 (256 threads) stage x[2304] -> x_staged
            const int wtid = tid - 32;
            for (int idx = wtid; idx < kHidden / 8; idx += 256) {
                const uint4 raw = __ldg(reinterpret_cast<const uint4*>(x) + idx);
                alignas(16) half2 h[4];
                if constexpr (PrepMode == 1) {
                    bf16x8_to_staged_h2(raw, h);
                } else {
                    bf16x8_to_h2(raw, h);
                }
                reinterpret_cast<uint4*>(x_staged)[idx] = *reinterpret_cast<const uint4*>(h);
            }
        }
        return;
    } else {
        __shared__ int s_ids[ActiveT * kTopK];
        __shared__ int s_wsum[4];
        __shared__ int s_wact[4];
        // All ActiveT <= 8 tokens select Top-8 IN PARALLEL across warps 0..ActiveT-1!
        if (warp < ActiveT) {
            warp_select_top8_128(router_logits + static_cast<std::int64_t>(warp) * kNumExperts, bias,
                                 route_scale, topk_ids + warp * kTopK, topk_weights + warp * kTopK,
                                 s_ids + warp * kTopK);
        }
        __syncthreads();

        unsigned long long mask = 0ULL;
        int cnt                 = 0;
        int off_inc             = 0;
        int act_inc             = 0;
        if (tid < 128) {
            const int e         = tid;
            constexpr int total = ActiveT * kTopK;
#pragma unroll
            for (int i = 0; i < total; ++i) {
                if (s_ids[i] == e) { mask |= (1ULL << i); }
            }
            cnt              = __popcll(mask);
            expert_counts[e] = cnt;
            off_inc          = cnt;
            act_inc          = (cnt > 0) ? 1 : 0;
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
        }
        __syncthreads();

        if (tid < 128) {
            const int e       = tid;
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
            int cursor = my_off;
            while (mask != 0ULL) {
                const int bit          = __ffsll(mask) - 1;
                sorted_pairs[cursor++] = bit;
                mask &= (mask - 1ULL);
            }
        }
    }
}

// ============================================================================
// 1c. Group C & Group D Mega-Fused Pre-MoE-to-Router Kernels (T == 1, 64 CTAs x 288t)
// ============================================================================

struct MoeAttnResSources {
    const __nv_bfloat16* sources[5];
};

// Group C: ZERO-BARRIER Fused AttnRes<S, Q=1> + mlp_norm + x_staged + Router + Bitonic Top-8!
// Works on ANY GPU (0 grid barriers -> zero deadlock risk, 0 spills, 100% bit-exact!).
template <int S, int PrepMode>
__global__ __launch_bounds__(288, 2) void attn_res_moe_router_t1_kernel(
    MoeAttnResSources src_args, const float* __restrict__ query,
    const __nv_bfloat16* __restrict__ mlp_norm_w, const __nv_bfloat16* __restrict__ router_w,
    const float* __restrict__ router_bias, float route_scale, float eps,
    __nv_bfloat16* __restrict__ mlp_out, float* __restrict__ router_logits,
    float* __restrict__ scores_buf, float* __restrict__ raws_buf,
    std::int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights,
    std::int32_t* __restrict__ done_counter, half* __restrict__ x_staged) {
    constexpr int kWarps   = 9;
    constexpr float kScale = 0.020833333333333332f;
    __shared__ float s_red[S * 2][kWarps];
    __shared__ float s_sc[S * 2];
    __shared__ float s_cs[32];
    __shared__ int s_cid[32];
    __shared__ int s_is_last;

    const int row0  = blockIdx.x * 2;
    const int tid   = threadIdx.x; // 0..287
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int idx   = tid * 8;

    // Issue router_w + bias loads upfront so HBM2 fetches overlap AttnRes compute!
    const uint4* w0 = reinterpret_cast<const uint4*>(router_w + static_cast<std::int64_t>(row0) * kHidden);
    const uint4* w1 = reinterpret_cast<const uint4*>(router_w + static_cast<std::int64_t>(row0 + 1) * kHidden);
    const uint4 wv0 = ld_cg_u4(w0 + tid);
    const uint4 wv1 = ld_cg_u4(w1 + tid);
    float b0 = 0.0f, b1 = 0.0f;
    if (tid == 0) {
        b0 = __ldg(router_bias + row0);
        b1 = __ldg(router_bias + row0 + 1);
    }

    // 1-pass AttnRes: compute BOTH sq[s] AND raw_dot[s] in 1 __syncthreads()!
    const float4 q0 = __ldg(reinterpret_cast<const float4*>(query + idx));
    const float4 q1 = __ldg(reinterpret_cast<const float4*>(query + idx + 4));
    float v_reg[S][8];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        load_bf16x8(src_args.sources[s] + idx, v_reg[s]);
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { sq = fmaf(v_reg[s][j], v_reg[s][j], sq); }
        float dt = fmaf(q0.x, v_reg[s][0],
                        fmaf(q0.y, v_reg[s][1], fmaf(q0.z, v_reg[s][2], q0.w * v_reg[s][3])));
        dt = fmaf(q1.x, v_reg[s][4],
                  fmaf(q1.y, v_reg[s][5], fmaf(q1.z, v_reg[s][6], fmaf(q1.w, v_reg[s][7], dt))));
        sq = warp_sum(sq);
        dt = warp_sum(dt);
        if (lane == 0) {
            s_red[s][warp]     = sq;
            s_red[S + s][warp] = dt;
        }
    }
    __syncthreads();

    for (int k = warp; k < 2 * S; k += kWarps) {
        const float sum = warp_sum((lane < kWarps) ? s_red[k][lane] : 0.0f);
        if (lane == 0) { s_sc[k] = sum; }
    }
    __syncthreads();

    float inv_s[S], sc[S], mx = -1e30f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        inv_s[s] = rsqrtf(s_sc[s] * (1.0f / 2304.0f) + eps);
        sc[s]    = s_sc[S + s] * inv_s[s] * kScale;
        mx       = fmaxf(mx, sc[s]);
    }
    float ws[S], den = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        ws[s] = expf(sc[s] - mx);
        den += ws[s];
    }
    const float inv_d = 1.0f / den;
#pragma unroll
    for (int s = 0; s < S; ++s) { ws[s] = (ws[s] * inv_d) * inv_s[s]; }

    float mixed[8], sq_m = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) { acc = fmaf(ws[s], v_reg[s][j], acc); }
        mixed[j] = acc;
        sq_m     = fmaf(acc, acc, sq_m);
    }
    sq_m = warp_sum(sq_m);
    if (lane == 0) { s_red[0][warp] = sq_m; }
    __syncthreads();
    if (warp == 0) {
        const float sum = warp_sum((lane < kWarps) ? s_red[0][lane] : 0.0f);
        if (lane == 0) { s_sc[0] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    const float inv_out = s_sc[0];
    float nw[8];
    load_bf16x8(mlp_norm_w + idx, nw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { mixed[j] = mixed[j] * inv_out * nw[j]; }
    const uint4 xpack = pack_bf16x8(mixed);

    if (blockIdx.x == 0) {
        reinterpret_cast<uint4*>(mlp_out)[tid] = xpack;
        if constexpr (PrepMode != 0) {
            alignas(16) half2 h[4];
            if constexpr (PrepMode == 1) {
                bf16x8_to_staged_h2(xpack, h);
            } else {
                bf16x8_to_h2(xpack, h);
            }
            reinterpret_cast<uint4*>(x_staged)[tid] = *reinterpret_cast<const uint4*>(h);
        }
    }

    const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&wv0);
    const auto* wb1 = reinterpret_cast<const __nv_bfloat162*>(&wv1);
    const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&xpack);
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f0 = __bfloat1622float2(wb0[j]);
        const float2 f1 = __bfloat1622float2(wb1[j]);
        const float2 xf = __bfloat1622float2(xb[j]);
        a0              = fmaf(f0.x, xf.x, fmaf(f0.y, xf.y, a0));
        a1              = fmaf(f1.x, xf.x, fmaf(f1.y, xf.y, a1));
    }
    const float w0s = warp_sum(a0), w1s = warp_sum(a1);
    if (lane == 0) {
        s_red[0][warp] = w0s;
        s_red[1][warp] = w1s;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = (((s_red[0][0] + s_red[0][1]) + (s_red[0][2] + s_red[0][3])) +
                          ((s_red[0][4] + s_red[0][5]) + (s_red[0][6] + s_red[0][7]))) +
                         s_red[0][8];
        const float s1 = (((s_red[1][0] + s_red[1][1]) + (s_red[1][2] + s_red[1][3])) +
                          ((s_red[1][4] + s_red[1][5]) + (s_red[1][6] + s_red[1][7]))) +
                         s_red[1][8];
        router_logits[row0 + 0] = s0;
        router_logits[row0 + 1] = s1;
        const float r0          = 1.0f / (1.0f + expf(-s0));
        const float r1          = 1.0f / (1.0f + expf(-s1));
        st_cg_f32(&raws_buf[row0 + 0], r0);
        st_cg_f32(&raws_buf[row0 + 1], r1);
        st_cg_f32(&scores_buf[row0 + 0], r0 + b0);
        st_cg_f32(&scores_buf[row0 + 1], r1 + b1);
        __threadfence();
        s_is_last = (atomicAdd(done_counter, 1) == 63) ? 1 : 0;
    }
    __syncthreads();
    if (!s_is_last) { return; }
    if (tid == 0) { done_counter[0] = 0; }
    asm volatile("" ::: "memory");
    block_bitonic_top8_128(scores_buf, raws_buf, route_scale, s_cs, s_cid, topk_ids, topk_weights);
}

// Group D: Mega-Fused 2-Barrier 64x288 Pre-MoE Core (ALL 5 stages in ONE launch at T == 1!)
//   Phase 0+1 (0 bars): Replicated AttnRes<S,1>+Prep + Down [4608->64] (0 loops, 0 z HBM!)
//   -- Grid Barrier 1 (64 CTAs) --
//   Phase 2: UpBlend (64 CTAs x 36 ch x 8t = 288t exact!) -> y_buf + partial_sq[64]
//   -- Grid Barrier 2 (64 CTAs) --
//   Phase 3+4: PostNorm + x_staged + Router (288t x 1 uint4, 0 loops!) + Bitonic Top-8!
template <int S, int PrepMode>
__global__ __launch_bounds__(288, 2) void attn_res_loop_ctrl_moe_router_t1_kernel(
    MoeAttnResSources src_args, const float* __restrict__ query,
    const __nv_bfloat16* __restrict__ msg, const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ gf_w, const __nv_bfloat16* __restrict__ norm_w,
    const __nv_bfloat16* __restrict__ router_w, const float* __restrict__ router_bias,
    float route_scale, float eps, __nv_bfloat16* __restrict__ cur_buf,
    __nv_bfloat16* __restrict__ h_buf, __nv_bfloat16* __restrict__ y_buf,
    float* __restrict__ partial_sq, __nv_bfloat16* __restrict__ mlp_out,
    unsigned int* __restrict__ bar_word, float* __restrict__ router_logits,
    float* __restrict__ scores_buf, float* __restrict__ raws_buf,
    std::int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights,
    std::int32_t* __restrict__ router_done, half* __restrict__ x_staged) {
    constexpr int kWarps   = 9;
    constexpr float kScale = 0.020833333333333332f;
    __shared__ float s_red[S * 2][kWarps];
    __shared__ float s_sc[S * 2];
    __shared__ float s_h[64];
    __shared__ float s_cs[32];
    __shared__ int s_cid[32];
    __shared__ float s_inv_y;
    __shared__ int s_is_last;

    const int bid = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, idx = tid * 8;

    // Issue down_w loads upfront so HBM2 fetches overlap AttnRes+Prep!
    const uint4* dw_ptr = reinterpret_cast<const uint4*>(down_w + static_cast<std::int64_t>(bid) * 4608);
    const uint4 dw_cur  = ld_cg_u4(dw_ptr + tid);
    const uint4 dw_msg  = ld_cg_u4(dw_ptr + 288 + tid);

    const float4 q0 = __ldg(reinterpret_cast<const float4*>(query + idx));
    const float4 q1 = __ldg(reinterpret_cast<const float4*>(query + idx + 4));
    float v_reg[S][8];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        load_bf16x8(src_args.sources[s] + idx, v_reg[s]);
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { sq = fmaf(v_reg[s][j], v_reg[s][j], sq); }
        float dt = fmaf(q0.x, v_reg[s][0],
                        fmaf(q0.y, v_reg[s][1], fmaf(q0.z, v_reg[s][2], q0.w * v_reg[s][3])));
        dt = fmaf(q1.x, v_reg[s][4],
                  fmaf(q1.y, v_reg[s][5], fmaf(q1.z, v_reg[s][6], fmaf(q1.w, v_reg[s][7], dt))));
        sq = warp_sum(sq);
        dt = warp_sum(dt);
        if (lane == 0) {
            s_red[s][warp]     = sq;
            s_red[S + s][warp] = dt;
        }
    }
    __syncthreads();

    for (int k = warp; k < 2 * S; k += kWarps) {
        const float sum = warp_sum((lane < kWarps) ? s_red[k][lane] : 0.0f);
        if (lane == 0) { s_sc[k] = sum; }
    }
    __syncthreads();

    float inv_s[S], sc[S], mx = -1e30f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        inv_s[s] = rsqrtf(s_sc[s] * (1.0f / 2304.0f) + eps);
        sc[s]    = s_sc[S + s] * inv_s[s] * kScale;
        mx       = fmaxf(mx, sc[s]);
    }
    float ws[S], den = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        ws[s] = expf(sc[s] - mx);
        den += ws[s];
    }
    const float inv_d = 1.0f / den;
#pragma unroll
    for (int s = 0; s < S; ++s) { ws[s] = (ws[s] * inv_d) * inv_s[s]; }

    float cur_v[8], sq_c = 0.0f;
    uint4 cur_raw;
    auto* cur_b = reinterpret_cast<__nv_bfloat162*>(&cur_raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float a = 0.0f, b = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            a = fmaf(ws[s], v_reg[s][2 * j], a);
            b = fmaf(ws[s], v_reg[s][2 * j + 1], b);
        }
        cur_b[j]         = __floats2bfloat162_rn(a, b);
        const float2 bf  = __bfloat1622float2(cur_b[j]);
        cur_v[2 * j]     = bf.x;
        cur_v[2 * j + 1] = bf.y;
        sq_c             = fmaf(bf.x, bf.x, fmaf(bf.y, bf.y, sq_c));
    }
    if (bid == 0) { st_cg_u4(reinterpret_cast<uint4*>(cur_buf) + tid, cur_raw); }

    float msg_v[8], sq_m = 0.0f;
    load_bf16x8(msg + idx, msg_v);
#pragma unroll
    for (int j = 0; j < 8; ++j) { sq_m = fmaf(msg_v[j], msg_v[j], sq_m); }
    sq_c = warp_sum(sq_c);
    sq_m = warp_sum(sq_m);
    if (lane == 0) {
        s_red[0][warp] = sq_c;
        s_red[1][warp] = sq_m;
    }
    __syncthreads();
    if (warp < 2) {
        const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_sc[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    // Phase 1: Down [4608->64] directly from registers (ZERO loops, ZERO z HBM!)
    const float ic  = s_sc[0], im = s_sc[1];
    const auto* wbc = reinterpret_cast<const __nv_bfloat162*>(&dw_cur);
    const auto* wbm = reinterpret_cast<const __nv_bfloat162*>(&dw_msg);
    float acc_c = 0.0f, acc_m = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 wc = __bfloat1622float2(wbc[j]);
        const float2 wm = __bfloat1622float2(wbm[j]);
        acc_c           = fmaf(wc.x, cur_v[2 * j], fmaf(wc.y, cur_v[2 * j + 1], acc_c));
        acc_m           = fmaf(wm.x, msg_v[2 * j], fmaf(wm.y, msg_v[2 * j + 1], acc_m));
    }
    const float w_acc = warp_sum(acc_c * ic + acc_m * im);
    if (lane == 0) { s_red[0][warp] = w_acc; }
    __syncthreads();
    if (tid == 0) {
        const float hval = (((s_red[0][0] + s_red[0][1]) + (s_red[0][2] + s_red[0][3])) +
                            ((s_red[0][4] + s_red[0][5]) + (s_red[0][6] + s_red[0][7]))) +
                           s_red[0][8];
        const __nv_bfloat16 hv = __float2bfloat16_rn(hval);
        st_cg_u16(&h_buf[bid], *reinterpret_cast<const std::uint16_t*>(&hv));
    }

    grid_barrier_64(bar_word); // Barrier 1

    // Phase 2: UpBlend (64 CTAs x 36 ch x 8t = 288t exact!)
    if (tid < 64) {
        const std::uint16_t hu = ld_cg_u16(&h_buf[tid]);
        s_h[tid]               = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&hu));
    }
    __syncthreads();

    const int ch_loc = tid >> 3, sub = tid & 7, ch = bid * 36 + ch_loc, k_off = sub * 8;
    const uint4 gw = ld_cg_u4(gf_w + static_cast<std::int64_t>(ch) * 64 + k_off);
    const uint4 fw = ld_cg_u4(gf_w + static_cast<std::int64_t>(2304 + ch) * 64 + k_off);
    const auto* gb = reinterpret_cast<const __nv_bfloat162*>(&gw);
    const auto* fb = reinterpret_cast<const __nv_bfloat162*>(&fw);
    float g_acc = 0.0f, f_acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 g0 = __bfloat1622float2(gb[j]);
        const float2 f0 = __bfloat1622float2(fb[j]);
        g_acc = fmaf(g0.x, s_h[k_off + 2 * j], fmaf(g0.y, s_h[k_off + 2 * j + 1], g_acc));
        f_acc = fmaf(f0.x, s_h[k_off + 2 * j], fmaf(f0.y, s_h[k_off + 2 * j + 1], f_acc));
    }
    g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 1, 8);
    g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 2, 8);
    g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 4, 8);
    f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 1, 8);
    f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 2, 8);
    f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 4, 8);

    float y_sq = 0.0f;
    if (sub == 0) {
        const std::uint16_t cu16 = ld_cg_u16(&cur_buf[ch]);
        const float cv           = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&cu16));
        const float mv           = __bfloat162float(__ldg(msg + ch));
        const float y            = (1.0f - tanhf(f_acc)) * cv + tanhf(g_acc) * mv;
        y_sq                     = y * y;
        const __nv_bfloat16 ybf  = __float2bfloat16_rn(y);
        st_cg_u16(&y_buf[ch], *reinterpret_cast<const std::uint16_t*>(&ybf));
    }
    const float wsum = warp_sum(y_sq);
    if (lane == 0) { s_red[0][warp] = wsum; }
    __syncthreads();
    if (tid == 0) {
        const float tot = (((s_red[0][0] + s_red[0][1]) + (s_red[0][2] + s_red[0][3])) +
                           ((s_red[0][4] + s_red[0][5]) + (s_red[0][6] + s_red[0][7]))) +
                          s_red[0][8];
        st_cg_f32(&partial_sq[bid], tot);
    }

    grid_barrier_64(bar_word); // Barrier 2

    // Phase 3+4: PostNorm + x_staged + Router + Bitonic Top-8!
    const int row0  = bid * 2;
    const uint4* w0 = reinterpret_cast<const uint4*>(router_w + static_cast<std::int64_t>(row0) * kHidden);
    const uint4* w1 = reinterpret_cast<const uint4*>(router_w + static_cast<std::int64_t>(row0 + 1) * kHidden);
    const uint4 wv0 = ld_cg_u4(w0 + tid);
    const uint4 wv1 = ld_cg_u4(w1 + tid);
    const uint4 yv  = ld_cg_u4(reinterpret_cast<const uint4*>(y_buf) + tid);
    const uint4 nwv = __ldg(reinterpret_cast<const uint4*>(norm_w) + tid);
    float b0 = 0.0f, b1 = 0.0f;
    if (tid == 0) {
        b0 = __ldg(router_bias + row0);
        b1 = __ldg(router_bias + row0 + 1);
    }

    if (warp == 0) {
        const float acc = warp_reduce_64_f4(partial_sq, lane);
        if (lane == 0) { s_inv_y = rsqrtf(acc * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    const float inv_y = s_inv_y;

    uint4 xpack;
    auto* xbp       = reinterpret_cast<__nv_bfloat162*>(&xpack);
    const auto* ybp = reinterpret_cast<const __nv_bfloat162*>(&yv);
    const auto* nwp = reinterpret_cast<const __nv_bfloat162*>(&nwv);
    const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&wv0);
    const auto* wb1 = reinterpret_cast<const __nv_bfloat162*>(&wv1);
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 yf  = __bfloat1622float2(ybp[j]);
        const float2 nwf = __bfloat1622float2(nwp[j]);
        xbp[j]           = __floats2bfloat162_rn(yf.x * inv_y * nwf.x, yf.y * inv_y * nwf.y);
        const float2 xf  = __bfloat1622float2(xbp[j]);
        const float2 f0  = __bfloat1622float2(wb0[j]);
        const float2 f1  = __bfloat1622float2(wb1[j]);
        a0               = fmaf(f0.x, xf.x, fmaf(f0.y, xf.y, a0));
        a1               = fmaf(f1.x, xf.x, fmaf(f1.y, xf.y, a1));
    }

    if (bid == 0) {
        reinterpret_cast<uint4*>(mlp_out)[tid] = xpack;
        if constexpr (PrepMode != 0) {
            alignas(16) half2 h[4];
            if constexpr (PrepMode == 1) {
                bf16x8_to_staged_h2(xpack, h);
            } else {
                bf16x8_to_h2(xpack, h);
            }
            reinterpret_cast<uint4*>(x_staged)[tid] = *reinterpret_cast<const uint4*>(h);
        }
    }

    const float w0s = warp_sum(a0), w1s = warp_sum(a1);
    if (lane == 0) {
        s_red[0][warp] = w0s;
        s_red[1][warp] = w1s;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = (((s_red[0][0] + s_red[0][1]) + (s_red[0][2] + s_red[0][3])) +
                          ((s_red[0][4] + s_red[0][5]) + (s_red[0][6] + s_red[0][7]))) +
                         s_red[0][8];
        const float s1 = (((s_red[1][0] + s_red[1][1]) + (s_red[1][2] + s_red[1][3])) +
                          ((s_red[1][4] + s_red[1][5]) + (s_red[1][6] + s_red[1][7]))) +
                         s_red[1][8];
        router_logits[row0 + 0] = s0;
        router_logits[row0 + 1] = s1;
        const float r0          = 1.0f / (1.0f + expf(-s0));
        const float r1          = 1.0f / (1.0f + expf(-s1));
        st_cg_f32(&raws_buf[row0 + 0], r0);
        st_cg_f32(&raws_buf[row0 + 1], r1);
        st_cg_f32(&scores_buf[row0 + 0], r0 + b0);
        st_cg_f32(&scores_buf[row0 + 1], r1 + b1);
        __threadfence();
        s_is_last = (atomicAdd(router_done, 1) == 63) ? 1 : 0;
    }
    __syncthreads();
    if (!s_is_last) { return; }
    if (tid == 0) { router_done[0] = 0; }
    asm volatile("" ::: "memory");
    block_bitonic_top8_128(scores_buf, raws_buf, route_scale, s_cs, s_cid, topk_ids, topk_weights);
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
// 2. Warp-Aligned Zero-Spill NVFP4 T == 1 Kernels (128 threads/CTA, <=64 regs!)
//    - w13: 8 threads/row (128B exact cache line, 9 exact K-iters, 64-reg 8 blks/SM
//           -> 693 GB/s on V100!, 3 warp shuffles, 0 barriers in reduce!)
//    - w2:  288 CTAs (8 rows/CTA, 8 half-warps handle all 8 experts in parallel,
//           2 threads/row x uint4 = 32B exact HBM2 sector, 2-deep prefetch, 1 shuffle!)
// ============================================================================

template <bool PreStaged>
__global__ __launch_bounds__(128, 8) void moe_nvfp4_t1_w13_kernel(
    const void* __restrict__ x_in, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom, const std::int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights, int num_slabs, float gate_cap, float up_cap,
    half* __restrict__ inter_out) {
    __shared__ uint4 s_x[144][2];

    const int tid = threadIdx.x; // 0..127
    if constexpr (PreStaged) {
        const auto* xp = static_cast<const uint4*>(x_in);
        auto* sx_flat  = reinterpret_cast<uint4*>(s_x);
        for (int idx = tid; idx < 288; idx += 128) { sx_flat[idx] = __ldg(xp + idx); }
    } else {
        const auto* x = static_cast<const __nv_bfloat16*>(x_in);
        for (int g = tid; g < 144; g += 128) {
            const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(x + g * 16));
            const uint4 x1 = __ldg(reinterpret_cast<const uint4*>(x + g * 16 + 8));
            alignas(16) half2 xa0[4];
            alignas(16) half2 xa1[4];
            bf16x8_to_staged_h2(x0, xa0);
            bf16x8_to_staged_h2(x1, xa1);
            s_x[g][0] = *reinterpret_cast<const uint4*>(xa0);
            s_x[g][1] = *reinterpret_cast<const uint4*>(xa1);
        }
    }
    __syncthreads();

    const int p_loc       = tid >> 3; // 0..15 (4 pairs/warp, 100% warp-aligned!)
    const int k_lane      = tid & 7;  // 0..7  (8 threads x 16B = 128B exact cache line!)
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

    uint4 gw         = ld_cg_u4(gc_ptr + k_lane * 16);
    uint4 uw         = ld_cg_u4(uc_ptr + k_lane * 16);
    std::uint16_t gs = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + k_lane * 2));
    std::uint16_t us = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + k_lane * 2));

#pragma unroll 1
    for (int it = 0; it < 9; ++it) {
        const int q  = it * 8 + k_lane;
        const int g0 = q * 2;
        uint4 gw_n{}, uw_n{};
        std::uint16_t gs_n = 0, us_n = 0;
        if (it < 8) {
            gw_n = ld_cg_u4(gc_ptr + (q + 8) * 16);
            uw_n = ld_cg_u4(uc_ptr + (q + 8) * 16);
            gs_n = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + (q + 8) * 2));
            us_n = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + (q + 8) * 2));
        }

        half2 d0[4], d1[4];
        const uint4 a0 = s_x[g0][0];
        const uint4 a1 = s_x[g0][1];
        nvfp4_decode8_h2(gw.x, rebias, d0);
        nvfp4_decode8_h2(gw.y, rebias, d1);
        acc_g = fmaf(h2_dot16_u4_f32(a0, a1, d0, d1), decode_e4m3_f32(gs & 0xFFu), acc_g);
        nvfp4_decode8_h2(uw.x, rebias, d0);
        nvfp4_decode8_h2(uw.y, rebias, d1);
        acc_u = fmaf(h2_dot16_u4_f32(a0, a1, d0, d1), decode_e4m3_f32(us & 0xFFu), acc_u);

        const uint4 b0 = s_x[g0 + 1][0];
        const uint4 b1 = s_x[g0 + 1][1];
        nvfp4_decode8_h2(gw.z, rebias, d0);
        nvfp4_decode8_h2(gw.w, rebias, d1);
        acc_g = fmaf(h2_dot16_u4_f32(b0, b1, d0, d1), decode_e4m3_f32(gs >> 8), acc_g);
        nvfp4_decode8_h2(uw.z, rebias, d0);
        nvfp4_decode8_h2(uw.w, rebias, d1);
        acc_u = fmaf(h2_dot16_u4_f32(b0, b1, d0, d1), decode_e4m3_f32(us >> 8), acc_u);

        gw = gw_n;
        uw = uw_n;
        gs = gs_n;
        us = us_n;
    }

    acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, 4, 8);
    acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, 2, 8);
    acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, 1, 8);
    acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, 4, 8);
    acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, 2, 8);
    acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, 1, 8);

    if (k_lane == 0) {
        const float outer = __ldg(w13_amax + eid) / __ldg(w13_denom);
        const float y     = softcapped_swiglu(acc_g * outer, acc_u * outer, gate_cap, up_cap) *
                            topk_weights[slot];
        const int blk8  = j_local & ~7;
        const int p8    = j_local & 7;
        const int dst_j = slab * kInterPerSlab + blk8 + ((p8 & 3) * 2 + (p8 >> 2));
        inter_out[static_cast<std::int64_t>(slot) * inter_local + dst_j] = __float2half_rn(y);
    }
}

__global__ __launch_bounds__(128, 8) void moe_nvfp4_t1_w2_kernel(
    const half* __restrict__ inter_in, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom, const std::int32_t* __restrict__ topk_ids,
    int num_slabs, __nv_bfloat16* __restrict__ out) {
    __shared__ uint4 s_inter[8][36][2];
    __shared__ float s_exp[8][8];
    __shared__ int s_eid[8];
    __shared__ float s_outer[8];

    const int tid    = threadIdx.x;    // 0..127
    const int slot   = tid >> 4;       // 0..7 (ALL 8 experts in parallel, 1 half-warp/expert!)
    const int r_loc  = (tid >> 1) & 7; // 0..7 (8 rows/CTA -> 288 CTAs = 3.6 CTAs/SM!)
    const int k_lane = tid & 1;        // 0..1 (2 threads x uint4 = 32B exact HBM2 sector!)
    const int row    = blockIdx.x * 8 + r_loc;

    const float inv_denom = 1.0f / __ldg(w2_denom);
    if (tid < kTopK) {
        const int eid = topk_ids[tid];
        s_eid[tid]    = eid;
        s_outer[tid]  = __ldg(w2_amax + eid) * inv_denom;
    }

    const int inter_local = num_slabs * kInterPerSlab;
    const half2 rebias    = __float2half2_rn(16384.0f);
    float total           = 0.0f;

    for (int slab = 0; slab < num_slabs; ++slab) {
        for (int idx = tid; idx < 288; idx += 128) {
            const int slt       = idx / 36;
            const int g         = idx - slt * 36;
            const half* src_ptr = inter_in + static_cast<std::int64_t>(slt) * inter_local +
                                  slab * kInterPerSlab + g * 16;
            s_inter[slt][g][0] = __ldg(reinterpret_cast<const uint4*>(src_ptr));
            s_inter[slt][g][1] = __ldg(reinterpret_cast<const uint4*>(src_ptr + 8));
        }
        __syncthreads();

        const int eid            = s_eid[slot];
        const std::int64_t ridx  = (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + row;
        const std::uint8_t* cptr = w2_codes + ridx * 288;
        const std::uint8_t* sptr = w2_scales + ridx * 36;

        float acc          = 0.0f;
        uint4 w_cur        = ld_cg_u4(cptr + k_lane * 16);
        std::uint16_t s_c  = __ldg(reinterpret_cast<const std::uint16_t*>(sptr + k_lane * 2));

#pragma unroll 1
        for (int it = 0; it < 9; ++it) {
            const int q = it * 2 + k_lane; // 0..17 (18 uint4 = 288B EXACT!)
            uint4 w_nxt{};
            std::uint16_t s_n = 0;
            if (it < 8) {
                w_nxt = ld_cg_u4(cptr + (q + 2) * 16);
                s_n   = __ldg(reinterpret_cast<const std::uint16_t*>(sptr + (q + 2) * 2));
            }
            half2 wd0[4], wd1[4];
            const uint4 a0 = s_inter[slot][2 * q][0];
            const uint4 a1 = s_inter[slot][2 * q][1];
            nvfp4_decode8_h2(w_cur.x, rebias, wd0);
            nvfp4_decode8_h2(w_cur.y, rebias, wd1);
            acc = fmaf(h2_dot16_u4_f32(a0, a1, wd0, wd1), decode_e4m3_f32(s_c & 0xFFu), acc);

            const uint4 b0 = s_inter[slot][2 * q + 1][0];
            const uint4 b1 = s_inter[slot][2 * q + 1][1];
            nvfp4_decode8_h2(w_cur.z, rebias, wd0);
            nvfp4_decode8_h2(w_cur.w, rebias, wd1);
            acc = fmaf(h2_dot16_u4_f32(b0, b1, wd0, wd1), decode_e4m3_f32(s_c >> 8), acc);

            w_cur = w_nxt;
            s_c   = s_n;
        }
        total = fmaf(acc, s_outer[slot], total);
        if (num_slabs > 1) { __syncthreads(); }
    }

    total += __shfl_xor_sync(0xFFFFFFFFu, total, 1, 2);
    if (k_lane == 0) { s_exp[slot][r_loc] = total; }
    __syncthreads();

    if (tid < 8) {
        const float sum = ((s_exp[0][tid] + s_exp[1][tid]) + (s_exp[2][tid] + s_exp[3][tid])) +
                          ((s_exp[4][tid] + s_exp[5][tid]) + (s_exp[6][tid] + s_exp[7][tid]));
        out[blockIdx.x * 8 + tid] = __float2bfloat16_rn(sum);
    }
}

// ============================================================================
// 3. Warp-Aligned Zero-Spill MXFP8 T == 1 Kernels (128 threads/CTA, 0 SMEM reduce!)
// ============================================================================

template <bool PreStaged>
__global__ __launch_bounds__(128, 6) void moe_mxfp8_t1_w13_kernel(
    const void* __restrict__ x_in, const std::uint8_t* __restrict__ w13_codes,
    const std::uint8_t* __restrict__ w13_scales, const std::int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights, int num_slabs, float gate_cap, float up_cap,
    half* __restrict__ inter_out) {
    alignas(16) __shared__ half2 s_x[72][16];

    const int tid = threadIdx.x;
    if constexpr (PreStaged) {
        const auto* xp = static_cast<const uint4*>(x_in);
        auto* sx_flat  = reinterpret_cast<uint4*>(s_x);
        for (int idx = tid; idx < 288; idx += 128) { sx_flat[idx] = __ldg(xp + idx); }
    } else {
        const auto* x = static_cast<const __nv_bfloat16*>(x_in);
        for (int idx = tid; idx < 288; idx += 128) {
            const int g    = idx >> 2; // 0..71
            const int q    = idx & 3;  // 0..3
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + g * 32 + q * 8));
            alignas(16) half2 xh[4];
            bf16x8_to_h2(xv, xh);
            *reinterpret_cast<uint4*>(&s_x[g][q * 4]) = *reinterpret_cast<const uint4*>(xh);
        }
    }
    __syncthreads();

    const int p_loc       = tid >> 3; // 0..15
    const int k_lane      = tid & 7;  // 0..7
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

    uint4 gv0        = ld_cg_u4(gc_ptr + k_lane * 32);
    uint4 gv1        = ld_cg_u4(gc_ptr + k_lane * 32 + 16);
    uint4 uv0        = ld_cg_u4(uc_ptr + k_lane * 32);
    uint4 uv1        = ld_cg_u4(uc_ptr + k_lane * 32 + 16);
    std::uint8_t gsb = __ldg(gs_ptr + k_lane);
    std::uint8_t usb = __ldg(us_ptr + k_lane);

#pragma unroll 1
    for (int it = 0; it < 9; ++it) {
        const int g = it * 8 + k_lane;
        uint4 ngv0{}, ngv1{}, nuv0{}, nuv1{};
        std::uint8_t ngsb = 0, nusb = 0;
        if (it < 8) {
            const int gn = g + 8;
            ngv0 = ld_cg_u4(gc_ptr + gn * 32);
            ngv1 = ld_cg_u4(gc_ptr + gn * 32 + 16);
            nuv0 = ld_cg_u4(uc_ptr + gn * 32);
            nuv1 = ld_cg_u4(uc_ptr + gn * 32 + 16);
            ngsb = __ldg(gs_ptr + gn);
            nusb = __ldg(us_ptr + gn);
        }
        const float gs = decode_ue8m0_f32(gsb) * 256.0f;
        const float us = decode_ue8m0_f32(usb) * 256.0f;
        half2 gp0 = __half2half2(__ushort_as_half(0));
        half2 gp1 = __half2half2(__ushort_as_half(0));
        half2 up0 = __half2half2(__ushort_as_half(0));
        half2 up1 = __half2half2(__ushort_as_half(0));

        const std::uint32_t gw[8] = {gv0.x, gv0.y, gv0.z, gv0.w, gv1.x, gv1.y, gv1.z, gv1.w};
        const std::uint32_t uw[8] = {uv0.x, uv0.y, uv0.z, uv0.w, uv1.x, uv1.y, uv1.z, uv1.w};
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const half2 x0 = s_x[g][w * 2 + 0];
            const half2 x1 = s_x[g][w * 2 + 1];
            half2 lo, hi;
            fp8_decode4_h2_div256(gw[w], lo, hi);
            gp0 = __hfma2(x0, lo, gp0);
            gp1 = __hfma2(x1, hi, gp1);
            fp8_decode4_h2_div256(uw[w], lo, hi);
            up0 = __hfma2(x0, lo, up0);
            up1 = __hfma2(x1, hi, up1);
        }
        const float2 gf = __half22float2(__hadd2(gp0, gp1));
        const float2 uf = __half22float2(__hadd2(up0, up1));
        acc_g           = fmaf(gf.x + gf.y, gs, acc_g);
        acc_u           = fmaf(uf.x + uf.y, us, acc_u);

        gv0 = ngv0; gv1 = ngv1;
        uv0 = nuv0; uv1 = nuv1;
        gsb = ngsb; usb = nusb;
    }

    acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, 4, 8);
    acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, 2, 8);
    acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, 1, 8);
    acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, 4, 8);
    acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, 2, 8);
    acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, 1, 8);

    if (k_lane == 0) {
        const float y = softcapped_swiglu(acc_g, acc_u, gate_cap, up_cap) * topk_weights[slot];
        inter_out[static_cast<std::int64_t>(slot) * inter_local + j_global] = __float2half_rn(y);
    }
}

__global__ __launch_bounds__(128, 6) void moe_mxfp8_t1_w2_kernel(
    const half* __restrict__ inter_in, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const std::int32_t* __restrict__ topk_ids,
    int num_slabs, __nv_bfloat16* __restrict__ out) {
    alignas(16) __shared__ half2 s_inter[8][18][16];
    __shared__ float s_warp[4][8];
    __shared__ int s_eid[8];

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int r_loc  = lane >> 2; // 0..7
    const int k_lane = lane & 3;  // 0..3 (4 threads x uint4 = 64B coalesced, 9 exact iters!)
    const int row    = blockIdx.x * 8 + r_loc;

    if (tid < kTopK) { s_eid[tid] = topk_ids[tid]; }

    const int inter_local = num_slabs * kInterPerSlab;
    const int slot_a      = warp * 2;
    const int slot_b      = slot_a + 1;
    float total           = 0.0f;

    for (int slab = 0; slab < num_slabs; ++slab) {
        for (int idx = tid; idx < 288; idx += 128) {
            const int g_flat = idx >> 1; // 0..143
            const int hlf    = idx & 1;  // 0..1
            const int slt    = g_flat / 18;
            const int grp    = g_flat - slt * 18;
            const half* sp   = inter_in + static_cast<std::int64_t>(slt) * inter_local +
                               slab * kInterPerSlab + grp * 32 + hlf * 16;
            *reinterpret_cast<uint4*>(&s_inter[slt][grp][hlf * 8 + 0]) =
                __ldg(reinterpret_cast<const uint4*>(sp));
            *reinterpret_cast<uint4*>(&s_inter[slt][grp][hlf * 8 + 4]) =
                __ldg(reinterpret_cast<const uint4*>(sp + 8));
        }
        __syncthreads();

        const int eid_a = s_eid[slot_a];
        const int eid_b = s_eid[slot_b];
        const std::int64_t ridx_a =
            (static_cast<std::int64_t>(slab) * kNumExperts + eid_a) * kHidden + row;
        const std::int64_t ridx_b =
            (static_cast<std::int64_t>(slab) * kNumExperts + eid_b) * kHidden + row;
        const std::uint8_t* ca = w2_codes + ridx_a * 576;
        const std::uint8_t* cb = w2_codes + ridx_b * 576;
        const std::uint8_t* sa = w2_scales + ridx_a * 18;
        const std::uint8_t* sb = w2_scales + ridx_b * 18;

        uint4 va         = ld_cg_u4(ca + k_lane * 16);
        uint4 vb         = ld_cg_u4(cb + k_lane * 16);
        std::uint8_t sab = __ldg(sa + (k_lane >> 1));
        std::uint8_t sbb = __ldg(sb + (k_lane >> 1));

#pragma unroll 1
        for (int it = 0; it < 9; ++it) {
            const int hi = it * 4 + k_lane; // 0..35
            const int g  = hi >> 1;         // 0..17
            const int h  = hi & 1;          // 0..1
            uint4 nva{}, nvb{};
            std::uint8_t nsab = 0, nsbb = 0;
            if (it < 8) {
                const int hin = hi + 4;
                nva  = ld_cg_u4(ca + hin * 16);
                nvb  = ld_cg_u4(cb + hin * 16);
                nsab = __ldg(sa + (hin >> 1));
                nsbb = __ldg(sb + (hin >> 1));
            }
            const float wa = decode_ue8m0_f32(sab) * 256.0f;
            const float wb = decode_ue8m0_f32(sbb) * 256.0f;

            const std::uint32_t wsa[4] = {va.x, va.y, va.z, va.w};
            const std::uint32_t wsb[4] = {vb.x, vb.y, vb.z, vb.w};
            half2 pa0 = __half2half2(__ushort_as_half(0));
            half2 pa1 = __half2half2(__ushort_as_half(0));
            half2 pb0 = __half2half2(__ushort_as_half(0));
            half2 pb1 = __half2half2(__ushort_as_half(0));
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                half2 lo, hi2;
                fp8_decode4_h2_div256(wsa[w], lo, hi2);
                pa0 = __hfma2(s_inter[slot_a][g][h * 8 + w * 2 + 0], lo, pa0);
                pa1 = __hfma2(s_inter[slot_a][g][h * 8 + w * 2 + 1], hi2, pa1);
                fp8_decode4_h2_div256(wsb[w], lo, hi2);
                pb0 = __hfma2(s_inter[slot_b][g][h * 8 + w * 2 + 0], lo, pb0);
                pb1 = __hfma2(s_inter[slot_b][g][h * 8 + w * 2 + 1], hi2, pb1);
            }
            const float2 fa = __half22float2(__hadd2(pa0, pa1));
            const float2 fb = __half22float2(__hadd2(pb0, pb1));
            total           = fmaf(fa.x + fa.y, wa, fmaf(fb.x + fb.y, wb, total));

            va  = nva;
            vb  = nvb;
            sab = nsab;
            sbb = nsbb;
        }
        if (num_slabs > 1) { __syncthreads(); }
    }

    total += __shfl_xor_sync(0xFFFFFFFFu, total, 2, 4);
    total += __shfl_xor_sync(0xFFFFFFFFu, total, 1, 4);
    if (k_lane == 0) { s_warp[warp][r_loc] = total; }
    __syncthreads();

    if (tid < 8) {
        const float sum = (s_warp[0][tid] + s_warp[1][tid]) + (s_warp[2][tid] + s_warp[3][tid]);
        out[blockIdx.x * 8 + tid] = __float2bfloat16_rn(sum);
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
    alignas(16) half2 h[4];
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
// 5. Warp-Aligned ZERO-SPILL Grouped MoE for T = 2..32 (NVFP4 & MXFP8)
//    - NVFP4 w13: SMEM s_x[2][144][8] (9 KB) + <=64 regs (8 blks/SM) -> 693 GB/s!
//    - NVFP4 w2:  2 threads/row x uint4 (32B exact sector) + 2-deep prefetch -> 530+ GB/s!
// ============================================================================

// TileM=4 (18KB SMEM, 4 blks/SM) + specialized mc=1/2/3..4 + 2-deep weight/scale prefetch (-274us/step!)
__global__ __launch_bounds__(128, 4) void moe_nvfp4_grouped_w13_coalesced_kernel(
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

    __shared__ uint4 s_x[4][144][2];
    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];
    __shared__ float s_w[4];

    const int tid         = threadIdx.x; // 0..127
    const int p_loc       = tid >> 3;    // 0..15 (4 pairs/warp)
    const int k_lane      = tid & 7;     // 0..7  (128B exact cache line!)
    const int inter_local = num_slabs * kInterPerSlab;
    const int j_global    = blockIdx.x * 16 + p_loc;
    const int slab        = j_global / kInterPerSlab;
    const int j_local     = j_global - slab * kInterPerSlab;
    const int blk8        = j_local & ~7;
    const int p8          = j_local & 7;
    const int dst_j       = slab * kInterPerSlab + blk8 + ((p8 & 3) * 2 + (p8 >> 2));

    const std::int64_t row_base =
        (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kW13RowsSlab + 2 * j_local;
    const std::uint8_t* gc_ptr = w13_codes + row_base * 1152;
    const std::uint8_t* uc_ptr = gc_ptr + 1152;
    const std::uint8_t* gs_ptr = w13_scales + row_base * 144;
    const std::uint8_t* us_ptr = gs_ptr + 144;

    const half2 rebias = __float2half2_rn(16384.0f);
    const float outer  = __ldg(w13_amax + eid) / __ldg(w13_denom);

    auto finish_m = [&](int m, float ag, float au) {
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 4, 8);
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 2, 8);
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 1, 8);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 4, 8);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 2, 8);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 1, 8);
        if (k_lane == 0) {
            const float y = softcapped_swiglu(ag * outer, au * outer, gate_cap, up_cap) * s_w[m];
            inter_out[static_cast<std::int64_t>(s_pairs[m]) * inter_local + dst_j] = __float2half_rn(y);
        }
    };

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
            s_w[tid]       = topk_weights[pair];
        }
        __syncthreads();

        // Stage up to 4 tokens into shared memory ONCE for all 16 row-pairs!
        const int total_g = m_cnt * 144;
        for (int idx = tid; idx < total_g; idx += 128) {
            const int m    = idx / 144;
            const int g    = idx - m * 144;
            const half* xp = x_staged + static_cast<std::int64_t>(s_toks[m]) * kHidden + g * 16;
            s_x[m][g][0]   = __ldg(reinterpret_cast<const uint4*>(xp));
            s_x[m][g][1]   = __ldg(reinterpret_cast<const uint4*>(xp + 8));
        }
        __syncthreads();

        uint4 gw         = ld_cg_u4(gc_ptr + k_lane * 16);
        uint4 uw         = ld_cg_u4(uc_ptr + k_lane * 16);
        std::uint16_t gs = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + k_lane * 2));
        std::uint16_t us = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + k_lane * 2));

        if (m_cnt == 1) {
            float ag = 0.0f, au = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 9; ++it) {
                const int q  = it * 8 + k_lane;
                const int g0 = q * 2;
                uint4 gw_n{}, uw_n{};
                std::uint16_t gs_n = 0, us_n = 0;
                if (it < 8) {
                    gw_n = ld_cg_u4(gc_ptr + (q + 8) * 16);
                    uw_n = ld_cg_u4(uc_ptr + (q + 8) * 16);
                    gs_n = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + (q + 8) * 2));
                    us_n = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + (q + 8) * 2));
                }

                half2 d0[4], d1[4];
                const uint4 a0 = s_x[0][g0][0];
                const uint4 a1 = s_x[0][g0][1];
                nvfp4_decode8_h2(gw.x, rebias, d0);
                nvfp4_decode8_h2(gw.y, rebias, d1);
                ag = fmaf(h2_dot16_u4_f32(a0, a1, d0, d1), decode_e4m3_f32(gs & 0xFFu), ag);
                nvfp4_decode8_h2(uw.x, rebias, d0);
                nvfp4_decode8_h2(uw.y, rebias, d1);
                au = fmaf(h2_dot16_u4_f32(a0, a1, d0, d1), decode_e4m3_f32(us & 0xFFu), au);

                const uint4 b0 = s_x[0][g0 + 1][0];
                const uint4 b1 = s_x[0][g0 + 1][1];
                nvfp4_decode8_h2(gw.z, rebias, d0);
                nvfp4_decode8_h2(gw.w, rebias, d1);
                ag = fmaf(h2_dot16_u4_f32(b0, b1, d0, d1), decode_e4m3_f32(gs >> 8), ag);
                nvfp4_decode8_h2(uw.z, rebias, d0);
                nvfp4_decode8_h2(uw.w, rebias, d1);
                au = fmaf(h2_dot16_u4_f32(b0, b1, d0, d1), decode_e4m3_f32(us >> 8), au);

                gw = gw_n; uw = uw_n; gs = gs_n; us = us_n;
            }
            finish_m(0, ag, au);
        } else if (m_cnt == 2) {
            float ag0 = 0.0f, ag1 = 0.0f;
            float au0 = 0.0f, au1 = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 9; ++it) {
                const int q  = it * 8 + k_lane;
                const int g0 = q * 2;
                uint4 gw_n{}, uw_n{};
                std::uint16_t gs_n = 0, us_n = 0;
                if (it < 8) {
                    gw_n = ld_cg_u4(gc_ptr + (q + 8) * 16);
                    uw_n = ld_cg_u4(uc_ptr + (q + 8) * 16);
                    gs_n = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + (q + 8) * 2));
                    us_n = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + (q + 8) * 2));
                }

                half2 d0[4], d1[4];
                auto dot2 = [&](std::uint32_t wlo, std::uint32_t whi, int grp, float sc,
                                float& acc0, float& acc1) {
                    nvfp4_decode8_h2(wlo, rebias, d0);
                    nvfp4_decode8_h2(whi, rebias, d1);
                    acc0 = fmaf(h2_dot16_u4_f32(s_x[0][grp][0], s_x[0][grp][1], d0, d1), sc, acc0);
                    acc1 = fmaf(h2_dot16_u4_f32(s_x[1][grp][0], s_x[1][grp][1], d0, d1), sc, acc1);
                };
                dot2(gw.x, gw.y, g0,     decode_e4m3_f32(gs & 0xFFu), ag0, ag1);
                dot2(gw.z, gw.w, g0 + 1, decode_e4m3_f32(gs >> 8),    ag0, ag1);
                dot2(uw.x, uw.y, g0,     decode_e4m3_f32(us & 0xFFu), au0, au1);
                dot2(uw.z, uw.w, g0 + 1, decode_e4m3_f32(us >> 8),    au0, au1);

                gw = gw_n; uw = uw_n; gs = gs_n; us = us_n;
            }
            finish_m(0, ag0, au0);
            finish_m(1, ag1, au1);
        } else {
            float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
            float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 9; ++it) {
                const int q  = it * 8 + k_lane;
                const int g0 = q * 2;
                uint4 gw_n{}, uw_n{};
                std::uint16_t gs_n = 0, us_n = 0;
                if (it < 8) {
                    gw_n = ld_cg_u4(gc_ptr + (q + 8) * 16);
                    uw_n = ld_cg_u4(uc_ptr + (q + 8) * 16);
                    gs_n = __ldg(reinterpret_cast<const std::uint16_t*>(gs_ptr + (q + 8) * 2));
                    us_n = __ldg(reinterpret_cast<const std::uint16_t*>(us_ptr + (q + 8) * 2));
                }

                half2 d0[4], d1[4];
                auto dot4 = [&](std::uint32_t wlo, std::uint32_t whi, int grp, float sc,
                                float& acc0, float& acc1, float& acc2, float& acc3) {
                    nvfp4_decode8_h2(wlo, rebias, d0);
                    nvfp4_decode8_h2(whi, rebias, d1);
                    acc0 = fmaf(h2_dot16_u4_f32(s_x[0][grp][0], s_x[0][grp][1], d0, d1), sc, acc0);
                    acc1 = fmaf(h2_dot16_u4_f32(s_x[1][grp][0], s_x[1][grp][1], d0, d1), sc, acc1);
                    acc2 = fmaf(h2_dot16_u4_f32(s_x[2][grp][0], s_x[2][grp][1], d0, d1), sc, acc2);
                    if (m_cnt == 4) {
                        acc3 = fmaf(h2_dot16_u4_f32(s_x[3][grp][0], s_x[3][grp][1], d0, d1), sc, acc3);
                    }
                };
                dot4(gw.x, gw.y, g0,     decode_e4m3_f32(gs & 0xFFu), ag0, ag1, ag2, ag3);
                dot4(gw.z, gw.w, g0 + 1, decode_e4m3_f32(gs >> 8),    ag0, ag1, ag2, ag3);
                dot4(uw.x, uw.y, g0,     decode_e4m3_f32(us & 0xFFu), au0, au1, au2, au3);
                dot4(uw.z, uw.w, g0 + 1, decode_e4m3_f32(us >> 8),    au0, au1, au2, au3);

                gw = gw_n; uw = uw_n; gs = gs_n; us = us_n;
            }
            finish_m(0, ag0, au0);
            finish_m(1, ag1, au1);
            finish_m(2, ag2, au2);
            if (m_cnt == 4) { finish_m(3, ag3, au3); }
        }
        if (m_base + 4 < end) { __syncthreads(); }
    }
}

// 128 rows/CTA (RowIters=2 x 64 rows -> 18 CTAs/expert, 2x less SMEM loading + specialized mc=1/2/3..4 -> 505+ GB/s!)
__global__ __launch_bounds__(128, 8) void moe_nvfp4_grouped_w2_coalesced_kernel(
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

    __shared__ uint4 s_in[4][36][2];
    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];

    const int tid           = threadIdx.x; // 0..127
    const int r_loc         = tid >> 1;    // 0..63
    const int k_lane        = tid & 1;     // 0..1  (2 threads x uint4 = 32B exact sector!)
    const int blocks_per_sl = kHidden / 128; // 18
    const int slab          = static_cast<int>(blockIdx.x) / blocks_per_sl;
    const int bx            = static_cast<int>(blockIdx.x) - slab * blocks_per_sl;
    const int inter_local   = num_slabs * kInterPerSlab;

    const float outer  = __ldg(w2_amax + eid) / __ldg(w2_denom);
    const half2 rebias = __float2half2_rn(16384.0f);

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
        }
        __syncthreads();

        for (int idx = tid; idx < m_cnt * 36; idx += 128) {
            const int m     = idx / 36;
            const int g     = idx - m * 36;
            const half* src = inter_staged + static_cast<std::int64_t>(s_pairs[m]) * inter_local +
                              slab * kInterPerSlab + g * 16;
            s_in[m][g][0]   = __ldg(reinterpret_cast<const uint4*>(src));
            s_in[m][g][1]   = __ldg(reinterpret_cast<const uint4*>(src + 8));
        }
        __syncthreads();

#pragma unroll
        for (int ri = 0; ri < 2; ++ri) {
            const int row           = bx * 128 + ri * 64 + r_loc;
            const std::int64_t ridx = (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + row;
            const std::uint8_t* cp  = w2_codes + ridx * 288;
            const std::uint8_t* sp  = w2_scales + ridx * 36;

            if (m_cnt == 1) {
                float acc         = 0.0f;
                uint4 w_cur       = ld_cg_u4(cp + k_lane * 16);
                std::uint16_t s_c = __ldg(reinterpret_cast<const std::uint16_t*>(sp + k_lane * 2));
#pragma unroll 1
                for (int it = 0; it < 9; ++it) {
                    const int q = it * 2 + k_lane;
                    uint4 w_nxt{};
                    std::uint16_t s_n = 0;
                    if (it < 8) {
                        w_nxt = ld_cg_u4(cp + (q + 2) * 16);
                        s_n   = __ldg(reinterpret_cast<const std::uint16_t*>(sp + (q + 2) * 2));
                    }
                    half2 d0[4], d1[4];
                    nvfp4_decode8_h2(w_cur.x, rebias, d0);
                    nvfp4_decode8_h2(w_cur.y, rebias, d1);
                    acc = fmaf(h2_dot16_u4_f32(s_in[0][2 * q][0], s_in[0][2 * q][1], d0, d1),
                               decode_e4m3_f32(s_c & 0xFFu), acc);

                    nvfp4_decode8_h2(w_cur.z, rebias, d0);
                    nvfp4_decode8_h2(w_cur.w, rebias, d1);
                    acc = fmaf(h2_dot16_u4_f32(s_in[0][2 * q + 1][0], s_in[0][2 * q + 1][1], d0, d1),
                               decode_e4m3_f32(s_c >> 8), acc);

                    w_cur = w_nxt;
                    s_c   = s_n;
                }
                acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 1, 2);
                if (k_lane == 0) {
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[0]) * kHidden + row,
                              acc * outer);
                }
            } else if (m_cnt == 2) {
                float a0 = 0.0f, a1 = 0.0f;
                uint4 w_cur       = ld_cg_u4(cp + k_lane * 16);
                std::uint16_t s_c = __ldg(reinterpret_cast<const std::uint16_t*>(sp + k_lane * 2));
#pragma unroll 1
                for (int it = 0; it < 9; ++it) {
                    const int q = it * 2 + k_lane;
                    uint4 w_nxt{};
                    std::uint16_t s_n = 0;
                    if (it < 8) {
                        w_nxt = ld_cg_u4(cp + (q + 2) * 16);
                        s_n   = __ldg(reinterpret_cast<const std::uint16_t*>(sp + (q + 2) * 2));
                    }
                    const float fs0 = decode_e4m3_f32(s_c & 0xFFu);
                    const float fs1 = decode_e4m3_f32(s_c >> 8);
                    half2 d0[4], d1[4];
                    nvfp4_decode8_h2(w_cur.x, rebias, d0);
                    nvfp4_decode8_h2(w_cur.y, rebias, d1);
                    a0 = fmaf(h2_dot16_u4_f32(s_in[0][2 * q][0], s_in[0][2 * q][1], d0, d1), fs0, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_in[1][2 * q][0], s_in[1][2 * q][1], d0, d1), fs0, a1);

                    nvfp4_decode8_h2(w_cur.z, rebias, d0);
                    nvfp4_decode8_h2(w_cur.w, rebias, d1);
                    a0 = fmaf(h2_dot16_u4_f32(s_in[0][2 * q + 1][0], s_in[0][2 * q + 1][1], d0, d1), fs1, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_in[1][2 * q + 1][0], s_in[1][2 * q + 1][1], d0, d1), fs1, a1);

                    w_cur = w_nxt;
                    s_c   = s_n;
                }
                a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1, 2);
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 1, 2);
                if (k_lane == 0) {
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[0]) * kHidden + row, a0 * outer);
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[1]) * kHidden + row, a1 * outer);
                }
            } else {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                uint4 w_cur       = ld_cg_u4(cp + k_lane * 16);
                std::uint16_t s_c = __ldg(reinterpret_cast<const std::uint16_t*>(sp + k_lane * 2));
#pragma unroll 1
                for (int it = 0; it < 9; ++it) {
                    const int q = it * 2 + k_lane;
                    uint4 w_nxt{};
                    std::uint16_t s_n = 0;
                    if (it < 8) {
                        w_nxt = ld_cg_u4(cp + (q + 2) * 16);
                        s_n   = __ldg(reinterpret_cast<const std::uint16_t*>(sp + (q + 2) * 2));
                    }
                    const float fs0 = decode_e4m3_f32(s_c & 0xFFu);
                    const float fs1 = decode_e4m3_f32(s_c >> 8);
                    half2 d0[4], d1[4];
                    nvfp4_decode8_h2(w_cur.x, rebias, d0);
                    nvfp4_decode8_h2(w_cur.y, rebias, d1);
                    a0 = fmaf(h2_dot16_u4_f32(s_in[0][2 * q][0], s_in[0][2 * q][1], d0, d1), fs0, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_in[1][2 * q][0], s_in[1][2 * q][1], d0, d1), fs0, a1);
                    a2 = fmaf(h2_dot16_u4_f32(s_in[2][2 * q][0], s_in[2][2 * q][1], d0, d1), fs0, a2);
                    if (m_cnt == 4) {
                        a3 = fmaf(h2_dot16_u4_f32(s_in[3][2 * q][0], s_in[3][2 * q][1], d0, d1), fs0, a3);
                    }
                    nvfp4_decode8_h2(w_cur.z, rebias, d0);
                    nvfp4_decode8_h2(w_cur.w, rebias, d1);
                    a0 = fmaf(h2_dot16_u4_f32(s_in[0][2 * q + 1][0], s_in[0][2 * q + 1][1], d0, d1), fs1, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_in[1][2 * q + 1][0], s_in[1][2 * q + 1][1], d0, d1), fs1, a1);
                    a2 = fmaf(h2_dot16_u4_f32(s_in[2][2 * q + 1][0], s_in[2][2 * q + 1][1], d0, d1), fs1, a2);
                    if (m_cnt == 4) {
                        a3 = fmaf(h2_dot16_u4_f32(s_in[3][2 * q + 1][0], s_in[3][2 * q + 1][1], d0, d1), fs1, a3);
                    }
                    w_cur = w_nxt;
                    s_c   = s_n;
                }
                a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1, 2);
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 1, 2);
                a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 1, 2);
                a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 1, 2);
                if (k_lane == 0) {
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[0]) * kHidden + row, a0 * outer);
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[1]) * kHidden + row, a1 * outer);
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[2]) * kHidden + row, a2 * outer);
                    if (m_cnt == 4) {
                        atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[3]) * kHidden + row, a3 * outer);
                    }
                }
            }
        }
        if (m_base + 4 < end) { __syncthreads(); }
    }
}

// Warp-aligned ZERO-SPILL grouped MXFP8 for T = 2..32 (SMEM s_x[4][72][4] + specialized mc=1/2/3..4 + pf2)
__global__ __launch_bounds__(128, 4) void moe_mxfp8_grouped_w13_coalesced_kernel(
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

    __shared__ uint4 s_x[4][72][4];
    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];
    __shared__ float s_w[4];

    const int tid         = threadIdx.x;
    const int p_loc       = tid >> 3; // 0..15
    const int k_lane      = tid & 7;  // 0..7
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

    auto finish_m = [&](int m, float ag, float au) {
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 4, 8);
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 2, 8);
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 1, 8);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 4, 8);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 2, 8);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 1, 8);
        if (k_lane == 0) {
            const float y = softcapped_swiglu(ag, au, gate_cap, up_cap) * s_w[m];
            inter_out[static_cast<std::int64_t>(s_pairs[m]) * inter_local + j_global] = __float2half_rn(y);
        }
    };

    for (int m_base = start; m_base < end; m_base += 4) {
        const int m_cnt = min(4, end - m_base);
        if (tid < m_cnt) {
            const int pair = sorted_pairs[m_base + tid];
            s_pairs[tid]   = pair;
            s_toks[tid]    = pair / kTopK;
            s_w[tid]       = topk_weights[pair];
        }
        __syncthreads();

        const int total_v = m_cnt * 288;
        for (int idx = tid; idx < total_v; idx += 128) {
            const int m    = idx / 288;
            const int rem  = idx - m * 288;
            const int g    = rem >> 2;
            const int q    = rem & 3;
            const half* xp = x_fp16 + static_cast<std::int64_t>(s_toks[m]) * kHidden + g * 32 + q * 8;
            s_x[m][g][q]   = __ldg(reinterpret_cast<const uint4*>(xp));
        }
        __syncthreads();

        uint4 gv0        = ld_cg_u4(gc_ptr + k_lane * 32);
        uint4 gv1        = ld_cg_u4(gc_ptr + k_lane * 32 + 16);
        uint4 uv0        = ld_cg_u4(uc_ptr + k_lane * 32);
        uint4 uv1        = ld_cg_u4(uc_ptr + k_lane * 32 + 16);
        std::uint8_t gsb = __ldg(gs_ptr + k_lane);
        std::uint8_t usb = __ldg(us_ptr + k_lane);

        if (m_cnt == 1) {
            float ag = 0.0f, au = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 9; ++it) {
                const int g = it * 8 + k_lane;
                uint4 ngv0{}, ngv1{}, nuv0{}, nuv1{};
                std::uint8_t ngsb = 0, nusb = 0;
                if (it < 8) {
                    const int gn = g + 8;
                    ngv0 = ld_cg_u4(gc_ptr + gn * 32);
                    ngv1 = ld_cg_u4(gc_ptr + gn * 32 + 16);
                    nuv0 = ld_cg_u4(uc_ptr + gn * 32);
                    nuv1 = ld_cg_u4(uc_ptr + gn * 32 + 16);
                    ngsb = __ldg(gs_ptr + gn);
                    nusb = __ldg(us_ptr + gn);
                }
                const float gs = decode_ue8m0_f32(gsb) * 256.0f;
                const float us = decode_ue8m0_f32(usb) * 256.0f;
                half2 d0[4], d1[4];

                const uint4 x0 = s_x[0][g][0], x1 = s_x[0][g][1];
                fp8_decode4_h2_div256(gv0.x, d0[0], d0[1]);
                fp8_decode4_h2_div256(gv0.y, d0[2], d0[3]);
                fp8_decode4_h2_div256(gv0.z, d1[0], d1[1]);
                fp8_decode4_h2_div256(gv0.w, d1[2], d1[3]);
                ag = fmaf(h2_dot16_u4_f32(x0, x1, d0, d1), gs, ag);
                fp8_decode4_h2_div256(uv0.x, d0[0], d0[1]);
                fp8_decode4_h2_div256(uv0.y, d0[2], d0[3]);
                fp8_decode4_h2_div256(uv0.z, d1[0], d1[1]);
                fp8_decode4_h2_div256(uv0.w, d1[2], d1[3]);
                au = fmaf(h2_dot16_u4_f32(x0, x1, d0, d1), us, au);

                const uint4 x2 = s_x[0][g][2], x3 = s_x[0][g][3];
                fp8_decode4_h2_div256(gv1.x, d0[0], d0[1]);
                fp8_decode4_h2_div256(gv1.y, d0[2], d0[3]);
                fp8_decode4_h2_div256(gv1.z, d1[0], d1[1]);
                fp8_decode4_h2_div256(gv1.w, d1[2], d1[3]);
                ag = fmaf(h2_dot16_u4_f32(x2, x3, d0, d1), gs, ag);
                fp8_decode4_h2_div256(uv1.x, d0[0], d0[1]);
                fp8_decode4_h2_div256(uv1.y, d0[2], d0[3]);
                fp8_decode4_h2_div256(uv1.z, d1[0], d1[1]);
                fp8_decode4_h2_div256(uv1.w, d1[2], d1[3]);
                au = fmaf(h2_dot16_u4_f32(x2, x3, d0, d1), us, au);

                gv0 = ngv0; gv1 = ngv1; uv0 = nuv0; uv1 = nuv1; gsb = ngsb; usb = nusb;
            }
            finish_m(0, ag, au);
        } else if (m_cnt == 2) {
            float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 9; ++it) {
                const int g = it * 8 + k_lane;
                uint4 ngv0{}, ngv1{}, nuv0{}, nuv1{};
                std::uint8_t ngsb = 0, nusb = 0;
                if (it < 8) {
                    const int gn = g + 8;
                    ngv0 = ld_cg_u4(gc_ptr + gn * 32);
                    ngv1 = ld_cg_u4(gc_ptr + gn * 32 + 16);
                    nuv0 = ld_cg_u4(uc_ptr + gn * 32);
                    nuv1 = ld_cg_u4(uc_ptr + gn * 32 + 16);
                    ngsb = __ldg(gs_ptr + gn);
                    nusb = __ldg(us_ptr + gn);
                }
                const float gs = decode_ue8m0_f32(gsb) * 256.0f;
                const float us = decode_ue8m0_f32(usb) * 256.0f;
                half2 d0[4], d1[4];
                auto dot2 = [&](const uint4& v, int q0, float sc, float& a0, float& a1) {
                    fp8_decode4_h2_div256(v.x, d0[0], d0[1]);
                    fp8_decode4_h2_div256(v.y, d0[2], d0[3]);
                    fp8_decode4_h2_div256(v.z, d1[0], d1[1]);
                    fp8_decode4_h2_div256(v.w, d1[2], d1[3]);
                    a0 = fmaf(h2_dot16_u4_f32(s_x[0][g][q0], s_x[0][g][q0 + 1], d0, d1), sc, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_x[1][g][q0], s_x[1][g][q0 + 1], d0, d1), sc, a1);
                };
                dot2(gv0, 0, gs, ag0, ag1);
                dot2(gv1, 2, gs, ag0, ag1);
                dot2(uv0, 0, us, au0, au1);
                dot2(uv1, 2, us, au0, au1);
                gv0 = ngv0; gv1 = ngv1; uv0 = nuv0; uv1 = nuv1; gsb = ngsb; usb = nusb;
            }
            finish_m(0, ag0, au0);
            finish_m(1, ag1, au1);
        } else {
            float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
            float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
#pragma unroll 1
            for (int it = 0; it < 9; ++it) {
                const int g = it * 8 + k_lane;
                uint4 ngv0{}, ngv1{}, nuv0{}, nuv1{};
                std::uint8_t ngsb = 0, nusb = 0;
                if (it < 8) {
                    const int gn = g + 8;
                    ngv0 = ld_cg_u4(gc_ptr + gn * 32);
                    ngv1 = ld_cg_u4(gc_ptr + gn * 32 + 16);
                    nuv0 = ld_cg_u4(uc_ptr + gn * 32);
                    nuv1 = ld_cg_u4(uc_ptr + gn * 32 + 16);
                    ngsb = __ldg(gs_ptr + gn);
                    nusb = __ldg(us_ptr + gn);
                }
                const float gs = decode_ue8m0_f32(gsb) * 256.0f;
                const float us = decode_ue8m0_f32(usb) * 256.0f;
                half2 d0[4], d1[4];
                auto dot4 = [&](const uint4& v, int q0, float sc,
                                float& a0, float& a1, float& a2, float& a3) {
                    fp8_decode4_h2_div256(v.x, d0[0], d0[1]);
                    fp8_decode4_h2_div256(v.y, d0[2], d0[3]);
                    fp8_decode4_h2_div256(v.z, d1[0], d1[1]);
                    fp8_decode4_h2_div256(v.w, d1[2], d1[3]);
                    a0 = fmaf(h2_dot16_u4_f32(s_x[0][g][q0], s_x[0][g][q0 + 1], d0, d1), sc, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_x[1][g][q0], s_x[1][g][q0 + 1], d0, d1), sc, a1);
                    a2 = fmaf(h2_dot16_u4_f32(s_x[2][g][q0], s_x[2][g][q0 + 1], d0, d1), sc, a2);
                    if (m_cnt == 4) {
                        a3 = fmaf(h2_dot16_u4_f32(s_x[3][g][q0], s_x[3][g][q0 + 1], d0, d1), sc, a3);
                    }
                };
                dot4(gv0, 0, gs, ag0, ag1, ag2, ag3);
                dot4(gv1, 2, gs, ag0, ag1, ag2, ag3);
                dot4(uv0, 0, us, au0, au1, au2, au3);
                dot4(uv1, 2, us, au0, au1, au2, au3);
                gv0 = ngv0; gv1 = ngv1; uv0 = nuv0; uv1 = nuv1; gsb = ngsb; usb = nusb;
            }
            finish_m(0, ag0, au0);
            finish_m(1, ag1, au1);
            finish_m(2, ag2, au2);
            if (m_cnt == 4) { finish_m(3, ag3, au3); }
        }
        if (m_base + 4 < end) { __syncthreads(); }
    }
}

// 128 rows/CTA (RowIters=4 x 32 rows -> 18 CTAs/expert, 2x less SMEM loading + specialized mc=1/2/3..4 + pf2!)
__global__ __launch_bounds__(128, 6) void moe_mxfp8_grouped_w2_coalesced_kernel(
    const half* __restrict__ inter_fp16, const std::uint8_t* __restrict__ w2_codes,
    const std::uint8_t* __restrict__ w2_scales, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ num_active, const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ sorted_pairs, int num_slabs, float* __restrict__ accum_out) {
    if (static_cast<int>(blockIdx.y) >= num_active[0]) { return; }
    const int eid   = active_experts[blockIdx.y];
    const int start = expert_offsets[eid];
    const int end   = expert_offsets[eid + 1];
    if (start >= end) { return; }

    __shared__ uint4 s_in[4][18][4];
    __shared__ int s_toks[4];
    __shared__ int s_pairs[4];

    const int tid           = threadIdx.x;
    const int r_loc         = tid >> 2;      // 0..31
    const int k_lane        = tid & 3;       // 0..3 (4 threads x uint4 = 64B coalesced)
    const int blocks_per_sl = kHidden / 128; // 18
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

        for (int idx = tid; idx < m_cnt * 72; idx += 128) {
            const int g_flat = idx >> 2;
            const int q      = idx & 3;
            const int m      = g_flat / 18;
            const int g      = g_flat - m * 18;
            const half* src  = inter_fp16 + static_cast<std::int64_t>(s_pairs[m]) * inter_local +
                               slab * kInterPerSlab + g * 32 + q * 8;
            s_in[m][g][q]    = __ldg(reinterpret_cast<const uint4*>(src));
        }
        __syncthreads();

#pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
            const int row           = bx * 128 + ri * 32 + r_loc;
            const std::int64_t ridx = (static_cast<std::int64_t>(slab) * kNumExperts + eid) * kHidden + row;
            const std::uint8_t* cp  = w2_codes + ridx * 576;
            const std::uint8_t* sp  = w2_scales + ridx * 18;

            uint4 v_cur        = ld_cg_u4(cp + k_lane * 16);
            std::uint8_t sc_b  = __ldg(sp + (k_lane >> 1));

            if (m_cnt == 1) {
                float acc = 0.0f;
#pragma unroll 1
                for (int it = 0; it < 9; ++it) {
                    const int hi = it * 4 + k_lane;
                    const int g  = hi >> 1;
                    const int q0 = (hi & 1) * 2;
                    uint4 v_nxt{};
                    std::uint8_t sc_n = 0;
                    if (it < 8) {
                        const int hin = hi + 4;
                        v_nxt = ld_cg_u4(cp + hin * 16);
                        sc_n  = __ldg(sp + (hin >> 1));
                    }
                    const float wsc = decode_ue8m0_f32(sc_b) * 256.0f;
                    half2 d0[4], d1[4];
                    fp8_decode4_h2_div256(v_cur.x, d0[0], d0[1]);
                    fp8_decode4_h2_div256(v_cur.y, d0[2], d0[3]);
                    fp8_decode4_h2_div256(v_cur.z, d1[0], d1[1]);
                    fp8_decode4_h2_div256(v_cur.w, d1[2], d1[3]);
                    acc   = fmaf(h2_dot16_u4_f32(s_in[0][g][q0], s_in[0][g][q0 + 1], d0, d1), wsc, acc);
                    v_cur = v_nxt;
                    sc_b  = sc_n;
                }
                acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 2, 4);
                acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 1, 4);
                if (k_lane == 0) {
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[0]) * kHidden + row, acc);
                }
            } else if (m_cnt == 2) {
                float a0 = 0.0f, a1 = 0.0f;
#pragma unroll 1
                for (int it = 0; it < 9; ++it) {
                    const int hi = it * 4 + k_lane;
                    const int g  = hi >> 1;
                    const int q0 = (hi & 1) * 2;
                    uint4 v_nxt{};
                    std::uint8_t sc_n = 0;
                    if (it < 8) {
                        const int hin = hi + 4;
                        v_nxt = ld_cg_u4(cp + hin * 16);
                        sc_n  = __ldg(sp + (hin >> 1));
                    }
                    const float wsc = decode_ue8m0_f32(sc_b) * 256.0f;
                    half2 d0[4], d1[4];
                    fp8_decode4_h2_div256(v_cur.x, d0[0], d0[1]);
                    fp8_decode4_h2_div256(v_cur.y, d0[2], d0[3]);
                    fp8_decode4_h2_div256(v_cur.z, d1[0], d1[1]);
                    fp8_decode4_h2_div256(v_cur.w, d1[2], d1[3]);
                    a0    = fmaf(h2_dot16_u4_f32(s_in[0][g][q0], s_in[0][g][q0 + 1], d0, d1), wsc, a0);
                    a1    = fmaf(h2_dot16_u4_f32(s_in[1][g][q0], s_in[1][g][q0 + 1], d0, d1), wsc, a1);
                    v_cur = v_nxt;
                    sc_b  = sc_n;
                }
                a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 2, 4);
                a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1, 4);
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 2, 4);
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 1, 4);
                if (k_lane == 0) {
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[0]) * kHidden + row, a0);
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[1]) * kHidden + row, a1);
                }
            } else {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 1
                for (int it = 0; it < 9; ++it) {
                    const int hi = it * 4 + k_lane;
                    const int g  = hi >> 1;
                    const int q0 = (hi & 1) * 2;
                    uint4 v_nxt{};
                    std::uint8_t sc_n = 0;
                    if (it < 8) {
                        const int hin = hi + 4;
                        v_nxt = ld_cg_u4(cp + hin * 16);
                        sc_n  = __ldg(sp + (hin >> 1));
                    }
                    const float wsc = decode_ue8m0_f32(sc_b) * 256.0f;
                    half2 d0[4], d1[4];
                    fp8_decode4_h2_div256(v_cur.x, d0[0], d0[1]);
                    fp8_decode4_h2_div256(v_cur.y, d0[2], d0[3]);
                    fp8_decode4_h2_div256(v_cur.z, d1[0], d1[1]);
                    fp8_decode4_h2_div256(v_cur.w, d1[2], d1[3]);
                    a0 = fmaf(h2_dot16_u4_f32(s_in[0][g][q0], s_in[0][g][q0 + 1], d0, d1), wsc, a0);
                    a1 = fmaf(h2_dot16_u4_f32(s_in[1][g][q0], s_in[1][g][q0 + 1], d0, d1), wsc, a1);
                    a2 = fmaf(h2_dot16_u4_f32(s_in[2][g][q0], s_in[2][g][q0 + 1], d0, d1), wsc, a2);
                    if (m_cnt == 4) {
                        a3 = fmaf(h2_dot16_u4_f32(s_in[3][g][q0], s_in[3][g][q0 + 1], d0, d1), wsc, a3);
                    }
                    v_cur = v_nxt;
                    sc_b  = sc_n;
                }
                a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 2, 4);
                a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1, 4);
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 2, 4);
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 1, 4);
                a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 2, 4);
                a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 1, 4);
                a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 2, 4);
                a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 1, 4);
                if (k_lane == 0) {
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[0]) * kHidden + row, a0);
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[1]) * kHidden + row, a1);
                    atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[2]) * kHidden + row, a2);
                    if (m_cnt == 4) {
                        atomicAdd(accum_out + static_cast<std::int64_t>(s_toks[3]) * kHidden + row, a3);
                    }
                }
            }
        }
        if (m_base + 4 < end) { __syncthreads(); }
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
                               Inkling2MoeScratch& scratch, cudaStream_t stream,
                               std::int32_t prep_mode) {
    const int k      = x.ne[0];
    const int tokens = x.ne[1];
    const int n      = router_weight.ne[1];
    if (scratch.router_done != nullptr && k == kHidden && n == kNumExperts &&
        x.dtype == DType::BF16 && router_weight.dtype == DType::BF16 &&
        tokens >= 1 && tokens <= 8) {
        const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
        const auto* wp = static_cast<const __nv_bfloat16*>(router_weight.data);
        const auto* bp = static_cast<const float*>(router_bias_f32.data);
        half* x_staged = nullptr;
        const int pm   = (prep_mode == 1 || prep_mode == 2) ? prep_mode : 0;
        if (pm != 0) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            x_staged = get_moe_x_scratch(dev, static_cast<std::size_t>(tokens) * kHidden);
        }
        // T == 1 ultra-fast 4-warp Bitonic Top-8 path (uses accum_f32[0..255], unused by T==1 MoE!)
        if (tokens == 1 && scratch.accum_f32 != nullptr) {
            float* scores_buf = scratch.accum_f32;
            float* raws_buf   = scratch.accum_f32 + kNumExperts;
            if (pm == 1) {
                moe_router_t1_bitonic_288t_kernel<1><<<64, 288, 0, stream>>>(
                    xp, wp, bp, route_scale, scratch.router_logits, scores_buf, raws_buf,
                    scratch.topk_ids, scratch.topk_weights, scratch.router_done, x_staged);
            } else if (pm == 2) {
                moe_router_t1_bitonic_288t_kernel<2><<<64, 288, 0, stream>>>(
                    xp, wp, bp, route_scale, scratch.router_logits, scores_buf, raws_buf,
                    scratch.topk_ids, scratch.topk_weights, scratch.router_done, x_staged);
            } else {
                moe_router_t1_bitonic_288t_kernel<0><<<64, 288, 0, stream>>>(
                    xp, wp, bp, route_scale, scratch.router_logits, scores_buf, raws_buf,
                    scratch.topk_ids, scratch.topk_weights, scratch.router_done, nullptr);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        auto launch = [&]<int ActiveT>() {
            if (pm == 1) {
                moe_router_fused_kernel<ActiveT, 1><<<64, 288, 0, stream>>>(
                    xp, wp, bp, route_scale, scratch.router_logits, scratch.topk_ids,
                    scratch.topk_weights, scratch.expert_counts, scratch.expert_offsets,
                    scratch.sorted_pairs, scratch.active_experts, scratch.num_active,
                    scratch.router_done, x_staged, scratch.accum_f32);
            } else if (pm == 2) {
                moe_router_fused_kernel<ActiveT, 2><<<64, 288, 0, stream>>>(
                    xp, wp, bp, route_scale, scratch.router_logits, scratch.topk_ids,
                    scratch.topk_weights, scratch.expert_counts, scratch.expert_offsets,
                    scratch.sorted_pairs, scratch.active_experts, scratch.num_active,
                    scratch.router_done, x_staged, scratch.accum_f32);
            } else {
                moe_router_fused_kernel<ActiveT, 0><<<64, 288, 0, stream>>>(
                    xp, wp, bp, route_scale, scratch.router_logits, scratch.topk_ids,
                    scratch.topk_weights, scratch.expert_counts, scratch.expert_offsets,
                    scratch.sorted_pairs, scratch.active_experts, scratch.num_active,
                    scratch.router_done, nullptr, nullptr);
            }
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

void inkling2_attn_res_moe_router_fused(const void* const* source_ptrs, std::int32_t num_sources,
                                        const Tensor& query_f32, const Tensor& mlp_norm_w,
                                        const Tensor& router_weight, const Tensor& router_bias_f32,
                                        float route_scale, float eps, Tensor& mlp_out,
                                        Inkling2MoeScratch& scratch, cudaStream_t stream,
                                        std::int32_t prep_mode) {
    const int hidden = mlp_out.ne[0];
    const int tokens = mlp_out.ne[1];
    if (tokens == 1 && hidden == kHidden && num_sources >= 1 && num_sources <= 5 &&
        router_weight.ne[1] == kNumExperts && scratch.router_done != nullptr &&
        scratch.accum_f32 != nullptr) {
        MoeAttnResSources sa{};
        for (int s = 0; s < num_sources; ++s) {
            sa.sources[s] = static_cast<const __nv_bfloat16*>(source_ptrs[s]);
        }
        const auto* qp = static_cast<const float*>(query_f32.data);
        const auto* np = static_cast<const __nv_bfloat16*>(mlp_norm_w.data);
        const auto* wp = static_cast<const __nv_bfloat16*>(router_weight.data);
        const auto* bp = static_cast<const float*>(router_bias_f32.data);
        auto* mo       = static_cast<__nv_bfloat16*>(mlp_out.data);
        float* sc_buf  = scratch.accum_f32;
        float* rw_buf  = scratch.accum_f32 + kNumExperts;
        const int pm   = (prep_mode == 1 || prep_mode == 2) ? prep_mode : 0;
        half* x_staged = nullptr;
        if (pm != 0) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            x_staged = get_moe_x_scratch(dev, kHidden);
        }
        auto launch = [&]<int S>() {
            if (pm == 1) {
                attn_res_moe_router_t1_kernel<S, 1><<<64, 288, 0, stream>>>(
                    sa, qp, np, wp, bp, route_scale, eps, mo, scratch.router_logits, sc_buf, rw_buf,
                    scratch.topk_ids, scratch.topk_weights, scratch.router_done, x_staged);
            } else if (pm == 2) {
                attn_res_moe_router_t1_kernel<S, 2><<<64, 288, 0, stream>>>(
                    sa, qp, np, wp, bp, route_scale, eps, mo, scratch.router_logits, sc_buf, rw_buf,
                    scratch.topk_ids, scratch.topk_weights, scratch.router_done, x_staged);
            } else {
                attn_res_moe_router_t1_kernel<S, 0><<<64, 288, 0, stream>>>(
                    sa, qp, np, wp, bp, route_scale, eps, mo, scratch.router_logits, sc_buf, rw_buf,
                    scratch.topk_ids, scratch.topk_weights, scratch.router_done, nullptr);
            }
        };
        switch (num_sources) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 3: launch.template operator()<3>(); break;
        case 4: launch.template operator()<4>(); break;
        case 5: launch.template operator()<5>(); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const void* nw[1] = {mlp_norm_w.data};
    void* outs[1]     = {mlp_out.data};
    inkling2_attn_res_mix(source_ptrs, num_sources, query_f32, nw, outs, 1, hidden, tokens, eps,
                          stream);
    inkling2_moe_router_fused(mlp_out, router_weight, router_bias_f32, route_scale, scratch, stream,
                              prep_mode);
}

void inkling2_attn_res_loop_ctrl_moe_router_fused(
    const void* const* source_ptrs, std::int32_t num_sources, const Tensor& query_f32,
    const Tensor& msg, const Tensor& down_w, const Tensor& gain_forget_w,
    const Tensor& mlp_norm_w, const Tensor& router_weight, const Tensor& router_bias_f32,
    float route_scale, float eps, Tensor& scratch_cur, Tensor& scratch_z, Tensor& scratch_h,
    Tensor& scratch_gf, std::int32_t* ctrl_done, Tensor& mlp_out, Inkling2MoeScratch& scratch,
    cudaStream_t stream, std::int32_t prep_mode) {
    const int hidden = mlp_out.ne[0];
    const int tokens = mlp_out.ne[1];
    if (tokens == 1 && hidden == kHidden && num_sources >= 1 && num_sources <= 5 &&
        down_w.ne[1] == 64 && router_weight.ne[1] == kNumExperts && ctrl_done != nullptr &&
        scratch.router_done != nullptr && scratch.accum_f32 != nullptr &&
        scratch_gf.bytes() >= 4864 && get_moe_device_sm_count() >= 64) {
        MoeAttnResSources sa{};
        for (int s = 0; s < num_sources; ++s) {
            sa.sources[s] = static_cast<const __nv_bfloat16*>(source_ptrs[s]);
        }
        const auto* qp = static_cast<const float*>(query_f32.data);
        const auto* mp = static_cast<const __nv_bfloat16*>(msg.data);
        const auto* dp = static_cast<const __nv_bfloat16*>(down_w.data);
        const auto* gp = static_cast<const __nv_bfloat16*>(gain_forget_w.data);
        const auto* np = static_cast<const __nv_bfloat16*>(mlp_norm_w.data);
        const auto* wp = static_cast<const __nv_bfloat16*>(router_weight.data);
        const auto* bp = static_cast<const float*>(router_bias_f32.data);
        auto* cb       = static_cast<__nv_bfloat16*>(scratch_cur.data);
        auto* hb       = static_cast<__nv_bfloat16*>(scratch_h.data);
        auto* yb       = static_cast<__nv_bfloat16*>(scratch_gf.data);
        auto* psq      = reinterpret_cast<float*>(static_cast<char*>(scratch_gf.data) + 4608);
        auto* mo       = static_cast<__nv_bfloat16*>(mlp_out.data);
        auto* bar      = reinterpret_cast<unsigned int*>(ctrl_done);
        float* sc_buf  = scratch.accum_f32;
        float* rw_buf  = scratch.accum_f32 + kNumExperts;
        const int pm   = (prep_mode == 1 || prep_mode == 2) ? prep_mode : 0;
        half* x_staged = nullptr;
        if (pm != 0) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            x_staged = get_moe_x_scratch(dev, kHidden);
        }
        auto launch = [&]<int S>() {
            if (pm == 1) {
                attn_res_loop_ctrl_moe_router_t1_kernel<S, 1><<<64, 288, 0, stream>>>(
                    sa, qp, mp, dp, gp, np, wp, bp, route_scale, eps, cb, hb, yb, psq, mo, bar,
                    scratch.router_logits, sc_buf, rw_buf, scratch.topk_ids, scratch.topk_weights,
                    scratch.router_done, x_staged);
            } else if (pm == 2) {
                attn_res_loop_ctrl_moe_router_t1_kernel<S, 2><<<64, 288, 0, stream>>>(
                    sa, qp, mp, dp, gp, np, wp, bp, route_scale, eps, cb, hb, yb, psq, mo, bar,
                    scratch.router_logits, sc_buf, rw_buf, scratch.topk_ids, scratch.topk_weights,
                    scratch.router_done, x_staged);
            } else {
                attn_res_loop_ctrl_moe_router_t1_kernel<S, 0><<<64, 288, 0, stream>>>(
                    sa, qp, mp, dp, gp, np, wp, bp, route_scale, eps, cb, hb, yb, psq, mo, bar,
                    scratch.router_logits, sc_buf, rw_buf, scratch.topk_ids, scratch.topk_weights,
                    scratch.router_done, nullptr);
            }
        };
        switch (num_sources) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 3: launch.template operator()<3>(); break;
        case 4: launch.template operator()<4>(); break;
        case 5: launch.template operator()<5>(); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Fallback for T >= 2 (uses single-launch fused loop_control for T=2..8!) or < 64 SMs
    inkling2_attn_res_mix_and_prep(source_ptrs, num_sources, query_f32, msg, scratch_cur, scratch_z,
                                   eps, stream);
    inkling2_loop_control(scratch_cur, msg, down_w, gain_forget_w, &mlp_norm_w, scratch_z,
                          scratch_h, scratch_gf, mlp_out, eps, stream, /*skip_prep=*/true,
                          ctrl_done);
    inkling2_moe_router_fused(mlp_out, router_weight, router_bias_f32, route_scale, scratch, stream,
                              prep_mode);
}

void inkling2_moe_nvfp4_forward(const Tensor& x, const void* w13_codes, const void* w13_scales,
                                const float* w13_amax, const float* w13_denom,
                                const void* w2_codes, const void* w2_scales,
                                const float* w2_amax, const float* w2_denom,
                                std::int32_t num_slabs, float gate_cap, float up_cap,
                                Inkling2MoeScratch& scratch, Tensor& out, cudaStream_t stream,
                                bool skip_prep, bool skip_cast_out) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        const int inter_local = num_slabs * kInterPerSlab;
        const dim3 g1(inter_local / 16, kTopK);
        if (skip_prep) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            const half* x_staged = get_moe_x_scratch(dev, kHidden);
            moe_nvfp4_t1_w13_kernel<true><<<g1, 128, 0, stream>>>(
                x_staged, static_cast<const std::uint8_t*>(w13_codes),
                static_cast<const std::uint8_t*>(w13_scales), w13_amax, w13_denom, scratch.topk_ids,
                scratch.topk_weights, num_slabs, gate_cap, up_cap,
                static_cast<half*>(scratch.intermediate_bf16));
        } else {
            moe_nvfp4_t1_w13_kernel<false><<<g1, 128, 0, stream>>>(
                x.data, static_cast<const std::uint8_t*>(w13_codes),
                static_cast<const std::uint8_t*>(w13_scales), w13_amax, w13_denom, scratch.topk_ids,
                scratch.topk_weights, num_slabs, gate_cap, up_cap,
                static_cast<half*>(scratch.intermediate_bf16));
        }
        CUDA_CHECK(cudaGetLastError());

        moe_nvfp4_t1_w2_kernel<<<kHidden / 8, 128, 0, stream>>>(
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
    if (!skip_prep) {
        prestage_x_fp16_kernel<true><<<tokens, 288, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), x_staged, tokens);
        CUDA_CHECK(cudaGetLastError());
    }

    const int max_active        = std::min(kNumExperts, tokens * kTopK);
    const std::size_t out_elems = static_cast<std::size_t>(tokens) * kHidden;

    if (tokens <= 32) {
        const dim3 g1(num_slabs * (kInterPerSlab / 16), max_active);
        moe_nvfp4_grouped_w13_coalesced_kernel<<<g1, 128, 0, stream>>>(
            x_staged, static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), w13_amax, w13_denom,
            scratch.active_experts, scratch.num_active, scratch.expert_offsets,
            scratch.sorted_pairs, scratch.topk_weights, num_slabs, gate_cap, up_cap,
            static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        if (!skip_prep) {
            CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));
        }

        const dim3 g2(num_slabs * (kHidden / 128), max_active);
        moe_nvfp4_grouped_w2_coalesced_kernel<<<g2, 128, 0, stream>>>(
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

        if (!skip_prep) {
            CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));
        }

        const dim3 g2(num_slabs * (kHidden / 32), max_active);
        moe_nvfp4_grouped_w2_qpn_kernel<3><<<g2, 96, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), w2_amax, w2_denom, scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, num_slabs,
            scratch.accum_f32);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!skip_cast_out) {
        const std::size_t n4 = out_elems / 4;
        cast_f32_to_bf16_vec4_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
            scratch.accum_f32, static_cast<__nv_bfloat16*>(out.data), n4);
        CUDA_CHECK(cudaGetLastError());
    }
}

void inkling2_moe_mxfp8_forward(const Tensor& x, const void* w13_codes, const void* w13_scales,
                                const void* w2_codes, const void* w2_scales,
                                std::int32_t num_slabs, float gate_cap, float up_cap,
                                Inkling2MoeScratch& scratch, Tensor& out, cudaStream_t stream,
                                bool skip_prep, bool skip_cast_out) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        const int inter_local = num_slabs * kInterPerSlab;
        const dim3 g1(inter_local / 16, kTopK);
        if (skip_prep) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            const half* x_fp16 = get_moe_x_scratch(dev, kHidden);
            moe_mxfp8_t1_w13_kernel<true><<<g1, 128, 0, stream>>>(
                x_fp16, static_cast<const std::uint8_t*>(w13_codes),
                static_cast<const std::uint8_t*>(w13_scales), scratch.topk_ids, scratch.topk_weights,
                num_slabs, gate_cap, up_cap, static_cast<half*>(scratch.intermediate_bf16));
        } else {
            moe_mxfp8_t1_w13_kernel<false><<<g1, 128, 0, stream>>>(
                x.data, static_cast<const std::uint8_t*>(w13_codes),
                static_cast<const std::uint8_t*>(w13_scales), scratch.topk_ids, scratch.topk_weights,
                num_slabs, gate_cap, up_cap, static_cast<half*>(scratch.intermediate_bf16));
        }
        CUDA_CHECK(cudaGetLastError());

        moe_mxfp8_t1_w2_kernel<<<kHidden / 8, 128, 0, stream>>>(
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
    if (!skip_prep) {
        prestage_x_fp16_kernel<false><<<tokens, 288, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), x_fp16, tokens);
        CUDA_CHECK(cudaGetLastError());
    }

    const int max_active        = std::min(kNumExperts, tokens * kTopK);
    const std::size_t out_elems = static_cast<std::size_t>(tokens) * kHidden;

    if (tokens <= 32) {
        const dim3 g1(num_slabs * (kInterPerSlab / 16), max_active);
        moe_mxfp8_grouped_w13_coalesced_kernel<<<g1, 128, 0, stream>>>(
            x_fp16, static_cast<const std::uint8_t*>(w13_codes),
            static_cast<const std::uint8_t*>(w13_scales), scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, scratch.topk_weights,
            num_slabs, gate_cap, up_cap, static_cast<half*>(scratch.intermediate_bf16));
        CUDA_CHECK(cudaGetLastError());

        if (!skip_prep) {
            CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));
        }

        const dim3 g2(num_slabs * (kHidden / 128), max_active);
        moe_mxfp8_grouped_w2_coalesced_kernel<<<g2, 128, 0, stream>>>(
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

        if (!skip_prep) {
            CUDA_CHECK(cudaMemsetAsync(scratch.accum_f32, 0, out_elems * sizeof(float), stream));
        }

        const dim3 g2(num_slabs * (kHidden / 32), max_active);
        moe_mxfp8_grouped_w2_qpn_kernel<3><<<g2, 96, 0, stream>>>(
            static_cast<const half*>(scratch.intermediate_bf16),
            static_cast<const std::uint8_t*>(w2_codes),
            static_cast<const std::uint8_t*>(w2_scales), scratch.active_experts,
            scratch.num_active, scratch.expert_offsets, scratch.sorted_pairs, num_slabs,
            scratch.accum_f32);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!skip_cast_out) {
        const std::size_t n4 = out_elems / 4;
        cast_f32_to_bf16_vec4_kernel<<<(n4 + 255) / 256, 256, 0, stream>>>(
            scratch.accum_f32, static_cast<__nv_bfloat16*>(out.data), n4);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops
