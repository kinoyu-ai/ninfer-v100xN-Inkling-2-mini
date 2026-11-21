#include "ninfer/ops/inkling2.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

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

// ============================================================================
// 1. Plain RMSNorm (NO +1 offset)
// ============================================================================

// Fast specialized path for H == 2304 (288 threads = 9 warps, 8 elements/thread = 1 uint4!)
__global__ void inkling2_rmsnorm_h2304_kernel(const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ w,
                                              __nv_bfloat16* __restrict__ out,
                                              int tokens, float eps) {
    __shared__ float s_warp[9];
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;    // 0..8
    const int lane = tid & 31;
    const int idx  = tid * 8;

    float v[8];
    load_bf16x8(x + static_cast<std::int64_t>(row) * 2304 + idx, v);
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

    __nv_bfloat16* orow = out + static_cast<std::int64_t>(row) * 2304 + idx;
    if (w != nullptr) {
        float wv[8];
        load_bf16x8(w + idx, wv);
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = v[j] * inv * wv[j]; }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] *= inv; }
    }
    store_bf16x8(orow, v);
}

__global__ void inkling2_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ out,
                                        int hidden, int tokens, float eps) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid = threadIdx.x;

    const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(row) * hidden;
    __nv_bfloat16* orow       = out + static_cast<std::int64_t>(row) * hidden;

    float sq = 0.0f;
    for (int i = tid; i < hidden; i += 256) {
        const float v = __bfloat162float(xrow[i]);
        sq            = fmaf(v, v, sq);
    }
    const float inv_rms = rsqrtf(block_sum_256(sq) / static_cast<float>(hidden) + eps);
    for (int i = tid; i < hidden; i += 256) {
        float v = __bfloat162float(xrow[i]) * inv_rms;
        if (w != nullptr) { v *= __bfloat162float(__ldg(w + i)); }
        orow[i] = __float2bfloat16_rn(v);
    }
}

// ============================================================================
// 2. Vocab-sharded embedding gather
// ============================================================================

__global__ void inkling2_embed_gather_shard_kernel(const std::int32_t* __restrict__ ids,
                                                   const __nv_bfloat16* __restrict__ table,
                                                   __nv_bfloat16* __restrict__ out,
                                                   int hidden, int tokens, int vocab_offset,
                                                   int vocab_count) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tok      = ids[row];
    const int local    = tok - vocab_offset;
    __nv_bfloat16* dst = out + static_cast<std::int64_t>(row) * hidden;

    if (local >= 0 && local < vocab_count) {
        const __nv_bfloat16* src = table + static_cast<std::int64_t>(local) * hidden;
        const int h8             = hidden >> 3;
        for (int i = threadIdx.x; i < h8; i += 256) {
            reinterpret_cast<uint4*>(dst)[i] = __ldg(reinterpret_cast<const uint4*>(src) + i);
        }
    } else {
        const int h8  = hidden >> 3;
        const uint4 z{0, 0, 0, 0};
        for (int i = threadIdx.x; i < h8; i += 256) { reinterpret_cast<uint4*>(dst)[i] = z; }
    }
}

// ============================================================================
// 3. Block Attention Residuals (288 threads = 9 warps, 8 BF16 = uint4/thread!)
//    Half the warps & barrier latency of 576-thread version + 16-byte coalesced I/O!
// ============================================================================

struct AttnResArgs {
    const __nv_bfloat16* sources[5];
    const __nv_bfloat16* norm_w[4];
    __nv_bfloat16* outs[4];
};

template <int S, int Q>
__device__ __forceinline__ void attn_res_mix_h2304_body(
    const AttnResArgs& args, const float* __restrict__ queries, int row, float eps) {
    constexpr int kWarps = 9;
    __shared__ float s_red[20][kWarps];
    __shared__ float s_scalar[20];

    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;    // 0..8
    const int lane = tid & 31;
    const int idx  = tid * 8;     // 0..2296

    float v_reg[S][8];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        load_bf16x8(args.sources[s] + static_cast<std::int64_t>(row) * 2304 + idx, v_reg[s]);
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { sq = fmaf(v_reg[s][j], v_reg[s][j], sq); }
        sq = warp_sum(sq);
        if (lane == 0) { s_red[s][warp] = sq; }
    }
    __syncthreads();

    if (warp < S) {
        const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

#pragma unroll
    for (int s = 0; s < S; ++s) {
        const float inv = s_scalar[s];
#pragma unroll
        for (int j = 0; j < 8; ++j) { v_reg[s][j] *= inv; }
    }

#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float* qp = queries + static_cast<std::int64_t>(q) * 2304 + idx;
        const float4 q0 = __ldg(reinterpret_cast<const float4*>(qp));
        const float4 q1 = __ldg(reinterpret_cast<const float4*>(qp + 4));
#pragma unroll
        for (int s = 0; s < S; ++s) {
            float dot = fmaf(q0.x, v_reg[s][0],
                             fmaf(q0.y, v_reg[s][1], fmaf(q0.z, v_reg[s][2], q0.w * v_reg[s][3])));
            dot = fmaf(q1.x, v_reg[s][4],
                       fmaf(q1.y, v_reg[s][5], fmaf(q1.z, v_reg[s][6], fmaf(q1.w, v_reg[s][7], dot))));
            dot = warp_sum(dot);
            if (lane == 0) { s_red[q * S + s][warp] = dot; }
        }
    }
    __syncthreads();

    constexpr float kScale = 0.020833333333333332f; // 1 / 48 exact
    for (int qs = warp; qs < Q * S; qs += kWarps) {
        const float dot = warp_sum((lane < kWarps) ? s_red[qs][lane] : 0.0f);
        if (lane == 0) { s_scalar[qs] = dot * kScale; }
    }
    __syncthreads();

    float probs[Q][S];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        float mx = -1e30f;
#pragma unroll
        for (int s = 0; s < S; ++s) { mx = fmaxf(mx, s_scalar[q * S + s]); }
        float den = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float p = expf(s_scalar[q * S + s] - mx);
            probs[q][s]   = p;
            den += p;
        }
        const float inv_d = 1.0f / den;
#pragma unroll
        for (int s = 0; s < S; ++s) { probs[q][s] *= inv_d; }
    }

    const bool has_post_norm = (args.norm_w[0] != nullptr);
    if (!has_post_norm) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            float o[8] = {};
#pragma unroll
            for (int s = 0; s < S; ++s) {
                const float p = probs[q][s];
#pragma unroll
                for (int j = 0; j < 8; ++j) { o[j] = fmaf(p, v_reg[s][j], o[j]); }
            }
            store_bf16x8(args.outs[q] + static_cast<std::int64_t>(row) * 2304 + idx, o);
        }
    } else {
        float mixed[Q][8];
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            float sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < S; ++s) { acc = fmaf(probs[q][s], v_reg[s][j], acc); }
                mixed[q][j] = acc;
                sq          = fmaf(acc, acc, sq);
            }
            sq = warp_sum(sq);
            if (lane == 0) { s_red[q][warp] = sq; }
        }
        __syncthreads();
        if (warp < Q) {
            const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float inv = s_scalar[q];
            float nw[8];
            load_bf16x8(args.norm_w[q] + idx, nw);
#pragma unroll
            for (int j = 0; j < 8; ++j) { mixed[q][j] = mixed[q][j] * inv * nw[j]; }
            store_bf16x8(args.outs[q] + static_cast<std::int64_t>(row) * 2304 + idx, mixed[q]);
        }
    }
}

template <int S, int Q>
__global__ __launch_bounds__(288, 2) void inkling2_attn_res_mix_h2304_kernel(
    AttnResArgs args, const float* __restrict__ queries, int tokens, float eps) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    attn_res_mix_h2304_body<S, Q>(args, queries, row, eps);
}

// Paired Cross-Pass (S=2, Q=4) + Pre-Attn (S1, Q1) in ONE launch (grid = (tokens, 2))!
template <int S1, int Q1>
__global__ __launch_bounds__(288, 2) void inkling2_attn_res_mix_pair_h2304_kernel(
    AttnResArgs a0, const float* __restrict__ q0, AttnResArgs a1, const float* __restrict__ q1,
    int tokens, float eps) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    if (blockIdx.y == 0) {
        attn_res_mix_h2304_body<2, 4>(a0, q0, row, eps);
    } else {
        attn_res_mix_h2304_body<S1, Q1>(a1, q1, row, eps);
    }
}

// Generic fallback for non-2304 hidden sizes (unit tests)
template <int S, int Q>
__global__ void inkling2_attn_res_mix_kernel(AttnResArgs args, const float* __restrict__ queries,
                                             int hidden, int tokens, float eps) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid = threadIdx.x;
    constexpr int kMaxE = 16;
    const int n_e       = (hidden + 255) / 256;

    float v_reg[S][kMaxE];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const __nv_bfloat16* src = args.sources[s] + static_cast<std::int64_t>(row) * hidden;
        float sq                 = 0.0f;
#pragma unroll
        for (int e = 0; e < kMaxE; ++e) {
            const int idx = tid + e * 256;
            if (idx < hidden) {
                const float val = __bfloat162float(__ldg(src + idx));
                v_reg[s][e]     = val;
                sq              = fmaf(val, val, sq);
            } else {
                v_reg[s][e] = 0.0f;
            }
        }
        const float inv = rsqrtf(block_sum_256(sq) / static_cast<float>(hidden) + eps);
#pragma unroll
        for (int e = 0; e < kMaxE; ++e) { v_reg[s][e] *= inv; }
    }

    const float scale = rsqrtf(static_cast<float>(hidden));

#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float* qrow = queries + static_cast<std::int64_t>(q) * hidden;
        float scores[S];
        float max_s = -1e30f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            float dot = 0.0f;
            for (int e = 0; e < n_e; ++e) {
                const int idx = tid + e * 256;
                if (idx < hidden) { dot = fmaf(__ldg(qrow + idx), v_reg[s][e], dot); }
            }
            const float sc = block_sum_256(dot) * scale;
            scores[s]      = sc;
            max_s          = fmaxf(max_s, sc);
        }

        float denom = 0.0f;
        float probs[S];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            probs[s] = expf(scores[s] - max_s);
            denom += probs[s];
        }
        const float inv_denom = 1.0f / denom;
#pragma unroll
        for (int s = 0; s < S; ++s) { probs[s] *= inv_denom; }

        const __nv_bfloat16* nw = args.norm_w[q];
        __nv_bfloat16* dst      = args.outs[q] + static_cast<std::int64_t>(row) * hidden;

        if (nw == nullptr) {
            for (int e = 0; e < n_e; ++e) {
                const int idx = tid + e * 256;
                if (idx < hidden) {
                    float acc = 0.0f;
#pragma unroll
                    for (int s = 0; s < S; ++s) { acc = fmaf(probs[s], v_reg[s][e], acc); }
                    dst[idx] = __float2bfloat16_rn(acc);
                }
            }
        } else {
            float out_r[kMaxE];
            float sq = 0.0f;
            for (int e = 0; e < n_e; ++e) {
                const int idx = tid + e * 256;
                if (idx < hidden) {
                    float acc = 0.0f;
#pragma unroll
                    for (int s = 0; s < S; ++s) { acc = fmaf(probs[s], v_reg[s][e], acc); }
                    out_r[e] = acc;
                    sq       = fmaf(acc, acc, sq);
                }
            }
            const float inv_rms = rsqrtf(block_sum_256(sq) / static_cast<float>(hidden) + eps);
            for (int e = 0; e < n_e; ++e) {
                const int idx = tid + e * 256;
                if (idx < hidden) {
                    dst[idx] = __float2bfloat16_rn(out_r[e] * inv_rms * __bfloat162float(__ldg(nw + idx)));
                }
            }
        }
    }
}

// ============================================================================
// 4. Ultra-Fast Cross-Pass Loop Controller (H = 2304, Rank = 64)
//    - Prep: 576 threads/token (uint2 loads/stores, 1 single barrier for both cur & msg!)
//    - Down: [4608 -> 64] via multi-warp Split-K BF16 linear (64 CTAs saturating V100!)
//    - Up + Blend (+ partial sq_sum for RMSNorm): 72 CTAs reading gf_w [4608, 64] ONCE
//      for up to 8 tokens, ZERO gf HBM buffer, full 80-SM HBM bandwidth!
//    - Post-RMSNorm finish (when post_norm_w != nullptr): 1 tiny 576-thread pass using
//      the 72 precomputed partial sq sums!
// ============================================================================

__global__ void loop_control_prep_h2304_kernel(const __nv_bfloat16* __restrict__ cur,
                                               const __nv_bfloat16* __restrict__ msg,
                                               __nv_bfloat16* __restrict__ z,
                                               int tokens, float eps) {
    __shared__ float s_red[2][18];
    __shared__ float s_inv[2];

    const int row  = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..575
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 4;

    const float4 vc = load_bf16x4(cur + static_cast<std::int64_t>(row) * 2304 + idx);
    const float4 vm = load_bf16x4(msg + static_cast<std::int64_t>(row) * 2304 + idx);
    const float sqc = warp_sum(vc.x * vc.x + vc.y * vc.y + vc.z * vc.z + vc.w * vc.w);
    const float sqm = warp_sum(vm.x * vm.x + vm.y * vm.y + vm.z * vm.z + vm.w * vm.w);
    if (lane == 0) {
        s_red[0][warp] = sqc;
        s_red[1][warp] = sqm;
    }
    __syncthreads();
    if (warp < 2) {
        const float sum = warp_sum((lane < 18) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_inv[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    const float ic      = s_inv[0];
    const float im      = s_inv[1];
    __nv_bfloat16* zrow = z + static_cast<std::int64_t>(row) * 4608;
    store_bf16x4(zrow + idx, vc.x * ic, vc.y * ic, vc.z * ic, vc.w * ic);
    store_bf16x4(zrow + 2304 + idx, vm.x * im, vm.y * im, vm.z * im, vm.w * im);
}

// Fused Gain/Forget Up-Proj (K=64) + Tanh Blend + Optional Partial SqSum!
// Processes up to 8 tokens per CTA (grid.x = 72 -> 32 channels/CTA, grid.y = ceil(T/8)).
// Each group of 4 lanes computes 1 channel i: reads gf_w[i, 0..63] & gf_w[2304+i, 0..63] ONCE
// (2 uint4 = 16 BF16 per thread -> 100% coalesced 128-byte cache lines!) and contracts with
// ALL T_blk <= 8 tokens held in shared memory s_h[8][64]!
template <int MaxTB>
__global__ __launch_bounds__(128, 4) void loop_control_up_blend_k64_kernel(
    const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ gf_w,
    const __nv_bfloat16* __restrict__ cur, const __nv_bfloat16* __restrict__ msg,
    __nv_bfloat16* __restrict__ out, float* __restrict__ partial_sq, int tokens) {
    __shared__ float s_h[MaxTB][64];
    __shared__ float s_sq[MaxTB][4];

    const int tid    = threadIdx.x; // 0..127
    const int warp   = tid >> 5;    // 0..3
    const int lane   = tid & 31;
    const int ch_loc = tid >> 2;    // 0..31 (32 channels per block)
    const int sub    = tid & 3;     // 0..3 (4 threads per channel, 16 K elements each)
    const int ch     = blockIdx.x * 32 + ch_loc; // 0..2303
    const int t_base = blockIdx.y * MaxTB;
    const int t_cnt  = min(MaxTB, tokens - t_base);

    // Load h[t_base..t_base+t_cnt-1, 0..63] into shared memory
    for (int e = tid; e < t_cnt * 64; e += 128) {
        const int lt = e >> 6;
        const int k  = e & 63;
        s_h[lt][k]   = __bfloat162float(__ldg(h + static_cast<std::int64_t>(t_base + lt) * 64 + k));
    }
    __syncthreads();

    // Load 16 gain weights and 16 forget weights ONCE for all t_cnt tokens (streaming .cg)!
    const __nv_bfloat16* gw_ptr = gf_w + static_cast<std::int64_t>(ch) * 64 + sub * 16;
    const __nv_bfloat16* fw_ptr = gf_w + static_cast<std::int64_t>(2304 + ch) * 64 + sub * 16;
    const uint4 gw0             = ld_global_cg_u4(gw_ptr);
    const uint4 gw1             = ld_global_cg_u4(gw_ptr + 8);
    const uint4 fw0             = ld_global_cg_u4(fw_ptr);
    const uint4 fw1             = ld_global_cg_u4(fw_ptr + 8);

    float gwf[16], fwf[16];
    const auto* gb0 = reinterpret_cast<const __nv_bfloat162*>(&gw0);
    const auto* gb1 = reinterpret_cast<const __nv_bfloat162*>(&gw1);
    const auto* fb0 = reinterpret_cast<const __nv_bfloat162*>(&fw0);
    const auto* fb1 = reinterpret_cast<const __nv_bfloat162*>(&fw1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 g0  = __bfloat1622float2(gb0[j]);
        const float2 g1  = __bfloat1622float2(gb1[j]);
        const float2 f0  = __bfloat1622float2(fb0[j]);
        const float2 f1  = __bfloat1622float2(fb1[j]);
        gwf[2 * j]       = g0.x;
        gwf[2 * j + 1]   = g0.y;
        gwf[8 + 2 * j]   = g1.x;
        gwf[8 + 2 * j + 1] = g1.y;
        fwf[2 * j]       = f0.x;
        fwf[2 * j + 1]   = f0.y;
        fwf[8 + 2 * j]   = f1.x;
        fwf[8 + 2 * j + 1] = f1.y;
    }

    const int k_off = sub * 16;
#pragma unroll
    for (int lt = 0; lt < MaxTB; ++lt) {
        float y_sq = 0.0f;
        if (lt < t_cnt) {
            float g_acc = 0.0f;
            float f_acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float hv = s_h[lt][k_off + j];
                g_acc          = fmaf(gwf[j], hv, g_acc);
                f_acc          = fmaf(fwf[j], hv, f_acc);
            }
            // Reduce across the 4 threads owning this channel
            g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 1, 4);
            g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 2, 4);
            f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 1, 4);
            f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 2, 4);

            if (sub == 0) {
                const std::int64_t off = static_cast<std::int64_t>(t_base + lt) * 2304 + ch;
                const float c          = __bfloat162float(__ldg(cur + off));
                const float m          = __bfloat162float(__ldg(msg + off));
                const float y          = (1.0f - tanhf(f_acc)) * c + tanhf(g_acc) * m;
                out[off]               = __float2bfloat16_rn(y);
                y_sq                   = y * y;
            }
        }
        if (partial_sq != nullptr && lt < t_cnt) {
            const float wsum = warp_sum(y_sq);
            if (lane == 0) { s_sq[lt][warp] = wsum; }
        }
    }

    if (partial_sq != nullptr) {
        __syncthreads();
        if (tid < t_cnt) {
            const float tot = s_sq[tid][0] + s_sq[tid][1] + s_sq[tid][2] + s_sq[tid][3];
            partial_sq[static_cast<std::int64_t>(t_base + tid) * 72 + blockIdx.x] = tot;
        }
    }
}

// Fast post-RMSNorm finish using the 72 precomputed partial sq sums (<<<T, 576>>>)
__global__ void loop_control_post_norm_h2304_kernel(__nv_bfloat16* __restrict__ out,
                                                    const __nv_bfloat16* __restrict__ norm_w,
                                                    const float* __restrict__ partial_sq,
                                                    int tokens, float eps) {
    __shared__ float s_w[3];
    __shared__ float s_inv;

    const int row  = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..575
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (warp < 3) {
        const int p   = warp * 32 + lane;
        const float v = (p < 72) ? __ldg(partial_sq + static_cast<std::int64_t>(row) * 72 + p) : 0.0f;
        const float w = warp_sum(v);
        if (lane == 0) { s_w[warp] = w; }
    }
    __syncthreads();
    if (tid == 0) {
        const float total_sq = s_w[0] + s_w[1] + s_w[2];
        s_inv                = rsqrtf(total_sq * (1.0f / 2304.0f) + eps);
    }
    __syncthreads();

    const float inv     = s_inv;
    const int idx       = tid * 4;
    __nv_bfloat16* orow = out + static_cast<std::int64_t>(row) * 2304 + idx;
    const float4 v      = load_bf16x4(orow);
    const float4 nw     = load_bf16x4(norm_w + idx);
    store_bf16x4(orow, v.x * inv * nw.x, v.y * inv * nw.y, v.z * inv * nw.z, v.w * inv * nw.w);
}

// ============================================================================
// 4b. Batched 3-Controller (Q, K, V) Loop Control Kernels (grid.z = 3!)
//     Runs all 3 pre-attention controllers simultaneously -> 192 fewer launches/step
//     and 216 CTAs (2.7 full waves on 80 SMs) instead of 3 serial 72-CTA waves!
// ============================================================================

struct LoopCtrlQkvArgs {
    const __nv_bfloat16* cur[3];
    const __nv_bfloat16* msg[3];
    const __nv_bfloat16* down[3];
    const __nv_bfloat16* gf[3];
    const __nv_bfloat16* norm[3];
    __nv_bfloat16* out[3];
};

__global__ void loop_control_prep_qkv_h2304_kernel(LoopCtrlQkvArgs args,
                                                   __nv_bfloat16* __restrict__ z_all,
                                                   int tokens, float eps) {
    __shared__ float s_red[2][18];
    __shared__ float s_inv[2];

    const int row  = blockIdx.x;
    const int c    = blockIdx.y; // 0, 1, 2
    if (row >= tokens || c >= 3) { return; }
    const int tid  = threadIdx.x; // 0..575
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 4;

    const float4 vc = load_bf16x4(args.cur[c] + static_cast<std::int64_t>(row) * 2304 + idx);
    const float4 vm = load_bf16x4(args.msg[c] + static_cast<std::int64_t>(row) * 2304 + idx);
    const float sqc = warp_sum(vc.x * vc.x + vc.y * vc.y + vc.z * vc.z + vc.w * vc.w);
    const float sqm = warp_sum(vm.x * vm.x + vm.y * vm.y + vm.z * vm.z + vm.w * vm.w);
    if (lane == 0) {
        s_red[0][warp] = sqc;
        s_red[1][warp] = sqm;
    }
    __syncthreads();
    if (warp < 2) {
        const float sum = warp_sum((lane < 18) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_inv[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    const float ic = s_inv[0];
    const float im = s_inv[1];
    __nv_bfloat16* zrow =
        z_all + (static_cast<std::int64_t>(c) * tokens + row) * 4608;
    store_bf16x4(zrow + idx, vc.x * ic, vc.y * ic, vc.z * ic, vc.w * ic);
    store_bf16x4(zrow + 2304 + idx, vm.x * im, vm.y * im, vm.z * im, vm.w * im);
}

// 192 CTAs (grid = (64, 3)): [4608 -> 64] down-projection for all 3 controllers simultaneously!
template <int ActiveT, int WarpsPerRow>
__global__ __launch_bounds__(WarpsPerRow * 32, 2) void loop_control_down_qkv_small_t_kernel(
    LoopCtrlQkvArgs args, const __nv_bfloat16* __restrict__ z_all,
    __nv_bfloat16* __restrict__ h_all) {
    __shared__ float smem[ActiveT][WarpsPerRow];
    const int row  = blockIdx.x; // 0..63
    const int c    = blockIdx.y; // 0..2
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __nv_bfloat16* z    = z_all + static_cast<std::int64_t>(c) * ActiveT * 4608;
    const __nv_bfloat16* wrow = args.down[c] + static_cast<std::int64_t>(row) * 4608;
    __nv_bfloat16* h_out      = h_all + static_cast<std::int64_t>(c) * ActiveT * 64;

    constexpr int kK8  = 4608 / 8; // 576
    float acc[ActiveT] = {};

    for (int i = threadIdx.x; i < kK8; i += WarpsPerRow * 32) {
        const uint4 wv = ld_global_cg_u4(reinterpret_cast<const uint4*>(wrow) + i);
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
                z + static_cast<std::int64_t>(ti) * 4608) + i);
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
        const int ti = lane;
        float sum    = 0.0f;
#pragma unroll
        for (int s = 0; s < WarpsPerRow; ++s) { sum += smem[ti][s]; }
        h_out[static_cast<std::int64_t>(ti) * 64 + row] = __float2bfloat16_rn(sum);
    }
}

template <int MaxTB>
__global__ __launch_bounds__(128, 4) void loop_control_up_blend_qkv_k64_kernel(
    LoopCtrlQkvArgs args, const __nv_bfloat16* __restrict__ h_all,
    float* __restrict__ partial_sq_all, int tokens) {
    __shared__ float s_h[MaxTB][64];
    __shared__ float s_sq[MaxTB][4];

    const int c      = blockIdx.z; // 0..2
    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int ch_loc = tid >> 2;
    const int sub    = tid & 3;
    const int ch     = blockIdx.x * 32 + ch_loc;
    const int t_base = blockIdx.y * MaxTB;
    const int t_cnt  = min(MaxTB, tokens - t_base);

    const __nv_bfloat16* h    = h_all + static_cast<std::int64_t>(c) * tokens * 64;
    const __nv_bfloat16* gf_w = args.gf[c];
    const __nv_bfloat16* cur  = args.cur[c];
    const __nv_bfloat16* msg  = args.msg[c];
    __nv_bfloat16* out        = args.out[c];
    const bool want_sq        = (args.norm[c] != nullptr);
    float* partial_sq         = want_sq ? (partial_sq_all + static_cast<std::int64_t>(c) * tokens * 72) : nullptr;

    for (int e = tid; e < t_cnt * 64; e += 128) {
        const int lt = e >> 6;
        const int k  = e & 63;
        s_h[lt][k]   = __bfloat162float(__ldg(h + static_cast<std::int64_t>(t_base + lt) * 64 + k));
    }
    __syncthreads();

    const __nv_bfloat16* gw_ptr = gf_w + static_cast<std::int64_t>(ch) * 64 + sub * 16;
    const __nv_bfloat16* fw_ptr = gf_w + static_cast<std::int64_t>(2304 + ch) * 64 + sub * 16;
    const uint4 gw0             = ld_global_cg_u4(gw_ptr);
    const uint4 gw1             = ld_global_cg_u4(gw_ptr + 8);
    const uint4 fw0             = ld_global_cg_u4(fw_ptr);
    const uint4 fw1             = ld_global_cg_u4(fw_ptr + 8);

    float gwf[16], fwf[16];
    const auto* gb0 = reinterpret_cast<const __nv_bfloat162*>(&gw0);
    const auto* gb1 = reinterpret_cast<const __nv_bfloat162*>(&gw1);
    const auto* fb0 = reinterpret_cast<const __nv_bfloat162*>(&fw0);
    const auto* fb1 = reinterpret_cast<const __nv_bfloat162*>(&fw1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 g0    = __bfloat1622float2(gb0[j]);
        const float2 g1    = __bfloat1622float2(gb1[j]);
        const float2 f0    = __bfloat1622float2(fb0[j]);
        const float2 f1    = __bfloat1622float2(fb1[j]);
        gwf[2 * j]         = g0.x;
        gwf[2 * j + 1]     = g0.y;
        gwf[8 + 2 * j]     = g1.x;
        gwf[8 + 2 * j + 1] = g1.y;
        fwf[2 * j]         = f0.x;
        fwf[2 * j + 1]     = f0.y;
        fwf[8 + 2 * j]     = f1.x;
        fwf[8 + 2 * j + 1] = f1.y;
    }

    const int k_off = sub * 16;
#pragma unroll
    for (int lt = 0; lt < MaxTB; ++lt) {
        float y_sq = 0.0f;
        if (lt < t_cnt) {
            float g_acc = 0.0f;
            float f_acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float hv = s_h[lt][k_off + j];
                g_acc          = fmaf(gwf[j], hv, g_acc);
                f_acc          = fmaf(fwf[j], hv, f_acc);
            }
            g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 1, 4);
            g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 2, 4);
            f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 1, 4);
            f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 2, 4);

            if (sub == 0) {
                const std::int64_t off = static_cast<std::int64_t>(t_base + lt) * 2304 + ch;
                const float cv         = __bfloat162float(__ldg(cur + off));
                const float mv         = __bfloat162float(__ldg(msg + off));
                const float y          = (1.0f - tanhf(f_acc)) * cv + tanhf(g_acc) * mv;
                out[off]               = __float2bfloat16_rn(y);
                y_sq                   = y * y;
            }
        }
        if (want_sq && lt < t_cnt) {
            const float wsum = warp_sum(y_sq);
            if (lane == 0) { s_sq[lt][warp] = wsum; }
        }
    }

    if (want_sq) {
        __syncthreads();
        if (tid < t_cnt) {
            const float tot = s_sq[tid][0] + s_sq[tid][1] + s_sq[tid][2] + s_sq[tid][3];
            partial_sq[static_cast<std::int64_t>(t_base + tid) * 72 + blockIdx.x] = tot;
        }
    }
}

__global__ void loop_control_post_norm_qkv_h2304_kernel(LoopCtrlQkvArgs args,
                                                        const float* __restrict__ partial_sq_all,
                                                        int tokens, float eps) {
    __shared__ float s_w[3];
    __shared__ float s_inv;

    const int row  = blockIdx.x;
    const int c    = blockIdx.y; // 0..2
    if (row >= tokens || c >= 3 || args.norm[c] == nullptr) { return; }
    const int tid  = threadIdx.x; // 0..575
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float* partial_sq = partial_sq_all + static_cast<std::int64_t>(c) * tokens * 72;
    if (warp < 3) {
        const int p   = warp * 32 + lane;
        const float v = (p < 72) ? __ldg(partial_sq + static_cast<std::int64_t>(row) * 72 + p) : 0.0f;
        const float w = warp_sum(v);
        if (lane == 0) { s_w[warp] = w; }
    }
    __syncthreads();
    if (tid == 0) {
        const float total_sq = s_w[0] + s_w[1] + s_w[2];
        s_inv                = rsqrtf(total_sq * (1.0f / 2304.0f) + eps);
    }
    __syncthreads();

    const float inv     = s_inv;
    const int idx       = tid * 4;
    __nv_bfloat16* orow = args.out[c] + static_cast<std::int64_t>(row) * 2304 + idx;
    const float4 v      = load_bf16x4(orow);
    const float4 nw     = load_bf16x4(args.norm[c] + idx);
    store_bf16x4(orow, v.x * inv * nw.x, v.y * inv * nw.y, v.z * inv * nw.z, v.w * inv * nw.w);
}

// Generic fallbacks for non-2304 hidden sizes
__global__ void loop_control_prep_kernel(const __nv_bfloat16* __restrict__ cur,
                                         const __nv_bfloat16* __restrict__ msg,
                                         __nv_bfloat16* __restrict__ z,
                                         int hidden, int tokens, float eps) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid = threadIdx.x;

    const __nv_bfloat16* crow = cur + static_cast<std::int64_t>(row) * hidden;
    const __nv_bfloat16* mrow = msg + static_cast<std::int64_t>(row) * hidden;
    __nv_bfloat16* zrow       = z + static_cast<std::int64_t>(row) * (2 * hidden);

    float sq_c = 0.0f;
    float sq_m = 0.0f;
    for (int i = tid; i < hidden; i += 256) {
        const float vc = __bfloat162float(__ldg(crow + i));
        const float vm = __bfloat162float(__ldg(mrow + i));
        sq_c           = fmaf(vc, vc, sq_c);
        sq_m           = fmaf(vm, vm, sq_m);
    }
    const float inv_c = rsqrtf(block_sum_256(sq_c) / static_cast<float>(hidden) + eps);
    const float inv_m = rsqrtf(block_sum_256(sq_m) / static_cast<float>(hidden) + eps);
    for (int i = tid; i < hidden; i += 256) {
        zrow[i]          = __float2bfloat16_rn(__bfloat162float(__ldg(crow + i)) * inv_c);
        zrow[hidden + i] = __float2bfloat16_rn(__bfloat162float(__ldg(mrow + i)) * inv_m);
    }
}

__global__ void loop_control_blend_kernel(const __nv_bfloat16* __restrict__ cur,
                                          const __nv_bfloat16* __restrict__ msg,
                                          const __nv_bfloat16* __restrict__ gf,
                                          const __nv_bfloat16* __restrict__ norm_w,
                                          __nv_bfloat16* __restrict__ out,
                                          int hidden, int tokens, float eps) {
    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid = threadIdx.x;

    const __nv_bfloat16* crow = cur + static_cast<std::int64_t>(row) * hidden;
    const __nv_bfloat16* mrow = msg + static_cast<std::int64_t>(row) * hidden;
    const __nv_bfloat16* grow = gf + static_cast<std::int64_t>(row) * (2 * hidden);
    const __nv_bfloat16* frow = grow + hidden;
    __nv_bfloat16* orow       = out + static_cast<std::int64_t>(row) * hidden;

    if (norm_w == nullptr) {
        for (int i = tid; i < hidden; i += 256) {
            const float c = __bfloat162float(__ldg(crow + i));
            const float m = __bfloat162float(__ldg(mrow + i));
            const float g = tanhf(__bfloat162float(__ldg(grow + i)));
            const float f = tanhf(__bfloat162float(__ldg(frow + i)));
            orow[i]       = __float2bfloat16_rn((1.0f - f) * c + g * m);
        }
    } else {
        constexpr int kMaxE = 16;
        const int n_e       = (hidden + 255) / 256;
        float vals[kMaxE];
        float sq = 0.0f;
        for (int e = 0; e < n_e; ++e) {
            const int i = tid + e * 256;
            if (i < hidden) {
                const float c = __bfloat162float(__ldg(crow + i));
                const float m = __bfloat162float(__ldg(mrow + i));
                const float g = tanhf(__bfloat162float(__ldg(grow + i)));
                const float f = tanhf(__bfloat162float(__ldg(frow + i)));
                const float v = (1.0f - f) * c + g * m;
                vals[e]       = v;
                sq            = fmaf(v, v, sq);
            }
        }
        const float inv_rms = rsqrtf(block_sum_256(sq) / static_cast<float>(hidden) + eps);
        for (int e = 0; e < n_e; ++e) {
            const int i = tid + e * 256;
            if (i < hidden) {
                orow[i] = __float2bfloat16_rn(vals[e] * inv_rms * __bfloat162float(__ldg(norm_w + i)));
            }
        }
    }
}

template <int S>
void dispatch_attn_res_q(const AttnResArgs& args, const float* queries, int num_queries,
                         int hidden, int tokens, float eps, cudaStream_t stream) {
    if (hidden == 2304) {
        switch (num_queries) {
        case 1:
            inkling2_attn_res_mix_h2304_kernel<S, 1><<<tokens, 288, 0, stream>>>(args, queries, tokens, eps);
            return;
        case 3:
            inkling2_attn_res_mix_h2304_kernel<S, 3><<<tokens, 288, 0, stream>>>(args, queries, tokens, eps);
            return;
        case 4:
            inkling2_attn_res_mix_h2304_kernel<S, 4><<<tokens, 288, 0, stream>>>(args, queries, tokens, eps);
            return;
        default:
            throw std::invalid_argument("inkling2_attn_res_mix: num_queries must be 1, 3, or 4");
        }
    }
    switch (num_queries) {
    case 1:
        inkling2_attn_res_mix_kernel<S, 1><<<tokens, 256, 0, stream>>>(args, queries, hidden, tokens, eps);
        return;
    case 3:
        inkling2_attn_res_mix_kernel<S, 3><<<tokens, 256, 0, stream>>>(args, queries, hidden, tokens, eps);
        return;
    case 4:
        inkling2_attn_res_mix_kernel<S, 4><<<tokens, 256, 0, stream>>>(args, queries, hidden, tokens, eps);
        return;
    default:
        throw std::invalid_argument("inkling2_attn_res_mix: num_queries must be 1, 3, or 4");
    }
}

} // namespace

void inkling2_rmsnorm(const Tensor& x, const Tensor* weight, Tensor& out, float eps,
                      cudaStream_t stream) {
    const int hidden = x.ne[0];
    const int tokens = x.ne[1];
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || out.ne[0] != hidden ||
        out.ne[1] != tokens) {
        throw std::invalid_argument("inkling2_rmsnorm: shape/dtype mismatch");
    }
    if (hidden == 2304) {
        inkling2_rmsnorm_h2304_kernel<<<tokens, 288, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            weight ? static_cast<const __nv_bfloat16*>(weight->data) : nullptr,
            static_cast<__nv_bfloat16*>(out.data), tokens, eps);
    } else {
        inkling2_rmsnorm_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            weight ? static_cast<const __nv_bfloat16*>(weight->data) : nullptr,
            static_cast<__nv_bfloat16*>(out.data), hidden, tokens, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_embed_gather_shard(const Tensor& token_ids, const Tensor& table,
                                 std::int32_t vocab_offset, std::int32_t vocab_count,
                                 Tensor& out, cudaStream_t stream) {
    const int tokens = token_ids.ne[0];
    const int hidden = table.ne[0];
    if (token_ids.dtype != DType::I32 || table.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        out.ne[0] != hidden || out.ne[1] != tokens || (hidden % 8 != 0)) {
        throw std::invalid_argument("inkling2_embed_gather_shard: invalid args");
    }
    inkling2_embed_gather_shard_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(token_ids.data),
        static_cast<const __nv_bfloat16*>(table.data), static_cast<__nv_bfloat16*>(out.data),
        hidden, tokens, vocab_offset, vocab_count);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attn_res_mix(const void* const* source_ptrs, std::int32_t num_sources,
                           const Tensor& queries_f32, const void* const* post_norm_w_ptrs,
                           void* const* out_ptrs, std::int32_t num_queries, std::int32_t hidden,
                           std::int32_t tokens, float eps, cudaStream_t stream) {
    if (num_sources < 1 || num_sources > 5 || num_queries < 1 || num_queries > 4 ||
        queries_f32.dtype != DType::FP32 || hidden > 4096 || tokens <= 0) {
        throw std::invalid_argument("inkling2_attn_res_mix: invalid parameters");
    }
    AttnResArgs args{};
    for (int s = 0; s < num_sources; ++s) {
        args.sources[s] = static_cast<const __nv_bfloat16*>(source_ptrs[s]);
    }
    for (int q = 0; q < num_queries; ++q) {
        args.norm_w[q] = post_norm_w_ptrs ? static_cast<const __nv_bfloat16*>(post_norm_w_ptrs[q]) : nullptr;
        args.outs[q]   = static_cast<__nv_bfloat16*>(out_ptrs[q]);
    }
    const auto* qp = static_cast<const float*>(queries_f32.data);
    switch (num_sources) {
    case 1: dispatch_attn_res_q<1>(args, qp, num_queries, hidden, tokens, eps, stream); break;
    case 2: dispatch_attn_res_q<2>(args, qp, num_queries, hidden, tokens, eps, stream); break;
    case 3: dispatch_attn_res_q<3>(args, qp, num_queries, hidden, tokens, eps, stream); break;
    case 4: dispatch_attn_res_q<4>(args, qp, num_queries, hidden, tokens, eps, stream); break;
    case 5: dispatch_attn_res_q<5>(args, qp, num_queries, hidden, tokens, eps, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attn_res_mix_pair(const void* const* src0, std::int32_t num_src0,
                                const Tensor& q0_f32, void* const* out0, std::int32_t num_q0,
                                const void* const* src1, std::int32_t num_src1,
                                const Tensor& q1_f32, const void* const* norm1, void* const* out1,
                                std::int32_t num_q1, std::int32_t hidden, std::int32_t tokens,
                                float eps, cudaStream_t stream) {
    if (hidden == 2304 && num_src0 == 2 && num_q0 == 4 && num_src1 >= 1 && num_src1 <= 4 &&
        (num_q1 == 1 || num_q1 == 3) && norm1 == nullptr) {
        AttnResArgs a0{}, a1{};
        for (int s = 0; s < 2; ++s) { a0.sources[s] = static_cast<const __nv_bfloat16*>(src0[s]); }
        for (int q = 0; q < 4; ++q) { a0.outs[q] = static_cast<__nv_bfloat16*>(out0[q]); }
        for (int s = 0; s < num_src1; ++s) { a1.sources[s] = static_cast<const __nv_bfloat16*>(src1[s]); }
        for (int q = 0; q < num_q1; ++q) { a1.outs[q] = static_cast<__nv_bfloat16*>(out1[q]); }
        const auto* qp0 = static_cast<const float*>(q0_f32.data);
        const auto* qp1 = static_cast<const float*>(q1_f32.data);
        const dim3 grid(tokens, 2);
        auto launch = [&]<int S1>() {
            if (num_q1 == 1) {
                inkling2_attn_res_mix_pair_h2304_kernel<S1, 1><<<grid, 288, 0, stream>>>(a0, qp0, a1, qp1, tokens, eps);
            } else {
                inkling2_attn_res_mix_pair_h2304_kernel<S1, 3><<<grid, 288, 0, stream>>>(a0, qp0, a1, qp1, tokens, eps);
            }
        };
        switch (num_src1) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 3: launch.template operator()<3>(); break;
        case 4: launch.template operator()<4>(); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    inkling2_attn_res_mix(src0, num_src0, q0_f32, nullptr, out0, num_q0, hidden, tokens, eps, stream);
    inkling2_attn_res_mix(src1, num_src1, q1_f32, norm1, out1, num_q1, hidden, tokens, eps, stream);
}

void inkling2_loop_control(const Tensor& cur, const Tensor& msg, const Tensor& down_w,
                           const Tensor& gain_forget_w, const Tensor* post_norm_w,
                           Tensor& scratch_z, Tensor& scratch_h, Tensor& scratch_gf,
                           Tensor& out, float eps, cudaStream_t stream) {
    const int hidden = cur.ne[0];
    const int tokens = cur.ne[1];
    const int rank   = down_w.ne[1];

    if (hidden == 2304 && rank == 64 && tokens <= 32) {
        // Fast fused path for Inkling-2-Mini decode/verify!
        loop_control_prep_h2304_kernel<<<tokens, 576, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(cur.data),
            static_cast<const __nv_bfloat16*>(msg.data),
            static_cast<__nv_bfloat16*>(scratch_z.data), tokens, eps);
        CUDA_CHECK(cudaGetLastError());

        inkling2_bf16_linear(scratch_z, down_w, scratch_h, stream);

        // Reuse scratch_gf (which is >= 4608 * T * 2 B >> 72 * T * 4 B!) for 72 partial sq floats/token
        float* partial_sq = post_norm_w ? static_cast<float*>(scratch_gf.data) : nullptr;
        if (tokens == 1) {
            loop_control_up_blend_k64_kernel<1><<<dim3(72, 1), 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(scratch_h.data),
                static_cast<const __nv_bfloat16*>(gain_forget_w.data),
                static_cast<const __nv_bfloat16*>(cur.data),
                static_cast<const __nv_bfloat16*>(msg.data),
                static_cast<__nv_bfloat16*>(out.data), partial_sq, tokens);
        } else if (tokens <= 4) {
            loop_control_up_blend_k64_kernel<4><<<dim3(72, 1), 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(scratch_h.data),
                static_cast<const __nv_bfloat16*>(gain_forget_w.data),
                static_cast<const __nv_bfloat16*>(cur.data),
                static_cast<const __nv_bfloat16*>(msg.data),
                static_cast<__nv_bfloat16*>(out.data), partial_sq, tokens);
        } else {
            const dim3 grid(72, (tokens + 7) / 8);
            loop_control_up_blend_k64_kernel<8><<<grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(scratch_h.data),
                static_cast<const __nv_bfloat16*>(gain_forget_w.data),
                static_cast<const __nv_bfloat16*>(cur.data),
                static_cast<const __nv_bfloat16*>(msg.data),
                static_cast<__nv_bfloat16*>(out.data), partial_sq, tokens);
        }
        CUDA_CHECK(cudaGetLastError());

        if (post_norm_w != nullptr) {
            loop_control_post_norm_h2304_kernel<<<tokens, 576, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data),
                static_cast<const __nv_bfloat16*>(post_norm_w->data), partial_sq, tokens, eps);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Generic / prefill path (T > 32 uses fast tiled Tensor Core GEMM for down & gain_forget)
    if (hidden == 2304) {
        loop_control_prep_h2304_kernel<<<tokens, 576, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(cur.data),
            static_cast<const __nv_bfloat16*>(msg.data),
            static_cast<__nv_bfloat16*>(scratch_z.data), tokens, eps);
    } else {
        loop_control_prep_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(cur.data),
            static_cast<const __nv_bfloat16*>(msg.data),
            static_cast<__nv_bfloat16*>(scratch_z.data), hidden, tokens, eps);
    }
    CUDA_CHECK(cudaGetLastError());

    inkling2_bf16_linear(scratch_z, down_w, scratch_h, stream);
    inkling2_bf16_linear(scratch_h, gain_forget_w, scratch_gf, stream);

    loop_control_blend_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(cur.data),
        static_cast<const __nv_bfloat16*>(msg.data),
        static_cast<const __nv_bfloat16*>(scratch_gf.data),
        post_norm_w ? static_cast<const __nv_bfloat16*>(post_norm_w->data) : nullptr,
        static_cast<__nv_bfloat16*>(out.data), hidden, tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_loop_control_qkv(const Tensor& cur0, const Tensor& cur1, const Tensor& cur2,
                               const Tensor& msg0, const Tensor& msg1, const Tensor& msg2,
                               const Tensor& down0, const Tensor& down1, const Tensor& down2,
                               const Tensor& gf0, const Tensor& gf1, const Tensor& gf2,
                               const Tensor* norm0, const Tensor* norm1, const Tensor* norm2,
                               Tensor& scratch_z, Tensor& scratch_h, Tensor& scratch_gf,
                               Tensor& out0, Tensor& out1, Tensor& out2,
                               float eps, cudaStream_t stream) {
    const int hidden = cur0.ne[0];
    const int tokens = cur0.ne[1];
    const int rank   = down0.ne[1];

    if (hidden == 2304 && rank == 64 && tokens >= 1 && tokens <= 32) {
        LoopCtrlQkvArgs args{};
        args.cur[0]  = static_cast<const __nv_bfloat16*>(cur0.data);
        args.cur[1]  = static_cast<const __nv_bfloat16*>(cur1.data);
        args.cur[2]  = static_cast<const __nv_bfloat16*>(cur2.data);
        args.msg[0]  = static_cast<const __nv_bfloat16*>(msg0.data);
        args.msg[1]  = static_cast<const __nv_bfloat16*>(msg1.data);
        args.msg[2]  = static_cast<const __nv_bfloat16*>(msg2.data);
        args.down[0] = static_cast<const __nv_bfloat16*>(down0.data);
        args.down[1] = static_cast<const __nv_bfloat16*>(down1.data);
        args.down[2] = static_cast<const __nv_bfloat16*>(down2.data);
        args.gf[0]   = static_cast<const __nv_bfloat16*>(gf0.data);
        args.gf[1]   = static_cast<const __nv_bfloat16*>(gf1.data);
        args.gf[2]   = static_cast<const __nv_bfloat16*>(gf2.data);
        args.norm[0] = norm0 ? static_cast<const __nv_bfloat16*>(norm0->data) : nullptr;
        args.norm[1] = norm1 ? static_cast<const __nv_bfloat16*>(norm1->data) : nullptr;
        args.norm[2] = norm2 ? static_cast<const __nv_bfloat16*>(norm2->data) : nullptr;
        args.out[0]  = static_cast<__nv_bfloat16*>(out0.data);
        args.out[1]  = static_cast<__nv_bfloat16*>(out1.data);
        args.out[2]  = static_cast<__nv_bfloat16*>(out2.data);

        auto* z_all = static_cast<__nv_bfloat16*>(scratch_z.data);
        auto* h_all = static_cast<__nv_bfloat16*>(scratch_h.data);
        auto* psq   = static_cast<float*>(scratch_gf.data);

        // 1. Prep all 3 controllers in one launch (grid = (tokens, 3), 576 threads)
        loop_control_prep_qkv_h2304_kernel<<<dim3(tokens, 3), 576, 0, stream>>>(
            args, z_all, tokens, eps);
        CUDA_CHECK(cudaGetLastError());

        // 2. Down-project all 3 controllers in ONE launch!
        if (tokens <= 8) {
            const dim3 dgrid(64, 3);
            switch (tokens) {
            case 1: loop_control_down_qkv_small_t_kernel<1, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 2: loop_control_down_qkv_small_t_kernel<2, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 3: loop_control_down_qkv_small_t_kernel<3, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 4: loop_control_down_qkv_small_t_kernel<4, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 5: loop_control_down_qkv_small_t_kernel<5, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 6: loop_control_down_qkv_small_t_kernel<6, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 7: loop_control_down_qkv_small_t_kernel<7, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            case 8: loop_control_down_qkv_small_t_kernel<8, 8><<<dgrid, 256, 0, stream>>>(args, z_all, h_all); break;
            }
            CUDA_CHECK(cudaGetLastError());
        } else {
            Tensor z0(z_all, DType::BF16, {4608, tokens});
            Tensor z1(z_all + static_cast<std::int64_t>(tokens) * 4608, DType::BF16, {4608, tokens});
            Tensor z2(z_all + static_cast<std::int64_t>(2 * tokens) * 4608, DType::BF16, {4608, tokens});
            Tensor h0(h_all, DType::BF16, {64, tokens});
            Tensor h1(h_all + static_cast<std::int64_t>(tokens) * 64, DType::BF16, {64, tokens});
            Tensor h2(h_all + static_cast<std::int64_t>(2 * tokens) * 64, DType::BF16, {64, tokens});
            inkling2_bf16_linear_triple(z0, down0, h0, z1, down1, h1, z2, down2, h2, stream);
        }

        // 3. Up-project + tanh blend all 3 controllers in ONE 216-CTA launch!
        if (tokens == 1) {
            loop_control_up_blend_qkv_k64_kernel<1><<<dim3(72, 1, 3), 128, 0, stream>>>(
                args, h_all, psq, tokens);
        } else if (tokens <= 4) {
            loop_control_up_blend_qkv_k64_kernel<4><<<dim3(72, 1, 3), 128, 0, stream>>>(
                args, h_all, psq, tokens);
        } else {
            loop_control_up_blend_qkv_k64_kernel<8>
                <<<dim3(72, (tokens + 7) / 8, 3), 128, 0, stream>>>(args, h_all, psq, tokens);
        }
        CUDA_CHECK(cudaGetLastError());

        // 4. Post-RMSNorm all 3 controllers in ONE launch (576 threads)!
        if (norm0 != nullptr || norm1 != nullptr || norm2 != nullptr) {
            loop_control_post_norm_qkv_h2304_kernel<<<dim3(tokens, 3), 576, 0, stream>>>(
                args, psq, tokens, eps);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Fallback for prefill (T > 32) or non-standard shapes
    Tensor sz(scratch_z.data, DType::BF16, {2 * hidden, tokens});
    Tensor sh(scratch_h.data, DType::BF16, {rank, tokens});
    Tensor sg(scratch_gf.data, DType::BF16, {2 * hidden, tokens});
    inkling2_loop_control(cur0, msg0, down0, gf0, norm0, sz, sh, sg, out0, eps, stream);
    inkling2_loop_control(cur1, msg1, down1, gf1, norm1, sz, sh, sg, out1, eps, stream);
    inkling2_loop_control(cur2, msg2, down2, gf2, norm2, sz, sh, sg, out2, eps, stream);
}

} // namespace ninfer::ops
