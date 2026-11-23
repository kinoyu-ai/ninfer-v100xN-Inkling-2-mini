#include "ninfer/ops/inkling2.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <atomic>
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

__device__ __forceinline__ float4 ld_global_cg_f4(const void* ptr) {
    float4 val;
    asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        : "l"(ptr));
    return val;
}

__device__ __forceinline__ std::uint16_t ld_global_cg_u16(const void* ptr) {
    std::uint16_t val;
    asm("ld.global.cg.u16 %0, [%1];\n" : "=h"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ void st_global_cg_f32(float* ptr, float v) {
    asm volatile("st.global.cg.f32 [%0], %1;\n" :: "l"(ptr), "f"(v) : "memory");
}

__device__ __forceinline__ void st_global_cg_u16(void* ptr, std::uint16_t v) {
    asm volatile("st.global.cg.u16 [%0], %1;\n" :: "l"(ptr), "h"(v) : "memory");
}

// Single-word sense-reversing Volta PTX 6.0+ GPU-scope release-acquire grid barrier.
// Pre-zeroed once at startup; bit 31 self-reverses on each barrier (100% CUDA-Graph safe!).
// Caller MUST ensure gridDim.x <= sm_count (guaranteed co-resident without occupancy deadlock).
__device__ __forceinline__ void grid_barrier_n(unsigned int* bar_word, int G, int local_bid) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int delta =
            (local_bid == 0) ? (0x80000001u - static_cast<unsigned int>(G)) : 1u;
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

__device__ __forceinline__ float warp_reduce_72_f4(const float* __restrict__ arr72, int lane) {
    float v = 0.0f;
    if (lane < 18) {
        const float4 f = ld_global_cg_f4(arr72 + lane * 4);
        v              = (f.x + f.y) + (f.z + f.w);
    }
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    return v;
}

int get_current_device_sm_count() {
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
    constexpr int kTot   = S + Q * S; // <= 25 (S<=5, Q<=4)
    __shared__ float s_red[kTot][kWarps];
    __shared__ float s_val[kTot];

    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;    // 0..8
    const int lane = tid & 31;
    const int idx  = tid * 8;     // 0..2296

    // Single pass: compute BOTH sq[s] AND raw_dot[q][s] before ONE shared reduction (2 fewer barriers!)
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
            if (lane == 0) { s_red[S + q * S + s][warp] = dot; }
        }
    }
    __syncthreads();

    for (int k = warp; k < kTot; k += kWarps) {
        const float sum = warp_sum((lane < kWarps) ? s_red[k][lane] : 0.0f);
        if (lane == 0) { s_val[k] = sum; }
    }
    __syncthreads();

    constexpr float kScale = 0.020833333333333332f; // 1 / 48 exact
    float inv_s[S];
#pragma unroll
    for (int s = 0; s < S; ++s) { inv_s[s] = rsqrtf(s_val[s] * (1.0f / 2304.0f) + eps); }

    float ws[Q][S];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        float sc[S];
        float mx = -1e30f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            sc[s] = s_val[S + q * S + s] * inv_s[s] * kScale;
            mx    = fmaxf(mx, sc[s]);
        }
        float den = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float p = expf(sc[s] - mx);
            ws[q][s]      = p;
            den += p;
        }
        const float inv_d = 1.0f / den;
#pragma unroll
        for (int s = 0; s < S; ++s) { ws[q][s] = (ws[q][s] * inv_d) * inv_s[s]; }
    }

    const bool has_post_norm = (args.norm_w[0] != nullptr);
    if (!has_post_norm) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            float o[8] = {};
#pragma unroll
            for (int s = 0; s < S; ++s) {
                const float w = ws[q][s];
#pragma unroll
                for (int j = 0; j < 8; ++j) { o[j] = fmaf(w, v_reg[s][j], o[j]); }
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
                for (int s = 0; s < S; ++s) { acc = fmaf(ws[q][s], v_reg[s][j], acc); }
                mixed[q][j] = acc;
                sq          = fmaf(acc, acc, sq);
            }
            sq = warp_sum(sq);
            if (lane == 0) { s_red[q][warp] = sq; }
        }
        __syncthreads();
        if (warp < Q) {
            const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_val[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float inv = s_val[q];
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

// Split-Q parallel kernel for Q == 3: grid = (tokens, 3) -> 3x SM parallelism, 0 B stack!
template <int S>
__global__ __launch_bounds__(288, 2) void inkling2_attn_res_mix_h2304_splitq3_kernel(
    AttnResArgs args, const float* __restrict__ queries, int tokens, float eps) {
    constexpr int kWarps = 9;
    __shared__ float s_red[2 * S][kWarps];
    __shared__ float s_val[2 * S];

    const int row = blockIdx.x;
    const int q   = blockIdx.y; // 0, 1, 2
    if (row >= tokens || q >= 3) { return; }
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;

    const float* qp = queries + static_cast<std::int64_t>(q) * 2304 + idx;
    const float4 q0 = __ldg(reinterpret_cast<const float4*>(qp));
    const float4 q1 = __ldg(reinterpret_cast<const float4*>(qp + 4));

    float v_reg[S][8];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        load_bf16x8(args.sources[s] + static_cast<std::int64_t>(row) * 2304 + idx, v_reg[s]);
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { sq = fmaf(v_reg[s][j], v_reg[s][j], sq); }
        float dot = fmaf(q0.x, v_reg[s][0],
                         fmaf(q0.y, v_reg[s][1], fmaf(q0.z, v_reg[s][2], q0.w * v_reg[s][3])));
        dot = fmaf(q1.x, v_reg[s][4],
                   fmaf(q1.y, v_reg[s][5], fmaf(q1.z, v_reg[s][6], fmaf(q1.w, v_reg[s][7], dot))));
        sq  = warp_sum(sq);
        dot = warp_sum(dot);
        if (lane == 0) {
            s_red[s][warp]     = sq;
            s_red[S + s][warp] = dot;
        }
    }
    __syncthreads();

    for (int k = warp; k < 2 * S; k += kWarps) {
        const float sum = warp_sum((lane < kWarps) ? s_red[k][lane] : 0.0f);
        if (lane == 0) { s_val[k] = sum; }
    }
    __syncthreads();

    constexpr float kScale = 0.020833333333333332f;
    float inv_s[S], sc[S];
    float mx = -1e30f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        inv_s[s] = rsqrtf(s_val[s] * (1.0f / 2304.0f) + eps);
        sc[s]    = s_val[S + s] * inv_s[s] * kScale;
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

    // Ternary select avoids dynamic struct indexing -> 0 bytes stack frame!
    const __nv_bfloat16* nw_ptr = (q == 0) ? args.norm_w[0] : ((q == 1) ? args.norm_w[1] : args.norm_w[2]);
    __nv_bfloat16* out_ptr      = (q == 0) ? args.outs[0] : ((q == 1) ? args.outs[1] : args.outs[2]);

    float mixed[8];
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) { acc = fmaf(ws[s], v_reg[s][j], acc); }
        mixed[j] = acc;
        sq       = fmaf(acc, acc, sq);
    }
    if (nw_ptr == nullptr) {
        store_bf16x8(out_ptr + static_cast<std::int64_t>(row) * 2304 + idx, mixed);
        return;
    }
    sq = warp_sum(sq);
    if (lane == 0) { s_red[0][warp] = sq; }
    __syncthreads();
    if (warp == 0) {
        const float sum = warp_sum((lane < kWarps) ? s_red[0][lane] : 0.0f);
        if (lane == 0) { s_val[0] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    const float inv = s_val[0];
    float nw[8];
    load_bf16x8(nw_ptr + idx, nw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { mixed[j] = mixed[j] * inv * nw[j]; }
    store_bf16x8(out_ptr + static_cast<std::int64_t>(row) * 2304 + idx, mixed);
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

// ============================================================================
// 3b. FUSED AttnRes Mix + Loop Control Prep (eliminates 48 prep launches + HBM re-reads!)
//     Exact BF16 rounding boundary before computing inv_rms for z = [norm(cur), norm(msg)]!
// ============================================================================

template <int S>
__global__ __launch_bounds__(288, 2) void attn_res_mix_and_prep_h2304_kernel(
    AttnResArgs args, const float* __restrict__ query, const __nv_bfloat16* __restrict__ msg,
    __nv_bfloat16* __restrict__ out_mixed, __nv_bfloat16* __restrict__ out_z, int tokens,
    float eps) {
    constexpr int kWarps = 9;
    __shared__ float s_red[5][kWarps];
    __shared__ float s_scalar[5];

    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;    // 0..8
    const int lane = tid & 31;
    const int idx  = tid * 8;

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

    const float* qp = query + idx;
    const float4 q0 = __ldg(reinterpret_cast<const float4*>(qp));
    const float4 q1 = __ldg(reinterpret_cast<const float4*>(qp + 4));
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const float inv = s_scalar[s];
#pragma unroll
        for (int j = 0; j < 8; ++j) { v_reg[s][j] *= inv; }
        float dot = fmaf(q0.x, v_reg[s][0],
                         fmaf(q0.y, v_reg[s][1], fmaf(q0.z, v_reg[s][2], q0.w * v_reg[s][3])));
        dot = fmaf(q1.x, v_reg[s][4],
                   fmaf(q1.y, v_reg[s][5], fmaf(q1.z, v_reg[s][6], fmaf(q1.w, v_reg[s][7], dot))));
        dot = warp_sum(dot);
        if (lane == 0) { s_red[s][warp] = dot; }
    }
    __syncthreads();

    constexpr float kScale = 0.020833333333333332f;
    if (warp < S) {
        const float dot = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_scalar[warp] = dot * kScale; }
    }
    __syncthreads();

    float mx = -1e30f;
#pragma unroll
    for (int s = 0; s < S; ++s) { mx = fmaxf(mx, s_scalar[s]); }
    float probs[S];
    float den = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const float p = expf(s_scalar[s] - mx);
        probs[s]      = p;
        den += p;
    }
    const float inv_d = 1.0f / den;
#pragma unroll
    for (int s = 0; s < S; ++s) { probs[s] *= inv_d; }

    // Mix cur, store BF16 to out_mixed, and round to BF16 in registers for bit-identical prep!
    float cur_v[8];
    uint4 cur_raw;
    auto* cur_b = reinterpret_cast<__nv_bfloat162*>(&cur_raw);
    float sq_c  = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float a = 0.0f, b = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            a = fmaf(probs[s], v_reg[s][2 * j], a);
            b = fmaf(probs[s], v_reg[s][2 * j + 1], b);
        }
        cur_b[j]           = __floats2bfloat162_rn(a, b);
        const float2 bf    = __bfloat1622float2(cur_b[j]);
        cur_v[2 * j]       = bf.x;
        cur_v[2 * j + 1]   = bf.y;
        sq_c               = fmaf(bf.x, bf.x, fmaf(bf.y, bf.y, sq_c));
    }
    *reinterpret_cast<uint4*>(out_mixed + static_cast<std::int64_t>(row) * 2304 + idx) = cur_raw;

    // Load msg3 and compute BOTH sq_c and sq_m in ONE barrier!
    float msg_v[8];
    load_bf16x8(msg + static_cast<std::int64_t>(row) * 2304 + idx, msg_v);
    float sq_m = 0.0f;
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
        if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    const float ic = s_scalar[0];
    const float im = s_scalar[1];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        cur_v[j] *= ic;
        msg_v[j] *= im;
    }
    __nv_bfloat16* zrow = out_z + static_cast<std::int64_t>(row) * 4608;
    store_bf16x8(zrow + idx, cur_v);
    store_bf16x8(zrow + 2304 + idx, msg_v);
}

// Paired Cross-Pass AttnRes + QKV Loop Control Prep in ONE launch (grid = (tokens, 2), 288 threads):
//   blockIdx.y == 0: computes msgs[0..3], writes msgs[0..3], writes norm(msgs[0..2]) -> z_all[c, row, 2304..4607]
//   blockIdx.y == 1: computes cur_mix[0..Q1-1], writes cur_outs, writes norm(cur_mix) -> z_all[c, row, 0..2303]
template <int S1, int Q1>
__global__ __launch_bounds__(288, 2) void attn_res_mix_pair_and_prep_qkv_h2304_kernel(
    AttnResArgs a0, const float* __restrict__ q0, AttnResArgs a1, const float* __restrict__ q1,
    __nv_bfloat16* __restrict__ z_all, int tokens, float eps) {
    constexpr int kWarps = 9;
    __shared__ float s_red[16][kWarps];
    __shared__ float s_scalar[16];

    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..287
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;
    constexpr float kScale = 0.020833333333333332f;

    if (blockIdx.y == 0) {
        // S = 2, Q = 4: bank_srcs -> msgs[0..3] AND z_all[c=0..2, row, 2304..4607]
        float v_reg[2][8];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            load_bf16x8(a0.sources[s] + static_cast<std::int64_t>(row) * 2304 + idx, v_reg[s]);
            float sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) { sq = fmaf(v_reg[s][j], v_reg[s][j], sq); }
            sq = warp_sum(sq);
            if (lane == 0) { s_red[s][warp] = sq; }
        }
        __syncthreads();
        if (warp < 2) {
            const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const float inv = s_scalar[s];
#pragma unroll
            for (int j = 0; j < 8; ++j) { v_reg[s][j] *= inv; }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float* qp = q0 + static_cast<std::int64_t>(q) * 2304 + idx;
            const float4 qv0 = __ldg(reinterpret_cast<const float4*>(qp));
            const float4 qv1 = __ldg(reinterpret_cast<const float4*>(qp + 4));
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                float dot = fmaf(qv0.x, v_reg[s][0],
                                 fmaf(qv0.y, v_reg[s][1], fmaf(qv0.z, v_reg[s][2], qv0.w * v_reg[s][3])));
                dot = fmaf(qv1.x, v_reg[s][4],
                           fmaf(qv1.y, v_reg[s][5], fmaf(qv1.z, v_reg[s][6], fmaf(qv1.w, v_reg[s][7], dot))));
                dot = warp_sum(dot);
                if (lane == 0) { s_red[q * 2 + s][warp] = dot; }
            }
        }
        __syncthreads();
        if (warp < 8) {
            const float dot = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_scalar[warp] = dot * kScale; }
        }
        __syncthreads();

        float m_reg[3][8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float s0v = s_scalar[q * 2 + 0];
            const float s1v = s_scalar[q * 2 + 1];
            const float mx  = fmaxf(s0v, s1v);
            const float e0  = expf(s0v - mx);
            const float e1  = expf(s1v - mx);
            const float inv = 1.0f / (e0 + e1);
            const float p0  = e0 * inv;
            const float p1  = e1 * inv;

            uint4 raw;
            auto* b = reinterpret_cast<__nv_bfloat162*>(&raw);
            float sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float x0 = fmaf(p0, v_reg[0][2 * j], p1 * v_reg[1][2 * j]);
                const float x1 = fmaf(p0, v_reg[0][2 * j + 1], p1 * v_reg[1][2 * j + 1]);
                b[j]           = __floats2bfloat162_rn(x0, x1);
                if (q < 3) {
                    const float2 bf     = __bfloat1622float2(b[j]);
                    m_reg[q][2 * j]     = bf.x;
                    m_reg[q][2 * j + 1] = bf.y;
                    sq                  = fmaf(bf.x, bf.x, fmaf(bf.y, bf.y, sq));
                }
            }
            *reinterpret_cast<uint4*>(a0.outs[q] + static_cast<std::int64_t>(row) * 2304 + idx) = raw;
            if (q < 3) {
                sq = warp_sum(sq);
                if (lane == 0) { s_red[q][warp] = sq; }
            }
        }
        __syncthreads();
        if (warp < 3) {
            const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float im = s_scalar[c];
#pragma unroll
            for (int j = 0; j < 8; ++j) { m_reg[c][j] *= im; }
            __nv_bfloat16* zrow = z_all + (static_cast<std::int64_t>(c) * tokens + row) * 4608;
            store_bf16x8(zrow + 2304 + idx, m_reg[c]);
        }
    } else {
        // S = S1, Q = Q1 (1 or 3): cur_srcs -> cur_outs[0..Q1-1] AND z_all[c=0..2, row, 0..2303]
        float v_reg[S1][8];
#pragma unroll
        for (int s = 0; s < S1; ++s) {
            load_bf16x8(a1.sources[s] + static_cast<std::int64_t>(row) * 2304 + idx, v_reg[s]);
            float sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) { sq = fmaf(v_reg[s][j], v_reg[s][j], sq); }
            sq = warp_sum(sq);
            if (lane == 0) { s_red[s][warp] = sq; }
        }
        __syncthreads();
        if (warp < S1) {
            const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < S1; ++s) {
            const float inv = s_scalar[s];
#pragma unroll
            for (int j = 0; j < 8; ++j) { v_reg[s][j] *= inv; }
        }
#pragma unroll
        for (int q = 0; q < Q1; ++q) {
            const float* qp  = q1 + static_cast<std::int64_t>(q) * 2304 + idx;
            const float4 qv0 = __ldg(reinterpret_cast<const float4*>(qp));
            const float4 qv1 = __ldg(reinterpret_cast<const float4*>(qp + 4));
#pragma unroll
            for (int s = 0; s < S1; ++s) {
                float dot = fmaf(qv0.x, v_reg[s][0],
                                 fmaf(qv0.y, v_reg[s][1], fmaf(qv0.z, v_reg[s][2], qv0.w * v_reg[s][3])));
                dot = fmaf(qv1.x, v_reg[s][4],
                           fmaf(qv1.y, v_reg[s][5], fmaf(qv1.z, v_reg[s][6], fmaf(qv1.w, v_reg[s][7], dot))));
                dot = warp_sum(dot);
                if (lane == 0) { s_red[q * S1 + s][warp] = dot; }
            }
        }
        __syncthreads();
        for (int qs = warp; qs < Q1 * S1; qs += kWarps) {
            const float dot = warp_sum((lane < kWarps) ? s_red[qs][lane] : 0.0f);
            if (lane == 0) { s_scalar[qs] = dot * kScale; }
        }
        __syncthreads();

        float c_reg[Q1][8];
#pragma unroll
        for (int q = 0; q < Q1; ++q) {
            float mx = -1e30f;
#pragma unroll
            for (int s = 0; s < S1; ++s) { mx = fmaxf(mx, s_scalar[q * S1 + s]); }
            float probs[S1];
            float den = 0.0f;
#pragma unroll
            for (int s = 0; s < S1; ++s) {
                const float p = expf(s_scalar[q * S1 + s] - mx);
                probs[s]      = p;
                den += p;
            }
            const float inv_d = 1.0f / den;
#pragma unroll
            for (int s = 0; s < S1; ++s) { probs[s] *= inv_d; }

            uint4 raw;
            auto* b  = reinterpret_cast<__nv_bfloat162*>(&raw);
            float sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float x0 = 0.0f, x1 = 0.0f;
#pragma unroll
                for (int s = 0; s < S1; ++s) {
                    x0 = fmaf(probs[s], v_reg[s][2 * j], x0);
                    x1 = fmaf(probs[s], v_reg[s][2 * j + 1], x1);
                }
                b[j]                = __floats2bfloat162_rn(x0, x1);
                const float2 bf     = __bfloat1622float2(b[j]);
                c_reg[q][2 * j]     = bf.x;
                c_reg[q][2 * j + 1] = bf.y;
                sq                  = fmaf(bf.x, bf.x, fmaf(bf.y, bf.y, sq));
            }
            *reinterpret_cast<uint4*>(a1.outs[q] + static_cast<std::int64_t>(row) * 2304 + idx) = raw;
            sq = warp_sum(sq);
            if (lane == 0) { s_red[q][warp] = sq; }
        }
        __syncthreads();
        if (warp < Q1) {
            const float sum = warp_sum((lane < kWarps) ? s_red[warp][lane] : 0.0f);
            if (lane == 0) { s_scalar[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
        }
        __syncthreads();

        if constexpr (Q1 == 1) {
            const float ic = s_scalar[0];
#pragma unroll
            for (int j = 0; j < 8; ++j) { c_reg[0][j] *= ic; }
            uint4 zpack;
            auto* zb = reinterpret_cast<__nv_bfloat162*>(&zpack);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                zb[j] = __floats2bfloat162_rn(c_reg[0][2 * j], c_reg[0][2 * j + 1]);
            }
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                __nv_bfloat16* zrow = z_all + (static_cast<std::int64_t>(c) * tokens + row) * 4608;
                *reinterpret_cast<uint4*>(zrow + idx) = zpack;
            }
        } else {
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float ic = s_scalar[c];
#pragma unroll
                for (int j = 0; j < 8; ++j) { c_reg[c][j] *= ic; }
                __nv_bfloat16* zrow = z_all + (static_cast<std::int64_t>(c) * tokens + row) * 4608;
                store_bf16x8(zrow + idx, c_reg[c]);
            }
        }
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
    __shared__ float s_red[2][9];
    __shared__ float s_inv[2];

    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..287 (9 warps, 8 BF16/thread)
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;

    float vc[8], vm[8];
    load_bf16x8(cur + static_cast<std::int64_t>(row) * 2304 + idx, vc);
    load_bf16x8(msg + static_cast<std::int64_t>(row) * 2304 + idx, vm);
    float sqc = 0.0f, sqm = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        sqc = fmaf(vc[j], vc[j], sqc);
        sqm = fmaf(vm[j], vm[j], sqm);
    }
    sqc = warp_sum(sqc);
    sqm = warp_sum(sqm);
    if (lane == 0) {
        s_red[0][warp] = sqc;
        s_red[1][warp] = sqm;
    }
    __syncthreads();
    if (warp < 2) {
        const float sum = warp_sum((lane < 9) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_inv[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    const float ic = s_inv[0];
    const float im = s_inv[1];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        vc[j] *= ic;
        vm[j] *= im;
    }
    __nv_bfloat16* zrow = z + static_cast<std::int64_t>(row) * 4608;
    store_bf16x8(zrow + idx, vc);
    store_bf16x8(zrow + 2304 + idx, vm);
}

// Fast 2-deep prefetched Single Down-Proj ([4608 -> 64], 64 CTAs x 256 threads = 8 warps/row!)
template <int ActiveT>
__global__ __launch_bounds__(256, 4) void loop_control_down_single_pf2_kernel(
    const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ down_w,
    __nv_bfloat16* __restrict__ h_out) {
    __shared__ float smem[ActiveT][8];
    const int row  = blockIdx.x;  // 0..63
    const int tid  = threadIdx.x; // 0..255
    const int warp = tid >> 5;
    const int lane = tid & 31;

    constexpr int kK8  = 4608 / 8; // 576
    const uint4* wp    = reinterpret_cast<const uint4*>(down_w + static_cast<std::int64_t>(row) * 4608);
    float acc[ActiveT] = {};

    uint4 w_cur = ld_global_cg_u4(wp + tid);
    for (int i = tid; i < kK8; i += 256) {
        uint4 w_nxt{};
        if (i + 256 < kK8) { w_nxt = ld_global_cg_u4(wp + i + 256); }
        const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
        const float2 w0 = __bfloat1622float2(wb[0]);
        const float2 w1 = __bfloat1622float2(wb[1]);
        const float2 w2 = __bfloat1622float2(wb[2]);
        const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                z + static_cast<std::int64_t>(ti) * 4608) + i);
            const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const float2 x0 = __bfloat1622float2(xb[0]);
            const float2 x1 = __bfloat1622float2(xb[1]);
            const float2 x2 = __bfloat1622float2(xb[2]);
            const float2 x3 = __bfloat1622float2(xb[3]);
            acc[ti] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, acc[ti]));
            acc[ti] = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, acc[ti]));
            acc[ti] = fmaf(w2.x, x2.x, fmaf(w2.y, x2.y, acc[ti]));
            acc[ti] = fmaf(w3.x, x3.x, fmaf(w3.y, x3.y, acc[ti]));
        }
        w_cur = w_nxt;
    }

#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        const float ws = warp_sum(acc[ti]);
        if (lane == 0) { smem[ti][warp] = ws; }
    }
    __syncthreads();
    if (warp == 0 && lane < ActiveT) {
        const float sum = ((smem[lane][0] + smem[lane][1]) + (smem[lane][2] + smem[lane][3])) +
                          ((smem[lane][4] + smem[lane][5]) + (smem[lane][6] + smem[lane][7]));
        h_out[static_cast<std::int64_t>(lane) * 64 + row] = __float2bfloat16_rn(sum);
    }
}

// Fused Gain/Forget Up-Proj (K=64) + Tanh Blend + partial sq_sum for fast 288t Post-RMSNorm!
// Processes up to 8 tokens per CTA (grid.x = 72 -> 32 channels/CTA, grid.y = ceil(T/8)).
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
            const float tot = (s_sq[tid][0] + s_sq[tid][1]) + (s_sq[tid][2] + s_sq[tid][3]);
            partial_sq[static_cast<std::int64_t>(t_base + tid) * 72 + blockIdx.x] = tot;
        }
    }
}

// Fast post-RMSNorm finish using the 72 precomputed partial sq sums (<<<T, 288>>> uint4, 9 warps!)
__global__ void loop_control_post_norm_h2304_kernel(__nv_bfloat16* __restrict__ out,
                                                    const __nv_bfloat16* __restrict__ norm_w,
                                                    const float* __restrict__ partial_sq,
                                                    int tokens, float eps) {
    __shared__ float s_w[3];
    __shared__ float s_inv;

    const int row = blockIdx.x;
    if (row >= tokens) { return; }
    const int tid  = threadIdx.x; // 0..287 (9 warps, 8 BF16/thread)
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
    const int idx       = tid * 8;
    __nv_bfloat16* orow = out + static_cast<std::int64_t>(row) * 2304 + idx;
    float v[8], nw[8];
    load_bf16x8(orow, v);
    load_bf16x8(norm_w + idx, nw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = v[j] * inv * nw[j]; }
    store_bf16x8(orow, v);
}

// ============================================================================
// 4b. Batched 3-Controller (Q, K, V) Loop Control Kernels (ZERO stack frame!)
//     Flat scalar pointers avoid 192B local stack frame; 8-warp 2-deep prefetch in down;
//     parallel 288t uint4 post-RMSNorm avoids Last-CTA serialization!
// ============================================================================

__global__ void loop_control_prep_qkv_h2304_kernel(
    const __nv_bfloat16* __restrict__ cur0, const __nv_bfloat16* __restrict__ cur1,
    const __nv_bfloat16* __restrict__ cur2, const __nv_bfloat16* __restrict__ msg0,
    const __nv_bfloat16* __restrict__ msg1, const __nv_bfloat16* __restrict__ msg2,
    __nv_bfloat16* __restrict__ z_all, int tokens, float eps) {
    __shared__ float s_red[2][9];
    __shared__ float s_inv[2];

    const int row = blockIdx.x;
    const int c   = blockIdx.y; // 0, 1, 2
    if (row >= tokens || c >= 3) { return; }
    const int tid  = threadIdx.x; // 0..287 (9 warps, 8 BF16/thread)
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx  = tid * 8;

    const __nv_bfloat16* cur_p = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const __nv_bfloat16* msg_p = (c == 0) ? msg0 : ((c == 1) ? msg1 : msg2);
    float vc[8], vm[8];
    load_bf16x8(cur_p + static_cast<std::int64_t>(row) * 2304 + idx, vc);
    load_bf16x8(msg_p + static_cast<std::int64_t>(row) * 2304 + idx, vm);
    float sqc = 0.0f, sqm = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        sqc = fmaf(vc[j], vc[j], sqc);
        sqm = fmaf(vm[j], vm[j], sqm);
    }
    sqc = warp_sum(sqc);
    sqm = warp_sum(sqm);
    if (lane == 0) {
        s_red[0][warp] = sqc;
        s_red[1][warp] = sqm;
    }
    __syncthreads();
    if (warp < 2) {
        const float sum = warp_sum((lane < 9) ? s_red[warp][lane] : 0.0f);
        if (lane == 0) { s_inv[warp] = rsqrtf(sum * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();

    const float ic = s_inv[0];
    const float im = s_inv[1];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        vc[j] *= ic;
        vm[j] *= im;
    }
    __nv_bfloat16* zrow = z_all + (static_cast<std::int64_t>(c) * tokens + row) * 4608;
    store_bf16x8(zrow + idx, vc);
    store_bf16x8(zrow + 2304 + idx, vm);
}

// 192 CTAs (grid = (64, 3)), 256 threads (8 warps/row) + 2-deep weight prefetch, ZERO stack frame!
template <int ActiveT>
__global__ __launch_bounds__(256, 4) void loop_control_down_qkv_pf2_kernel(
    const __nv_bfloat16* __restrict__ down0, const __nv_bfloat16* __restrict__ down1,
    const __nv_bfloat16* __restrict__ down2, const __nv_bfloat16* __restrict__ z_all,
    __nv_bfloat16* __restrict__ h_all) {
    __shared__ float smem[ActiveT][8];
    const int row  = blockIdx.x; // 0..63
    const int c    = blockIdx.y; // 0..2
    const int tid  = threadIdx.x; // 0..255
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const __nv_bfloat16* down = (c == 0) ? down0 : ((c == 1) ? down1 : down2);
    const __nv_bfloat16* z    = z_all + static_cast<std::int64_t>(c) * ActiveT * 4608;
    const uint4* wp           = reinterpret_cast<const uint4*>(down + static_cast<std::int64_t>(row) * 4608);
    __nv_bfloat16* h_out      = h_all + static_cast<std::int64_t>(c) * ActiveT * 64;

    constexpr int kK8  = 4608 / 8; // 576
    float acc[ActiveT] = {};

    uint4 w_cur = ld_global_cg_u4(wp + tid);
    for (int i = tid; i < kK8; i += 256) {
        uint4 w_nxt{};
        if (i + 256 < kK8) { w_nxt = ld_global_cg_u4(wp + i + 256); }
        const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
        const float2 w0 = __bfloat1622float2(wb[0]);
        const float2 w1 = __bfloat1622float2(wb[1]);
        const float2 w2 = __bfloat1622float2(wb[2]);
        const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                z + static_cast<std::int64_t>(ti) * 4608) + i);
            const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const float2 x0 = __bfloat1622float2(xb[0]);
            const float2 x1 = __bfloat1622float2(xb[1]);
            const float2 x2 = __bfloat1622float2(xb[2]);
            const float2 x3 = __bfloat1622float2(xb[3]);
            acc[ti] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, acc[ti]));
            acc[ti] = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, acc[ti]));
            acc[ti] = fmaf(w2.x, x2.x, fmaf(w2.y, x2.y, acc[ti]));
            acc[ti] = fmaf(w3.x, x3.x, fmaf(w3.y, x3.y, acc[ti]));
        }
        w_cur = w_nxt;
    }

#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        const float ws = warp_sum(acc[ti]);
        if (lane == 0) { smem[ti][warp] = ws; }
    }
    __syncthreads();

    if (warp == 0 && lane < ActiveT) {
        const float sum = ((smem[lane][0] + smem[lane][1]) + (smem[lane][2] + smem[lane][3])) +
                          ((smem[lane][4] + smem[lane][5]) + (smem[lane][6] + smem[lane][7]));
        h_out[static_cast<std::int64_t>(lane) * 64 + row] = __float2bfloat16_rn(sum);
    }
}

template <int MaxTB>
__global__ __launch_bounds__(128, 4) void loop_control_up_blend_qkv_k64_kernel(
    const __nv_bfloat16* __restrict__ cur0, const __nv_bfloat16* __restrict__ cur1,
    const __nv_bfloat16* __restrict__ cur2, const __nv_bfloat16* __restrict__ msg0,
    const __nv_bfloat16* __restrict__ msg1, const __nv_bfloat16* __restrict__ msg2,
    const __nv_bfloat16* __restrict__ gf0, const __nv_bfloat16* __restrict__ gf1,
    const __nv_bfloat16* __restrict__ gf2, const __nv_bfloat16* __restrict__ norm0,
    const __nv_bfloat16* __restrict__ norm1, const __nv_bfloat16* __restrict__ norm2,
    __nv_bfloat16* __restrict__ out0, __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2, const __nv_bfloat16* __restrict__ h_all,
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

    const __nv_bfloat16* h      = h_all + static_cast<std::int64_t>(c) * tokens * 64;
    const __nv_bfloat16* gf_w   = (c == 0) ? gf0 : ((c == 1) ? gf1 : gf2);
    const __nv_bfloat16* cur    = (c == 0) ? cur0 : ((c == 1) ? cur1 : cur2);
    const __nv_bfloat16* msg    = (c == 0) ? msg0 : ((c == 1) ? msg1 : msg2);
    const __nv_bfloat16* norm_w = (c == 0) ? norm0 : ((c == 1) ? norm1 : norm2);
    __nv_bfloat16* out          = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const bool want_sq          = (norm_w != nullptr);
    float* partial_sq           = want_sq ? (partial_sq_all + static_cast<std::int64_t>(c) * tokens * 72) : nullptr;

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
            const float tot = (s_sq[tid][0] + s_sq[tid][1]) + (s_sq[tid][2] + s_sq[tid][3]);
            partial_sq[static_cast<std::int64_t>(t_base + tid) * 72 + blockIdx.x] = tot;
        }
    }
}

__global__ void loop_control_post_norm_qkv_h2304_kernel(
    const __nv_bfloat16* __restrict__ norm0, const __nv_bfloat16* __restrict__ norm1,
    const __nv_bfloat16* __restrict__ norm2, __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1, __nv_bfloat16* __restrict__ out2,
    const float* __restrict__ partial_sq_all, int tokens, float eps) {
    __shared__ float s_w[3];
    __shared__ float s_inv;

    const int row               = blockIdx.x;
    const int c                 = blockIdx.y; // 0..2
    const __nv_bfloat16* norm_p = (c == 0) ? norm0 : ((c == 1) ? norm1 : norm2);
    if (row >= tokens || c >= 3 || norm_p == nullptr) { return; }
    __nv_bfloat16* out_p = (c == 0) ? out0 : ((c == 1) ? out1 : out2);
    const int tid        = threadIdx.x; // 0..287 (9 warps, 8 BF16/thread)
    const int warp       = tid >> 5;
    const int lane       = tid & 31;

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
    const int idx       = tid * 8;
    __nv_bfloat16* orow = out_p + static_cast<std::int64_t>(row) * 2304 + idx;
    float v[8], nw[8];
    load_bf16x8(orow, v);
    load_bf16x8(norm_p + idx, nw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = v[j] * inv * nw[j]; }
    store_bf16x8(orow, v);
}

// ============================================================================
// 4c. Fused Pass-Transition Bank Swap (1 vectorized launch replacing 3 D2D copies)
// ============================================================================

__global__ void swap_pass_banks_vec8_kernel(
    const uint4* __restrict__ src_bank0, const uint4* __restrict__ prelude,
    uint4* __restrict__ cur, uint4* __restrict__ bank0, uint4* __restrict__ bank_out,
    std::size_t n8) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n8) { return; }
    const uint4 b0 = __ldg(src_bank0 + i);
    const uint4 cv = __ldg(cur + i);
    const uint4 pv = __ldg(prelude + i);
    bank0[i]       = b0;
    bank_out[i]    = cv;
    cur[i]         = pv;
}

__global__ void swap_pass_banks_scalar_kernel(
    const __nv_bfloat16* __restrict__ src_bank0, const __nv_bfloat16* __restrict__ prelude,
    __nv_bfloat16* __restrict__ cur, __nv_bfloat16* __restrict__ bank0,
    __nv_bfloat16* __restrict__ bank_out, std::size_t elems) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elems) { return; }
    const __nv_bfloat16 b0 = __ldg(src_bank0 + i);
    const __nv_bfloat16 cv = __ldg(cur + i);
    const __nv_bfloat16 pv = __ldg(prelude + i);
    bank0[i]               = b0;
    bank_out[i]            = cv;
    cur[i]                 = pv;
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

// ============================================================================
// 4d. Single-Launch Fused Loop Control (Down + UpBlend + PostNorm) & Mega-Fused Group B (QKV)
//     Uses single-word Volta PTX 6.0+ GPU-scope release-acquire grid barriers (72 CTAs <= 80 SMs)!
// ============================================================================

// Single-launch fused Single Loop Control (Down + UpBlend + PostNorm) for ActiveT in [1..8] (72x256, 2 bars!)
template <int ActiveT>
__global__ __launch_bounds__(256, 2) void loop_control_fused_single_t_kernel(
    const __nv_bfloat16* __restrict__ cur, const __nv_bfloat16* __restrict__ msg,
    const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ gf_w, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ h_buf, float* __restrict__ partial_sq,
    unsigned int* __restrict__ bar_word, __nv_bfloat16* __restrict__ out, float eps) {
    constexpr int G = 72;
    __shared__ float s_h[ActiveT][64];
    __shared__ float s_smem[ActiveT][8];
    __shared__ float s_inv[ActiveT];

    const int bid  = blockIdx.x;  // 0..71
    const int tid  = threadIdx.x; // 0..255
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Phase 1: Down [4608 -> 64] (CTAs 0..63, 2-deep weight prefetch)
    if (bid < 64) {
        constexpr int kK8 = 4608 / 8; // 576
        const uint4* wp =
            reinterpret_cast<const uint4*>(down_w + static_cast<std::int64_t>(bid) * 4608);
        float acc[ActiveT] = {};
        uint4 w_cur        = ld_global_cg_u4(wp + tid);
        for (int i = tid; i < kK8; i += 256) {
            uint4 w_nxt{};
            if (i + 256 < kK8) { w_nxt = ld_global_cg_u4(wp + i + 256); }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const float2 w0 = __bfloat1622float2(wb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
            for (int ti = 0; ti < ActiveT; ++ti) {
                const uint4 xv  = __ldg(reinterpret_cast<const uint4*>(
                                            z + static_cast<std::int64_t>(ti) * 4608) + i);
                const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&xv);
                const float2 x0 = __bfloat1622float2(xb[0]);
                const float2 x1 = __bfloat1622float2(xb[1]);
                const float2 x2 = __bfloat1622float2(xb[2]);
                const float2 x3 = __bfloat1622float2(xb[3]);
                acc[ti] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, acc[ti]));
                acc[ti] = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, acc[ti]));
                acc[ti] = fmaf(w2.x, x2.x, fmaf(w2.y, x2.y, acc[ti]));
                acc[ti] = fmaf(w3.x, x3.x, fmaf(w3.y, x3.y, acc[ti]));
            }
            w_cur = w_nxt;
        }
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            const float ws = warp_sum(acc[ti]);
            if (lane == 0) { s_smem[ti][warp] = ws; }
        }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int ti = 0; ti < ActiveT; ++ti) {
                const float tot =
                    ((s_smem[ti][0] + s_smem[ti][1]) + (s_smem[ti][2] + s_smem[ti][3])) +
                    ((s_smem[ti][4] + s_smem[ti][5]) + (s_smem[ti][6] + s_smem[ti][7]));
                const __nv_bfloat16 hv = __float2bfloat16_rn(tot);
                st_global_cg_u16(&h_buf[static_cast<std::int64_t>(ti) * 64 + bid],
                                 *reinterpret_cast<const std::uint16_t*>(&hv));
            }
        }
    }

    grid_barrier_n(bar_word, G, bid); // Barrier 1

    // Phase 2: UpBlend (ALL 72 CTAs, 32 ch/CTA x 8t = 256t, reading gf_w ONCE across ActiveT tokens!)
    for (int e = tid; e < ActiveT * 64; e += 256) {
        const std::uint16_t hu = ld_global_cg_u16(&h_buf[e]);
        s_h[e >> 6][e & 63]    = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&hu));
    }
    __syncthreads();

    const int ch_loc = tid >> 3;
    const int sub    = tid & 7;
    const int ch     = bid * 32 + ch_loc;
    const int k_off  = sub * 8;
    const uint4 gw   = ld_global_cg_u4(gf_w + static_cast<std::int64_t>(ch) * 64 + k_off);
    const uint4 fw   = ld_global_cg_u4(gf_w + static_cast<std::int64_t>(2304 + ch) * 64 + k_off);
    float gwf[8], fwf[8];
    const auto* gb = reinterpret_cast<const __nv_bfloat162*>(&gw);
    const auto* fb = reinterpret_cast<const __nv_bfloat162*>(&fw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 g0  = __bfloat1622float2(gb[j]);
        const float2 f0  = __bfloat1622float2(fb[j]);
        gwf[2 * j]       = g0.x;
        gwf[2 * j + 1]   = g0.y;
        fwf[2 * j]       = f0.x;
        fwf[2 * j + 1]   = f0.y;
    }

    float my_y[ActiveT];
#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        float g_acc = 0.0f, f_acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float hv = s_h[ti][k_off + j];
            g_acc          = fmaf(gwf[j], hv, g_acc);
            f_acc          = fmaf(fwf[j], hv, f_acc);
        }
        g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 1, 8);
        g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 2, 8);
        g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 4, 8);
        f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 1, 8);
        f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 2, 8);
        f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 4, 8);
        float y_sq = 0.0f;
        if (sub == 0) {
            const std::int64_t off = static_cast<std::int64_t>(ti) * 2304 + ch;
            const float y          = (1.0f - tanhf(f_acc)) * __bfloat162float(__ldg(cur + off)) +
                                     tanhf(g_acc) * __bfloat162float(__ldg(msg + off));
            my_y[ti] = __bfloat162float(__float2bfloat16_rn(y));
            y_sq     = y * y;
        }
        const float wsum = warp_sum(y_sq);
        if (lane == 0) { s_smem[ti][warp] = wsum; }
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            const float tot =
                ((s_smem[ti][0] + s_smem[ti][1]) + (s_smem[ti][2] + s_smem[ti][3])) +
                ((s_smem[ti][4] + s_smem[ti][5]) + (s_smem[ti][6] + s_smem[ti][7]));
            st_global_cg_f32(&partial_sq[ti * G + bid], tot);
        }
    }

    grid_barrier_n(bar_word, G, bid); // Barrier 2

    // Phase 3: Post-RMSNorm directly from registers (all ActiveT tokens reduced in parallel!)
    if (warp < ActiveT) {
        const float acc = warp_reduce_72_f4(&partial_sq[warp * G], lane);
        if (lane == 0) { s_inv[warp] = rsqrtf(acc * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    if (sub == 0) {
        const float nw = __bfloat162float(__ldg(norm_w + ch));
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            out[static_cast<std::int64_t>(ti) * 2304 + ch] =
                __float2bfloat16_rn(my_y[ti] * s_inv[ti] * nw);
        }
    }
}

// Mega-Fused Group B (Pre-Attn Core QKV, T == 1): 72 CTAs x 256t, 3 release-acquire barriers!
// Collapses AttnResPair + PrepQKV + DownQKV + UpBlendQKV + PostNormQKV (4 launches -> 1 launch!).
// Writes ONLY msg3_out + final normalized (out0, out1, out2)! Zero z_all, zero unnormed out HBM!
template <int S1, int Q1>
__global__ __launch_bounds__(256, 2) void attn_res_pair_loop_ctrl_qkv_72cta_kernel(
    const __nv_bfloat16* __restrict__ bank0, const __nv_bfloat16* __restrict__ bank1,
    const float* __restrict__ cp_q, __nv_bfloat16* __restrict__ msg3_out,
    AttnResArgs cur_args, const float* __restrict__ cur_q,
    const __nv_bfloat16* __restrict__ down0, const __nv_bfloat16* __restrict__ down1,
    const __nv_bfloat16* __restrict__ down2, const __nv_bfloat16* __restrict__ gf0,
    const __nv_bfloat16* __restrict__ gf1, const __nv_bfloat16* __restrict__ gf2,
    const __nv_bfloat16* __restrict__ norm0, const __nv_bfloat16* __restrict__ norm1,
    const __nv_bfloat16* __restrict__ norm2, __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1, __nv_bfloat16* __restrict__ out2,
    float* __restrict__ partial, __nv_bfloat16* __restrict__ h_all,
    unsigned int* __restrict__ bar_word, float eps) {
    constexpr int G      = 72;
    constexpr int kBankN = 11;
    constexpr int kCurG  = (S1 > 1) ? (S1 * (S1 - 1) / 2) : 0;
    constexpr int kCurN  = S1 + kCurG + Q1 * S1;
    constexpr int kTot   = kBankN + kCurN;
    constexpr float kScale = 0.020833333333333332f;

    __shared__ float s_tot[kTot];
    __shared__ float s_bws[4][2];
    __shared__ float s_binv[3];
    __shared__ float s_cws[Q1][S1];
    __shared__ float s_cinv[Q1];
    __shared__ float s_b32[2][32];
    __shared__ float s_c32[S1][32];
    __shared__ float s_msg32[3][32];
    __shared__ float s_cur32[Q1][32];
    __shared__ float s_h[3][64];
    __shared__ float s_smem[3][8];
    __shared__ float s_inv_y[3];

    const int bid  = blockIdx.x;  // 0..71
    const int tid  = threadIdx.x; // 0..255
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int ch0  = bid * 32 + lane;

    // ---- Phase 0: Parallel 72-CTA Bank + Cur AttnRes + Gram (all queries in parallel across warps!) ----
    if (warp == 0) {
        const float b0 = __bfloat162float(__ldg(bank0 + ch0));
        const float b1 = __bfloat162float(__ldg(bank1 + ch0));
        s_b32[0][lane] = b0;
        s_b32[1][lane] = b1;
        const float sq0 = warp_sum(b0 * b0);
        const float sq1 = warp_sum(b1 * b1);
        const float g01 = warp_sum(b0 * b1);
        if (lane == 0) {
            st_global_cg_f32(&partial[0 * G + bid], sq0);
            st_global_cg_f32(&partial[1 * G + bid], sq1);
            st_global_cg_f32(&partial[2 * G + bid], g01);
        }
    } else if (warp == 1) {
        float cr[S1];
#pragma unroll
        for (int s = 0; s < S1; ++s) {
            cr[s]          = __bfloat162float(__ldg(cur_args.sources[s] + ch0));
            s_c32[s][lane] = cr[s];
            const float sq = warp_sum(cr[s] * cr[s]);
            if (lane == 0) { st_global_cg_f32(&partial[(kBankN + s) * G + bid], sq); }
        }
        if constexpr (S1 > 1) {
            int gi = 0;
#pragma unroll
            for (int a = 0; a < S1; ++a) {
#pragma unroll
                for (int b = a + 1; b < S1; ++b) {
                    const float gd = warp_sum(cr[a] * cr[b]);
                    if (lane == 0) { st_global_cg_f32(&partial[(kBankN + S1 + gi) * G + bid], gd); }
                    ++gi;
                }
            }
        }
    }
    __syncthreads();

    if (warp < 4) {
        const float qv = __ldg(cp_q + static_cast<std::int64_t>(warp) * 2304 + ch0);
        const float d0 = warp_sum(qv * s_b32[0][lane]);
        const float d1 = warp_sum(qv * s_b32[1][lane]);
        if (lane == 0) {
            st_global_cg_f32(&partial[(3 + warp * 2 + 0) * G + bid], d0);
            st_global_cg_f32(&partial[(3 + warp * 2 + 1) * G + bid], d1);
        }
    } else if (warp < 4 + Q1) {
        const int q    = warp - 4;
        const float qv = __ldg(cur_q + static_cast<std::int64_t>(q) * 2304 + ch0);
#pragma unroll
        for (int s = 0; s < S1; ++s) {
            const float dt = warp_sum(qv * s_c32[s][lane]);
            if (lane == 0) {
                st_global_cg_f32(&partial[(kBankN + S1 + kCurG + q * S1 + s) * G + bid], dt);
            }
        }
    }

    grid_barrier_n(bar_word, G, bid); // Barrier 1

    for (int k = warp; k < kTot; k += 8) {
        const float s = warp_reduce_72_f4(&partial[k * G], lane);
        if (lane == 0) { s_tot[k] = s; }
    }
    __syncthreads();

    if (tid == 0) {
        const float ib0 = rsqrtf(s_tot[0] * (1.0f / 2304.0f) + eps);
        const float ib1 = rsqrtf(s_tot[1] * (1.0f / 2304.0f) + eps);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float s0 = s_tot[3 + q * 2 + 0] * ib0 * kScale;
            const float s1 = s_tot[3 + q * 2 + 1] * ib1 * kScale;
            const float mx = fmaxf(s0, s1);
            const float e0 = expf(s0 - mx), e1 = expf(s1 - mx), id = 1.0f / (e0 + e1);
            const float w0 = (e0 * id) * ib0, w1 = (e1 * id) * ib1;
            s_bws[q][0] = w0;
            s_bws[q][1] = w1;
            if (q < 3) {
                const float n2 = w0 * w0 * s_tot[0] + w1 * w1 * s_tot[1] + 2.0f * w0 * w1 * s_tot[2];
                s_binv[q]      = rsqrtf(fmaxf(n2, 0.0f) * (1.0f / 2304.0f) + eps);
            }
        }
    } else if (tid == 32) {
        float ic_s[S1];
#pragma unroll
        for (int s = 0; s < S1; ++s) { ic_s[s] = rsqrtf(s_tot[kBankN + s] * (1.0f / 2304.0f) + eps); }
#pragma unroll
        for (int q = 0; q < Q1; ++q) {
            float sc[S1], mx = -1e30f;
#pragma unroll
            for (int s = 0; s < S1; ++s) {
                sc[s] = s_tot[kBankN + S1 + kCurG + q * S1 + s] * ic_s[s] * kScale;
                mx    = fmaxf(mx, sc[s]);
            }
            float ws[S1], den = 0.0f;
#pragma unroll
            for (int s = 0; s < S1; ++s) { ws[s] = expf(sc[s] - mx); den += ws[s]; }
            const float id = 1.0f / den;
#pragma unroll
            for (int s = 0; s < S1; ++s) { ws[s] = (ws[s] * id) * ic_s[s]; s_cws[q][s] = ws[s]; }
            float n2 = 0.0f;
#pragma unroll
            for (int s = 0; s < S1; ++s) { n2 = fmaf(ws[s] * ws[s], s_tot[kBankN + s], n2); }
            if constexpr (S1 > 1) {
                int gi = 0;
#pragma unroll
                for (int a = 0; a < S1; ++a) {
#pragma unroll
                    for (int b = a + 1; b < S1; ++b) {
                        n2 = fmaf(2.0f * ws[a] * ws[b], s_tot[kBankN + S1 + gi], n2);
                        ++gi;
                    }
                }
            }
            s_cinv[q] = rsqrtf(fmaxf(n2, 0.0f) * (1.0f / 2304.0f) + eps);
        }
    }
    __syncthreads();

    if (warp == 0) {
        const float b0 = s_b32[0][lane], b1 = s_b32[1][lane];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float mv   = fmaf(s_bws[c][0], b0, s_bws[c][1] * b1);
            s_msg32[c][lane] = __bfloat162float(__float2bfloat16_rn(mv));
        }
        msg3_out[ch0] = __float2bfloat16_rn(fmaf(s_bws[3][0], b0, s_bws[3][1] * b1));
    } else if (warp == 1) {
#pragma unroll
        for (int q = 0; q < Q1; ++q) {
            float cv = 0.0f;
#pragma unroll
            for (int s = 0; s < S1; ++s) { cv = fmaf(s_cws[q][s], s_c32[s][lane], cv); }
            s_cur32[q][lane] = __bfloat162float(__float2bfloat16_rn(cv));
        }
    }

    // ---- Phase 1: Down QKV (CTAs 0..63, 3 controllers simultaneously for 3x activation reuse!) ----
    if (bid < 64) {
        const uint4* wp0 = reinterpret_cast<const uint4*>(down0 + static_cast<std::int64_t>(bid) * 4608);
        const uint4* wp1 = reinterpret_cast<const uint4*>(down1 + static_cast<std::int64_t>(bid) * 4608);
        const uint4* wp2 = reinterpret_cast<const uint4*>(down2 + static_cast<std::int64_t>(bid) * 4608);

        float cws_z[Q1][S1];
#pragma unroll
        for (int q = 0; q < Q1; ++q) {
#pragma unroll
            for (int s = 0; s < S1; ++s) { cws_z[q][s] = s_cws[q][s] * s_cinv[q]; }
        }
        float bws_z[3][2];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            bws_z[c][0] = s_bws[c][0] * s_binv[c];
            bws_z[c][1] = s_bws[c][1] * s_binv[c];
        }

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
        uint4 w0_cur = ld_global_cg_u4(wp0 + tid);
        uint4 w1_cur = ld_global_cg_u4(wp1 + tid);
        uint4 w2_cur = ld_global_cg_u4(wp2 + tid);
        for (int i = tid; i < 576; i += 256) {
            uint4 w0_nxt{}, w1_nxt{}, w2_nxt{};
            if (i + 256 < 576) {
                w0_nxt = ld_global_cg_u4(wp0 + i + 256);
                w1_nxt = ld_global_cg_u4(wp1 + i + 256);
                w2_nxt = ld_global_cg_u4(wp2 + i + 256);
            }
            const auto* wb0 = reinterpret_cast<const __nv_bfloat162*>(&w0_cur);
            const auto* wb1 = reinterpret_cast<const __nv_bfloat162*>(&w1_cur);
            const auto* wb2 = reinterpret_cast<const __nv_bfloat162*>(&w2_cur);
            if (i < 288) {
                float zc[Q1][8] = {};
#pragma unroll
                for (int s = 0; s < S1; ++s) {
                    float sv[8];
                    load_bf16x8(cur_args.sources[s] + i * 8, sv);
#pragma unroll
                    for (int q = 0; q < Q1; ++q) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) { zc[q][j] = fmaf(cws_z[q][s], sv[j], zc[q][j]); }
                    }
                }
                constexpr int q1 = (Q1 == 3) ? 1 : 0;
                constexpr int q2 = (Q1 == 3) ? 2 : 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f0 = __bfloat1622float2(wb0[j]);
                    const float2 f1 = __bfloat1622float2(wb1[j]);
                    const float2 f2 = __bfloat1622float2(wb2[j]);
                    acc0 = fmaf(f0.x, zc[0][2 * j], fmaf(f0.y, zc[0][2 * j + 1], acc0));
                    acc1 = fmaf(f1.x, zc[q1][2 * j], fmaf(f1.y, zc[q1][2 * j + 1], acc1));
                    acc2 = fmaf(f2.x, zc[q2][2 * j], fmaf(f2.y, zc[q2][2 * j + 1], acc2));
                }
            } else {
                float bv0[8], bv1[8];
                load_bf16x8(bank0 + (i - 288) * 8, bv0);
                load_bf16x8(bank1 + (i - 288) * 8, bv1);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f0 = __bfloat1622float2(wb0[j]);
                    const float2 f1 = __bfloat1622float2(wb1[j]);
                    const float2 f2 = __bfloat1622float2(wb2[j]);
                    const float z0x = fmaf(bws_z[0][0], bv0[2 * j], bws_z[0][1] * bv1[2 * j]);
                    const float z0y = fmaf(bws_z[0][0], bv0[2 * j + 1], bws_z[0][1] * bv1[2 * j + 1]);
                    const float z1x = fmaf(bws_z[1][0], bv0[2 * j], bws_z[1][1] * bv1[2 * j]);
                    const float z1y = fmaf(bws_z[1][0], bv0[2 * j + 1], bws_z[1][1] * bv1[2 * j + 1]);
                    const float z2x = fmaf(bws_z[2][0], bv0[2 * j], bws_z[2][1] * bv1[2 * j]);
                    const float z2y = fmaf(bws_z[2][0], bv0[2 * j + 1], bws_z[2][1] * bv1[2 * j + 1]);
                    acc0 = fmaf(f0.x, z0x, fmaf(f0.y, z0y, acc0));
                    acc1 = fmaf(f1.x, z1x, fmaf(f1.y, z1y, acc1));
                    acc2 = fmaf(f2.x, z2x, fmaf(f2.y, z2y, acc2));
                }
            }
            w0_cur = w0_nxt;
            w1_cur = w1_nxt;
            w2_cur = w2_nxt;
        }
        const float ws0 = warp_sum(acc0), ws1 = warp_sum(acc1), ws2 = warp_sum(acc2);
        if (lane == 0) {
            s_smem[0][warp] = ws0;
            s_smem[1][warp] = ws1;
            s_smem[2][warp] = ws2;
        }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float tot =
                    ((s_smem[c][0] + s_smem[c][1]) + (s_smem[c][2] + s_smem[c][3])) +
                    ((s_smem[c][4] + s_smem[c][5]) + (s_smem[c][6] + s_smem[c][7]));
                const __nv_bfloat16 hv = __float2bfloat16_rn(tot);
                st_global_cg_u16(&h_all[c * 64 + bid], *reinterpret_cast<const std::uint16_t*>(&hv));
            }
        }
    }

    grid_barrier_n(bar_word, G, bid); // Barrier 2

    // ---- Phase 2: UpBlend QKV (ALL 72 CTAs, 32 ch/CTA x 3 controllers -> keep y[0..2] in registers!) ----
    if (tid < 192) {
        const std::uint16_t hu  = ld_global_cg_u16(&h_all[tid]);
        s_h[tid >> 6][tid & 63] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&hu));
    }
    __syncthreads();

    const int ch_loc = tid >> 3, sub = tid & 7, ch = bid * 32 + ch_loc, k_off = sub * 8;
    float my_y[3]                   = {};
    const __nv_bfloat16* gf_ptrs[3] = {gf0, gf1, gf2};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const uint4 gw = ld_global_cg_u4(gf_ptrs[c] + static_cast<std::int64_t>(ch) * 64 + k_off);
        const uint4 fw =
            ld_global_cg_u4(gf_ptrs[c] + static_cast<std::int64_t>(2304 + ch) * 64 + k_off);
        const auto* gb = reinterpret_cast<const __nv_bfloat162*>(&gw);
        const auto* fb = reinterpret_cast<const __nv_bfloat162*>(&fw);
        float g_acc = 0.0f, f_acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 g0 = __bfloat1622float2(gb[j]);
            const float2 f0 = __bfloat1622float2(fb[j]);
            g_acc = fmaf(g0.x, s_h[c][k_off + 2 * j], fmaf(g0.y, s_h[c][k_off + 2 * j + 1], g_acc));
            f_acc = fmaf(f0.x, s_h[c][k_off + 2 * j], fmaf(f0.y, s_h[c][k_off + 2 * j + 1], f_acc));
        }
        g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 1, 8);
        g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 2, 8);
        g_acc += __shfl_xor_sync(0xFFFFFFFFu, g_acc, 4, 8);
        f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 1, 8);
        f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 2, 8);
        f_acc += __shfl_xor_sync(0xFFFFFFFFu, f_acc, 4, 8);
        float y_sq = 0.0f;
        if (sub == 0) {
            const int cq  = (Q1 == 3) ? c : 0;
            const float y = (1.0f - tanhf(f_acc)) * s_cur32[cq][ch_loc] +
                            tanhf(g_acc) * s_msg32[c][ch_loc];
            my_y[c] = __bfloat162float(__float2bfloat16_rn(y));
            y_sq    = y * y;
        }
        const float wsum = warp_sum(y_sq);
        if (lane == 0) { s_smem[c][warp] = wsum; }
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float tot =
                ((s_smem[c][0] + s_smem[c][1]) + (s_smem[c][2] + s_smem[c][3])) +
                ((s_smem[c][4] + s_smem[c][5]) + (s_smem[c][6] + s_smem[c][7]));
            st_global_cg_f32(&partial[c * G + bid], tot);
        }
    }

    grid_barrier_n(bar_word, G, bid); // Barrier 3

    // ---- Phase 3: PostNorm QKV directly from registers ----
    if (warp < 3) {
        const float acc = warp_reduce_72_f4(&partial[warp * G], lane);
        if (lane == 0) { s_inv_y[warp] = rsqrtf(acc * (1.0f / 2304.0f) + eps); }
    }
    __syncthreads();
    if (sub == 0) {
        out0[ch] = __float2bfloat16_rn(my_y[0] * s_inv_y[0] * __bfloat162float(__ldg(norm0 + ch)));
        out1[ch] = __float2bfloat16_rn(my_y[1] * s_inv_y[1] * __bfloat162float(__ldg(norm1 + ch)));
        out2[ch] = __float2bfloat16_rn(my_y[2] * s_inv_y[2] * __bfloat162float(__ldg(norm2 + ch)));
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
            inkling2_attn_res_mix_h2304_splitq3_kernel<S>
                <<<dim3(tokens, 3), 288, 0, stream>>>(args, queries, tokens, eps);
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

void inkling2_attn_res_mix_and_prep(const void* const* source_ptrs, std::int32_t num_sources,
                                    const Tensor& query_f32, const Tensor& msg,
                                    Tensor& out_mixed, Tensor& out_z, float eps,
                                    cudaStream_t stream) {
    const int tokens = out_mixed.ne[1];
    if (out_mixed.ne[0] != 2304 || num_sources < 1 || num_sources > 5) {
        void* outs[1] = {out_mixed.data};
        inkling2_attn_res_mix(source_ptrs, num_sources, query_f32, nullptr, outs, 1,
                              out_mixed.ne[0], tokens, eps, stream);
        loop_control_prep_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(out_mixed.data),
            static_cast<const __nv_bfloat16*>(msg.data),
            static_cast<__nv_bfloat16*>(out_z.data), out_mixed.ne[0], tokens, eps);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    AttnResArgs args{};
    for (int s = 0; s < num_sources; ++s) {
        args.sources[s] = static_cast<const __nv_bfloat16*>(source_ptrs[s]);
    }
    const auto* qp = static_cast<const float*>(query_f32.data);
    const auto* mp = static_cast<const __nv_bfloat16*>(msg.data);
    auto* om       = static_cast<__nv_bfloat16*>(out_mixed.data);
    auto* oz       = static_cast<__nv_bfloat16*>(out_z.data);
    switch (num_sources) {
    case 1: attn_res_mix_and_prep_h2304_kernel<1><<<tokens, 288, 0, stream>>>(args, qp, mp, om, oz, tokens, eps); break;
    case 2: attn_res_mix_and_prep_h2304_kernel<2><<<tokens, 288, 0, stream>>>(args, qp, mp, om, oz, tokens, eps); break;
    case 3: attn_res_mix_and_prep_h2304_kernel<3><<<tokens, 288, 0, stream>>>(args, qp, mp, om, oz, tokens, eps); break;
    case 4: attn_res_mix_and_prep_h2304_kernel<4><<<tokens, 288, 0, stream>>>(args, qp, mp, om, oz, tokens, eps); break;
    case 5: attn_res_mix_and_prep_h2304_kernel<5><<<tokens, 288, 0, stream>>>(args, qp, mp, om, oz, tokens, eps); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_attn_res_mix_pair_and_prep_qkv(
    const void* const* bank_srcs, const Tensor& cp_q_f32, void* const* msg_outs,
    const void* const* cur_srcs, std::int32_t num_cur_srcs, const Tensor& cur_q_f32,
    void* const* cur_outs, std::int32_t num_cur_q, Tensor& out_z_3t, float eps,
    cudaStream_t stream) {
    const int tokens = out_z_3t.ne[1] / 3;
    AttnResArgs a0{}, a1{};
    for (int s = 0; s < 2; ++s) { a0.sources[s] = static_cast<const __nv_bfloat16*>(bank_srcs[s]); }
    for (int q = 0; q < 4; ++q) { a0.outs[q] = static_cast<__nv_bfloat16*>(msg_outs[q]); }
    for (int s = 0; s < num_cur_srcs; ++s) { a1.sources[s] = static_cast<const __nv_bfloat16*>(cur_srcs[s]); }
    for (int q = 0; q < num_cur_q; ++q) { a1.outs[q] = static_cast<__nv_bfloat16*>(cur_outs[q]); }
    const auto* qp0 = static_cast<const float*>(cp_q_f32.data);
    const auto* qp1 = static_cast<const float*>(cur_q_f32.data);
    auto* oz        = static_cast<__nv_bfloat16*>(out_z_3t.data);
    const dim3 grid(tokens, 2);
    auto launch = [&]<int S1>() {
        if (num_cur_q == 1) {
            attn_res_mix_pair_and_prep_qkv_h2304_kernel<S1, 1><<<grid, 288, 0, stream>>>(a0, qp0, a1, qp1, oz, tokens, eps);
        } else {
            attn_res_mix_pair_and_prep_qkv_h2304_kernel<S1, 3><<<grid, 288, 0, stream>>>(a0, qp0, a1, qp1, oz, tokens, eps);
        }
    };
    switch (num_cur_srcs) {
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    case 4: launch.template operator()<4>(); break;
    default: throw std::invalid_argument("inkling2_attn_res_mix_pair_and_prep_qkv: bad num_cur_srcs");
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_loop_control(const Tensor& cur, const Tensor& msg, const Tensor& down_w,
                           const Tensor& gain_forget_w, const Tensor* post_norm_w,
                           Tensor& scratch_z, Tensor& scratch_h, Tensor& scratch_gf,
                           Tensor& out, float eps, cudaStream_t stream,
                           bool skip_prep, std::int32_t* done_counter) {
    const int hidden = cur.ne[0];
    const int tokens = cur.ne[1];
    const int rank   = down_w.ne[1];

    if (hidden == 2304 && rank == 64 && tokens <= 32) {
        if (!skip_prep) {
            loop_control_prep_h2304_kernel<<<tokens, 288, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(cur.data),
                static_cast<const __nv_bfloat16*>(msg.data),
                static_cast<__nv_bfloat16*>(scratch_z.data), tokens, eps);
            CUDA_CHECK(cudaGetLastError());
        }

        // Single-launch fused Down + UpBlend + PostNorm (72 CTAs <= 80 SMs, 2 release-acquire bars!)
        // Empirically beats 3-launch by 10-13% for T in [1..6]; T=7..8 keeps full 3-launch occupancy.
        if (tokens >= 1 && tokens <= 6 && post_norm_w != nullptr && done_counter != nullptr &&
            scratch_gf.bytes() >= static_cast<std::size_t>(tokens) * 72ULL * sizeof(float) &&
            get_current_device_sm_count() >= 72) {
            const auto* cp = static_cast<const __nv_bfloat16*>(cur.data);
            const auto* mp = static_cast<const __nv_bfloat16*>(msg.data);
            const auto* zp = static_cast<const __nv_bfloat16*>(scratch_z.data);
            const auto* dp = static_cast<const __nv_bfloat16*>(down_w.data);
            const auto* gp = static_cast<const __nv_bfloat16*>(gain_forget_w.data);
            const auto* np = static_cast<const __nv_bfloat16*>(post_norm_w->data);
            auto* hp       = static_cast<__nv_bfloat16*>(scratch_h.data);
            auto* psq      = static_cast<float*>(scratch_gf.data);
            auto* bar      = reinterpret_cast<unsigned int*>(done_counter);
            auto* op       = static_cast<__nv_bfloat16*>(out.data);
            auto launch = [&]<int ActiveT>() {
                loop_control_fused_single_t_kernel<ActiveT><<<72, 256, 0, stream>>>(
                    cp, mp, zp, dp, gp, np, hp, psq, bar, op, eps);
            };
            switch (tokens) {
            case 1: launch.template operator()<1>(); break;
            case 2: launch.template operator()<2>(); break;
            case 3: launch.template operator()<3>(); break;
            case 4: launch.template operator()<4>(); break;
            case 5: launch.template operator()<5>(); break;
            case 6: launch.template operator()<6>(); break;
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        if (tokens <= 8) {
            const auto* zp = static_cast<const __nv_bfloat16*>(scratch_z.data);
            const auto* dp = static_cast<const __nv_bfloat16*>(down_w.data);
            auto* hp       = static_cast<__nv_bfloat16*>(scratch_h.data);
            switch (tokens) {
            case 1: loop_control_down_single_pf2_kernel<1><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 2: loop_control_down_single_pf2_kernel<2><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 3: loop_control_down_single_pf2_kernel<3><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 4: loop_control_down_single_pf2_kernel<4><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 5: loop_control_down_single_pf2_kernel<5><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 6: loop_control_down_single_pf2_kernel<6><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 7: loop_control_down_single_pf2_kernel<7><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            case 8: loop_control_down_single_pf2_kernel<8><<<64, 256, 0, stream>>>(zp, dp, hp); break;
            }
            CUDA_CHECK(cudaGetLastError());
        } else {
            inkling2_bf16_linear(scratch_z, down_w, scratch_h, stream);
        }

        float* partial_sq  = post_norm_w ? static_cast<float*>(scratch_gf.data) : nullptr;
        const auto* nw_ptr = post_norm_w ? static_cast<const __nv_bfloat16*>(post_norm_w->data) : nullptr;

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
        } else if (tokens <= 8) {
            loop_control_up_blend_k64_kernel<8><<<dim3(72, 1), 128, 0, stream>>>(
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
            loop_control_post_norm_h2304_kernel<<<tokens, 288, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), nw_ptr, partial_sq, tokens, eps);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Generic / prefill path (T > 32 uses fast tiled Tensor Core GEMM for down & gain_forget)
    if (!skip_prep) {
        if (hidden == 2304) {
            loop_control_prep_h2304_kernel<<<tokens, 288, 0, stream>>>(
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
    }

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
                               float eps, cudaStream_t stream,
                               bool skip_prep, std::int32_t* done_counters) {
    const int hidden = cur0.ne[0];
    const int tokens = cur0.ne[1];
    const int rank   = down0.ne[1];

    if (hidden == 2304 && rank == 64 && tokens >= 1 && tokens <= 32) {
        const auto* c0 = static_cast<const __nv_bfloat16*>(cur0.data);
        const auto* c1 = static_cast<const __nv_bfloat16*>(cur1.data);
        const auto* c2 = static_cast<const __nv_bfloat16*>(cur2.data);
        const auto* m0 = static_cast<const __nv_bfloat16*>(msg0.data);
        const auto* m1 = static_cast<const __nv_bfloat16*>(msg1.data);
        const auto* m2 = static_cast<const __nv_bfloat16*>(msg2.data);
        const auto* d0 = static_cast<const __nv_bfloat16*>(down0.data);
        const auto* d1 = static_cast<const __nv_bfloat16*>(down1.data);
        const auto* d2 = static_cast<const __nv_bfloat16*>(down2.data);
        const auto* g0 = static_cast<const __nv_bfloat16*>(gf0.data);
        const auto* g1 = static_cast<const __nv_bfloat16*>(gf1.data);
        const auto* g2 = static_cast<const __nv_bfloat16*>(gf2.data);
        const auto* n0 = norm0 ? static_cast<const __nv_bfloat16*>(norm0->data) : nullptr;
        const auto* n1 = norm1 ? static_cast<const __nv_bfloat16*>(norm1->data) : nullptr;
        const auto* n2 = norm2 ? static_cast<const __nv_bfloat16*>(norm2->data) : nullptr;
        auto* o0       = static_cast<__nv_bfloat16*>(out0.data);
        auto* o1       = static_cast<__nv_bfloat16*>(out1.data);
        auto* o2       = static_cast<__nv_bfloat16*>(out2.data);

        auto* z_all = static_cast<__nv_bfloat16*>(scratch_z.data);
        auto* h_all = static_cast<__nv_bfloat16*>(scratch_h.data);
        auto* psq   = static_cast<float*>(scratch_gf.data);

        if (!skip_prep) {
            loop_control_prep_qkv_h2304_kernel<<<dim3(tokens, 3), 288, 0, stream>>>(
                c0, c1, c2, m0, m1, m2, z_all, tokens, eps);
            CUDA_CHECK(cudaGetLastError());
        }
        (void)done_counters; // QKV T>=2 retains full 192/216-CTA 80-SM parallelism; T==1 uses Group B.

        if (tokens <= 8) {
            const dim3 dgrid(64, 3);
            switch (tokens) {
            case 1: loop_control_down_qkv_pf2_kernel<1><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 2: loop_control_down_qkv_pf2_kernel<2><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 3: loop_control_down_qkv_pf2_kernel<3><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 4: loop_control_down_qkv_pf2_kernel<4><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 5: loop_control_down_qkv_pf2_kernel<5><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 6: loop_control_down_qkv_pf2_kernel<6><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 7: loop_control_down_qkv_pf2_kernel<7><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
            case 8: loop_control_down_qkv_pf2_kernel<8><<<dgrid, 256, 0, stream>>>(d0, d1, d2, z_all, h_all); break;
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

        const bool any_norm = (n0 != nullptr || n1 != nullptr || n2 != nullptr);

        if (tokens == 1) {
            loop_control_up_blend_qkv_k64_kernel<1><<<dim3(72, 1, 3), 128, 0, stream>>>(
                c0, c1, c2, m0, m1, m2, g0, g1, g2, n0, n1, n2, o0, o1, o2, h_all, psq, tokens);
        } else if (tokens <= 4) {
            loop_control_up_blend_qkv_k64_kernel<4><<<dim3(72, 1, 3), 128, 0, stream>>>(
                c0, c1, c2, m0, m1, m2, g0, g1, g2, n0, n1, n2, o0, o1, o2, h_all, psq, tokens);
        } else if (tokens <= 8) {
            loop_control_up_blend_qkv_k64_kernel<8><<<dim3(72, 1, 3), 128, 0, stream>>>(
                c0, c1, c2, m0, m1, m2, g0, g1, g2, n0, n1, n2, o0, o1, o2, h_all, psq, tokens);
        } else {
            loop_control_up_blend_qkv_k64_kernel<8>
                <<<dim3(72, (tokens + 7) / 8, 3), 128, 0, stream>>>(
                    c0, c1, c2, m0, m1, m2, g0, g1, g2, n0, n1, n2, o0, o1, o2, h_all, psq, tokens);
        }
        CUDA_CHECK(cudaGetLastError());

        if (any_norm) {
            loop_control_post_norm_qkv_h2304_kernel<<<dim3(tokens, 3), 288, 0, stream>>>(
                n0, n1, n2, o0, o1, o2, psq, tokens, eps);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    // Fallback for prefill (T > 32) or non-standard shapes
    auto* z_all = static_cast<__nv_bfloat16*>(scratch_z.data);
    Tensor sz0(z_all, DType::BF16, {2 * hidden, tokens});
    Tensor sz1(z_all + static_cast<std::int64_t>(tokens) * 2 * hidden, DType::BF16, {2 * hidden, tokens});
    Tensor sz2(z_all + static_cast<std::int64_t>(2 * tokens) * 2 * hidden, DType::BF16, {2 * hidden, tokens});
    Tensor sh(scratch_h.data, DType::BF16, {rank, tokens});
    Tensor sg(scratch_gf.data, DType::BF16, {2 * hidden, tokens});
    inkling2_loop_control(cur0, msg0, down0, gf0, norm0, sz0, sh, sg, out0, eps, stream, skip_prep);
    inkling2_loop_control(cur1, msg1, down1, gf1, norm1, sz1, sh, sg, out1, eps, stream, skip_prep);
    inkling2_loop_control(cur2, msg2, down2, gf2, norm2, sz2, sh, sg, out2, eps, stream, skip_prep);
}

void inkling2_attn_res_pair_loop_ctrl_qkv_fused(
    const void* const* bank_srcs, const Tensor& cp_q_f32, void* const* tmp_msg_outs,
    const void* const* cur_srcs, std::int32_t num_cur_srcs, const Tensor& cur_q_f32,
    void* const* tmp_cur_outs, std::int32_t num_cur_q,
    const Tensor& down0, const Tensor& down1, const Tensor& down2,
    const Tensor& gf0, const Tensor& gf1, const Tensor& gf2,
    const Tensor& norm0, const Tensor& norm1, const Tensor& norm2,
    Tensor& scratch_z_3t, Tensor& scratch_h_3t, Tensor& scratch_gf_3t,
    Tensor& out0, Tensor& out1, Tensor& out2,
    float eps, cudaStream_t stream, std::int32_t* done_counters) {
    const int hidden = out0.ne[0];
    const int tokens = out0.ne[1];
    const int rank   = down0.ne[1];

    // Mega-fused 1-launch path at T == 1 (72 CTAs <= 80 SMs, 3 release-acquire barriers, 0 spills!)
    if (tokens == 1 && hidden == 2304 && rank == 64 && num_cur_srcs >= 1 && num_cur_srcs <= 4 &&
        (num_cur_q == 1 || num_cur_q == 3) && done_counters != nullptr &&
        scratch_gf_3t.bytes() >= 33ULL * 72ULL * sizeof(float) &&
        get_current_device_sm_count() >= 72) {
        AttnResArgs cur_a{};
        for (int s = 0; s < num_cur_srcs; ++s) {
            cur_a.sources[s] = static_cast<const __nv_bfloat16*>(cur_srcs[s]);
        }
        const auto* b0   = static_cast<const __nv_bfloat16*>(bank_srcs[0]);
        const auto* b1   = static_cast<const __nv_bfloat16*>(bank_srcs[1]);
        const auto* cpq  = static_cast<const float*>(cp_q_f32.data);
        auto* msg3       = static_cast<__nv_bfloat16*>(tmp_msg_outs[3]);
        const auto* curq = static_cast<const float*>(cur_q_f32.data);
        const auto* d0   = static_cast<const __nv_bfloat16*>(down0.data);
        const auto* d1   = static_cast<const __nv_bfloat16*>(down1.data);
        const auto* d2   = static_cast<const __nv_bfloat16*>(down2.data);
        const auto* g0   = static_cast<const __nv_bfloat16*>(gf0.data);
        const auto* g1   = static_cast<const __nv_bfloat16*>(gf1.data);
        const auto* g2   = static_cast<const __nv_bfloat16*>(gf2.data);
        const auto* n0   = static_cast<const __nv_bfloat16*>(norm0.data);
        const auto* n1   = static_cast<const __nv_bfloat16*>(norm1.data);
        const auto* n2   = static_cast<const __nv_bfloat16*>(norm2.data);
        auto* o0         = static_cast<__nv_bfloat16*>(out0.data);
        auto* o1         = static_cast<__nv_bfloat16*>(out1.data);
        auto* o2         = static_cast<__nv_bfloat16*>(out2.data);
        auto* part       = static_cast<float*>(scratch_gf_3t.data);
        auto* hall       = static_cast<__nv_bfloat16*>(scratch_h_3t.data);
        auto* bar        = reinterpret_cast<unsigned int*>(done_counters);

        auto launch = [&]<int S1>() {
            if (num_cur_q == 1) {
                attn_res_pair_loop_ctrl_qkv_72cta_kernel<S1, 1><<<72, 256, 0, stream>>>(
                    b0, b1, cpq, msg3, cur_a, curq, d0, d1, d2, g0, g1, g2, n0, n1, n2,
                    o0, o1, o2, part, hall, bar, eps);
            } else {
                attn_res_pair_loop_ctrl_qkv_72cta_kernel<S1, 3><<<72, 256, 0, stream>>>(
                    b0, b1, cpq, msg3, cur_a, curq, d0, d1, d2, g0, g1, g2, n0, n1, n2,
                    o0, o1, o2, part, hall, bar, eps);
            }
        };
        switch (num_cur_srcs) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 3: launch.template operator()<3>(); break;
        case 4: launch.template operator()<4>(); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Fallback for T >= 2 (uses fused loop_control_qkv for T=2..8!) or GPUs with < 72 SMs
    inkling2_attn_res_mix_pair_and_prep_qkv(bank_srcs, cp_q_f32, tmp_msg_outs, cur_srcs,
                                            num_cur_srcs, cur_q_f32, tmp_cur_outs, num_cur_q,
                                            scratch_z_3t, eps, stream);
    Tensor msg0(tmp_msg_outs[0], DType::BF16, {hidden, tokens});
    Tensor msg1(tmp_msg_outs[1], DType::BF16, {hidden, tokens});
    Tensor msg2(tmp_msg_outs[2], DType::BF16, {hidden, tokens});
    Tensor cur0(tmp_cur_outs[0], DType::BF16, {hidden, tokens});
    Tensor cur1(tmp_cur_outs[num_cur_q == 1 ? 0 : 1], DType::BF16, {hidden, tokens});
    Tensor cur2(tmp_cur_outs[num_cur_q == 1 ? 0 : 2], DType::BF16, {hidden, tokens});
    inkling2_loop_control_qkv(cur0, cur1, cur2, msg0, msg1, msg2, down0, down1, down2,
                              gf0, gf1, gf2, &norm0, &norm1, &norm2, scratch_z_3t, scratch_h_3t,
                              scratch_gf_3t, out0, out1, out2, eps, stream, /*skip_prep=*/true,
                              done_counters);
}

void inkling2_swap_pass_banks(const Tensor& src_bank0, const Tensor& prelude,
                              Tensor& cur, Tensor& bank0, Tensor& bank_out,
                              cudaStream_t stream) {
    const std::size_t elems = static_cast<std::size_t>(cur.ne[0]) * cur.ne[1];
    if (elems == 0) { return; }
    if ((elems % 8) == 0) {
        const std::size_t n8 = elems / 8;
        swap_pass_banks_vec8_kernel<<<static_cast<unsigned>((n8 + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint4*>(src_bank0.data), static_cast<const uint4*>(prelude.data),
            static_cast<uint4*>(cur.data), static_cast<uint4*>(bank0.data),
            static_cast<uint4*>(bank_out.data), n8);
    } else {
        swap_pass_banks_scalar_kernel<<<static_cast<unsigned>((elems + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src_bank0.data),
            static_cast<const __nv_bfloat16*>(prelude.data),
            static_cast<__nv_bfloat16*>(cur.data), static_cast<__nv_bfloat16*>(bank0.data),
            static_cast<__nv_bfloat16*>(bank_out.data), elems);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
