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

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
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

__device__ __forceinline__ uint4 ld_global_cg_u4(const void* ptr) {
    uint4 bits;
    // Non-volatile read-only streaming load: allows nvcc to schedule/pipeline multiple HBM2 loads in flight!
    asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
        : "l"(ptr));
    return bits;
}

// ============================================================================
// 1. T == 1 High-Bandwidth Volta BF16 GEMV
//    - 1 row/warp (4 rows/CTA -> 1075 CTAs QKV / 576 CTAs O-proj) + 2-deep
//      software prefetch (765 GB/s QKV, 673 GB/s O-proj on V100, <= 40 regs!).
//    - Multi-warp Split-K (WarpsPerRow=8) for small N <= 512 (saturates all 80 SMs!).
// ============================================================================

template <typename OutT>
__global__ __launch_bounds__(128, 8) void bf16_gemv_1rpw_pf2_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 4 + warp;
    if (row >= n) { return; }

    const int k8    = k >> 3;
    const uint4* wp = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    const uint4* xp = reinterpret_cast<const uint4*>(x);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (lane < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + lane);
        uint4 x_cur = __ldg(xp + lane);
        for (int i = lane; i < k8; i += 32) {
            uint4 w_nxt{}, x_nxt{};
            if (i + 32 < k8) {
                w_nxt = ld_global_cg_u4(wp + i + 32);
                x_nxt = __ldg(xp + i + 32);
            }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&x_cur);
            const float2 w0 = __bfloat1622float2(wb[0]), x0 = __bfloat1622float2(xb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]), x1 = __bfloat1622float2(xb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]), x2 = __bfloat1622float2(xb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]), x3 = __bfloat1622float2(xb[3]);
            a0 = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, a0));
            a1 = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, a1));
            a2 = fmaf(w2.x, x2.x, fmaf(w2.y, x2.y, a2));
            a3 = fmaf(w3.x, x3.x, fmaf(w3.y, x3.y, a3));
            w_cur = w_nxt;
            x_cur = x_nxt;
        }
    }

    float sum = warp_reduce_sum((a0 + a1) + (a2 + a3));
    if (lane == 0) {
        if (bias != nullptr) { sum += __bfloat162float(__ldg(bias + row)); }
        if (residual != nullptr) { sum += __bfloat162float(__ldg(residual + row)); }
        if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
            out[row] = __float2bfloat16_rn(sum);
        } else {
            out[row] = sum;
        }
    }
}

// ============================================================================
// 2. Small-T (T = 1..8) Coalesced Multi-Token SIMT Kernel
//    - Small N (<= 512): 1 CTA per output row, SplitK warps (WarpsPerRow = 8) ->
//      64..512 CTAs saturate all 80 SMs AND read each weight row ONCE across ALL T tokens!
//    - Large N (> 512): 4 warps/CTA, RowsPerWarp = 2, reads each weight row ONCE across ALL T!
//    - Zero BF16->FP16 conversions, 100% coalesced 128-byte weight lines, exact FP32 accum!
// ============================================================================

template <typename OutT, int ActiveT, int WarpsPerRow>
__global__ __launch_bounds__(WarpsPerRow * 32, 2) void bf16_small_t_splitk_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k) {
    __shared__ float smem[ActiveT][WarpsPerRow];
    const int row  = blockIdx.x;
    if (row >= n) { return; }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __nv_bfloat16* wrow = w + static_cast<std::int64_t>(row) * k;
    const int k8              = k >> 3;
    float acc[ActiveT]        = {};

    for (int i = threadIdx.x; i < k8; i += WarpsPerRow * 32) {
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
                x + static_cast<std::int64_t>(ti) * k) + i);
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
        const float ws = warp_reduce_sum(acc[ti]);
        if (lane == 0) { smem[ti][warp] = ws; }
    }
    __syncthreads();

    if (warp == 0 && lane < ActiveT) {
        const int ti = lane;
        float sum    = 0.0f;
#pragma unroll
        for (int s = 0; s < WarpsPerRow; ++s) { sum += smem[ti][s]; }
        if (bias != nullptr) { sum += __bfloat162float(__ldg(bias + row)); }
        const std::int64_t idx = static_cast<std::int64_t>(ti) * n + row;
        if (residual != nullptr) { sum += __bfloat162float(__ldg(residual + idx)); }
        if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
            out[idx] = __float2bfloat16_rn(sum);
        } else {
            out[idx] = sum;
        }
    }
}

// 16 threads/row (8 rows/CTA, 256B coalesced, 2-deep weight prefetch, low registers -> 709 GB/s T=4!)
template <typename OutT, int ActiveT>
__global__ __launch_bounds__(128, 6) void bf16_small_t_16tpr_pf2_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k) {
    const int r_loc = threadIdx.x >> 4; // 0..7
    const int kl    = threadIdx.x & 15; // 0..15
    const int row   = blockIdx.x * 8 + r_loc;
    if (row >= n) { return; }

    const int k8       = k >> 3;
    const uint4* wp    = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    float acc[ActiveT] = {};

    if (kl < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + kl);
        for (int i = kl; i < k8; i += 16) {
            uint4 w_nxt{};
            if (i + 16 < k8) { w_nxt = ld_global_cg_u4(wp + i + 16); }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const float2 w0 = __bfloat1622float2(wb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
            for (int ti = 0; ti < ActiveT; ++ti) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(ti) * k) + i);
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
    }

    const float bval = (kl == 0 && bias != nullptr) ? __bfloat162float(__ldg(bias + row)) : 0.0f;
#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        float sum = acc[ti];
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 8, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1, 16);
        if (kl == 0) {
            sum += bval;
            const std::int64_t idx = static_cast<std::int64_t>(ti) * n + row;
            if (residual != nullptr) { sum += __bfloat162float(__ldg(residual + idx)); }
            if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
                out[idx] = __float2bfloat16_rn(sum);
            } else {
                out[idx] = sum;
            }
        }
    }
}

// 32 threads/row (4 rows/CTA -> 576 CTAs for O-proj!) with 2-deep weight prefetch for T <= 4
template <typename OutT, int ActiveT>
__global__ __launch_bounds__(128, 6) void bf16_small_t_32tpr_pf2_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 4 + warp;
    if (row >= n) { return; }

    const int k8       = k >> 3;
    const uint4* wp    = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    float acc[ActiveT] = {};

    if (lane < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + lane);
        for (int i = lane; i < k8; i += 32) {
            uint4 w_nxt{};
            if (i + 32 < k8) { w_nxt = ld_global_cg_u4(wp + i + 32); }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const float2 w0 = __bfloat1622float2(wb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
            for (int ti = 0; ti < ActiveT; ++ti) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(ti) * k) + i);
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
    }

    const float bval = (lane == 0 && bias != nullptr) ? __bfloat162float(__ldg(bias + row)) : 0.0f;
#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        float sum = warp_reduce_sum(acc[ti]);
        if (lane == 0) {
            sum += bval;
            const std::int64_t idx = static_cast<std::int64_t>(ti) * n + row;
            if (residual != nullptr) { sum += __bfloat162float(__ldg(residual + idx)); }
            if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
                out[idx] = __float2bfloat16_rn(sum);
            } else {
                out[idx] = sum;
            }
        }
    }
}

// Coalesced T = 9..16 SIMT kernel (16tpr, reads weights ONCE at ~550 GB/s -> 2x faster than QPN!)
template <typename OutT>
__global__ __launch_bounds__(128, 4) void bf16_t9_16_16tpr_pf2_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k, int t) {
    const int r_loc = threadIdx.x >> 4;
    const int kl    = threadIdx.x & 15;
    const int row   = blockIdx.x * 8 + r_loc;
    if (row >= n) { return; }

    const int k8  = k >> 3;
    const uint4* wp = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    float acc[16] = {};

    if (kl < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + kl);
        for (int i = kl; i < k8; i += 16) {
            uint4 w_nxt{};
            if (i + 16 < k8) { w_nxt = ld_global_cg_u4(wp + i + 16); }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const float2 w0 = __bfloat1622float2(wb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]);
            for (int ti = 0; ti < t; ++ti) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(ti) * k) + i);
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
    }

    const float bval = (kl == 0 && bias != nullptr) ? __bfloat162float(__ldg(bias + row)) : 0.0f;
    for (int ti = 0; ti < t; ++ti) {
        float sum = acc[ti];
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 8, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1, 16);
        if (kl == 0) {
            sum += bval;
            const std::int64_t idx = static_cast<std::int64_t>(ti) * n + row;
            if (residual != nullptr) { sum += __bfloat162float(__ldg(residual + idx)); }
            if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
                out[idx] = __float2bfloat16_rn(sum);
            } else {
                out[idx] = sum;
            }
        }
    }
}

template <typename OutT, int ActiveT>
void dispatch_bf16_small_t_fixed(const __nv_bfloat16* x, const __nv_bfloat16* w, OutT* out,
                                 const __nv_bfloat16* bias, const __nv_bfloat16* residual,
                                 int n, int k, cudaStream_t stream) {
    if (n <= 512 && k >= 512) {
        constexpr int kSplit = 8;
        bf16_small_t_splitk_kernel<OutT, ActiveT, kSplit>
            <<<n, kSplit * 32, 0, stream>>>(x, w, out, bias, residual, n, k);
    } else if (ActiveT <= 4 && n <= 2560 && k >= 1024) {
        const int blocks = (n + 3) / 4;
        bf16_small_t_32tpr_pf2_kernel<OutT, ActiveT>
            <<<blocks, 128, 0, stream>>>(x, w, out, bias, residual, n, k);
    } else {
        const int blocks = (n + 7) / 8;
        bf16_small_t_16tpr_pf2_kernel<OutT, ActiveT>
            <<<blocks, 128, 0, stream>>>(x, w, out, bias, residual, n, k);
    }
}

// ============================================================================
// 3. Medium-T (T = 9..32) Multi-Tile Volta QPN Tensor Core Kernel (hmma.884)
//    kTiles in {2, 3, 4} reads weights ONCE for up to 32 tokens (vs 3-4x re-reads!).
// ============================================================================

template <typename OutT, int kTiles, int SplitK>
__global__ __launch_bounds__(
    SplitK * 32, (kTiles <= 2 ? 16 : 4) / SplitK < 1 ? 1 : (kTiles <= 2 ? 16 : 4) / SplitK)
void bf16_volta_qpn_ktiles_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k, int t) {
    __shared__ float partial[SplitK][kTiles * 8 * 32];
    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int qp      = (lane >> 2) & 3;
    const int r       = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col     = blockIdx.x * 32 + qp * 8 + r;
    const int use_col = (col < n) ? col : 0;

    const int blocks = k / 64;
    const int begin  = warp * (blocks / SplitK) + min(warp, blocks % SplitK);
    const int end    = begin + blocks / SplitK + (warp < blocks % SplitK);

    float c[kTiles][2][8] = {};
    for (int block = begin; block < end; ++block) {
        uint4 words[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            words[e] = ld_global_cg_u4(
                reinterpret_cast<const uint4*>(
                    w + static_cast<std::int64_t>(use_col) * k + block * 64 + e * 8));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&words[e]);
            half2 bh[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(wb[j]);
                bh[j]          = __floats2half2_rn(f.x, f.y);
            }
            const auto* bb = reinterpret_cast<const unsigned*>(bh);
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int t_row = tile * 8 + r;
                half2 ah[4];
                if (t_row < t) {
                    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                        x + static_cast<std::int64_t>(t_row) * k + block * 64 + e * 8));
                    const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __bfloat1622float2(xb[j]);
                        ah[j]          = __floats2half2_rn(f.x, f.y);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { ah[j] = __half2half2(__ushort_as_half(0)); }
                }
                const auto* aa = reinterpret_cast<const unsigned*>(ah);
                volta_mma_qp_n(c[tile][0], aa[0], aa[1], bb[0], bb[1]);
                volta_mma_qp_n(c[tile][1], aa[2], aa[3], bb[2], bb[3]);
            }
        }
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            partial[warp][(tile * 8 + row) * 32 + qp * 8 + cl] = c[tile][0][i] + c[tile][1][i];
        }
    }
    __syncthreads();

    constexpr int kOut = kTiles * 8 * 32;
    for (int e = threadIdx.x; e < kOut; e += SplitK * 32) {
        const int grow = e >> 5;
        const int cl   = e & 31;
        const int gcol = blockIdx.x * 32 + cl;
        if (grow < t && gcol < n) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) { sum += partial[s][e]; }
            if (bias != nullptr) { sum += __bfloat162float(__ldg(bias + gcol)); }
            const std::int64_t idx = static_cast<std::int64_t>(grow) * n + gcol;
            if (residual != nullptr) { sum += __bfloat162float(__ldg(residual + idx)); }
            if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
                out[idx] = __float2bfloat16_rn(sum);
            } else {
                out[idx] = sum;
            }
        }
    }
}

// ============================================================================
// 4. Large-T (T > 32, Prefill) Volta Shared-Memory Tiled Tensor Core GEMM
// ============================================================================

template <typename OutT>
__global__ __launch_bounds__(128, 2) void bf16_volta_tiled_gemm_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    OutT* __restrict__ out, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, int n, int k, int t) {
    constexpr int BM             = 64;
    constexpr int BN             = 64;
    constexpr int BK             = 32;
    constexpr int SMEM_STRIDE_H2 = (BK / 2) + 4;

    __shared__ half2 smem_a[BM][SMEM_STRIDE_H2];
    __shared__ half2 smem_b[BN][SMEM_STRIDE_H2];

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1;
    const int warp_n = warp & 1;

    const int bm_base = blockIdx.y * BM;
    const int bn_base = blockIdx.x * BN;

    float acc[4][8];
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[nb][i] = 0.0f; }
    }

    for (int bk = 0; bk < k; bk += BK) {
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int vec_id = tid + step * 128;
            const int r      = vec_id >> 2;
            const int c8     = (vec_id & 3) * 8;

            const int gr_a = bm_base + r;
            if (gr_a < t) {
                const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(gr_a) * k + bk + c8));
                const auto* sb = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f           = __bfloat1622float2(sb[j]);
                    smem_a[r][(c8 >> 1) + j] = __floats2half2_rn(f.x, f.y);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    smem_a[r][(c8 >> 1) + j] = __half2half2(__ushort_as_half(0));
                }
            }

            const int gr_b = bn_base + r;
            if (gr_b < n) {
                const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                    w + static_cast<std::int64_t>(gr_b) * k + bk + c8));
                const auto* sb = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f           = __bfloat1622float2(sb[j]);
                    smem_b[r][(c8 >> 1) + j] = __floats2half2_rn(f.x, f.y);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    smem_b[r][(c8 >> 1) + j] = __half2half2(__ushort_as_half(0));
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            half2 a_frag[4];
            volta_load_qp(a_frag, &smem_a[warp_m * 32][ks * 4], SMEM_STRIDE_H2);
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                half2 b_frag[4];
                volta_load_k(b_frag, &smem_b[warp_n * 32 + nb * 8][ks * 4], SMEM_STRIDE_H2);
                volta_mma_qk(acc[nb], a_frag, b_frag);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int grow = bm_base + warp_m * 32 + volta_d_get_i(l);
            const int gcol = bn_base + warp_n * 32 + nb * 8 + volta_d_get_j(l);
            if (grow < t && gcol < n) {
                float val = acc[nb][l];
                if (bias != nullptr) { val += __bfloat162float(__ldg(bias + gcol)); }
                const std::int64_t idx = static_cast<std::int64_t>(grow) * n + gcol;
                if (residual != nullptr) { val += __bfloat162float(__ldg(residual + idx)); }
                if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
                    out[idx] = __float2bfloat16_rn(val);
                } else {
                    out[idx] = val;
                }
            }
        }
    }
}

// ============================================================================
// 5. Fused Unembed + Exact Monotonic Softcap Argmax (ZERO Logits HBM Write, ZERO tanhf in Loop!)
//    Because f(x) = cap * tanh(x / cap) is strictly increasing for cap > 0, argmax(f(x)) == argmax(x)!
//    We find exact raw argmax across valid_cols in one pass over output_head, then apply
//    softcap * tanhf(v / softcap) ONCE per token to the winning scalar in reduce_block_argmax!
// ============================================================================

// 2-deep software-prefetched Unembed Argmax: 883 GB/s at T=1 (32tpr) & 798 GB/s at T=4 (16tpr)!
template <int ActiveT, int Tpr>
__global__ __launch_bounds__(128, 6) void unembed_small_t_argmax_pf2_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    int valid_cols, int k, int num_blocks, float* __restrict__ block_vals,
    std::int32_t* __restrict__ block_ids) {
    constexpr int Rpb = 128 / Tpr; // 4 (Tpr=32) or 8 (Tpr=16)
    __shared__ float s_val[ActiveT][Rpb];
    __shared__ int s_idx[ActiveT][Rpb];

    const int r_loc = threadIdx.x / Tpr;
    const int kl    = threadIdx.x & (Tpr - 1);
    const int k8    = k >> 3;

    float best_v[ActiveT];
    int best_id[ActiveT];
#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        best_v[ti]  = -std::numeric_limits<float>::infinity();
        best_id[ti] = 0;
    }

    const int stride_rows = num_blocks * Rpb;
    for (int row = blockIdx.x * Rpb + r_loc; row < valid_cols; row += stride_rows) {
        const uint4* wp    = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
        float acc[ActiveT] = {};
        if (kl < k8) {
            uint4 w_cur = ld_global_cg_u4(wp + kl);
            for (int i = kl; i < k8; i += Tpr) {
                uint4 w_nxt{};
                if (i + Tpr < k8) { w_nxt = ld_global_cg_u4(wp + i + Tpr); }
                const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
                const float2 w0 = __bfloat1622float2(wb[0]);
                const float2 w1 = __bfloat1622float2(wb[1]);
                const float2 w2 = __bfloat1622float2(wb[2]);
                const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
                for (int ti = 0; ti < ActiveT; ++ti) {
                    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                        x + static_cast<std::int64_t>(ti) * k) + i);
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
        }
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            float val = acc[ti];
#pragma unroll
            for (int off = Tpr >> 1; off > 0; off >>= 1) {
                val += __shfl_xor_sync(0xFFFFFFFFu, val, off, Tpr);
            }
            if (kl == 0 && (val > best_v[ti] || (val == best_v[ti] && row < best_id[ti]))) {
                best_v[ti]  = val;
                best_id[ti] = row;
            }
        }
    }

    if (kl == 0) {
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            s_val[ti][r_loc] = best_v[ti];
            s_idx[ti][r_loc] = best_id[ti];
        }
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
#pragma unroll
        for (int ti = 0; ti < ActiveT; ++ti) {
            float v = (lane < Rpb) ? s_val[ti][lane] : -std::numeric_limits<float>::infinity();
            int id  = (lane < Rpb) ? s_idx[ti][lane] : 0;
            warp_reduce_argmax(v, id);
            if (lane == 0) {
                block_vals[static_cast<std::int64_t>(ti) * num_blocks + blockIdx.x] = v;
                block_ids[static_cast<std::int64_t>(ti) * num_blocks + blockIdx.x]  = id;
            }
        }
    }
}

// Fused Multi-Tile QPN Unembed Argmax for T = 9..32 (reads output_head ONCE for all T tokens!)
template <int kTiles, int SplitK>
__global__ __launch_bounds__(
    SplitK * 32, (kTiles <= 2 ? 16 : 4) / SplitK < 1 ? 1 : (kTiles <= 2 ? 16 : 4) / SplitK)
void unembed_qpn_argmax_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ w, int valid_cols, int k, int t,
                               int num_blocks, float* __restrict__ block_vals,
                               std::int32_t* __restrict__ block_ids) {
    __shared__ float partial[SplitK][kTiles * 8 * 32];
    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int qp     = (lane >> 2) & 3;
    const int r      = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int blocks = k / 64;
    const int begin  = warp * (blocks / SplitK) + min(warp, blocks % SplitK);
    const int end    = begin + blocks / SplitK + (warp < blocks % SplitK);

    // Track per-CTA best across grid-strided 32-col tiles (num_blocks = 320!)
    __shared__ float s_best_v[kTiles * 8];
    __shared__ int s_best_id[kTiles * 8];
    if (threadIdx.x < kTiles * 8) {
        s_best_v[threadIdx.x]  = -std::numeric_limits<float>::infinity();
        s_best_id[threadIdx.x] = 0;
    }
    __syncthreads();

    const int total_tiles = (valid_cols + 31) / 32;
    for (int tile_n = blockIdx.x; tile_n < total_tiles; tile_n += num_blocks) {
        const int col     = tile_n * 32 + qp * 8 + r;
        const int use_col = (col < valid_cols) ? col : 0;

        float c[kTiles][2][8] = {};
        for (int block = begin; block < end; ++block) {
            uint4 words[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                words[e] = ld_global_cg_u4(
                    reinterpret_cast<const uint4*>(
                        w + static_cast<std::int64_t>(use_col) * k + block * 64 + e * 8));
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&words[e]);
                half2 bh[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(wb[j]);
                    bh[j]          = __floats2half2_rn(f.x, f.y);
                }
                const auto* bb = reinterpret_cast<const unsigned*>(bh);
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int t_row = tile * 8 + r;
                    half2 ah[4];
                    if (t_row < t) {
                        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                            x + static_cast<std::int64_t>(t_row) * k + block * 64 + e * 8));
                        const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __bfloat1622float2(xb[j]);
                            ah[j]          = __floats2half2_rn(f.x, f.y);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) { ah[j] = __half2half2(__ushort_as_half(0)); }
                    }
                    const auto* aa = reinterpret_cast<const unsigned*>(ah);
                    volta_mma_qp_n(c[tile][0], aa[0], aa[1], bb[0], bb[1]);
                    volta_mma_qp_n(c[tile][1], aa[2], aa[3], bb[2], bb[3]);
                }
            }
        }

#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partial[warp][(tile * 8 + row) * 32 + qp * 8 + cl] = c[tile][0][i] + c[tile][1][i];
            }
        }
        __syncthreads();

        // Each warp reduces its assigned rows (32 cols -> 1 max per row via warp_reduce_argmax!)
        for (int grow = warp; grow < t; grow += SplitK) {
            const int gcol = tile_n * 32 + lane;
            float val      = -std::numeric_limits<float>::infinity();
            if (gcol < valid_cols) {
                val = 0.0f;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) { val += partial[s][grow * 32 + lane]; }
            }
            int id = gcol;
            warp_reduce_argmax(val, id);
            if (lane == 0) {
                const float prev_v = s_best_v[grow];
                const int prev_i   = s_best_id[grow];
                if (val > prev_v || (val == prev_v && id < prev_i)) {
                    s_best_v[grow]  = val;
                    s_best_id[grow] = id;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < t) {
        const int grow = threadIdx.x;
        block_vals[static_cast<std::int64_t>(grow) * num_blocks + blockIdx.x] = s_best_v[grow];
        block_ids[static_cast<std::int64_t>(grow) * num_blocks + blockIdx.x]  = s_best_id[grow];
    }
}

__global__ void reduce_block_argmax_kernel(const float* __restrict__ block_vals,
                                           const std::int32_t* __restrict__ block_ids,
                                           int num_blocks, int stride, float softcap,
                                           float* __restrict__ out_val,
                                           std::int32_t* __restrict__ out_id) {
    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float* bv        = block_vals + static_cast<std::int64_t>(row) * stride;
    const std::int32_t* bi = block_ids + static_cast<std::int64_t>(row) * stride;

    float v = -std::numeric_limits<float>::infinity();
    int id  = 0;
    for (int i = tid; i < num_blocks; i += 256) {
        const float cv = bv[i];
        const int ci   = bi[i];
        if (cv > v || (cv == v && ci < id)) {
            v  = cv;
            id = ci;
        }
    }
    warp_reduce_argmax(v, id);
    if (lane == 0) {
        s_val[warp] = v;
        s_idx[warp] = id;
    }
    __syncthreads();
    if (warp == 0) {
        v  = (lane < 8) ? s_val[lane] : -std::numeric_limits<float>::infinity();
        id = (lane < 8) ? s_idx[lane] : 0;
        warp_reduce_argmax(v, id);
        if (lane == 0) {
            if (softcap > 0.0f && isfinite(v)) { v = softcap * tanhf(v / softcap); }
            out_val[row] = v;
            out_id[row]  = id;
        }
    }
}

__global__ void softcap_mask_argmax_f32_kernel(float* __restrict__ logits, int n, int t,
                                               float softcap, int valid_cols,
                                               float* __restrict__ out_vals,
                                               std::int32_t* __restrict__ out_ids) {
    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    const int row  = blockIdx.x;
    if (row >= t) { return; }
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float* lrow  = logits + static_cast<std::int64_t>(row) * n;
    float best_v = -std::numeric_limits<float>::infinity();
    int best_id  = 0;

    for (int col = tid; col < n; col += 256) {
        float val;
        if (col < valid_cols) {
            const float raw = lrow[col];
            val             = (softcap > 0.0f) ? (softcap * tanhf(raw / softcap)) : raw;
        } else {
            val = -std::numeric_limits<float>::infinity();
        }
        lrow[col] = val;
        if (val > best_v || (val == best_v && col < best_id)) {
            best_v  = val;
            best_id = col;
        }
    }
    if (out_vals != nullptr && out_ids != nullptr) {
        warp_reduce_argmax(best_v, best_id);
        if (lane == 0) {
            s_val[warp] = best_v;
            s_idx[warp] = best_id;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < 8) ? s_val[lane] : -std::numeric_limits<float>::infinity();
            int id  = (lane < 8) ? s_idx[lane] : 0;
            warp_reduce_argmax(v, id);
            if (lane == 0) {
                out_vals[row] = v;
                out_ids[row]  = id;
            }
        }
    }
}

// ============================================================================
// 6. Fused DSpark Markov Step: base_logits[b, v] + dot(latent[b], fold_w[v]) + fold_b[v] -> argmax
//    Supports multi-batch (B <= 8) reading fold_w [256, 100512] ONCE for ALL B lanes!
// ============================================================================

template <typename BaseT, typename LatentT, int MaxB>
__global__ __launch_bounds__(256, 2) void markov_logits_argmax_batched_kernel(
    const BaseT* __restrict__ base_logits, const LatentT* __restrict__ latent,
    const __nv_bfloat16* __restrict__ fold_w, const float* __restrict__ fold_b,
    int n_shard, int valid_cols, int batch, int base_row_stride, int base_row_offset,
    float* __restrict__ block_vals, std::int32_t* __restrict__ block_ids, int num_blocks_n) {
    constexpr int kWarps = 8;
    __shared__ float s_lat[MaxB][256];
    __shared__ float s_val[MaxB][kWarps];
    __shared__ int s_idx[MaxB][kWarps];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int b = 0; b < batch; ++b) {
        if constexpr (std::is_same_v<LatentT, __nv_bfloat16>) {
            s_lat[b][tid] = __bfloat162float(latent[static_cast<std::int64_t>(b) * 256 + tid]);
        } else {
            s_lat[b][tid] = latent[static_cast<std::int64_t>(b) * 256 + tid];
        }
    }
    __syncthreads();

    float lx[MaxB][8];
#pragma unroll
    for (int b = 0; b < MaxB; ++b) {
        if (b < batch) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { lx[b][j] = s_lat[b][lane * 8 + j]; }
        }
    }

    float best_v[MaxB];
    int best_id[MaxB];
#pragma unroll
    for (int b = 0; b < MaxB; ++b) {
        best_v[b]  = -std::numeric_limits<float>::infinity();
        best_id[b] = 0;
    }

    const int rows_per_block = (valid_cols + num_blocks_n - 1) / num_blocks_n;
    const int r_begin        = blockIdx.x * rows_per_block;
    const int r_end          = min(valid_cols, r_begin + rows_per_block);

    // 4 rows/warp/iter -> 4x HBM2 MLP on fold_w + coalesced prefetch of fold_b & base_logits!
    for (int v0 = r_begin + warp * 4; v0 < r_end; v0 += kWarps * 4) {
        if (v0 + 3 < r_end) {
            const uint4 wv0 = ld_global_cg_u4(fold_w + static_cast<std::int64_t>(v0 + 0) * 256 + lane * 8);
            const uint4 wv1 = ld_global_cg_u4(fold_w + static_cast<std::int64_t>(v0 + 1) * 256 + lane * 8);
            const uint4 wv2 = ld_global_cg_u4(fold_w + static_cast<std::int64_t>(v0 + 2) * 256 + lane * 8);
            const uint4 wv3 = ld_global_cg_u4(fold_w + static_cast<std::int64_t>(v0 + 3) * 256 + lane * 8);

            // Coalesced prefetch of 4 fold_b + 4 base_logits across lanes 0..3 while fold_w streams!
            const float fb_mine = (lane < 4) ? __ldg(fold_b + v0 + lane) : 0.0f;
            float base_mine[MaxB];
#pragma unroll
            for (int b = 0; b < MaxB; ++b) {
                if (b < batch && lane < 4) {
                    const std::int64_t brow =
                        static_cast<std::int64_t>(b) * base_row_stride + base_row_offset;
                    if constexpr (std::is_same_v<BaseT, __nv_bfloat16>) {
                        base_mine[b] = __bfloat162float(__ldg(base_logits + brow * n_shard + v0 + lane));
                    } else {
                        base_mine[b] = __ldg(base_logits + brow * n_shard + v0 + lane);
                    }
                } else {
                    base_mine[b] = 0.0f;
                }
            }

            const uint4 wvs[4] = {wv0, wv1, wv2, wv3};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wvs[r]);
                float wf[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(wb[j]);
                    wf[2 * j]      = f.x;
                    wf[2 * j + 1]  = f.y;
                }
                const float fb_r = __shfl_sync(0xFFFFFFFFu, fb_mine, r);
                const int vr     = v0 + r;
#pragma unroll
                for (int b = 0; b < MaxB; ++b) {
                    if (b < batch) {
                        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a0 = fmaf(wf[2 * j], lx[b][2 * j], a0);
                            a1 = fmaf(wf[2 * j + 1], lx[b][2 * j + 1], a1);
                        }
                        const float acc    = warp_reduce_sum(a0 + a1);
                        const float base_r = __shfl_sync(0xFFFFFFFFu, base_mine[b], r);
                        if (lane == 0) {
                            const float total = base_r + acc + fb_r;
                            if (total > best_v[b] || (total == best_v[b] && vr < best_id[b])) {
                                best_v[b]  = total;
                                best_id[b] = vr;
                            }
                        }
                    }
                }
            }
        } else {
            for (int v = v0; v < r_end; ++v) {
                const uint4 wv = ld_global_cg_u4(
                    fold_w + static_cast<std::int64_t>(v) * 256 + lane * 8);
                const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
                float wf[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(wb[j]);
                    wf[2 * j]      = f.x;
                    wf[2 * j + 1]  = f.y;
                }
                const float fb_v = (lane == 0) ? __ldg(fold_b + v) : 0.0f;
#pragma unroll
                for (int b = 0; b < MaxB; ++b) {
                    if (b < batch) {
                        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a0 = fmaf(wf[2 * j], lx[b][2 * j], a0);
                            a1 = fmaf(wf[2 * j + 1], lx[b][2 * j + 1], a1);
                        }
                        const float acc = warp_reduce_sum(a0 + a1);
                        if (lane == 0) {
                            const std::int64_t brow =
                                static_cast<std::int64_t>(b) * base_row_stride + base_row_offset;
                            float base_v;
                            if constexpr (std::is_same_v<BaseT, __nv_bfloat16>) {
                                base_v = __bfloat162float(__ldg(base_logits + brow * n_shard + v));
                            } else {
                                base_v = __ldg(base_logits + brow * n_shard + v);
                            }
                            const float total = base_v + acc + fb_v;
                            if (total > best_v[b] || (total == best_v[b] && v < best_id[b])) {
                                best_v[b]  = total;
                                best_id[b] = v;
                            }
                        }
                    }
                }
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < MaxB; ++b) {
            if (b < batch) {
                s_val[b][warp] = best_v[b];
                s_idx[b][warp] = best_id[b];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int b = 0; b < MaxB; ++b) {
            if (b < batch) {
                float v = (lane < kWarps) ? s_val[b][lane] : -std::numeric_limits<float>::infinity();
                int id  = (lane < kWarps) ? s_idx[b][lane] : 0;
                warp_reduce_argmax(v, id);
                if (lane == 0) {
                    block_vals[static_cast<std::int64_t>(b) * num_blocks_n + blockIdx.x] = v;
                    block_ids[static_cast<std::int64_t>(b) * num_blocks_n + blockIdx.x]  = id;
                }
            }
        }
    }
}

template <typename OutT>
void dispatch_bf16_linear(const __nv_bfloat16* x, const __nv_bfloat16* w, OutT* out,
                          const __nv_bfloat16* bias, const __nv_bfloat16* residual,
                          int n, int k, int t, cudaStream_t stream) {
    if (k <= 0 || n <= 0 || t <= 0 || (k % 64 != 0)) {
        throw std::invalid_argument("inkling2_bf16_linear: requires positive N, T and K % 64 == 0");
    }
    if (t == 1) {
        if (n <= 512 && k >= 512) {
            constexpr int kSplit = 8;
            bf16_small_t_splitk_kernel<OutT, 1, kSplit>
                <<<n, kSplit * 32, 0, stream>>>(x, w, out, bias, residual, n, k);
        } else {
            const int blocks = (n + 3) / 4;
            bf16_gemv_1rpw_pf2_kernel<OutT>
                <<<blocks, 128, 0, stream>>>(x, w, out, bias, residual, n, k);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t <= 8) {
        switch (t) {
        case 2: dispatch_bf16_small_t_fixed<OutT, 2>(x, w, out, bias, residual, n, k, stream); break;
        case 3: dispatch_bf16_small_t_fixed<OutT, 3>(x, w, out, bias, residual, n, k, stream); break;
        case 4: dispatch_bf16_small_t_fixed<OutT, 4>(x, w, out, bias, residual, n, k, stream); break;
        case 5: dispatch_bf16_small_t_fixed<OutT, 5>(x, w, out, bias, residual, n, k, stream); break;
        case 6: dispatch_bf16_small_t_fixed<OutT, 6>(x, w, out, bias, residual, n, k, stream); break;
        case 7: dispatch_bf16_small_t_fixed<OutT, 7>(x, w, out, bias, residual, n, k, stream); break;
        case 8: dispatch_bf16_small_t_fixed<OutT, 8>(x, w, out, bias, residual, n, k, stream); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t <= 16 && k >= 512) {
        const int blocks = (n + 7) / 8;
        bf16_t9_16_16tpr_pf2_kernel<OutT>
            <<<blocks, 128, 0, stream>>>(x, w, out, bias, residual, n, k, t);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t <= 32) {
        const int grid_x = (n + 31) / 32;
        if (k == 64) {
            if (t <= 16) {
                bf16_volta_qpn_ktiles_kernel<OutT, 2, 1><<<grid_x, 32, 0, stream>>>(x, w, out, bias, residual, n, k, t);
            } else if (t <= 24) {
                bf16_volta_qpn_ktiles_kernel<OutT, 3, 1><<<grid_x, 32, 0, stream>>>(x, w, out, bias, residual, n, k, t);
            } else {
                bf16_volta_qpn_ktiles_kernel<OutT, 4, 1><<<grid_x, 32, 0, stream>>>(x, w, out, bias, residual, n, k, t);
            }
        } else if (t <= 16) {
            if (n < 1024 && k >= 512) {
                bf16_volta_qpn_ktiles_kernel<OutT, 2, 8><<<grid_x, 256, 0, stream>>>(x, w, out, bias, residual, n, k, t);
            } else {
                bf16_volta_qpn_ktiles_kernel<OutT, 2, 4><<<grid_x, 128, 0, stream>>>(x, w, out, bias, residual, n, k, t);
            }
        } else if (t <= 24) {
            bf16_volta_qpn_ktiles_kernel<OutT, 3, 4><<<grid_x, 128, 0, stream>>>(x, w, out, bias, residual, n, k, t);
        } else {
            bf16_volta_qpn_ktiles_kernel<OutT, 4, 4><<<grid_x, 128, 0, stream>>>(x, w, out, bias, residual, n, k, t);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid((n + 63) / 64, (t + 63) / 64);
    bf16_volta_tiled_gemm_kernel<OutT>
        <<<grid, 128, 0, stream>>>(x, w, out, bias, residual, n, k, t);
    CUDA_CHECK(cudaGetLastError());
}

struct ArgmaxScratch {
    float* vals       = nullptr;
    std::int32_t* ids = nullptr;
    std::size_t cap   = 0;
};

ArgmaxScratch& get_argmax_scratch(int device, std::size_t needed) {
    static ArgmaxScratch scratches[8];
    ArgmaxScratch& s         = scratches[device & 7];
    const std::size_t target = std::max<std::size_t>(needed, 32768);
    if (s.cap < target) {
        if (s.vals) { (void)cudaFree(s.vals); }
        if (s.ids) { (void)cudaFree(s.ids); }
        CUDA_CHECK(cudaMalloc(&s.vals, target * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.ids, target * sizeof(std::int32_t)));
        s.cap = target;
    }
    return s;
}

template <int ActiveT>
void launch_unembed_small_t_argmax(const __nv_bfloat16* x, const __nv_bfloat16* w, int valid,
                                   int k, float softcap, ArgmaxScratch& scratch, float* max_vals,
                                   std::int32_t* max_ids, cudaStream_t stream) {
    constexpr int kTpr   = (ActiveT == 1) ? 32 : 16;
    constexpr int kRpb   = 128 / kTpr;
    const int max_blks   = (valid + kRpb - 1) / kRpb;
    const int num_blocks = std::min(320, max_blks);
    unembed_small_t_argmax_pf2_kernel<ActiveT, kTpr><<<num_blocks, 128, 0, stream>>>(
        x, w, valid, k, num_blocks, scratch.vals, scratch.ids);
    CUDA_CHECK(cudaGetLastError());
    reduce_block_argmax_kernel<<<ActiveT, 256, 0, stream>>>(
        scratch.vals, scratch.ids, num_blocks, num_blocks, softcap, max_vals, max_ids);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void inkling2_ensure_argmax_scratch(int device, std::size_t min_blocks) {
    int prev = 0;
    CUDA_CHECK(cudaGetDevice(&prev));
    if (prev != device) { CUDA_CHECK(cudaSetDevice(device)); }
    (void)get_argmax_scratch(device, min_blocks);
    if (prev != device) { CUDA_CHECK(cudaSetDevice(prev)); }
}

void inkling2_bf16_linear(const Tensor& x, const Tensor& w, Tensor& out, cudaStream_t stream,
                          const Tensor* bias, const Tensor* residual) {
    if (x.dtype != DType::BF16 || w.dtype != DType::BF16) {
        throw std::invalid_argument("inkling2_bf16_linear: x and w must be BF16");
    }
    const int k = x.ne[0];
    const int t = x.ne[1];
    const int n = w.ne[1];
    if (w.ne[0] != k || out.ne[0] != n || out.ne[1] != t) {
        throw std::invalid_argument("inkling2_bf16_linear: shape mismatch");
    }
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const auto* wp = static_cast<const __nv_bfloat16*>(w.data);
    const auto* bp = bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr;
    const auto* rp = residual ? static_cast<const __nv_bfloat16*>(residual->data) : nullptr;
    if (out.dtype == DType::BF16) {
        dispatch_bf16_linear(xp, wp, static_cast<__nv_bfloat16*>(out.data), bp, rp, n, k, t, stream);
    } else if (out.dtype == DType::FP32) {
        dispatch_bf16_linear(xp, wp, static_cast<float*>(out.data), bp, rp, n, k, t, stream);
    } else {
        throw std::invalid_argument("inkling2_bf16_linear: out must be BF16 or FP32");
    }
}

// ============================================================================
// 7. Fused Paired & Triple BF16 Linear (eliminates tail waves & 108 launches/step!)
// ============================================================================

template <int ActiveT, int WarpsPerRow>
__global__ __launch_bounds__(WarpsPerRow * 32, 2) void bf16_pair_small_t_splitk_kernel(
    const __nv_bfloat16* __restrict__ x0, const __nv_bfloat16* __restrict__ w0,
    __nv_bfloat16* __restrict__ out0, int n0,
    const __nv_bfloat16* __restrict__ x1, const __nv_bfloat16* __restrict__ w1,
    __nv_bfloat16* __restrict__ out1, int n1, int k) {
    __shared__ float smem[ActiveT][WarpsPerRow];
    const bool second         = (static_cast<int>(blockIdx.x) >= n0);
    const int row             = second ? (static_cast<int>(blockIdx.x) - n0) : static_cast<int>(blockIdx.x);
    const int n               = second ? n1 : n0;
    if (row >= n) { return; }
    const __nv_bfloat16* x    = second ? x1 : x0;
    const __nv_bfloat16* w    = second ? w1 : w0;
    __nv_bfloat16* out        = second ? out1 : out0;

    const int warp            = threadIdx.x >> 5;
    const int lane            = threadIdx.x & 31;
    const __nv_bfloat16* wrow = w + static_cast<std::int64_t>(row) * k;
    const int k8              = k >> 3;
    float acc[ActiveT]        = {};

    for (int i = threadIdx.x; i < k8; i += WarpsPerRow * 32) {
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
                x + static_cast<std::int64_t>(ti) * k) + i);
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
        const float ws = warp_reduce_sum(acc[ti]);
        if (lane == 0) { smem[ti][warp] = ws; }
    }
    __syncthreads();

    if (warp == 0 && lane < ActiveT) {
        const int ti = lane;
        float sum    = 0.0f;
#pragma unroll
        for (int s = 0; s < WarpsPerRow; ++s) { sum += smem[ti][s]; }
        out[static_cast<std::int64_t>(ti) * n + row] = __float2bfloat16_rn(sum);
    }
}

// Flat scalar argument struct: ZERO array indexing -> ZERO 96-byte local stack frame!
struct Bf16MultiArgs {
    const __nv_bfloat16* x0;
    const __nv_bfloat16* x1;
    const __nv_bfloat16* x2;
    const __nv_bfloat16* x3;
    const __nv_bfloat16* x4;
    const __nv_bfloat16* w0;
    const __nv_bfloat16* w1;
    const __nv_bfloat16* w2;
    const __nv_bfloat16* w3;
    const __nv_bfloat16* w4;
    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    __nv_bfloat16* out2;
    __nv_bfloat16* out3;
    __nv_bfloat16* out4;
    int n0;
    int n1;
    int n2;
    int n3;
    int n4;
    int b0;
    int b01;
    int b012;
    int b0123;
};

template <int NumSegs>
__device__ __forceinline__ void select_multi_seg(const Bf16MultiArgs& args, int& blk,
                                                 const __nv_bfloat16*& x,
                                                 const __nv_bfloat16*& w,
                                                 __nv_bfloat16*& out, int& n) {
    if constexpr (NumSegs == 5) {
        if (blk >= args.b0123) {
            blk -= args.b0123;
            x = args.x4; w = args.w4; out = args.out4; n = args.n4;
            return;
        }
        if (blk >= args.b012) {
            blk -= args.b012;
            x = args.x3; w = args.w3; out = args.out3; n = args.n3;
            return;
        }
    }
    if (blk >= args.b01) {
        blk -= args.b01;
        x   = args.x2;
        w   = args.w2;
        out = args.out2;
        n   = args.n2;
    } else if (blk >= args.b0) {
        blk -= args.b0;
        x   = args.x1;
        w   = args.w1;
        out = args.out1;
        n   = args.n1;
    } else {
        x   = args.x0;
        w   = args.w0;
        out = args.out0;
        n   = args.n0;
    }
}

template <int NumSegs>
__global__ __launch_bounds__(128, 8) void bf16_multi_gemv_1rpw_pf2_kernel(
    Bf16MultiArgs args, int k) {
    int blk = blockIdx.x;
    const __nv_bfloat16 *x, *w;
    __nv_bfloat16* out;
    int n;
    select_multi_seg<NumSegs>(args, blk, x, w, out, n);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blk * 4 + warp;
    if (row >= n) { return; }

    const int k8    = k >> 3;
    const uint4* wp = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    const uint4* xp = reinterpret_cast<const uint4*>(x);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (lane < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + lane);
        uint4 x_cur = __ldg(xp + lane);
        for (int i = lane; i < k8; i += 32) {
            uint4 w_nxt{}, x_nxt{};
            if (i + 32 < k8) {
                w_nxt = ld_global_cg_u4(wp + i + 32);
                x_nxt = __ldg(xp + i + 32);
            }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&x_cur);
            const float2 w0 = __bfloat1622float2(wb[0]), x0 = __bfloat1622float2(xb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]), x1 = __bfloat1622float2(xb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]), x2 = __bfloat1622float2(xb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]), x3 = __bfloat1622float2(xb[3]);
            a0 = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, a0));
            a1 = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, a1));
            a2 = fmaf(w2.x, x2.x, fmaf(w2.y, x2.y, a2));
            a3 = fmaf(w3.x, x3.x, fmaf(w3.y, x3.y, a3));
            w_cur = w_nxt;
            x_cur = x_nxt;
        }
    }
    const float sum = warp_reduce_sum((a0 + a1) + (a2 + a3));
    if (lane == 0) { out[row] = __float2bfloat16_rn(sum); }
}

template <int NumSegs, int ActiveT>
__global__ __launch_bounds__(128, 6) void bf16_multi_small_t_16tpr_pf2_kernel(
    Bf16MultiArgs args, int k) {
    int blk = blockIdx.x;
    const __nv_bfloat16 *x, *w;
    __nv_bfloat16* out;
    int n;
    select_multi_seg<NumSegs>(args, blk, x, w, out, n);

    const int r_loc = threadIdx.x >> 4;
    const int kl    = threadIdx.x & 15;
    const int row   = blk * 8 + r_loc;
    if (row >= n) { return; }

    const int k8       = k >> 3;
    const uint4* wp    = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    float acc[ActiveT] = {};

    if (kl < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + kl);
        for (int i = kl; i < k8; i += 16) {
            uint4 w_nxt{};
            if (i + 16 < k8) { w_nxt = ld_global_cg_u4(wp + i + 16); }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const float2 w0 = __bfloat1622float2(wb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
            for (int ti = 0; ti < ActiveT; ++ti) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(ti) * k) + i);
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
    }

#pragma unroll
    for (int ti = 0; ti < ActiveT; ++ti) {
        float sum = acc[ti];
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 8, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1, 16);
        if (kl == 0) { out[static_cast<std::int64_t>(ti) * n + row] = __float2bfloat16_rn(sum); }
    }
}

template <int NumSegs>
__global__ __launch_bounds__(128, 4) void bf16_multi_t9_16_16tpr_pf2_kernel(
    Bf16MultiArgs args, int k, int t) {
    int blk = blockIdx.x;
    const __nv_bfloat16 *x, *w;
    __nv_bfloat16* out;
    int n;
    select_multi_seg<NumSegs>(args, blk, x, w, out, n);

    const int r_loc = threadIdx.x >> 4;
    const int kl    = threadIdx.x & 15;
    const int row   = blk * 8 + r_loc;
    if (row >= n) { return; }

    const int k8    = k >> 3;
    const uint4* wp = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
    float acc[16]   = {};

    if (kl < k8) {
        uint4 w_cur = ld_global_cg_u4(wp + kl);
        for (int i = kl; i < k8; i += 16) {
            uint4 w_nxt{};
            if (i + 16 < k8) { w_nxt = ld_global_cg_u4(wp + i + 16); }
            const auto* wb  = reinterpret_cast<const __nv_bfloat162*>(&w_cur);
            const float2 w0 = __bfloat1622float2(wb[0]);
            const float2 w1 = __bfloat1622float2(wb[1]);
            const float2 w2 = __bfloat1622float2(wb[2]);
            const float2 w3 = __bfloat1622float2(wb[3]);
            for (int ti = 0; ti < t; ++ti) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(ti) * k) + i);
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
    }

    for (int ti = 0; ti < t; ++ti) {
        float sum = acc[ti];
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 8, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2, 16);
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1, 16);
        if (kl == 0) { out[static_cast<std::int64_t>(ti) * n + row] = __float2bfloat16_rn(sum); }
    }
}

template <int NumSegs, int kTiles, int SplitK>
__global__ __launch_bounds__(
    SplitK * 32, (kTiles <= 2 ? 16 : 4) / SplitK < 1 ? 1 : (kTiles <= 2 ? 16 : 4) / SplitK)
void bf16_multi_volta_qpn_ktiles_kernel(Bf16MultiArgs args, int k, int t) {
    __shared__ float partial[SplitK][kTiles * 8 * 32];
    int blk = blockIdx.x;
    const __nv_bfloat16 *x, *w;
    __nv_bfloat16* out;
    int n;
    select_multi_seg<NumSegs>(args, blk, x, w, out, n);

    const int lane    = threadIdx.x & 31;
    const int warp    = threadIdx.x >> 5;
    const int qp      = (lane >> 2) & 3;
    const int r       = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col     = blk * 32 + qp * 8 + r;
    const int use_col = (col < n) ? col : 0;

    const int blocks = k / 64;
    const int begin  = warp * (blocks / SplitK) + min(warp, blocks % SplitK);
    const int end    = begin + blocks / SplitK + (warp < blocks % SplitK);

    float c[kTiles][2][8] = {};
    for (int block = begin; block < end; ++block) {
        uint4 words[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            words[e] = ld_global_cg_u4(
                reinterpret_cast<const uint4*>(
                    w + static_cast<std::int64_t>(use_col) * k + block * 64 + e * 8));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const auto* wb = reinterpret_cast<const __nv_bfloat162*>(&words[e]);
            half2 bh[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(wb[j]);
                bh[j]          = __floats2half2_rn(f.x, f.y);
            }
            const auto* bb = reinterpret_cast<const unsigned*>(bh);
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int t_row = tile * 8 + r;
                half2 ah[4];
                if (t_row < t) {
                    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                        x + static_cast<std::int64_t>(t_row) * k + block * 64 + e * 8));
                    const auto* xb  = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __bfloat1622float2(xb[j]);
                        ah[j]          = __floats2half2_rn(f.x, f.y);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { ah[j] = __half2half2(__ushort_as_half(0)); }
                }
                const auto* aa = reinterpret_cast<const unsigned*>(ah);
                volta_mma_qp_n(c[tile][0], aa[0], aa[1], bb[0], bb[1]);
                volta_mma_qp_n(c[tile][1], aa[2], aa[3], bb[2], bb[3]);
            }
        }
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            partial[warp][(tile * 8 + row) * 32 + qp * 8 + cl] = c[tile][0][i] + c[tile][1][i];
        }
    }
    __syncthreads();

    constexpr int kOut = kTiles * 8 * 32;
    for (int e = threadIdx.x; e < kOut; e += SplitK * 32) {
        const int grow = e >> 5;
        const int cl   = e & 31;
        const int gcol = blk * 32 + cl;
        if (grow < t && gcol < n) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) { sum += partial[s][e]; }
            out[static_cast<std::int64_t>(grow) * n + gcol] = __float2bfloat16_rn(sum);
        }
    }
}

template <int NumSegs>
void dispatch_bf16_multi(const Bf16MultiArgs& base_args, int k, int t, cudaStream_t stream) {
    auto set_blocks = [&](Bf16MultiArgs& a, int rows_per_blk) -> int {
        a.b0         = (a.n0 + rows_per_blk - 1) / rows_per_blk;
        const int b1 = (a.n1 + rows_per_blk - 1) / rows_per_blk;
        const int b2 = (a.n2 + rows_per_blk - 1) / rows_per_blk;
        a.b01        = a.b0 + b1;
        if constexpr (NumSegs == 3) {
            return a.b01 + b2;
        } else {
            a.b012       = a.b01 + b2;
            const int b3 = (a.n3 + rows_per_blk - 1) / rows_per_blk;
            const int b4 = (a.n4 + rows_per_blk - 1) / rows_per_blk;
            a.b0123      = a.b012 + b3;
            return a.b0123 + b4;
        }
    };

    Bf16MultiArgs args = base_args;
    if (t == 1) {
        const int g = set_blocks(args, 4);
        bf16_multi_gemv_1rpw_pf2_kernel<NumSegs><<<g, 128, 0, stream>>>(args, k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t <= 8) {
        const int g = set_blocks(args, 8);
        switch (t) {
        case 2: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 2><<<g, 128, 0, stream>>>(args, k); break;
        case 3: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 3><<<g, 128, 0, stream>>>(args, k); break;
        case 4: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 4><<<g, 128, 0, stream>>>(args, k); break;
        case 5: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 5><<<g, 128, 0, stream>>>(args, k); break;
        case 6: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 6><<<g, 128, 0, stream>>>(args, k); break;
        case 7: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 7><<<g, 128, 0, stream>>>(args, k); break;
        case 8: bf16_multi_small_t_16tpr_pf2_kernel<NumSegs, 8><<<g, 128, 0, stream>>>(args, k); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (t <= 16) {
        const int g = set_blocks(args, 8);
        bf16_multi_t9_16_16tpr_pf2_kernel<NumSegs><<<g, 128, 0, stream>>>(args, k, t);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int g = set_blocks(args, 32);
    if (t <= 24) {
        bf16_multi_volta_qpn_ktiles_kernel<NumSegs, 3, 4><<<g, 128, 0, stream>>>(args, k, t);
    } else {
        bf16_multi_volta_qpn_ktiles_kernel<NumSegs, 4, 4><<<g, 128, 0, stream>>>(args, k, t);
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_bf16_linear_pair(const Tensor& x0, const Tensor& w0, Tensor& out0,
                               const Tensor& x1, const Tensor& w1, Tensor& out1,
                               cudaStream_t stream) {
    const int k  = x0.ne[0];
    const int t  = x0.ne[1];
    const int n0 = w0.ne[1];
    const int n1 = w1.ne[1];
    if (x0.dtype == DType::BF16 && x1.dtype == DType::BF16 && out0.dtype == DType::BF16 &&
        out1.dtype == DType::BF16 && x1.ne[0] == k && x1.ne[1] == t && (k % 64 == 0) &&
        t >= 1 && t <= 8 && n0 <= 512 && n1 <= 512 && k >= 512) {
        const auto* xp0 = static_cast<const __nv_bfloat16*>(x0.data);
        const auto* wp0 = static_cast<const __nv_bfloat16*>(w0.data);
        auto* op0       = static_cast<__nv_bfloat16*>(out0.data);
        const auto* xp1 = static_cast<const __nv_bfloat16*>(x1.data);
        const auto* wp1 = static_cast<const __nv_bfloat16*>(w1.data);
        auto* op1       = static_cast<__nv_bfloat16*>(out1.data);
        const int grid  = n0 + n1;
        switch (t) {
        case 1: bf16_pair_small_t_splitk_kernel<1, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 2: bf16_pair_small_t_splitk_kernel<2, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 3: bf16_pair_small_t_splitk_kernel<3, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 4: bf16_pair_small_t_splitk_kernel<4, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 5: bf16_pair_small_t_splitk_kernel<5, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 6: bf16_pair_small_t_splitk_kernel<6, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 7: bf16_pair_small_t_splitk_kernel<7, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        case 8: bf16_pair_small_t_splitk_kernel<8, 8><<<grid, 256, 0, stream>>>(xp0, wp0, op0, n0, xp1, wp1, op1, n1, k); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    inkling2_bf16_linear(x0, w0, out0, stream);
    inkling2_bf16_linear(x1, w1, out1, stream);
}

void inkling2_bf16_linear_triple(const Tensor& x0, const Tensor& w0, Tensor& out0,
                                 const Tensor& x1, const Tensor& w1, Tensor& out1,
                                 const Tensor& x2, const Tensor& w2, Tensor& out2,
                                 cudaStream_t stream) {
    const int k = x0.ne[0];
    const int t = x0.ne[1];
    if (x0.dtype == DType::BF16 && x1.dtype == DType::BF16 && x2.dtype == DType::BF16 &&
        out0.dtype == DType::BF16 && out1.dtype == DType::BF16 && out2.dtype == DType::BF16 &&
        x1.ne[0] == k && x2.ne[0] == k && x1.ne[1] == t && x2.ne[1] == t &&
        (k % 64 == 0) && t >= 1 && t <= 32) {
        Bf16MultiArgs args{};
        args.x0   = static_cast<const __nv_bfloat16*>(x0.data);
        args.x1   = static_cast<const __nv_bfloat16*>(x1.data);
        args.x2   = static_cast<const __nv_bfloat16*>(x2.data);
        args.w0   = static_cast<const __nv_bfloat16*>(w0.data);
        args.w1   = static_cast<const __nv_bfloat16*>(w1.data);
        args.w2   = static_cast<const __nv_bfloat16*>(w2.data);
        args.out0 = static_cast<__nv_bfloat16*>(out0.data);
        args.out1 = static_cast<__nv_bfloat16*>(out1.data);
        args.out2 = static_cast<__nv_bfloat16*>(out2.data);
        args.n0   = w0.ne[1];
        args.n1   = w1.ne[1];
        args.n2   = w2.ne[1];
        dispatch_bf16_multi<3>(args, k, t, stream);
        return;
    }
    inkling2_bf16_linear(x0, w0, out0, stream);
    inkling2_bf16_linear(x1, w1, out1, stream);
    inkling2_bf16_linear(x2, w2, out2, stream);
}

void inkling2_bf16_linear_5way(const Tensor& x0, const Tensor& w0, Tensor& out0,
                               const Tensor& x1, const Tensor& w1, Tensor& out1,
                               const Tensor& x2, const Tensor& w2, Tensor& out2,
                               const Tensor& x3, const Tensor& w3, Tensor& out3,
                               const Tensor& x4, const Tensor& w4, Tensor& out4,
                               cudaStream_t stream) {
    const int k = x0.ne[0];
    const int t = x0.ne[1];
    if (x0.dtype == DType::BF16 && x1.dtype == DType::BF16 && x2.dtype == DType::BF16 &&
        x3.dtype == DType::BF16 && x4.dtype == DType::BF16 &&
        out0.dtype == DType::BF16 && out1.dtype == DType::BF16 && out2.dtype == DType::BF16 &&
        out3.dtype == DType::BF16 && out4.dtype == DType::BF16 &&
        x1.ne[0] == k && x2.ne[0] == k && x3.ne[0] == k && x4.ne[0] == k &&
        x1.ne[1] == t && x2.ne[1] == t && x3.ne[1] == t && x4.ne[1] == t &&
        (k % 64 == 0) && t >= 1 && t <= 32) {
        Bf16MultiArgs args{};
        args.x0   = static_cast<const __nv_bfloat16*>(x0.data);
        args.x1   = static_cast<const __nv_bfloat16*>(x1.data);
        args.x2   = static_cast<const __nv_bfloat16*>(x2.data);
        args.x3   = static_cast<const __nv_bfloat16*>(x3.data);
        args.x4   = static_cast<const __nv_bfloat16*>(x4.data);
        args.w0   = static_cast<const __nv_bfloat16*>(w0.data);
        args.w1   = static_cast<const __nv_bfloat16*>(w1.data);
        args.w2   = static_cast<const __nv_bfloat16*>(w2.data);
        args.w3   = static_cast<const __nv_bfloat16*>(w3.data);
        args.w4   = static_cast<const __nv_bfloat16*>(w4.data);
        args.out0 = static_cast<__nv_bfloat16*>(out0.data);
        args.out1 = static_cast<__nv_bfloat16*>(out1.data);
        args.out2 = static_cast<__nv_bfloat16*>(out2.data);
        args.out3 = static_cast<__nv_bfloat16*>(out3.data);
        args.out4 = static_cast<__nv_bfloat16*>(out4.data);
        args.n0   = w0.ne[1];
        args.n1   = w1.ne[1];
        args.n2   = w2.ne[1];
        args.n3   = w3.ne[1];
        args.n4   = w4.ne[1];
        dispatch_bf16_multi<5>(args, k, t, stream);
        return;
    }
    inkling2_bf16_linear_triple(x0, w0, out0, x1, w1, out1, x2, w2, out2, stream);
    inkling2_bf16_linear_pair(x3, w3, out3, x4, w4, out4, stream);
}

void inkling2_unembed_project(const Tensor& x, const Tensor& w, float softcap,
                              std::int32_t valid_shard_cols, Tensor* logits_out,
                              Tensor* max_vals, Tensor* max_ids, cudaStream_t stream) {
    const int k = x.ne[0];
    const int t = x.ne[1];
    const int n = w.ne[1];
    if (x.dtype != DType::BF16 || w.dtype != DType::BF16 || w.ne[0] != k) {
        throw std::invalid_argument("inkling2_unembed_project: invalid x/w");
    }
    const int valid = std::max(0, std::min(valid_shard_cols, n));
    const auto* xp  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* wp  = static_cast<const __nv_bfloat16*>(w.data);

    // Fast fused argmax-only path (NO FP32 logits written, NO tanhf in vocab loop!) for ALL T = 1..64!
    if (logits_out == nullptr && max_vals != nullptr && max_ids != nullptr) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        auto* mv = static_cast<float*>(max_vals->data);
        auto* mi = static_cast<std::int32_t*>(max_ids->data);
        if (t <= 8) {
            auto& sc = get_argmax_scratch(dev, static_cast<std::size_t>(t) * 320);
            switch (t) {
            case 1: launch_unembed_small_t_argmax<1>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 2: launch_unembed_small_t_argmax<2>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 3: launch_unembed_small_t_argmax<3>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 4: launch_unembed_small_t_argmax<4>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 5: launch_unembed_small_t_argmax<5>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 6: launch_unembed_small_t_argmax<6>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 7: launch_unembed_small_t_argmax<7>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            case 8: launch_unembed_small_t_argmax<8>(xp, wp, valid, k, softcap, sc, mv, mi, stream); return;
            }
        }
        // T > 8: chunk in groups of up to 32 rows using multi-tile QPN fused argmax!
        for (int t0 = 0; t0 < t; t0 += 32) {
            const int tb         = std::min(32, t - t0);
            const int num_blocks = std::min(320, (valid + 31) / 32);
            auto& sc             = get_argmax_scratch(dev, static_cast<std::size_t>(tb) * num_blocks);
            const auto* xsub     = xp + static_cast<std::int64_t>(t0) * k;
            if (tb <= 8) {
                unembed_qpn_argmax_kernel<1, 4><<<num_blocks, 128, 0, stream>>>(
                    xsub, wp, valid, k, tb, num_blocks, sc.vals, sc.ids);
            } else if (tb <= 16) {
                unembed_qpn_argmax_kernel<2, 4><<<num_blocks, 128, 0, stream>>>(
                    xsub, wp, valid, k, tb, num_blocks, sc.vals, sc.ids);
            } else if (tb <= 24) {
                unembed_qpn_argmax_kernel<3, 4><<<num_blocks, 128, 0, stream>>>(
                    xsub, wp, valid, k, tb, num_blocks, sc.vals, sc.ids);
            } else {
                unembed_qpn_argmax_kernel<4, 4><<<num_blocks, 128, 0, stream>>>(
                    xsub, wp, valid, k, tb, num_blocks, sc.vals, sc.ids);
            }
            CUDA_CHECK(cudaGetLastError());
            reduce_block_argmax_kernel<<<tb, 256, 0, stream>>>(
                sc.vals, sc.ids, num_blocks, num_blocks, softcap, mv + t0, mi + t0);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    if (logits_out == nullptr || logits_out->dtype != DType::FP32) {
        throw std::invalid_argument("inkling2_unembed_project: requires FP32 logits_out or argmax outputs");
    }
    dispatch_bf16_linear<float>(xp, wp, static_cast<float*>(logits_out->data), nullptr, nullptr, n, k, t, stream);
    softcap_mask_argmax_f32_kernel<<<t, 256, 0, stream>>>(
        static_cast<float*>(logits_out->data), n, t, softcap, valid,
        max_vals ? static_cast<float*>(max_vals->data) : nullptr,
        max_ids ? static_cast<std::int32_t*>(max_ids->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_markov_logits_argmax(const Tensor& base_logits, const Tensor& latent,
                                   const Tensor& fold_w, const Tensor& fold_b,
                                   std::int32_t valid_shard_cols, Tensor& max_vals,
                                   Tensor& max_ids, cudaStream_t stream) {
    inkling2_markov_logits_argmax_strided(base_logits, latent, fold_w, fold_b, valid_shard_cols,
                                          1, 0, max_vals, max_ids, stream);
}

void inkling2_markov_logits_argmax_strided(const Tensor& base_logits, const Tensor& latent,
                                           const Tensor& fold_w, const Tensor& fold_b,
                                           std::int32_t valid_shard_cols,
                                           std::int32_t base_row_stride,
                                           std::int32_t base_row_offset, Tensor& max_vals,
                                           Tensor& max_ids, cudaStream_t stream) {
    const int n_shard = base_logits.ne[0];
    const int batch   = latent.ne[1];
    const int valid   = std::max(0, std::min(valid_shard_cols, n_shard));
    if (latent.ne[0] != 256 || batch <= 0 || batch > 8 || fold_w.ne[0] != 256 ||
        fold_w.ne[1] != n_shard || fold_b.ne[0] != n_shard || fold_b.dtype != DType::FP32) {
        throw std::invalid_argument("inkling2_markov_logits_argmax: shape/dtype mismatch");
    }
    constexpr int kBlocksN = 160;
    int dev                = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    auto& scratch = get_argmax_scratch(dev, static_cast<std::size_t>(batch) * kBlocksN);

    const auto* fw = static_cast<const __nv_bfloat16*>(fold_w.data);
    const auto* fb = static_cast<const float*>(fold_b.data);

    auto launch = [&]<typename BaseT, typename LatentT>() {
        const auto* bp = static_cast<const BaseT*>(base_logits.data);
        const auto* lp = static_cast<const LatentT*>(latent.data);
        if (batch == 1) {
            markov_logits_argmax_batched_kernel<BaseT, LatentT, 1><<<kBlocksN, 256, 0, stream>>>(
                bp, lp, fw, fb, n_shard, valid, batch, base_row_stride, base_row_offset,
                scratch.vals, scratch.ids, kBlocksN);
        } else if (batch <= 4) {
            markov_logits_argmax_batched_kernel<BaseT, LatentT, 4><<<kBlocksN, 256, 0, stream>>>(
                bp, lp, fw, fb, n_shard, valid, batch, base_row_stride, base_row_offset,
                scratch.vals, scratch.ids, kBlocksN);
        } else {
            markov_logits_argmax_batched_kernel<BaseT, LatentT, 8><<<kBlocksN, 256, 0, stream>>>(
                bp, lp, fw, fb, n_shard, valid, batch, base_row_stride, base_row_offset,
                scratch.vals, scratch.ids, kBlocksN);
        }
    };

    if (base_logits.dtype == DType::BF16 && latent.dtype == DType::BF16) {
        launch.template operator()<__nv_bfloat16, __nv_bfloat16>();
    } else if (base_logits.dtype == DType::FP32 && latent.dtype == DType::BF16) {
        launch.template operator()<float, __nv_bfloat16>();
    } else if (base_logits.dtype == DType::FP32 && latent.dtype == DType::FP32) {
        launch.template operator()<float, float>();
    } else {
        launch.template operator()<__nv_bfloat16, float>();
    }
    CUDA_CHECK(cudaGetLastError());

    reduce_block_argmax_kernel<<<batch, 256, 0, stream>>>(
        scratch.vals, scratch.ids, kBlocksN, kBlocksN, 0.0f, static_cast<float*>(max_vals.data),
        static_cast<std::int32_t*>(max_ids.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
