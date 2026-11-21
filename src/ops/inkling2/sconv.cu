#include "ninfer/ops/inkling2.h"

#include "core/device.h"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr float kInvSqrt5 = 0.44721359549995793928f; // 1 / sqrt(5)
constexpr float kInvSqrt9 = 0.33333333333333333333f; // 1 / 3 = 1 / sqrt(9)

__device__ __forceinline__ Bf16x8Pack pkt_load(const Bf16x8Pack* address) {
    uint4 bits;
    asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(address));
    return *reinterpret_cast<const Bf16x8Pack*>(&bits);
}

__device__ __forceinline__ void pkt_store(Bf16x8Pack* address, const Bf16x8Pack& value) {
    const auto& bits = *reinterpret_cast<const uint4*>(&value);
    asm volatile("st.volatile.global.v4.b32 [%4],{%0,%1,%2,%3};"
                 :
                 : "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w), "l"(address)
                 : "memory");
}

// ============================================================================
// 1. Static causal short convolution (W = 4)
// ============================================================================

__global__ void sconv_static_prefill_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ weight,
                                            float* __restrict__ state,
                                            __nv_bfloat16* __restrict__ out,
                                            const __nv_bfloat16* __restrict__ residual_add,
                                            int channels, int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) { return; }

    const float w0 = __bfloat162float(__ldg(weight + c * 4 + 0));
    const float w1 = __bfloat162float(__ldg(weight + c * 4 + 1));
    const float w2 = __bfloat162float(__ldg(weight + c * 4 + 2));
    const float w3 = __bfloat162float(__ldg(weight + c * 4 + 3));

    float s0 = state[0 * channels + c];
    float s1 = state[1 * channels + c];
    float s2 = state[2 * channels + c];

    for (int t = 0; t < tokens; ++t) {
        const std::int64_t off = static_cast<std::int64_t>(t) * channels + c;
        const float xt         = __bfloat162float(x[off]);
        float y                = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * kInvSqrt5;
        if (residual_add != nullptr) { y += __bfloat162float(residual_add[off]); }
        out[off] = __float2bfloat16_rn(y);
        s0       = s1;
        s1       = s2;
        s2       = xt;
    }

    state[0 * channels + c] = s0;
    state[1 * channels + c] = s1;
    state[2 * channels + c] = s2;
}

__global__ void sconv_static_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ weight,
                                           float* __restrict__ state_pool,
                                           const std::int32_t* __restrict__ lane_ids,
                                           __nv_bfloat16* __restrict__ out,
                                           const __nv_bfloat16* __restrict__ residual_add,
                                           int channels, int batch_size,
                                           std::size_t lane_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const int lane       = lane_ids[b];
    float* st            = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    const float w0       = __bfloat162float(__ldg(weight + c * 4 + 0));
    const float w1       = __bfloat162float(__ldg(weight + c * 4 + 1));
    const float w2       = __bfloat162float(__ldg(weight + c * 4 + 2));
    const float w3       = __bfloat162float(__ldg(weight + c * 4 + 3));
    const float s0       = st[0 * channels + c];
    const float s1       = st[1 * channels + c];
    const float s2       = st[2 * channels + c];
    const std::int64_t o = static_cast<std::int64_t>(b) * channels + c;
    const float xt       = __bfloat162float(x[o]);

    float y = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * kInvSqrt5;
    if (residual_add != nullptr) { y += __bfloat162float(residual_add[o]); }
    out[o] = __float2bfloat16_rn(y);

    st[0 * channels + c] = s1;
    st[1 * channels + c] = s2;
    st[2 * channels + c] = xt;
}

__global__ void sconv_static_verify_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ weight,
                                           const float* __restrict__ state_pool,
                                           float* __restrict__ verify_pool,
                                           const std::int32_t* __restrict__ lane_ids,
                                           __nv_bfloat16* __restrict__ out,
                                           const __nv_bfloat16* __restrict__ residual_add,
                                           int channels, int batch_size, int verify_len,
                                           std::size_t lane_stride, int max_verify_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const int lane     = lane_ids[b];
    const float* st_in = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    float* vst_lane =
        verify_pool + static_cast<std::size_t>(lane) * static_cast<std::size_t>(max_verify_len) * lane_stride;

    const float w0 = __bfloat162float(__ldg(weight + c * 4 + 0));
    const float w1 = __bfloat162float(__ldg(weight + c * 4 + 1));
    const float w2 = __bfloat162float(__ldg(weight + c * 4 + 2));
    const float w3 = __bfloat162float(__ldg(weight + c * 4 + 3));

    float s0 = st_in[0 * channels + c];
    float s1 = st_in[1 * channels + c];
    float s2 = st_in[2 * channels + c];

    for (int v = 0; v < verify_len; ++v) {
        const std::int64_t off = static_cast<std::int64_t>(b * verify_len + v) * channels + c;
        const float xt         = __bfloat162float(x[off]);
        float y                = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * kInvSqrt5;
        if (residual_add != nullptr) { y += __bfloat162float(residual_add[off]); }
        out[off] = __float2bfloat16_rn(y);
        s0       = s1;
        s1       = s2;
        s2       = xt;

        float* vst            = vst_lane + static_cast<std::size_t>(v) * lane_stride;
        vst[0 * channels + c] = s0;
        vst[1 * channels + c] = s1;
        vst[2 * channels + c] = s2;
    }
}

// Fused K+V Static Pair (grid.z = 2 -> 48 fewer launches/step!)
__global__ void sconv_kv_static_decode_kernel(
    const __nv_bfloat16* __restrict__ k_in, const __nv_bfloat16* __restrict__ wk, float* __restrict__ st_k,
    const __nv_bfloat16* __restrict__ v_in, const __nv_bfloat16* __restrict__ wv, float* __restrict__ st_v,
    const std::int32_t* __restrict__ lane_ids, __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v, int channels, int batch_size, std::size_t lane_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const bool is_v             = (blockIdx.z == 1);
    const __nv_bfloat16* x      = is_v ? v_in : k_in;
    const __nv_bfloat16* weight = is_v ? wv : wk;
    float* state_pool           = is_v ? st_v : st_k;
    __nv_bfloat16* out          = is_v ? out_v : out_k;

    const int lane       = lane_ids[b];
    float* st            = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    const float w0       = __bfloat162float(__ldg(weight + c * 4 + 0));
    const float w1       = __bfloat162float(__ldg(weight + c * 4 + 1));
    const float w2       = __bfloat162float(__ldg(weight + c * 4 + 2));
    const float w3       = __bfloat162float(__ldg(weight + c * 4 + 3));
    const float s0       = st[0 * channels + c];
    const float s1       = st[1 * channels + c];
    const float s2       = st[2 * channels + c];
    const std::int64_t o = static_cast<std::int64_t>(b) * channels + c;
    const float xt       = __bfloat162float(x[o]);

    out[o]               = __float2bfloat16_rn((w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * kInvSqrt5);
    st[0 * channels + c] = s1;
    st[1 * channels + c] = s2;
    st[2 * channels + c] = xt;
}

__global__ void sconv_kv_static_verify_kernel(
    const __nv_bfloat16* __restrict__ k_in, const __nv_bfloat16* __restrict__ wk,
    const float* __restrict__ st_k, float* __restrict__ vst_k,
    const __nv_bfloat16* __restrict__ v_in, const __nv_bfloat16* __restrict__ wv,
    const float* __restrict__ st_v, float* __restrict__ vst_v,
    const std::int32_t* __restrict__ lane_ids, __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v, int channels, int batch_size, int verify_len,
    std::size_t lane_stride, int max_verify_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const bool is_v             = (blockIdx.z == 1);
    const __nv_bfloat16* x      = is_v ? v_in : k_in;
    const __nv_bfloat16* weight = is_v ? wv : wk;
    const float* state_pool     = is_v ? st_v : st_k;
    float* verify_pool          = is_v ? vst_v : vst_k;
    __nv_bfloat16* out          = is_v ? out_v : out_k;

    const int lane     = lane_ids[b];
    const float* st_in = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    float* vst_lane =
        verify_pool + static_cast<std::size_t>(lane) * static_cast<std::size_t>(max_verify_len) * lane_stride;

    const float w0 = __bfloat162float(__ldg(weight + c * 4 + 0));
    const float w1 = __bfloat162float(__ldg(weight + c * 4 + 1));
    const float w2 = __bfloat162float(__ldg(weight + c * 4 + 2));
    const float w3 = __bfloat162float(__ldg(weight + c * 4 + 3));

    float s0 = st_in[0 * channels + c];
    float s1 = st_in[1 * channels + c];
    float s2 = st_in[2 * channels + c];

    for (int v = 0; v < verify_len; ++v) {
        const std::int64_t off = static_cast<std::int64_t>(b * verify_len + v) * channels + c;
        const float xt         = __bfloat162float(x[off]);
        out[off]               = __float2bfloat16_rn((w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * kInvSqrt5);
        s0                     = s1;
        s1                     = s2;
        s2                     = xt;

        float* vst            = vst_lane + static_cast<std::size_t>(v) * lane_stride;
        vst[0 * channels + c] = s0;
        vst[1 * channels + c] = s1;
        vst[2 * channels + c] = s2;
    }
}

// ============================================================================
// 2. Dynamic causal short convolution (W = 4)
// ============================================================================

__global__ void sconv_dynamic_prefill_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ dyn_w,
                                             const __nv_bfloat16* __restrict__ static_w,
                                             float* __restrict__ state,
                                             __nv_bfloat16* __restrict__ out,
                                             int channels, int num_kv_heads, int tokens) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) { return; }

    const int head_dim = channels / num_kv_heads;
    const int head     = c / head_dim;

    const float sw0 = __bfloat162float(__ldg(static_w + c * 4 + 0));
    const float sw1 = __bfloat162float(__ldg(static_w + c * 4 + 1));
    const float sw2 = __bfloat162float(__ldg(static_w + c * 4 + 2));
    const float sw3 = __bfloat162float(__ldg(static_w + c * 4 + 3));

    float s0 = state[0 * channels + c];
    float s1 = state[1 * channels + c];
    float s2 = state[2 * channels + c];

    for (int t = 0; t < tokens; ++t) {
        const std::int64_t off  = static_cast<std::int64_t>(t) * channels + c;
        const __nv_bfloat16* dw = dyn_w + (static_cast<std::int64_t>(t) * num_kv_heads + head) * 4;
        const float w0          = sw0 + __bfloat162float(__ldg(dw + 0));
        const float w1          = sw1 + __bfloat162float(__ldg(dw + 1));
        const float w2          = sw2 + __bfloat162float(__ldg(dw + 2));
        const float w3          = sw3 + __bfloat162float(__ldg(dw + 3));

        const float xt = __bfloat162float(x[off]);
        out[off]       = __float2bfloat16_rn((w0 * xt + w1 * s2 + w2 * s1 + w3 * s0 + xt) * kInvSqrt9);
        s0             = s1;
        s1             = s2;
        s2             = xt;
    }

    state[0 * channels + c] = s0;
    state[1 * channels + c] = s1;
    state[2 * channels + c] = s2;
}

__global__ void sconv_dynamic_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ dyn_w,
                                            const __nv_bfloat16* __restrict__ static_w,
                                            float* __restrict__ state_pool,
                                            const std::int32_t* __restrict__ lane_ids,
                                            __nv_bfloat16* __restrict__ out,
                                            int channels, int num_kv_heads, int batch_size,
                                            std::size_t lane_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const int lane     = lane_ids[b];
    const int head_dim = channels / num_kv_heads;
    const int head     = c / head_dim;
    float* st          = state_pool + static_cast<std::size_t>(lane) * lane_stride;

    const __nv_bfloat16* dw = dyn_w + (static_cast<std::int64_t>(b) * num_kv_heads + head) * 4;
    const float w0 = __bfloat162float(__ldg(static_w + c * 4 + 0)) + __bfloat162float(__ldg(dw + 0));
    const float w1 = __bfloat162float(__ldg(static_w + c * 4 + 1)) + __bfloat162float(__ldg(dw + 1));
    const float w2 = __bfloat162float(__ldg(static_w + c * 4 + 2)) + __bfloat162float(__ldg(dw + 2));
    const float w3 = __bfloat162float(__ldg(static_w + c * 4 + 3)) + __bfloat162float(__ldg(dw + 3));

    const float s0       = st[0 * channels + c];
    const float s1       = st[1 * channels + c];
    const float s2       = st[2 * channels + c];
    const std::int64_t o = static_cast<std::int64_t>(b) * channels + c;
    const float xt       = __bfloat162float(x[o]);

    out[o]               = __float2bfloat16_rn((w0 * xt + w1 * s2 + w2 * s1 + w3 * s0 + xt) * kInvSqrt9);
    st[0 * channels + c] = s1;
    st[1 * channels + c] = s2;
    st[2 * channels + c] = xt;
}

__global__ void sconv_dynamic_verify_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ dyn_w,
                                            const __nv_bfloat16* __restrict__ static_w,
                                            const float* __restrict__ state_pool,
                                            float* __restrict__ verify_pool,
                                            const std::int32_t* __restrict__ lane_ids,
                                            __nv_bfloat16* __restrict__ out,
                                            int channels, int num_kv_heads, int batch_size,
                                            int verify_len, std::size_t lane_stride,
                                            int max_verify_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const int lane     = lane_ids[b];
    const int head_dim = channels / num_kv_heads;
    const int head     = c / head_dim;
    const float* st_in = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    float* vst_lane =
        verify_pool + static_cast<std::size_t>(lane) * static_cast<std::size_t>(max_verify_len) * lane_stride;

    const float sw0 = __bfloat162float(__ldg(static_w + c * 4 + 0));
    const float sw1 = __bfloat162float(__ldg(static_w + c * 4 + 1));
    const float sw2 = __bfloat162float(__ldg(static_w + c * 4 + 2));
    const float sw3 = __bfloat162float(__ldg(static_w + c * 4 + 3));

    float s0 = st_in[0 * channels + c];
    float s1 = st_in[1 * channels + c];
    float s2 = st_in[2 * channels + c];

    for (int v = 0; v < verify_len; ++v) {
        const int tok           = b * verify_len + v;
        const std::int64_t off  = static_cast<std::int64_t>(tok) * channels + c;
        const __nv_bfloat16* dw = dyn_w + (static_cast<std::int64_t>(tok) * num_kv_heads + head) * 4;
        const float w0          = sw0 + __bfloat162float(__ldg(dw + 0));
        const float w1          = sw1 + __bfloat162float(__ldg(dw + 1));
        const float w2          = sw2 + __bfloat162float(__ldg(dw + 2));
        const float w3          = sw3 + __bfloat162float(__ldg(dw + 3));

        const float xt = __bfloat162float(x[off]);
        out[off]       = __float2bfloat16_rn((w0 * xt + w1 * s2 + w2 * s1 + w3 * s0 + xt) * kInvSqrt9);
        s0             = s1;
        s1             = s2;
        s2             = xt;

        float* vst            = vst_lane + static_cast<std::size_t>(v) * lane_stride;
        vst[0 * channels + c] = s0;
        vst[1 * channels + c] = s1;
        vst[2 * channels + c] = s2;
    }
}

__global__ void sconv_kv_dynamic_decode_kernel(
    const __nv_bfloat16* __restrict__ k_in, const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ wk, float* __restrict__ st_k,
    const __nv_bfloat16* __restrict__ v_in, const __nv_bfloat16* __restrict__ dv,
    const __nv_bfloat16* __restrict__ wv, float* __restrict__ st_v,
    const std::int32_t* __restrict__ lane_ids, __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v, int channels, int num_kv_heads, int batch_size,
    std::size_t lane_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const bool is_v               = (blockIdx.z == 1);
    const __nv_bfloat16* x        = is_v ? v_in : k_in;
    const __nv_bfloat16* dyn_w    = is_v ? dv : dk;
    const __nv_bfloat16* static_w = is_v ? wv : wk;
    float* state_pool             = is_v ? st_v : st_k;
    __nv_bfloat16* out            = is_v ? out_v : out_k;

    const int lane     = lane_ids[b];
    const int head_dim = channels / num_kv_heads;
    const int head     = c / head_dim;
    float* st          = state_pool + static_cast<std::size_t>(lane) * lane_stride;

    const __nv_bfloat16* dw = dyn_w + (static_cast<std::int64_t>(b) * num_kv_heads + head) * 4;
    const float w0 = __bfloat162float(__ldg(static_w + c * 4 + 0)) + __bfloat162float(__ldg(dw + 0));
    const float w1 = __bfloat162float(__ldg(static_w + c * 4 + 1)) + __bfloat162float(__ldg(dw + 1));
    const float w2 = __bfloat162float(__ldg(static_w + c * 4 + 2)) + __bfloat162float(__ldg(dw + 2));
    const float w3 = __bfloat162float(__ldg(static_w + c * 4 + 3)) + __bfloat162float(__ldg(dw + 3));

    const float s0       = st[0 * channels + c];
    const float s1       = st[1 * channels + c];
    const float s2       = st[2 * channels + c];
    const std::int64_t o = static_cast<std::int64_t>(b) * channels + c;
    const float xt       = __bfloat162float(x[o]);

    out[o]               = __float2bfloat16_rn((w0 * xt + w1 * s2 + w2 * s1 + w3 * s0 + xt) * kInvSqrt9);
    st[0 * channels + c] = s1;
    st[1 * channels + c] = s2;
    st[2 * channels + c] = xt;
}

__global__ void sconv_kv_dynamic_verify_kernel(
    const __nv_bfloat16* __restrict__ k_in, const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ wk, const float* __restrict__ st_k, float* __restrict__ vst_k,
    const __nv_bfloat16* __restrict__ v_in, const __nv_bfloat16* __restrict__ dv,
    const __nv_bfloat16* __restrict__ wv, const float* __restrict__ st_v, float* __restrict__ vst_v,
    const std::int32_t* __restrict__ lane_ids, __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v, int channels, int num_kv_heads, int batch_size,
    int verify_len, std::size_t lane_stride, int max_verify_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= channels || b >= batch_size) { return; }

    const bool is_v               = (blockIdx.z == 1);
    const __nv_bfloat16* x        = is_v ? v_in : k_in;
    const __nv_bfloat16* dyn_w    = is_v ? dv : dk;
    const __nv_bfloat16* static_w = is_v ? wv : wk;
    const float* state_pool       = is_v ? st_v : st_k;
    float* verify_pool            = is_v ? vst_v : vst_k;
    __nv_bfloat16* out            = is_v ? out_v : out_k;

    const int lane     = lane_ids[b];
    const int head_dim = channels / num_kv_heads;
    const int head     = c / head_dim;
    const float* st_in = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    float* vst_lane =
        verify_pool + static_cast<std::size_t>(lane) * static_cast<std::size_t>(max_verify_len) * lane_stride;

    const float sw0 = __bfloat162float(__ldg(static_w + c * 4 + 0));
    const float sw1 = __bfloat162float(__ldg(static_w + c * 4 + 1));
    const float sw2 = __bfloat162float(__ldg(static_w + c * 4 + 2));
    const float sw3 = __bfloat162float(__ldg(static_w + c * 4 + 3));

    float s0 = st_in[0 * channels + c];
    float s1 = st_in[1 * channels + c];
    float s2 = st_in[2 * channels + c];

    for (int v = 0; v < verify_len; ++v) {
        const int tok           = b * verify_len + v;
        const std::int64_t off  = static_cast<std::int64_t>(tok) * channels + c;
        const __nv_bfloat16* dw = dyn_w + (static_cast<std::int64_t>(tok) * num_kv_heads + head) * 4;
        const float w0          = sw0 + __bfloat162float(__ldg(dw + 0));
        const float w1          = sw1 + __bfloat162float(__ldg(dw + 1));
        const float w2          = sw2 + __bfloat162float(__ldg(dw + 2));
        const float w3          = sw3 + __bfloat162float(__ldg(dw + 3));

        const float xt = __bfloat162float(x[off]);
        out[off]       = __float2bfloat16_rn((w0 * xt + w1 * s2 + w2 * s1 + w3 * s0 + xt) * kInvSqrt9);
        s0             = s1;
        s1             = s2;
        s2             = xt;

        float* vst            = vst_lane + static_cast<std::size_t>(v) * lane_stride;
        vst[0 * channels + c] = s0;
        vst[1 * channels + c] = s1;
        vst[2 * channels + c] = s2;
    }
}

// ============================================================================
// 3. FUSED TP2 Direct-P2P Packet AllReduce + Static SConv (+ Residual + Block Copy)!
//    Matches allreduce_packet_sum_kernel's exact 64-thread, 8-BF16/thread epoch protocol!
//    Eliminates 96 sconv_static launches + 8 block D2D memcpys + 96 HBM round-trips per step!
// ============================================================================

__global__ void tp2_packet_allreduce_sconv_decode_kernel(
    const Bf16x8Pack* __restrict__ source, Bf16x8Pack* __restrict__ local,
    Bf16x8Pack* __restrict__ peer, unsigned* __restrict__ epochs, int packs, int epoch_packs,
    const __nv_bfloat16* __restrict__ sconv_w, float* __restrict__ state_pool,
    const std::int32_t* __restrict__ lane_ids, int channels, std::size_t lane_stride,
    Bf16x8Pack* __restrict__ out, const Bf16x8Pack* __restrict__ residual_add,
    const Bf16x8Pack* __restrict__ block_copy_src, Bf16x8Pack* __restrict__ block_copy_dst) {
    const int i          = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned epoch = epochs[blockIdx.x] & 1U;

    if (i < packs) {
        // Optionally copy block boundary cur -> blocks[num_blocks] right here (zero extra launch!)
        if (block_copy_src != nullptr && block_copy_dst != nullptr) {
            store_vec(block_copy_dst + i, load_vec<Bf16x8Pack>(block_copy_src + i));
        }

        // Exact two-epoch packet push & receive (bit-identical to allreduce_packet_sum_kernel)
        auto a = load_vec<Bf16x8Pack>(source + i);
        Bf16x8Pack empty;
        auto* a_bits     = reinterpret_cast<std::uint16_t*>(&a);
        auto* empty_bits = reinterpret_cast<std::uint16_t*>(&empty);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (a_bits[j] == 0xffffU) { a_bits[j] = 0x7fc0U; }
            empty_bits[j] = 0xffffU;
        }
        const int offset = static_cast<int>(epoch) * epoch_packs + i;
        pkt_store(peer + offset, a);

        Bf16x8Pack b;
        while (true) {
            b                = pkt_load(local + offset);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&b);
            bool missing     = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) { missing |= (bits[j] == 0xffffU); }
            if (!missing) { break; }
        }
        pkt_store(local + offset, empty);

        // Compute exact BF16-rounded AllReduce sum (8 channels), then apply SConv + residual in registers!
        const int packs_per_row = channels >> 3;
        const int row           = i / packs_per_row;
        const int c0            = (i - row * packs_per_row) << 3;
        const int lane          = lane_ids[row];
        float* st               = state_pool + static_cast<std::size_t>(lane) * lane_stride;

        Bf16x8Pack res_pack{};
        if (residual_add != nullptr) { res_pack = load_vec<Bf16x8Pack>(residual_add + i); }

        Bf16x8Pack y_pack{};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            // Exact BF16 rounding boundary matching separate AllReduce -> SConv!
            const __nv_bfloat162 summed = __floats2bfloat162_rn(
                __low2float(a.pair[p]) + __low2float(b.pair[p]),
                __high2float(a.pair[p]) + __high2float(b.pair[p]));
            const float2 xf = __bfloat1622float2(summed);
            const float x2[2] = {xf.x, xf.y};
            float r2[2]       = {0.0f, 0.0f};
            if (residual_add != nullptr) {
                const float2 rf = __bfloat1622float2(res_pack.pair[p]);
                r2[0]           = rf.x;
                r2[1]           = rf.y;
            }
            float y2[2];
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int c    = c0 + p * 2 + sub;
                const float xt = x2[sub];
                const float w0 = __bfloat162float(__ldg(sconv_w + c * 4 + 0));
                const float w1 = __bfloat162float(__ldg(sconv_w + c * 4 + 1));
                const float w2 = __bfloat162float(__ldg(sconv_w + c * 4 + 2));
                const float w3 = __bfloat162float(__ldg(sconv_w + c * 4 + 3));
                const float s0 = st[0 * channels + c];
                const float s1 = st[1 * channels + c];
                const float s2 = st[2 * channels + c];
                float y        = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * kInvSqrt5;
                if (residual_add != nullptr) { y += r2[sub]; }
                y2[sub]              = y;
                st[0 * channels + c] = s1;
                st[1 * channels + c] = s2;
                st[2 * channels + c] = xt;
            }
            y_pack.pair[p] = __floats2bfloat162_rn(y2[0], y2[1]);
        }
        store_vec(out + i, y_pack);
    }
    __syncthreads();
    if (threadIdx.x == 0) { epochs[blockIdx.x] = epoch ^ 1U; }
}

// ============================================================================
// 4. Verify Commit
// ============================================================================

__global__ void sconv_commit_verify_kernel(float* __restrict__ state_base,
                                           const float* __restrict__ verify_base,
                                           std::size_t lane_state_floats, int max_verify_len,
                                           const std::int32_t* __restrict__ lane_ids,
                                           const std::int32_t* __restrict__ accepted_counts,
                                           int batch_size) {
    const int b = blockIdx.y;
    if (b >= batch_size) { return; }
    const int acc = accepted_counts[b];
    if (acc <= 0 || acc > max_verify_len) { return; }
    const int lane      = lane_ids[b];
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= lane_state_floats) { return; }

    const float* src = verify_base +
        (static_cast<std::size_t>(lane) * static_cast<std::size_t>(max_verify_len) +
         static_cast<std::size_t>(acc - 1)) * lane_state_floats;
    float* dst = state_base + static_cast<std::size_t>(lane) * lane_state_floats;
    dst[i]     = src[i];
}

} // namespace

void inkling2_sconv_static_prefill(const Tensor& x, const Tensor& weight, float* state_lane,
                                   Tensor& out, const Tensor* residual_add, cudaStream_t stream) {
    const int channels = x.ne[0];
    const int tokens   = x.ne[1];
    const int blocks   = (channels + 255) / 256;
    sconv_static_prefill_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data), state_lane,
        static_cast<__nv_bfloat16*>(out.data),
        residual_add ? static_cast<const __nv_bfloat16*>(residual_add->data) : nullptr,
        channels, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_static_decode(const Tensor& x, const Tensor& weight, float* state_pool,
                                  const std::int32_t* lane_ids, std::int32_t batch_size,
                                  Tensor& out, const Tensor* residual_add, cudaStream_t stream,
                                  std::size_t state_lane_stride) {
    const int channels       = x.ne[0];
    const std::size_t stride = state_lane_stride ? state_lane_stride : static_cast<std::size_t>(3 * channels);
    const dim3 grid((channels + 255) / 256, batch_size);
    sconv_static_decode_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data), state_pool, lane_ids,
        static_cast<__nv_bfloat16*>(out.data),
        residual_add ? static_cast<const __nv_bfloat16*>(residual_add->data) : nullptr,
        channels, batch_size, stride);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_static_verify(const Tensor& x, const Tensor& weight, const float* state_pool,
                                  float* verify_states_pool, const std::int32_t* lane_ids,
                                  std::int32_t batch_size, std::int32_t verify_len,
                                  Tensor& out, const Tensor* residual_add, cudaStream_t stream,
                                  std::size_t state_lane_stride, std::int32_t max_verify_len) {
    const int channels       = x.ne[0];
    const std::size_t stride = state_lane_stride ? state_lane_stride : static_cast<std::size_t>(3 * channels);
    const int max_v          = max_verify_len ? max_verify_len : verify_len;
    const dim3 grid((channels + 255) / 256, batch_size);
    sconv_static_verify_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data), state_pool, verify_states_pool, lane_ids,
        static_cast<__nv_bfloat16*>(out.data),
        residual_add ? static_cast<const __nv_bfloat16*>(residual_add->data) : nullptr,
        channels, batch_size, verify_len, stride, max_v);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_dynamic_prefill(const Tensor& x, const Tensor& dyn_w, const Tensor& static_w,
                                    std::int32_t num_kv_heads, float* state_lane, Tensor& out,
                                    cudaStream_t stream) {
    const int channels = x.ne[0];
    const int tokens   = x.ne[1];
    const int blocks   = (channels + 255) / 256;
    sconv_dynamic_prefill_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(dyn_w.data),
        static_cast<const __nv_bfloat16*>(static_w.data), state_lane,
        static_cast<__nv_bfloat16*>(out.data), channels, num_kv_heads, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_dynamic_decode(const Tensor& x, const Tensor& dyn_w, const Tensor& static_w,
                                   std::int32_t num_kv_heads, float* state_pool,
                                   const std::int32_t* lane_ids, std::int32_t batch_size,
                                   Tensor& out, cudaStream_t stream,
                                   std::size_t state_lane_stride) {
    const int channels       = x.ne[0];
    const std::size_t stride = state_lane_stride ? state_lane_stride : static_cast<std::size_t>(3 * channels);
    const dim3 grid((channels + 255) / 256, batch_size);
    sconv_dynamic_decode_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(dyn_w.data),
        static_cast<const __nv_bfloat16*>(static_w.data), state_pool, lane_ids,
        static_cast<__nv_bfloat16*>(out.data), channels, num_kv_heads, batch_size, stride);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_dynamic_verify(const Tensor& x, const Tensor& dyn_w, const Tensor& static_w,
                                   std::int32_t num_kv_heads, const float* state_pool,
                                   float* verify_states_pool, const std::int32_t* lane_ids,
                                   std::int32_t batch_size, std::int32_t verify_len,
                                   Tensor& out, cudaStream_t stream,
                                   std::size_t state_lane_stride, std::int32_t max_verify_len) {
    const int channels       = x.ne[0];
    const std::size_t stride = state_lane_stride ? state_lane_stride : static_cast<std::size_t>(3 * channels);
    const int max_v          = max_verify_len ? max_verify_len : verify_len;
    const dim3 grid((channels + 255) / 256, batch_size);
    sconv_dynamic_verify_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(dyn_w.data),
        static_cast<const __nv_bfloat16*>(static_w.data), state_pool, verify_states_pool, lane_ids,
        static_cast<__nv_bfloat16*>(out.data), channels, num_kv_heads, batch_size, verify_len,
        stride, max_v);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_commit_verify(float* state_base, const float* verify_base,
                                  std::size_t lane_state_floats, std::int32_t max_verify_len,
                                  const std::int32_t* lane_ids, const std::int32_t* accepted_counts,
                                  std::int32_t batch_size, cudaStream_t stream) {
    if (batch_size <= 0 || lane_state_floats == 0) { return; }
    const dim3 grid(static_cast<unsigned>((lane_state_floats + 255) / 256), batch_size);
    sconv_commit_verify_kernel<<<grid, 256, 0, stream>>>(
        state_base, verify_base, lane_state_floats, max_verify_len, lane_ids, accepted_counts,
        batch_size);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_kv_static_decode(const Tensor& k, const Tensor& wk, float* state_k,
                                     const Tensor& v, const Tensor& wv, float* state_v,
                                     const std::int32_t* lane_ids, std::int32_t batch_size,
                                     Tensor& out_k, Tensor& out_v, cudaStream_t stream,
                                     std::size_t state_lane_stride) {
    const int channels = k.ne[0];
    const dim3 grid((channels + 255) / 256, batch_size, 2);
    sconv_kv_static_decode_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(wk.data),
        state_k, static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const __nv_bfloat16*>(wv.data), state_v, lane_ids,
        static_cast<__nv_bfloat16*>(out_k.data), static_cast<__nv_bfloat16*>(out_v.data),
        channels, batch_size, state_lane_stride);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_kv_static_verify(const Tensor& k, const Tensor& wk, const float* state_k,
                                     float* verify_k, const Tensor& v, const Tensor& wv,
                                     const float* state_v, float* verify_v,
                                     const std::int32_t* lane_ids, std::int32_t batch_size,
                                     std::int32_t verify_len, Tensor& out_k, Tensor& out_v,
                                     cudaStream_t stream, std::size_t state_lane_stride,
                                     std::int32_t max_verify_len) {
    const int channels = k.ne[0];
    const dim3 grid((channels + 255) / 256, batch_size, 2);
    sconv_kv_static_verify_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(wk.data),
        state_k, verify_k, static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const __nv_bfloat16*>(wv.data), state_v, verify_v, lane_ids,
        static_cast<__nv_bfloat16*>(out_k.data), static_cast<__nv_bfloat16*>(out_v.data),
        channels, batch_size, verify_len, state_lane_stride, max_verify_len);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_kv_dynamic_decode(const Tensor& k, const Tensor& dyn_k, const Tensor& wk,
                                      float* state_k, const Tensor& v, const Tensor& dyn_v,
                                      const Tensor& wv, float* state_v, std::int32_t num_kv_heads,
                                      const std::int32_t* lane_ids, std::int32_t batch_size,
                                      Tensor& out_k, Tensor& out_v, cudaStream_t stream,
                                      std::size_t state_lane_stride) {
    const int channels = k.ne[0];
    const dim3 grid((channels + 255) / 256, batch_size, 2);
    sconv_kv_dynamic_decode_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(dyn_k.data),
        static_cast<const __nv_bfloat16*>(wk.data), state_k,
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(dyn_v.data),
        static_cast<const __nv_bfloat16*>(wv.data), state_v, lane_ids,
        static_cast<__nv_bfloat16*>(out_k.data), static_cast<__nv_bfloat16*>(out_v.data),
        channels, num_kv_heads, batch_size, state_lane_stride);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_sconv_kv_dynamic_verify(const Tensor& k, const Tensor& dyn_k, const Tensor& wk,
                                      const float* state_k, float* verify_k, const Tensor& v,
                                      const Tensor& dyn_v, const Tensor& wv, const float* state_v,
                                      float* verify_v, std::int32_t num_kv_heads,
                                      const std::int32_t* lane_ids, std::int32_t batch_size,
                                      std::int32_t verify_len, Tensor& out_k, Tensor& out_v,
                                      cudaStream_t stream, std::size_t state_lane_stride,
                                      std::int32_t max_verify_len) {
    const int channels = k.ne[0];
    const dim3 grid((channels + 255) / 256, batch_size, 2);
    sconv_kv_dynamic_verify_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(dyn_k.data),
        static_cast<const __nv_bfloat16*>(wk.data), state_k, verify_k,
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(dyn_v.data),
        static_cast<const __nv_bfloat16*>(wv.data), state_v, verify_v, lane_ids,
        static_cast<__nv_bfloat16*>(out_k.data), static_cast<__nv_bfloat16*>(out_v.data),
        channels, num_kv_heads, batch_size, verify_len, state_lane_stride, max_verify_len);
    CUDA_CHECK(cudaGetLastError());
}

void inkling2_tp2_packet_allreduce_sconv_decode(
    const Tensor& partial_in, void* local_storage, void* peer_storage, std::size_t epoch_bytes,
    const Tensor& sconv_weight, float* state_pool, const std::int32_t* lane_ids,
    std::int32_t batch_size, std::size_t state_lane_stride, Tensor& out,
    const Tensor* residual_add, const Tensor* block_copy_src, Tensor* block_copy_dst,
    cudaStream_t stream) {
    const int channels    = partial_in.ne[0];
    const int packs       = static_cast<int>(partial_in.bytes() / sizeof(Bf16x8Pack));
    const int epoch_packs = static_cast<int>(epoch_bytes / sizeof(Bf16x8Pack));
    const int blocks      = (packs + 63) / 64;
    auto* epochs          = reinterpret_cast<unsigned*>(
        static_cast<std::uint8_t*>(local_storage) + 2 * epoch_bytes);
    tp2_packet_allreduce_sconv_decode_kernel<<<blocks, 64, 0, stream>>>(
        static_cast<const Bf16x8Pack*>(partial_in.data),
        static_cast<Bf16x8Pack*>(local_storage), static_cast<Bf16x8Pack*>(peer_storage), epochs,
        packs, epoch_packs, static_cast<const __nv_bfloat16*>(sconv_weight.data), state_pool,
        lane_ids, channels, state_lane_stride, static_cast<Bf16x8Pack*>(out.data),
        residual_add ? static_cast<const Bf16x8Pack*>(residual_add->data) : nullptr,
        block_copy_src ? static_cast<const Bf16x8Pack*>(block_copy_src->data) : nullptr,
        block_copy_dst ? static_cast<Bf16x8Pack*>(block_copy_dst->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
