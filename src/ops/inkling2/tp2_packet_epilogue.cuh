#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

constexpr int kTp2FusedHidden        = 2304;
constexpr int kTp2FusedPacks         = kTp2FusedHidden / 8; // 288
constexpr std::size_t kTp2FusedPktBytes =
    static_cast<std::size_t>(2 * kTp2FusedPacks) * 16ULL; // 9216 B
constexpr float kTp2FusedInvSqrt5    = 0.44721359549995793928f;

struct alignas(16) Tp2Bf16x8Pack {
    __nv_bfloat162 pair[4];
};

static_assert(sizeof(Tp2Bf16x8Pack) == 16);

__device__ __forceinline__ void tp2_pkt_store(Tp2Bf16x8Pack* dst, Tp2Bf16x8Pack v) {
    const auto* w = reinterpret_cast<const unsigned*>(&v);
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(dst), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3])
                 : "memory");
}

__device__ __forceinline__ Tp2Bf16x8Pack tp2_pkt_load(const Tp2Bf16x8Pack* src) {
    Tp2Bf16x8Pack v;
    auto* w = reinterpret_cast<unsigned*>(&v);
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                 : "l"(src)
                 : "memory");
    return v;
}

// Zero-barrier per-pack (8 channels) TP2 direct-P2P Packet AllReduce + static SConv
// (+ optional residual_add and optional block_copy).
//
// Preconditions:
//   - Caller is threads 0..7 (warp 0) of a 288-CTA launch (`pack_idx = blockIdx.x` in [0..287]).
//   - All 288 CTAs fit co-resident on BOTH peer GPUs (verified by caller via occupancy check).
//   - `s_local_bf16[0..7]` already holds this CTA's 8 BF16 partial output values.
//   - `pkt_local` / `pkt_peer` each point to [2 * 288] Tp2Bf16x8Pack (initialized to 0xFF)
//     followed immediately by [288] unsigned per-pack epochs (initialized to 0).
//
// Latency-hiding order:
//   1. Immediately pushes `a` over NVLink (`tp2_pkt_store`) so peer transfer starts ASAP.
//   2. While packet is in flight, prefetches read-only sconv weights, sconv state (s0..s2),
//      residual_add, and executes optional block_copy in parallel across threads 0..7!
//   3. Polls peer packet (`tp2_pkt_load`), clears local mailbox, flips per-pack epoch,
//      computes exact BF16-rounded sum + exact FP32 SConv + residual -> BF16 out & state!
__device__ __forceinline__ void tp2_packet_ar_sconv_epilogue_w0(
    int pack_idx, __nv_bfloat16* s_local_bf16, Tp2Bf16x8Pack* s_peer_pack,
    void* __restrict__ pkt_local_raw, void* __restrict__ pkt_peer_raw,
    const __nv_bfloat16* __restrict__ sconv_w, float* __restrict__ state_pool,
    const std::int32_t* __restrict__ lane_ids, std::size_t lane_stride,
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ residual_add,
    const void* __restrict__ block_copy_src_raw, void* __restrict__ block_copy_dst_raw) {
    const int tid = threadIdx.x; // 0..7
    __syncwarp(0xFFu);

    auto* pkt_local = static_cast<Tp2Bf16x8Pack*>(pkt_local_raw);
    auto* pkt_peer  = static_cast<Tp2Bf16x8Pack*>(pkt_peer_raw);
    auto* epochs    = reinterpret_cast<unsigned*>(
        static_cast<std::uint8_t*>(pkt_local_raw) + kTp2FusedPktBytes);

    unsigned epoch = 0;
    int offset     = 0;
    Tp2Bf16x8Pack empty{};
    if (tid == 0) {
        epoch           = epochs[pack_idx] & 1U;
        Tp2Bf16x8Pack a = *reinterpret_cast<const Tp2Bf16x8Pack*>(s_local_bf16);
        auto* a_bits     = reinterpret_cast<std::uint16_t*>(&a);
        auto* empty_bits = reinterpret_cast<std::uint16_t*>(&empty);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (a_bits[j] == 0xffffU) { a_bits[j] = 0x7fc0U; }
            empty_bits[j] = 0xffffU;
        }
        offset = static_cast<int>(epoch) * kTp2FusedPacks + pack_idx;
        // Fire NVLink packet immediately!
        tp2_pkt_store(pkt_peer + offset, a);
        *reinterpret_cast<Tp2Bf16x8Pack*>(s_local_bf16) = a;
    }

    // Overlap HBM reads (sconv state, weights, residual, block_copy) with NVLink RTT!
    const int c      = pack_idx * 8 + tid;
    const int lane   = lane_ids[0];
    float* st        = state_pool + static_cast<std::size_t>(lane) * lane_stride;
    const float s0   = st[0 * kTp2FusedHidden + c];
    const float s1   = st[1 * kTp2FusedHidden + c];
    const float s2   = st[2 * kTp2FusedHidden + c];
    const uint2 wraw = __ldg(reinterpret_cast<const uint2*>(sconv_w + c * 4));
    const float rval =
        (residual_add != nullptr) ? __bfloat162float(__ldg(residual_add + c)) : 0.0f;

    if (tid == 0) {
        if (block_copy_src_raw != nullptr && block_copy_dst_raw != nullptr) {
            const auto* bsrc = static_cast<const Tp2Bf16x8Pack*>(block_copy_src_raw);
            auto* bdst       = static_cast<Tp2Bf16x8Pack*>(block_copy_dst_raw);
            bdst[pack_idx]   = bsrc[pack_idx];
        }

        Tp2Bf16x8Pack b;
        while (true) {
            b                = tp2_pkt_load(pkt_local + offset);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&b);
            bool missing     = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) { missing |= (bits[j] == 0xffffU); }
            if (!missing) { break; }
        }
        tp2_pkt_store(pkt_local + offset, empty);
        epochs[pack_idx] = epoch ^ 1U;
        *s_peer_pack     = b;
    }
    __syncwarp(0xFFu);

    const float af   = __bfloat162float(s_local_bf16[tid]);
    const float bf   = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(s_peer_pack)[tid]);
    const float xf   = __bfloat162float(__float2bfloat16_rn(af + bf));

    const auto* wb   = reinterpret_cast<const __nv_bfloat162*>(&wraw);
    const float2 wa  = __bfloat1622float2(wb[0]);
    const float2 wb2 = __bfloat1622float2(wb[1]);

    float y = (wa.x * s0 + wa.y * s1 + wb2.x * s2 + wb2.y * xf + xf) * kTp2FusedInvSqrt5;
    if (residual_add != nullptr) { y += rval; }

    out[c]                    = __float2bfloat16_rn(y);
    st[0 * kTp2FusedHidden + c] = s1;
    st[1 * kTp2FusedHidden + c] = s2;
    st[2 * kTp2FusedHidden + c] = xf;
}

} // namespace ninfer::ops::detail
