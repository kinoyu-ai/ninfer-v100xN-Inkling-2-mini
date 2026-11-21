#pragma once

#include <array>
#include <cstdint>

namespace ninfer::targets::inkling2_mini::detail {

inline constexpr std::int32_t kHiddenSize       = 2304;
inline constexpr std::int32_t kNumPhysLayers    = 24;
inline constexpr std::int32_t kCoreStart        = 6;
inline constexpr std::int32_t kCoreEnd          = 18;
inline constexpr std::int32_t kCoreLen          = kCoreEnd - kCoreStart; // 12
inline constexpr std::int32_t kNumPasses        = 3;
inline constexpr std::int32_t kNumLogicalLayers = 48; // 6 + 3*12 + 6
inline constexpr std::int32_t kGateRank         = 64;

inline constexpr std::int32_t kNumQHeads        = 24;
inline constexpr std::int32_t kNumKVHeads       = 8;
inline constexpr std::int32_t kHeadDim          = 128;
inline constexpr std::int32_t kDRel             = 16;
inline constexpr std::int32_t kRelExtent        = 1024;
inline constexpr std::int32_t kSwaWindow        = 1024;
inline constexpr std::int32_t kSwaRingCap       = 1040; // 1024 + 16 headroom for verify block

inline constexpr std::int32_t kNumExperts       = 128;
inline constexpr std::int32_t kTopK             = 8;
inline constexpr std::int32_t kMoeInterSize     = 1152;
inline constexpr std::int32_t kMoeInterPerSlab  = 576;
inline constexpr std::int32_t kNvfp4LayerEnd    = 20; // physical 0..19 NVFP4, 20..23 MXFP8

inline constexpr std::int32_t kPaddedVocabSize  = 201024;
inline constexpr std::int32_t kUnpaddedVocab    = 200058;
inline constexpr std::int32_t kAttnResBlockSize = 6;
inline constexpr std::int32_t kMaxContextNative = 262144;

inline constexpr float kRmsNormEps            = 1e-6f;
inline constexpr float kQkClipThreshold       = 22.627416997969522f; // 2 * sqrt(128)
inline constexpr float kRelClipThreshold      = 8.0f;
inline constexpr float kLogScalingFloor       = 8192.0f;
inline constexpr float kLogScalingAlpha       = 0.1f;
inline constexpr float kFinalLogitSoftcap     = 30.0f;
inline constexpr float kRouteScale            = 8.0f;
inline constexpr float kMlpGateSoftcap        = 4.0f;
inline constexpr float kMlpUpSoftcap          = 25.0f;

// DSpark MTP Drafter constants
inline constexpr std::int32_t kMtpNumLayers     = 3;
inline constexpr std::int32_t kMtpInterSize     = 9216;
inline constexpr std::int32_t kMtpMarkovRank    = 256;
inline constexpr std::int32_t kMtpMaxBlockSize  = 8; // 1 anchor + up to 7 draft tokens
inline constexpr float kMtpRopeTheta            = 1000000.0f;
inline constexpr std::array<std::int32_t, 4> kMtpTargetLayerIds = {5, 23, 41, 48};

inline constexpr bool is_full_phys_layer(std::int32_t ip) noexcept {
    return ip == 5 || ip == 11 || ip == 17 || ip == 23;
}

inline constexpr bool is_local_phys_layer(std::int32_t ip) noexcept {
    return !is_full_phys_layer(ip);
}

inline constexpr bool is_core_phys_layer(std::int32_t ip) noexcept {
    return ip >= kCoreStart && ip < kCoreEnd;
}

inline constexpr bool is_nvfp4_phys_layer(std::int32_t ip) noexcept {
    return ip < kNvfp4LayerEnd;
}

inline constexpr std::int32_t phys_of_logical(std::int32_t il) noexcept {
    if (il < kCoreStart) { return il; }
    if (il < kCoreStart + kNumPasses * kCoreLen) {
        return kCoreStart + ((il - kCoreStart) % kCoreLen);
    }
    return kCoreEnd + (il - (kCoreStart + kNumPasses * kCoreLen));
}

// 8 global logical layers: 5, 11, 17, 23, 29, 35, 41, 47 -> global index 0..7
inline constexpr std::int32_t global_layer_index(std::int32_t il) noexcept {
    return (il % 6 == 5) ? (il / 6) : -1;
}

// 40 local logical layers -> local index 0..39
inline constexpr std::int32_t local_layer_index(std::int32_t il) noexcept {
    return (il % 6 == 5) ? -1 : (il - il / 6);
}

inline constexpr std::int32_t kNumGlobalLogicalLayers = 8;
inline constexpr std::int32_t kNumLocalLogicalLayers  = 40;

} // namespace ninfer::targets::inkling2_mini::detail
