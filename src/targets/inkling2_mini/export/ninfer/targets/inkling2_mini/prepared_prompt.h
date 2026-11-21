#pragma once

#include <ninfer/targets/inkling2_mini/frontend.h>

#include <cstdint>
#include <vector>

namespace ninfer::targets::inkling2_mini {

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    bool allow_prefix_reuse   = true;
    double prepare_seconds    = 0.0;
    double tokenize_seconds   = 0.0;
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::targets::inkling2_mini
