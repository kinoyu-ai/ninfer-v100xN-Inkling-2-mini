#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::inkling2_mini::frontend_internal {

struct AddedToken {
    int id = 0;
    std::string content;
    bool special = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    [[nodiscard]] std::vector<TokenId> encode(std::string_view text, bool parse_added_tokens = true,
                                              const PreparationControl& control = {}) const;
    [[nodiscard]] std::string decode_token_bytes(TokenId id, bool skip_special_tokens = false) const;
    [[nodiscard]] bool is_special_token(TokenId id) const noexcept;
    [[nodiscard]] bool is_valid_token(TokenId id) const noexcept;
    [[nodiscard]] std::span<const TokenId> default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

private:
    std::vector<std::string> id_to_token_;
    std::vector<bool> valid_token_ids_;
    std::vector<bool> special_token_ids_;
    std::vector<bool> added_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    std::unordered_map<std::string, int> bpe_merge_ranks_;
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_{};
    std::vector<TokenId> default_stop_token_ids_;
    bool ignore_merges_ = true;
};

} // namespace ninfer::targets::inkling2_mini::frontend_internal
