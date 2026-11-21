#include "targets/inkling2_mini/impl/frontend/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace ninfer::targets::inkling2_mini::frontend_internal {
namespace {

using Json    = nlohmann::json;
namespace uni = ninfer::text::unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

class EncodeControl {
public:
    explicit EncodeControl(const PreparationControl& control) : control_(control) { check(); }
    void check() const {
        if (control_.cancellation.requested()) {
            throw RequestError(RequestErrorKind::Cancelled, "tokenization cancelled");
        }
        if (control_.deadline != std::chrono::steady_clock::time_point{} &&
            std::chrono::steady_clock::now() >= control_.deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout, "tokenization timed out");
        }
    }
    void step() {
        if ((++work_ & 1023U) == 0) { check(); }
    }

private:
    const PreparationControl& control_;
    std::size_t work_ = 0;
};

std::string merge_pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + 1 + right.size());
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

std::unordered_map<std::uint32_t, char> build_byte_level_decoder() {
    std::unordered_map<std::uint32_t, char> decoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= 255; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t cp = visible ? static_cast<std::uint32_t>(byte) : next++;
        decoder.emplace(cp, static_cast<char>(static_cast<unsigned char>(byte)));
    }
    return decoder;
}

std::array<std::string, 256> build_byte_level_encoder() {
    std::array<std::string, 256> encoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= 255; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t cp = visible ? static_cast<std::uint32_t>(byte) : next++;
        encoder[static_cast<std::size_t>(byte)] =
            uni::codepoint_to_utf8(static_cast<std::int32_t>(cp));
    }
    return encoder;
}

// Unicode category helpers matching o200k_base regex:
//   is_U: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
//   is_l: [\p{Ll}\p{Lm}\p{Lo}\p{M}]
bool is_newline(std::int32_t cp) noexcept { return cp == '\r' || cp == '\n'; }

bool is_upper_group(std::int32_t cp) noexcept {
    switch (utf8proc_category(cp)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
        return true;
    default:
        return false;
    }
}

bool is_lower_group(std::int32_t cp) noexcept {
    switch (utf8proc_category(cp)) {
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
        return true;
    default:
        return false;
    }
}

bool is_word_prefix(std::int32_t cp) noexcept {
    return !is_newline(cp) && !uni::is_letter(cp) && !uni::is_number(cp);
}

bool is_punct_char(std::int32_t cp) noexcept {
    return !uni::is_whitespace(cp) && !uni::is_letter(cp) && !uni::is_number(cp);
}

bool ascii_ci_matches(std::string_view text, std::size_t offset, std::string_view suffix) {
    if (offset + suffix.size() > text.size()) { return false; }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(text[offset + i])) !=
            std::tolower(static_cast<unsigned char>(suffix[i]))) {
            return false;
        }
    }
    return true;
}

// Matches optional (?i:'s|'t|'re|'ve|'m|'ll|'d) at codepoint index `pos`, returning new end index.
std::size_t match_contraction(std::string_view text, const std::vector<uni::CodepointSpan>& spans,
                              std::size_t pos) {
    if (pos >= spans.size() || spans[pos].value != '\'') { return pos; }
    const std::size_t off                 = spans[pos].offset;
    constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (std::string_view suf : suffixes) {
        if (ascii_ci_matches(text, off + 1, suf)) {
            std::size_t end = pos + 1;
            while (end < spans.size() && spans[end].offset < off + 1 + suf.size()) { ++end; }
            return end;
        }
    }
    return pos;
}

std::size_t span_end_offset(std::string_view text, const std::vector<uni::CodepointSpan>& spans,
                            std::size_t end) {
    return (end >= spans.size()) ? text.size() : spans[end].offset;
}

// Match branch 1 body: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
// Returns end index (> s) on match, or 0 on no match.
std::size_t match_o200k_branch1_body(const std::vector<uni::CodepointSpan>& spans, std::size_t s) {
    const std::size_t n = spans.size();
    std::size_t e1      = s;
    while (e1 < n && is_upper_group(spans[e1].value)) { ++e1; }
    if (e1 < n && is_lower_group(spans[e1].value)) {
        std::size_t e = e1;
        while (e < n && is_lower_group(spans[e].value)) { ++e; }
        return e;
    }
    // Backtrack inside [s, e1) to find the last codepoint in is_lower_group (Lm/Lo/M)
    for (std::size_t k = e1; k > s; --k) {
        if (is_lower_group(spans[k - 1].value)) { return k; }
    }
    return 0;
}

// Match branch 2 body: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
std::size_t match_o200k_branch2_body(const std::vector<uni::CodepointSpan>& spans, std::size_t s) {
    const std::size_t n = spans.size();
    if (s >= n || !is_upper_group(spans[s].value)) { return 0; }
    std::size_t e = s;
    while (e < n && is_upper_group(spans[e].value)) { ++e; }
    while (e < n && is_lower_group(spans[e].value)) { ++e; }
    return e;
}

std::vector<std::string_view> o200k_split_words(std::string_view text, EncodeControl& control) {
    const std::vector<uni::CodepointSpan> spans =
        uni::utf8_codepoints(text, "inkling2 Tokenizer::encode input");
    const std::size_t n = spans.size();
    std::vector<std::string_view> words;

    for (std::size_t i = 0; i < n;) {
        control.step();
        const std::size_t begin_off = spans[i].offset;
        const std::int32_t cp       = spans[i].value;

        // Branch 1: [^\r\n\p{L}\p{N}]? U* l+ contraction?
        {
            std::size_t end = 0;
            if (is_word_prefix(cp)) { end = match_o200k_branch1_body(spans, i + 1); }
            if (end == 0) { end = match_o200k_branch1_body(spans, i); }
            if (end > i) {
                end = match_contraction(text, spans, end);
                words.emplace_back(
                    text.substr(begin_off, span_end_offset(text, spans, end) - begin_off));
                i = end;
                continue;
            }
        }

        // Branch 2: [^\r\n\p{L}\p{N}]? U+ l* contraction?
        {
            std::size_t end = 0;
            if (is_word_prefix(cp)) { end = match_o200k_branch2_body(spans, i + 1); }
            if (end == 0) { end = match_o200k_branch2_body(spans, i); }
            if (end > i) {
                end = match_contraction(text, spans, end);
                words.emplace_back(
                    text.substr(begin_off, span_end_offset(text, spans, end) - begin_off));
                i = end;
                continue;
            }
        }

        // Branch 3: \p{N}{1,3}
        if (uni::is_number(cp)) {
            std::size_t end = i + 1;
            while (end < n && (end - i) < 3 && uni::is_number(spans[end].value)) { ++end; }
            words.emplace_back(
                text.substr(begin_off, span_end_offset(text, spans, end) - begin_off));
            i = end;
            continue;
        }

        // Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
        if ((cp == ' ' && i + 1 < n && is_punct_char(spans[i + 1].value)) || is_punct_char(cp)) {
            std::size_t end = (cp == ' ') ? (i + 1) : i;
            while (end < n && is_punct_char(spans[end].value)) {
                control.step();
                ++end;
            }
            while (end < n && (is_newline(spans[end].value) || spans[end].value == '/')) {
                control.step();
                ++end;
            }
            words.emplace_back(
                text.substr(begin_off, span_end_offset(text, spans, end) - begin_off));
            i = end;
            continue;
        }

        // Branches 5, 6, 7: \s*[\r\n]+ | \s+(?!\S) | \s+
        if (uni::is_whitespace(cp)) {
            std::size_t run_end      = i;
            std::size_t last_newline = std::string_view::npos;
            while (run_end < n && uni::is_whitespace(spans[run_end].value)) {
                control.step();
                if (is_newline(spans[run_end].value)) { last_newline = run_end; }
                ++run_end;
            }
            if (last_newline != std::string_view::npos) {
                const std::size_t end = last_newline + 1;
                words.emplace_back(
                    text.substr(begin_off, span_end_offset(text, spans, end) - begin_off));
                i = end;
                continue;
            }
            if (run_end == n) {
                words.emplace_back(text.substr(begin_off));
                i = run_end;
                continue;
            }
            if (run_end - i >= 2) {
                const std::size_t end = run_end - 1;
                words.emplace_back(
                    text.substr(begin_off, span_end_offset(text, spans, end) - begin_off));
                i = end;
                continue;
            }
            words.emplace_back(
                text.substr(begin_off, span_end_offset(text, spans, run_end) - begin_off));
            i = run_end;
            continue;
        }

        // Fallback single codepoint
        words.emplace_back(text.substr(begin_off, spans[i].length));
        ++i;
    }
    return words;
}

std::string byte_level_encode(std::string_view text) {
    static const std::array<std::string, 256> kEncoder = build_byte_level_encoder();
    std::string out;
    out.reserve(text.size() * 2);
    for (const unsigned char b : text) { out += kEncoder[b]; }
    return out;
}

std::vector<std::string> byte_level_symbols(std::string_view text) {
    static const std::array<std::string, 256> kEncoder = build_byte_level_encoder();
    std::vector<std::string> syms;
    syms.reserve(text.size());
    for (const unsigned char b : text) { syms.push_back(kEncoder[b]); }
    return syms;
}

int parse_id_json(const Json& val, std::string_view label) {
    if (!val.is_number_integer()) {
        throw std::invalid_argument(std::string(label) + ": expected integer id");
    }
    const std::int64_t v = val.get<std::int64_t>();
    if (v < 0 || v > kMaxTokenId) {
        throw std::invalid_argument(std::string(label) + ": id out of range");
    }
    return static_cast<int>(v);
}

} // namespace

Tokenizer::Tokenizer(TokenizerResources resources) {
    if (resources.tokenizer_json.empty() || resources.tokenizer_config_json.empty() ||
        resources.generation_config_json.empty()) {
        throw std::invalid_argument("inkling2 tokenizer resources are empty");
    }
    const Json root   = Json::parse(resources.tokenizer_json);
    const Json cfg    = Json::parse(resources.tokenizer_config_json);
    const Json gen    = Json::parse(resources.generation_config_json);
    const Json& model = root.at("model");
    if (model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("inkling2 tokenizer model.type must be BPE");
    }
    ignore_merges_ = model.value("ignore_merges", true);

    const Json& vocab = model.at("vocab");
    int max_id        = -1;
    for (const auto& item : vocab.items()) {
        max_id = std::max(max_id, parse_id_json(item.value(), "model.vocab"));
    }
    id_to_token_.resize(static_cast<std::size_t>(max_id + 1));
    valid_token_ids_.assign(static_cast<std::size_t>(max_id + 1), false);
    special_token_ids_.assign(static_cast<std::size_t>(max_id + 1), false);
    added_token_ids_.assign(static_cast<std::size_t>(max_id + 1), false);
    vocab_token_to_id_.reserve(vocab.size());

    for (const auto& item : vocab.items()) {
        const int id = parse_id_json(item.value(), "model.vocab");
        id_to_token_[static_cast<std::size_t>(id)]     = item.key();
        valid_token_ids_[static_cast<std::size_t>(id)] = true;
        vocab_token_to_id_.emplace(item.key(), id);
    }

    auto register_added = [&](int id, const std::string& content, bool special) {
        const auto idx = static_cast<std::size_t>(id);
        if (idx >= id_to_token_.size()) {
            id_to_token_.resize(idx + 1);
            valid_token_ids_.resize(idx + 1, false);
            special_token_ids_.resize(idx + 1, false);
            added_token_ids_.resize(idx + 1, false);
        }
        if (added_token_ids_[idx]) { return; }
        id_to_token_[idx]       = content;
        valid_token_ids_[idx]   = true;
        special_token_ids_[idx] = special;
        added_token_ids_[idx]   = true;
        added_tokens_.push_back(AddedToken{.id = id, .content = content, .special = special});
    };

    if (root.contains("added_tokens") && root.at("added_tokens").is_array()) {
        for (const Json& item : root.at("added_tokens")) {
            register_added(parse_id_json(item.at("id"), "added_tokens.id"),
                           item.at("content").get<std::string>(), item.value("special", false));
        }
    }
    if (cfg.contains("added_tokens_decoder") && cfg.at("added_tokens_decoder").is_object()) {
        for (const auto& item : cfg.at("added_tokens_decoder").items()) {
            int id = 0;
            const auto& k = item.key();
            auto [ptr, ec] = std::from_chars(k.data(), k.data() + k.size(), id);
            if (ec != std::errc{} || ptr != k.data() + k.size()) { continue; }
            register_added(id, item.value().at("content").get<std::string>(),
                           item.value().value("special", false));
        }
    }

    // Sort longest content first so overlapping prefixes match greedily
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                  if (a.content.size() != b.content.size()) {
                      return a.content.size() > b.content.size();
                  }
                  return a.id < b.id;
              });
    for (std::size_t i = 0; i < added_tokens_.size(); ++i) {
        if (!added_tokens_[i].content.empty()) {
            added_token_candidates_[static_cast<unsigned char>(added_tokens_[i].content[0])]
                .push_back(i);
        }
    }

    const Json& merges = model.at("merges");
    bpe_merge_ranks_.reserve(merges.size());
    int rank = 0;
    for (const Json& item : merges) {
        if (item.is_array() && item.size() == 2) {
            bpe_merge_ranks_.emplace(
                merge_pair_key(item[0].get<std::string>(), item[1].get<std::string>()), rank++);
        } else if (item.is_string()) {
            const std::string s = item.get<std::string>();
            const auto sp       = s.find(' ');
            if (sp != std::string::npos) {
                bpe_merge_ranks_.emplace(
                    merge_pair_key(std::string_view(s).substr(0, sp),
                                   std::string_view(s).substr(sp + 1)),
                    rank++);
            }
        }
    }

    if (gen.contains("eos_token_id")) {
        const Json& eos = gen.at("eos_token_id");
        if (eos.is_number_integer()) {
            default_stop_token_ids_.push_back(parse_id_json(eos, "eos_token_id"));
        } else if (eos.is_array()) {
            for (const Json& e : eos) {
                default_stop_token_ids_.push_back(parse_id_json(e, "eos_token_id"));
            }
        }
    }
    if (default_stop_token_ids_.empty()) { default_stop_token_ids_.push_back(200006); }
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool parse_added_tokens,
                                       const PreparationControl& control_in) const {
    EncodeControl control(control_in);
    std::vector<TokenId> ids;
    if (text.empty()) { return ids; }

    auto encode_ordinary = [&](std::string_view chunk) {
        if (chunk.empty()) { return; }
        // Inkling-2-Mini has normalizer: null (NO NFC normalization!)
        for (const std::string_view word : o200k_split_words(chunk, control)) {
            control.check();
            if (ignore_merges_) {
                const std::string bl = byte_level_encode(word);
                const auto direct    = vocab_token_to_id_.find(bl);
                if (direct != vocab_token_to_id_.end()) {
                    ids.push_back(direct->second);
                    continue;
                }
            }
            std::vector<std::string> symbols = byte_level_symbols(word);
            while (symbols.size() > 1) {
                control.check();
                int best_rank       = std::numeric_limits<int>::max();
                std::size_t best_at = symbols.size();
                for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                    control.step();
                    const auto it =
                        bpe_merge_ranks_.find(merge_pair_key(symbols[i], symbols[i + 1]));
                    if (it != bpe_merge_ranks_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_at   = i;
                    }
                }
                if (best_at == symbols.size()) { break; }
                symbols[best_at] += symbols[best_at + 1];
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_at + 1));
            }
            for (const std::string& sym : symbols) {
                const auto it = vocab_token_to_id_.find(sym);
                if (it == vocab_token_to_id_.end()) {
                    throw std::invalid_argument("inkling2 tokenizer: unknown BPE symbol");
                }
                ids.push_back(it->second);
            }
        }
    };

    if (!parse_added_tokens) {
        encode_ordinary(text);
        return ids;
    }

    std::size_t ordinary_begin = 0;
    std::size_t pos            = 0;
    while (pos < text.size()) {
        control.step();
        const AddedToken* matched = nullptr;
        const auto& cands = added_token_candidates_[static_cast<unsigned char>(text[pos])];
        for (const std::size_t idx : cands) {
            const AddedToken& tok = added_tokens_[idx];
            if (tok.content.size() <= text.size() - pos &&
                text.compare(pos, tok.content.size(), tok.content) == 0) {
                matched = &tok;
                break;
            }
        }
        if (matched == nullptr) {
            ++pos;
            continue;
        }
        if (pos > ordinary_begin) {
            encode_ordinary(text.substr(ordinary_begin, pos - ordinary_begin));
        }
        ids.push_back(matched->id);
        pos += matched->content.size();
        ordinary_begin = pos;
    }
    if (ordinary_begin < text.size()) { encode_ordinary(text.substr(ordinary_begin)); }
    return ids;
}

std::string Tokenizer::decode_token_bytes(TokenId id, bool skip_special_tokens) const {
    static const std::unordered_map<std::uint32_t, char> kDecoder = build_byte_level_decoder();
    if (!is_valid_token(id)) {
        throw std::out_of_range("inkling2 tokenizer: invalid token id " + std::to_string(id));
    }
    const auto idx = static_cast<std::size_t>(id);
    if (skip_special_tokens && special_token_ids_[idx]) { return {}; }
    const std::string& token = id_to_token_[idx];
    if (added_token_ids_[idx]) { return token; }

    std::string bytes;
    const auto cps = uni::utf8_codepoints(token, "inkling2 decode_token_bytes");
    bytes.reserve(cps.size());
    for (const auto& cp : cps) {
        const auto it = kDecoder.find(static_cast<std::uint32_t>(cp.value));
        if (it == kDecoder.end()) {
            throw std::invalid_argument("inkling2 tokenizer: invalid byte-level codepoint");
        }
        bytes.push_back(it->second);
    }
    return bytes;
}

bool Tokenizer::is_special_token(TokenId id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < special_token_ids_.size() &&
           special_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_valid_token(TokenId id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < valid_token_ids_.size() &&
           valid_token_ids_[static_cast<std::size_t>(id)];
}

} // namespace ninfer::targets::inkling2_mini::frontend_internal
