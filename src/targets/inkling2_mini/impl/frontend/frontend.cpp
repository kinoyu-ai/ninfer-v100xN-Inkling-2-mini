#include <ninfer/targets/inkling2_mini/frontend.h>
#include <ninfer/targets/inkling2_mini/prepared_prompt.h>

#include "targets/inkling2_mini/impl/frontend/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::inkling2_mini {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend_internal;

constexpr TokenId kTokMessageModel         = 200001; // <|message_model|>
constexpr TokenId kTokContentText          = 200004; // <|content_text|>
constexpr TokenId kTokContentThinking      = 200008; // <|content_thinking|>
constexpr TokenId kTokEndMessage           = 200010; // <|end_message|>
constexpr TokenId kTokContentToolError     = 200022; // <|content_tool_error|>
constexpr TokenId kTokContentInvokeToolJsn = 200049; // <|content_invoke_tool_json|>
constexpr TokenId kTokContentInvokeToolTxt = 200057; // <|content_invoke_tool_text|>

constexpr std::string_view kReplacementCharacter = "\xef\xbf\xbd";

std::size_t decode_utf8_prefix(std::string_view bytes, std::string& out) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto lead    = static_cast<unsigned char>(bytes[offset]);
        std::size_t length = 0;
        unsigned char low  = 0x80U;
        unsigned char high = 0xbfU;
        if (lead <= 0x7fU) {
            out.push_back(static_cast<char>(lead));
            ++offset;
            continue;
        }
        if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
            if (lead == 0xe0U) { low = 0xa0U; }
            if (lead == 0xedU) { high = 0x9fU; }
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
            if (lead == 0xf0U) { low = 0x90U; }
            if (lead == 0xf4U) { high = 0x8fU; }
        } else {
            out.append(kReplacementCharacter);
            ++offset;
            continue;
        }
        std::size_t index = 1;
        bool well_formed  = true;
        for (; index < length && offset + index < bytes.size(); ++index) {
            const auto byte             = static_cast<unsigned char>(bytes[offset + index]);
            const unsigned char minimum = index == 1 ? low : 0x80U;
            const unsigned char maximum = index == 1 ? high : 0xbfU;
            if (byte < minimum || byte > maximum) {
                well_formed = false;
                break;
            }
        }
        if (!well_formed) {
            out.append(kReplacementCharacter);
            offset += index;
            continue;
        }
        if (offset + length > bytes.size()) { return offset; }
        out.append(bytes.substr(offset, length));
        offset += length;
    }
    return offset;
}

std::string_view role_token_of(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        return "<|message_system|>";
    case ChatRole::User:
        return "<|message_user|>";
    case ChatRole::Assistant:
        return "<|message_model|>";
    case ChatRole::Tool:
        return "<|message_tool|>";
    }
    throw std::invalid_argument("unsupported chat role");
}

std::string render_thinking_effort(const PromptOptions& options) {
    std::string_view effort_str = "0.9";
    if (!options.enable_thinking) {
        effort_str = "0";
    } else if (options.reasoning_effort) {
        switch (*options.reasoning_effort) {
        case ReasoningEffort::Low:
            effort_str = "0.2";
            break;
        case ReasoningEffort::Medium:
            effort_str = "0.7";
            break;
        case ReasoningEffort::XHigh:
            effort_str = "0.99";
            break;
        }
    }
    std::string out = "<|message_system|><|content_text|>Thinking effort level: ";
    out.append(effort_str);
    out.append("<|end_message|>");
    return out;
}

// Renders messages strictly following thinkingmachines/Inkling-2-Mini chat_template.jinja.
std::string render_chat_prompt(const PromptInput& input) {
    std::string out;

    // Optional tools declaration block
    if (!input.options.tool_jsons.empty()) {
        Json specs = Json::array();
        for (const std::string& raw : input.options.tool_jsons) {
            const Json tool = Json::parse(raw);
            const Json& fn  = tool.contains("function") ? tool.at("function") : tool;
            Json spec       = Json::object();
            spec["description"] =
                (fn.contains("description") && fn.at("description").is_string())
                    ? fn.at("description").get<std::string>()
                    : "";
            spec["name"]       = fn.at("name");
            spec["parameters"] = fn.contains("parameters") ? fn.at("parameters") : Json::object();
            spec["type"] = (tool.contains("type") && tool.at("type").is_string())
                               ? tool.at("type").get<std::string>()
                               : "function";
            specs.push_back(std::move(spec));
        }
        out += "<|message_system|>tool_declare<|content_xml|>";
        out += specs.dump(-1, ' ', false, Json::error_handler_t::strict);
        out += "<|end_message|>";
    }

    bool effort_emitted = false;
    for (const ChatMessage& msg : input.messages) {
        for (const MessagePart& part : msg.parts) {
            if (part.kind == MessagePartKind::Media) {
                throw std::invalid_argument(
                    "inkling2-mini target currently supports text prompts only");
            }
        }
        const bool is_system = (msg.role == ChatRole::System || msg.role == ChatRole::Developer);
        if (!effort_emitted && !is_system) {
            out += render_thinking_effort(input.options);
            effort_emitted = true;
        }

        std::string rtok(role_token_of(msg.role));
        if (msg.role == ChatRole::Tool && !msg.tool_call_id.empty()) {
            for (const ChatMessage& prev : input.messages) {
                if (prev.role == ChatRole::Assistant) {
                    for (const ToolCall& tc : prev.tool_calls) {
                        if (tc.id == msg.tool_call_id && !tc.name.empty()) { rtok += tc.name; }
                    }
                }
            }
        }

        if (msg.role == ChatRole::Assistant && !msg.reasoning_content.empty()) {
            out += rtok;
            out += "<|content_thinking|>";
            out += msg.reasoning_content;
            out += "<|end_message|>";
        }

        std::string joined_text;
        for (const MessagePart& part : msg.parts) { joined_text += part.text; }
        const bool emit_text =
            !joined_text.empty() ||
            !(msg.role == ChatRole::Assistant &&
              (!msg.tool_calls.empty() || !msg.reasoning_content.empty()));
        if (emit_text) {
            out += rtok;
            out += "<|content_text|>";
            out += joined_text;
            out += "<|end_message|>";
        }

        if (msg.role == ChatRole::Assistant && !msg.tool_calls.empty()) {
            for (const ToolCall& tc : msg.tool_calls) {
                out += "<|message_model|>";
                out += tc.name;
                out += "<|content_invoke_tool_json|>";
                Json args = tc.arguments_json.empty() ? Json::object() : Json::parse(tc.arguments_json);
                Json payload;
                payload["name"] = tc.name;
                payload["args"] = std::move(args);
                out += payload.dump(-1, ' ', false, Json::error_handler_t::strict);
                out += "<|end_message|>";
            }
        }

        if (msg.role == ChatRole::Assistant) { out += "<|content_model_end_sampling|>"; }
    }

    if (!effort_emitted) { out += render_thinking_effort(input.options); }
    return out;
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    auto add_id               = [&](TokenId id) {
        if (!tokenizer.is_valid_token(id)) {
            throw std::invalid_argument("stop token id outside inkling2 vocabulary: " +
                                        std::to_string(id));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), id) ==
            result.token_ids.end()) {
            result.token_ids.push_back(id);
        }
    };
    if (caller.include_model_defaults) {
        for (const TokenId id : tokenizer.default_stop_token_ids()) { add_id(id); }
    }
    for (const TokenId id : caller.token_ids) { add_id(id); }

    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ninfer::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        result.strings.push_back(stop);
    }
    return result;
}

void append_delta(PublishedOutput& out, OutputChannel ch, std::string text) {
    if (text.empty()) { return; }
    if (!out.empty() && out.back().channel == ch) {
        out.back().text += text;
    } else {
        out.push_back(OutputDelta{.channel = ch, .text = std::move(text)});
    }
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

std::size_t channel_idx(OutputChannel ch) noexcept {
    return ch == OutputChannel::Reasoning ? 0 : 1;
}

enum class StreamPhase : std::uint8_t {
    BetweenMessages,
    InHeader,
    InReasoning,
    InContent,
};

struct DecoderState {
    StreamPhase phase = StreamPhase::BetweenMessages;
    std::string utf8_pending;
    std::array<std::string, 2> stop_pending;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
};

struct StopMatch {
    bool found                     = false;
    std::uint32_t committed_tokens = 0;
    PublishedOutput output;
};

void feed_channel(DecoderState& st, OutputChannel ch, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* match) {
    if (text.empty()) { return; }
    std::string combined = st.stop_pending[channel_idx(ch)];
    combined.append(text);

    if (match != nullptr && !match->found) {
        for (const StopString& stop : policy.strings) {
            if (stop.channel != ch) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            PublishedOutput cand = emitted;
            append_delta(cand, ch, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(cand, ch, stop.text); }
            *match = StopMatch{
                .found            = true,
                .committed_tokens = committed_tokens,
                .output           = std::move(cand),
            };
            return;
        }
    }

    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel == ch) { hold = std::max(hold, longest_suffix_prefix(combined, stop.text)); }
    }
    append_delta(emitted, ch, combined.substr(0, combined.size() - hold));
    st.stop_pending[channel_idx(ch)] = combined.substr(combined.size() - hold);
    st.decoded_bytes += text.size();
}

void flush_active_channel(DecoderState& st, const StopPolicy& policy, PublishedOutput& emitted,
                          std::uint32_t committed_tokens) {
    const OutputChannel ch =
        (st.phase == StreamPhase::InReasoning) ? OutputChannel::Reasoning : OutputChannel::Content;
    if (!st.utf8_pending.empty()) {
        st.utf8_pending.clear();
        if (st.phase == StreamPhase::InReasoning || st.phase == StreamPhase::InContent) {
            feed_channel(st, ch, kReplacementCharacter, policy, emitted, committed_tokens, nullptr);
        }
    }
    for (std::size_t idx = 0; idx < 2; ++idx) {
        if (!st.stop_pending[idx].empty()) {
            const OutputChannel c = (idx == 0) ? OutputChannel::Reasoning : OutputChannel::Content;
            append_delta(emitted, c, std::move(st.stop_pending[idx]));
            st.stop_pending[idx].clear();
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// PreparedPrompt
// ---------------------------------------------------------------------------

PreparedPrompt::PreparedPrompt() noexcept                                  = default;
PreparedPrompt::~PreparedPrompt()                                          = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept                  = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept       = default;
PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr || data_->token_ids.empty()) {
        throw std::logic_error("inkling2 prepared prompt is empty");
    }
    return PromptSummary{
        .prompt_tokens = static_cast<std::uint32_t>(data_->token_ids.size()),
        .has_media     = false,
    };
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    return PromptPreparationStats{
        .seconds          = data_->prepare_seconds,
        .tokenize_seconds = data_->tokenize_seconds,
    };
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::logic_error("inkling2 prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::logic_error("inkling2 prepared prompt is empty"); }
    PreparedPromptData out = std::move(*prompt.data_);
    prompt.data_.reset();
    return out;
}

// ---------------------------------------------------------------------------
// OutputSession
// ---------------------------------------------------------------------------

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tok, StopPolicy pol, OutputOptions opts)
        : tokenizer(std::move(tok)), policy(std::move(pol)), raw(opts.raw),
          preserve_special(opts.raw || opts.preserve_special_tokens) {
        if (raw) { state.phase = StreamPhase::InContent; }
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    bool raw              = false;
    bool preserve_special = false;
    DecoderState state;
    DecoderState preview_state;
    PublishedOutput preview_output;
    bool preview_ready = false;
};

OutputSession::OutputSession() noexcept                                = default;
OutputSession::~OutputSession()                                        = default;
OutputSession::OutputSession(OutputSession&&) noexcept                 = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept      = default;
OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

runtime::OutputDecision OutputSession::preview(std::span<const TokenId> tokens,
                                               std::uint32_t budget_remaining,
                                               FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (tokens.empty() || tokens.size() > budget_remaining) {
        throw std::invalid_argument("invalid output preview token span");
    }

    impl_->preview_state = impl_->state;
    impl_->preview_output.clear();

    auto complete = [&](std::uint32_t count, FinishReason reason) {
        impl_->preview_ready = true;
        return runtime::OutputDecision{.accepted_tokens = count, .finish_reason = reason};
    };

    for (std::size_t idx = 0; idx < tokens.size(); ++idx) {
        const std::uint32_t count = static_cast<std::uint32_t>(idx + 1);
        const TokenId tok         = tokens[idx];
        if (!impl_->tokenizer->is_valid_token(tok)) {
            throw std::out_of_range("generated token outside inkling2 vocabulary: " +
                                    std::to_string(tok));
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), tok) !=
            impl_->policy.token_ids.end();

        if (stop_token) {
            if (impl_->policy.publish_stop_token && impl_->raw) {
                const std::string bytes = impl_->tokenizer->decode_token_bytes(tok, false);
                feed_channel(impl_->preview_state, OutputChannel::Content, bytes, impl_->policy,
                             impl_->preview_output, count, nullptr);
            }
            flush_active_channel(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            impl_->preview_state.terminal = true;
            return complete(count, FinishReason::StopToken);
        }

        if (!impl_->raw) {
            if (tok == kTokMessageModel) {
                flush_active_channel(impl_->preview_state, impl_->policy, impl_->preview_output,
                                     count);
                impl_->preview_state.phase = StreamPhase::InHeader;
                continue;
            }
            if (tok == kTokContentThinking) {
                impl_->preview_state.utf8_pending.clear();
                impl_->preview_state.phase = StreamPhase::InReasoning;
                continue;
            }
            if (tok == kTokContentText || tok == kTokContentToolError ||
                tok == kTokContentInvokeToolJsn || tok == kTokContentInvokeToolTxt) {
                impl_->preview_state.utf8_pending.clear();
                impl_->preview_state.phase = StreamPhase::InContent;
                continue;
            }
            if (tok == kTokEndMessage) {
                flush_active_channel(impl_->preview_state, impl_->policy, impl_->preview_output,
                                     count);
                impl_->preview_state.phase = StreamPhase::BetweenMessages;
                continue;
            }
            if (impl_->preview_state.phase == StreamPhase::InHeader) {
                // Suppress tool/speaker header names between <|message_model|> and <|content_*|>
                continue;
            }
            if (impl_->preview_state.phase == StreamPhase::BetweenMessages) {
                // If model emits raw text without a header tag, treat as content
                impl_->preview_state.phase = StreamPhase::InContent;
            }
        }

        if (impl_->preview_state.phase == StreamPhase::InReasoning) {
            ++impl_->preview_state.reasoning_tokens;
        }

        const std::string bytes =
            impl_->tokenizer->decode_token_bytes(tok, !impl_->preserve_special);
        if (bytes.empty()) { continue; }

        impl_->preview_state.utf8_pending += bytes;
        std::string text;
        const std::size_t resolved = decode_utf8_prefix(impl_->preview_state.utf8_pending, text);
        impl_->preview_state.utf8_pending.erase(0, resolved);
        if (text.empty()) { continue; }

        const OutputChannel ch = (impl_->preview_state.phase == StreamPhase::InReasoning)
                                     ? OutputChannel::Reasoning
                                     : OutputChannel::Content;
        StopMatch match;
        feed_channel(impl_->preview_state, ch, text, impl_->policy, impl_->preview_output, count,
                     &match);
        if (match.found) {
            impl_->preview_state.utf8_pending.clear();
            impl_->preview_state.stop_pending = {};
            impl_->preview_state.terminal     = true;
            impl_->preview_output             = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == budget_remaining) {
        flush_active_channel(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        impl_->preview_state.terminal = true;
        return complete(count, limit_reason);
    }
    return complete(count, FinishReason::None);
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    impl_->preview_state = impl_->state;
    impl_->preview_output.clear();
    flush_active_channel(impl_->preview_state, impl_->policy, impl_->preview_output, 0);
    impl_->preview_state.terminal = true;
    impl_->preview_ready          = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() noexcept {
    if (impl_ == nullptr || !impl_->preview_ready) { return {}; }
    impl_->state         = std::move(impl_->preview_state);
    impl_->preview_ready = false;
    PublishedOutput out  = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    return out;
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

// ---------------------------------------------------------------------------
// Frontend
// ---------------------------------------------------------------------------

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, std::uint32_t max_ctx)
        : tokenizer(std::make_shared<const fi::Tokenizer>(fi::TokenizerResources{
              .tokenizer_json         = resources.tokenizer_json,
              .tokenizer_config_json  = resources.tokenizer_config_json,
              .generation_config_json = resources.generation_config_json,
          })),
          max_context(max_ctx) {
        for (const TokenId id : tokenizer->default_stop_token_ids()) {
            defaults.token_ids.push_back(id);
        }
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    std::uint32_t max_context = 2048;
    StopPolicy defaults;
};

Frontend::Frontend(const Frontend&)            = default;
Frontend& Frontend::operator=(const Frontend&) = default;
Frontend::Frontend(Frontend&&) noexcept        = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                          = default;
Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto t0 = Clock::now();
    const std::string rendered = render_chat_prompt(input);
    const auto t_tok0          = Clock::now();
    std::vector<TokenId> ids   = impl_->tokenizer->encode(rendered, true, control);
    const auto t_end           = Clock::now();
    if (ids.empty()) { throw std::invalid_argument("prepared prompt has zero tokens"); }

    auto data               = std::make_unique<PreparedPromptData>();
    data->token_ids         = std::move(ids);
    data->allow_prefix_reuse = true;
    data->tokenize_seconds  = std::chrono::duration<double>(t_end - t_tok0).count();
    data->prepare_seconds   = std::chrono::duration<double>(t_end - t0).count();
    return PreparedPrompt(std::move(data));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    const std::string rendered = render_chat_prompt(input);
    return static_cast<std::uint32_t>(impl_->tokenizer->encode(rendered, true, control).size());
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    if (token_ids.empty()) {
        throw std::invalid_argument("prepare_tokens requires at least one token");
    }
    for (const TokenId id : token_ids) {
        if (!impl_->tokenizer->is_valid_token(id)) {
            throw std::invalid_argument("prepare_tokens: token id out of vocabulary: " +
                                        std::to_string(id));
        }
    }
    auto data               = std::make_unique<PreparedPromptData>();
    data->token_ids         = std::move(token_ids);
    data->allow_prefix_reuse = allow_prefix_identity;
    return PreparedPrompt(std::move(data));
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept {
    return PromptCapabilities{
        .enable_thinking = true,
        .reasoning_effort =
            ReasoningEffortCapabilities{
                .low            = true,
                .medium         = true,
                .xhigh          = true,
                .default_effort = ReasoningEffort::Medium,
            },
    };
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

OutputSession Frontend::make_output_session(const PreparedPrompt& /*prompt*/,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output) const {
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    return OutputSession(std::make_unique<OutputSession::Impl>(impl_->tokenizer, std::move(policy),
                                                               output));
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

Frontend make_frontend(const FrontendResources& resources, std::uint32_t max_context) {
    return Frontend(std::make_shared<Frontend::Impl>(resources, max_context));
}

} // namespace ninfer::targets::inkling2_mini
