#pragma once

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
    [[nodiscard]] std::string rendered_content(bool add_vision_id = false,
                                               int* image_count   = nullptr,
                                               int* video_count   = nullptr) const;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    bool add_vision_id = false;
    // Records a second reusable boundary at the end of the message before the last one (see
    // PromptIdentity::stable_checkpoint). Off by default: it costs one extra prefix
    // tokenization per request.
    bool message_boundary_checkpoint = false;
    // How many recent message boundaries to retain when `message_boundary_checkpoint` is on,
    // counting the newest (the stable boundary). Clients can rewrite an assistant message
    // several turns back -- dropping the tool_use of an interrupted tool call also drops its
    // tool_result -- so one boundary before the final message cannot cover the divergence.
    // 0 or 1 keeps only the stable boundary.
    std::uint32_t boundary_history = 1;
    // Records reusable boundaries in the static leading block before the first conversation
    // message: one after the tool definitions, and one after the whole block (tool definitions and
    // instruction/system messages). Agent frontends send that block first and it is identical
    // across sessions, so checkpoints there let a new conversation restore an already-prefilled
    // prefix. The tool boundary survives changes in the system text that follows it (for example
    // an environment line). See PromptIdentity::tools_checkpoint/prefix_checkpoint.
    bool shared_prefix_checkpoint = false;
    std::vector<std::string> tool_jsons;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::optional<RewriteCheckpointByteSpec> stable_checkpoint;
    // Older message boundaries, oldest first, retained for clients that rewrite a message
    // several turns back.
    std::vector<RewriteCheckpointByteSpec> message_boundaries;
    std::optional<RewriteCheckpointByteSpec> prefix_checkpoint;
    // End of the tool definitions, before the leading instruction text that follows them.
    std::optional<RewriteCheckpointByteSpec> tools_checkpoint;
};

enum class ChatTemplateSemantics : std::uint8_t {
    ThinkingToggle,
    ReasoningEffort,
};

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate resolve(std::string_view source);

    [[nodiscard]] PromptCapabilities capabilities() const noexcept;
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options = {}) const;

private:
    explicit CompiledChatTemplate(ChatTemplateSemantics semantics) noexcept
        : semantics_(semantics) {}

    ChatTemplateSemantics semantics_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
