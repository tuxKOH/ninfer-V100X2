#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class ResidentPrefixIdentity {
public:
    // Persistence for the disk-backed shared prefix and spilled parked conversations. The format
    // is the compact in-memory layout, length-prefixed; see sequence_image_io.h for the envelope.
    void save(std::ostream& out) const;
    static ResidentPrefixIdentity load(std::istream& in);
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;

    // True when the first `count` identity entries are identical. Used to tell whether two
    // retained checkpoints at the same frontier describe the same token/position lineage.
    [[nodiscard]] bool same_prefix(const ResidentPrefixIdentity& other,
                                   std::size_t count) const noexcept;

    // Approximate heap size of the metadata, for host budget accounting.
    [[nodiscard]] std::size_t bytes() const noexcept;

    // Stable content hash of the first `count` identity entries, used to deduplicate persisted
    // shared prefixes. Not cryptographic; a 64-bit FNV-1a over the compact representation.
    [[nodiscard]] std::uint64_t fingerprint(std::size_t count) const noexcept;

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  const std::vector<TokenId>& resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

// Content key for a persisted prefix: the identity fields together with the token IDs they were
// assigned from. Two prompts of equal length but different tokens (different system prompts of
// the same size) must not share a key.
[[nodiscard]] std::uint64_t prefix_fingerprint(std::span<const TokenId> tokens,
                                               const ResidentPrefixIdentity& identity,
                                               std::size_t count) noexcept;

} // namespace ninfer::targets::qwen3_6::detail
