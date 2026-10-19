#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include "targets/qwen3_6/impl/runtime/image_stream.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

bool same_grid(const VisionGrid& left, const VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<TokenSpan>& left, const std::vector<TokenSpan>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const TokenSpan& a, const TokenSpan& b) {
                                                         return a.begin == b.begin &&
                                                                a.count == b.count;
                                                     });
}

bool same_item(const VisionItem& left, const VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool prefix_item_count(const std::vector<VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (first.count == 0 || last.count == 0 ||
            last.begin > std::numeric_limits<std::size_t>::max() - last.count) {
            return false;
        }
        const std::size_t end = last.begin + last.count;
        if (end <= tokens) {
            if (saw_suffix) { return false; }
            ++*count;
        } else if (first.begin >= tokens) {
            saw_suffix = true;
        } else {
            // A reusable frontier may not divide the consumers of one Vision item.
            return false;
        }
    }
    return true;
}

} // namespace

void ResidentPrefixIdentity::reserve(std::size_t tokens) {
    token_types_.reserve(tokens);
    for (auto& axis : positions_) { axis.reserve(tokens); }
}

void ResidentPrefixIdentity::clear() noexcept {
    token_types_.clear();
    for (auto& axis : positions_) { axis.clear(); }
    vision_items_.clear();
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_ = prompt.vision_items;
}

void ResidentPrefixIdentity::append_generated(std::size_t count, std::int32_t rope_delta) {
    const std::size_t begin = size();
    if (count > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated prefix identity length overflows size_t");
    }
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated prefix position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated MRoPE position exceeds int32");
        }
        token_types_.push_back(0);
        for (auto& axis : positions_) { axis.push_back(static_cast<std::int32_t>(position)); }
    }
}

void ResidentPrefixIdentity::truncate(std::size_t tokens) {
    if (tokens > size()) {
        throw std::out_of_range("cannot extend resident prefix identity by truncation");
    }
    std::size_t retained_items = 0;
    if (!prefix_item_count(vision_items_, tokens, &retained_items)) {
        throw std::logic_error("resident prefix truncation divides a Vision item");
    }
    token_types_.resize(tokens);
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
}

bool ResidentPrefixIdentity::same_prefix(const ResidentPrefixIdentity& other,
                                         std::size_t count) const noexcept {
    if (count > size() || count > other.size()) { return false; }
    if (!std::equal(token_types_.begin(),
                    token_types_.begin() + static_cast<std::ptrdiff_t>(count),
                    other.token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        if (!std::equal(positions_[axis].begin(),
                        positions_[axis].begin() + static_cast<std::ptrdiff_t>(count),
                        other.positions_[axis].begin())) {
            return false;
        }
    }
    std::size_t own_items   = 0;
    std::size_t other_items = 0;
    if (!prefix_item_count(vision_items_, count, &own_items) ||
        !prefix_item_count(other.vision_items_, count, &other_items) ||
        own_items != other_items) {
        return false;
    }
    for (std::size_t i = 0; i < own_items; ++i) {
        if (!same_item(vision_items_[i], other.vision_items_[i])) { return false; }
    }
    return true;
}

std::size_t ResidentPrefixIdentity::bytes() const noexcept {
    std::size_t total = token_types_.capacity();
    for (const auto& axis : positions_) { total += axis.capacity() * sizeof(std::int32_t); }
    // Vision items are rare in this deployment; count a conservative per-item estimate.
    total += vision_items_.capacity() * sizeof(VisionItem);
    return total;
}

bool ResidentPrefixIdentity::matches(const PreparedPromptData& prompt, std::size_t count) const {
    const std::size_t prompt_tokens = prompt.token_ids.size();
    if (count > prompt_tokens || count > size() || prompt.token_types.size() != prompt_tokens ||
        prompt.positions.size() != 3 * prompt_tokens) {
        return false;
    }
    if (!std::equal(prompt.token_types.begin(),
                    prompt.token_types.begin() + static_cast<std::ptrdiff_t>(count),
                    token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * prompt_tokens);
        if (!std::equal(begin, begin + static_cast<std::ptrdiff_t>(count),
                        positions_[axis].begin())) {
            return false;
        }
    }

    std::size_t incoming_items = 0;
    std::size_t resident_items = 0;
    if (!prefix_item_count(prompt.vision_items, count, &incoming_items) ||
        !prefix_item_count(vision_items_, count, &resident_items) ||
        incoming_items != resident_items) {
        return false;
    }
    for (std::size_t i = 0; i < incoming_items; ++i) {
        if (!same_item(prompt.vision_items[i], vision_items_[i])) { return false; }
    }
    return true;
}

void ResidentPrefixIdentity::save(std::ostream& out) const {
    using namespace image_stream;
    // The envelope (magic/version) is written by sequence_image_io.h; this is the payload.
    put_size(out, token_types_.size());
    put_bytes(out, token_types_.data(), token_types_.size());
    for (const auto& positions : positions_) { put_vector<std::int32_t>(out, positions); }
    put_size(out, vision_items_.size());
    for (const VisionItem& item : vision_items_) {
        put_scalar<std::uint8_t>(out, static_cast<std::uint8_t>(item.modality));
        put_scalar<std::int32_t>(out, item.grid.temporal);
        put_scalar<std::int32_t>(out, item.grid.height);
        put_scalar<std::int32_t>(out, item.grid.width);
        put_size(out, item.patch_begin);
        put_size(out, item.patch_count);
        put_bytes(out, item.content_digest.data(), item.content_digest.size());
        put_size(out, item.timestamps.size());
        for (const double timestamp : item.timestamps) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &timestamp, sizeof(bits));
            put_scalar<std::uint64_t>(out, bits);
        }
        put_size(out, item.token_spans.size());
        for (const TokenSpan& span : item.token_spans) {
            put_size(out, span.begin);
            put_size(out, span.count);
        }
    }
}

ResidentPrefixIdentity ResidentPrefixIdentity::load(std::istream& in) {
    using namespace image_stream;
    constexpr std::size_t kMaximumTokens  = 1U << 20;  // 1,048,576 tokens
    constexpr std::size_t kMaximumItems   = 8192;
    constexpr std::size_t kMaximumSpans   = 1U << 20;
    constexpr std::size_t kMaximumStamps  = 1U << 20;
    ResidentPrefixIdentity identity;
    identity.token_types_ = get_vector<std::uint8_t>(in, kMaximumTokens);
    for (auto& positions : identity.positions_) {
        positions = get_vector<std::int32_t>(in, kMaximumTokens * 3);
        if (positions.size() != identity.token_types_.size()) {
            throw std::runtime_error("sequence image identity positions do not match its tokens");
        }
    }
    const std::size_t items = get_size(in, kMaximumItems);
    identity.vision_items_.reserve(items);
    for (std::size_t i = 0; i < items; ++i) {
        VisionItem item;
        item.modality      = static_cast<PromptModality>(get_scalar<std::uint8_t>(in));
        item.grid.temporal = get_scalar<std::int32_t>(in);
        item.grid.height   = get_scalar<std::int32_t>(in);
        item.grid.width    = get_scalar<std::int32_t>(in);
        item.patch_begin   = get_size(in, std::numeric_limits<std::size_t>::max());
        item.patch_count   = get_size(in, std::numeric_limits<std::size_t>::max());
        get_bytes(in, item.content_digest.data(), item.content_digest.size());
        const std::size_t stamps = get_size(in, kMaximumStamps);
        item.timestamps.reserve(stamps);
        for (std::size_t s = 0; s < stamps; ++s) {
            const std::uint64_t bits = get_scalar<std::uint64_t>(in);
            double timestamp         = 0.0;
            std::memcpy(&timestamp, &bits, sizeof(timestamp));
            item.timestamps.push_back(timestamp);
        }
        const std::size_t spans = get_size(in, kMaximumSpans);
        item.token_spans.reserve(spans);
        for (std::size_t s = 0; s < spans; ++s) {
            item.token_spans.push_back(
                TokenSpan{.begin = get_size(in, std::numeric_limits<std::size_t>::max()),
                          .count = get_size(in, std::numeric_limits<std::size_t>::max())});
        }
        identity.vision_items_.push_back(std::move(item));
    }
    return identity;
}

std::uint64_t ResidentPrefixIdentity::fingerprint(std::size_t count) const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix     = [&](const void* data, std::size_t bytes) {
        const auto* bytes_ptr = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            hash ^= bytes_ptr[i];
            hash *= 1099511628211ULL;
        }
    };
    count = std::min(count, token_types_.size());
    mix(token_types_.data(), count);
    for (const auto& axis : positions_) { mix(axis.data(), count * sizeof(std::int32_t)); }
    for (const VisionItem& item : vision_items_) {
        if (item.token_spans.empty() || item.token_spans.front().begin >= count) { continue; }
        mix(&item.grid, sizeof(item.grid));
        mix(&item.patch_begin, sizeof(item.patch_begin));
        mix(&item.patch_count, sizeof(item.patch_count));
        mix(item.content_digest.data(), item.content_digest.size());
        mix(item.timestamps.data(), item.timestamps.size() * sizeof(double));
        for (const TokenSpan& span : item.token_spans) {
            mix(&span.begin, sizeof(span.begin));
            mix(&span.count, sizeof(span.count));
        }
    }
    return hash;
}

std::uint64_t prefix_fingerprint(std::span<const TokenId> tokens,
                                 const ResidentPrefixIdentity& identity,
                                 std::size_t count) noexcept {
    const std::size_t length = std::min(count, tokens.size());
    std::uint64_t hash       = 1469598103934665603ULL;
    const auto* bytes        = reinterpret_cast<const std::uint8_t*>(tokens.data());
    for (std::size_t i = 0; i < length * sizeof(TokenId); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    hash ^= identity.fingerprint(length);
    hash *= 1099511628211ULL;
    return hash;
}

bool prefix_matches(const PreparedPromptData& prompt, const std::vector<TokenId>& resident_tokens,
                    const ResidentPrefixIdentity& resident_identity, std::size_t count) {
    if (count > prompt.token_ids.size() || count > resident_tokens.size()) { return false; }
    return std::equal(prompt.token_ids.begin(),
                      prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count),
                      resident_tokens.begin()) &&
           resident_identity.matches(prompt, count);
}

} // namespace ninfer::targets::qwen3_6::detail
