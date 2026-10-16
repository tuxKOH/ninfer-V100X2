#pragma once

// Host-resident parking: the complete continuation state of a finished conversation, held off the
// GPU so that another conversation can own the KV pool. Prefix reuse keeps one finished
// conversation per lane resident; parking extends that to as many conversations as the host byte
// budget allows, at the cost of one device-to-host copy each way.

#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
// For kMtpDecodeMaximumDrafts: the parked draft train is sized exactly like the live one.
#include <ninfer/targets/qwen3_6/round_state.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

/**
 * One parked conversation.
 *
 * Everything the target keeps on the device besides the request itself is mirrored here: each
 * rank's packed KV page payload (text, and backend under MTP), each rank's GDN recurrent and
 * convolution slot, and rank 0's tail hidden. The host ledger and its ResidentPrefixIdentity are
 * what authorize a later restore, so a parked entry is matched by exactly the token-prefix
 * predicate a resident lane is matched by -- parking adds storage, not a new notion of identity.
 *
 * A parked entry deliberately carries no rewrite checkpoint. Dropping it costs only the turn or
 * response checkpoint reuse opportunity, never correctness: the execution frontier is preserved,
 * and a conversation that comes back resumes through the resident AppendAtFrontier path from that
 * frontier. `bytes` is the total host cost of the entry and is what the store budgets against.
 */
struct ParkedSequence {
    std::uint64_t id = 0;
    std::vector<TokenId> ledger;
    ResidentPrefixIdentity prefix_identity;
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::uint32_t mtp_kv_valid       = 0;
    std::int32_t rope_delta          = 0;
    bool tail_hidden_valid           = false;
    std::uint32_t mtp_draft_count    = 0;
    std::array<TokenId, kMtpDecodeMaximumDrafts> mtp_drafts{};

    // Device payloads. Index 0 is rank 0 and index `rank` is that peer, matching the sequence
    // bundle's own rank ordering; every vector is sized to the device count. `backend` and
    // `tail_hidden` exist only for the ranks that have one.
    std::vector<std::unique_ptr<PagedKVImage>> text;
    std::vector<std::unique_ptr<PagedKVImage>> backend;
    std::vector<std::vector<std::byte>> linear_state;
    std::vector<std::vector<std::byte>> tail_hidden;

    std::size_t bytes       = 0;
    std::uint64_t last_used = 0;
};

/**
 * Bounded, prefix-keyed store of parked conversations.
 *
 * Capacity is a byte budget, not a count: inserting past it evicts least-recently-used entries
 * until the new entry fits, and an entry larger than the whole budget is refused rather than
 * stored. Nothing here touches a device; the caller owns every copy in and out.
 *
 * This is a cache, not a queue: an entry can be evicted at any insert, so a caller that plans a
 * restore must re-establish the entry after parking whatever occupied the lane, and tolerate it
 * having gone away.
 */
class ParkedSequenceStore {
public:
    ParkedSequenceStore() noexcept = default;
    explicit ParkedSequenceStore(std::uint64_t budget_bytes) noexcept
        : budget_bytes_(budget_bytes) {}

    [[nodiscard]] bool enabled() const noexcept { return budget_bytes_ != 0; }
    [[nodiscard]] std::uint64_t budget_bytes() const noexcept { return budget_bytes_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }
    [[nodiscard]] std::size_t entry_count() const noexcept { return entries_.size(); }

    // Stores `entry` under a fresh id, evicting least-recently-used entries until it fits. Returns
    // the assigned id, or nullopt when the store is disabled or the entry alone exceeds the whole
    // budget -- in which case nothing is stored and the entry is left untouched on the caller's
    // side. `now` is a caller-supplied monotonic tick ordered the same way at every call site.
    std::optional<std::uint64_t> insert(std::unique_ptr<ParkedSequence> entry, std::uint64_t now);

    // The most valuable parked entry whose resident prefix is a prefix of `prompt`: the largest
    // resident frontier wins, and a tie goes to the most recently used. Returns nullptr when the
    // store is empty or nothing matches. The pointer stays valid until the next insert or erase.
    [[nodiscard]] const ParkedSequence* match(const PreparedPromptData& prompt) const;

    [[nodiscard]] const ParkedSequence* find_by_id(std::uint64_t id) const noexcept;

    // Marks an entry as just used, so it is the last candidate for the next insertion's eviction.
    void touch(std::uint64_t id, std::uint64_t now) noexcept;
    void erase(std::uint64_t id) noexcept;
    void clear() noexcept;

private:
    void evict_least_recently_used() noexcept;

    std::vector<std::unique_ptr<ParkedSequence>> entries_;
    std::uint64_t budget_bytes_ = 0;
    std::size_t used_bytes_     = 0;
    std::uint64_t next_id_      = 1;
};

inline std::optional<std::uint64_t> ParkedSequenceStore::insert(
    std::unique_ptr<ParkedSequence> entry, std::uint64_t now) {
    if (!enabled() || entry == nullptr || entry->bytes > budget_bytes_) { return std::nullopt; }
    while (used_bytes_ + entry->bytes > budget_bytes_) {
        const std::size_t before = entries_.size();
        evict_least_recently_used();
        if (entries_.size() == before) { return std::nullopt; }
    }
    entry->id        = next_id_++;
    entry->last_used = now;
    used_bytes_ += entry->bytes;
    const std::uint64_t id = entry->id;
    entries_.push_back(std::move(entry));
    return id;
}

inline const ParkedSequence* ParkedSequenceStore::match(const PreparedPromptData& prompt) const {
    const ParkedSequence* best = nullptr;
    for (const std::unique_ptr<ParkedSequence>& entry : entries_) {
        if (entry->execution_frontier == 0) { continue; }
        if (!prefix_matches(prompt, entry->ledger, entry->prefix_identity,
                            entry->execution_frontier)) {
            continue;
        }
        if (best == nullptr || entry->execution_frontier > best->execution_frontier ||
            (entry->execution_frontier == best->execution_frontier &&
             entry->last_used > best->last_used)) {
            best = entry.get();
        }
    }
    return best;
}

inline const ParkedSequence* ParkedSequenceStore::find_by_id(std::uint64_t id) const noexcept {
    for (const std::unique_ptr<ParkedSequence>& entry : entries_) {
        if (entry->id == id) { return entry.get(); }
    }
    return nullptr;
}

inline void ParkedSequenceStore::touch(std::uint64_t id, std::uint64_t now) noexcept {
    for (const std::unique_ptr<ParkedSequence>& entry : entries_) {
        if (entry->id == id) {
            entry->last_used = now;
            return;
        }
    }
}

inline void ParkedSequenceStore::erase(std::uint64_t id) noexcept {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if ((*it)->id == id) {
            used_bytes_ -= (*it)->bytes;
            entries_.erase(it);
            return;
        }
    }
}

inline void ParkedSequenceStore::clear() noexcept {
    entries_.clear();
    used_bytes_ = 0;
}

inline void ParkedSequenceStore::evict_least_recently_used() noexcept {
    if (entries_.empty()) { return; }
    std::size_t victim = 0;
    for (std::size_t i = 1; i < entries_.size(); ++i) {
        if (entries_[i]->last_used < entries_[victim]->last_used) { victim = i; }
    }
    used_bytes_ -= entries_[victim]->bytes;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(victim));
}

} // namespace ninfer::targets::qwen3_6::detail
