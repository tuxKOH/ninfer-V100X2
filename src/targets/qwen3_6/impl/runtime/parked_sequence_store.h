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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// How many *older* rewrite checkpoints a lane (or parked entry) keeps besides the current one.
// A stale or forked replay of the same conversation matches one of these boundaries, so the
// request resumes at a turn boundary instead of re-prefilling the whole history. The normal
// continuation only ever uses the newest checkpoint, so this is purely a branch-recovery cache.
inline constexpr std::size_t kCheckpointHistoryDepth = 3;

// How many recent message boundaries a lane (and each parked entry) pins outside the ring. One
// boundary is roughly 150 MB of recurrent state across both ranks on the 27B TP2 profile, so
// this is a memory/latency trade: eight boundaries cover several turns of client rewrites.
inline constexpr std::size_t kBoundaryHistoryDepth = 8;

// How many recent response (assistant-header) checkpoints are pinned. A client re-renders the
// assistant message it just received -- tool-call argument JSON and thinking text do not always
// round-trip -- so a tool turn's continuation diverges at that message's header. That header is a
// response checkpoint, not a message boundary, and boundary captures would evict it from the
// depth-3 ring within a turn.
inline constexpr std::size_t kResponseHistoryDepth = 3;

// Sparse long-range checkpoints, one per kLongRangeSpacing tokens, so a client snapshot that lags
// the resident conversation by many turns (background agents resend an older full history) can
// resume within one spacing step instead of re-prefilling the whole gap.
inline constexpr std::uint32_t kLongRangeSpacing = 16384;
inline constexpr std::size_t kLongRangeDepth     = 4;

/**
 * One rewrite checkpoint retained off the device: the boundary, each rank's recurrent slot at
 * that boundary, and rank 0's hidden there. Copies are host buffers, newest-first in a small
 * bounded history.
 */
struct HostCheckpoint {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
    // The token and position lineage this state was produced from (prefix up to `frontier`).
    // Selection MUST validate a replay against this, not against the lane's current ledger: two
    // branches of one conversation share a ledger prefix up to their fork, so a checkpoint from
    // the other branch would otherwise be selected by tokens it never actually saw.
    std::vector<TokenId> ledger;
    ResidentPrefixIdentity identity;
    std::vector<std::vector<std::byte>> state;
    std::vector<std::byte> hidden;
    std::size_t bytes = 0;
};

// True when two retained checkpoints describe the same lineage at the same boundary. Without
// provenance the boundary alone is the key (legacy/unit-test entries).
inline bool same_checkpoint_boundary(const HostCheckpoint& a, const HostCheckpoint& b) noexcept {
    if (a.frontier != b.frontier || a.kind != b.kind) { return false; }
    if (a.ledger.size() < a.frontier || b.ledger.size() < b.frontier) { return true; }
    return std::equal(a.ledger.begin(),
                      a.ledger.begin() + static_cast<std::ptrdiff_t>(a.frontier),
                      b.ledger.begin()) &&
           a.identity.same_prefix(b.identity, a.frontier);
}

// Inserts `checkpoint` at the front of `history`, replacing an entry for the same boundary and
// dropping the oldest beyond `depth`. Boundaries are keyed by kind + frontier + lineage; after a
// branch resume the frontiers in the ring are not monotonic, so callers search all.
inline void push_host_checkpoint(std::vector<HostCheckpoint>& history, HostCheckpoint checkpoint,
                                 std::size_t depth = kCheckpointHistoryDepth) {
    if (checkpoint.frontier == 0 || depth == 0) { return; }
    for (auto it = history.begin(); it != history.end(); ++it) {
        if (same_checkpoint_boundary(*it, checkpoint)) {
            history.erase(it);
            break;
        }
    }
    history.insert(history.begin(), std::move(checkpoint));
    if (history.size() > depth) { history.resize(depth); }
}

// Pins a message boundary, newest first, dropping a duplicate frontier and the oldest entry past
// the depth. Unlike the rewrite-checkpoint ring these are never rotated away by later turns.
inline void push_pinned_boundary(std::vector<HostCheckpoint>& history, HostCheckpoint checkpoint,
                                 std::size_t depth = kBoundaryHistoryDepth) {
    if (checkpoint.frontier == 0 || depth == 0) { return; }
    for (auto it = history.begin(); it != history.end(); ++it) {
        if (it->frontier == checkpoint.frontier) {
            history.erase(it);
            break;
        }
    }
    history.insert(history.begin(), std::move(checkpoint));
    if (history.size() > depth) { history.resize(depth); }
}

/**
 * One parked conversation.
 *
 * Everything the target keeps on the device besides the request itself is mirrored here: each
 * rank's packed KV page payload (text, and backend under MTP), each rank's GDN recurrent and
 * convolution slot, and rank 0's tail hidden. The host ledger and its ResidentPrefixIdentity are
 * what authorize a later restore, so a parked entry is matched by exactly the token-prefix
 * predicate a resident lane is matched by -- parking adds storage, not a new notion of identity.
 *
 * A parked entry also mirrors the lane's rewrite checkpoint when one is valid. Without it, a
 * conversation whose replay diverges inside its last response -- which is the norm when the
 * client re-renders a preserved-thinking turn -- could only ever be matched at the execution
 * frontier, so every park/restore cycle would re-prefill the whole conversation. With it, the
 * entry can be matched at the checkpoint frontier and resume through the same checkpoint-restore
 * path a resident lane uses. `bytes` is the total host cost of the entry and is what the store
 * budgets against.
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

    // The lane's rewrite checkpoint, mirrored only when `checkpoint_valid`. `checkpoint_state`
    // is the per-rank recurrent slot at the checkpoint frontier and `checkpoint_hidden` is rank
    // 0's hidden state there; both are device payloads, like the fields above.
    bool checkpoint_valid                             = false;
    RewriteCheckpointKind checkpoint_kind             = RewriteCheckpointKind::TurnClosure;
    std::uint32_t checkpoint_frontier                 = 0;
    std::vector<TokenId> checkpoint_ledger;
    ResidentPrefixIdentity checkpoint_identity;
    std::vector<std::vector<std::byte>> checkpoint_state;
    std::vector<std::byte> checkpoint_hidden;

    // Older boundaries of the same conversation, newest-first. A forked or stale replay of the
    // conversation can match one of these when it no longer matches the current checkpoint.
    std::vector<HostCheckpoint> checkpoint_history;
    // Pinned recent message boundaries, newest first; see SequenceState::boundary_history.
    std::vector<HostCheckpoint> boundary_history;
    // Pinned recent response (assistant-header) checkpoints, newest first.
    std::vector<HostCheckpoint> response_history;
    // Sparse long-range checkpoints, newest first; see kLongRangeSpacing.
    std::vector<HostCheckpoint> long_range_history;

    // The shared leading-block checkpoint, pinned outside the ring (see SequenceState).
    HostCheckpoint shared_prefix;
    // The tool-definitions anchor, pinned the same way.
    HostCheckpoint tools_prefix;

    // Payload location. A spilled or preloaded entry keeps only the matching metadata in RAM and
    // streams its images back when a request selects it; `bytes` still reports the full payload.
    bool disk_backed = false;
    std::string disk_path;
    std::vector<std::uint32_t> text_pages;
    std::vector<std::uint32_t> backend_pages;
    // Set for entries loaded from disk that must survive eviction (the shared prefix file).
    bool pinned = false;

    std::size_t bytes       = 0;
    std::uint64_t last_used = 0;
};

// Heap bytes the matcher keeps when an entry is disk-backed. Budgeted with the payload so a pile
// of small spilled conversations cannot grow host RAM without bound.
inline std::size_t sequence_metadata_bytes(const ParkedSequence& entry) noexcept {
    std::size_t bytes = entry.ledger.size() * sizeof(TokenId) + entry.prefix_identity.bytes();
    const auto add = [&](const HostCheckpoint& checkpoint) {
        bytes += checkpoint.ledger.size() * sizeof(TokenId) + checkpoint.identity.bytes();
    };
    if (entry.checkpoint_valid) {
        bytes += entry.checkpoint_ledger.size() * sizeof(TokenId) + entry.checkpoint_identity.bytes();
    }
    for (const HostCheckpoint& checkpoint : entry.checkpoint_history) { add(checkpoint); }
    for (const HostCheckpoint& checkpoint : entry.boundary_history) { add(checkpoint); }
    for (const HostCheckpoint& checkpoint : entry.response_history) { add(checkpoint); }
    for (const HostCheckpoint& checkpoint : entry.long_range_history) { add(checkpoint); }
    add(entry.shared_prefix);
    add(entry.tools_prefix);
    return bytes;
}

// Drops the device payload copies from a metadata-only disk index entry, keeping everything the
// matcher reads: frontiers, ledger, identity and the checkpoint lineage.
inline void strip_sequence_payload(ParkedSequence& entry) {
    entry.text.clear();
    entry.backend.clear();
    entry.linear_state.clear();
    entry.tail_hidden.clear();
    entry.checkpoint_state.clear();
    entry.checkpoint_hidden.clear();
    for (HostCheckpoint& checkpoint : entry.checkpoint_history) {
        checkpoint.state.clear();
        checkpoint.hidden.clear();
    }
    for (HostCheckpoint& checkpoint : entry.boundary_history) {
        checkpoint.state.clear();
        checkpoint.hidden.clear();
    }
    for (HostCheckpoint& checkpoint : entry.response_history) {
        checkpoint.state.clear();
        checkpoint.hidden.clear();
    }
    for (HostCheckpoint& checkpoint : entry.long_range_history) {
        checkpoint.state.clear();
        checkpoint.hidden.clear();
    }
    entry.shared_prefix.state.clear();
    entry.shared_prefix.hidden.clear();
    entry.tools_prefix.state.clear();
    entry.tools_prefix.hidden.clear();
}

// Which pinned cross-conversation boundary a plan resumes: the end of the tool definitions or the
// end of the whole static leading block. Both live outside the checkpoint ring.
enum class SharedPrefixSlot : std::uint8_t {
    None    = 0,
    Tools   = 1,
    Leading = 2,
    Message   = 3,
    Response  = 4,
    LongRange = 5,
};

// Which retained boundary a parked conversation can be resumed from. `checkpoint_index` 0 is
// the entry's current checkpoint, i+1 is checkpoint_history[i].
struct ParkedCheckpointMatch {
    const ParkedSequence* entry = nullptr;
    std::uint32_t frontier      = 0;
    std::uint32_t checkpoint_index = 0;
    // The match is one of the entry's pinned shared prefixes rather than a current/historical one.
    SharedPrefixSlot shared = SharedPrefixSlot::None;
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
    // side. `protect_id` names an entry this insert must never evict: when the new entry cannot
    // fit without it, the insert is refused before anything is evicted, so a caller that is about
    // to consume that entry can park an incumbent without sacrificing it. `now` is a
    // caller-supplied monotonic tick ordered the same way at every call site.
    std::optional<std::uint64_t> insert(std::unique_ptr<ParkedSequence> entry, std::uint64_t now,
                                        std::uint64_t protect_id = 0);

    // The most valuable parked entry whose resident prefix is a prefix of `prompt`: the largest
    // resident frontier wins, and a tie goes to the most recently used. Returns nullptr when the
    // store is empty or nothing matches. The pointer stays valid until the next insert or erase.
    [[nodiscard]] std::vector<const ParkedSequence*> candidates() const;
    [[nodiscard]] const ParkedSequence* match(const PreparedPromptData& prompt) const;

    // The best rewrite-checkpoint boundary across every entry (current and historical). A
    // boundary matches when its frontier is a prefix of `prompt`; the largest frontier wins and
    // a tie goes to the most recently used entry. Used only compared against an exact match at
    // the execution frontier, so an exact resume always wins at an equal frontier.
    [[nodiscard]] std::optional<ParkedCheckpointMatch>
    match_checkpoint(const PreparedPromptData& prompt) const;

    // Disk spill support: the caller writes the removed entry out, then reinserts a metadata-only
    // index entry (or drops it when the write fails).
    std::unique_ptr<ParkedSequence> remove_least_recently_used();

    [[nodiscard]] const ParkedSequence* find_by_id(std::uint64_t id) const noexcept;
    // Non-const: a restore takes ownership of the entry's host payloads (they are moved into the
    // lane) before the entry is erased.
    [[nodiscard]] ParkedSequence* find_by_id(std::uint64_t id) noexcept;

    // Marks an entry as just used, so it is the last candidate for the next insertion's eviction.
    void touch(std::uint64_t id, std::uint64_t now) noexcept;
    void erase(std::uint64_t id) noexcept;
    void clear() noexcept;

private:
    bool evict_least_recently_used(std::uint64_t protect_id) noexcept;

    std::vector<std::unique_ptr<ParkedSequence>> entries_;
    std::uint64_t budget_bytes_ = 0;
    std::size_t used_bytes_     = 0;
    std::uint64_t next_id_      = 1;
};

inline std::optional<std::uint64_t> ParkedSequenceStore::insert(
    std::unique_ptr<ParkedSequence> entry, std::uint64_t now, std::uint64_t protect_id) {
    if (!enabled() || entry == nullptr || entry->bytes > budget_bytes_) { return std::nullopt; }
    if (used_bytes_ + entry->bytes > budget_bytes_) {
        // Refuse before evicting anything when even dropping every unprotected entry cannot make
        // room: a refused insert must leave the store exactly as it found it.
        std::size_t protected_bytes = 0;
        for (const std::unique_ptr<ParkedSequence>& stored : entries_) {
            if (stored->id == protect_id) { protected_bytes += stored->bytes; }
        }
        if (protected_bytes + entry->bytes > budget_bytes_) { return std::nullopt; }
        while (used_bytes_ + entry->bytes > budget_bytes_) {
            if (!evict_least_recently_used(protect_id)) { return std::nullopt; }
        }
    }
    entry->id        = next_id_++;
    entry->last_used = now;
    used_bytes_ += entry->bytes;
    const std::uint64_t id = entry->id;
    entries_.push_back(std::move(entry));
    return id;
}

// Matching is defined over an arbitrary collection of candidates so the resident store and the
// disk-backed index (metadata-only entries) share one predicate. Each candidate must outlive the
// call; the returned pointer names the entry the caller should restore.
inline const ParkedSequence* match_parked_entries(std::span<const ParkedSequence* const> entries,
                                                  const PreparedPromptData& prompt) {
    const ParkedSequence* best = nullptr;
    for (const ParkedSequence* entry : entries) {
        if (entry == nullptr || entry->execution_frontier == 0) { continue; }
        if (!prefix_matches(prompt, entry->ledger, entry->prefix_identity,
                            entry->execution_frontier)) {
            continue;
        }
        if (best == nullptr || entry->execution_frontier > best->execution_frontier ||
            (entry->execution_frontier == best->execution_frontier &&
             entry->last_used > best->last_used)) {
            best = entry;
        }
    }
    return best;
}

inline std::optional<ParkedCheckpointMatch> match_parked_checkpoints(
    std::span<const ParkedSequence* const> entries, const PreparedPromptData& prompt) {
    std::optional<ParkedCheckpointMatch> best;
    const auto consider = [&](const ParkedSequence& entry, std::uint32_t frontier,
                              std::uint32_t index, SharedPrefixSlot shared,
                              const std::vector<TokenId>& ledger,
                              const ResidentPrefixIdentity& identity) {
        if (frontier == 0 || frontier > entry.execution_frontier ||
            frontier > ledger.size() || frontier > prompt.token_ids.size()) {
            return;
        }
        // The parked KV (entry lineage) and the checkpoint's own lineage must both contain the
        // prompt prefix. The first keeps the copied KV pages valid; the second keeps the
        // checkpoint's recurrent state valid when the entry's history spans a branch.
        if (!prefix_matches(prompt, entry.ledger, entry.prefix_identity, frontier)) { return; }
        if (&ledger != &entry.ledger && !prefix_matches(prompt, ledger, identity, frontier)) {
            return;
        }
        if (best && (frontier < best->frontier ||
                     (frontier == best->frontier && entry.last_used <= best->entry->last_used))) {
            return;
        }
        best = ParkedCheckpointMatch{&entry, frontier, index, shared};
    };
    for (const ParkedSequence* entry : entries) {
        if (entry == nullptr) { continue; }
        // A checkpoint's own lineage decides whether it is valid for this prompt. Only legacy
        // entries without provenance fall back to the entry ledger.
        if (entry->checkpoint_valid) {
            const bool own = !entry->checkpoint_ledger.empty();
            consider(*entry, entry->checkpoint_frontier, 0, SharedPrefixSlot::None,
                     own ? entry->checkpoint_ledger : entry->ledger,
                     own ? entry->checkpoint_identity : entry->prefix_identity);
        }
        for (std::size_t i = 0; i < entry->checkpoint_history.size(); ++i) {
            const HostCheckpoint& checkpoint = entry->checkpoint_history[i];
            const bool own                   = !checkpoint.ledger.empty();
            consider(*entry, checkpoint.frontier, static_cast<std::uint32_t>(i + 1),
                     SharedPrefixSlot::None, own ? checkpoint.ledger : entry->ledger,
                     own ? checkpoint.identity : entry->prefix_identity);
        }
        // The pinned shared boundaries survive ring rotation and are offered to any conversation
        // that shares the block. The tool anchor stays valid when the instruction text changes.
        const auto consider_shared = [&](const HostCheckpoint& pinned, SharedPrefixSlot slot) {
            if (pinned.frontier == 0) { return; }
            const bool own = !pinned.ledger.empty();
            consider(*entry, pinned.frontier, 0, slot, own ? pinned.ledger : entry->ledger,
                     own ? pinned.identity : entry->prefix_identity);
        };
        consider_shared(entry->tools_prefix, SharedPrefixSlot::Tools);
        consider_shared(entry->shared_prefix, SharedPrefixSlot::Leading);
        for (const HostCheckpoint& boundary : entry->boundary_history) {
            consider_shared(boundary, SharedPrefixSlot::Message);
        }
        for (const HostCheckpoint& response : entry->response_history) {
            consider_shared(response, SharedPrefixSlot::Response);
        }
        for (const HostCheckpoint& checkpoint : entry->long_range_history) {
            consider_shared(checkpoint, SharedPrefixSlot::LongRange);
        }
    }
    return best;
}

inline std::vector<const ParkedSequence*> ParkedSequenceStore::candidates() const {
    std::vector<const ParkedSequence*> pointers;
    pointers.reserve(entries_.size());
    for (const std::unique_ptr<ParkedSequence>& entry : entries_) { pointers.push_back(entry.get()); }
    return pointers;
}

inline const ParkedSequence* ParkedSequenceStore::match(const PreparedPromptData& prompt) const {
    const std::vector<const ParkedSequence*> entries = candidates();
    return match_parked_entries(entries, prompt);
}

inline std::optional<ParkedCheckpointMatch> ParkedSequenceStore::match_checkpoint(
    const PreparedPromptData& prompt) const {
    const std::vector<const ParkedSequence*> entries = candidates();
    return match_parked_checkpoints(entries, prompt);
}

// Removes and returns the least-recently-used entry so the caller can spill it to disk and index
// it there. Returns nullptr when the store is empty. The caller owns the returned payload.
inline std::unique_ptr<ParkedSequence> ParkedSequenceStore::remove_least_recently_used() {
    if (entries_.empty()) { return nullptr; }
    std::size_t oldest = 0;
    for (std::size_t i = 1; i < entries_.size(); ++i) {
        if (entries_[i]->last_used < entries_[oldest]->last_used) { oldest = i; }
    }
    std::unique_ptr<ParkedSequence> entry = std::move(entries_[oldest]);
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(oldest));
    used_bytes_ -= entry->bytes;
    return entry;
}


inline const ParkedSequence* ParkedSequenceStore::find_by_id(std::uint64_t id) const noexcept {
    for (const std::unique_ptr<ParkedSequence>& entry : entries_) {
        if (entry->id == id) { return entry.get(); }
    }
    return nullptr;
}

inline ParkedSequence* ParkedSequenceStore::find_by_id(std::uint64_t id) noexcept {
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

inline bool ParkedSequenceStore::evict_least_recently_used(std::uint64_t protect_id) noexcept {
    std::optional<std::size_t> victim;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i]->id == protect_id) { continue; }
        if (!victim || entries_[i]->last_used < entries_[*victim]->last_used) { victim = i; }
    }
    if (!victim) { return false; }
    used_bytes_ -= entries_[*victim]->bytes;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(*victim));
    return true;
}

} // namespace ninfer::targets::qwen3_6::detail
