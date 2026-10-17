// Unit tests for the host-side park store. The store owns no device state, so this runs without a
// GPU: it exercises the byte budget, least-recently-used eviction, and the prefix-identity match
// that decides which parked conversation a request resumes.

#include "targets/qwen3_6/impl/runtime/parked_sequence_store.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

using ninfer::TokenId;
using ninfer::targets::qwen3_6::PreparedPromptData;
using ninfer::targets::qwen3_6::RewriteCheckpointKind;
using ninfer::targets::qwen3_6::detail::HostCheckpoint;
using ninfer::targets::qwen3_6::detail::ParkedSequence;
using ninfer::targets::qwen3_6::detail::ParkedSequenceStore;
using ninfer::targets::qwen3_6::detail::ResidentPrefixIdentity;

namespace {

int failures = 0;

void check(bool ok, const char* message) {
    if (!ok) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}

PreparedPromptData prompt_of(const std::vector<TokenId>& tokens) {
    PreparedPromptData prompt;
    prompt.token_ids = tokens;
    prompt.token_types.assign(tokens.size(), 0);
    prompt.positions.resize(3 * tokens.size());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            prompt.positions[axis * tokens.size() + i] = static_cast<std::int32_t>(i);
        }
    }
    return prompt;
}

HostCheckpoint checkpoint_of(const std::vector<TokenId>& tokens, std::uint32_t frontier) {
    const PreparedPromptData prompt = prompt_of(tokens);
    HostCheckpoint checkpoint;
    checkpoint.kind     = RewriteCheckpointKind::ResponseReplay;
    checkpoint.frontier = frontier;
    checkpoint.ledger   = tokens;
    checkpoint.identity.assign(prompt);
    return checkpoint;
}

std::unique_ptr<ParkedSequence> entry_of(const PreparedPromptData& prompt, std::size_t bytes) {
    auto entry                = std::make_unique<ParkedSequence>();
    entry->ledger             = prompt.token_ids;
    entry->prefix_identity    = ResidentPrefixIdentity{};
    entry->prefix_identity.assign(prompt);
    entry->execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    entry->ledger_frontier    = entry->execution_frontier;
    entry->bytes              = bytes;
    return entry;
}

} // namespace

int main() {
    // A default store is disabled: it stores nothing and matches nothing, so enabling parking is
    // always an explicit choice rather than a side effect of constructing the program.
    ParkedSequenceStore disabled;
    check(!disabled.enabled(), "a default store must be disabled");
    check(!disabled.insert(entry_of(prompt_of({1, 2, 3}), 16), 1).has_value(),
          "a disabled store must refuse an insert");

    // Two parked conversations with distinct prefixes, sized to sit inside the budget together.
    const PreparedPromptData alpha = prompt_of({10, 11, 12, 13});
    const PreparedPromptData beta  = prompt_of({20, 21});

    ParkedSequenceStore store(100);
    check(store.enabled(), "an explicit budget must enable the store");
    const auto alpha_id = store.insert(entry_of(alpha, 40), 1);
    const auto beta_id  = store.insert(entry_of(beta, 40), 2);
    check(alpha_id.has_value() && beta_id.has_value(), "both entries must fit the budget");
    check(store.used_bytes() == 80, "used bytes must count every stored entry");
    check(store.entry_count() == 2, "both entries must be stored");

    // Matching is by token-prefix identity, and a longer resident frontier wins: a prompt that
    // extends alpha matches alpha, not beta.
    const PreparedPromptData alpha_extended = prompt_of({10, 11, 12, 13, 14, 15});
    const ParkedSequence* hit              = store.match(alpha_extended);
    check(hit != nullptr && hit->id == *alpha_id, "an extended prefix must match its own entry");

    // A prompt that is not a prefix of anything parked matches nothing, rather than the nearest
    // entry: parking must never resume a conversation the caller did not ask for.
    check(store.match(prompt_of({30, 31})) == nullptr, "an unrelated prompt must not match");

    // Identity is part of the match, not just the token ids: the same ids under a different
    // position regime are a different conversation.
    PreparedPromptData shifted = alpha_extended;
    for (std::size_t i = 0; i < shifted.token_ids.size(); ++i) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            shifted.positions[axis * shifted.token_ids.size() + i] += 7;
        }
    }
    check(store.match(shifted) == nullptr, "changed positions must not reuse a parked entry");

    // Longest frontier wins when two entries both prefix-match.
    ParkedSequenceStore nested(100);
    const PreparedPromptData short_prefix = prompt_of({40, 41});
    const PreparedPromptData long_prefix  = prompt_of({40, 41, 42, 43});
    (void)nested.insert(entry_of(short_prefix, 10), 1);
    const auto long_id = nested.insert(entry_of(long_prefix, 10), 2);
    const ParkedSequence* best = nested.match(prompt_of({40, 41, 42, 43, 44}));
    check(best != nullptr && best->id == *long_id, "the longest matching frontier must win");

    // An entry larger than the whole budget is refused outright, leaving the store untouched.
    ParkedSequenceStore tight(50);
    check(!tight.insert(entry_of(alpha, 51), 1).has_value(),
          "an entry larger than the budget must be refused");
    check(tight.used_bytes() == 0 && tight.entry_count() == 0,
          "a refused insert must not change the store");

    // Growing past the budget evicts least-recently-used entries, and `touch` protects one.
    ParkedSequenceStore lru(100);
    const auto first  = lru.insert(entry_of(prompt_of({50, 51}), 40), 1);
    const auto second = lru.insert(entry_of(prompt_of({52, 53}), 40), 2);
    check(first.has_value() && second.has_value(), "the first two entries must fit");
    lru.touch(*first, 3);
    const auto third = lru.insert(entry_of(prompt_of({54, 55}), 40), 4);
    check(third.has_value(), "a third entry must be admitted by eviction");
    check(lru.entry_count() == 2, "eviction must keep the store inside its budget");
    check(lru.used_bytes() == 80, "eviction must release the evicted entry's bytes");
    check(lru.find_by_id(*first) != nullptr, "a touched entry must survive eviction");
    check(lru.find_by_id(*second) == nullptr, "the least recently used entry must be evicted");

    // The rewrite checkpoint is a second match frontier. A replay that diverges inside the last
    // response matches no execution frontier, but it still contains the checkpoint prefix, so the
    // checkpoint matcher finds the entry that the exact matcher must not.
    const PreparedPromptData gamma = prompt_of({60, 61, 62, 63, 64, 65});
    auto checkpoint_entry          = entry_of(gamma, 10);
    checkpoint_entry->checkpoint_valid    = true;
    checkpoint_entry->checkpoint_kind     = RewriteCheckpointKind::ResponseReplay;
    checkpoint_entry->checkpoint_frontier = 4;
    ParkedSequenceStore checkpointed(100);
    (void)checkpointed.insert(std::move(checkpoint_entry), 1);
    const PreparedPromptData diverged = prompt_of({60, 61, 62, 63, 90, 91, 92});
    check(checkpointed.match(diverged) == nullptr,
          "a replay that diverged inside the last response must not match the full frontier");
    const auto checkpoint_hit = checkpointed.match_checkpoint(diverged);
    check(checkpoint_hit.has_value() && checkpoint_hit->frontier == 4 &&
              checkpoint_hit->checkpoint_index == 0,
          "the current checkpoint frontier must match a diverged replay");
    // A prompt that diverges before the checkpoint is still unusable.
    check(!checkpointed.match_checkpoint(prompt_of({60, 61, 99, 63, 64, 65})).has_value(),
          "a replay that diverged before the checkpoint must not match");

    // A historical boundary is a second-chance resume point for a stale or forked replay. It is
    // reported with its index so the caller can import that boundary's state.
    const PreparedPromptData delta = prompt_of({70, 71, 72, 73, 74, 75, 76});
    auto history_entry             = entry_of(delta, 10);
    HostCheckpoint historical;
    historical.kind     = RewriteCheckpointKind::ResponseReplay;
    historical.frontier = 3;
    history_entry->checkpoint_history.push_back(std::move(historical));
    ParkedSequenceStore historical_store(100);
    (void)historical_store.insert(std::move(history_entry), 1);
    const auto historical_hit =
        historical_store.match_checkpoint(prompt_of({70, 71, 72, 99, 99, 99}));
    check(historical_hit.has_value() && historical_hit->frontier == 3 &&
              historical_hit->checkpoint_index == 1,
          "a historical checkpoint boundary must match a stale replay");
    // Provenance: a retained checkpoint belongs to the branch it was captured from. Two branches
    // of one conversation share a token prefix only up to their fork, so a checkpoint from the
    // other branch must never be selected merely because the current lane ledger matches at its
    // frontier -- its recurrent state is from tokens this replay never had.
    auto forked_entry = entry_of(prompt_of({1, 2, 3, 4, 5, 6}), 10);
    forked_entry->checkpoint_history.push_back(checkpoint_of({1, 2, 9, 9, 9, 9}, 5));
    ParkedSequenceStore forked_store(100);
    (void)forked_store.insert(std::move(forked_entry), 1);
    check(!forked_store.match_checkpoint(prompt_of({1, 2, 3, 4, 5, 6, 7})).has_value(),
          "a checkpoint from another branch must not be selected by the entry's tokens");
    // Nor may its state be paired with this entry's KV just because the prompt matches the
    // checkpoint's own lineage: both the KV and the state must describe the same prefix.
    check(!forked_store.match_checkpoint(prompt_of({1, 2, 9, 9, 9, 9, 0})).has_value(),
          "a checkpoint must not be paired with KV from a different branch");
    // When the entry itself belongs to that lineage, the boundary is selectable.
    auto matching_entry = entry_of(prompt_of({1, 2, 9, 9, 9, 9}), 10);
    matching_entry->checkpoint_history.push_back(checkpoint_of({1, 2, 9, 9, 9, 9}, 5));
    ParkedSequenceStore matching_store(100);
    (void)matching_store.insert(std::move(matching_entry), 1);
    const auto provenance_hit =
        matching_store.match_checkpoint(prompt_of({1, 2, 9, 9, 9, 9, 0}));
    check(provenance_hit.has_value() && provenance_hit->frontier == 5 &&
              provenance_hit->checkpoint_index == 1,
          "a replay matching both the entry and the checkpoint lineage must select it");

    // The newest matching boundary wins even when several are retained.
    auto two_history_entry = entry_of(delta, 10);
    HostCheckpoint older;
    older.kind     = RewriteCheckpointKind::ResponseReplay;
    older.frontier = 2;
    HostCheckpoint newer = older;
    newer.frontier       = 5;
    two_history_entry->checkpoint_history.push_back(std::move(older));
    two_history_entry->checkpoint_history.push_back(std::move(newer));
    ParkedSequenceStore two_history_store(100);
    (void)two_history_store.insert(std::move(two_history_entry), 1);
    const auto newest_match = two_history_store.match_checkpoint(delta);
    check(newest_match.has_value() && newest_match->frontier == 5,
          "the largest matching retained boundary must win");

    // A protected insert may evict other entries, but never the entry it protects.
    ParkedSequenceStore guarded(100);
    const auto guarded_first  = guarded.insert(entry_of(prompt_of({70, 71}), 40), 1);
    const auto guarded_second = guarded.insert(entry_of(prompt_of({72, 73}), 40), 2);
    const auto guarded_third =
        guarded.insert(entry_of(prompt_of({74, 75}), 40), 3, *guarded_first);
    check(guarded_third.has_value(), "a protected insert must still be admitted");
    check(guarded.find_by_id(*guarded_first) != nullptr,
          "a protected entry must survive the insert");
    check(guarded.find_by_id(*guarded_second) == nullptr,
          "the unprotected least-recently-used entry must be evicted");

    // When the new entry cannot fit beside the protected one, the insert is refused before any
    // eviction, leaving every stored entry intact.
    ParkedSequenceStore protected_full(100);
    const auto protected_big   = protected_full.insert(entry_of(prompt_of({80, 81}), 80), 1);
    const auto protected_small = protected_full.insert(entry_of(prompt_of({82, 83}), 20), 2);
    check(!protected_full.insert(entry_of(prompt_of({84, 85}), 30), 3, *protected_big)
               .has_value(),
          "an insert that cannot fit beside the protected entry must be refused");
    check(protected_full.used_bytes() == 100 && protected_full.entry_count() == 2,
          "a refused protected insert must not evict anything");
    check(protected_full.find_by_id(*protected_small) != nullptr,
          "a refused protected insert must leave the other entries intact");

    // Erasing releases the bytes it accounted for.
    lru.erase(*first);
    check(lru.used_bytes() == 40 && lru.entry_count() == 1, "erase must release its entry");
    lru.clear();
    check(lru.used_bytes() == 0 && lru.entry_count() == 0, "clear must empty the store");

    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
