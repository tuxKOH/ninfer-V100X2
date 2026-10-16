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

    // Erasing releases the bytes it accounted for.
    lru.erase(*first);
    check(lru.used_bytes() == 40 && lru.entry_count() == 1, "erase must release its entry");
    lru.clear();
    check(lru.used_bytes() == 0 && lru.entry_count() == 0, "clear must empty the store");

    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
