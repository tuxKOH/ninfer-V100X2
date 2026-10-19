#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

inline bool reuse_trace_enabled() {
    static const bool enabled = std::getenv("NINFER_REUSE_TRACE") != nullptr;
    return enabled;
}

template <typename... Args>
void reuse_trace(const char* format, Args... args) {
    if (!reuse_trace_enabled()) { return; }
    std::fprintf(stderr, "[reuse] ");
    std::fprintf(stderr, format, args...);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * kPreparedVisionPatchFeatures) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash && backend_kv_cache() != nullptr) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    if (ram_kv.gpu_tokens != 0) {
        base->text_kv_page_entitlement = decoder->text_kv.pool().page_group_count();
        base->backend_kv_page_entitlement = decoder->mtp_cache() != nullptr
            ? std::min(decoder->mtp_cache()->pool().page_group_count(), pages_for_tokens(capacity)) : 0U;
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_6::VisionControl>(qwen3_6::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_6::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            // The CPU encoder stages its intermediates in host buffers; only the merged output
            // uses the device transient, which the plan accounts for separately. The swap mode
            // allocates the scratch on demand, so the startup workspace is not the envelope.
            if (!vision_cpu && !vision_kv_swap &&
                schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.rewrite_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.rewrite_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->rewrite_checkpoint = candidate;
    }
    if (prompt.identity.stable_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.stable_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "stable checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->stable_checkpoint = candidate;
    }
    for (const RewriteCheckpointSpec& candidate : prompt.identity.message_checkpoints) {
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "message checkpoint frontier must lie at or inside the prompt frontier");
        }
        if (!base->message_checkpoints.empty() &&
            candidate.frontier <= base->message_checkpoints.back().frontier) {
            throw std::invalid_argument("message checkpoints must be strictly ascending");
        }
        if (base->stable_checkpoint &&
            candidate.frontier >= base->stable_checkpoint->frontier) {
            throw std::invalid_argument("message checkpoint must precede the stable boundary");
        }
        base->message_checkpoints.push_back(candidate);
    }
    if (prompt.identity.prefix_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.prefix_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "shared prefix checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->prefix_checkpoint = candidate;
    }
    if (prompt.identity.tools_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.tools_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "tool checkpoint frontier must lie at or inside the prompt frontier");
        }
        if (base->prefix_checkpoint && candidate.frontier >= base->prefix_checkpoint->frontier) {
            throw std::invalid_argument("tool checkpoint must precede the shared prefix");
        }
        base->tools_checkpoint = candidate;
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->rewrite_checkpoint &&
                 base->rewrite_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL) +
        (base->stable_checkpoint &&
                 base->stable_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL) +
        static_cast<std::size_t>(std::count_if(
            base->message_checkpoints.begin(), base->message_checkpoints.end(),
            [&](const RewriteCheckpointSpec& checkpoint) {
                return checkpoint.frontier < base->summary.prompt_tokens;
            })) +
        (speculative_backend != SpeculativeBackend::DFlash && base->prefix_checkpoint &&
                 base->prefix_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL) +
        (speculative_backend != SpeculativeBackend::DFlash && base->tools_checkpoint &&
                 base->tools_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;

    if (reuse_trace_enabled()) {
        reuse_trace(
            "plan lane=%u prompt=%zu retained=%d allow=%d reusable=%d exec=%u ledger_frontier=%u "
            "ledger=%zu text_kv=%u mtp_kv=%u tail_hidden=%d ckpt=%d kind=%d ckpt_frontier=%u "
            "history=%zu",
            lane, prompt.token_ids.size(), sequence.retained ? 1 : 0,
            base.allow_prefix_reuse ? 1 : 0, prompt.identity.reusable ? 1 : 0,
            sequence.execution_frontier, sequence.ledger_frontier, sequence.ledger.size(),
            sequence.text_kv_valid, sequence.mtp_kv_valid, sequence.tail_hidden_valid ? 1 : 0,
            sequence.rewrite_checkpoint.valid ? 1 : 0,
            static_cast<int>(sequence.rewrite_checkpoint.kind),
            sequence.rewrite_checkpoint.frontier, sequence.checkpoint_history.size());
        for (std::size_t h = 0; h < sequence.checkpoint_history.size(); ++h) {
            const HostCheckpoint& host = sequence.checkpoint_history[h];
            reuse_trace("  hist[%zu] kind=%d frontier=%u ledger=%zu own=%d", h,
                        static_cast<int>(host.kind), host.frontier, host.ledger.size(),
                        host.ledger.empty() ? 0 : 1);
        }
    }

    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained) {
        const bool dflash_append_ready =
            speculative_backend != SpeculativeBackend::DFlash ||
            sequence.dflash_context_frontier == sequence.execution_frontier;
        const bool append_matches =
            sequence.execution_frontier != 0 && dflash_append_ready &&
            qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                            sequence.execution_frontier);
        reuse_trace("append_check frontier=%u dflash_ready=%d match=%d",
                    sequence.execution_frontier, dflash_append_ready ? 1 : 0,
                    append_matches ? 1 : 0);
        if (append_matches) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = sequence.execution_frontier;
        } else {
            // The replay diverged from the lane's frontier. Resume from the largest retained
            // boundary that is still a prefix of it: the current checkpoint first, then the
            // history. A stale or forked replay of the same conversation usually matches an
            // older turn boundary, which turns its full prefill into a suffix prefill. After a
            // branch resume the history is not sorted, so every entry is considered.
            std::uint32_t best_frontier = 0;
            std::uint32_t best_index    = 0;  // 0 = no checkpoint selected
            RewriteCheckpointKind best_kind = RewriteCheckpointKind::TurnClosure;
            SharedPrefixSlot best_shared = SharedPrefixSlot::None;
            // Two independent checks are required. (1) The lane's own ledger/KV must contain
            // this prompt prefix -- otherwise the resident KV pages the resume would read were
            // produced for a different branch. (2) The checkpoint's OWN lineage must contain it
            // too -- otherwise its host recurrent state was produced for a different branch
            // (branches share a prefix only up to their fork). The checkpoint frontier must also
            // still be materialized: a branch switch trims the KV to an older boundary.
            const auto consider_checkpoint = [&](RewriteCheckpointKind kind, std::uint32_t frontier,
                                                 std::uint32_t index, SharedPrefixSlot shared,
                                                 const std::vector<TokenId>& ledger,
                                                 const qwen3_6::detail::ResidentPrefixIdentity&
                                                     identity) {
                const bool in_range =
                    frontier != 0 && frontier <= prompt.token_ids.size() &&
                    frontier > best_frontier && frontier <= ledger.size() &&
                    frontier <= sequence.text_kv_valid;
                bool lane_match = false;
                bool own_match  = false;
                if (in_range) {
                    lane_match = qwen3_6::detail::prefix_matches(
                        prompt, sequence.ledger, sequence.prefix_identity, frontier);
                    own_match =
                        qwen3_6::detail::prefix_matches(prompt, ledger, identity, frontier);
                }
                reuse_trace(
                    "  cand index=%u kind=%d frontier=%u in_range=%d lane_match=%d own_match=%d",
                    index, static_cast<int>(kind), frontier, in_range ? 1 : 0,
                    lane_match ? 1 : 0, own_match ? 1 : 0);
                if (!in_range || !lane_match || !own_match) { return; }
                best_frontier = frontier;
                best_index    = index;
                best_kind     = kind;
                best_shared   = shared;
            };
            if (sequence.rewrite_checkpoint.valid) {
                const bool own = !sequence.checkpoint_ledger.empty();
                consider_checkpoint(
                    sequence.rewrite_checkpoint.kind, sequence.rewrite_checkpoint.frontier, 0,
                    SharedPrefixSlot::None,
                    own ? sequence.checkpoint_ledger : sequence.ledger,
                    own ? sequence.checkpoint_identity : sequence.prefix_identity);
            }
            // The pinned shared boundaries are offered to any conversation whose prompt starts
            // with the same static block, not only to the conversation that captured them. The
            // tool anchor survives a changed instruction text; the full leading block is the
            // longer prefix when both match. DFlash keeps its draft context on the device and the
            // host checkpoint does not mirror it, so it is excluded exactly like the ring entries.
            const auto consider_shared = [&](const HostCheckpoint& pinned, SharedPrefixSlot slot) {
                if (pinned.frontier == 0) { return; }
                const bool own = !pinned.ledger.empty();
                consider_checkpoint(pinned.kind, pinned.frontier, 0, slot,
                                    own ? pinned.ledger : sequence.ledger,
                                    own ? pinned.identity : sequence.prefix_identity);
            };
            if (speculative_backend != SpeculativeBackend::DFlash) {
                consider_shared(sequence.tools_prefix, SharedPrefixSlot::Tools);
                consider_shared(sequence.shared_prefix, SharedPrefixSlot::Leading);
                // The boundary the lane last resumed from: the fork point another branch returns
                // to, kept outside the depth-limited histories.
                consider_shared(sequence.resume_anchor, SharedPrefixSlot::Anchor);
                // Recent message boundaries, pinned outside the ring: a client that rewrote an
                // assistant message a few turns back resumes at the last boundary before it.
                for (const HostCheckpoint& boundary : sequence.boundary_history) {
                    consider_shared(boundary, SharedPrefixSlot::Message);
                }
                for (const HostCheckpoint& response : sequence.response_history) {
                    consider_shared(response, SharedPrefixSlot::Response);
                }
                for (const HostCheckpoint& checkpoint : sequence.long_range_history) {
                    consider_shared(checkpoint, SharedPrefixSlot::LongRange);
                }
            }
            // DFlash keeps its checkpoint state in its own context, so the host history (which
            // mirrors only the recurrent slot and hidden) cannot restore it.
            if (speculative_backend != SpeculativeBackend::DFlash) {
                for (std::size_t i = 0; i < sequence.checkpoint_history.size(); ++i) {
                    const HostCheckpoint& host = sequence.checkpoint_history[i];
                    const bool own             = !host.ledger.empty();
                    consider_checkpoint(host.kind, host.frontier,
                                        static_cast<std::uint32_t>(i + 1),
                                        SharedPrefixSlot::None,
                                        own ? host.ledger : sequence.ledger,
                                        own ? host.identity : sequence.prefix_identity);
                }
            }
            if (best_index != 0 || best_frontier != 0) {
                plan->reuse                 = restore_path(best_kind);
                plan->reuse_base            = best_frontier;
                plan->checkpoint_index      = best_index;
                plan->restore_shared_prefix = best_shared;
            }
        }
    }

    // A conversation whose KV was parked by an earlier eviction resumes from the host store. It
    // is matched by the same token-prefix identity the resident lane above is matched by, so
    // parking widens how many conversations can be resumed without widening what "the same
    // conversation" means. The store is consulted even when the resident lane matched: a
    // different resident conversation may match only the shared leading block while a parked
    // entry of this conversation holds its whole history, and the longer resume wins. A tie keeps
    // the resident plan, which needs no host-to-device restore.
    const qwen3_6::detail::ParkedSequence* parked_entry = nullptr;
    if (base.allow_prefix_reuse && prompt.identity.reusable &&
        (parked.enabled() || !disk_entries.empty()) &&
        speculative_backend != SpeculativeBackend::DFlash) {
        const std::vector<const qwen3_6::detail::ParkedSequence*> candidates = parked_candidates();
        const qwen3_6::detail::ParkedSequence* exact_entry =
            qwen3_6::detail::match_parked_entries(candidates, prompt);
        // A replay that diverged inside the parked conversation's last response matches no
        // execution frontier, but it still contains a retained checkpoint's prefix. Both
        // frontiers are valid reuse points, so take the longer one: a checkpoint resume only
        // costs the suffix re-prefill a resident lane would pay anyway. An exact resume needs the
        // parked tail hidden under MTP, and a checkpoint resume does not, so an entry whose tail
        // hidden was invalidated (a cancellation during a speculative round) is still useful.
        const std::optional<qwen3_6::detail::ParkedCheckpointMatch> checkpoint_match =
            qwen3_6::detail::match_parked_checkpoints(candidates, prompt);
        // A restore materializes the image's pages into the request's own allocation, whose
        // entitlement is sized from this prompt and output budget. An image produced by an older
        // layout can exceed it; taking that entry would throw from attach_from_host inside the
        // worker's execution unit and fail the whole engine, so decline it here and let the
        // request prefill from its own prefix instead.
        // A disk-backed entry keeps only page COUNTS in RAM; a materialized one has the images.
        const auto parked_images_fit = [&](const qwen3_6::detail::ParkedSequence& entry) {
            // A disk-backed entry is only usable while its file exists; a deleted or manually
            // removed image must fall back to a normal prefill instead of failing at restore.
            if (entry.disk_backed) {
                std::error_code error;
                if (!std::filesystem::exists(entry.disk_path, error) || error) { return false; }
            }
            const auto pages_at = [](const auto& images, const std::vector<std::uint32_t>& disk,
                                     std::size_t rank) -> std::uint32_t {
                if (rank < images.size() && images[rank] != nullptr) { return images[rank]->pages(); }
                return rank < disk.size() ? disk[rank] : 0U;
            };
            const std::size_t ranks =
                std::max(entry.text.size(), entry.text_pages.size());
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                if (pages_at(entry.text, entry.text_pages, rank) >
                    plan->text_kv_page_entitlement) {
                    return false;
                }
            }
            const std::size_t backend_ranks =
                std::max(entry.backend.size(), entry.backend_pages.size());
            for (std::size_t rank = 0; rank < backend_ranks; ++rank) {
                if (pages_at(entry.backend, entry.backend_pages, rank) >
                    plan->backend_kv_page_entitlement) {
                    return false;
                }
            }
            return true;
        };
        const bool exact_usable = exact_entry != nullptr &&
                                  (speculative_backend != SpeculativeBackend::Mtp ||
                                   exact_entry->tail_hidden_valid) &&
                                  parked_images_fit(*exact_entry);
        const bool checkpoint_usable =
            checkpoint_match.has_value() &&
            (speculative_backend != SpeculativeBackend::Mtp ||
             (decoder->mtp_cache() != nullptr &&
              checkpoint_match->entry->mtp_kv_valid >= checkpoint_match->frontier - 1)) &&
            parked_images_fit(*checkpoint_match->entry);
        // The longer parked candidate: a checkpoint resume costs only the suffix prefill an
        // exact resume pays anyway, so an entry whose tail hidden was invalidated (a
        // cancellation during a speculative round) is still useful at its checkpoint frontier.
        const qwen3_6::detail::ParkedSequence* candidate = nullptr;
        std::uint32_t candidate_frontier                 = 0;
        bool candidate_checkpoint                        = false;
        if (checkpoint_usable &&
            (!exact_usable || checkpoint_match->frontier > exact_entry->execution_frontier)) {
            candidate          = checkpoint_match->entry;
            candidate_frontier = checkpoint_match->frontier;
            candidate_checkpoint = true;
        } else if (exact_usable) {
            candidate          = exact_entry;
            candidate_frontier = exact_entry->execution_frontier;
        }
        // Only take the parked state when it beats what the resident lane already offers by
        // enough to pay for the host-to-device restore and the incumbent's parking. A parked
        // entry can match at the shared leading block (or a warmup boundary) while sharing
        // almost nothing else; restoring it then costs a multi-gigabyte copy to save a few
        // hundred tokens of prefill.
        constexpr std::uint32_t kMinimumParkedReuseTokens = 2048;
        constexpr std::uint32_t kMinimumParkedReuseGain   = 1024;
        const std::uint32_t resident_base = plan->reuse_base;
        const bool worth_restoring =
            candidate != nullptr && candidate_frontier > resident_base &&
            candidate_frontier >= kMinimumParkedReuseTokens &&
            candidate_frontier >= resident_base + kMinimumParkedReuseGain;
        if (!worth_restoring && candidate != nullptr) {
            reuse_trace("parked declined frontier=%u resident=%u", candidate_frontier,
                        resident_base);
        }
        if (worth_restoring) {
            // Mark the target most-recently-used at plan time: an admission-time eviction of
            // another lane (or disk-budget enforcement) between planning and start_prefill must
            // not take the entry this plan already names.
            if (is_disk_entry_id(candidate->id)) {
                for (std::unique_ptr<ParkedSequence>& entry : disk_entries) {
                    if (entry->id == candidate->id) {
                        entry->last_used = park_clock++;
                        break;
                    }
                }
            } else {
                parked.touch(candidate->id, park_clock++);
            }
            parked_entry                = candidate;
            plan->reuse                 = ReusePath::RestoreParkedKv;
            plan->reuse_base            = candidate_frontier;
            plan->parked_id             = candidate->id;
            plan->parked_checkpoint     = candidate_checkpoint;
            plan->checkpoint_index      = candidate_checkpoint ? checkpoint_match->checkpoint_index : 0;
            plan->restore_shared_prefix = candidate_checkpoint ? checkpoint_match->shared
                                                               : SharedPrefixSlot::None;
        }
        reuse_trace("parked enabled=1 exact_entry=%d match_frontier=%u usable=%d resident=%u "
                    "result=%d base=%u",
                    exact_entry != nullptr ? 1 : 0,
                    checkpoint_match.has_value() ? checkpoint_match->frontier : 0U,
                    checkpoint_usable ? 1 : 0, resident_base,
                    static_cast<int>(plan->reuse), plan->reuse_base);
    } else if (reuse_trace_enabled() && plan->reuse == ReusePath::FullReset) {
        reuse_trace("parked skipped enabled=%d allow=%d reusable=%d", parked.enabled() ? 1 : 0,
                    base.allow_prefix_reuse ? 1 : 0, prompt.identity.reusable ? 1 : 0);
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        // A checkpoint resume restores its own hidden, so it does not need the parked tail
        // hidden; an exact resume does.
        const bool parked_ready =
            parked_entry != nullptr &&
            (parked_entry->tail_hidden_valid || plan->parked_checkpoint) &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || parked_entry->mtp_kv_valid >= plan->reuse_base - 1);
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && sequence.tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || sequence.mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = is_rewrite_checkpoint_restore(plan->reuse) &&
                                      decoder->mtp_cache() != nullptr && plan->reuse_base != 0 &&
                                      sequence.mtp_kv_valid >= plan->reuse_base - 1;
        bool checkpoint_fallback = false;
        if (plan->reuse == ReusePath::AppendAtFrontier && !append_ready &&
            decoder->mtp_cache() != nullptr) {
            // The append frontier is unusable only because the tail hidden is not authoritative
            // (a cancellation during a speculative round). The lane's rewrite checkpoints are
            // still intact, and resuming from the largest matching one is exactly what a
            // resident lane would do.
            std::uint32_t best_frontier = 0;
            std::uint32_t best_index    = 0;
            RewriteCheckpointKind best_kind = RewriteCheckpointKind::TurnClosure;
            SharedPrefixSlot best_shared = SharedPrefixSlot::None;
            const auto consider_checkpoint = [&](RewriteCheckpointKind kind, std::uint32_t frontier,
                                                 std::uint32_t index, SharedPrefixSlot shared,
                                                 const std::vector<TokenId>& ledger,
                                                 const qwen3_6::detail::ResidentPrefixIdentity&
                                                     identity) {
                const bool in_range =
                    frontier != 0 && frontier <= prompt.token_ids.size() &&
                    frontier > best_frontier && frontier <= ledger.size() &&
                    frontier <= sequence.text_kv_valid;
                const bool mtp_ok = in_range && sequence.mtp_kv_valid >= frontier - 1;
                bool lane_match   = false;
                bool own_match    = false;
                if (in_range) {
                    lane_match = qwen3_6::detail::prefix_matches(
                        prompt, sequence.ledger, sequence.prefix_identity, frontier);
                    own_match =
                        qwen3_6::detail::prefix_matches(prompt, ledger, identity, frontier);
                }
                reuse_trace(
                    "  fallback_cand index=%u kind=%d frontier=%u in_range=%d mtp_ok=%d "
                    "lane_match=%d own_match=%d",
                    index, static_cast<int>(kind), frontier, in_range ? 1 : 0, mtp_ok ? 1 : 0,
                    lane_match ? 1 : 0, own_match ? 1 : 0);
                if (!mtp_ok || !lane_match || !own_match) { return; }
                best_frontier = frontier;
                best_index    = index;
                best_kind     = kind;
                best_shared   = shared;
            };
            if (sequence.rewrite_checkpoint.valid) {
                const bool own = !sequence.checkpoint_ledger.empty();
                consider_checkpoint(
                    sequence.rewrite_checkpoint.kind, sequence.rewrite_checkpoint.frontier, 0,
                    SharedPrefixSlot::None,
                    own ? sequence.checkpoint_ledger : sequence.ledger,
                    own ? sequence.checkpoint_identity : sequence.prefix_identity);
            }
            const auto consider_shared = [&](const HostCheckpoint& pinned, SharedPrefixSlot slot) {
                if (pinned.frontier == 0) { return; }
                const bool own = !pinned.ledger.empty();
                consider_checkpoint(pinned.kind, pinned.frontier, 0, slot,
                                    own ? pinned.ledger : sequence.ledger,
                                    own ? pinned.identity : sequence.prefix_identity);
            };
            if (speculative_backend != SpeculativeBackend::DFlash) {
                consider_shared(sequence.tools_prefix, SharedPrefixSlot::Tools);
                consider_shared(sequence.shared_prefix, SharedPrefixSlot::Leading);
                // The boundary the lane last resumed from: the fork point another branch returns
                // to, kept outside the depth-limited histories.
                consider_shared(sequence.resume_anchor, SharedPrefixSlot::Anchor);
                // Recent message boundaries, pinned outside the ring: a client that rewrote an
                // assistant message a few turns back resumes at the last boundary before it.
                for (const HostCheckpoint& boundary : sequence.boundary_history) {
                    consider_shared(boundary, SharedPrefixSlot::Message);
                }
                for (const HostCheckpoint& response : sequence.response_history) {
                    consider_shared(response, SharedPrefixSlot::Response);
                }
                for (const HostCheckpoint& checkpoint : sequence.long_range_history) {
                    consider_shared(checkpoint, SharedPrefixSlot::LongRange);
                }
            }
            for (std::size_t i = 0; i < sequence.checkpoint_history.size(); ++i) {
                const HostCheckpoint& host = sequence.checkpoint_history[i];
                const bool own             = !host.ledger.empty();
                consider_checkpoint(host.kind, host.frontier, static_cast<std::uint32_t>(i + 1),
                                    SharedPrefixSlot::None,
                                    own ? host.ledger : sequence.ledger,
                                    own ? host.identity : sequence.prefix_identity);
            }
            if (best_frontier != 0) {
                plan->reuse                 = restore_path(best_kind);
                plan->reuse_base            = best_frontier;
                plan->checkpoint_index      = best_index;
                plan->parked_id             = 0;
                plan->parked_checkpoint     = false;
                plan->restore_shared_prefix = best_shared;
                parked_entry                = nullptr;
                checkpoint_fallback         = true;
            }
        }
        reuse_trace(
            "mtp_gates reuse=%d base=%u append_ready=%d checkpoint_ready=%d parked_ready=%d "
            "fallback=%d",
            static_cast<int>(plan->reuse), plan->reuse_base, append_ready ? 1 : 0,
            checkpoint_ready ? 1 : 0, parked_ready ? 1 : 0, checkpoint_fallback ? 1 : 0);
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready &&
            !parked_ready && !checkpoint_fallback) {
            plan->reuse             = ReusePath::FullReset;
            plan->reuse_base        = 0;
            plan->parked_id         = 0;
            plan->parked_checkpoint = false;
            plan->checkpoint_index  = 0;
            parked_entry            = nullptr;
            reuse_trace("mtp_gates demote=full_reset");
        }
    }
    reuse_trace("chosen reuse=%d base=%u index=%u parked=%llu action=%d",
                static_cast<int>(plan->reuse), plan->reuse_base, plan->checkpoint_index,
                static_cast<unsigned long long>(plan->parked_id),
                static_cast<int>(plan->rewrite_checkpoint_action));

    if (is_rewrite_checkpoint_restore(plan->reuse) &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !sequence.kv ||
         dflash->full.has_value() != sequence.kv->backend.has_value() ||
         sequence.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse            = ReusePath::FullReset;
        plan->reuse_base       = 0;
        plan->checkpoint_index = 0;
    }

    const std::optional<RewriteCheckpointSpec>& desired = base.rewrite_checkpoint;
    // A parked plan names the lane's *replaced* occupant only until the restore runs, so the
    // occupant's checkpoint says nothing about the conversation being resumed: never let it
    // authorize a retention the restored state cannot back. A historical plan likewise replaces
    // the lane's current checkpoint with an older boundary before execution, so an action
    // validated against the pre-restore checkpoint would fail (or retain the wrong state).
    const bool existing_checkpoint_matches =
        desired && plan->reuse != ReusePath::FullReset &&
        plan->reuse != ReusePath::RestoreParkedKv && plan->checkpoint_index == 0 &&
        sequence.rewrite_checkpoint.valid &&
        sequence.rewrite_checkpoint.frontier == desired->frontier &&
        qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        desired->frontier);
    if (!desired) {
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::Drop;
    } else if (existing_checkpoint_matches) {
        plan->rewrite_checkpoint_action = sequence.rewrite_checkpoint.kind == desired->kind
                                              ? RewriteCheckpointAction::KeepExisting
                                              : RewriteCheckpointAction::ReclassifyExisting;
    } else if (desired->frontier > plan->reuse_base) {
        plan->rewrite_checkpoint_action  = RewriteCheckpointAction::CaptureNew;
        plan->rewrite_checkpoint_capture = desired;
    } else {
        // The selected continuation state is already past the desired boundary. It remains a
        // valid hit; do not replay an otherwise reusable prefix merely to materialize an older
        // auxiliary snapshot. A later request can still use the checkpoint currently retained.
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
    }

    // The stable message boundary is captured in addition to (and before) the response
    // checkpoint whenever it lies in the prefix this request is about to prefill. A boundary the
    // selected continuation already covers needs no capture; its state is not snapshotted.
    if (base.stable_checkpoint && base.stable_checkpoint->frontier > plan->reuse_base &&
        base.stable_checkpoint->frontier < plan->summary.prompt_tokens &&
        (!desired || desired->frontier != base.stable_checkpoint->frontier)) {
        plan->stable_checkpoint_capture = base.stable_checkpoint;
    }
    // Every recent boundary this request prefills is captured and pinned, so a later request
    // whose client rewrote an assistant message (dropping an interrupted tool_use) can resume at
    // the last boundary before the rewrite instead of re-prefilling the conversation.
    for (const RewriteCheckpointSpec& boundary : base.message_checkpoints) {
        if (boundary.frontier > plan->reuse_base &&
            boundary.frontier < plan->summary.prompt_tokens &&
            (!desired || desired->frontier != boundary.frontier) &&
            (!base.stable_checkpoint || boundary.frontier < base.stable_checkpoint->frontier)) {
            plan->message_checkpoint_captures.push_back(boundary);
        }
    }
    // The tool anchor is captured before the shared leading boundary. Its state is pinned for the
    // lane (and any parked entry) exactly like the leading boundary, and it stays reusable when
    // the instruction text after the tool definitions changes.
    if (speculative_backend != SpeculativeBackend::DFlash && base.tools_checkpoint &&
        base.tools_checkpoint->frontier > plan->reuse_base &&
        base.tools_checkpoint->frontier < plan->summary.prompt_tokens &&
        (!desired || desired->frontier != base.tools_checkpoint->frontier) &&
        (!base.prefix_checkpoint ||
         base.tools_checkpoint->frontier < base.prefix_checkpoint->frontier) &&
        (!base.stable_checkpoint ||
         base.tools_checkpoint->frontier < base.stable_checkpoint->frontier)) {
        plan->tools_checkpoint_capture = base.tools_checkpoint;
    }
    // The shared leading boundary is captured next when this request prefills it and pinned for
    // the lane (and any parked entry), where any other conversation with the same static leading
    // block can match it. DFlash has no host-mirrored draft context, so it never participates.
    if (speculative_backend != SpeculativeBackend::DFlash && base.prefix_checkpoint &&
        base.prefix_checkpoint->frontier > plan->reuse_base &&
        base.prefix_checkpoint->frontier < plan->summary.prompt_tokens &&
        (!desired || desired->frontier != base.prefix_checkpoint->frontier) &&
        (!base.stable_checkpoint ||
         base.prefix_checkpoint->frontier < base.stable_checkpoint->frontier)) {
        plan->prefix_checkpoint_capture = base.prefix_checkpoint;
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (is_rewrite_checkpoint_restore(plan->reuse)) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (plan->reuse == ReusePath::RestoreParkedKv) {
            // A restored lane resumes exactly as an appended one does, so it takes the same MTP
            // bridge: a suffix to prefill, or an exact hit that samples from the parked hidden.
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_6::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    // Every frontier capture that lies before the prompt end forces an extra prefill step: the
    // executor clamps that step to the boundary so the scheduler can capture there. The service
    // projection is what admission freezes, so an uncounted split here overruns it at decode.
    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->rewrite_checkpoint_capture &&
                 plan->rewrite_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL) +
        (plan->stable_checkpoint_capture &&
                 plan->stable_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL) +
        static_cast<std::size_t>(std::count_if(
            plan->message_checkpoint_captures.begin(), plan->message_checkpoint_captures.end(),
            [&](const RewriteCheckpointSpec& checkpoint) {
                return checkpoint.frontier < plan->summary.prompt_tokens;
            })) +
        (plan->prefix_checkpoint_capture &&
                 plan->prefix_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL) +
        (plan->tools_checkpoint_capture &&
                 plan->tools_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
