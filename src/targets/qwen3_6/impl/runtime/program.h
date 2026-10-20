#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/parked_disk_io.h"
#include "targets/qwen3_6/impl/runtime/parked_sequence_store.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

struct SequenceImageGeometry;

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    std::optional<qwen3_6::RewriteCheckpointSpec> stable_checkpoint;
    std::vector<qwen3_6::RewriteCheckpointSpec> message_checkpoints;
    std::optional<qwen3_6::RewriteCheckpointSpec> prefix_checkpoint;
    std::optional<qwen3_6::RewriteCheckpointSpec> tools_checkpoint;
    bool allow_prefix_reuse = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    // Identification of the parked entry a RestoreParkedKv plan resumes, resolved against the
    // store when the plan is executed rather than carried as a pointer, which an intervening
    // park could invalidate.
    std::uint64_t parked_id = 0;
    // True when the parked entry is resumed at its rewrite-checkpoint frontier rather than its
    // execution frontier: the replay diverged inside the last response, so only the checkpoint
    // prefix is reusable and `reuse_base` names that frontier.
    bool parked_checkpoint = false;
    // Which retained checkpoint a checkpoint restore resumes from: 0 is the current one, i+1 is
    // history[i]. Used for both resident lanes and parked entries.
    std::uint32_t checkpoint_index = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    // An extra capture at the last stable message boundary, when one lies ahead of the reuse
    // frontier. It lands in the host checkpoint history and lets a client that rewrites or drops
    // its final message resume from the common message prefix.
    std::optional<qwen3_6::RewriteCheckpointSpec> stable_checkpoint_capture;
    // Recent message boundaries captured in prompt order, oldest first, excluding the stable one.
    std::vector<qwen3_6::RewriteCheckpointSpec> message_checkpoint_captures;
    // A capture after the static leading block; shared by every session with that prefix.
    std::optional<qwen3_6::RewriteCheckpointSpec> prefix_checkpoint_capture;
    // A capture after the tool definitions, before the volatile leading instruction text.
    std::optional<qwen3_6::RewriteCheckpointSpec> tools_checkpoint_capture;
    // The selected boundary is one of the lane's pinned shared prefixes rather than a ring entry.
    SharedPrefixSlot restore_shared_prefix = SharedPrefixSlot::None;
    // ShareLanePrefix: the source lane whose allocation supplies the reusable prefix pages, the
    // pinned boundary holding its recurrent state, and how many leading pages are borrowed.
    std::uint32_t share_lane              = 0;
    SharedPrefixSlot share_slot           = SharedPrefixSlot::None;
    std::uint32_t shared_text_kv_pages    = 0;
    std::uint32_t shared_backend_kv_pages = 0;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
    // Each peer owns its text KV pool: TP splits heads; PP splits layers. Lockstep pool
    // operations preserve matching page identifiers and block tables across compute devices.
    std::array<std::optional<PagedKVAllocation>, kMaximumExecutionDevices - 1> text_peers;
    // Peer backend allocations use the same lockstep page ownership. In PP2 the peer's
    // symmetric MTP reservation is unused by computation.
    std::array<std::optional<PagedKVAllocation>, kMaximumExecutionDevices - 1> backend_peers;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    bool greedy_target                   = false;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    RewriteCheckpoint rewrite_checkpoint;
    // Token/position lineage the current device checkpoint was captured from. Selection and
    // rotation must use this, not the lane's live ledger, which may have moved to another branch.
    std::vector<TokenId> checkpoint_ledger;
    ResidentPrefixIdentity checkpoint_identity;
    // Older rewrite-checkpoint boundaries of this conversation, newest-first. Kept host-side so
    // a stale or forked replay can resume at one of them without growing device memory.
    std::vector<HostCheckpoint> checkpoint_history;
    // Recent message boundaries, newest first, pinned outside the ring. A client can rewrite an
    // assistant message several turns back (dropping an interrupted tool_use also drops its
    // tool_result), and each boundary before such a message is a resume point the depth-3 ring
    // would have evicted.
    std::vector<HostCheckpoint> boundary_history;
    // Recent response (assistant-header) checkpoints, newest first, pinned outside the ring. A
    // client re-renders the assistant message it received, so a tool turn diverges at that
    // message's header; only a response checkpoint resumes there.
    std::vector<HostCheckpoint> response_history;
    // Sparse long-range checkpoints, newest first, so a client snapshot lagging many turns behind
    // still resumes within about kLongRangeSpacing tokens.
    std::vector<HostCheckpoint> long_range_history;
    // The shared leading-block boundary, pinned outside the ring: it is identical across agent
    // sessions, so it must survive ring rotation to stay usable by a later conversation.
    HostCheckpoint shared_prefix;
    // The same for the end of the tool definitions, which precede the leading instruction text
    // and therefore survive changes in it (environment lines, dates).
    HostCheckpoint tools_prefix;
    // The boundary this lane last resumed from when a request had to restart mid-conversation
    // (a rewrite, a fork, a lagging replay). Two branches of one conversation share tokens up to
    // their fork, and this boundary is the branch's return point; it lives outside the
    // depth-limited histories because the captures made while re-prefilling the new branch would
    // evict it first.
    HostCheckpoint resume_anchor;
    TokenSpan ram_query;
    std::uint32_t ram_rank_frontier = 0;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        // Captured once, at its frontier, into the host checkpoint history; see RequestPlanImpl.
        std::optional<RewriteCheckpointSpec> stable_checkpoint_capture;
        bool stable_checkpoint_captured  = false;
        std::vector<RewriteCheckpointSpec> message_checkpoint_captures;
        std::size_t message_capture_index = 0;
        std::optional<RewriteCheckpointSpec> prefix_checkpoint_capture;
        bool prefix_checkpoint_captured  = false;
        std::optional<RewriteCheckpointSpec> tools_checkpoint_capture;
        bool tools_checkpoint_captured   = false;
        // Where the resumed state's hidden at base-1 lives, captured before start_prefill clears
        // tail_hidden_valid for a suffix. Retention uses it when nothing has been staged yet.
        enum class ResumeHidden : std::uint8_t { None, Tail, Checkpoint };
        ResumeHidden resumed_hidden = ResumeHidden::None;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        // Column of the last processed token's hidden in the most recent chunk. A cancelled
        // prefill keeps it so the next request can resume at `cursor` through the ordinary
        // append path instead of re-prefilling the whole prompt.
        std::int32_t tail_column         = -1;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

// Rank 1's complete runtime mirror: its own arenas, its own shard of the weights, its own halved
// decoder state, and its own RoundState. It owns no bookkeeping -- lanes, page accounting,
// sampling and the pinned host round buffers all live once, on rank 0.
struct PeerRuntime {
    PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                const SequencePlanImpl& plan, int rank);

    PeerRuntime(const PeerRuntime&)            = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    const int rank;
    DeviceContext& device;
    const LoadedModelData& model;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<DFlashPersistentState> dflash;
    // Rank 1's own GDN replay records: the speculative verify round records this device's own
    // head/channel shard and folds it here, so the two devices commit the same accepted prefix
    // from records neither ever exchanges.
    std::optional<GdnReplayRecords> replay_records;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    // Rank 1's OWN penalty counters. `ops::SamplingConfig::token_counts` is a raw device pointer,
    // and the MTP round's acceptance runs on both devices (see the replicated-accept note in
    // mtp_impl.h), so rank 1 must never be handed rank 0's. It is not a cache: rank 0 remains the
    // source of truth (ProgramImplCore::install_sampling zeroes both, and the one increment that
    // happens on rank 0 alone -- prefill's bonus token -- is copied across before the first decode
    // round), and thereafter both are advanced by the same Op over bit-identical inputs.
    Tensor token_counts;
    std::optional<DecodeGraphPeerBridge> graph_bridge;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, std::span<const LoadedModelData* const> peer_models,
                    const SequencePlanImpl& plan, ExecutionContext& execution);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    // `max_tokens` bounds one step below the configured prefill chunk, so a scheduler can slice a
    // long prefill and give other lanes decode rounds in between; zero uses the full chunk.
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient,
                                                                std::uint32_t max_tokens = 0);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane,
                                                                  std::uint32_t max_tokens = 0);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    // Shutdown dump: parks retained lanes and writes every RAM-parked conversation to disk.
    void persist_state();
    // A cancelled decode stops at a resolved round boundary, so its lane holds exactly the state
    // a completed turn leaves behind. Keeps that state for the conversation's next request when
    // it is decode-ready; otherwise clears the lane like abort_lane.
    void retain_cancelled_lane(std::uint32_t lane) noexcept;
    // A cancelled staged prefill has already materialized a KV prefix and the hidden state of the
    // last token it processed. Keeps that prefix as an append frontier (with the hidden as its
    // MTP bridge input) so the conversation's next request resumes there instead of re-prefilling
    // everything. Returns false -- and leaves the lane untouched -- when the prefix cannot be
    // retained exactly (no chunk completed, a Vision consumer span would be split, or the staged
    // state is inconsistent); the caller then clears the lane as before.
    [[nodiscard]] bool retain_cancelled_prefill_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    // Copies a retained lane's complete continuation state to the park store and clears the lane,
    // returning true when it was parked. Best-effort and infallible by contract: a lane that
    // cannot be parked (parking disabled, RAM KV, Vision, or no room left in the store) is
    // reported false and left untouched, so the caller can fall back to dropping it.
    // `protect_id` names a stored entry the insert must not evict (the plan's restore target).
    // With `detach == false` the state is snapshotted without disturbing the lane: the pages are
    // copied rather than released, so a caller that is about to trim the lane for another branch
    // keeps the outgoing branch for a later return while still resuming the one it planned.
    [[nodiscard]] bool park_retained_lane(std::uint32_t lane, std::uint64_t protect_id = 0,
                                          bool detach = true) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    ExecutionContext& execution;
    DeviceContext& device;
    const int device_count;
    const int pp;
    const std::uint32_t pipeline_chunks;
    std::optional<PinnedHostBuffer> pipeline_upload;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const RamKvOptions ram_kv;
    const bool volta_long_attention;
    // Recent message boundaries pinned per lane/entry; see EngineOptions::boundary_history.
    const std::uint32_t boundary_history_depth;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    // Vision runs on host threads: no device Vision workspace is reserved or validated.
    const bool vision_cpu;
    // GPU Vision scratch is demand-allocated per encode by swapping the persistent arena to host.
    const bool vision_kv_swap;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t graph_allowance_bytes;
    // Measured graph residency per device (index = rank). At tp1 only [0] is populated. At tp2 the
    // single cross-device graph materializes driver state on BOTH devices, and each is checked
    // against the SAME per-device allowance -- graph_allowance_bytes is a per-device budget, like
    // every other field in device_reservation_bytes.
    std::array<std::size_t, kMaximumExecutionDevices> graph_observed_bytes{0, 0};
    // Node count of ONE captured decode graph (the first profile of the captured family). At tp2
    // one graph holds both devices' nodes, so this is the direct measurement of whether the peer's
    // half of the schedule was captured rather than left out.
    std::size_t graph_node_count = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    // YaRN rotary state, resolved once at construction.
    //
    // `rope_frequency_storage[rank]` is a dedicated 32-float device allocation on rank `rank`'s OWN
    // device, made once here and never touched again: CUDA Graph capture bakes the pointer into the
    // replayed rope launch node, so it must outlive every replay, and at device_count 2 each rank ropes its
    // own head-local q/k on its own device, where a pointer into the other rank's allocation is not
    // addressable. It is deliberately NOT part of the planned persistent arena: under
    // `RopeMode::Native` nothing is allocated at all, which is what keeps a native plan and its
    // memory summary byte-identical to the pre-YaRN engine.
    //
    // `rope_frequency[rank]` is the descriptor every text rope call site reads (through
    // `ExecutionCore::rope_frequency` -> `TextContext`); a null `inv_frequency` IS the native path.
    std::array<DeviceBuffer, kMaximumExecutionDevices> rope_frequency_storage;
    std::array<ops::RopeFrequencyOverride, kMaximumExecutionDevices> rope_frequency{};
    const RopeMode rope_mode;
    const std::uint32_t effective_max_context;
    const double yarn_mscale;
    std::vector<std::unique_ptr<PeerRuntime>> peers;
    std::optional<ops::PeerEvents> peer_events;
    std::array<schedule::PeerCore, kMaximumExecutionDevices - 1> peer_cores{};
    [[nodiscard]] std::span<const schedule::PeerCore> peer_views() const noexcept {
        return std::span(peer_cores).first(pp == 1 ? peers.size() : 0);
    }
    [[nodiscard]] const schedule::PeerCore* pipeline_view() const noexcept {
        return pp == 2 ? &peer_cores[0] : nullptr;
    }
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    // Conversations whose resident frontier did not fit the KV pool, held in host memory so they
    // can be resumed exactly instead of re-prefilled. Disabled (zero budget) by default.
    ParkedSequenceStore parked;
    // Reusable pinned window for GDN slot images. Park/restore move one ~73 MiB slot per rank;
    // transferring straight between the device and the pageable vector that stores the image
    // would serialize each transfer through the driver's bounce buffer. Sized lazily from the
    // pool's slot geometry and reused by every rank.
    std::optional<PinnedHostBuffer> slot_image_host;
    // Disk-backed tier: metadata-only entries whose KV/state payload lives in a file. Pinned
    // entries were loaded from the shared-prefix file; the rest are spilled parked conversations.
    std::vector<std::unique_ptr<ParkedSequence>> disk_entries;
    std::uint64_t next_disk_entry = 1;
    std::uint64_t disk_spilled_bytes = 0;
    // Paths and budget for the disk tier; see EngineOptions.
    const std::string shared_prefix_save_path;
    const std::string shared_prefix_load_path;
    const std::string shared_prefix_dir;
    // (frontier, identity fingerprint) pairs already persisted in the directory.
    std::set<std::pair<std::uint32_t, std::uint64_t>> saved_prefix_keys;
    const std::string park_spill_dir;
    const std::uint64_t park_spill_bytes;
    const std::uint64_t kv_park_io_bytes;
    const std::uint32_t kv_park_prefetch;
    const std::uint64_t kv_park_max_age_sec;
    // Background FIFO spill writer; null when spill I/O is synchronous or no directory is set.
    std::unique_ptr<ParkedDiskIo> disk_io;
    // While the startup index is being built, budget enforcement must not age-expire entries:
    // the prefetch that runs right after loading has first claim on the most recent images.
    bool disk_index_loading_ = false;
    bool saved_tools_prefix    = false;
    bool saved_leading_prefix  = false;
    // Host image of the persistent arena chunks evicted for the Vision encode in flight. Empty
    // outside an encode.
    std::optional<PinnedHostBuffer> vision_swap_host_;
    std::size_t vision_swap_bytes_ = 0;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    // Debug-only capture of the round's full-vocabulary logits, raw BF16 bits. OFF by default and
    // completely inert until enable_logits_capture(true) allocates the host buffer: with it off,
    // copy_round_logits() is a predictable branch and nothing is allocated or copied, so no
    // production path (tp1 or tp2) is changed. With it on, the buffer is overwritten every time a
    // token is sampled from io.logits at prefill finalization, for both tp1 and tp2 (io is rank
    // 0's window; logits_tp2 gathers into it the same way the tp1 leaf writes it), so no
    // device_count-specific code is needed. Prefill only -- the decode path is CUDA-graph-capturable and an
    // unconditional memcpy inside a captured region is rejected at capture time; a probe that
    // requests exactly one output token completes on prefill's own sampled token and never runs a
    // decode round, which is what the parity harness (tools/tp2/parity.cpp) uses. Valid only
    // immediately after Engine::generate()/wait() returns to the calling thread for a single-lane
    // engine -- concurrent lanes would overwrite it out of order.
    std::vector<std::uint16_t> logits_capture;
    bool logits_capture_enabled = false;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    // Rank 1's own pinned ordinary ingress: rank 0's record with every row's
    // `sampling[row].token_counts` nulled. Rank 1 mirrors the ordinary round for its half of the
    // weights and never samples (the output head is vocabulary-split and sampling belongs to rank
    // 0), so the sampling configs in its frame are inert -- but they must not carry a rank-0
    // DEVICE address into rank 1's frame: dereferencing one from rank 1 is a silent cross-device
    // fault. A separate pinned buffer rather than a patched copy at issue time, for the same
    // reason as `mtp_peer_host_ingress`: the upload is inside the captured decode graph, which
    // re-reads this exact host address at every replay.

    bool peer_egress_check_enabled     = false;
    std::uint64_t peer_egress_rounds     = 0;
    std::uint64_t peer_egress_mismatches = 0;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    // Rank 1's own pinned MTP ingress: byte-for-byte rank 0's record except that each row's
    // `sampling[row].token_counts` names rank 1's counter lane. It has to be a separate pinned
    // buffer rather than a patched copy made at issue time, because the ingress upload is inside
    // the captured decode graph and the graph re-reads this exact host address at every replay.

    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;


    std::size_t workspace_logical_peak_bytes = 0;

    // Monotonic tick ordering park-store use, advanced once per park and once per plan that
    // targets a parked entry. Only relative order matters, so any monotone counter will do.
    std::uint64_t park_clock = 0;

    // See logits_capture's comment. Read-only; the caller owns thread-safety (single-lane, read
    // after wait() returns). Empty while capture is disabled.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return logits_capture;
    }

    // Turns the debug capture above on or off. Enabling sizes the host buffer to the round's
    // logits row count; disabling releases it. Must not be called while a round is in flight (the
    // executor holds its execution mutex across this call).
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at device_count == 2, read rank 1's MTP egress
    // back and compare it field for field with rank 0's. The two ranks run the acceptance Op over
    // bit-identical inputs, so their egress records are argued to agree; enabling this turns that
    // induction into a measurement. Costs one ~1 KiB device-to-host copy plus a host compare per
    // round while enabled, and nothing at all while off. Counters are cumulative over the
    // Program's lifetime; a mismatch is counted (per row, per field) rather than thrown, so a test
    // can read the totals after a clean run.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept {
        return peer_egress_rounds;
    }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept {
        return peer_egress_mismatches;
    }

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    // Copies rank 0's counter lane onto rank 1. Called once, after prefill's bonus token is
    // sampled -- the only counter increment that happens on rank 0 and not on rank 1.
    [[nodiscard]] static Tensor token_counts_lane(const Tensor& storage, std::uint32_t lane);
    void publish_peer_token_counts(const SequenceState& sequence);
    // Mirrors `mtp_host_ingress` into `mtp_peer_host_ingress`, swapping every row's counter
    // pointer for rank 1's. No-op at tp1 or without MTP.
    void publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes);
    void publish_peer_dflash_ingress(std::span<const std::uint32_t> lanes);
    // Mirrors `ordinary_host_ingress` into `ordinary_peer_host_ingress` with every row's counter
    // pointer nulled. No-op at tp1 or without an ordinary frame.
    void publish_peer_ordinary_ingress();
    // Debug-only: reads rank 1's MTP egress back and compares it, field for field, with rank 0's.
    // No-op unless the check is enabled and a peer exists.
    void check_peer_mtp_egress(std::size_t rows);
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    static void set_peer_i32(Tensor& tensor, std::int32_t value, DeviceContext& rank_device);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void copy_round_logits();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request,
                                                             std::uint32_t max_tokens = 0);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    // `shared_text_pages`/`shared_backend_pages` name an entitlement prefix borrowed from another
    // lane's allocation (ShareLanePrefix); the pool only charges the owned remainder.
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages, std::uint32_t shared_text_pages = 0,
                             std::uint32_t shared_backend_pages = 0);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    // Materializes a parked conversation into an empty lane: reserves the KV bundle, copies the
    // parked pages and GDN slot back on every rank, and restores the ledger, identity and
    // frontiers, so the resident reuse path can resume from the parked frontier unchanged. With
    // `resume_at_checkpoint` the lane is put back at the entry's rewrite-checkpoint frontier
    // instead: the checkpoint's GDN slot and hidden are materialized and the checkpoint becomes
    // the lane's current one, so the existing checkpoint-restore path can re-prefill the suffix.
    // Disk tier.
    [[nodiscard]] SequenceImageGeometry image_geometry() const;
    // Per-rank KV pools, in rank order (rank 0 first). Each rank owns a distinct pool object and
    // a restored image must be bound to the pool it will attach to.
    void storage_pools(std::vector<PagedKVPool*>& text,
                       std::vector<PagedKVPool*>& backend) const;
    [[nodiscard]] std::vector<const ParkedSequence*> parked_candidates() const;
    [[nodiscard]] const ParkedSequence* find_disk_entry(std::uint64_t id) const noexcept;
    static bool is_disk_entry_id(std::uint64_t id) noexcept;
    void load_shared_prefix_entries();
    // Reloads conversations spilled by an earlier process from their metadata sidecars.
    void load_parked_spill_entries();
    static void remove_sequence_files(const std::string& image_path) noexcept;
    // Keeps the shared-prefix directory bounded: removes the oldest images and their disk index
    // entries when more than kSharedPrefixDirFiles are present.
    void enforce_prefix_dir_cap();
    // Pins the response checkpoint just captured at this request's prompt end. Called at the
    // capture site so a later request's plan can never select an entry that a subsequent pin
    // evicts before the restore runs.
    void pin_response_checkpoint(SequenceState& sequence);
    // Retains the boundary a request resumed from, so a return to the branch it replaced can
    // restore it even after the depth-limited histories have rotated it out.
    static void pin_resume_anchor(SequenceState& sequence, const HostCheckpoint& checkpoint);
    void write_shared_prefix(SequenceState& sequence, const HostCheckpoint& pinned, bool tools);
    // Writes an entry to the spill directory and replaces it with a metadata-only index entry.
    // With the background writer this only submits the job; the index entry appears when the
    // worker drains completions. Returns nullptr on success; on failure it returns the
    // still-intact entry so the caller can reinsert or drop it deliberately.
    std::unique_ptr<ParkedSequence> spill_entry(std::unique_ptr<ParkedSequence> entry,
                                                std::uint64_t protect_id = 0);
    // Moves finished background writes into disk_entries and prunes superseded states. Runs only
    // on the worker thread.
    void drain_disk_completions();
    // Startup: reads the MRU spilled conversations back into the RAM tier (fixed MRU order).
    void prefetch_disk_entries();
    // Deletes spilled images idle longer than kv_park_max_age_sec. Byte-budget LRU still applies.
    void sweep_expired_disk_entries();
    // Drops older parked states that the incoming lineage strictly contains (same tokens,
    // positions and media up to a shorter frontier). One current copy per conversation; the
    // plan's protected target and pinned entries are never removed.
    static bool supersedes(const ParkedSequence& older, const ParkedSequence& incoming) noexcept;
    std::uint64_t superseded_ram_bytes(const ParkedSequence& incoming,
                                       std::uint64_t protect_id) const;
    void prune_superseded_ram(const ParkedSequence& incoming, std::uint64_t protect_id);
    // `keep_id` is the entry the incoming state was just written as; the equal-frontier state
    // would otherwise supersede itself and be deleted on the spot.
    void prune_superseded_disk(const ParkedSequence& incoming, std::uint64_t protect_id,
                               std::uint64_t keep_id = 0);
    void enforce_disk_budget();
    void restore_disk_entry(SequenceState& sequence, const ParkedSequence& meta,
                            std::uint32_t text_pages, std::uint32_t backend_pages,
                            const RequestPlanImpl& plan);
    void restore_parked_lane(SequenceState& sequence, ParkedSequence& entry,
                             std::uint32_t text_pages, std::uint32_t backend_pages,
                             bool resume_at_checkpoint, std::uint32_t checkpoint_index,
                             SharedPrefixSlot resume_at_shared_prefix = SharedPrefixSlot::None,
                             std::uint32_t resume_frontier = 0);
    // Copies a checkpoint's per-rank recurrent slot and rank-0 hidden from host. `into_current`
    // also materializes it into the running slot (a parked restore); otherwise only the
    // checkpoint slot is written and the resident checkpoint path copies it forward.
    void import_checkpoint_state(SequenceState& sequence, const HostCheckpoint& checkpoint,
                                 bool into_current);
    // Pinned staging window sized to one GDN slot image, shared by every rank's park/restore.
    [[nodiscard]] std::span<std::byte> slot_image_staging();
    // Copies a pageable slot image into the pinned window and imports it into `slot`.
    void import_slot_staged(LinearAttentionStatePool& state, std::int32_t slot,
                            std::span<const std::byte> image, cudaStream_t stream);
    [[nodiscard]] HostCheckpoint export_checkpoint(SequenceState& sequence);
    // Moves the lane's current checkpoint into its host history, if one is valid.
    void rotate_device_checkpoint(SequenceState& sequence);
    // Resolves a plan that resumes from a historical boundary: preserves the current device
    // checkpoint, imports the selected one, and (when the request will capture a new checkpoint)
    // keeps the selected boundary in the history too.
    void restore_historical_checkpoint(SequenceState& sequence, std::uint32_t checkpoint_index,
                                       bool preserve);
    void refresh_ram_kv(SequenceState& sequence, std::uint32_t frontier);
    // Demand allocation behind VisionSwapHooks: evicts enough of the persistent arena to host
    // memory for one Vision encode; restore puts the evicted bytes back. Both no-op unless
    // vision_kv_swap is on.
    void evict_vision_workspace(std::size_t bytes);
    void restore_vision_workspace();
    static void vision_swap_evict_hook(void* self, std::size_t bytes);
    static void vision_swap_restore_hook(void* self);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] std::array<qwen3_6::PagedKVCacheView, kMaximumExecutionDevices - 1>
    mtp_kv_views_peer(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
