#pragma once

#include <chrono>
#include <cstddef>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency = 8;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;
// Default aggregate merged visual tokens per prompt. The budget itself is an option with no
// compiled upper bound; `kUnlimitedVisionTokens` disables the budget check and sizes the Vision
// tokens by the context ceiling instead.
inline constexpr std::uint32_t kDefaultVisionTokenBudget = 32768;
inline constexpr std::uint32_t kUnlimitedVisionTokens    = 0;
// Marks a sequence plan whose request transient is allocated on demand rather than frozen at
// startup; used by the unlimited Vision budget, where the worst case is the whole context.
inline constexpr std::size_t kDynamicRequestTransientBytes = static_cast<std::size_t>(-1);

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

// Rotary position-embedding regime. `Native` is the checkpoint's own RoPE: the registered native
// context capacity is the ceiling and nothing about the rope path changes. `Yarn` applies YaRN
// frequency correction (`--yarn-factor` x `--yarn-origin`), which raises the addressable context
// ceiling to `yarn_origin * yarn_factor` and substitutes a corrected inverse-frequency table plus
// an `mscale` multiplier inside the rope kernels. Native is bit-for-bit the pre-YaRN path.
enum class RopeMode : std::uint8_t {
    Native,
    Yarn,
};

// Product ceiling on any YaRN-extended context, independent of factor/origin.
inline constexpr std::uint32_t kMaximumYarnContext = 1048576;

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

enum class VisionDevice : std::uint8_t {
    Gpu,
    Cpu,
};

// Experimental, approximate RAM-KV retrieval. Zero disables it. The window bounds GPU KV
// residency; the archive retains the original packed KV bytes, not a second quantization.
struct RamKvOptions {
    std::uint32_t gpu_tokens = 0;
    std::uint64_t budget_bytes = 32000000000ULL;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    int device = 0;
    // Tensor-parallel degree: 1, 2 or 4. Parallel execution requires `devices` to name exactly
    // tp distinct ids of the same compute capability. The 27B package supports TP2 Text/MTP
    // its optional DFlash2 backend, and Vision; TP4 supports SM70 Qwen3.8 NVFP4/native FP8
    // Text/MTP without Vision. The 35B-A3B package requires TP1.
    int tp = 1;
    // Pipeline degree. PP2 owns complete consecutive layer ranges on two SM70 devices;
    // it requires TP1, one active request, Text, and None/MTP speculation.
    int pp = 1;
    // Number of chunks submitted in one PP prefill wave. One is the serial control profile.
    std::uint32_t pipeline_chunks = 8;
    // Explicit compute device ids. Empty means "derive from `device`" (i.e. {device});
    // this lets callers that construct EngineOptions directly (tests, embedders) omit it. When
    // non-empty its size must equal tp * pp.
    std::vector<int> devices;
    // Optional second GPU used only as a VMM-backed expert-weight store. This keeps tp == 1:
    // attention, KV, and all scheduling remain on `device`; only supported MoE artifacts may use
    // the storage pages. -1 disables the mode.
    int storage_device = -1;
    std::uint32_t max_context          = 2048; // Exact logical ceiling of each request.
    // Rotary regime. `RopeMode::Yarn` raises the ceiling `max_context` is validated against from
    // the variant's registered native capacity to `yarn_origin * yarn_factor` (capped at
    // `kMaximumYarnContext`); `yarn_origin` must equal that registered native capacity. Native
    // leaves every rope call site on the checkpoint's own constant frequency table.
    RopeMode rope_mode                 = RopeMode::Native;
    double yarn_factor                 = 4.0;
    std::uint32_t yarn_origin          = 262144;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    // Absolute bound from lifetime acquisition to admission: it covers media acquisition, prompt
    // preparation and time spent waiting behind another request's prefill (one prefill owner at a
    // time). Zero disables the deadline entirely, so a request waits as long as the Engine needs.
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // Maximum prompt tokens one staged prefill step processes while another lane is decoding.
    // A prefill step has a large fixed cost, so the slice bounds the GPU work but cannot by
    // itself give the decoder much time. Zero restores whole-chunk steps. Must be a multiple of
    // 128 when nonzero.
    std::uint32_t prefill_slice_tokens = 256;
    // Decode rounds run between two prefill steps while another lane is decoding. This is the
    // effective fairness knob: a prefill step costs ~140 ms of fixed overhead, so one round per
    // step leaves the decoder at ~12 tok/s regardless of slice size; more rounds trade prefill
    // throughput for decoder responsiveness. Must be at least 1.
    std::uint32_t prefill_decode_rounds = 1;
    // Interval at which the executor emits an OutputSink::keepalive() ping while a request is
    // producing no output -- a multi-minute prefill, or time spent queued behind another
    // request's prefill. The consumer (e.g. an SSE handler) turns this into a transport-level
    // liveness frame so the connection is not torn down as idle. Zero disables the ping.
    std::uint32_t stream_keepalive_ms   = 15000;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    // Complete-history SM70 TP2 long attention with coarser KV partitions and FP32 PV.
    // Numerically qualified, not output-bit-equivalent to the default profile.
    bool volta_long_attention          = false;
    RamKvOptions ram_kv;
    // Host budget, in bytes, for parked conversations. Zero (the default) disables parking: a
    // finished conversation's KV stays resident for prefix reuse and is dropped the moment another
    // conversation needs the lane. With a budget, that KV and its GDN state are instead copied to
    // host memory when the lane is reclaimed, so a conversation that comes back resumes exactly
    // without re-prefilling its prompt.
    std::uint64_t kv_park_bytes = 0;
    // Background spill writer: payload bytes of submitted-but-undrained spill images. Spill
    // writes run on one deterministic FIFO thread so the serving worker never blocks on a
    // multi-GB image; 0 restores the old synchronous write on the worker.
    std::uint64_t kv_park_io_bytes = 8ULL << 30;
    // At startup, read this many most-recently-used spilled conversations back into the RAM
    // tier in MRU order, so the first request after a restart avoids a disk restore. 0 disables.
    std::uint32_t kv_park_prefetch = 0;
    // Delete spilled conversation images (and sidecars) that have not been written or restored
    // for this many seconds. The byte budget already evicts least-recently-used entries when it
    // is exceeded; this additionally removes long-idle leftovers below that budget. 0 disables.
    std::uint64_t kv_park_max_age_sec = 0;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    // Where the Vision encoder executes. `Gpu` keeps the quantized backbone in the TP device
    // arena and runs the existing CUDA pipeline. `Cpu` retains the backbone payloads in host
    // memory, runs the encoder on host threads, and only copies the merged embeddings to the
    // device, which frees the vision weight and scratch reservations at the cost of CPU speed.
    VisionDevice vision_device             = VisionDevice::Gpu;
    // Aggregate merged visual tokens per prompt. The value is not capped: a positive budget sizes
    // the startup Vision scratch and request transient; `kUnlimitedVisionTokens` (0) disables the
    // budget check, sizes the encoder limits by max_context, and allocates the request transient
    // on demand. Unlimited requires a CPU Vision encoder or `vision_kv_swap`, since neither
    // allocation can be pre-reserved.
    std::uint32_t vision_max_tokens         = kDefaultVisionTokenBudget;
    // GPU Vision normally carries its full encode scratch in the startup workspace reservation.
    // With this set, that reservation is skipped (the memory goes to KV/runtime capacity
    // instead) and each image's encode allocates its scratch on demand. To make room, the
    // persistent runtime arena -- this rank's KV and per-lane state, including other resident
    // lanes -- is temporarily copied to host memory and released, then restored afterwards.
    // Each image therefore pays a host round trip of roughly the requested scratch size.
    // Requires enable_vision with VisionDevice::Gpu.
    bool vision_kv_swap                    = false;
    // Captures an extra reusable checkpoint at the end of the message before the last one. A
    // client that rewrites or drops its final message (ephemeral system-reminders, an
    // interrupted tool result) can then resume from the common message prefix instead of
    // re-prefilling the whole prompt.
    bool message_boundary_checkpoints      = false;
    // Recent message boundaries retained, counting the stable one. Clients can rewrite an
    // assistant message several turns back; each retained boundary is a resume point before it.
    // 0 or 1 keeps only the stable boundary.
    std::uint32_t boundary_history         = 1;
    // Captures a reusable checkpoint after the static leading block (tool definitions and system
    // messages). Agent sessions share that block, so a new conversation can restore a prefix the
    // server already prefilled instead of starting from token zero.
    bool shared_prefix_checkpoints         = false;
    // Persist the shared-prefix resume state (KV pages, recurrent slots and hidden) to this path
    // after it is captured, and reload it at startup so a restart does not re-prefill the block.
    std::string shared_prefix_save_path;
    std::string shared_prefix_load_path;
    // Directory holding one image per distinct shared prefix (leading block and tool anchor).
    // Unlike the single-file paths, every distinct prefix captured by this process is kept, and
    // every file in the directory is loaded at startup.
    std::string shared_prefix_dir;
    // Spill parked conversations to this directory when the host parking budget is exhausted.
    // An empty path disables the disk tier; `park_spill_bytes` bounds it (0 = same as
    // kv_park_bytes).
    std::string park_spill_dir;
    std::uint64_t park_spill_bytes         = 0;
    bool use_cuda_graph                    = true;
    LoadProgress load_progress;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
    // Liveness ping emitted periodically while the request produces no output (see
    // EngineOptions::stream_keepalive_ms). Consumers that do not care may ignore it;
    // a consumer that owns a transport turns it into a keepalive frame. A throw from here is
    // treated exactly like a throw from publish(): the request is cancelled. It runs on the
    // same thread as publish(), so implementations need no extra synchronization.
    virtual void keepalive() {}
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

enum class PrefixReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreTurnCheckpoint,
    RestoreResponseCheckpoint,
    // The resident frontier was parked to host memory by an earlier eviction and is being brought
    // back. Reported only when it is materialized; from the model's point of view it resumes
    // exactly like AppendAtFrontier at the parked frontier.
    RestoreParkedKv,
    // The reusable prefix lives in another lane's allocation and is reference-counted into this
    // lane instead of being copied or re-prefilled. The shared pages are copied on write where
    // the two continuations diverge.
    ShareLanePrefix,
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    // Rotary regime as the target runtime resolved it. `effective_max_context` is the ceiling
    // `max_context` was admitted against; `yarn_mscale` is the rope-path cos/sin multiplier the
    // corrected frequency table carries (exactly 1 under `RopeMode::Native`).
    RopeMode rope_mode                        = RopeMode::Native;
    std::uint32_t effective_max_context       = 0;
    double yarn_mscale                        = 1.0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    ArenaMemorySummary request_transient;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Graph residency measured on THIS device (rank 0). The allowance above is a per-device
    // budget, so both figures are per-device and neither is a total to be divided.
    std::size_t cuda_graph_observed_bytes         = 0;
    // Graph residency measured on rank 1, and 0 at tp == 1. A tensor-parallel decode program is a
    // single cross-device graph, but instantiating it materializes driver and module state on both
    // devices, so the second device carries its own cost against the same allowance.
    std::size_t cuda_graph_peer_observed_bytes    = 0;
    // Each active rank's measured residency; inactive rows are zero.
    std::array<std::size_t, 4> cuda_graph_rank_observed_bytes{};
    // Node count of one captured decode graph, 0 when graphs are disabled. A parallel graph holds
    // all active ranks' nodes and the collectives' cross-device copies; event edges are not nodes.
    std::size_t cuda_graph_node_count             = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::uint32_t ram_kv_gpu_tokens = 0;
    std::uint64_t ram_kv_archive_bytes = 0; // Maximum packed archive, summed across ranks.
    std::uint64_t ram_kv_transfer_bytes = 0; // Actual H2D+D2H payload, summed across ranks.
    // GDN (linear-attention) recurrent + convolution state for THIS device, head-sharded by tp.
    std::size_t gdn_state_bytes = 0;
};

// One row of the per-device memory table. At tp * pp == 1 only `devices[0]` is
// meaningful. Every byte count is measured or planned for that device specifically -- nothing here
// is a model total divided by the tensor-parallel width.
struct DeviceMemoryReport {
    int device                              = -1;
    std::uint64_t weights_bytes             = 0; // this device's weight arena (its own shard)
    std::uint64_t kv_pool_bytes             = 0; // paged KV payload
    std::uint64_t gdn_state_bytes           = 0; // GDN recurrent + conv state
    std::uint64_t sequence_bytes            = 0; // whole persistent arena (KV + GDN + round state)
    std::uint64_t workspace_bytes           = 0; // workspace arena capacity
    std::uint64_t cuda_graph_bytes          = 0; // measured graph residency on THIS device
    std::uint64_t reserved_bytes            = 0; // weights + runtime reservation
    std::uint64_t free_after_startup_bytes  = 0;
    std::uint64_t total_bytes               = 0;
};

// Monotonic execution counters plus one boundary-consistent scheduler snapshot. Consumers derive
// interval throughput by subtracting two snapshots and dividing by their own monotonic wall time.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; resident prefix hits are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds         = 0;
    std::uint64_t decode_row_rounds     = 0;
    std::uint32_t running_requests      = 0;
    std::uint32_t prefilling_requests   = 0;
    std::uint32_t decode_ready_requests = 0;
    std::uint32_t waiting_requests      = 0;
    // Progress of the single request currently owning the prefill lane. At most one prefill
    // runs at a time, so these name that request: `prefill_prompt_tokens` is its full prompt
    // length, `prefill_processed_tokens` counts the tokens made resident so far (reused prefix
    // hits plus prefill work completed this run). Both are 0 when no prefill is in flight, and
    // the count climbs toward the prompt length as the lane advances. Unlike the counters above
    // these are instantaneous, not monotonic, so a reader must not difference them across
    // snapshots. A reader sees the lane go 0 -> the running progress -> 0; the final step clears
    // the lane before it publishes, so the peak observed value is (prompt length - final chunk),
    // not equality.
    std::uint32_t prefill_prompt_tokens    = 0;
    std::uint32_t prefill_processed_tokens = 0;
};

struct LoadSummary {
    int tp = 1;
    int pp = 1;
    std::array<DeviceMemoryReport, 4> devices{};
    // Rotary regime resolved at construction. `effective_max_context` is the ceiling `max_context`
    // was validated against (the variant's native capacity under `RopeMode::Native`,
    // `yarn_origin * yarn_factor` under `RopeMode::Yarn`); `yarn_mscale` is the rotary cos/sin
    // multiplier the YaRN table carries and is exactly 1 under Native.
    RopeMode rope_mode                    = RopeMode::Native;
    double yarn_factor                    = 0.0;
    std::uint32_t yarn_origin             = 0;
    std::uint32_t effective_max_context   = 0;
    double yarn_mscale                    = 1.0;
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
};

} // namespace ninfer
