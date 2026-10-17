// The Engine worker loop's exception boundary.
//
// The executor runs every request on one worker thread. Historically ANY exception raised while a
// round was being turned into published text reached worker_loop's catch-all, which calls
// fail_all() and RETURNS: the worker exits, `failed_` latches, and the Engine answers "inference
// engine is unavailable" for the rest of the process. A byte-level vocabulary makes ill-formed
// UTF-8 in generated text an ordinary outcome, so that boundary had to be split by SCOPE:
//
//   * request scope -- detokenization, stop/channel policy and publication of ONE OutputSession:
//     that request fails (or, after the decoder fix, simply publishes U+FFFD), the lane is torn
//     down, and the Engine keeps serving;
//   * engine scope -- the shared execution unit (Program/CUDA): every request fails and the
//     worker stops, exactly as before.
//
// This test drives the real ConcurrentExecutor with a scripted in-process Program, so both halves
// are asserted end to end: submit -> worker thread -> published text / error -> next request.
// It needs a CUDA device (the worker binds itself to one) but no artifact and no model.

#include "runtime/engine/concurrent_executor.h"

#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;
namespace rt  = ninfer::runtime;

using ninfer::TokenId;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

// ---------------------------------------------------------------------------------------------
// Frontend fixture: a synthetic byte-level vocabulary. Every entry is one byte of the GPT-2
// byte-level alphabet, so a test can hand the decoder any byte sequence it likes, one token at a
// time -- which is exactly how a real checkpoint emits text. 6 = <eos>.
//
//   id 0  'x' 0x78      id 10 'ä' 0xE4     id 11 '¸' 0xB8     id 12 'Ń' 0xAD   ("中" = E4 B8 AD)
//   id 13 'À' 0xC0      id 14 'õ' 0xF5     (bytes that can never lead)
//   id 15 'à' 0xE0      id 16 'í' 0xED     id 17 'ð' 0xF0     id 18 'ô' 0xF4
//        (the four leads with a NARROWED second-byte range -- Unicode 15 table 3-7)
//   id 19 'Ģ' 0x80      id 20 'ı' 0x8F     id 21 'Ĳ' 0x90     id 22 'Ł' 0x9F
//   id 23 'ł' 0xA0      id 24 '¿' 0xBF     id 35 'ĺ' 0x98     (continuation bytes at the bounds)
//   id 25 'ñ' 0xF1      id 26 'á' 0xE1     id 27 'Â' 0xC2     (Unicode 15 table 3-8 vector)
//   id 28 'a' 0x61      id 29 'b' 0x62     id 33 'c' 0x63     id 34 'd' 0x64
// ---------------------------------------------------------------------------------------------

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

const std::string& chat_template_source() {
    static const std::string source = [] {
        std::string text = read_file(NINFER_SOURCE_DIR
                                     "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
        if (!text.empty() && text.back() == '\n') { text.pop_back(); }
        return text;
    }();
    return source;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

q36::FrontendResources resources() {
    q36::FrontendResources result;
    result.chat_template_jinja  = chat_template_source();
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(30, "user\n"), added(31, "assistant\n"), added(32, "\n"),
         added(248045, "<|im_start|>", true), added(248046, "<|im_end|>", true),
         added(248053, "<|vision_start|>", true), added(248054, "<|vision_end|>", true),
         added(248056, "<|image_pad|>", true), added(248057, "<|video_pad|>", true),
         added(248068, "<think>"), added(248069, "</think>")});
    result.tokenizer_json =
        nlohmann::json{{"model",
                        {{"type", "BPE"},
                         {"vocab",
                          {{"x", 0},   {"ä", 10},  {"¸", 11},  {"Ń", 12},  {"À", 13}, {"õ", 14},
                           {"à", 15},  {"í", 16},  {"ð", 17},  {"ô", 18},  {"Ģ", 19}, {"ı", 20},
                           {"Ĳ", 21},  {"Ł", 22},  {"ł", 23},  {"¿", 24},  {"ñ", 25}, {"á", 26},
                           {"Â", 27},  {"a", 28},  {"b", 29},  {"c", 33},  {"d", 34}, {"ĺ", 35}}},
                         {"merges", nlohmann::json::array()}}},
                       {"added_tokens", tokens}}
            .dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{{"add_bos_token", false},
                                                  {"add_prefix_space", false},
                                                  {"pad_token", "<|endoftext|>"},
                                                  {"chat_template", result.chat_template_jinja},
                                                  {"added_tokens_decoder", std::move(decoder)}}
                                       .dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

// ---------------------------------------------------------------------------------------------
// Scripted Program. It performs no device work: every request pops one token script, prefill
// licenses its first token and each decode round licenses the next one.
// ---------------------------------------------------------------------------------------------

struct ScriptedPlan {
    rt::RequestPlanSummary value;

    [[nodiscard]] const rt::RequestPlanSummary& summary() const noexcept { return value; }
};

struct GenerationScript {
    std::vector<TokenId> tokens;
    // When set, decode_batch throws instead of licensing the token at this index. This is the
    // engine-scope leg: a Program/CUDA failure, not an output-session failure.
    int throw_at_decode_index = -1;
    // Number of prefill steps; the first token is licensed on the FINAL step and earlier steps
    // only make bounded progress. Because the Engine admits at most one prefill owner and refuses
    // admission while any prefill is in flight, a multi-step prefill is how a test holds the
    // admission gate shut. Each step sleeps |prefill_step_delay| so the worker observes real
    // wall-clock progress, and with it real admission boundaries, between steps.
    std::uint32_t prefill_steps = 1;
    std::chrono::milliseconds prefill_step_delay{0};
};

class ScriptedProgram {
public:
    explicit ScriptedProgram(std::uint32_t max_concurrency) : max_concurrency_(max_concurrency) {}

    void push_script(GenerationScript script) { queued_.push_back(std::move(script)); }

    // Extra service-work quanta reserved on top of the output budget. Real plans include prefill
    // work in service_work_quanta; this fixture charges one quantum per prefill step, so a
    // multi-step script needs headroom or the projection is overdrawn mid-prefill.
    void set_service_allowance(std::uint64_t quanta) { service_allowance_ = quanta; }

    [[nodiscard]] rt::AdmissionResources admission_capacity() const noexcept {
        return rt::AdmissionResources{
            .active_lanes = max_concurrency_, .main_kv_pages = max_concurrency_, .backend_kv_pages = 0};
    }

    // The budget comes from the request's own options, never from the script queue: with several
    // requests in flight the executor plans one while another is still holding the queue head.
    [[nodiscard]] ScriptedPlan plan_request_base(const q36::PreparedPrompt& prompt,
                                                 const rt::ResolvedExecutionOptions& execution) const {
        const std::uint32_t output_tokens = execution.requested_output_tokens;
        if (output_tokens == 0) { throw std::logic_error("scripted request has no output budget"); }
        return ScriptedPlan{rt::RequestPlanSummary{
            .prompt_tokens           = prompt.summary().prompt_tokens,
            .reusable_prompt_tokens  = 0,
            .requested_output_tokens = output_tokens,
            .effective_output_tokens = output_tokens,
            .effective_limit_reason  = ninfer::FinishReason::OutputLimit,
            .transient_bytes         = 0,
            .transient_alignment     = 1,
            .admission = rt::AdmissionResources{.active_lanes = 1, .main_kv_pages = 1},
            .service_work_quanta = output_tokens + service_allowance_,
        }};
    }

    [[nodiscard]] ScriptedPlan plan_request_for_lane(std::uint32_t, const q36::PreparedPrompt&,
                                                     const ScriptedPlan& base) const {
        return base;
    }

    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const ScriptedPlan&) const noexcept {
        return lane < max_concurrency_ && lanes_[lane].lifecycle == Lifecycle::Empty;
    }

    [[nodiscard]] bool can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                              const ScriptedPlan& plan) const noexcept {
        return can_admit_lane(lane, plan);
    }

    rt::PrefillStepResult start_prefill_lane(std::uint32_t lane, q36::PreparedPrompt&& prompt,
                                             ScriptedPlan&& plan, rt::TransientRegion) {
        if (queued_.empty()) { throw std::logic_error("scripted program has no remaining script"); }
        Lane& state         = lanes_[lane];
        state.script        = std::move(queued_.front());
        queued_.pop_front();
        state.lifecycle     = Lifecycle::Active;
        state.emitted       = 0;
        state.prompt_tokens = plan.summary().prompt_tokens;
        state.prefill_done  = 0;
        return prefill_step(lane);
    }

    rt::PrefillStepResult advance_prefill_lane(std::uint32_t lane) {
        if (lanes_[lane].lifecycle != Lifecycle::Active) {
            throw std::logic_error("scripted prefill step no longer matches Program state");
        }
        return prefill_step(lane);
    }

    void resolve_prefill_lane(std::uint32_t lane, bool finished) {
        if (lanes_[lane].lifecycle != Lifecycle::Active) {
            throw std::logic_error("prefill resolution no longer matches Program state");
        }
        if (finished) { clear(lane); }
    }

    rt::BatchedGeneratedRound decode_batch(std::span<const std::uint32_t> lanes,
                                           std::span<const rt::RoundBudget>) {
        batch_.clear();
        for (const std::uint32_t lane : lanes) {
            Lane& state = lanes_[lane];
            if (state.lifecycle != Lifecycle::Active) {
                throw std::logic_error("decode batch membership no longer matches Program state");
            }
            if (state.script.throw_at_decode_index >= 0 &&
                state.emitted == static_cast<std::size_t>(state.script.throw_at_decode_index)) {
                throw std::runtime_error("scripted device failure in decode_batch");
            }
            batch_.push_back(license(lane));
            // The round is now OWED a resolution for every one of its rows.
            state.lifecycle = Lifecycle::Pending;
        }
        maximum_batch_rows_ = std::max(maximum_batch_rows_, lanes.size());
        return rt::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(batch_.data(), batch_.size()),
            .row_counts = {},
            .row_stride = 1,
        };
    }

    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled) {
        if (accepted.size() != lanes.size() || terminal.size() != lanes.size() ||
            cancelled.size() != lanes.size()) {
            throw std::logic_error("pending batch resolution has inconsistent membership");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            // THE precondition the real Program enforces (ProgramImplCore::resolve_pending_batch:
            // "ordinary pending batch no longer matches Program state"). It is what forces the
            // executor to resolve the WHOLE round before tearing any lane down: aborting a failed
            // row first would leave that lane empty here and fail the round for every request in
            // it. Without this check the fake would happily accept the wrong ordering.
            if (lanes_[lanes[row]].lifecycle != Lifecycle::Pending) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            // Mirrors the real Program: a cancelled row clears the lane outright; an ordinary row
            // returns to Active unless it finished.
            lanes_[lanes[row]].lifecycle = Lifecycle::Active;
            if (cancelled[row] != 0 || terminal[row] != 0) { clear(lanes[row]); }
        }
    }

    void abort_lane(std::uint32_t lane) noexcept {
        if (lane < max_concurrency_) { clear(lane); }
    }

    // The fake does not model retained continuation state, so a cancelled lane is cleared for
    // both entry points; the executor contract only requires that the call is made.
    void retain_cancelled_lane(std::uint32_t lane) noexcept {
        if (lane < max_concurrency_) { clear(lane); }
    }

    [[nodiscard]] bool has_retained_lane(std::uint32_t) const noexcept { return false; }
    void evict_retained_lane(std::uint32_t) noexcept {}
    [[nodiscard]] ninfer::GenerationTimings generation_timings_lane(std::uint32_t) const noexcept {
        return {};
    }
    [[nodiscard]] ninfer::SpeculativeStats speculative_stats_lane(std::uint32_t) const noexcept {
        return {};
    }
    [[nodiscard]] ninfer::MemorySummary memory_summary() const noexcept { return {}; }
    void reset_memory_peaks() noexcept {}
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return {};
    }
    void enable_logits_capture(bool) noexcept {}

    // True once every lane is idle again -- the invariant a request-scope failure must preserve.
    [[nodiscard]] bool all_lanes_idle() const noexcept {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (lanes_[lane].lifecycle != Lifecycle::Empty) { return false; }
        }
        return true;
    }

    // Widest decode round this Program actually saw. A multi-row isolation test that never
    // produced a multi-row round would be asserting nothing.
    [[nodiscard]] std::size_t maximum_batch_rows() const noexcept { return maximum_batch_rows_; }

private:
    enum class Lifecycle : std::uint8_t {
        Empty,   // no request
        Active,  // owns a sequence, no round outstanding
        Pending, // a decode round licensed tokens and owes this lane a resolution
    };

    struct Lane {
        GenerationScript script;
        std::size_t emitted        = 0;
        TokenId token              = 0;
        std::uint32_t prompt_tokens = 0;
        std::uint32_t prefill_done  = 0;
        Lifecycle lifecycle        = Lifecycle::Empty;
    };

    // One bounded prefill step. Earlier steps report progress without licensing; the final step
    // licenses the first generated token and completes, exactly as the one-step fixture always did.
    rt::PrefillStepResult prefill_step(std::uint32_t lane) {
        Lane& state = lanes_[lane];
        if (state.script.prefill_step_delay.count() > 0) {
            std::this_thread::sleep_for(state.script.prefill_step_delay);
        }
        ++state.prefill_done;
        if (state.prefill_done < state.script.prefill_steps) {
            return rt::PrefillStepResult{
                .summary                 = rt::BeginSummary{.prompt_tokens = state.prompt_tokens},
                .round                   = rt::GeneratedRound{},
                .processed_prompt_tokens = 1,
                .complete                = false,
            };
        }
        state.token = license(lane);
        return rt::PrefillStepResult{
            .summary = rt::BeginSummary{.prompt_tokens = state.prompt_tokens,
                                        .reused_prompt_tokens = 0,
                                        .prefix_reuse_path = ninfer::PrefixReusePath::FullReset},
            .round   = rt::GeneratedRound{.tokens = std::span<const TokenId>(&state.token, 1)},
            .processed_prompt_tokens = 1,
            .complete                = true,
        };
    }

    TokenId license(std::uint32_t lane) {
        Lane& state = lanes_[lane];
        if (state.emitted >= state.script.tokens.size()) {
            throw std::logic_error("scripted lane licensed more tokens than its script holds");
        }
        state.token = state.script.tokens[state.emitted++];
        return state.token;
    }

    void clear(std::uint32_t lane) noexcept {
        lanes_[lane].lifecycle = Lifecycle::Empty;
        lanes_[lane].emitted   = 0;
    }

    const std::uint32_t max_concurrency_;
    std::array<Lane, ninfer::kMaximumConcurrency> lanes_{};
    std::deque<GenerationScript> queued_;
    std::vector<TokenId> batch_;
    std::size_t maximum_batch_rows_ = 0;
    std::uint64_t service_allowance_ = 0;
};

struct ScriptedPackage {
    using Program         = ScriptedProgram;
    using RequestBasePlan = ScriptedPlan;
    using RequestPlan     = ScriptedPlan;
};

struct ScriptedRequestMemory {
    void activate(std::size_t, std::size_t) {}
    void deactivate() noexcept {}
    [[nodiscard]] rt::TransientRegion region() const noexcept { return {}; }
    [[nodiscard]] ninfer::ArenaMemorySummary summary() const noexcept { return {}; }
    void reset_peak() noexcept {}
};

struct ScriptedLoaded {
    q36::Frontend frontend;
};

struct ScriptedInstance {
    using Package = ScriptedPackage;

    std::shared_ptr<ScriptedLoaded> loaded;
    std::unique_ptr<ScriptedProgram> program;
    ScriptedRequestMemory request_memory;
    rt::KvCapacityResolution kv_capacity_resolution;
};

using Executor = rt::ConcurrentExecutor<ScriptedInstance>;

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

struct Outcome {
    bool ok = false;
    std::string content;
    std::string error;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::None;
};

rt::ResolvedRequestOptions request_options(std::uint32_t output_tokens) {
    rt::ResolvedRequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    return options;
}

Executor::Submission submit_request(Executor& executor, const q36::Frontend& frontend,
                                    std::uint32_t output_tokens) {
    q36::PreparedPrompt prompt          = frontend.prepare_tokens({0});
    const ninfer::PromptSummary summary = prompt.summary();
    return executor.submit(std::move(prompt), summary, 0.0, request_options(output_tokens));
}

Outcome await(Executor::Submission submission) {
    Outcome outcome;
    try {
        const ninfer::GenerationResult result =
            submission.wait(nullptr, ninfer::CancellationView{});
        outcome.ok            = true;
        outcome.content       = result.content;
        outcome.finish_reason = result.finish_reason;
    } catch (const std::exception& error) { outcome.error = error.what(); }
    return outcome;
}

Outcome run_request(Executor& executor, const q36::Frontend& frontend,
                    std::uint32_t output_tokens) {
    try {
        return await(submit_request(executor, frontend, output_tokens));
    } catch (const std::exception& error) {
        Outcome outcome;
        outcome.error = error.what();
        return outcome;
    }
}

ninfer::EngineOptions executor_options(std::uint32_t max_concurrency = 1,
                                       std::uint32_t pending_timeout_ms = 30'000) {
    ninfer::EngineOptions options;
    options.max_concurrency      = max_concurrency;
    options.max_pending_requests = 4;
    options.pending_timeout_ms   = pending_timeout_ms;
    return options;
}

constexpr std::string_view kReplacement = "\xef\xbf\xbd";

std::string replaced(int count) {
    std::string text;
    for (int index = 0; index < count; ++index) { text.append(kReplacement); }
    return text;
}

// ---------------------------------------------------------------------------------------------
// Decoder conformance, fed through the family OutputSession exactly as the Engine feeds it: one
// generated token at a time. The sibling cases in tests/targets/qwen3_6/test_frontend.cpp cannot
// execute in every environment (that binary also reads an upstream author's local HF checkpoint
// path), and these are the cases that guard the deleted code-point range check, so they live here
// too -- in a binary that runs.
// ---------------------------------------------------------------------------------------------

void test_decoder_conformance(const q36::Frontend& frontend) {
    struct Case {
        std::string_view name;
        std::vector<TokenId> tokens;
        std::string text;
    };
    // Byte tokens: 'x'=0 0x78, 10=0xE4, 11=0xB8, 12=0xAD, 13=0xC0, 14=0xF5, 15=0xE0, 16=0xED,
    // 17=0xF0, 18=0xF4, 19=0x80, 20=0x8F, 21=0x90, 22=0x9F, 23=0xA0, 24=0xBF, 25=0xF1, 26=0xE1,
    // 27=0xC2, 28='a', 29='b', 33='c', 34='d', 35=0x98.
    const std::vector<Case> cases{
        // --- well-formed, including every narrowed bound at its extreme -------------------------
        {"split code point across tokens", {10, 11, 12}, "\xe4\xb8\xad"},               // U+4E2D
        {"lowest three-byte (E0 A0 80)", {15, 23, 19}, "\xe0\xa0\x80"},                 // U+0800
        {"last before surrogates (ED 9F BF)", {16, 22, 24}, "\xed\x9f\xbf"},            // U+D7FF
        {"lowest four-byte (F0 90 80 80)", {17, 21, 19, 19}, "\xf0\x90\x80\x80"},      // U+10000
        {"highest code point (F4 8F BF BF)", {18, 20, 24, 24}, "\xf4\x8f\xbf\xbf"},    // U+10FFFF
        {"astral code point (F0 9F 98 80)", {17, 22, 35, 19}, "\xf0\x9f\x98\x80"},     // U+1F600

        // --- bytes that can never lead ---------------------------------------------------------
        {"lone continuation byte", {11}, replaced(1)},
        {"invalid lead 0xC0", {13}, replaced(1)},
        {"invalid lead 0xF5", {14}, replaced(1)},

        // --- the narrowed second-byte bounds: what the deleted range check used to catch --------
        // E0 80 80 is an overlong U+0000; the E0 lower bound (A0) rejects it at the second byte,
        // so the maximal subpart is <E0> alone and each 80 then fails on its own.
        {"overlong three-byte (E0 80 80)", {15, 19, 19}, replaced(3)},
        // ED A0 80 is the surrogate U+D800; the ED upper bound (9F) rejects it.
        {"surrogate (ED A0 80)", {16, 23, 19}, replaced(3)},
        // F0 80 is an overlong four-byte; the F0 lower bound (90) rejects it.
        {"overlong four-byte (F0 80)", {17, 19}, replaced(2)},
        // F4 90 would be U+110000; the F4 upper bound (8F) rejects it.
        {"above U+10FFFF (F4 90)", {18, 21}, replaced(2)},

        // --- truncation and resynchronization --------------------------------------------------
        {"truncated sequence then ASCII", {10, 0}, replaced(1) + "x"},
        {"truncated sequence then invalid lead", {10, 11, 13}, replaced(2)},
        {"replacement then split code point", {11, 10, 11, 12}, replaced(1) + "\xe4\xb8\xad"},

        // --- Unicode 15 table 3-8, the canonical U+FFFD substitution vector ---------------------
        // 61 F1 80 80 E1 80 C2 62 80 63 80 BF 64
        //   -> "a" FFFD FFFD FFFD "b" FFFD "c" FFFD FFFD "d"
        {"Unicode table 3-8 vector",
         {28, 25, 19, 19, 26, 19, 27, 29, 19, 33, 19, 24, 34},
         "a" + replaced(3) + "b" + replaced(1) + "c" + replaced(2) + "d"},
    };

    for (const Case& item : cases) {
        auto prompt             = frontend.prepare_tokens({0});
        auto session            = frontend.make_output_session(prompt, {});
        std::uint32_t remaining = static_cast<std::uint32_t>(item.tokens.size()) + 1;
        std::string published;
        bool threw = false;
        try {
            for (const TokenId token : item.tokens) {
                const auto decision = session.preview(std::array<TokenId, 1>{token}, remaining,
                                                      ninfer::FinishReason::OutputLimit);
                remaining -= decision.accepted_tokens;
                for (const ninfer::OutputDelta& delta : session.commit_preview()) {
                    if (delta.channel == ninfer::OutputChannel::Content) { published += delta.text; }
                }
            }
        } catch (...) { threw = true; }
        expect(!threw, "generated-token detokenization threw instead of replacing");
        if (published != item.text) {
            ++failures;
            std::cerr << "FAIL: decoder case '" << item.name << "' published " << published.size()
                      << " bytes, expected " << item.text.size() << '\n';
        }
    }

    // A budget that ends on a still-completable prefix terminalizes to exactly one U+FFFD.
    auto prompt  = frontend.prepare_tokens({0});
    auto session = frontend.make_output_session(prompt, {});
    const auto held =
        session.preview(std::array<TokenId, 1>{10}, 1, ninfer::FinishReason::OutputLimit);
    expect(held.finish_reason == ninfer::FinishReason::OutputLimit,
           "held UTF-8 prefix did not end at the token budget");
    std::string tail;
    for (const ninfer::OutputDelta& delta : session.commit_preview()) {
        if (delta.channel == ninfer::OutputChannel::Content) { tail += delta.text; }
    }
    expect(tail == replaced(1), "truncated tail did not terminalize to one replacement character");
}

// A request whose generated tokens are ill-formed UTF-8 completes with replacement characters,
// and the Engine is still serviceable afterwards. Before the boundary was split by scope, the
// first such request killed the worker thread and every later request answered "inference engine
// is unavailable".
void test_invalid_utf8_request_completes(const q36::Frontend& frontend,
                                         const ninfer::DeviceContext& device) {
    ScriptedInstance instance{.loaded  = std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                              .program = std::make_unique<ScriptedProgram>(1),
                              .request_memory         = {},
                              .kv_capacity_resolution = {}};
    // 11 = a lone 0xB8 continuation byte (prefill's own token), then a code point split across
    // three decode tokens, then a second lone continuation byte.
    instance.program->push_script(GenerationScript{.tokens = {11, 10, 11, 12, 11}});
    instance.program->push_script(GenerationScript{.tokens = {0}});
    ScriptedProgram& program = *instance.program;
    Executor executor(instance, executor_options(), device);

    const Outcome first = run_request(executor, frontend, 5);
    expect(first.ok, "ill-formed UTF-8 request did not complete");
    expect(first.content == std::string(kReplacement) + "\xe4\xb8\xad" + std::string(kReplacement),
           "ill-formed UTF-8 request published the wrong text");
    expect(first.finish_reason == ninfer::FinishReason::OutputLimit,
           "ill-formed UTF-8 request did not finish at its token budget");

    const Outcome second = run_request(executor, frontend, 1);
    expect(second.ok && second.content == "x",
           "the Engine did not serve the next request after ill-formed UTF-8");
    expect(program.all_lanes_idle(), "a lane was left active after the requests completed");
}

// A genuine request-scope failure -- here an out-of-vocabulary generated token, which the output
// session rejects -- must fail exactly that request, on both the prefill and the decode path,
// and leave the Engine serving.
void test_request_scope_failure_is_isolated(const q36::Frontend& frontend,
                                            const ninfer::DeviceContext& device,
                                            bool fail_on_prefill) {
    constexpr TokenId out_of_vocabulary = 900'000;
    ScriptedInstance instance{.loaded  = std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                              .program = std::make_unique<ScriptedProgram>(1),
                              .request_memory         = {},
                              .kv_capacity_resolution = {}};
    if (fail_on_prefill) {
        instance.program->push_script(GenerationScript{.tokens = {out_of_vocabulary}});
    } else {
        instance.program->push_script(GenerationScript{.tokens = {0, out_of_vocabulary}});
    }
    instance.program->push_script(GenerationScript{.tokens = {0}});
    ScriptedProgram& program = *instance.program;
    Executor executor(instance, executor_options(), device);

    const Outcome failed = run_request(executor, frontend, fail_on_prefill ? 1U : 2U);
    expect(!failed.ok, "an out-of-vocabulary generated token did not fail its request");
    expect(failed.error.find("outside the checkpoint vocabulary") != std::string::npos,
           "the failing request did not receive its own error");
    expect(program.all_lanes_idle(), "a request-scope failure left its lane active");

    const Outcome next = run_request(executor, frontend, 1);
    expect(next.ok && next.content == "x",
           "the Engine stopped serving after a request-scope failure");
}

// Multi-row isolation -- the reason row_errors exists. Two requests share decode rounds; one of
// them raises a request-scope failure mid-round. The OTHER row of that same batch must be
// unaffected: its text stays complete and exact, and it finishes normally.
void test_multi_row_isolation(const q36::Frontend& frontend,
                              const ninfer::DeviceContext& device) {
    constexpr TokenId out_of_vocabulary = 900'000;
    constexpr int kCodePoints           = 8;
    ScriptedInstance instance{.loaded  = std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                              .program = std::make_unique<ScriptedProgram>(2),
                              .request_memory         = {},
                              .kv_capacity_resolution = {}};
    // Healthy neighbour: "中" (E4 B8 AD) eight times, three byte tokens each.
    std::vector<TokenId> healthy;
    std::string expected;
    for (int index = 0; index < kCodePoints; ++index) {
        healthy.insert(healthy.end(), {10, 11, 12});
        expected += "\xe4\xb8\xad";
    }
    instance.program->push_script(GenerationScript{.tokens = healthy});
    // Neighbour that fails inside a shared decode round: two good tokens, then a token the output
    // session rejects.
    instance.program->push_script(GenerationScript{.tokens = {0, 0, out_of_vocabulary}});
    ScriptedProgram& program = *instance.program;
    Executor executor(instance, executor_options(2), device);

    Outcome healthy_outcome;
    Outcome failing_outcome;
    try {
        auto healthy_submission = submit_request(executor, frontend,
                                                 static_cast<std::uint32_t>(healthy.size()));
        auto failing_submission = submit_request(executor, frontend, 3);
        healthy_outcome         = await(std::move(healthy_submission));
        failing_outcome         = await(std::move(failing_submission));
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: multi-row submission threw: " << error.what() << '\n';
        return;
    }

    expect(program.maximum_batch_rows() >= 2,
           "no decode round ever carried two rows -- multi-row isolation was not exercised");
    expect(!failing_outcome.ok, "the failing neighbour did not fail");
    expect(failing_outcome.error.find("outside the checkpoint vocabulary") != std::string::npos,
           "the failing neighbour did not receive its own error");
    expect(healthy_outcome.ok, "a neighbour's request-scope failure also failed the healthy row");
    expect(healthy_outcome.content == expected,
           "the healthy row's text was disturbed by its neighbour's failure");
    expect(healthy_outcome.finish_reason == ninfer::FinishReason::OutputLimit,
           "the healthy row did not finish at its own token budget");
    expect(program.all_lanes_idle(), "a lane was left active after a multi-row failure");
}

// Admission is gated on "no prefill owner": while one request is prefilling, the worker never
// reaches the admission stage, so a queued request can only be served once that prefill finishes.
// A single long prompt (an agent conversation with a huge reconstructed context) therefore holds
// the gate shut for its whole prefill. The deadline policy is the only defense against starving
// the queue: a bounded timeout fails the waiter, and a DISABLED timeout (0) lets it wait and then
// complete. Both halves are asserted here with a scripted multi-step prefill.
void test_admission_deadline_expiry_and_disable(const q36::Frontend& frontend,
                                                const ninfer::DeviceContext& device) {
    constexpr std::uint32_t kSlowPrefillSteps = 6;
    constexpr auto kSlowPrefillStepDelay      = std::chrono::milliseconds(25);

    const auto run = [&](std::uint32_t pending_timeout_ms) {
        ScriptedInstance instance{.loaded =
                                      std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                                  .program                = std::make_unique<ScriptedProgram>(2),
                                  .request_memory         = {},
                                  .kv_capacity_resolution = {}};
        instance.program->set_service_allowance(kSlowPrefillSteps);
        // A: holds the prefill owner across six slow steps, then decodes two more tokens.
        instance.program->push_script(GenerationScript{.tokens = {0, 0, 0},
                                                       .prefill_steps = kSlowPrefillSteps,
                                                       .prefill_step_delay = kSlowPrefillStepDelay});
        // B: single-step prefill; admissible only after A's prefill owner clears.
        instance.program->push_script(GenerationScript{.tokens = {28, 29}});
        ScriptedProgram& program = *instance.program;
        Executor executor(instance, executor_options(2, pending_timeout_ms), device);

        auto first  = submit_request(executor, frontend, 3);
        auto second = submit_request(executor, frontend, 2);
        const Outcome second_outcome = await(std::move(second));
        const Outcome first_outcome  = await(std::move(first));
        expect(program.all_lanes_idle(), "an admission-deadline case left a lane active");
        return std::pair<Outcome, Outcome>{first_outcome, second_outcome};
    };

    // Bounded deadline (40 ms) is far shorter than A's ~150 ms prefill, so the waiter expires.
    const auto bounded = run(40);
    expect(bounded.first.ok && bounded.first.content == "xxx",
           "the prefilling request was disturbed by the queued request's timeout");
    expect(!bounded.second.ok, "a request behind a long prefill did not hit its admission deadline");
    expect(bounded.second.error.find("expired while waiting for admission") != std::string::npos,
           "the expired waiter did not report an admission queue timeout");

    // Disabled deadline (0): the same waiter must survive the prefill and complete normally.
    const auto disabled = run(0);
    expect(disabled.first.ok && disabled.first.content == "xxx",
           "the prefilling request failed while the waiter was unbounded");
    expect(disabled.second.ok && disabled.second.content == "ab",
           "a request with a disabled admission deadline did not complete after the prefill");
    expect(disabled.second.finish_reason == ninfer::FinishReason::OutputLimit,
           "the disabled-deadline waiter did not finish at its own token budget");
}

// The other half of the contract: a failure of the shared execution unit still fails every
// request and takes the Engine out of service. Narrowing the boundary must not weaken this.
void test_engine_scope_failure_still_fails_all(const q36::Frontend& frontend,
                                               const ninfer::DeviceContext& device) {
    ScriptedInstance instance{.loaded  = std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                              .program = std::make_unique<ScriptedProgram>(1),
                              .request_memory         = {},
                              .kv_capacity_resolution = {}};
    instance.program->push_script(
        GenerationScript{.tokens = {0, 0}, .throw_at_decode_index = 1});
    instance.program->push_script(GenerationScript{.tokens = {0}});
    Executor executor(instance, executor_options(), device);

    const Outcome failed = run_request(executor, frontend, 2);
    expect(!failed.ok, "a device failure did not fail the in-flight request");
    expect(failed.error.find("scripted device failure") != std::string::npos,
           "the in-flight request did not receive the device failure");

    const Outcome next = run_request(executor, frontend, 1);
    expect(!next.ok, "the Engine kept serving after a device failure");
    expect(next.error.find("unavailable") != std::string::npos,
           "a request after a device failure was not refused as unavailable");
}

// A request owning the prefill lane publishes its token progress in the runtime-stats snapshot:
// a watcher sees the prompt length, watches the processed count climb toward it, then both clear
// to zero once the request leaves the lane. This is the readout a server uses to show a long
// prompt's prefill advancing instead of a bare "prefilling=1".
//
// The scripted prefill consumes exactly one prompt token per step, so the prompt is made as long
// as the step count. Progress is then physical: it must climb to (prompt - 1) -- the final step
// clears the lane before its completion publish, so equality is never observed -- and must never
// exceed the prompt length.
void test_prefill_progress_is_published(const q36::Frontend& frontend,
                                        const ninfer::DeviceContext& device) {
    constexpr std::uint32_t kPromptTokens = 20;
    constexpr auto kStepDelay             = std::chrono::milliseconds(15);

    ScriptedInstance instance{.loaded  = std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                              .program = std::make_unique<ScriptedProgram>(1),
                              .request_memory         = {},
                              .kv_capacity_resolution = {}};
    instance.program->set_service_allowance(kPromptTokens);
    instance.program->push_script(GenerationScript{.tokens             = {0, 0, 0},
                                                   .prefill_steps      = kPromptTokens,
                                                   .prefill_step_delay = kStepDelay});
    Executor executor(instance, executor_options(), device);

    q36::PreparedPrompt prompt         = frontend.prepare_tokens(std::vector<TokenId>(kPromptTokens, 0));
    const ninfer::PromptSummary summary = prompt.summary();
    const std::uint32_t prompt_tokens   = summary.prompt_tokens;
    auto submission = executor.submit(std::move(prompt), summary, 0.0, request_options(3));

    // Sample the lane until the request is admitted and then has left it.
    std::uint32_t observed_prompt = 0;
    std::uint32_t max_processed   = 0;
    std::vector<std::uint32_t> processed;
    bool saw_prefill       = false;
    bool saw_cleared_after = false;
    for (int guard = 0; guard < 8000; ++guard) {
        const ninfer::RuntimeStats stats = executor.runtime_stats();
        if (stats.prefilling_requests != 0) {
            saw_prefill     = true;
            observed_prompt = stats.prefill_prompt_tokens;
            if (stats.prefill_processed_tokens > max_processed) {
                max_processed = stats.prefill_processed_tokens;
            }
            processed.push_back(stats.prefill_processed_tokens);
        } else if (saw_prefill) {
            saw_cleared_after = stats.prefill_prompt_tokens == 0 &&
                                stats.prefill_processed_tokens == 0;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const Outcome outcome = await(std::move(submission));

    expect(outcome.ok, "the prefill-progress sampling request did not complete");
    expect(saw_prefill, "the prefill lane was never observed active");
    expect(prompt_tokens == kPromptTokens,
           "the scripted prompt length did not match the prefill step count");
    expect(observed_prompt == prompt_tokens,
           "the published prefill prompt length did not match the request summary");
    bool non_decreasing = true;
    for (std::size_t i = 1; i < processed.size(); ++i) {
        if (processed[i] < processed[i - 1]) { non_decreasing = false; }
    }
    expect(processed.size() >= 2, "too few prefill progress samples were captured");
    expect(non_decreasing, "published prefill progress moved backwards");
    expect(max_processed <= prompt_tokens, "published prefill progress exceeded the prompt length");
    expect(max_processed + 1 >= prompt_tokens,
           "published prefill progress stalled short of the prompt; final observable step missed");
    expect(saw_cleared_after, "prefill progress was not cleared once the request left the lane");
}

// A streaming consumer must receive periodic liveness pings while a request produces no output,
// so a multi-minute prefill does not leave the transport silent (and idle-dropped). The pings are
// driven from the same consumer thread that owns the sink, so this fixture observes them directly:
// the count at the first publish is proof a ping fired *before* any content, i.e. during prefill.
struct CountingSink final : ninfer::OutputSink {
    void publish(ninfer::OutputDelta delta) override {
        if (!saw_publish) {
            saw_publish                 = true;
            keepalives_at_first_publish = keepalives;
        }
        bytes += delta.text.size();
    }
    void keepalive() override { ++keepalives; }

    std::size_t bytes = 0;
    int keepalives    = 0;
    bool saw_publish  = false;
    int keepalives_at_first_publish = 0;
};

ninfer::EngineOptions keepalive_options(std::uint32_t interval_ms) {
    ninfer::EngineOptions options       = executor_options();
    options.stream_keepalive_ms         = interval_ms;
    return options;
}

// Shared fixture: a prefill long enough (kPromptTokens steps spaced by step_delay) that a
// keepalive interval of a fraction of that duration must fire several times.
ScriptedInstance long_prefill_instance(const q36::Frontend& frontend, std::uint32_t prompt_tokens,
                                       std::chrono::milliseconds step_delay) {
    ScriptedInstance instance{.loaded  = std::make_shared<ScriptedLoaded>(ScriptedLoaded{frontend}),
                              .program = std::make_unique<ScriptedProgram>(1),
                              .request_memory         = {},
                              .kv_capacity_resolution = {}};
    instance.program->set_service_allowance(prompt_tokens);
    instance.program->push_script(GenerationScript{.tokens             = {0, 0, 0},
                                                   .prefill_steps      = prompt_tokens,
                                                   .prefill_step_delay = step_delay});
    return instance;
}

ninfer::GenerationResult run_with_sink(Executor& executor, const q36::Frontend& frontend,
                                       std::uint32_t prompt_tokens, CountingSink& sink) {
    q36::PreparedPrompt prompt          = frontend.prepare_tokens(std::vector<TokenId>(prompt_tokens, 0));
    const ninfer::PromptSummary summary = prompt.summary();
    auto submission = executor.submit(std::move(prompt), summary, 0.0, request_options(3));
    return submission.wait(&sink, ninfer::CancellationView{});
}

void test_stream_keepalive_fires_during_prefill(const q36::Frontend& frontend,
                                                const ninfer::DeviceContext& device) {
    constexpr std::uint32_t kPromptTokens        = 20;
    constexpr auto kStepDelay                    = std::chrono::milliseconds(15);
    constexpr std::uint32_t kKeepaliveIntervalMs = 50;

    ScriptedInstance instance = long_prefill_instance(frontend, kPromptTokens, kStepDelay);
    Executor executor(instance, keepalive_options(kKeepaliveIntervalMs), device);

    CountingSink sink;
    const ninfer::GenerationResult result = run_with_sink(executor, frontend, kPromptTokens, sink);

    expect(result.generated_token_ids.size() == 3,
           "the keepalive request produced the wrong number of tokens");
    expect(sink.saw_publish, "the keepalive request published no content");
    expect(sink.keepalives_at_first_publish > 0,
           "no keepalive was emitted before the first token: a long prefill stayed silent");
}

void test_stream_keepalive_disabled(const q36::Frontend& frontend,
                                    const ninfer::DeviceContext& device) {
    constexpr std::uint32_t kPromptTokens = 20;
    constexpr auto kStepDelay             = std::chrono::milliseconds(15);

    ScriptedInstance instance = long_prefill_instance(frontend, kPromptTokens, kStepDelay);
    Executor executor(instance, keepalive_options(0), device);

    CountingSink sink;
    const ninfer::GenerationResult result = run_with_sink(executor, frontend, kPromptTokens, sink);

    expect(result.generated_token_ids.size() == 3,
           "the disabled-keepalive request produced the wrong number of tokens");
    expect(sink.keepalives == 0,
           "a keepalive was emitted even though the interval was disabled");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "no CUDA device: skipping\n";
        return 77;
    }
    const ninfer::DeviceContext device(0);
    const q36::Frontend frontend = q36::FrontendTestAccess::create_component(resources(), false);

    test_decoder_conformance(frontend);
    test_invalid_utf8_request_completes(frontend, device);
    test_multi_row_isolation(frontend, device);
    test_request_scope_failure_is_isolated(frontend, device, true);
    test_request_scope_failure_is_isolated(frontend, device, false);
    test_admission_deadline_expiry_and_disable(frontend, device);
    test_engine_scope_failure_still_fails_all(frontend, device);
    test_prefill_progress_is_published(frontend, device);
    test_stream_keepalive_fires_during_prefill(frontend, device);
    test_stream_keepalive_disabled(frontend, device);
    return failures == 0 ? 0 : 1;
}
