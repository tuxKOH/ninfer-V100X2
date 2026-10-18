#pragma once
// Host implementation of the Vision encoder, selected by --vision-device cpu.
//
// The CUDA path (VisionContext) keeps the quantized backbone in the device arena and runs every
// projection through the target's quantized GEMM kernels. This path instead retains the same
// payloads in host memory (TensorPlacement::Host), dequantizes one projection at a time, and runs
// the identical math with AVX2/FMA host threads. Only the merged embeddings leave the host, as one
// H2D copy into the caller's device transient.
//
// Numeric notes. The device path rounds every intermediate activation to BF16; this one keeps FP32
// intermediates and rounds only the final merged embeddings. The two are therefore equivalent but
// not bit-identical, which is fine for Vision: prefix-reuse identity is carried by the media digest
// and consumer spans, never by encoder output bits.

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "targets/qwen3_6/impl/runtime/cpu_vision_detail.h"

#include <ninfer/targets/qwen3_6/vision_control.h>

#include "core/device.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {


// Dequantized backbone weights, shared by every request in the process. A fresh encoder is built
// per prefill session, and without this cache each image re-decoded all 411M codes (~0.3 GB of
// payload) before the first projection. The cache costs about 1.7 GB of host memory once warm and
// is keyed by the model's weight set, not by the encoder.
struct CpuVisionWeightCache {
    struct Entry {
        cpu_vision_detail::Matrix matrix;
        bool ready = false;
    };
    std::mutex mutex;
    Entry patch_embedding;
    Entry merger_fc1;
    Entry merger_fc2;
    std::vector<Entry> qkv;
    std::vector<Entry> output;
    std::vector<Entry> fc1;
    std::vector<Entry> fc2;

    explicit CpuVisionWeightCache(std::size_t layers)
        : qkv(layers), output(layers), fc1(layers), fc2(layers) {}
};

class CpuVisionEncoder {
public:
    explicit CpuVisionEncoder(const LoadedModelData& model)
        : weights_(&require_vision(model)),
          workers_(cpu_vision_detail::worker_count()),
          gemm_rows_(cpu_vision_detail::select_gemm_rows()),
          cache_(cpu_vision_detail::weight_cache_enabled()
                     ? shared_cache(weights_)
                     : std::make_shared<CpuVisionWeightCache>(vision_layers_)) {
        if (weights_->common.layers.size() != static_cast<std::size_t>(vision_layers_)) {
            throw std::invalid_argument("CPU Vision weight set has an unexpected layer count");
        }
    }

    void encode(const schedule::VisionItemView& item, Tensor& output, cudaStream_t stream) {
        using namespace cpu_vision_detail;
        namespace sc = schedule;
        if (item.control == nullptr) { throw std::invalid_argument("CPU Vision item control is null"); }
        const qwen3_6::VisionItemControl& control = *item.control;
        const auto P = static_cast<int>(control.patch_count);
        const auto T = static_cast<int>(control.merged_count);
        constexpr int H  = sc::VisionScheduleConfig::hidden;
        constexpr int D  = sc::VisionScheduleConfig::head_dim;
        constexpr int NH = sc::VisionScheduleConfig::heads;
        constexpr int IN = sc::VisionScheduleConfig::intermediate;
        constexpr int MH = sc::VisionScheduleConfig::merger_hidden;
        constexpr int OH = sc::VisionScheduleConfig::out_hidden;
        constexpr int PD = sc::VisionScheduleConfig::patch_dim;
        constexpr float eps = sc::VisionScheduleConfig::norm_eps;
        const auto patches_expected = static_cast<std::size_t>(P) * PD;
        if (P <= 0 || T <= 0 || P != T * sc::VisionScheduleConfig::merge_unit ||
            item.patches.size() != patches_expected ||
            control.position_ids.size() != static_cast<std::size_t>(2) * P ||
            control.position_table_indices.size() != static_cast<std::size_t>(4) * P ||
            control.position_table_weights.size() != static_cast<std::size_t>(4) * P ||
            control.cu_seqlens.size() < 2) {
            throw std::invalid_argument("CPU Vision item has an inconsistent shape");
        }
        if (output.dtype != DType::BF16 || output.data == nullptr || output.ne[0] != OH ||
            output.ne[1] != T) {
            throw std::invalid_argument("CPU Vision output must be a BF16 [H,V] tensor");
        }
        const auto& common = weights_->common;

        std::vector<float> x(static_cast<std::size_t>(H) * P);
        std::vector<float> h(static_cast<std::size_t>(H) * P);
        std::vector<float> qkv(static_cast<std::size_t>(3) * H * P);
        std::vector<float> attended(static_cast<std::size_t>(H) * P);
        std::vector<float> projected(static_cast<std::size_t>(H) * P);
        std::vector<float> up(static_cast<std::size_t>(IN) * P);

        // Patch embedding: x = patch_embed * patches + bias.
        {
            const Matrix& matrix = matrix_for(common.patch_embedding, cache_->patch_embedding);
            std::vector<float> patches(static_cast<std::size_t>(PD) * P);
            for (std::size_t i = 0; i < patches.size(); ++i) {
                patches[i] = bf16_to_f32(item.patches[i]);
            }
            gemm(matrix, patches.data(), P, x.data());
            std::vector<float> bias;
            to_float(common.patch_embedding_bias, bias);
            for (int p = 0; p < P; ++p) {
                for (int c = 0; c < H; ++c) { x[c + static_cast<std::size_t>(p) * H] += bias[c]; }
            }
        }
        // Interpolated absolute position table.
        {
            constexpr int PE = sc::VisionScheduleConfig::position_embeddings;
            std::vector<float> table;
            to_float(common.position_embedding, table); // [rows, hidden] row-major
            for (int p = 0; p < P; ++p) {
                for (int corner = 0; corner < 4; ++corner) {
                    const std::size_t control_index =
                        static_cast<std::size_t>(p) * 4 + corner;
                    const std::int32_t row = control.position_table_indices[control_index];
                    const float w          = control.position_table_weights[control_index];
                    if (w == 0.0F || row < 0 || row >= PE) { continue; }
                    const float* source = table.data() + static_cast<std::size_t>(row) * H;
                    float* dest         = x.data() + static_cast<std::size_t>(p) * H;
                    for (int c = 0; c < H; ++c) { dest[c] += w * source[c]; }
                }
            }
        }

        std::vector<float> qkv_bias, out_bias, fc1_bias, fc2_bias, norm_w, norm_b;
        for (std::size_t layer = 0; layer < common.layers.size(); ++layer) {
            const auto& source = common.layers[layer];
            // Attention block: norm -> qkv -> rope -> segmented attention -> projection -> residual.
            to_float(source.norm1_weight, norm_w);
            to_float(source.norm1_bias, norm_b);
            layer_norm(x, norm_w, norm_b, H, P, eps, h);
            gemm(matrix_for(source.qkv, cache_->qkv[layer]), h.data(), P, qkv.data());
            to_float(source.qkv_bias, qkv_bias);
            for (int p = 0; p < P; ++p) {
                for (int c = 0; c < 3 * H; ++c) {
                    qkv[c + static_cast<std::size_t>(p) * 3 * H] += qkv_bias[c];
                }
            }
            rope(qkv, control.position_ids, P);
            attention(qkv, control.cu_seqlens, P, attended);
            gemm(matrix_for(source.output, cache_->output[layer]), attended.data(), P,
                 projected.data());
            to_float(source.output_bias, out_bias);
            for (int p = 0; p < P; ++p) {
                for (int c = 0; c < H; ++c) {
                    x[c + static_cast<std::size_t>(p) * H] +=
                        projected[c + static_cast<std::size_t>(p) * H] + out_bias[c];
                }
            }
            // MLP block: norm -> fc1 -> GELU(tanh) -> fc2 -> residual.
            to_float(source.norm2_weight, norm_w);
            to_float(source.norm2_bias, norm_b);
            layer_norm(x, norm_w, norm_b, H, P, eps, h);
            gemm(matrix_for(source.fc1, cache_->fc1[layer]), h.data(), P, up.data());
            to_float(source.fc1_bias, fc1_bias);
            bias_gelu_tanh(up, fc1_bias, IN, P);
            gemm(matrix_for(source.fc2, cache_->fc2[layer]), up.data(), P, projected.data());
            to_float(source.fc2_bias, fc2_bias);
            for (int p = 0; p < P; ++p) {
                for (int c = 0; c < H; ++c) {
                    x[c + static_cast<std::size_t>(p) * H] +=
                        projected[c + static_cast<std::size_t>(p) * H] + fc2_bias[c];
                }
            }
        }

        // Merger: norm -> [merger_hidden, V] view -> fc1 -> GELU(exact) -> fc2 -> output.
        to_float(weights_->common.merger_norm_weight, norm_w);
        to_float(weights_->common.merger_norm_bias, norm_b);
        layer_norm(x, norm_w, norm_b, H, P, eps, h);
        std::vector<float> merger_hidden(static_cast<std::size_t>(MH) * T);
        gemm(matrix_for(weights_->common.merger_fc1, cache_->merger_fc1), h.data(), T,
             merger_hidden.data());
        std::vector<float> merger_bias;
        to_float(weights_->common.merger_fc1_bias, merger_bias);
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < MH; ++c) {
                float& value = merger_hidden[c + static_cast<std::size_t>(t) * MH];
                value += merger_bias[c];
                value = gelu_exact(value);
            }
        }
        std::vector<float> merged(static_cast<std::size_t>(OH) * T);
        gemm(matrix_for(weights_->merger_fc2, cache_->merger_fc2), merger_hidden.data(), T,
             merged.data());
        to_float(weights_->merger_fc2_bias, merger_bias);
        output_bits_.resize(static_cast<std::size_t>(OH) * T);
        for (std::size_t i = 0; i < merged.size(); ++i) {
            output_bits_[i] = f32_to_bf16(merged[i] + merger_bias[i % static_cast<std::size_t>(OH)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data, output_bits_.data(),
                                   output_bits_.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        // The staging buffer is reused by the next item; the scatter that consumes this output
        // runs later on the same stream, so waiting here is the simplest correct ordering.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

private:
    static constexpr int vision_layers_ = schedule::VisionScheduleConfig::layers;

    static const VisionWeights& require_vision(const LoadedModelData& model) {
        if (!model.vision) {
            throw std::invalid_argument("CPU Vision requested without weights");
        }
        return *model.vision;
    }

    static std::shared_ptr<CpuVisionWeightCache> shared_cache(const VisionWeights* weights) {
        // The registry owns the cache for the process lifetime. Encoders are built and destroyed
        // with each prefill session, so a weak reference here would let the cache expire between
        // requests and silently reintroduce a full dequantization per image. There is one weight
        // set per loaded model, and the entry is keyed by its address.
        static std::mutex registry_mutex;
        static std::unordered_map<const VisionWeights*, std::shared_ptr<CpuVisionWeightCache>>
            registry;
        std::lock_guard lock(registry_mutex);
        const auto found = registry.find(weights);
        if (found != registry.end()) { return found->second; }
        auto cache        = std::make_shared<CpuVisionWeightCache>(vision_layers_);
        registry[weights] = cache;
        return cache;
    }

    // Dequantizes a projection once per process. The parallel row stripes make the first image's
    // decode cheap; every later image reads the cached FP32 matrix.
    const cpu_vision_detail::Matrix& matrix_for(const Weight& weight,
                                                CpuVisionWeightCache::Entry& cached) {
        std::lock_guard lock(cache_->mutex);
        if (!cached.ready) {
            const int rows = static_cast<int>(weight.shape[0]);
            const int cols = static_cast<int>(weight.shape[1]);
            if (rows <= 0 || cols <= 0) {
                throw std::invalid_argument("CPU Vision: empty weight");
            }
            cached.matrix.resize(rows, cols);
            // Validate geometry and format once, outside the worker threads.
            cpu_vision_detail::dequantize_rows(weight, cached.matrix, 0, 0);
            const unsigned stripes =
                std::max(1U, std::min(workers_, static_cast<unsigned>(rows)));
            cpu_vision_detail::parallel_for(stripes, stripes, [&](std::size_t stripe) {
                const int begin = static_cast<int>(static_cast<std::size_t>(rows) * stripe /
                                                   stripes);
                const int end   = static_cast<int>(static_cast<std::size_t>(rows) * (stripe + 1) /
                                                   stripes);
                cpu_vision_detail::dequantize_rows(weight, cached.matrix, begin, end);
            });
            cached.ready = true;
        }
        return cached.matrix;
    }

    void gemm(const cpu_vision_detail::Matrix& weight, const float* input, int tokens,
              float* out) const {
        const int rows = weight.rows;
        const int cols = weight.cols;
        const unsigned stripes = std::max(1U, std::min(workers_, static_cast<unsigned>(rows)));
        cpu_vision_detail::parallel_for(stripes, stripes, [&](std::size_t stripe) {
            const int begin =
                static_cast<int>(static_cast<std::size_t>(rows) * stripe / stripes);
            const int end =
                static_cast<int>(static_cast<std::size_t>(rows) * (stripe + 1) / stripes);
            gemm_rows_(weight.data.data(), input, rows, cols, tokens, out, begin, end);
        });
    }

    void layer_norm(const std::vector<float>& x, const std::vector<float>& weight,
                    const std::vector<float>& bias, int hidden, int tokens, float eps,
                    std::vector<float>& out) const {
        cpu_vision_detail::parallel_for(static_cast<std::size_t>(tokens), workers_,
                                        [&](std::size_t token) {
            normalise_token(x.data() + token * static_cast<std::size_t>(hidden),
                            weight.data(), bias.data(), hidden, eps,
                            out.data() + token * static_cast<std::size_t>(hidden));
        });
    }

    static void normalise_token(const float* src, const float* weight, const float* bias,
                                int hidden, float eps, float* dst) {
        double sum = 0.0;
        for (int c = 0; c < hidden; ++c) { sum += src[c]; }
        const float mean = static_cast<float>(sum / hidden);
        double m2 = 0.0;
        for (int c = 0; c < hidden; ++c) {
            const double d = src[c] - mean;
            m2 += d * d;
        }
        const float inv = 1.0F / std::sqrt(static_cast<float>(m2 / hidden) + eps);
        for (int c = 0; c < hidden; ++c) {
            dst[c] = (src[c] - mean) * inv * weight[c] + bias[c];
        }
    }

    // bias + GELU(tanh) over a [tokens, width] buffer, the hot tail of every MLP block.
    static void bias_gelu_tanh(std::vector<float>& data, const std::vector<float>& bias,
                               int width, int tokens) {
        cpu_vision_detail::bias_gelu_tanh(data.data(), bias.data(), width, tokens);
    }

    static float gelu_exact(float value) {
        return 0.5F * value * (1.0F + std::erf(value * 0.70710678118654752440F));
    }

    // 2D rotary over q and k, head_dim 72: the first 36 and second 36 halves rotate pairwise with
    // the CUDA Vision2D convention (pairs 0..17 on the height axis, 18..35 on the width axis).
    static void rope(std::vector<float>& qkv, const std::vector<std::int32_t>& positions, int P) {
        namespace sc = schedule;
        constexpr int D  = sc::VisionScheduleConfig::head_dim;
        constexpr int NH = sc::VisionScheduleConfig::heads;
        constexpr int H  = sc::VisionScheduleConfig::hidden;
        for (int p = 0; p < P; ++p) {
            float cos_table[36];
            float sin_table[36];
            for (int pair = 0; pair < 36; ++pair) {
                const int axis = pair / 18;
                const int pos  = positions[static_cast<std::size_t>(axis) * P + p];
                const float angle =
                    static_cast<float>(pos) * cpu_vision_detail::kVisionInvFrequency[pair % 18];
                cos_table[pair] = std::cos(angle);
                sin_table[pair] = std::sin(angle);
            }
            float* q = qkv.data() + static_cast<std::size_t>(p) * 3 * H;
            float* k = q + H;
            for (int head = 0; head < NH; ++head) {
                for (int half = 0; half < 2; ++half) {
                    float* vector = half == 0 ? q : k;
                    float* base   = vector + head * D;
                    for (int lane = 0; lane < 18; ++lane) {
                        const int i0 = 2 * lane;
                        const int i1 = 2 * lane + 1;
                        const int j0 = i0 + 36;
                        const int j1 = i1 + 36;
                        const float a0 = base[i0];
                        const float b0 = base[j0];
                        const float a1 = base[i1];
                        const float b1 = base[j1];
                        base[i0] = a0 * cos_table[i0] - b0 * sin_table[i0];
                        base[j0] = b0 * cos_table[i0] + a0 * sin_table[i0];
                        base[i1] = a1 * cos_table[i1] - b1 * sin_table[i1];
                        base[j1] = b1 * cos_table[i1] + a1 * sin_table[i1];
                    }
                }
            }
        }
    }

    void attention(const std::vector<float>& qkv, const std::vector<std::int32_t>& cu_seqlens,
                   int P, std::vector<float>& out) const {
        namespace sc = schedule;
        constexpr int D  = sc::VisionScheduleConfig::head_dim;
        constexpr int NH = sc::VisionScheduleConfig::heads;
        constexpr int H  = sc::VisionScheduleConfig::hidden;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        __builtin_cpu_init();
        if (D == 72 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            attention_avx2(qkv, cu_seqlens, P, NH, H, D, out);
            return;
        }
#endif
        const float scale = 1.0F / std::sqrt(static_cast<float>(D));
        const std::size_t segments = cu_seqlens.size() > 0 ? cu_seqlens.size() - 1 : 0;
        for (std::size_t segment = 0; segment < segments; ++segment) {
            const int begin = cu_seqlens[segment];
            const int end   = cu_seqlens[segment + 1];
            if (begin < 0 || end <= begin || end > P) { continue; }
            const int length = end - begin;
            // Parallelize over heads; each head owns its own queries, score scratch and output.
            cpu_vision_detail::parallel_for(static_cast<std::size_t>(NH), workers_, [&](std::size_t head) {
                std::vector<float> scores(static_cast<std::size_t>(length));
                for (int p = begin; p < end; ++p) {
                    const float* q = qkv.data() + static_cast<std::size_t>(p) * 3 * H +
                                     static_cast<std::size_t>(head) * D;
                    float max_score = -INFINITY;
                    for (int j = begin; j < end; ++j) {
                        const float* k = qkv.data() + static_cast<std::size_t>(j) * 3 * H + H +
                                         static_cast<std::size_t>(head) * D;
                        float dot = 0.0F;
                        for (int d = 0; d < D; ++d) { dot += q[d] * k[d]; }
                        dot *= scale;
                        scores[static_cast<std::size_t>(j - begin)] = dot;
                        max_score = std::max(max_score, dot);
                    }
                    float sum = 0.0F;
                    for (int j = begin; j < end; ++j) {
                        float& score = scores[static_cast<std::size_t>(j - begin)];
                        score        = std::exp(score - max_score);
                        sum += score;
                    }
                    const float inv = 1.0F / sum;
                    float* dst = out.data() + static_cast<std::size_t>(p) * H +
                                 static_cast<std::size_t>(head) * D;
                    for (int d = 0; d < D; ++d) { dst[d] = 0.0F; }
                    for (int j = begin; j < end; ++j) {
                        const float weight = scores[static_cast<std::size_t>(j - begin)] * inv;
                        const float* v = qkv.data() + static_cast<std::size_t>(j) * 3 * H + 2 * H +
                                         static_cast<std::size_t>(head) * D;
                        for (int d = 0; d < D; ++d) { dst[d] += weight * v[d]; }
                    }
                }
            });
        }
    }

    const VisionWeights* weights_                  = nullptr;
    unsigned workers_                              = 1;
    cpu_vision_detail::GemmRowRange gemm_rows_     = nullptr;
    std::shared_ptr<CpuVisionWeightCache> cache_;
    std::vector<std::uint16_t> output_bits_;

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    void attention_avx2(const std::vector<float>& qkv,
                        const std::vector<std::int32_t>& cu_seqlens, int P, int NH, int H,
                        int D, std::vector<float>& out) const {
        const std::size_t segments = cu_seqlens.size() > 1 ? cu_seqlens.size() - 1 : 0;
        for (std::size_t segment = 0; segment < segments; ++segment) {
            const int begin = cu_seqlens[segment];
            const int end   = cu_seqlens[segment + 1];
            if (begin < 0 || end <= begin || end > P) { continue; }
            cpu_vision_detail::parallel_for(
                static_cast<std::size_t>(NH), workers_, [&](std::size_t head) {
                    cpu_vision_detail::attention_head(qkv.data(), begin, end, H, D, head,
                                                      out.data());
                });
        }
    }
#endif
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
