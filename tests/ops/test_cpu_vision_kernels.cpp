// Host Vision encoder kernels: the optimized AVX2 paths must agree with their naive definitions.
// These are the pieces a real image exercises most, and they are pure host code, so they can be
// pinned without a device or an artifact.

#define NINFER_QWEN36_RUNTIME_NS test_cpu_vision_runtime
#include "targets/qwen3_6/impl/runtime/cpu_vision_detail.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace cv = ninfer::targets::qwen3_6::detail::test_cpu_vision_runtime::schedule::
    cpu_vision_detail;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::printf("FAIL: %s\n", message);
    }
}

constexpr int kHidden = 1152;
constexpr int kHeadDim = 72;
constexpr int kHeads = 16;

void reference_attention_head(const float* qkv, int begin, int end, std::size_t head,
                              float* out) {
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> scores(static_cast<std::size_t>(end - begin));
    for (int p = begin; p < end; ++p) {
        const float* q = qkv + static_cast<std::size_t>(p) * 3 * kHidden + head * kHeadDim;
        float max_score = -INFINITY;
        for (int j = begin; j < end; ++j) {
            const float* k = qkv + static_cast<std::size_t>(j) * 3 * kHidden + kHidden +
                             head * kHeadDim;
            float dot = 0.0F;
            for (int d = 0; d < kHeadDim; ++d) { dot += q[d] * k[d]; }
            dot *= scale;
            scores[static_cast<std::size_t>(j - begin)] = dot;
            max_score = std::max(max_score, dot);
        }
        float sum = 0.0F;
        for (int j = begin; j < end; ++j) {
            float& value = scores[static_cast<std::size_t>(j - begin)];
            value        = std::exp(value - max_score);
            sum += value;
        }
        float* dst = out + static_cast<std::size_t>(p) * kHidden + head * kHeadDim;
        for (int d = 0; d < kHeadDim; ++d) { dst[d] = 0.0F; }
        for (int j = begin; j < end; ++j) {
            const float w = scores[static_cast<std::size_t>(j - begin)] / sum;
            const float* v = qkv + static_cast<std::size_t>(j) * 3 * kHidden + 2 * kHidden +
                             head * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) { dst[d] += w * v[d]; }
        }
    }
}

void check_attention() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) { return; }
    const int patches = 37;
    std::vector<float> qkv(static_cast<std::size_t>(3) * kHidden * patches);
    for (std::size_t i = 0; i < qkv.size(); ++i) {
        qkv[i] = std::sin(static_cast<float>(i) * 0.011F);
    }
    std::vector<float> fast(static_cast<std::size_t>(kHidden) * patches);
    std::vector<float> slow(fast.size());
    double max_error = 0.0;
    // Two segments exercise the per-segment boundary handling.
    for (const auto& bounds : {std::pair<int, int>{0, 20}, std::pair<int, int>{20, patches}}) {
        for (int head = 0; head < kHeads; ++head) {
            cv::attention_head(qkv.data(), bounds.first, bounds.second, kHidden, kHeadDim, head,
                               fast.data());
            reference_attention_head(qkv.data(), bounds.first, bounds.second, head, slow.data());
            for (int p = bounds.first; p < bounds.second; ++p) {
                const std::size_t base = static_cast<std::size_t>(p) * kHidden +
                                         static_cast<std::size_t>(head) * kHeadDim;
                for (int d = 0; d < kHeadDim; ++d) {
                    max_error = std::max(max_error, std::fabs(double(fast[base + d]) -
                                                             slow[base + d]));
                }
            }
        }
    }
    check(max_error < 1e-5, "AVX2 attention disagrees with the scalar reference");
#endif
}

void check_bias_gelu() {
    const int width  = 4304;
    const int tokens = 9;
    std::vector<float> bias(width);
    std::vector<float> data(static_cast<std::size_t>(width) * tokens);
    for (int c = 0; c < width; ++c) { bias[c] = std::sin(static_cast<float>(c) * 0.017F); }
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = std::cos(static_cast<float>(i) * 0.013F) * 6.0F;
    }
    std::vector<float> expected = data;
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < width; ++c) {
            const std::size_t index = static_cast<std::size_t>(t) * width + c;
            expected[index] = cv::gelu_tanh_scalar(expected[index] + bias[c]);
        }
    }
    cv::bias_gelu_tanh(data.data(), bias.data(), width, tokens);
    double max_error = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        max_error = std::max(max_error, std::fabs(double(data[i]) - expected[i]));
    }
    check(max_error < 1e-5, "vectorized bias+GELU(tanh) disagrees with the scalar formula");
}

void check_gemm() {
    const int rows   = 33;
    const int cols   = 1152;
    const int tokens = 11;
    std::vector<float> weight(static_cast<std::size_t>(rows) * cols);
    std::vector<float> input(static_cast<std::size_t>(cols) * tokens);
    for (std::size_t i = 0; i < weight.size(); ++i) {
        weight[i] = std::sin(static_cast<float>(i) * 0.001F);
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.002F);
    }
    std::vector<float> expected(static_cast<std::size_t>(rows) * tokens);
    for (int t = 0; t < tokens; ++t) {
        const float* x = input.data() + static_cast<std::size_t>(t) * cols;
        float* y       = expected.data() + static_cast<std::size_t>(t) * rows;
        for (int r = 0; r < rows; ++r) {
            const float* w = weight.data() + static_cast<std::size_t>(r) * cols;
            float acc      = 0.0F;
            for (int c = 0; c < cols; ++c) { acc += w[c] * x[c]; }
            y[r] = acc;
        }
    }
    std::vector<float> out(expected.size());
    cv::gemm_rows_generic(weight.data(), input.data(), rows, cols, tokens, out.data(), 0, rows);
    double max_error = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        max_error = std::max(max_error, std::fabs(double(out[i]) - expected[i]));
    }
    check(max_error < 1e-4, "row-blocked GEMM disagrees with the naive reference");
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        std::fill(out.begin(), out.end(), 0.0F);
        cv::gemm_rows_avx2(weight.data(), input.data(), rows, cols, tokens, out.data(), 0, rows);
        max_error = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            max_error = std::max(max_error, std::fabs(double(out[i]) - expected[i]));
        }
        check(max_error < 1e-3, "AVX2 GEMM disagrees with the naive reference");
    }
#endif
}

} // namespace

int main() {
    check_attention();
    check_bias_gelu();
    check_gemm();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
