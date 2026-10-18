// The CPU Vision encoder decodes the same row-split payloads the CUDA kernels receive. This test
// pins that decoder to the test-owned reference for every format the Vision backbone uses, at the
// projection widths the backbone actually runs, and covers the FP16 subnormal scale values the
// artifact converter emits for small-magnitude groups.

#define NINFER_QWEN36_RUNTIME_NS test_cpu_vision_runtime
#include "ops/quantized_weight.h"
#include "targets/qwen3_6/impl/runtime/cpu_vision_detail.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

std::vector<float> pattern(std::size_t count, float scale) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.37F) * scale;
    }
    return values;
}

void check_format(const char* name, ninfer::QType qtype, int rows, int cols, float magnitude) {
    const std::vector<float> source = pattern(static_cast<std::size_t>(rows) * cols, magnitude);
    const ninfer::test::quantized_weight::PackedWeight packed =
        ninfer::test::quantized_weight::pack_row_split_lowbit(source, rows, cols, qtype);
    cv::Matrix ours;
    cv::dequantize(packed.weight, ours);
    check(ours.rows == rows && ours.cols == cols, name);
    double max_error = 0.0;
    for (std::size_t i = 0; i < packed.dequant.size(); ++i) {
        max_error = std::max(max_error, std::fabs(double(packed.dequant[i]) - ours.data[i]));
    }
    if (max_error != 0.0) {
        std::printf("FAIL: %s rows=%d cols=%d magnitude=%g max_error=%g\n", name, rows, cols,
                    magnitude, max_error);
        ++failures;
    }
}

// Exhaustive FP16 decode against the compiler's native half type where it exists. This is the
// check that caught the subnormal exponent (127 - 15 - shift instead of 127 - 14 - shift): a
// halved subnormal scale silently halves every weight in that group.
void check_fp16_decode() {
#if defined(__FLT16_MAX__) && (defined(__x86_64__) || defined(__i386__))
    int mismatches = 0;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const auto half = static_cast<std::uint16_t>(bits);
        _Float16 native;
        std::memcpy(&native, &half, sizeof(native));
        const float expected = static_cast<float>(native);
        const float actual   = cv::f16_to_f32(half);
        if (std::isnan(expected)) {
            if (!std::isnan(actual)) { ++mismatches; }
            continue;
        }
        if (!(expected == actual)) { ++mismatches; }
    }
    check(mismatches == 0, "FP16 decode disagrees with the native half conversion");
#endif
}

} // namespace

int main() {
    using ninfer::QType;
    constexpr float kNormal = 4.0F;
    // Subnormal FP16 scales: the converter emits 2^-24..2^-14 prefixes for small groups.
    constexpr float kSubnormal = 1.0e-6F;
    for (const float magnitude : {kNormal, kSubnormal}) {
        check_format("q4 patch/patch", QType::Q4G64_F16S, 24, 256, magnitude);
        check_format("q5 attention/output", QType::Q5G64_F16S, 24, 256, magnitude);
        check_format("q6 patch_embedding", QType::Q6G64_F16S, 16, 320, magnitude);
        check_format("w8 merger", QType::W8G32_F16S, 16, 320, magnitude);
        // The MLP input width is 1152 and the fc2 input width is 4304; the latter pads to 4352,
        // so its last group is partial.
        check_format("q4 fc1 k=1152", QType::Q4G64_F16S, 8, 1152, magnitude);
        check_format("q5 fc2 k=4304", QType::Q5G64_F16S, 8, 4304, magnitude);
        check_format("w8 merger k=4608", QType::W8G32_F16S, 8, 4608, magnitude);
    }
    check_fp16_decode();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
