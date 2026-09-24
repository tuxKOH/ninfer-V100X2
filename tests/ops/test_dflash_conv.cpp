#include "ninfer/ops/dflash_conv.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int H = 5120;
constexpr int G = H / 16;
constexpr int D = G * 4;

int run_case(int width, int batch, int side) {
    std::vector<float> hidden(static_cast<std::size_t>(H) * width * batch);
    std::vector<float> dynamic(static_cast<std::size_t>(D) * width * batch);
    std::vector<float> base(static_cast<std::size_t>(H) * 4);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
    }
    for (std::size_t i = 0; i < dynamic.size(); ++i) {
        dynamic[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
    }
    for (std::size_t i = 0; i < base.size(); ++i) {
        base[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f;
    }
    round_to_bf16(hidden);
    round_to_bf16(dynamic);
    round_to_bf16(base);
    std::vector<std::uint16_t> expected(hidden.size());
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < width; ++t) {
            for (int c = 0; c < H; ++c) {
                double sum = 0.0;
                for (int tap = 0; tap < 2; ++tap) {
                    if (t < tap) { continue; }
                    const int group = c / 16;
                    const int coefficient = group + tap * G + side * G * 2 + (b * width + t) * D;
                    const int base_index = c + tap * H + side * H * 2;
                    const int input = (b * width + t - tap) * H + c;
                    sum += (static_cast<double>(base[base_index]) + dynamic[coefficient]) *
                           hidden[input];
                }
                expected[(b * width + t) * H + c] = f32_to_bf16(static_cast<float>(sum));
            }
        }
    }
    auto bits = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
        return out;
    };
    DeviceBuffer d_hidden = to_device(bits(hidden));
    DeviceBuffer d_dynamic = to_device(bits(dynamic));
    DeviceBuffer d_base = to_device(bits(base));
    GuardedDeviceBuffer d_out(hidden.size() * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor x(d_hidden.p, DType::BF16, {H, width, batch});
    Tensor coeff(d_dynamic.p, DType::BF16, {D, width, batch});
    Tensor weights(d_base.p, DType::BF16, {H, 2, 2});
    Tensor out(d_out.data(), DType::BF16, {H, width, batch});
    ops::dflash2_conv(x, coeff, weights, side, out, nullptr);
    cuda_synchronize();
    int failures = verify_exact("DFlash2 convolution", from_device<std::uint16_t>(d_out.data(),
                            hidden.size()), expected);
    failures += d_out.verify_guards("DFlash2 convolution guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "DFlash2 convolution: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    for (int side = 0; side < 2; ++side) {
        failures += run_case(1, 1, side);
        failures += run_case(4, 2, side);
    }
    return failures == 0 ? 0 : 1;
}
