#include "ninfer/ops/dflash_selector.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int V = 1025;
constexpr int R = 8;
constexpr int K = 3;
constexpr int B = 2;

int run_case() {
    std::vector<float> logits(static_cast<std::size_t>(V) * K * B);
    std::vector<float> gate(static_cast<std::size_t>(R) * K * B);
    std::vector<float> prev(static_cast<std::size_t>(V) * R);
    std::vector<float> next(static_cast<std::size_t>(V) * R);
    const std::vector<std::int32_t> anchors{4, 17};
    for (int col = 0; col < K * B; ++col) {
        for (int v = 0; v < V; ++v) {
            logits[col * V + v] = static_cast<float>((v * 7 + col * 3) % 53) / 8.0f;
        }
        for (int r = 0; r < R; ++r) { gate[col * R + r] = static_cast<float>(r + 1) / 64.0f; }
    }
    for (int v = 0; v < V; ++v) {
        for (int r = 0; r < R; ++r) {
            prev[v * R + r] = static_cast<float>((v + 3 * r) % 7 - 3) / 16.0f;
            next[v * R + r] = static_cast<float>((v * 3 + r) % 9 - 4) / 16.0f;
        }
    }
    round_to_bf16(logits);
    round_to_bf16(gate);
    round_to_bf16(prev);
    round_to_bf16(next);
    std::vector<std::int32_t> expected(K * B);
    for (int b = 0; b < B; ++b) {
        int parent = anchors[b];
        for (int t = 0; t < K; ++t) {
            const int col = b * K + t;
            std::vector<int> candidates(V);
            for (int v = 0; v < V; ++v) { candidates[v] = v; }
            std::partial_sort(candidates.begin(), candidates.begin() + 16, candidates.end(),
                              [&](int lhs, int rhs) {
                                  if (logits[col * V + lhs] != logits[col * V + rhs]) {
                                      return logits[col * V + lhs] > logits[col * V + rhs];
                                  }
                                  return lhs < rhs;
                              });
            int best = candidates[0];
            double best_score = -1.0e30;
            for (int i = 0; i < 16; ++i) {
                const int id = candidates[i];
                double score = logits[col * V + id];
                for (int r = 0; r < R; ++r) {
                    score += static_cast<double>(prev[parent * R + r]) * gate[col * R + r] *
                             next[id * R + r];
                }
                if (score > best_score) { best_score = score; best = id; }
            }
            parent = expected[col] = best;
        }
    }
    auto bits = [](const std::vector<float>& src) {
        std::vector<std::uint16_t> dst(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) { dst[i] = f32_to_bf16(src[i]); }
        return dst;
    };
    DeviceBuffer d_logits = to_device(bits(logits));
    DeviceBuffer d_gate = to_device(bits(gate));
    DeviceBuffer d_prev = to_device(bits(prev));
    DeviceBuffer d_next = to_device(bits(next));
    DeviceBuffer d_anchor = to_device(anchors);
    DeviceBuffer d_partial(static_cast<std::size_t>(16 * 2 * K * B) * sizeof(std::int64_t));
    DeviceBuffer d_topk(static_cast<std::size_t>(16 * K * B) * sizeof(std::int32_t));
    DeviceBuffer d_out(static_cast<std::size_t>(K * B) * sizeof(std::int32_t));
    Tensor logits_t(d_logits.p, DType::BF16, {V, K, B});
    Tensor gate_t(d_gate.p, DType::BF16, {R, K, B});
    Tensor prev_t(d_prev.p, DType::BF16, {V, R});
    Tensor next_t(d_next.p, DType::BF16, {V, R});
    Tensor anchor_t(d_anchor.p, DType::I32, {B});
    Tensor partial_t(d_partial.p, DType::I64, {16, 2, K * B});
    Tensor topk_t(d_topk.p, DType::I32, {16, K, B});
    Tensor out_t(d_out.p, DType::I32, {K, B});
    ops::dflash2_select(logits_t, gate_t, prev_t, next_t, anchor_t, partial_t, topk_t,
                        out_t, nullptr);
    cuda_synchronize();
    return verify_exact("DFlash2 selector", from_device<std::int32_t>(d_out, K * B), expected);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "DFlash2 selector: SKIP (CUDA unavailable)\n";
        return 77;
    }
    return run_case() == 0 ? 0 : 1;
}
