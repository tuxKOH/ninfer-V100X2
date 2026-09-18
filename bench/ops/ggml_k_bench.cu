#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/ggml_k/ggml_k.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    using namespace ninfer;
    constexpr std::array<int, 3> shapes[] = {{17408, 5120, 0}, {5120, 8704, 0},
                                            {124160, 5120, 1}, {7168, 5120, 0}};
    for (const auto shape : shapes) {
        const int n = shape[0], k = shape[1], bytes = shape[2] ? 210 : 144;
        std::vector<unsigned char> rows(std::size_t(n) * k / 256 * bytes, 0x65);
        const __half scale = __float2half_rn(0.0001f);
        for (std::size_t b = 0; b < rows.size(); b += bytes) {
            std::memcpy(rows.data() + b + (shape[2] ? 208 : 0), &scale, 2);
            if (!shape[2]) { std::memcpy(rows.data() + b + 2, &scale, 2); }
        }
        std::vector<std::uint64_t> descriptors(n);
        for (int r = 0; r < n; ++r) {
            descriptors[r] = (std::uint64_t(r) * k / 256 * bytes << 1) | shape[2];
        }
        DeviceBuffer codes(rows.size()), table(descriptors.size() * 8);
        codes.copy_from_host(rows.data(), rows.size());
        table.copy_from_host(descriptors.data(), table.bytes);
        Weight weight;
        weight.qtype = QType::GGML_K;
        weight.layout = QuantLayout::GgmlK256;
        weight.n = n; weight.k = k; weight.qdata = codes.p; weight.qhigh = table.p;
        for (const int t : {1, 4, 64}) {
            std::vector<__nv_bfloat16> input(std::size_t(k) * t, __float2bfloat16_rn(0.25f));
            DeviceBuffer x(input.size() * 2), y(std::size_t(n) * t * 2);
            x.copy_from_host(input.data(), x.bytes);
            Tensor xt(x.p, DType::BF16, {k, t}), yt(y.p, DType::BF16, {n, t});
            for (int i = 0; i < 3; ++i) { ops::detail::ggml_k_linear(xt, weight, yt, nullptr); }
            cudaEvent_t begin, end;
            CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
            CUDA_CHECK(cudaEventRecord(begin));
            for (int i = 0; i < 20; ++i) { ops::detail::ggml_k_linear(xt, weight, yt, nullptr); }
            CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            std::cout << "N=" << n << " K=" << k << " T=" << t << " q6=" << shape[2]
                      << " us=" << ms * 1000 / 20 << " GB/s=" << rows.size() / (ms / 20 * 1e6) << '\n';
            CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
        }
    }
}
