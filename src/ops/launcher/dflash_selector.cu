#include "ops/launcher/dflash_selector.h"
#include "core/device.h"

#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kTop = 16;
constexpr int kThreads = 256;
constexpr int kItems = 4;
constexpr int kTile = kThreads * kItems;
using Sort = cub::BlockRadixSort<std::uint64_t, kThreads, kItems>;

__device__ __forceinline__ std::uint64_t logit_key(float value, int id) {
    const std::uint32_t raw = __float_as_uint(value);
    const std::uint32_t ordered = (raw & 0x80000000u) ? ~raw : raw ^ 0x80000000u;
    return (static_cast<std::uint64_t>(ordered) << 32) |
           static_cast<std::uint32_t>(~id);
}

__global__ void dflash2_partial_topk(const __nv_bfloat16* logits, std::uint64_t* partial,
                                      int vocab, int parts) {
    const int part = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    const int first = part * kTile;
    std::uint64_t keys[kItems];
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int id = first + item * kThreads + static_cast<int>(threadIdx.x);
        keys[item] = id < vocab
                         ? logit_key(__bfloat162float(logits[static_cast<std::size_t>(col) * vocab + id]), id)
                         : 0ULL;
    }
    __shared__ Sort::TempStorage storage;
    Sort(storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * kItems + item;
        if (rank < kTop) {
            partial[(static_cast<std::size_t>(col) * parts + part) * kTop + rank] = keys[item];
        }
    }
}

__global__ void dflash2_finish_topk(const std::uint64_t* partial, int* topk, int parts) {
    const int col = static_cast<int>(blockIdx.x);
    std::uint64_t winners[kTop];
#pragma unroll
    for (int i = 0; i < kTop; ++i) { winners[i] = 0ULL; }
    for (int item = 0; item < parts * kTop; ++item) {
        const std::uint64_t key = partial[static_cast<std::size_t>(col) * parts * kTop + item];
        int insertion = kTop;
#pragma unroll
        for (int i = 0; i < kTop; ++i) {
            if (key > winners[i]) { insertion = i; break; }
        }
        if (insertion == kTop) { continue; }
        for (int i = kTop - 1; i > insertion; --i) { winners[i] = winners[i - 1]; }
        winners[insertion] = key;
    }
#pragma unroll
    for (int i = 0; i < kTop; ++i) {
        topk[col * kTop + i] = static_cast<int>(~static_cast<std::uint32_t>(winners[i]));
    }
}

__global__ void dflash2_walk(const __nv_bfloat16* logits, const __nv_bfloat16* gate,
                             const __nv_bfloat16* predecessor,
                             const __nv_bfloat16* successor, const int* anchors,
                             const int* topk, int* out, int vocab, int steps, int rank) {
    const int batch = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float scores[kTop];
    __shared__ int selected;
    if (tid == 0) { selected = anchors[batch]; }
    __syncthreads();
    for (int step = 0; step < steps; ++step) {
        const int col = batch * steps + step;
        const int* candidates = topk + col * kTop;
        const __nv_bfloat16* g = gate + static_cast<std::size_t>(col) * rank;
        for (int pass = 0; pass < 2; ++pass) {
            const int candidate = warp + pass * 8;
            const int id = candidates[candidate];
            float subtotal = 0.0f;
            for (int r = lane; r < rank; r += 32) {
                const float pred = __bfloat162float(predecessor[
                    static_cast<std::size_t>(selected) * rank + r]);
                const float factor = __bfloat162float(g[r]);
                const float succ = __bfloat162float(successor[
                    static_cast<std::size_t>(id) * rank + r]);
                subtotal += pred * factor * succ;
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                subtotal += __shfl_down_sync(0xffffffffu, subtotal, offset);
            }
            if (lane == 0) {
                scores[candidate] = __bfloat162float(logits[
                    static_cast<std::size_t>(col) * vocab + id]) + subtotal;
            }
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int candidate = 1; candidate < kTop; ++candidate) {
                if (scores[candidate] > scores[best]) { best = candidate; }
            }
            selected = candidates[best];
            out[col] = selected;
        }
        __syncthreads();
    }
}

} // namespace

void dflash2_select_launch(const Tensor& logits, const Tensor& gate, const Tensor& predecessor,
                           const Tensor& successor, const Tensor& anchors, const Tensor& partial,
                           Tensor& topk, Tensor& out, cudaStream_t stream) {
    const int vocab = logits.ne[0];
    const int steps = logits.ne[1];
    const int batch = logits.ne[2];
    const int parts = (vocab + kTile - 1) / kTile;
    dflash2_partial_topk<<<dim3(parts, steps * batch), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<std::uint64_t*>(partial.data), vocab, parts);
    CUDA_CHECK(cudaGetLastError());
    dflash2_finish_topk<<<steps * batch, 1, 0, stream>>>(
        static_cast<const std::uint64_t*>(partial.data), static_cast<int*>(topk.data), parts);
    CUDA_CHECK(cudaGetLastError());
    dflash2_walk<<<batch, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(successor.data),
        static_cast<const int*>(anchors.data), static_cast<const int*>(topk.data),
        static_cast<int*>(out.data), vocab, steps, gate.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
