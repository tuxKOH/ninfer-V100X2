#include "ops/linear/ggml_k/ggml_k_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/ggml_k/ggml_k_codec.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::bfloat16_t;
using GemmBf16 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
using GemmFp32 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

// One CTA owns a native K256 block.  For Q4_K a lane reads one packed byte and emits both
// values, so its two 32-wide stores are coalesced; Q6_K keeps one lane per logical value.
// The old per-value decoder reloaded the descriptor and reselected Q4/Q6 for every element.
__global__ void dequant_rows(const unsigned char* __restrict__ rows,
                             const std::uint64_t* __restrict__ descriptors,
                             ElementInput* __restrict__ out, int k) {
    const int row = static_cast<int>(blockIdx.y);
    const int block_index = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const std::uint64_t descriptor = descriptors[row];
    const bool q6 = (descriptor & 1u) != 0;
    const unsigned char* block = rows + (descriptor >> 1) +
                                 block_index * (q6 ? 210 : 144);
    auto* output = out + static_cast<std::int64_t>(row) * k + block_index * 256;
    if (q6) {
        if (tid >= 64) { return; }
        const int half = tid >> 5;
        const int lane = tid & 31;
        const unsigned low0 = __ldg(block + half * 64 + lane);
        const unsigned low1 = __ldg(block + half * 64 + 32 + lane);
        const unsigned high = __ldg(block + 128 + half * 32 + lane);
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
#pragma unroll
        for (int section = 0; section < 4; ++section) {
            const unsigned low = (section & 1) == 0 ? low0 : low1;
            const unsigned nibble = (low >> ((section >> 1) * 4)) & 15u;
            const unsigned high_bits = (high >> (section * 2)) & 3u;
            const int code = static_cast<int>(nibble | (high_bits << 4)) - 32;
            const int scale_index = half * 8 + section * 2 + (lane >> 4);
            const int scale = static_cast<int>(static_cast<signed char>(
                __ldg(block + 192 + scale_index)));
            output[half * 128 + section * 32 + lane] =
                ElementInput(d * static_cast<float>(scale * code));
        }
        return;
    }
    if (tid >= 128) { return; }
    const int pair = tid >> 5;
    const int lane = tid & 31;
    const unsigned code = block[16 + tid];
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const float dm = __half2float(*reinterpret_cast<const __half*>(block + 2));
    const unsigned char* scales = block + 4;
    const int g0 = pair * 2;
    const int g1 = g0 + 1;
    const int scale0 = g0 < 4 ? scales[g0] & 63
                              : (scales[g0 + 4] & 15) | ((scales[g0 - 4] >> 6) << 4);
    const int scale1 = g1 < 4 ? scales[g1] & 63
                              : (scales[g1 + 4] & 15) | ((scales[g1 - 4] >> 6) << 4);
    const int min0 = g0 < 4 ? scales[g0 + 4] & 63
                            : (scales[g0 + 4] >> 4) | ((scales[g0] >> 6) << 4);
    const int min1 = g1 < 4 ? scales[g1 + 4] & 63
                            : (scales[g1 + 4] >> 4) | ((scales[g1] >> 6) << 4);
    output[pair * 64 + lane] = ElementInput((d * scale0) * (code & 15) - dm * min0);
    output[pair * 64 + 32 + lane] = ElementInput((d * scale1) * (code >> 4) - dm * min1);
}

template <bool TiledGdn>
__global__ void bf16_to_fp16(const __nv_bfloat16* __restrict__ in,
                             ElementInput* __restrict__ out, std::int64_t count, int k) {
    // Every registered projection has an even K and token count. Convert two adjacent BF16
    // values per lane so the staging pass uses one 32-bit load and store rather than two scalar
    // transactions. Keep the odd-count tail for the public helper's general contract.
    const std::int64_t pair = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t index = pair * 2;
    if (index >= count) { return; }
    if constexpr (TiledGdn) {
        const int col = static_cast<int>(index % k);
        const int head = col / 128;
        const int grouped = k == 3072 ? (head & 7) * 3 + (head >> 3)
                                      : (head & 15) * 3 + (head >> 4);
        const std::int64_t source = (index / k) * k + grouped * 128 + (col & 127);
        if (index + 1 < count) {
            const auto values = *reinterpret_cast<const __nv_bfloat162*>(in + source);
            *reinterpret_cast<__half2*>(out + index) =
                __float22half2_rn(__bfloat1622float2(values));
        } else {
            out[index] = ElementInput(__bfloat162float(in[source]));
        }
    } else {
        if (index + 1 < count) {
            const auto values = *reinterpret_cast<const __nv_bfloat162*>(in + index);
            *reinterpret_cast<__half2*>(out + index) =
                __float22half2_rn(__bfloat1622float2(values));
        } else {
            out[index] = ElementInput(__bfloat162float(in[index]));
        }
    }
}

std::size_t gemm_workspace_bytes(int n, int k, int t) {
    const cutlass::gemm::GemmCoord shape(t, n, k);
    GemmBf16::Arguments bf16_args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    GemmFp32::Arguments fp32_args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    return std::max(GemmBf16::get_workspace_size(bf16_args),
                    GemmFp32::get_workspace_size(fp32_args));
}

struct Scratch {
    Tensor weight;
    Tensor input;
    DeviceSpan gemm;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, int n, int k, int t) {
    Scratch scratch;
    scratch.weight = allocator.alloc(DType::FP16, {k, n});
    scratch.input = allocator.alloc(DType::FP16, {k, t});
    const std::size_t bytes = gemm_workspace_bytes(n, k, t);
    if (bytes != 0) { scratch.gemm = allocator.alloc_bytes(bytes); }
    return scratch;
}

} // namespace

std::size_t ggml_k_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, tokens);
    return layout.peak_bytes(1);
}

void ggml_k_cutlass_sm70_launch(const Tensor& x, const Weight& w, const Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream,
                                std::int32_t weight_row_offset, bool add,
                                bool tiled_gdn_input) {
    const int n = out.ne[0];
    const int k = w.k;
    const int t = x.ne[1];
    auto scope = workspace.scope();
    const auto scratch = allocate_scratch(workspace, n, k, t);
    auto* weight = static_cast<ElementInput*>(scratch.weight.data);
    auto* input = static_cast<ElementInput*>(scratch.input.data);

    const auto* descriptors = static_cast<const std::uint64_t*>(w.qhigh) + weight_row_offset;
    dequant_rows<<<dim3(static_cast<unsigned>(k / 256), static_cast<unsigned>(n)),
                   128, 0, stream>>>(static_cast<const unsigned char*>(w.qdata), descriptors,
                                     weight, k);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(k) * t;
    const auto input_pairs = static_cast<unsigned>((input_count + 1) / 2);
    if (tiled_gdn_input) {
        bf16_to_fp16<true><<<static_cast<int>((input_pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), input, input_count, k);
    } else {
        bf16_to_fp16<false><<<static_cast<int>((input_pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), input, input_count, k);
    }
    CUDA_CHECK(cudaGetLastError());

    const cutlass::gemm::GemmCoord shape(t, n, k);
    const bool output_fp32 = out.dtype == DType::FP32;
    cutlass::Status status = cutlass::Status::kSuccess;
    if (output_fp32) {
        const int out_ld = static_cast<int>(out.nb[1] / sizeof(float));
        GemmFp32::Arguments args{
            shape, {input, k}, {weight, k}, {static_cast<float*>(out.data), out_ld},
            {static_cast<float*>(out.data), out_ld}, {1.0F, add ? 1.0F : 0.0F}, 1};
        GemmFp32 op;
        status = op.can_implement(args);
        if (status == cutlass::Status::kSuccess) { status = op.initialize(args, scratch.gemm.data, stream); }
        if (status == cutlass::Status::kSuccess) { status = op(stream); }
    } else {
        const int out_ld = static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16));
        GemmBf16::Arguments args{
            shape, {input, k}, {weight, k}, {static_cast<ElementOutput*>(out.data), out_ld},
            {static_cast<ElementOutput*>(out.data), out_ld}, {1.0F, add ? 1.0F : 0.0F}, 1};
        GemmBf16 op;
        status = op.can_implement(args);
        if (status == cutlass::Status::kSuccess) { status = op.initialize(args, scratch.gemm.data, stream); }
        if (status == cutlass::Status::kSuccess) { status = op(stream); }
    }
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("ggml_k_cutlass_sm70: CUTLASS GEMM failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
