#include "ops/launcher/dflash_conv.h"
#include "core/device.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {
__global__ void dflash2_conv_kernel(const __nv_bfloat16* x, const __nv_bfloat16* dyn,
                                    const __nv_bfloat16* base, __nv_bfloat16* out,
                                    int H, int W, int B, int side) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int b = blockIdx.z;
    if (c >= H || t >= W || b >= B) return;
    const int group = c / 16;
    const int groups = H / 16;
    float sum = 0.0f;
    for (int tap = 0; tap < 2; ++tap) {
        const int src = t - tap;
        if (src < 0) continue;
        // GGML layout is [group, tap, side, token] for dynamic and
        // [channel, tap, side] for base. The coefficients belong to the
        // output position, even when the input comes from an earlier tap.
        const int dyn_base = group + tap * groups + side * groups * 2 +
                             (b * W + t) * groups * 4;
        const float coeff = __bfloat162float(base[c + tap * H + side * H * 2]) +
                            __bfloat162float(dyn[dyn_base]);
        const std::size_t xidx = static_cast<std::size_t>(b) * H * W + static_cast<std::size_t>(src) * H + c;
        sum += coeff * __bfloat162float(x[xidx]);
    }
    out[static_cast<std::size_t>(b) * H * W + static_cast<std::size_t>(t) * H + c] = __float2bfloat16(sum);
}

void dflash2_conv_launch(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                         std::int32_t side, Tensor& out, cudaStream_t stream) {
    dim3 block(256, 1, 1);
    dim3 grid((hidden.ne[0] + 255) / 256, hidden.ne[1], hidden.ne[2]);
    dflash2_conv_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), static_cast<const __nv_bfloat16*>(dynamic.data),
        static_cast<const __nv_bfloat16*>(base.data), static_cast<__nv_bfloat16*>(out.data),
        hidden.ne[0], hidden.ne[1], hidden.ne[2], side);
    CUDA_CHECK(cudaGetLastError());
}
}
