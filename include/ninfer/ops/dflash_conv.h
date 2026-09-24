#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Qwen3.8 DFlash2 grouped dynamic causal convolution. `hidden` and `out` are BF16
// [H, W, B], `dynamic` is BF16 [2*K*G, W, B] (K=2, G=H/16), and `base` is BF16
// [H, K, 2].  Each batch row is an independent diffusion block; taps beyond its
// left edge read zero. `side` selects the attention/MLP input or output kernel.
void dflash2_conv(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                  std::int32_t side, Tensor& out, cudaStream_t stream);

}
