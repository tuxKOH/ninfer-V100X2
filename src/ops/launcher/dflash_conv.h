#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void dflash2_conv_launch(const Tensor&, const Tensor&, const Tensor&, std::int32_t, Tensor&, cudaStream_t);
}
