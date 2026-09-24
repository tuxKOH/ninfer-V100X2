#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void dflash2_select_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                           const Tensor&, Tensor&, Tensor&, cudaStream_t);
}
