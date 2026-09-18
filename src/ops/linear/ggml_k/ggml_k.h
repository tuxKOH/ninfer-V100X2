#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The artifact preserves each original Q4_K/Q6_K row and its embedded scales.
// The row descriptor low bit selects Q6_K; the remaining bits locate the row.
void ggml_k_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void ggml_k_project_split(const Tensor& x, const Weight& weight, const Tensor* outputs,
                          int count, bool add, cudaStream_t stream,
                          bool tiled_gdn_input = false);
void ggml_k_embedding(const Tensor& ids, const Weight& weight, Tensor& out,
                      cudaStream_t stream);

} // namespace ninfer::ops::detail
