#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t ggml_k_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                std::int32_t tokens);
void ggml_k_cutlass_sm70_launch(const Tensor& x, const Weight& w, const Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream,
                                std::int32_t weight_row_offset = 0, bool add = false,
                                bool tiled_gdn_input = false);

} // namespace ninfer::ops::detail
