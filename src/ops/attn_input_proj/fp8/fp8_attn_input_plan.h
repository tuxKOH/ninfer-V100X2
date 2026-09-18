#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

// --- tp2 column-shard siblings, instantiated at Fp8AttnInputTp2ColumnGeometry. -----------------
// Same kernel templates as the tp1 forms above, instantiated at the shard's halved N. Route
// selection (resolve_route) is inherited unchanged -- a pure function of (policy, token count).
// The workspace query is reused UNCHANGED from the tp1 form: K=5120 is unchanged by the shard
// (only the output row count N halves), the same rule NVFP4's shard of this family follows -- no
// `_shard` capacity function exists or is needed.

void fp8_attn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                    cudaStream_t stream);

void fp8_attn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, LinearPolicy policy,
                                   WorkspaceArena* workspace, cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
void launch_fp8_attn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
