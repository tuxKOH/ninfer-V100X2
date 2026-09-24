#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace ninfer::ops {
// Greedy DFlash2 lattice walk. logits=[V,K,B], gate=[R,K,B], predecessor/successor=[V,R],
// anchors=[B], partial=[16,ceil(V/1024),K*B] I64, topk=[16,K,B] I32,
// output draft tokens=[K,B]. Scratch is caller-owned for graph address stability.
void dflash2_select(const Tensor& logits, const Tensor& gate, const Tensor& predecessor,
                    const Tensor& successor, const Tensor& anchors, const Tensor& partial,
                    Tensor& topk, Tensor& out,
                    cudaStream_t stream);
}
