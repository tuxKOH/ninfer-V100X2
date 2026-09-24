#include "ninfer/ops/dflash_selector.h"
#include "ops/launcher/dflash_selector.h"
#include <stdexcept>
namespace ninfer::ops {
void dflash2_select(const Tensor& logits, const Tensor& gate, const Tensor& predecessor,
                    const Tensor& successor, const Tensor& anchors, const Tensor& partial,
                    Tensor& topk, Tensor& out,
                    cudaStream_t stream) {
    const int V = logits.ne[0], K = logits.ne[1], B = logits.ne[2], R = gate.ne[0];
    if (logits.dtype != DType::BF16 || gate.dtype != DType::BF16 || predecessor.dtype != DType::BF16 ||
        successor.dtype != DType::BF16 || anchors.dtype != DType::I32 || out.dtype != DType::I32 ||
        V <= 0 || K <= 0 || B <= 0 || R <= 0 || predecessor.ne[0] != V || predecessor.ne[1] != R ||
        successor.ne[0] != V || successor.ne[1] != R || gate.ne[1] != K || gate.ne[2] != B ||
        logits.ne[2] != B || anchors.ne[0] != B ||
        partial.dtype != DType::I64 || partial.ne[0] != 16 ||
        partial.ne[1] != (V + 1023) / 1024 || partial.ne[2] != K * B ||
        topk.dtype != DType::I32 || topk.ne[0] != 16 || topk.ne[1] != K ||
        topk.ne[2] != B || out.ne[0] != K || out.ne[1] != B)
        throw std::invalid_argument("dflash2_select: invalid tensor shape");
    if (!logits.is_contiguous() || !gate.is_contiguous() || !predecessor.is_contiguous() ||
        !successor.is_contiguous() || !anchors.is_contiguous() ||
        !partial.is_contiguous() || !topk.is_contiguous() || !out.is_contiguous())
        throw std::invalid_argument("dflash2_select: tensors must be contiguous");
    detail::dflash2_select_launch(logits, gate, predecessor, successor, anchors,
                                  partial, topk, out, stream);
}
}
