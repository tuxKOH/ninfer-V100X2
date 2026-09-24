#include "ninfer/ops/dflash_conv.h"
#include "ops/launcher/dflash_conv.h"
#include <stdexcept>

namespace ninfer::ops {
void dflash2_conv(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                  std::int32_t side, Tensor& out, cudaStream_t stream) {
    if (hidden.dtype != DType::BF16 || dynamic.dtype != DType::BF16 || base.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || hidden.ne[2] <= 0 ||
        hidden.ne[3] != 1 || dynamic.ne[0] != 1280 || dynamic.ne[1] != hidden.ne[1] ||
        dynamic.ne[2] != hidden.ne[2] || dynamic.ne[3] != 1 || base.ne[0] != hidden.ne[0] ||
        base.ne[1] != 2 || base.ne[2] != 2 || base.ne[3] != 1 || side < 0 || side > 1) {
        throw std::invalid_argument("dflash2_conv: invalid tensor shape or side");
    }
    for (int i = 0; i < 4; ++i) if (out.ne[i] != hidden.ne[i])
        throw std::invalid_argument("dflash2_conv: output shape mismatch");
    if (!hidden.is_contiguous() || !dynamic.is_contiguous() || !base.is_contiguous() || !out.is_contiguous() ||
        !hidden.data || !dynamic.data || !base.data || !out.data)
        throw std::invalid_argument("dflash2_conv: tensors must be contiguous and non-null");
    detail::dflash2_conv_launch(hidden, dynamic, base, side, out, stream);
}
}
