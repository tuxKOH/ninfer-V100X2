#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include <ninfer/targets/qwen3_6/vision_control.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct VisionUseSpan {
    std::uint32_t begin      = 0;
    std::uint32_t end        = 0;
    std::uint32_t item_index = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_6::VisionControl> control;
    std::vector<VisionUseSpan> uses;
};

// Demand allocation of the GPU Vision scratch. When present, an encode does not use the
// startup workspace at all: `evict` is called with the bytes the encode needs and must make
// enough device memory available for an allocation of exactly that size (the Program borrows
// its relocatable runtime arena, copying the shortfall to host memory; a request that already
// fits is left alone). `restore` is called after the scratch has been freed and must put every
// evicted byte back. Both callbacks run on the prefill thread with all device work serialized.
struct VisionSwapHooks {
    void* self                                  = nullptr;
    void (*evict)(void* self, std::size_t bytes) = nullptr;
    void (*restore)(void* self)                 = nullptr;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
