#include "runtime/engine/request_memory.h"

#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

class RequestMemory::Impl {
public:
    Impl(DeviceContext& context, std::size_t capacity) : device(context.device) {
        if (capacity == kDynamicRequestTransientBytes) {
            dynamic = true;
            return;
        }
        if (capacity != 0) {
            CUDA_CHECK(cudaSetDevice(device));
            arena = std::make_unique<DeviceArena>(capacity);
        }
    }

    ~Impl() {
        (void)cudaSetDevice(device);
        if (allocation != nullptr) { (void)cudaFree(allocation); }
        arena.reset();
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return arena != nullptr ? arena->capacity() : allocated_bytes;
    }

    int device = 0;
    std::unique_ptr<DeviceArena> arena;
    // Grow-on-demand buffer, used instead of the arena in dynamic mode.
    bool dynamic          = false;
    void* allocation      = nullptr;
    std::size_t allocated_bytes = 0;
    std::size_t active_bytes     = 0;
    std::size_t active_alignment = 1;
    std::size_t peak_bytes       = 0;
};

RequestMemory::RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes)
    : impl_(std::make_unique<Impl>(device, frozen_capacity_bytes)) {}

RequestMemory::~RequestMemory() = default;

void RequestMemory::activate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        if (alignment != 1) {
            throw std::invalid_argument("an empty transient region must use alignment one");
        }
        deactivate();
        return;
    }

    if (!is_power_of_two(alignment) || alignment > kDeviceAllocationAlignment) {
        throw std::invalid_argument("unsupported transient region alignment");
    }

    if (impl_->dynamic) {
        if (bytes > impl_->allocated_bytes) {
            CUDA_CHECK(cudaSetDevice(impl_->device));
            // Release the old buffer first: growth must fit the new size alone, not old + new.
            // A failed growth leaves no buffer; the next, smaller activation reallocates.
            if (impl_->allocation != nullptr) {
                (void)cudaFree(impl_->allocation);
                impl_->allocation      = nullptr;
                impl_->allocated_bytes = 0;
            }
            void* grown              = nullptr;
            const cudaError_t status = cudaMalloc(&grown, bytes);
            if (status != cudaSuccess) {
                // Request-scoped: the caller reports this to the requesting client only. Device
                // memory exhaustion must not take down the shared execution unit.
                (void)cudaGetLastError();
                throw RequestError(RequestErrorKind::Unavailable,
                                   std::string("request transient allocation failed: ") +
                                       cudaGetErrorString(status));
            }
            impl_->allocation      = grown;
            impl_->allocated_bytes = bytes;
        }
    } else if (impl_->arena == nullptr || bytes > impl_->arena->capacity()) {
        throw std::invalid_argument("request transient exceeds its frozen startup capacity");
    }

    impl_->active_bytes     = bytes;
    impl_->active_alignment = alignment;
    impl_->peak_bytes       = std::max(impl_->peak_bytes, bytes);
}

void RequestMemory::deactivate() noexcept {
    impl_->active_bytes     = 0;
    impl_->active_alignment = 1;
}

TransientRegion RequestMemory::region() const noexcept {
    if (impl_->active_bytes == 0) { return {}; }
    std::byte* base = static_cast<std::byte*>(impl_->arena != nullptr ? impl_->arena->base()
                                                                     : impl_->allocation);
    return {base, impl_->active_bytes, impl_->active_alignment};
}

ArenaMemorySummary RequestMemory::summary() const noexcept {
    return ArenaMemorySummary{
        impl_->capacity(),
        impl_->active_bytes,
        impl_->peak_bytes,
    };
}

void RequestMemory::reset_peak() noexcept { impl_->peak_bytes = impl_->active_bytes; }

} // namespace ninfer::runtime
