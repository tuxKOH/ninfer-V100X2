#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"

#include <cstddef>
#include <memory>

namespace ninfer {

struct DeviceContext;

namespace runtime {

// Owns the request transient allocation. Usually it is frozen at startup and a request may
// activate only a prefix of it. Constructing it with `kDynamicRequestTransientBytes` instead
// makes the buffer grow on demand: the first request that needs more allocates a larger one and
// later requests reuse it. Only the unlimited Vision budget uses that mode, where no finite
// worst case can be reserved. A growth the device cannot satisfy throws
// RequestError(Unavailable) rather than aborting, so the caller fails that one request.
class RequestMemory {
public:
    static constexpr std::size_t kDeviceAllocationAlignment = 256;

    RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes);
    ~RequestMemory();

    RequestMemory(const RequestMemory&)            = delete;
    RequestMemory& operator=(const RequestMemory&) = delete;
    RequestMemory(RequestMemory&&)                 = delete;
    RequestMemory& operator=(RequestMemory&&)      = delete;

    void activate(std::size_t bytes, std::size_t alignment);
    void deactivate() noexcept;

    [[nodiscard]] TransientRegion region() const noexcept;
    [[nodiscard]] ArenaMemorySummary summary() const noexcept;
    void reset_peak() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace ninfer
