// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::size_t kDefaultHostStagingBytes = 64U * 1024U * 1024U;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void startup_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("peer transport startup: ") + operation + ": " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

// Linux CUDA PCIe P2P is unsupported behind a translated IOMMU domain. A small
// allocation can nevertheless pass a copy probe while other mappings silently
// lose writes, so the domain restriction takes precedence over that probe.
std::string translated_iommu_domain(const ExecutionContext& ec) {
    std::string reason;
#if defined(__linux__)
    for (int rank = 0; rank < 2; ++rank) {
        char pci_bus_id[32]{};
        startup_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ec.dev[rank]->device),
                      "cudaDeviceGetPCIBusId");
        // CUDA may emit uppercase hexadecimal; Linux PCI sysfs names are lowercase.
        for (char& c : pci_bus_id) {
            if (c >= 'A' && c <= 'F') { c += 'a' - 'A'; }
        }
        std::ifstream domain_file(std::string("/sys/bus/pci/devices/") + pci_bus_id +
                                  "/iommu_group/type");
        std::string domain;
        domain_file >> domain;
        if (domain == "DMA" || domain == "DMA-FQ") {
            if (!reason.empty()) { reason += "; "; }
            reason += std::string("PCI ") + pci_bus_id + " uses translated IOMMU domain " + domain;
        }
    }
#else
    (void)ec;
#endif
    return reason;
}

// Startup-only storage. The exact payload catches drivers that advertise peer access
// but silently drop DMA writes (observed with V100s behind an IOMMU). Use the same
// pull API and destination compute stream as the collectives, without GPU peer loads.
class PeerTransferProbe {
public:
    explicit PeerTransferProbe(const ExecutionContext& ec) : ec_(ec) {
        startup_check(cudaGetDevice(&previous_), "cudaGetDevice");
    }

    ~PeerTransferProbe() {
        for (int rank = 0; rank < 2; ++rank) {
            cleanup(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
            if (source_[rank] != nullptr) { cleanup(cudaFree(source_[rank]), "cudaFree source"); }
            if (destination_[rank] != nullptr) {
                cleanup(cudaFree(destination_[rank]), "cudaFree destination");
            }
        }
        cleanup(cudaSetDevice(previous_), "restore device");
    }

    PeerTransferProbe(const PeerTransferProbe&) = delete;
    PeerTransferProbe& operator=(const PeerTransferProbe&) = delete;

    void set_device(int rank) const {
        startup_check(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
    }

    void initialize() {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            startup_check(cudaMalloc(&source_[rank], kBytes), "cudaMalloc source");
            startup_check(cudaMalloc(&destination_[rank], kBytes), "cudaMalloc destination");
            std::array<std::uint32_t, kWords> values{};
            for (std::size_t i = 0; i < kWords; ++i) { values[i] = pattern(rank, i); }
            startup_check(cudaMemcpyAsync(source_[rank], values.data(), kBytes,
                                           cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                          "initialize source");
            startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream), "retire source");
        }
    }

    // Empty means both complete copies matched their independent host patterns exactly.
    std::string qualify() {
        std::string mismatch;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cudaStream_t stream = ec_.dev[rank]->stream;
            startup_check(cudaMemsetAsync(destination_[rank], 0xcd, kBytes, stream),
                          "clear destination");
            startup_check(pull_peer(destination_[rank], source_[1 - rank], kBytes, stream),
                          "cross-device copy");
            startup_check(cudaStreamSynchronize(stream), "retire cross-device copy");
            std::array<std::uint32_t, kWords> actual{};
            startup_check(cudaMemcpy(actual.data(), destination_[rank], kBytes,
                                      cudaMemcpyDeviceToHost), "read destination");
            for (std::size_t i = 0; i < kWords; ++i) {
                if (actual[i] != pattern(1 - rank, i)) {
                    if (mismatch.empty()) {
                        mismatch = "device " + std::to_string(ec_.dev[1 - rank]->device) +
                                   " -> " + std::to_string(ec_.dev[rank]->device) +
                                   " data mismatch at word " + std::to_string(i);
                    }
                    break;
                }
            }
        }
        return mismatch;
    }

    // Validate the explicit fallback used by the collectives.  This is intentionally separate
    // from qualify(): the UVA D2D form exercises the driver's opaque staging path, while this
    // route must prove the two concrete D2H/H2D copies and their byte identity.
    std::string qualify_host_staged() {
        std::array<void*, 2> host{nullptr, nullptr};
        std::string mismatch;
        try {
            for (void*& slot : host) {
                startup_check(cudaHostAlloc(&slot, kBytes, cudaHostAllocPortable),
                              "cudaHostAlloc probe");
            }
            for (int rank = 0; rank < 2; ++rank) {
                set_device(rank);
                startup_check(cudaMemcpyAsync(host[rank], source_[rank], kBytes,
                                              cudaMemcpyDeviceToHost, ec_.dev[rank]->stream),
                              "probe device-to-host");
                startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream),
                              "retire probe device-to-host");
            }
            for (int rank = 0; rank < 2; ++rank) {
                set_device(rank);
                startup_check(cudaMemcpyAsync(destination_[rank], host[1 - rank], kBytes,
                                              cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                              "probe host-to-device");
                startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream),
                              "retire probe host-to-device");
                std::array<std::uint32_t, kWords> actual{};
                startup_check(cudaMemcpy(actual.data(), destination_[rank], kBytes,
                                         cudaMemcpyDeviceToHost), "read staged destination");
                for (std::size_t i = 0; i < kWords; ++i) {
                    if (actual[i] != pattern(1 - rank, i)) {
                        mismatch = "device " + std::to_string(ec_.dev[1 - rank]->device) +
                                   " -> " + std::to_string(ec_.dev[rank]->device) +
                                   " staged data mismatch at word " + std::to_string(i);
                        break;
                    }
                }
                if (!mismatch.empty()) { break; }
            }
        } catch (...) {
            for (void*& slot : host) {
                if (slot != nullptr) { (void)cudaFreeHost(slot); }
            }
            throw;
        }
        for (void*& slot : host) {
            if (slot != nullptr) { startup_check(cudaFreeHost(slot), "cudaFreeHost probe"); }
        }
        return mismatch;
    }

    void disable_peer_access() const {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            const cudaError_t status = cudaDeviceDisablePeerAccess(ec_.dev[1 - rank]->device);
            if (status == cudaErrorPeerAccessNotEnabled) {
                (void)cudaGetLastError();
            } else {
                startup_check(status, "cudaDeviceDisablePeerAccess");
            }
        }
    }

private:
    static constexpr std::size_t kWords = 4096;
    static constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

    static std::uint32_t pattern(int rank, std::size_t index) {
        return 0x4f000000U ^ (std::uint32_t(rank) << 20U) ^
               (static_cast<std::uint32_t>(index) * 65537U);
    }

    static void cleanup(cudaError_t status, const char* operation) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during peer probe %s: %s: %s\n",
                         operation, cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    const ExecutionContext& ec_;
    int previous_ = 0;
    std::array<void*, 2> source_{};
    std::array<void*, 2> destination_{};
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    ec.direct_peer_access = false;
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    PeerTransferProbe probe(ec);
    try {
        std::string direct_failure = translated_iommu_domain(ec);
        int forward = 0;
        int reverse = 0;
        if (direct_failure.empty()) {
            startup_check(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]),
                          "cudaDeviceCanAccessPeer forward");
            startup_check(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]),
                          "cudaDeviceCanAccessPeer reverse");
        }
        const bool supported = direct_failure.empty() && forward != 0 && reverse != 0;
        if (supported) {
            for (int rank = 0; rank < 2; ++rank) {
                probe.set_device(rank);
                const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else {
                    startup_check(status, "cudaDeviceEnablePeerAccess");
                }
            }
        } else {
            if (direct_failure.empty()) { direct_failure = "peer access unavailable"; }
            probe.disable_peer_access();
        }
        probe.initialize();
        if (supported) {
            direct_failure = probe.qualify();
            if (direct_failure.empty()) {
                ec.direct_peer_access = true;
                return true;
            }
            probe.disable_peer_access();
        }

        const std::string staged_failure = probe.qualify_host_staged();
        if (!staged_failure.empty()) {
            throw std::runtime_error("peer transport startup: host-staged validation failed: " +
                                     staged_failure);
        }
        std::fprintf(stderr,
                     "[ninfer] direct P2P disabled (%s); using verified pinned host-staged copies\n",
                     direct_failure.c_str());
        return false;
    } catch (...) {
        // Failed startup must not leave a partially enabled pair behind. Clear the
        // runtime's last error before cleanup; a fatal context error still prevents
        // further use, and the original startup exception remains authoritative.
        (void)cudaGetLastError();
        try { probe.disable_peer_access(); } catch (...) {}
        throw;
    }
}

PeerEvents::PeerEvents(const ExecutionContext& ec) {
    require_two_devices(ec, "PeerEvents: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    direct_transport_ = ec.direct_peer_access;
    for (int rank = 0; rank < 2; ++rank) {
        devices_[static_cast<std::size_t>(rank)] = ec.dev[rank]->device;
        streams_[static_cast<std::size_t>(rank)] = ec.dev[rank]->stream;
    }
    cudaEvent_t created[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int slot = 0; slot < 6; ++slot) {
        const int rank             = slot % 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    inputs_ready_ = {created[0], created[1]};
    transfer_ready_ = {created[2], created[3]};
    pull_done_      = {created[4], created[5]};
    try {
        // Graph capture may encounter the largest prefill all-reduce or vocabulary gather before
        // an eager call can grow the fallback buffer. The V100X2 4096-token prefill is below
        // 42 MiB, so 64 MiB per source covers every registered shape.
        ensure_host_staging(kDefaultHostStagingBytes);
    } catch (...) {
        for (cudaEvent_t& event : created) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
        for (void*& slot : host_staging_) {
            if (slot != nullptr) { (void)cudaFreeHost(slot); slot = nullptr; }
        }
        throw;
    }
    // A host slot is also read by the peer's H2D engine. Seed pull_done with a completed event so
    // the first staged collective may use the same inter-call lifetime edge as every later one.
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(pull_done_[static_cast<std::size_t>(rank)],
                                   ec.dev[rank]->stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[rank]->stream));
    }
    host_staging_ready_ = true;
}

PeerEvents::~PeerEvents() {
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &transfer_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
    for (void*& slot : host_staging_) {
        if (slot != nullptr) {
            for (cudaStream_t stream : streams_) {
                const cudaError_t sync = cudaStreamSynchronize(stream);
                if (sync != cudaSuccess) {
                    std::fprintf(stderr, "CUDA cleanup failed during stream synchronization: %s: %s\n",
                                 cudaGetErrorName(sync), cudaGetErrorString(sync));
                }
            }
            const cudaError_t status = cudaFreeHost(slot);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaFreeHost: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            slot = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), transfer_ready_(other.transfer_ready_),
      pull_done_(other.pull_done_), direct_transport_(other.direct_transport_),
      devices_(other.devices_), streams_(other.streams_),
      host_staging_(other.host_staging_), host_staging_bytes_(other.host_staging_bytes_),
      host_staging_ready_(other.host_staging_ready_) {
    other.inputs_ready_ = {nullptr, nullptr};
    other.transfer_ready_ = {nullptr, nullptr};
    other.pull_done_    = {nullptr, nullptr};
    other.devices_      = {0, 0};
    other.streams_      = {nullptr, nullptr};
    other.host_staging_ = {nullptr, nullptr};
    other.host_staging_bytes_ = 0;
    other.host_staging_ready_ = false;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    transfer_ready_.swap(other.transfer_ready_);
    pull_done_.swap(other.pull_done_);
    std::swap(direct_transport_, other.direct_transport_);
    devices_.swap(other.devices_);
    streams_.swap(other.streams_);
    host_staging_.swap(other.host_staging_);
    std::swap(host_staging_bytes_, other.host_staging_bytes_);
    std::swap(host_staging_ready_, other.host_staging_ready_);
    return *this;
}

void PeerEvents::ensure_host_staging(std::size_t bytes) const {
    if (direct_transport_ || bytes <= host_staging_bytes_) { return; }
    if (bytes == 0) { return; }
    for (cudaStream_t stream : streams_) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            throw std::logic_error("host-staged collective grew its pinned buffer during CUDA graph capture");
        }
    }
    // A resize is exceptional (normally the largest prefill chunk allocates it once). Retire both
    // streams before releasing the old host slots so no in-flight H2D node can observe freed data.
    for (cudaStream_t stream : streams_) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    for (void*& slot : host_staging_) {
        if (slot != nullptr) { CUDA_CHECK(cudaFreeHost(slot)); slot = nullptr; }
    }
    try {
        for (void*& slot : host_staging_) {
            CUDA_CHECK(cudaHostAlloc(&slot, bytes, cudaHostAllocPortable));
        }
        host_staging_bytes_ = bytes;
    } catch (...) {
        for (void*& slot : host_staging_) {
            if (slot != nullptr) { (void)cudaFreeHost(slot); slot = nullptr; }
        }
        host_staging_bytes_ = 0;
        throw;
    }
}

void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }
    events.ensure_host_staging(bytes);

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: direct P2P is one pull.  Behind translated IOMMU domains, make the two PCIe
    // transfers explicit: both D2H legs are issued first, then both ranks wait for the peer's
    // host slot and issue H2D.  The host slots are source-owned and the existing pull_done edge
    // still prevents a source buffer from being overwritten while its peer reads it.
    if (events.direct_transport()) {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
            CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(events.host_staging(rank), buffer[rank].data, bytes,
                                       cudaMemcpyDeviceToHost, local.stream));
            CUDA_CHECK(cudaEventRecord(events.transfer_ready(rank), local.stream));
        }
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.transfer_ready(1 - rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(staging[rank].data, events.host_staging(1 - rank), bytes,
                                       cudaMemcpyHostToDevice, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};
    events.ensure_host_staging(std::max(block[0], block[1]));

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each destination is written only by its own stream.  The staged branch first
    // exports each owned block to its source slot, then imports the peer slot after its event.
    if (events.direct_transport()) {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                       part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                       local.stream));
            CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                                 part[1 - rank].data, block[1 - rank], local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(events.host_staging(rank), part[rank].data, block[rank],
                                       cudaMemcpyDeviceToHost, local.stream));
            CUDA_CHECK(cudaEventRecord(events.transfer_ready(rank), local.stream));
            CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                       part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                       local.stream));
        }
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.transfer_ready(1 - rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[1 - rank]),
                                       events.host_staging(1 - rank), block[1 - rank],
                                       cudaMemcpyHostToDevice, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

void broadcast_rank0(const Tensor& source, const Tensor& destination,
                     const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec, "broadcast_rank0: requires two devices");
    require(events.live(), "broadcast_rank0: events must be live");
    require(source.data != nullptr && destination.data != nullptr &&
                source.dtype == destination.dtype && source.is_contiguous() &&
                destination.is_contiguous(), "broadcast_rank0: invalid tensors");
    for (int d = 0; d < 4; ++d) {
        require(source.ne[d] == destination.ne[d], "broadcast_rank0: shapes must match");
    }
    const std::size_t bytes = source.bytes();
    if (bytes == 0) { return; }
    events.ensure_host_staging(bytes);
#ifndef NDEBUG
    require_resident_on(source.data, ec.dev[0]->device, "broadcast_rank0: source is not on rank 0");
    require_resident_on(destination.data, ec.dev[1]->device,
                        "broadcast_rank0: destination is not on rank 1");
#endif
    const CurrentDeviceGuard guard;
    const DeviceContext& origin = *ec.dev[0];
    const DeviceContext& peer = *ec.dev[1];
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), origin.stream));
    if (events.direct_transport()) {
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.inputs_ready(0), 0));
        CUDA_CHECK(pull_peer(destination.data, source.data, bytes, peer.stream));
    } else {
        CurrentDeviceGuard::set(origin.device);
        CUDA_CHECK(cudaMemcpyAsync(events.host_staging(0), source.data, bytes,
                                   cudaMemcpyDeviceToHost, origin.stream));
        CUDA_CHECK(cudaEventRecord(events.transfer_ready(0), origin.stream));
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.transfer_ready(0), 0));
        CUDA_CHECK(cudaMemcpyAsync(destination.data, events.host_staging(0), bytes,
                                   cudaMemcpyHostToDevice, peer.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(1), peer.stream));
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaStreamWaitEvent(origin.stream, events.pull_done(1), 0));
}

} // namespace ninfer::ops
