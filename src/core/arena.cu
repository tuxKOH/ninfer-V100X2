#include "core/arena.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

// Physical-memory unit the relocatable arena unmaps at a time. Anything at least the driver's
// allocation granularity works; 64 MiB keeps the handle count modest for a multi-GiB KV arena
// while bounding the copy overshoot of one eviction.
constexpr std::size_t kRelocationChunkBytes = 64ULL << 20;

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

std::string driver_error_message(const char* prefix, CUresult result) {
    const char* name    = nullptr;
    const char* message = nullptr;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &message);
    return std::string(prefix) + ": " + (name != nullptr ? name : "CUDA_ERROR_UNKNOWN") + ": " +
           (message != nullptr ? message : "unknown driver error");
}

void driver_check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) { throw std::runtime_error(driver_error_message(operation, result)); }
}

void log_driver_error(const char* operation, CUresult result) noexcept {
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "%s\n", driver_error_message(operation, result).c_str());
    }
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", cudaFree(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(ptr));
        ptr = nullptr;
    }
}

} // namespace

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    p = ptr;
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_device(p);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = cudaMemset(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }

    base_ = ptr;
    cap_  = capacity_bytes;
    off_  = 0;
}

struct DeviceArena::Relocation {
    struct Chunk {
        CUdeviceptr address = 0;
        CUmemGenericAllocationHandle handle = 0;
        bool mapped = false;
    };
    struct Evicted {
        std::size_t chunk       = 0;
        std::size_t host_offset = 0;
    };

    int device               = 0;
    CUmemAllocationProp prop{};
    // The local device plus every device that can reach this one. Tensor-parallel allreduce
    // pulls through peer copies, and a VMM mapping is invisible to those unless the peer's
    // location is granted here; cudaDeviceEnablePeerAccess only covers cudaMalloc memory.
    std::vector<CUmemAccessDesc> access;
    std::size_t chunk_bytes  = 0;
    std::size_t virtual_bytes = 0;
    CUdeviceptr base         = 0;
    std::vector<Chunk> chunks;
    std::vector<Evicted> evicted;
    std::size_t mapped = 0;

    ~Relocation() { release_all(); }

    void release_all() noexcept {
        for (Chunk& chunk : chunks) {
            if (!chunk.mapped) { continue; }
            log_driver_error("cuMemUnmap", cuMemUnmap(chunk.address, chunk_bytes));
            log_driver_error("cuMemRelease", cuMemRelease(chunk.handle));
            chunk.mapped = false;
            chunk.handle = 0;
        }
        if (base != 0) {
            log_driver_error("cuMemAddressFree", cuMemAddressFree(base, virtual_bytes));
            base = 0;
        }
        mapped = 0;
        evicted.clear();
    }

    void require_current_device() const {
        int current = -1;
        const cudaError_t err = cudaGetDevice(&current);
        if (err != cudaSuccess) {
            throw std::runtime_error(cuda_error_message("cudaGetDevice failed", err));
        }
        if (current != device) {
            throw std::logic_error("relocatable DeviceArena used from a different device");
        }
    }

    void map_chunk(Chunk& chunk) {
        driver_check(cuMemCreate(&chunk.handle, chunk_bytes, &prop, 0), "cuMemCreate");
        const CUresult status = cuMemMap(chunk.address, chunk_bytes, 0, chunk.handle, 0);
        if (status != CUDA_SUCCESS) {
            (void)cuMemRelease(chunk.handle);
            chunk.handle = 0;
            throw std::runtime_error(driver_error_message("cuMemMap", status));
        }
        const CUresult access_status = cuMemSetAccess(chunk.address, chunk_bytes, access.data(),
                                                      access.size());
        if (access_status != CUDA_SUCCESS) {
            (void)cuMemUnmap(chunk.address, chunk_bytes);
            (void)cuMemRelease(chunk.handle);
            chunk.handle = 0;
            throw std::runtime_error(driver_error_message("cuMemSetAccess", access_status));
        }
        chunk.mapped = true;
        ++mapped;
    }
};

DeviceArena::DeviceArena(std::size_t capacity_bytes, RelocatableTag) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }
    auto relocation = std::make_unique<Relocation>();
    int device      = 0;
    const cudaError_t device_err = cudaGetDevice(&device);
    if (device_err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDevice failed", device_err));
    }
    relocation->device = device;

    driver_check(cuInit(0), "cuInit");
    relocation->prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    relocation->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    relocation->prop.location.id   = device;
    {
        CUmemAccessDesc local{};
        local.location = relocation->prop.location;
        local.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        relocation->access.push_back(local);
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess) {
        for (int other = 0; other < device_count; ++other) {
            if (other == device) { continue; }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, other, device) == cudaSuccess &&
                can_access != 0) {
                CUmemAccessDesc peer{};
                peer.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                peer.location.id   = other;
                peer.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                relocation->access.push_back(peer);
            }
        }
    }
    std::size_t granularity        = 0;
    driver_check(cuMemGetAllocationGranularity(&granularity, &relocation->prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
    const std::size_t chunk_bytes =
        (kRelocationChunkBytes + granularity - 1) / granularity * granularity;
    if (capacity_bytes > std::numeric_limits<std::size_t>::max() - (chunk_bytes - 1)) {
        throw std::overflow_error("relocatable arena address size overflows size_t");
    }
    const std::size_t virtual_bytes =
        (capacity_bytes + chunk_bytes - 1) / chunk_bytes * chunk_bytes;
    relocation->chunk_bytes   = chunk_bytes;
    relocation->virtual_bytes = virtual_bytes;
    driver_check(cuMemAddressReserve(&relocation->base, virtual_bytes, 0, 0, 0),
                 "cuMemAddressReserve");
    relocation->chunks.resize(virtual_bytes / chunk_bytes);
    for (std::size_t i = 0; i < relocation->chunks.size(); ++i) {
        Relocation::Chunk& chunk = relocation->chunks[i];
        chunk.address            = relocation->base + i * chunk_bytes;
        relocation->map_chunk(chunk);
    }

    cap_        = capacity_bytes;
    base_       = reinterpret_cast<void*>(relocation->base);
    off_        = 0;
    relocation_ = std::move(relocation);
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_ && relocation_ == nullptr) { free_device(base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_), relocation_(std::move(other.relocation_)) {
    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    // A relocatable base is a reserved virtual address owned by its Relocation, not a
    // cudaMalloc block; releasing it through cudaFree would be an invalid free.
    if (owns_ && relocation_ == nullptr) { free_device(base_); }
    relocation_.reset();
    base_       = other.base_;
    cap_        = other.cap_;
    off_        = other.off_;
    peak_       = other.peak_;
    owns_       = other.owns_;
    relocation_ = std::move(other.relocation_);

    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
    return *this;
}

std::size_t DeviceArena::relocation_span(std::size_t bytes) const {
    if (relocation_ == nullptr || bytes == 0) { return 0; }
    const std::size_t chunk_bytes = relocation_->chunk_bytes;
    const std::size_t count       = (bytes + chunk_bytes - 1) / chunk_bytes;
    if (count > relocation_->chunks.size()) {
        throw std::length_error("relocatable arena cannot release that many bytes");
    }
    return count * chunk_bytes;
}

std::size_t DeviceArena::evict_to_host(std::size_t bytes, void* host) {
    if (relocation_ == nullptr) { throw std::logic_error("DeviceArena is not relocatable"); }
    Relocation& relocation = *relocation_;
    if (bytes == 0) { return 0; }
    if (host == nullptr) { throw std::invalid_argument("DeviceArena eviction needs host storage"); }
    if (!relocation.evicted.empty()) {
        throw std::logic_error("relocatable DeviceArena is already evicted");
    }
    relocation.require_current_device();

    const std::size_t span  = relocation_span(bytes);
    const std::size_t count = span / relocation.chunk_bytes;
    if (count > relocation.mapped) {
        throw std::logic_error("relocatable arena has too little mapped memory to evict");
    }

    std::size_t offset = 0;
    for (std::size_t index = relocation.chunks.size();
         index-- > 0 && relocation.evicted.size() < count;) {
        Relocation::Chunk& chunk = relocation.chunks[index];
        if (!chunk.mapped) { continue; }
        const cudaError_t copy = cudaMemcpy(static_cast<unsigned char*>(host) + offset,
                                            reinterpret_cast<void*>(chunk.address),
                                            relocation.chunk_bytes, cudaMemcpyDeviceToHost);
        if (copy != cudaSuccess) {
            throw std::runtime_error(cuda_error_message("arena eviction copy failed", copy));
        }
        driver_check(cuMemUnmap(chunk.address, relocation.chunk_bytes), "cuMemUnmap");
        driver_check(cuMemRelease(chunk.handle), "cuMemRelease");
        chunk.mapped = false;
        chunk.handle = 0;
        --relocation.mapped;
        relocation.evicted.push_back(Relocation::Evicted{index, offset});
        offset += relocation.chunk_bytes;
    }
    if (offset != span) {
        // Unreachable: `count` mapped chunks were checked above.
        throw std::logic_error("relocatable arena released a short span");
    }
    return offset;
}

std::size_t DeviceArena::evicted_bytes() const noexcept {
    if (relocation_ == nullptr) { return 0; }
    return relocation_->evicted.size() * relocation_->chunk_bytes;
}

void DeviceArena::restore_from_host(const void* host, std::size_t bytes) {
    if (relocation_ == nullptr) { throw std::logic_error("DeviceArena is not relocatable"); }
    Relocation& relocation = *relocation_;
    if (relocation.evicted.empty()) {
        if (bytes != 0) {
            throw std::invalid_argument("relocatable arena has nothing to restore");
        }
        return;
    }
    if (host == nullptr) { throw std::invalid_argument("DeviceArena restore needs host storage"); }
    if (bytes != relocation.evicted.size() * relocation.chunk_bytes) {
        throw std::invalid_argument("relocatable arena restore span is inconsistent");
    }
    relocation.require_current_device();

    std::size_t done = 0;
    try {
        for (; done < relocation.evicted.size(); ++done) {
            const Relocation::Evicted& entry = relocation.evicted[done];
            Relocation::Chunk& chunk         = relocation.chunks[entry.chunk];
            if (chunk.mapped) { throw std::logic_error("relocatable arena chunk is already mapped"); }
            relocation.map_chunk(chunk);
            const cudaError_t copy = cudaMemcpy(
                reinterpret_cast<void*>(chunk.address),
                static_cast<const unsigned char*>(host) + entry.host_offset,
                relocation.chunk_bytes, cudaMemcpyHostToDevice);
            if (copy != cudaSuccess) {
                // Return the chunk to the evicted state so a retry sees a coherent arena.
                log_driver_error("cuMemUnmap", cuMemUnmap(chunk.address, relocation.chunk_bytes));
                log_driver_error("cuMemRelease", cuMemRelease(chunk.handle));
                chunk.mapped = false;
                chunk.handle = 0;
                --relocation.mapped;
                throw std::runtime_error(cuda_error_message("arena restore copy failed", copy));
            }
        }
    } catch (...) {
        relocation.evicted.erase(relocation.evicted.begin(),
                                 relocation.evicted.begin() + static_cast<std::ptrdiff_t>(done));
        throw;
    }
    relocation.evicted.clear();
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept { off_ = 0; }

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMallocHost failed", err));
    }

    data_ = ptr;
    size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_pinned(data_);
    data_ = other.data_;
    size_ = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

} // namespace ninfer
