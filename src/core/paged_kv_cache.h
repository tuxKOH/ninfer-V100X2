#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

/**
 * Non-owning, single-sequence view consumed by growing-cache Ops.
 *
 * Physical plane order is fixed by the owning homogeneous pool and validated by the consuming
 * Op. block_table is one contiguous I32 logical-block row.
 */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    Tensor read_indices; // Original logical page -> chronological resident-page index.
    Tensor read_table;   // Chronological resident-page index -> physical page.
    std::uint32_t read_capacity = 0;
};

/**
 * Non-owning multi-sequence view consumed by batched growing-cache Ops.
 *
 * Physical planes and the complete block-table matrix are shared by every logical row in one
 * invocation. block_tables is contiguous I32 [logical_pages, table_rows]; the consuming Op
 * receives its per-row table selectors separately.
 */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    Tensor read_indices;
    Tensor read_tables;
    std::uint32_t read_capacity = 0;
};

// A pool plane is storage-only. Consumers assign K/V/layer meaning to plane indices.
struct PagedKVPlaneSpec {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;
};

enum class PagedKVPlaneOrder : std::uint8_t {
    PageMajor,
    HeadMajor,
};

struct PagedKVPoolSpec {
    std::uint32_t page_group_count      = 0;
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
    PagedKVPlaneOrder plane_order       = PagedKVPlaneOrder::PageMajor;
    std::vector<PagedKVPlaneSpec> planes;
    bool ram_enabled = false;
};

struct PagedKVPlaneLayout {
    PagedKVPlaneSpec spec;
    TensorRegion storage;
};

struct PagedKVPoolLayout {
    PagedKVPoolSpec spec;
    std::vector<PagedKVPlaneLayout> planes;
    TensorRegion block_tables;
    TensorRegion read_indices;
    TensorRegion read_tables;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

[[nodiscard]] PagedKVPoolLayout plan_paged_kv_pool(LayoutBuilder& builder,
                                                   const PagedKVPoolSpec& spec);

class PagedKVAllocation;
class PagedKVImage;
struct PagedKVResize;

class PagedKVPool {
public:
    PagedKVPool(DeviceSpan backing, const PagedKVPoolLayout& layout);

    PagedKVPool(const PagedKVPool&)            = delete;
    PagedKVPool& operator=(const PagedKVPool&) = delete;
    PagedKVPool(PagedKVPool&&)                 = delete;
    PagedKVPool& operator=(PagedKVPool&&)      = delete;

    [[nodiscard]] std::uint32_t page_group_count() const noexcept;
    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept;
    [[nodiscard]] std::int32_t table_row_count() const noexcept;
    [[nodiscard]] std::size_t plane_count() const noexcept;
    [[nodiscard]] const Tensor& plane(std::size_t index) const;
    [[nodiscard]] const Tensor& block_tables() const noexcept;
    [[nodiscard]] Tensor block_table_row(std::int32_t row) const;
    [[nodiscard]] bool ram_enabled() const noexcept { return spec_.ram_enabled; }
    [[nodiscard]] const Tensor& read_indices() const noexcept { return read_indices_; }
    [[nodiscard]] const Tensor& read_tables() const noexcept { return read_tables_; }
    [[nodiscard]] std::size_t ram_archive_bytes() const noexcept;

    [[nodiscard]] std::uint32_t entitled_pages() const noexcept;
    [[nodiscard]] std::uint32_t mapped_pages() const noexcept;
    [[nodiscard]] std::uint32_t free_pages() const noexcept;
    [[nodiscard]] bool can_reserve(std::uint32_t page_entitlement) const noexcept;
    [[nodiscard]] PagedKVAllocation reserve(std::uint32_t page_entitlement);

    // Zeros only the named physical page groups across every storage plane.
    void zero_pages(std::span<const std::int32_t> page_ids, cudaStream_t stream = nullptr);

private:
    friend class PagedKVAllocation;
    friend void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

    [[nodiscard]] bool can_replace_entitlement(std::uint32_t old_pages,
                                               std::uint32_t new_pages) const noexcept;
    [[nodiscard]] std::vector<std::int32_t> take_pages(std::uint32_t count,
                                                       std::int32_t preferred_first);
    void return_pages(std::span<const std::int32_t> pages) noexcept;
    void add_entitlement(std::uint32_t pages) noexcept;
    void replace_entitlement(std::uint32_t old_pages, std::uint32_t new_pages) noexcept;
    void acquire_row(std::int32_t row);
    void release_row(std::int32_t row) noexcept;

    PagedKVPoolSpec spec_;
    std::vector<Tensor> planes_;
    Tensor block_tables_;
    Tensor read_indices_;
    Tensor read_tables_;
    std::vector<std::int32_t> free_page_ids_;
    std::vector<bool> row_in_use_;
    std::uint32_t entitled_pages_ = 0;
    std::uint32_t mapped_pages_   = 0;
};

class PagedKVAllocation {
public:
    PagedKVAllocation() noexcept;
    ~PagedKVAllocation();

    PagedKVAllocation(const PagedKVAllocation&)            = delete;
    PagedKVAllocation& operator=(const PagedKVAllocation&) = delete;
    PagedKVAllocation(PagedKVAllocation&& other) noexcept;
    PagedKVAllocation& operator=(PagedKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t page_entitlement() const noexcept;
    [[nodiscard]] std::uint32_t mapped_page_count() const noexcept;
    [[nodiscard]] std::uint32_t mapped_token_capacity() const noexcept;
    // Every storage plane's bytes for one page, times the mapped page count: exactly the host
    // image detach_to_host() produces, and so the host cost of parking this allocation.
    [[nodiscard]] std::size_t mapped_payload_bytes() const noexcept;
    [[nodiscard]] std::int32_t bound_row() const noexcept;
    [[nodiscard]] std::span<const std::int32_t> page_ids() const noexcept;
    [[nodiscard]] bool belongs_to(const PagedKVPool& pool) const noexcept;

    void set_page_entitlement(std::uint32_t pages);
    void cancel_unmapped_entitlement() noexcept;
    void materialize_pages(std::uint32_t pages, cudaStream_t stream = nullptr);
    void materialize_tokens(std::uint32_t tokens, cudaStream_t stream = nullptr);
    void trim_pages(std::uint32_t pages);
    void trim_tokens(std::uint32_t tokens);

    void bind_row(std::int32_t row, cudaStream_t stream = nullptr);
    void publish_mapping(cudaStream_t stream = nullptr) const;
    void unbind_row() noexcept;
    [[nodiscard]] Tensor block_table() const;

    // Copies every mapped logical page's packed payload to host and returns every physical page
    // to the pool, leaving this allocation empty (as release() would). The image is keyed by
    // LOGICAL page order: which physical page a logical page occupied is pool state that must not
    // survive parking, so attach_from_host() takes fresh pages and copies the payloads back
    // instead. Requires an unbound allocation and is rejected under RAM KV, whose archive and
    // retrieval window are a different (single-request) host scheme. The caller must have made
    // this pool's device current, since the payload copies are issued on it.
    [[nodiscard]] std::unique_ptr<PagedKVImage> detach_to_host(cudaStream_t stream = nullptr);
    // Restores a detached image into this allocation, which must be freshly reserved in the
    // image's own pool with at least image.pages() of entitlement and no mapped pages yet.
    void attach_from_host(const PagedKVImage& image, cudaStream_t stream = nullptr);

    void release() noexcept;
    // Ranking is storage-policy input supplied by the model family, not interpreted by core.
    void prefer_pages(std::span<const std::int32_t> ranked_pages);
    // A PP prefill wave may contain multiple not-yet-written chunks in already mapped pages.
    void require_pages(std::uint32_t begin, std::uint32_t end);
    [[nodiscard]] std::uint64_t ram_transfer_bytes() const noexcept;

private:
    friend class PagedKVPool;
    friend void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

    PagedKVAllocation(PagedKVPool& pool, std::uint32_t page_entitlement);
    void publish_range(std::uint32_t first_page, std::uint32_t page_count,
                       cudaStream_t stream) const;

    PagedKVPool* pool_ = nullptr;
    std::vector<std::int32_t> page_ids_;
    std::uint32_t page_entitlement_ = 0;
    std::int32_t bound_row_         = -1;
    struct RamState;
    std::unique_ptr<RamState> ram_;
    void materialize_ram_pages(std::uint32_t pages, cudaStream_t stream);
    void publish_ram_mapping(cudaStream_t stream) const;
};

/**
 * Host-resident image of one allocation's logical page payload.
 *
 * Layout is plane-major then logical page: plane p occupies plane_offsets()[p] ..
 * plane_offsets()[p] + plane_page_bytes()[p] * pages(), and logical page q of that plane sits at
 * the start of that span plus q * plane_page_bytes()[p]. Nothing here names a physical page or a
 * pool offset, so an image is portable across allocations of one pool but not across pools (the
 * plane geometry is a pool property, and `pool` witnesses that at attach time).
 */
class PagedKVImage {
public:
    PagedKVImage() noexcept;
    ~PagedKVImage();

    PagedKVImage(const PagedKVImage&)            = delete;
    PagedKVImage& operator=(const PagedKVImage&) = delete;
    PagedKVImage(PagedKVImage&&) noexcept;
    PagedKVImage& operator=(PagedKVImage&&) noexcept;

    [[nodiscard]] bool valid() const noexcept { return pool_ != nullptr; }
    [[nodiscard]] std::uint32_t pages() const noexcept { return pages_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return data_.size(); }
    [[nodiscard]] const std::vector<std::size_t>& plane_offsets() const noexcept {
        return plane_offsets_;
    }
    [[nodiscard]] const std::vector<std::size_t>& plane_page_bytes() const noexcept {
        return plane_page_bytes_;
    }

private:
    friend class PagedKVAllocation;

    PagedKVPool* pool_ = nullptr;
    std::uint32_t pages_ = 0;
    std::vector<std::size_t> plane_offsets_;
    std::vector<std::size_t> plane_page_bytes_;
    std::vector<unsigned char> data_;
};

struct PagedKVReservation {
    PagedKVPool* pool              = nullptr;
    std::uint32_t page_entitlement = 0;
};

// Reserves every requested pool or leaves all pools unchanged.
[[nodiscard]] std::vector<PagedKVAllocation>
reserve_paged_kv_bundle(std::span<const PagedKVReservation> reservations);

struct PagedKVResize {
    PagedKVAllocation* allocation  = nullptr;
    std::uint32_t mapped_pages     = 0;
    std::uint32_t page_entitlement = 0;
};

// Atomically validates a retained-claim/truncate resize vector, then applies it at a GPU boundary.
void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

} // namespace ninfer
