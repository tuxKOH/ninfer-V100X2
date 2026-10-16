#include "core/paged_kv_cache.h"

#include "core/device.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <sys/mman.h>

namespace ninfer {
namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void validate_distinct_pools(std::span<const PagedKVReservation> reservations) {
    for (std::size_t i = 0; i < reservations.size(); ++i) {
        if (reservations[i].pool == nullptr || reservations[i].page_entitlement == 0) {
            throw std::invalid_argument("Paged KV reservation must name a pool and pages");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (reservations[i].pool == reservations[j].pool) {
                throw std::invalid_argument("Paged KV bundle contains the same pool twice");
            }
        }
    }
}

} // namespace

PagedKVPoolLayout plan_paged_kv_pool(LayoutBuilder& builder, const PagedKVPoolSpec& spec) {
    const std::int32_t physical_pages = checked_i32(spec.page_group_count, "Paged KV page count");
    const std::int32_t logical_pages =
        checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    if (spec.planes.empty()) { throw std::invalid_argument("Paged KV pool must contain planes"); }

    PagedKVPoolLayout layout;
    layout.spec = spec;
    layout.planes.reserve(spec.planes.size());
    for (std::size_t index = 0; index < spec.planes.size(); ++index) {
        const PagedKVPlaneSpec& plane = spec.planes[index];
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Paged KV plane extents must be positive");
        }
        const std::string label = "Paged KV plane " + std::to_string(index);
        PagedKVPlaneLayout planned;
        planned.spec = plane;
        if (spec.plane_order == PagedKVPlaneOrder::PageMajor) {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, plane.head_extent, physical_pages},
                plane.alignment, label);
        } else {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, physical_pages, plane.head_extent},
                plane.alignment, label);
        }
        layout.planes.push_back(planned);
    }
    layout.block_tables = builder.add_tensor(DType::I32, {logical_pages, spec.table_rows}, 256,
                                             "Paged KV block tables");
    if (spec.ram_enabled) {
        if (spec.table_rows != 1 || spec.plane_order != PagedKVPlaneOrder::PageMajor) {
            throw std::invalid_argument("RAM KV requires one PageMajor table row");
        }
        layout.read_indices = builder.add_tensor(DType::I32, {logical_pages, 1}, 256,
                                                 "RAM KV read indices");
        layout.read_tables = builder.add_tensor(DType::I32, {logical_pages, 1}, 256,
                                                "RAM KV chronological read table");
    }
    return layout;
}

std::size_t PagedKVPoolLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const PagedKVPlaneLayout& plane : planes) { total += plane.storage.region.bytes; }
    return total;
}

std::size_t PagedKVPoolLayout::metadata_bytes() const noexcept {
    return block_tables.region.bytes + read_indices.region.bytes + read_tables.region.bytes;
}

PagedKVPool::PagedKVPool(DeviceSpan backing, const PagedKVPoolLayout& layout)
    : spec_(layout.spec), block_tables_(layout.block_tables.bind(backing)),
      row_in_use_(static_cast<std::size_t>(layout.spec.table_rows), false) {
    if (spec_.ram_enabled) {
        read_indices_ = layout.read_indices.bind(backing);
        read_tables_ = layout.read_tables.bind(backing);
    }
    if (layout.planes.size() != spec_.planes.size() || layout.planes.empty()) {
        throw std::invalid_argument("Paged KV layout plane inventory is inconsistent");
    }
    if (block_tables_.dtype != DType::I32 ||
        block_tables_.ne[0] !=
            checked_i32(spec_.logical_page_capacity, "Paged KV logical page capacity") ||
        block_tables_.ne[1] != spec_.table_rows) {
        throw std::logic_error("Paged KV block-table layout is inconsistent");
    }

    planes_.reserve(layout.planes.size());
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const PagedKVPlaneLayout& plane = layout.planes[index];
        if (plane.spec.dtype != spec_.planes[index].dtype ||
            plane.spec.leading_extent != spec_.planes[index].leading_extent ||
            plane.spec.head_extent != spec_.planes[index].head_extent) {
            throw std::logic_error("Paged KV plane layout does not match its spec");
        }
        planes_.push_back(plane.storage.bind(backing));
    }

    free_page_ids_.reserve(spec_.page_group_count);
    for (std::uint32_t page = 0; page < spec_.page_group_count; ++page) {
        free_page_ids_.push_back(static_cast<std::int32_t>(page));
    }
}

std::uint32_t PagedKVPool::page_group_count() const noexcept { return spec_.page_group_count; }

std::uint32_t PagedKVPool::logical_page_capacity() const noexcept {
    return spec_.logical_page_capacity;
}

std::int32_t PagedKVPool::table_row_count() const noexcept { return spec_.table_rows; }

std::size_t PagedKVPool::plane_count() const noexcept { return planes_.size(); }

const Tensor& PagedKVPool::plane(std::size_t index) const { return planes_.at(index); }

const Tensor& PagedKVPool::block_tables() const noexcept { return block_tables_; }

std::size_t PagedKVPool::ram_archive_bytes() const noexcept {
    std::size_t page_bytes = 0;
    for (const Tensor& plane : planes_) { page_bytes += plane.nb[3]; }
    return page_bytes * logical_page_capacity();
}

Tensor PagedKVPool::block_table_row(std::int32_t row) const {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV block-table row out of range");
    }
    return block_tables_.slice(1, row, 1).view(
        {static_cast<std::int32_t>(logical_page_capacity())});
}

std::uint32_t PagedKVPool::entitled_pages() const noexcept { return entitled_pages_; }

std::uint32_t PagedKVPool::mapped_pages() const noexcept { return mapped_pages_; }

std::uint32_t PagedKVPool::free_pages() const noexcept {
    return static_cast<std::uint32_t>(free_page_ids_.size());
}

bool PagedKVPool::can_reserve(std::uint32_t page_entitlement) const noexcept {
    return page_entitlement != 0 && page_entitlement <= logical_page_capacity() &&
           page_entitlement <= page_group_count() - entitled_pages_;
}

bool PagedKVPool::can_replace_entitlement(std::uint32_t old_pages,
                                          std::uint32_t new_pages) const noexcept {
    return old_pages <= entitled_pages_ && new_pages <= logical_page_capacity() &&
           new_pages <= page_group_count() - (entitled_pages_ - old_pages);
}

PagedKVAllocation PagedKVPool::reserve(std::uint32_t page_entitlement) {
    if (!can_reserve(page_entitlement)) { throw std::bad_alloc(); }
    PagedKVAllocation allocation(*this, page_entitlement);
    add_entitlement(page_entitlement);
    return allocation;
}

void PagedKVPool::zero_pages(std::span<const std::int32_t> page_ids, cudaStream_t stream) {
    if (page_ids.empty()) { return; }

    std::vector<std::int32_t> sorted(page_ids.begin(), page_ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= static_cast<std::int32_t>(page_group_count())) {
        throw std::out_of_range("Paged KV physical page is out of range");
    }
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("Paged KV physical pages must be distinct");
    }

    const auto zero_run = [&](std::int32_t first, std::int32_t count) {
        for (const Tensor& plane : planes_) {
            auto* base = static_cast<unsigned char*>(plane.data);
            if (spec_.plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemsetAsync(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                                           static_cast<std::size_t>(count) * plane.nb[3], stream));
            } else {
                CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::int64_t>(first) * plane.nb[2],
                                             plane.nb[3], 0,
                                             static_cast<std::size_t>(count) * plane.nb[2],
                                             static_cast<std::size_t>(plane.ne[3]), stream));
            }
        }
    };

    std::size_t begin = 0;
    while (begin < sorted.size()) {
        std::size_t end = begin + 1;
        while (end < sorted.size() && sorted[end] == sorted[end - 1] + 1) { ++end; }
        zero_run(sorted[begin], static_cast<std::int32_t>(end - begin));
        begin = end;
    }
}

std::vector<std::int32_t> PagedKVPool::take_pages(std::uint32_t count,
                                                  std::int32_t preferred_first) {
    if (count == 0) { return {}; }
    if (count > free_page_ids_.size()) {
        throw std::logic_error("Paged KV entitlement could not be materialized");
    }

    const auto run_at = [&](std::size_t begin) {
        if (begin + count > free_page_ids_.size()) { return false; }
        for (std::uint32_t offset = 1; offset < count; ++offset) {
            if (free_page_ids_[begin + offset] !=
                free_page_ids_[begin] + static_cast<std::int32_t>(offset)) {
                return false;
            }
        }
        return true;
    };

    std::size_t selected = free_page_ids_.size();
    if (preferred_first >= 0) {
        const auto it =
            std::lower_bound(free_page_ids_.begin(), free_page_ids_.end(), preferred_first);
        if (it != free_page_ids_.end() && *it == preferred_first) {
            const auto begin = static_cast<std::size_t>(it - free_page_ids_.begin());
            if (run_at(begin)) { selected = begin; }
        }
    }
    if (selected == free_page_ids_.size()) {
        for (std::size_t begin = 0; begin + count <= free_page_ids_.size(); ++begin) {
            if (run_at(begin)) {
                selected = begin;
                break;
            }
        }
    }

    std::vector<std::int32_t> out;
    out.reserve(count);
    if (selected != free_page_ids_.size()) {
        const auto first = free_page_ids_.begin() + static_cast<std::ptrdiff_t>(selected);
        const auto last  = first + static_cast<std::ptrdiff_t>(count);
        out.insert(out.end(), first, last);
        free_page_ids_.erase(first, last);
    } else {
        const auto last = free_page_ids_.begin() + static_cast<std::ptrdiff_t>(count);
        out.insert(out.end(), free_page_ids_.begin(), last);
        free_page_ids_.erase(free_page_ids_.begin(), last);
    }
    mapped_pages_ += count;
    return out;
}

void PagedKVPool::return_pages(std::span<const std::int32_t> pages) noexcept {
    if (pages.empty()) { return; }
    free_page_ids_.insert(free_page_ids_.end(), pages.begin(), pages.end());
    std::sort(free_page_ids_.begin(), free_page_ids_.end());
    mapped_pages_ -= static_cast<std::uint32_t>(pages.size());
}

void PagedKVPool::add_entitlement(std::uint32_t pages) noexcept { entitled_pages_ += pages; }

void PagedKVPool::replace_entitlement(std::uint32_t old_pages, std::uint32_t new_pages) noexcept {
    entitled_pages_ = entitled_pages_ - old_pages + new_pages;
}

void PagedKVPool::acquire_row(std::int32_t row) {
    if (row < 0 || row >= table_row_count()) {
        throw std::out_of_range("Paged KV block-table row out of range");
    }
    if (row_in_use_[static_cast<std::size_t>(row)]) {
        throw std::logic_error("Paged KV block-table row is already bound");
    }
    row_in_use_[static_cast<std::size_t>(row)] = true;
}

void PagedKVPool::release_row(std::int32_t row) noexcept {
    row_in_use_[static_cast<std::size_t>(row)] = false;
}

struct PagedKVAllocation::RamState {
    void* archive = MAP_FAILED;
    std::size_t bytes = 0;
    std::vector<std::size_t> plane_offsets;
    std::vector<bool> archived;
    std::vector<std::int32_t> preferred;
    mutable std::vector<std::int32_t> indices;
    mutable std::vector<std::int32_t> table;
    std::uint32_t resident = 0;
    std::uint64_t transfers = 0;
    bool dirty = true;
    std::uint32_t required_begin = 0;
    std::uint32_t required_end = 0;

    explicit RamState(const PagedKVPool& pool) : bytes(pool.ram_archive_bytes()),
        archived(pool.logical_page_capacity(), false),
        indices(pool.logical_page_capacity(), -1), table(pool.logical_page_capacity(), 0) {
        std::size_t offset = 0;
        for (std::size_t p = 0; p < pool.plane_count(); ++p) {
            plane_offsets.push_back(offset);
            offset += pool.plane(p).nb[3] * pool.logical_page_capacity();
        }
        archive = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (archive == MAP_FAILED) { throw std::bad_alloc(); }
    }
    ~RamState() { if (archive != MAP_FAILED) { munmap(archive, bytes); } }
};

PagedKVAllocation::PagedKVAllocation() noexcept = default;

PagedKVAllocation::PagedKVAllocation(PagedKVPool& pool, std::uint32_t page_entitlement)
    : pool_(&pool), page_entitlement_(page_entitlement) {
    page_ids_.reserve(pool.ram_enabled() ? pool.logical_page_capacity() : page_entitlement);
    if (pool.ram_enabled()) { ram_ = std::make_unique<RamState>(pool); }
}

PagedKVAllocation::~PagedKVAllocation() { release(); }

PagedKVAllocation::PagedKVAllocation(PagedKVAllocation&& other) noexcept
    : pool_(other.pool_), page_ids_(std::move(other.page_ids_)),
      page_entitlement_(other.page_entitlement_), bound_row_(other.bound_row_),
      ram_(std::move(other.ram_)) {
    other.pool_             = nullptr;
    other.page_entitlement_ = 0;
    other.bound_row_        = -1;
}

PagedKVAllocation& PagedKVAllocation::operator=(PagedKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    pool_                   = other.pool_;
    page_ids_               = std::move(other.page_ids_);
    page_entitlement_       = other.page_entitlement_;
    bound_row_              = other.bound_row_;
    ram_                    = std::move(other.ram_);
    other.pool_             = nullptr;
    other.page_entitlement_ = 0;
    other.bound_row_        = -1;
    return *this;
}

bool PagedKVAllocation::valid() const noexcept { return pool_ != nullptr; }

std::uint32_t PagedKVAllocation::page_entitlement() const noexcept { return page_entitlement_; }

std::uint32_t PagedKVAllocation::mapped_page_count() const noexcept {
    return ram_ ? ram_->resident : static_cast<std::uint32_t>(page_ids_.size());
}

std::uint32_t PagedKVAllocation::mapped_token_capacity() const noexcept {
    return static_cast<std::uint32_t>(page_ids_.size()) * kPagedKVPageSize;
}

std::size_t PagedKVAllocation::mapped_payload_bytes() const noexcept {
    if (!valid()) { return 0; }
    std::size_t per_page = 0;
    for (std::size_t p = 0; p < pool_->plane_count(); ++p) {
        per_page += static_cast<std::size_t>(pool_->plane(p).nb[3]);
    }
    return per_page * page_ids_.size();
}

std::int32_t PagedKVAllocation::bound_row() const noexcept { return bound_row_; }

std::span<const std::int32_t> PagedKVAllocation::page_ids() const noexcept { return page_ids_; }

bool PagedKVAllocation::belongs_to(const PagedKVPool& pool) const noexcept {
    return pool_ == &pool;
}

void PagedKVAllocation::set_page_entitlement(std::uint32_t pages) {
    if (!valid() || pages < mapped_page_count()) {
        throw std::invalid_argument("Paged KV entitlement is smaller than mapped pages");
    }
    if (!pool_->can_replace_entitlement(page_entitlement_, pages)) { throw std::bad_alloc(); }
    page_ids_.reserve(pages);
    pool_->replace_entitlement(page_entitlement_, pages);
    page_entitlement_ = pages;
}

void PagedKVAllocation::cancel_unmapped_entitlement() noexcept {
    if (!valid()) { return; }
    if (ram_) { return; } // The fixed resident budget is also the retained-prefix growth claim.
    const std::uint32_t mapped = mapped_page_count();
    pool_->replace_entitlement(page_entitlement_, mapped);
    page_entitlement_ = mapped;
}

void PagedKVAllocation::materialize_pages(std::uint32_t pages, cudaStream_t stream) {
    if (ram_) { materialize_ram_pages(pages, stream); return; }
    if (!valid() || pages < mapped_page_count() || pages > page_entitlement_) {
        throw std::invalid_argument("Paged KV materialize extent is outside entitlement");
    }
    const std::uint32_t old_count = mapped_page_count();
    const std::uint32_t count     = pages - old_count;
    if (count == 0) { return; }
    const std::int32_t preferred =
        page_ids_.empty() ? -1 : static_cast<std::int32_t>(page_ids_.back() + 1);
    std::vector<std::int32_t> acquired = pool_->take_pages(count, preferred);
    page_ids_.insert(page_ids_.end(), acquired.begin(), acquired.end());
    if (bound_row_ >= 0) { publish_range(old_count, count, stream); }
}

void PagedKVAllocation::materialize_tokens(std::uint32_t tokens, cudaStream_t stream) {
    materialize_pages(pages_for_tokens(tokens), stream);
}

void PagedKVAllocation::trim_pages(std::uint32_t pages) {
    if (!valid()) { throw std::logic_error("Cannot trim an empty Paged KV allocation"); }
    if (ram_) {
        if (pages > page_ids_.size()) { throw std::invalid_argument("RAM KV trim exceeds extent"); }
        std::vector<std::int32_t> returned;
        for (std::size_t p = pages; p < page_ids_.size(); ++p) {
            if (page_ids_[p] >= 0) { returned.push_back(page_ids_[p]); }
            ram_->archived[p] = false;
        }
        pool_->return_pages(returned);
        ram_->resident -= static_cast<std::uint32_t>(returned.size());
        if (pages != page_ids_.size()) { ram_->dirty = true; }
        page_ids_.resize(pages);
        return;
    }
    if (pages > mapped_page_count()) {
        throw std::invalid_argument("Paged KV trim extent exceeds mapped pages");
    }
    if (pages == mapped_page_count()) { return; }
    pool_->return_pages(std::span<const std::int32_t>(
        page_ids_.data() + pages, static_cast<std::size_t>(mapped_page_count() - pages)));
    page_ids_.resize(pages);
}

void PagedKVAllocation::trim_tokens(std::uint32_t tokens) { trim_pages(pages_for_tokens(tokens)); }

void PagedKVAllocation::bind_row(std::int32_t row, cudaStream_t stream) {
    if (!valid()) { throw std::logic_error("Cannot bind an empty Paged KV allocation"); }
    if (bound_row_ >= 0) { throw std::logic_error("Paged KV allocation is already bound"); }
    pool_->acquire_row(row);
    bound_row_ = row;
    publish_mapping(stream);
}

void PagedKVAllocation::publish_mapping(cudaStream_t stream) const {
    if (bound_row_ < 0) { throw std::logic_error("Paged KV allocation is not bound"); }
    if (ram_) { publish_ram_mapping(stream); return; }
    publish_range(0, mapped_page_count(), stream);
}

void PagedKVAllocation::publish_range(std::uint32_t first_page, std::uint32_t page_count,
                                      cudaStream_t stream) const {
    if (page_count == 0) { return; }
    Tensor row         = pool_->block_table_row(bound_row_);
    auto* destination  = static_cast<std::int32_t*>(row.data) + first_page;
    const auto* source = page_ids_.data() + first_page;
    CUDA_CHECK(cudaMemcpyAsync(destination, source,
                               static_cast<std::size_t>(page_count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
}

void PagedKVAllocation::unbind_row() noexcept {
    if (bound_row_ < 0) { return; }
    pool_->release_row(bound_row_);
    bound_row_ = -1;
}

Tensor PagedKVAllocation::block_table() const {
    if (bound_row_ < 0) { throw std::logic_error("Paged KV allocation is not bound"); }
    return pool_->block_table_row(bound_row_);
}

void PagedKVAllocation::release() noexcept {
    if (!valid()) { return; }
    unbind_row();
    if (ram_) {
        std::vector<std::int32_t> returned;
        for (auto id : page_ids_) { if (id >= 0) { returned.push_back(id); } }
        pool_->return_pages(returned);
        ram_.reset();
    } else { pool_->return_pages(page_ids_); }
    pool_->replace_entitlement(page_entitlement_, 0);
    page_ids_.clear();
    page_entitlement_ = 0;
    pool_             = nullptr;
}

PagedKVImage::PagedKVImage() noexcept = default;

PagedKVImage::~PagedKVImage() = default;

PagedKVImage::PagedKVImage(PagedKVImage&& other) noexcept
    : pool_(other.pool_), pages_(other.pages_),
      plane_offsets_(std::move(other.plane_offsets_)),
      plane_page_bytes_(std::move(other.plane_page_bytes_)), data_(std::move(other.data_)) {
    other.pool_  = nullptr;
    other.pages_ = 0;
}

PagedKVImage& PagedKVImage::operator=(PagedKVImage&& other) noexcept {
    if (this == &other) { return *this; }
    pool_             = other.pool_;
    pages_            = other.pages_;
    plane_offsets_    = std::move(other.plane_offsets_);
    plane_page_bytes_ = std::move(other.plane_page_bytes_);
    data_             = std::move(other.data_);
    other.pool_       = nullptr;
    other.pages_      = 0;
    return *this;
}

std::unique_ptr<PagedKVImage> PagedKVAllocation::detach_to_host(cudaStream_t stream) {
    if (!valid()) { throw std::logic_error("Cannot detach an empty Paged KV allocation"); }
    if (ram_) { throw std::logic_error("Paged KV detach is not supported with RAM KV"); }
    if (bound_row_ >= 0) {
        throw std::logic_error("Paged KV detach requires an unbound allocation");
    }

    auto image         = std::make_unique<PagedKVImage>();
    image->pool_       = pool_;
    image->pages_      = static_cast<std::uint32_t>(page_ids_.size());
    std::size_t offset = 0;
    for (std::size_t p = 0; p < pool_->plane_count(); ++p) {
        const std::size_t page_bytes = static_cast<std::size_t>(pool_->plane(p).nb[3]);
        image->plane_page_bytes_.push_back(page_bytes);
        image->plane_offsets_.push_back(offset);
        offset += page_bytes * image->pages_;
    }
    // Every byte written below, so the value-initialization pass is pure overhead. It stays
    // because a partially failed detach must not hand back uninitialized host memory.
    image->data_.resize(offset);

    for (std::size_t p = 0; p < pool_->plane_count(); ++p) {
        const Tensor& plane          = pool_->plane(p);
        const std::size_t page_bytes = image->plane_page_bytes_[p];
        unsigned char* host          = image->data_.data() + image->plane_offsets_[p];
        for (std::uint32_t page = 0; page < image->pages_; ++page) {
            const std::int32_t id = page_ids_[page];
            if (id < 0) { throw std::logic_error("Paged KV detach found an unmapped logical page"); }
            CUDA_CHECK(cudaMemcpyAsync(
                host + static_cast<std::size_t>(page) * page_bytes,
                static_cast<const unsigned char*>(plane.data) +
                    static_cast<std::size_t>(id) * page_bytes,
                page_bytes, cudaMemcpyDeviceToHost, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    release();
    return image;
}

void PagedKVAllocation::attach_from_host(const PagedKVImage& image, cudaStream_t stream) {
    if (!valid() || !image.valid()) {
        throw std::invalid_argument("Paged KV attach requires a live allocation and image");
    }
    if (image.pool_ != pool_) {
        throw std::invalid_argument("Paged KV image belongs to a different pool");
    }
    if (mapped_page_count() != 0) {
        throw std::logic_error("Paged KV attach requires an allocation with no mapped pages");
    }
    if (image.pages_ > page_entitlement_) {
        throw std::invalid_argument("Paged KV image exceeds the allocation entitlement");
    }
    if (image.plane_page_bytes_.size() != pool_->plane_count()) {
        throw std::invalid_argument("Paged KV image plane count does not match the pool");
    }
    for (std::size_t p = 0; p < pool_->plane_count(); ++p) {
        if (image.plane_page_bytes_[p] != static_cast<std::size_t>(pool_->plane(p).nb[3])) {
            throw std::invalid_argument("Paged KV image plane geometry does not match the pool");
        }
    }

    materialize_pages(image.pages_, stream);
    for (std::size_t p = 0; p < pool_->plane_count(); ++p) {
        const Tensor& plane          = pool_->plane(p);
        const std::size_t page_bytes = image.plane_page_bytes_[p];
        const unsigned char* host    = image.data_.data() + image.plane_offsets_[p];
        for (std::uint32_t page = 0; page < image.pages_; ++page) {
            const std::int32_t id = page_ids_[page];
            if (id < 0) { throw std::logic_error("Paged KV image attach materialized no page"); }
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<unsigned char*>(plane.data) +
                    static_cast<std::size_t>(id) * page_bytes,
                host + static_cast<std::size_t>(page) * page_bytes,
                page_bytes, cudaMemcpyHostToDevice, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (bound_row_ >= 0) { publish_mapping(stream); }
}

void PagedKVAllocation::prefer_pages(std::span<const std::int32_t> ranked_pages) {
    if (!ram_) { throw std::logic_error("page ranking requires RAM KV"); }
    ram_->preferred.assign(ranked_pages.begin(), ranked_pages.end());
    ram_->dirty = true;
}

void PagedKVAllocation::require_pages(std::uint32_t begin, std::uint32_t end) {
    if (!ram_ || begin > end || end > pool_->logical_page_capacity() ||
        end - begin > page_entitlement_) {
        throw std::invalid_argument("RAM KV required page span is invalid");
    }
    if (begin != ram_->required_begin || end != ram_->required_end) { ram_->dirty = true; }
    ram_->required_begin = begin;
    ram_->required_end = end;
}

std::uint64_t PagedKVAllocation::ram_transfer_bytes() const noexcept {
    return ram_ ? ram_->transfers : 0;
}

void PagedKVAllocation::materialize_ram_pages(std::uint32_t pages, cudaStream_t stream) {
    if (pages < page_ids_.size() || pages > pool_->logical_page_capacity()) {
        throw std::invalid_argument("RAM KV materialize extent is invalid");
    }
    const auto old_pages = static_cast<std::uint32_t>(page_ids_.size());
    if (pages == old_pages && !ram_->dirty) { return; }
    // A previous eager prefill unit may still be consuming this allocation's host-to-device
    // metadata upload. Do not mutate its stable host source until the stream boundary settles.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (pages - old_pages > page_entitlement_) {
        throw std::invalid_argument("RAM KV append exceeds resident window");
    }
    page_ids_.resize(pages, -1);
    std::vector<bool> wanted(pages, false);
    std::uint32_t count = 0;
    const auto add = [&](std::uint32_t p) {
        if (p < pages && !wanted[p] && count < page_entitlement_) {
            wanted[p] = true; ++count;
        }
    };
    // Every append page is mandatory, then a 4K recent band and a 512-token sink.
    // Existing selected pages stay physically in place. Original order is restored only in
    // the read table, never by moving resident payloads.
    for (auto p = ram_->required_begin; p < std::min(pages, ram_->required_end); ++p) { add(p); }
    for (auto p = old_pages; p < pages; ++p) { add(p); }
    for (auto p = pages; p > (pages > 64 ? pages - 64 : 0);) { add(--p); }
    for (std::uint32_t p = 0; p < std::min(8U, pages); ++p) { add(p); }
    for (auto p : ram_->preferred) { if (p >= 0) { add(static_cast<std::uint32_t>(p)); } }
    for (auto p = pages; p != 0;) { add(--p); }

    const auto copy_run = [&](std::uint32_t logical, std::int32_t physical, std::uint32_t count, bool to_ram) {
        for (std::size_t p = 0; p < pool_->plane_count(); ++p) {
            const Tensor& plane = pool_->plane(p);
            const auto page_bytes = static_cast<std::size_t>(plane.nb[3]);
            const auto bytes = page_bytes * count;
            auto* host = static_cast<unsigned char*>(ram_->archive) + ram_->plane_offsets[p] +
                         static_cast<std::size_t>(logical) * page_bytes;
            auto* gpu = static_cast<unsigned char*>(plane.data) + physical * page_bytes;
            CUDA_CHECK(cudaMemcpyAsync(to_ram ? static_cast<void*>(host) : gpu,
                                       to_ram ? static_cast<const void*>(gpu) : host, bytes,
                                       to_ram ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice,
                                       stream));
            ram_->transfers += bytes;
        }
    };
    std::vector<std::int32_t> evicted;
    for (std::uint32_t p = 0; p < pages;) {
        if (page_ids_[p] < 0 || wanted[p]) { ++p; continue; }
        std::uint32_t end = p + 1;
        while (end < pages && !wanted[end] && page_ids_[end] == page_ids_[end - 1] + 1) { ++end; }
        copy_run(p, page_ids_[p], end - p, true);
        for (; p < end; ++p) {
            ram_->archived[p] = true;
            evicted.push_back(page_ids_[p]);
            page_ids_[p] = -1;
        }
    }
    // The host archive must be ready before a subsequent load or allocator reuse. Boundaries
    // run outside graph capture; no host payload pointer is embedded in a decode graph.
    if (!evicted.empty()) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    pool_->return_pages(evicted);
    ram_->resident -= static_cast<std::uint32_t>(evicted.size());
    std::vector<std::int32_t> fresh;
    std::vector<bool> load(pages, false);
    for (std::uint32_t p = 0; p < pages; ++p) {
        if (!wanted[p] || page_ids_[p] >= 0) { continue; }
        auto acquired = pool_->take_pages(1, -1);
        page_ids_[p] = acquired.front(); ++ram_->resident;
        if (ram_->archived[p]) { load[p] = true; }
        else { fresh.push_back(page_ids_[p]); }
    }
    for (std::uint32_t p = 0; p < pages;) {
        if (!load[p]) { ++p; continue; }
        std::uint32_t end = p + 1;
        while (end < pages && load[end] && page_ids_[end] == page_ids_[end - 1] + 1) { ++end; }
        copy_run(p, page_ids_[p], end - p, false);
        p = end;
    }
    // Flash staging can gather page-rounded future rows before applying its causal mask.
    // Keep their codes/scales finite (0), so a masked uninitialized FP16 NaN cannot contaminate
    // softmax. Coalesce fresh physical runs; never clear restored or already-resident pages.
    pool_->zero_pages(fresh, stream);
    if (bound_row_ >= 0) { publish_ram_mapping(stream); }
}

void PagedKVAllocation::publish_ram_mapping(cudaStream_t stream) const {
    auto& indices = ram_->indices;
    auto& table = ram_->table;
    std::fill(indices.begin(), indices.end(), -1);
    std::int32_t count = 0;
    for (std::size_t p = 0; p < page_ids_.size(); ++p) {
        if (page_ids_[p] >= 0) { indices[p] = count; table[count++] = page_ids_[p]; }
    }
    // Padding is a valid resident address, but always causally masked by remapped positions.
    std::fill(table.begin() + count, table.end(), count == 0 ? 0 : table[count - 1]);
    Tensor original = pool_->block_table_row(bound_row_);
    CUDA_CHECK(cudaMemsetAsync(original.data, 0xff, original.bytes(), stream));
    if (!page_ids_.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(original.data, page_ids_.data(), page_ids_.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(pool_->read_indices().data, indices.data(),
                               indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool_->read_tables().data, table.data(), table.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    ram_->dirty = false;
}

std::vector<PagedKVAllocation>
reserve_paged_kv_bundle(std::span<const PagedKVReservation> reservations) {
    validate_distinct_pools(reservations);
    for (const PagedKVReservation& reservation : reservations) {
        if (!reservation.pool->can_reserve(reservation.page_entitlement)) {
            throw std::bad_alloc();
        }
    }

    std::vector<PagedKVAllocation> allocations;
    allocations.reserve(reservations.size());
    for (const PagedKVReservation& reservation : reservations) {
        allocations.push_back(reservation.pool->reserve(reservation.page_entitlement));
    }
    return allocations;
}

void resize_paged_kv_bundle(std::span<const PagedKVResize> changes) {
    for (std::size_t i = 0; i < changes.size(); ++i) {
        const PagedKVResize& change = changes[i];
        if (change.allocation == nullptr || !change.allocation->valid()) {
            throw std::invalid_argument("Paged KV resize must name a live allocation");
        }
        if (change.mapped_pages > change.allocation->mapped_page_count() ||
            change.mapped_pages > change.page_entitlement) {
            throw std::invalid_argument("Paged KV resize extents are inconsistent");
        }
        if (change.allocation->ram_ &&
            change.mapped_pages != change.allocation->mapped_page_count()) {
            throw std::invalid_argument("RAM KV resize changes only resident entitlement; trim logical tokens first");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (change.allocation == changes[j].allocation ||
                change.allocation->pool_ == changes[j].allocation->pool_) {
                throw std::invalid_argument("Paged KV resize contains the same pool twice");
            }
        }
        if (!change.allocation->pool_->can_replace_entitlement(change.allocation->page_entitlement_,
                                                               change.page_entitlement)) {
            throw std::bad_alloc();
        }
    }
    // Complete every potentially throwing host allocation before changing any pool accounting.
    for (const PagedKVResize& change : changes) {
        change.allocation->page_ids_.reserve(change.page_entitlement);
    }
    for (const PagedKVResize& change : changes) {
        if (!change.allocation->ram_) { change.allocation->trim_pages(change.mapped_pages); }
        change.allocation->set_page_entitlement(change.page_entitlement);
    }
}

} // namespace ninfer
