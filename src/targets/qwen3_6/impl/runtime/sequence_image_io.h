#pragma once

// Persistence for a complete parked conversation: KV page payloads, every rank's recurrent slot,
// the hidden states and the checkpoint lineage. One file is a self-describing image with explicit
// geometry, so a truncated or mismatched file is rejected before any device state is touched.
//
// Two callers share the format:
//   * the disk-backed shared prefix, written once after the leading block is captured and loaded
//     at startup so the first request after a restart does not re-prefill it;
//   * the RAM-to-disk spill tier, which writes the least-recently-used parked conversation(s) when
//     the host park budget is exhausted and keeps a metadata-only index in RAM for matching.

#include "targets/qwen3_6/impl/runtime/image_stream.h"
#include "targets/qwen3_6/impl/runtime/parked_sequence_store.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

inline constexpr std::uint32_t kSequenceImageMagic   = 0x3147514eU;  // 'NQG1'
inline constexpr std::uint32_t kSequenceImageVersion = 3U;  // v3 adds long-range checkpoints

// Per-rank geometry the image must match. Values come from the live decoder state, so a file
// written by a different model, quantization, TP width or page layout is refused.
struct SequenceImageGeometry {
    std::size_t rank_count              = 0;
    std::size_t slot_bytes              = 0;
    std::size_t hidden_bytes            = 0;
    std::size_t text_plane_count        = 0;
    std::size_t backend_plane_count     = 0;
    std::uint32_t text_page_capacity    = 0;
    std::uint32_t backend_page_capacity = 0;
    std::vector<std::size_t> text_plane_page_bytes;
    std::vector<std::size_t> backend_plane_page_bytes;
};

namespace sequence_image_io_detail {

inline void write_checkpoint(std::ostream& out, const HostCheckpoint& checkpoint) {
    using namespace image_stream;
    put_scalar<std::uint8_t>(out, static_cast<std::uint8_t>(checkpoint.kind));
    put_scalar<std::uint32_t>(out, checkpoint.frontier);
    put_vector<std::int32_t>(out, checkpoint.ledger);
    checkpoint.identity.save(out);
    put_size(out, checkpoint.state.size());
    for (const std::vector<std::byte>& state : checkpoint.state) {
        put_size(out, state.size());
        put_bytes(out, state.data(), state.size());
    }
    put_size(out, checkpoint.hidden.size());
    put_bytes(out, checkpoint.hidden.data(), checkpoint.hidden.size());
}

// A pinned boundary is optional: an entry may carry neither, one, or both. Absence is encoded
// explicitly so a file with no pinned checkpoint (a spilled conversation) stays readable.
inline void write_optional_checkpoint(std::ostream& out, const HostCheckpoint& checkpoint) {
    using namespace image_stream;
    put_scalar<std::uint8_t>(out, checkpoint.frontier != 0 ? 1U : 0U);
    if (checkpoint.frontier != 0) { write_checkpoint(out, checkpoint); }
}

inline HostCheckpoint read_checkpoint(std::istream& in, const SequenceImageGeometry& geometry) {
    using namespace image_stream;
    constexpr std::size_t kMaximumTokens = 1U << 20;
    HostCheckpoint checkpoint;
    checkpoint.kind     = static_cast<RewriteCheckpointKind>(get_scalar<std::uint8_t>(in));
    checkpoint.frontier = get_scalar<std::uint32_t>(in);
    checkpoint.ledger   = get_vector<TokenId>(in, kMaximumTokens);
    checkpoint.identity = ResidentPrefixIdentity::load(in);
    const std::size_t ranks = get_size(in, 64);
    if (ranks != geometry.rank_count) {
        throw std::runtime_error("sequence image checkpoint rank count does not match the engine");
    }
    checkpoint.state.reserve(ranks);
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        const std::size_t size = get_size(in, std::numeric_limits<std::size_t>::max());
        if (size != geometry.slot_bytes) {
            throw std::runtime_error("sequence image checkpoint slot size does not match the engine");
        }
        std::vector<std::byte> state(size);
        get_bytes(in, state.data(), size);
        checkpoint.state.push_back(std::move(state));
    }
    const std::size_t hidden = get_size(in, std::numeric_limits<std::size_t>::max());
    if (hidden != geometry.hidden_bytes) {
        throw std::runtime_error("sequence image checkpoint hidden size does not match the engine");
    }
    checkpoint.hidden.resize(hidden);
    get_bytes(in, checkpoint.hidden.data(), hidden);
    checkpoint.bytes = checkpoint.ledger.size() * sizeof(TokenId) + checkpoint.identity.bytes() +
                       checkpoint.hidden.size();
    for (const auto& state : checkpoint.state) { checkpoint.bytes += state.size(); }
    return checkpoint;
}

inline void write_image(std::ostream& out, const PagedKVImage& image) {
    using namespace image_stream;
    put_scalar<std::uint32_t>(out, image.pages());
    put_size(out, image.plane_page_bytes().size());
    for (const std::size_t bytes : image.plane_page_bytes()) { put_size(out, bytes); }
    put_size(out, image.bytes());
    // PagedKVImage keeps its payload private; plane_offsets() exposes the layout, and the payload
    // is plane-major with no gaps, so re-read it through the pool-independent accessor.
    put_bytes(out, image.payload().data(), image.bytes());
}

inline std::unique_ptr<PagedKVImage> read_image(std::istream& in, PagedKVPool* pool,
                                                std::size_t expected_planes,
                                                std::uint32_t page_capacity,
                                                const std::vector<std::size_t>& plane_bytes) {
    using namespace image_stream;
    const std::uint32_t pages = get_scalar<std::uint32_t>(in);
    if (pool == nullptr || pages == 0 || pages > page_capacity) {
        throw std::runtime_error("sequence image KV page count does not match the engine");
    }
    const std::size_t planes = get_size(in, 64);
    if (planes != expected_planes || planes != plane_bytes.size()) {
        throw std::runtime_error("sequence image KV plane count does not match the engine");
    }
    std::vector<std::size_t> stored_bytes(planes);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        stored_bytes[plane] = get_size(in, std::numeric_limits<std::size_t>::max());
    }
    const std::size_t size = get_size(in, std::numeric_limits<std::size_t>::max());
    std::vector<unsigned char> data(size);
    get_bytes(in, data.data(), size);
    return PagedKVImage::from_host(*pool, pages, std::move(stored_bytes), std::move(data));
}

} // namespace sequence_image_io_detail

namespace sequence_image_io_detail {

inline void write_entry(std::ostream& out, const ParkedSequence& entry) {
    using namespace image_stream;
        put_scalar<std::uint32_t>(out, entry.execution_frontier);
        put_scalar<std::uint32_t>(out, entry.ledger_frontier);
        put_scalar<std::uint32_t>(out, entry.mtp_kv_valid);
        put_scalar<std::int32_t>(out, entry.rope_delta);
        put_scalar<std::uint8_t>(out, entry.tail_hidden_valid ? 1U : 0U);
        put_scalar<std::uint32_t>(out, entry.mtp_draft_count);
        for (const TokenId draft : entry.mtp_drafts) { put_scalar<std::int32_t>(out, draft); }
        put_vector<std::int32_t>(out, entry.ledger);
        entry.prefix_identity.save(out);

        put_size(out, entry.linear_state.size());
        for (const std::vector<std::byte>& state : entry.linear_state) {
            put_size(out, state.size());
            put_bytes(out, state.data(), state.size());
        }
        put_size(out, entry.tail_hidden.size());
        for (const std::vector<std::byte>& hidden : entry.tail_hidden) {
            put_size(out, hidden.size());
            put_bytes(out, hidden.data(), hidden.size());
        }
        put_size(out, entry.text.size());
        for (const std::unique_ptr<PagedKVImage>& image : entry.text) {
            if (image == nullptr) { throw std::runtime_error("sequence image has no text KV payload"); }
            sequence_image_io_detail::write_image(out, *image);
        }
        put_size(out, entry.backend.size());
        for (const std::unique_ptr<PagedKVImage>& image : entry.backend) {
            if (image == nullptr) { throw std::runtime_error("sequence image has no backend KV payload"); }
            sequence_image_io_detail::write_image(out, *image);
        }

        put_scalar<std::uint8_t>(out, entry.checkpoint_valid ? 1U : 0U);
        if (entry.checkpoint_valid) {
            HostCheckpoint current;
            current.kind     = entry.checkpoint_kind;
            current.frontier = entry.checkpoint_frontier;
            current.ledger   = entry.checkpoint_ledger;
            current.identity = entry.checkpoint_identity;
            current.state    = entry.checkpoint_state;
            current.hidden   = entry.checkpoint_hidden;
            sequence_image_io_detail::write_checkpoint(out, current);
        }
        const auto write_history = [&](const std::vector<HostCheckpoint>& history) {
            put_size(out, history.size());
            for (const HostCheckpoint& checkpoint : history) {
                sequence_image_io_detail::write_checkpoint(out, checkpoint);
            }
        };
        write_history(entry.checkpoint_history);
        write_history(entry.boundary_history);
        write_history(entry.response_history);
        write_history(entry.long_range_history);
        write_optional_checkpoint(out, entry.shared_prefix);
        write_optional_checkpoint(out, entry.tools_prefix);
}

} // namespace sequence_image_io_detail

// Writes through a temporary file and renames it, so an interrupted write never leaves a partial
// image at the destination path. Each entry is a complete parked conversation; the shared-prefix
// file holds the tool anchor and the leading block when both were captured.
inline void write_sequence_images(std::span<const ParkedSequence* const> entries,
                                  const std::string& path) {
    using namespace image_stream;
    const std::filesystem::path target(path);
    const std::filesystem::path temporary = target.string() + ".tmp";
    {
        std::error_code directory_error;
        if (!target.parent_path().empty()) {
            std::filesystem::create_directories(target.parent_path(), directory_error);
        }
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot open sequence image for writing: " + path); }
        put_scalar<std::uint32_t>(out, kSequenceImageMagic);
        put_scalar<std::uint32_t>(out, kSequenceImageVersion);
        put_size(out, entries.size());
        for (const ParkedSequence* entry : entries) {
            if (entry == nullptr) { throw std::runtime_error("sequence image entry is null"); }
            sequence_image_io_detail::write_entry(out, *entry);
        }
        out.flush();
        if (!out) { throw std::runtime_error("sequence image write failed: " + path); }
    }
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw std::runtime_error("cannot publish sequence image: " + path);
    }
}

inline void write_sequence_image(const ParkedSequence& entry, const std::string& path) {
    const ParkedSequence* pointers[] = {&entry};
    write_sequence_images(pointers, path);
}

namespace sequence_image_io_detail {

inline std::unique_ptr<ParkedSequence> read_entry(std::istream& in,
                                                  const SequenceImageGeometry& geometry,
                                                  std::span<PagedKVPool* const> text_pools,
                                                  std::span<PagedKVPool* const> backend_pools,
                                                  std::uint32_t version) {
    using namespace image_stream;
    auto entry                = std::make_unique<ParkedSequence>();
    entry->execution_frontier = get_scalar<std::uint32_t>(in);
    entry->ledger_frontier    = get_scalar<std::uint32_t>(in);
    entry->mtp_kv_valid       = get_scalar<std::uint32_t>(in);
    entry->rope_delta         = get_scalar<std::int32_t>(in);
    entry->tail_hidden_valid  = get_scalar<std::uint8_t>(in) != 0;
    entry->mtp_draft_count    = get_scalar<std::uint32_t>(in);
    if (entry->mtp_draft_count > entry->mtp_drafts.size()) {
        throw std::runtime_error("sequence image draft count is out of range");
    }
    for (TokenId& draft : entry->mtp_drafts) { draft = get_scalar<std::int32_t>(in); }
    entry->ledger           = get_vector<TokenId>(in, 1U << 20);
    entry->prefix_identity  = ResidentPrefixIdentity::load(in);

    const auto read_rank_blobs = [&](std::vector<std::vector<std::byte>>& blobs,
                                     std::size_t expected_size, std::size_t expected_count,
                                     const char* what) {
        const std::size_t count = get_size(in, 64);
        if (count != expected_count) {
            throw std::runtime_error(std::string("sequence image ") + what +
                                     " rank count does not match the engine");
        }
        blobs.reserve(count);
        for (std::size_t rank = 0; rank < count; ++rank) {
            const std::size_t size = get_size(in, std::numeric_limits<std::size_t>::max());
            if (size != expected_size) {
                throw std::runtime_error(std::string("sequence image ") + what +
                                         " size does not match the engine");
            }
            std::vector<std::byte> blob(size);
            get_bytes(in, blob.data(), size);
            blobs.push_back(std::move(blob));
        }
    };
    read_rank_blobs(entry->linear_state, geometry.slot_bytes, geometry.rank_count, "state");
    read_rank_blobs(entry->tail_hidden, geometry.hidden_bytes, 1, "tail hidden");

    // Each TP rank owns a distinct pool object, and attach_from_host rejects an image built for
    // another pool even when the geometry matches, so every rank is bound to its own pool.
    const auto read_images_ranked = [&](std::vector<std::unique_ptr<PagedKVImage>>& images,
                                        std::size_t expected_count,
                                        std::span<PagedKVPool* const> pools, std::size_t planes,
                                        std::uint32_t capacity,
                                        const std::vector<std::size_t>& plane_bytes,
                                        const char* what) {
        const std::size_t count = get_size(in, 64);
        if (count != expected_count) {
            throw std::runtime_error(std::string("sequence image ") + what +
                                     " rank count does not match the engine");
        }
        images.reserve(count);
        for (std::size_t rank = 0; rank < count; ++rank) {
            PagedKVPool* pool = rank < pools.size() ? pools[rank] : nullptr;
            images.push_back(
                read_image(in, pool, planes, capacity, plane_bytes));
        }
    };
    read_images_ranked(entry->text, geometry.rank_count, text_pools,
                       geometry.text_plane_count, geometry.text_page_capacity,
                       geometry.text_plane_page_bytes, "text KV");
    const bool has_backend = geometry.backend_plane_count != 0;
    read_images_ranked(entry->backend, has_backend ? geometry.rank_count : 0, backend_pools,
                       geometry.backend_plane_count, geometry.backend_page_capacity,
                       geometry.backend_plane_page_bytes, "backend KV");

    entry->checkpoint_valid = get_scalar<std::uint8_t>(in) != 0;
    if (entry->checkpoint_valid) {
        HostCheckpoint current  = sequence_image_io_detail::read_checkpoint(in, geometry);
        entry->checkpoint_kind  = current.kind;
        entry->checkpoint_frontier = current.frontier;
        entry->checkpoint_ledger   = std::move(current.ledger);
        entry->checkpoint_identity = std::move(current.identity);
        entry->checkpoint_state    = std::move(current.state);
        entry->checkpoint_hidden   = std::move(current.hidden);
    }
    const auto read_history = [&](std::vector<HostCheckpoint>& history) {
        const std::size_t count = get_size(in, 64);
        history.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            history.push_back(sequence_image_io_detail::read_checkpoint(in, geometry));
        }
    };
    read_history(entry->checkpoint_history);
    read_history(entry->boundary_history);
    // v1 predates pinned response checkpoints; v2 predates the long-range ladder. Reading an older
    // image keeps a cache written by a previous build usable instead of discarding it.
    if (version >= 2) { read_history(entry->response_history); }
    if (version >= 3) { read_history(entry->long_range_history); }
    const auto read_optional = [&](HostCheckpoint& target) {
        if (get_scalar<std::uint8_t>(in) != 0) {
            target = read_checkpoint(in, geometry);
        }
    };
    read_optional(entry->shared_prefix);
    read_optional(entry->tools_prefix);

    std::size_t bytes = 0;
    for (const auto& image : entry->text) { bytes += image->bytes(); }
    for (const auto& image : entry->backend) { bytes += image->bytes(); }
    for (const auto& state : entry->linear_state) { bytes += state.size(); }
    for (const auto& hidden : entry->tail_hidden) { bytes += hidden.size(); }
    for (const auto& state : entry->checkpoint_state) { bytes += state.size(); }
    bytes += entry->checkpoint_hidden.size();
    bytes += entry->checkpoint_ledger.size() * sizeof(TokenId) + entry->checkpoint_identity.bytes();
    for (const HostCheckpoint& checkpoint : entry->checkpoint_history) { bytes += checkpoint.bytes; }
    for (const HostCheckpoint& checkpoint : entry->boundary_history) { bytes += checkpoint.bytes; }
    for (const HostCheckpoint& checkpoint : entry->response_history) { bytes += checkpoint.bytes; }
    for (const HostCheckpoint& checkpoint : entry->long_range_history) { bytes += checkpoint.bytes; }
    bytes += entry->shared_prefix.bytes + entry->tools_prefix.bytes;
    entry->bytes = bytes;
    return entry;
}

} // namespace sequence_image_io_detail

inline std::vector<std::unique_ptr<ParkedSequence>> read_sequence_images(
    const std::string& path, const SequenceImageGeometry& geometry,
    std::span<PagedKVPool* const> text_pools, std::span<PagedKVPool* const> backend_pools) {
    using namespace image_stream;
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open sequence image: " + path); }
    if (get_scalar<std::uint32_t>(in) != kSequenceImageMagic) {
        throw std::runtime_error("not a sequence image: " + path);
    }
    const std::uint32_t version = get_scalar<std::uint32_t>(in);
    if (version == 0 || version > kSequenceImageVersion) {
        throw std::runtime_error("unsupported sequence image version: " + path);
    }
    const std::size_t count = get_size(in, 64);
    std::vector<std::unique_ptr<ParkedSequence>> entries;
    entries.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        entries.push_back(sequence_image_io_detail::read_entry(in, geometry, text_pools,
                                                               backend_pools, version));
    }
    if (in.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("sequence image has trailing data: " + path);
    }
    return entries;
}

// --- Metadata sidecar -------------------------------------------------------------------------
// A spilled conversation's image holds gigabytes of payload; startup only needs the matching
// fields. A small `<image>.meta` sidecar carries them so a restart can index every spill without
// reading the payloads. Written after the image, removed with it.

inline constexpr std::uint32_t kSequenceMetaMagic   = 0x314d514eU;  // 'NQM1'
inline constexpr std::uint32_t kSequenceMetaVersion = 1U;

namespace sequence_image_io_detail {

inline void write_meta_checkpoint(std::ostream& out, const HostCheckpoint& checkpoint) {
    using namespace image_stream;
    put_scalar<std::uint8_t>(out, static_cast<std::uint8_t>(checkpoint.kind));
    put_scalar<std::uint32_t>(out, checkpoint.frontier);
    put_vector<std::int32_t>(out, checkpoint.ledger);
    checkpoint.identity.save(out);
}

inline void read_meta_checkpoint(std::istream& in, HostCheckpoint& checkpoint) {
    using namespace image_stream;
    constexpr std::size_t kMaximumTokens = 1U << 20;
    checkpoint.kind     = static_cast<RewriteCheckpointKind>(get_scalar<std::uint8_t>(in));
    checkpoint.frontier = get_scalar<std::uint32_t>(in);
    checkpoint.ledger   = get_vector<TokenId>(in, kMaximumTokens);
    checkpoint.identity = ResidentPrefixIdentity::load(in);
}

inline void write_meta_history(std::ostream& out, const std::vector<HostCheckpoint>& history) {
    using namespace image_stream;
    put_size(out, history.size());
    for (const HostCheckpoint& checkpoint : history) { write_meta_checkpoint(out, checkpoint); }
}

inline void read_meta_history(std::istream& in, std::vector<HostCheckpoint>& history) {
    using namespace image_stream;
    const std::size_t count = get_size(in, 64);
    history.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        HostCheckpoint checkpoint;
        read_meta_checkpoint(in, checkpoint);
        history.push_back(std::move(checkpoint));
    }
}

} // namespace sequence_image_io_detail

inline void write_sequence_metadata(const ParkedSequence& entry, std::size_t payload_bytes,
                                    const SequenceImageGeometry& geometry,
                                    const std::string& path) {
    using namespace image_stream;
    const std::filesystem::path target(path);
    const std::filesystem::path temporary = target.string() + ".tmp";
    {
        std::error_code directory_error;
        if (!target.parent_path().empty()) {
            std::filesystem::create_directories(target.parent_path(), directory_error);
        }
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot open sequence metadata for writing: " + path); }
        put_scalar<std::uint32_t>(out, kSequenceMetaMagic);
        put_scalar<std::uint32_t>(out, kSequenceMetaVersion);
        // Geometry is validated at startup so an artifact from another model/layout is skipped
        // instead of failing a restore later.
        put_size(out, geometry.rank_count);
        put_size(out, geometry.slot_bytes);
        put_size(out, geometry.hidden_bytes);
        put_size(out, geometry.text_plane_count);
        put_size(out, geometry.backend_plane_count);
        put_scalar<std::uint32_t>(out, entry.execution_frontier);
        put_scalar<std::uint32_t>(out, entry.ledger_frontier);
        put_scalar<std::uint32_t>(out, entry.mtp_kv_valid);
        put_scalar<std::int32_t>(out, entry.rope_delta);
        put_scalar<std::uint8_t>(out, entry.tail_hidden_valid ? 1U : 0U);
        put_scalar<std::uint32_t>(out, entry.mtp_draft_count);
        for (const TokenId draft : entry.mtp_drafts) { put_scalar<std::int32_t>(out, draft); }
        put_vector<std::int32_t>(out, entry.ledger);
        entry.prefix_identity.save(out);
        put_vector<std::uint32_t>(out, entry.text_pages);
        put_vector<std::uint32_t>(out, entry.backend_pages);
        put_size(out, payload_bytes);
        put_scalar<std::uint8_t>(out, entry.checkpoint_valid ? 1U : 0U);
        if (entry.checkpoint_valid) {
            HostCheckpoint current;
            current.kind     = entry.checkpoint_kind;
            current.frontier = entry.checkpoint_frontier;
            current.ledger   = entry.checkpoint_ledger;
            current.identity = entry.checkpoint_identity;
            sequence_image_io_detail::write_meta_checkpoint(out, current);
        }
        sequence_image_io_detail::write_meta_history(out, entry.checkpoint_history);
        sequence_image_io_detail::write_meta_history(out, entry.boundary_history);
        sequence_image_io_detail::write_meta_history(out, entry.response_history);
        sequence_image_io_detail::write_meta_history(out, entry.long_range_history);
        put_scalar<std::uint8_t>(out, entry.shared_prefix.frontier != 0 ? 1U : 0U);
        if (entry.shared_prefix.frontier != 0) {
            sequence_image_io_detail::write_meta_checkpoint(out, entry.shared_prefix);
        }
        put_scalar<std::uint8_t>(out, entry.tools_prefix.frontier != 0 ? 1U : 0U);
        if (entry.tools_prefix.frontier != 0) {
            sequence_image_io_detail::write_meta_checkpoint(out, entry.tools_prefix);
        }
        out.flush();
        if (!out) {
            throw std::runtime_error("sequence metadata write failed: " + path);
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw std::runtime_error("cannot publish sequence metadata: " + path);
    }
}

inline std::unique_ptr<ParkedSequence> read_sequence_metadata(
    const std::string& path, const SequenceImageGeometry& geometry) {
    using namespace image_stream;
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open sequence metadata: " + path); }
    if (get_scalar<std::uint32_t>(in) != kSequenceMetaMagic) {
        throw std::runtime_error("not sequence metadata: " + path);
    }
    if (get_scalar<std::uint32_t>(in) != kSequenceMetaVersion) {
        throw std::runtime_error("unsupported sequence metadata version: " + path);
    }
    constexpr std::size_t kMaximumSize = std::numeric_limits<std::size_t>::max();
    if (get_size(in, kMaximumSize) != geometry.rank_count ||
        get_size(in, kMaximumSize) != geometry.slot_bytes ||
        get_size(in, kMaximumSize) != geometry.hidden_bytes ||
        get_size(in, kMaximumSize) != geometry.text_plane_count ||
        get_size(in, kMaximumSize) != geometry.backend_plane_count) {
        throw std::runtime_error("sequence metadata geometry does not match the engine: " + path);
    }
    auto entry                = std::make_unique<ParkedSequence>();
    entry->execution_frontier = get_scalar<std::uint32_t>(in);
    entry->ledger_frontier    = get_scalar<std::uint32_t>(in);
    entry->mtp_kv_valid       = get_scalar<std::uint32_t>(in);
    entry->rope_delta         = get_scalar<std::int32_t>(in);
    entry->tail_hidden_valid  = get_scalar<std::uint8_t>(in) != 0;
    entry->mtp_draft_count    = get_scalar<std::uint32_t>(in);
    if (entry->mtp_draft_count > entry->mtp_drafts.size()) {
        throw std::runtime_error("sequence metadata draft count is out of range");
    }
    for (TokenId& draft : entry->mtp_drafts) { draft = get_scalar<std::int32_t>(in); }
    entry->ledger          = get_vector<TokenId>(in, 1U << 20);
    entry->prefix_identity = ResidentPrefixIdentity::load(in);
    entry->text_pages      = get_vector<std::uint32_t>(in, 64);
    entry->backend_pages   = get_vector<std::uint32_t>(in, 64);
    if (entry->text_pages.size() != geometry.rank_count ||
        (geometry.backend_plane_count != 0 &&
         entry->backend_pages.size() != geometry.rank_count) ||
        (geometry.backend_plane_count == 0 && !entry->backend_pages.empty())) {
        throw std::runtime_error("sequence metadata page counts do not match the engine: " + path);
    }
    entry->bytes           = get_size(in, std::numeric_limits<std::size_t>::max());
    entry->checkpoint_valid = get_scalar<std::uint8_t>(in) != 0;
    if (entry->checkpoint_valid) {
        HostCheckpoint current;
        sequence_image_io_detail::read_meta_checkpoint(in, current);
        entry->checkpoint_kind     = current.kind;
        entry->checkpoint_frontier = current.frontier;
        entry->checkpoint_ledger   = std::move(current.ledger);
        entry->checkpoint_identity = std::move(current.identity);
    }
    sequence_image_io_detail::read_meta_history(in, entry->checkpoint_history);
    sequence_image_io_detail::read_meta_history(in, entry->boundary_history);
    sequence_image_io_detail::read_meta_history(in, entry->response_history);
    sequence_image_io_detail::read_meta_history(in, entry->long_range_history);
    if (get_scalar<std::uint8_t>(in) != 0) {
        sequence_image_io_detail::read_meta_checkpoint(in, entry->shared_prefix);
    }
    if (get_scalar<std::uint8_t>(in) != 0) {
        sequence_image_io_detail::read_meta_checkpoint(in, entry->tools_prefix);
    }
    if (in.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("sequence metadata has trailing data: " + path);
    }
    return entry;
}

inline std::unique_ptr<ParkedSequence> read_sequence_image(
    const std::string& path, const SequenceImageGeometry& geometry,
    std::span<PagedKVPool* const> text_pools, std::span<PagedKVPool* const> backend_pools) {
    std::vector<std::unique_ptr<ParkedSequence>> entries =
        read_sequence_images(path, geometry, text_pools, backend_pools);
    if (entries.size() != 1) {
        throw std::runtime_error("sequence image does not hold exactly one entry: " + path);
    }
    return std::move(entries.front());
}

} // namespace ninfer::targets::qwen3_6::detail
