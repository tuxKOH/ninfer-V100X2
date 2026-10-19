// The disk-backed shared prefix and the RAM-to-disk spill tier share one image format. This
// round-trips the metadata and checkpoint lineage and checks that truncated or mismatched files
// are rejected before any device state is touched. No device or pool is needed: the KV payloads
// are absent here, matching the metadata-only shape the disk index keeps in RAM.
#include "targets/qwen3_6/impl/runtime/sequence_image_io.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace d = ninfer::targets::qwen3_6::detail;

int failures = 0;
void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

int main() {
    d::ParkedSequence entry;
    entry.execution_frontier = 40;
    entry.ledger_frontier    = 40;
    entry.mtp_kv_valid       = 40;
    entry.rope_delta         = 0;
    entry.tail_hidden_valid  = true;
    entry.tail_hidden.push_back(std::vector<std::byte>(8, std::byte{0x11}));
    entry.ledger.assign(40, 7);
    ninfer::targets::qwen3_6::PreparedPromptData prompt;
    prompt.token_ids.assign(40, 7);
    prompt.token_types.assign(40, 0);
    prompt.positions.assign(40 * 3, 0);
    entry.prefix_identity.assign(prompt);
    d::HostCheckpoint checkpoint;
    checkpoint.kind     = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure;
    checkpoint.frontier = 40;
    checkpoint.ledger   = entry.ledger;
    checkpoint.identity = entry.prefix_identity;
    checkpoint.hidden.assign(8, std::byte{0x3c});
    entry.boundary_history.push_back(checkpoint);
    entry.response_history.push_back(checkpoint);
    entry.long_range_history.push_back(checkpoint);
    entry.shared_prefix = checkpoint;
    entry.tools_prefix  = checkpoint;
    entry.bytes         = 123;

    const std::string path =
        (std::filesystem::temp_directory_path() / "ninfer_sequence_image_test.img").string();
    d::SequenceImageGeometry geometry;  // zero ranks: no KV payloads
    geometry.hidden_bytes = 8;         // matches the fixture's checkpoint hidden
    d::write_sequence_image(entry, path);

    std::vector<std::unique_ptr<d::ParkedSequence>> loaded =
        d::read_sequence_images(path, geometry, {}, {});
    check(loaded.size() == 1, "one entry survives the round trip");
    if (!loaded.empty()) {
        d::ParkedSequence& read = *loaded.front();
        check(read.execution_frontier == 40 && read.ledger_frontier == 40, "frontiers");
        check(read.mtp_kv_valid == 40 && read.rope_delta == 0, "mtp/rope");
        check(read.tail_hidden_valid, "tail hidden flag");
        check(read.ledger == entry.ledger, "ledger");
        check(read.prefix_identity.size() == entry.prefix_identity.size(), "identity size");
        check(read.checkpoint_valid == false, "checkpoint flag defaults off");
        check(read.boundary_history.size() == 1 && read.boundary_history[0].frontier == 40,
              "boundary history");
        check(read.response_history.size() == 1 && read.response_history[0].frontier == 40,
              "response history");
        check(read.long_range_history.size() == 1 && read.long_range_history[0].frontier == 40,
              "long-range history");
        check(read.boundary_history[0].hidden == checkpoint.hidden, "boundary hidden");
        check(read.shared_prefix.frontier == 40 && read.tools_prefix.frontier == 40, "pinned");
    }

    // The directory key must distinguish equal-length prefixes with different tokens: two system
    // prompts of the same size otherwise dedupe to one saved image.
    {
        ninfer::targets::qwen3_6::PreparedPromptData left;
        ninfer::targets::qwen3_6::PreparedPromptData right;
        left.token_ids.assign(40, 7);
        right.token_ids.assign(40, 8);
        left.token_types.assign(40, 0);
        right.token_types.assign(40, 0);
        left.positions.assign(40 * 3, 0);
        right.positions.assign(40 * 3, 0);
        d::ResidentPrefixIdentity left_identity;
        d::ResidentPrefixIdentity right_identity;
        left_identity.assign(left);
        right_identity.assign(right);
        check(d::prefix_fingerprint(left.token_ids, left_identity, 40) !=
                  d::prefix_fingerprint(right.token_ids, right_identity, 40),
              "equal-length prefixes with different tokens share a persistence key");
        check(d::prefix_fingerprint(left.token_ids, left_identity, 40) ==
                  d::prefix_fingerprint(left.token_ids, left_identity, 40),
              "prefix fingerprint is not stable");
    }

    // A pinned boundary is optional. An entry may carry neither (a spilled conversation), only
    // the tool anchor, or only the leading block; absence must not be confused with a malformed
    // checkpoint (the reader validates rank/hidden sizes of checkpoints that ARE present).
    const auto roundtrip_pinned = [&](bool tools, bool leading) {
        d::ParkedSequence partial;
        partial.execution_frontier = 40;
        partial.ledger_frontier    = 40;
        partial.mtp_kv_valid       = 40;
        partial.ledger             = entry.ledger;
        partial.prefix_identity    = entry.prefix_identity;
        partial.tail_hidden.push_back(std::vector<std::byte>(8, std::byte{0x11}));
        partial.shared_prefix = leading ? checkpoint : d::HostCheckpoint{};
        partial.tools_prefix  = tools ? checkpoint : d::HostCheckpoint{};
        const std::string partial_path =
            (std::filesystem::temp_directory_path() /
             ("ninfer_sequence_partial_" + std::to_string(tools) + std::to_string(leading) + ".img"))
                .string();
        d::write_sequence_image(partial, partial_path);
        std::vector<std::unique_ptr<d::ParkedSequence>> read =
            d::read_sequence_images(partial_path, geometry, {}, {});
        std::filesystem::remove(partial_path);
        return read.size() == 1 && read.front()->tools_prefix.frontier == (tools ? 40 : 0) &&
               read.front()->shared_prefix.frontier == (leading ? 40 : 0);
    };
    check(roundtrip_pinned(false, false), "entry with no pinned boundary round-trips");
    check(roundtrip_pinned(true, false), "entry with only the tool anchor round-trips");
    check(roundtrip_pinned(false, true), "entry with only the leading block round-trips");

    // The metadata sidecar lets a restart index a spilled conversation without reading its
    // payload. Round-trip it, including the page counts and per-history checkpoints.
    {
        entry.text_pages    = {3, 3};
        entry.backend_pages = {4, 4};
        d::SequenceImageGeometry meta_geometry = geometry;
        meta_geometry.rank_count              = 2;
        meta_geometry.slot_bytes              = 8;
        meta_geometry.text_plane_count        = 2;
        meta_geometry.backend_plane_count     = 2;
        const std::string meta_path =
            (std::filesystem::temp_directory_path() / "ninfer_sequence_meta.img.meta").string();
        d::write_sequence_metadata(entry, 987654, meta_geometry, meta_path);
        std::unique_ptr<d::ParkedSequence> meta =
            d::read_sequence_metadata(meta_path, meta_geometry);
        check(meta->execution_frontier == 40 && meta->ledger_frontier == 40 &&
                  meta->bytes == 987654 && meta->ledger == entry.ledger,
              "metadata sidecar did not round-trip");
        check(meta->text_pages == entry.text_pages &&
                  meta->backend_pages == entry.backend_pages,
              "metadata sidecar page counts");
        check(meta->boundary_history.size() == 1 &&
                  meta->response_history.size() == 1 &&
                  meta->long_range_history.size() == 1 &&
                  meta->shared_prefix.frontier == 40 && meta->tools_prefix.frontier == 40,
              "metadata sidecar checkpoints");
        d::SequenceImageGeometry wrong = meta_geometry;
        wrong.slot_bytes = 16;
        bool rejected                  = false;
        try {
            (void)d::read_sequence_metadata(meta_path, wrong);
        } catch (const std::exception&) { rejected = true; }
        check(rejected, "metadata sidecar accepted a foreign geometry");
        std::filesystem::remove(meta_path);
    }

    // Truncation is rejected rather than partially applied.
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
        std::ofstream out(std::filesystem::temp_directory_path() /
                              "ninfer_sequence_image_truncated.img",
                          std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
    }
    bool rejected = false;
    try {
        (void)d::read_sequence_images(
            (std::filesystem::temp_directory_path() / "ninfer_sequence_image_truncated.img")
                .string(),
            geometry, {}, {});
    } catch (const std::exception&) { rejected = true; }
    check(rejected, "truncated image is rejected");

    // A geometry mismatch (rank count) is rejected.
    d::SequenceImageGeometry wrong_ranks;
    wrong_ranks.rank_count = 2;
    wrong_ranks.slot_bytes = 8;
    rejected               = false;
    try {
        (void)d::read_sequence_images(path, wrong_ranks, {}, {});
    } catch (const std::exception&) { rejected = true; }
    check(rejected, "rank mismatch is rejected");

    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
