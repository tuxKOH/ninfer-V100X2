#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
using namespace ninfer::artifact;
using namespace ninfer::targets::qwen3_6_27b::detail;
using Json = nlohmann::json;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

std::uint64_t little_endian(const char* bytes, int count) {
    std::uint64_t result = 0;
    for (int i = 0; i < count; ++i) {
        result |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return result;
}

void verify(const char* path) {
    // Independent source-directory/scalar oracle. No GPU allocation or weight conversion.
    std::ifstream source(path, std::ios::binary);
    std::array<char, 32> header{};
    source.read(header.data(), header.size());
    require(source.good() && header[7] == 3, "QUASAR test requires its registered v3 file");
    std::string json_bytes(little_endian(header.data() + 8, 8), '\0');
    source.read(json_bytes.data(), json_bytes.size());
    require(source.good(), "truncated source directory");
    const Json directory = Json::parse(json_bytes);
    const auto payload_start = (header.size() + json_bytes.size() + 4095) / 4096 * 4096;
    std::unordered_map<std::string, Json> stored;
    for (const auto& object : directory.at("objects")) {
        stored.emplace(object.at("id").get<std::string>(), object);
    }
    const auto scalar = [&](std::uint64_t offset) {
        std::array<char, 4> bytes{};
        source.seekg(payload_start + offset);
        source.read(bytes.data(), bytes.size());
        require(source.good(), "source FP32 scalar is truncated");
        return static_cast<std::uint32_t>(little_endian(bytes.data(), 4));
    };

    Reader reader(path);
    require(reader.identity().model_id == "qwen3.8-27b" &&
                reader.identity().weights_id == "quasar-nvfp4", "QUASAR identity was not retained");
    require(reader.objects().size() == 1334, "QUASAR projected inventory is incomplete");
    for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp,
                               ninfer::SpeculativeBackend::DFlash}) {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
            {.speculative = backend}, 2);
        int projections = 0;
        const auto check = [&](const WeightPlan& weight, const std::string& divisor) {
            const auto& descriptor = std::get<TensorDescriptor>(binder.descriptor(weight.object));
            const auto& original = stored.at(descriptor.name);
            require(weight.format == NumericFormat::NVFP4 && original.at("format") == "nvfp4",
                    descriptor.name + ": wrong weight format");
            const auto payload = reader.payload(descriptor);
            require(payload.absolute_offset == payload_start + original.at("offset").get<std::uint64_t>() &&
                        payload.data.size() == original.at("bytes").get<std::uint64_t>(),
                    descriptor.name + ": not the exact original packed payload range");
            const auto elements = descriptor.shape[0] * descriptor.shape[1];
            // All registered shapes have aligned code planes; stored E2M1/E4M3 words are
            // followed by one FP32 weight divisor. Read it directly from the original file.
            const auto expected_weight = scalar(original.at("offset").get<std::uint64_t>() +
                                                elements / 2 + elements / 16);
            const auto expected_input = scalar(stored.at(divisor).at("offset").get<std::uint64_t>());
            require(weight.weight_scale_divisor_bits == expected_weight &&
                        weight.input_scale_divisor_bits == expected_input,
                    descriptor.name + ": scale pairing differs from source FP32 words");
            ++projections;
        };
        for (std::size_t layer = 0; layer < plan.bindings.text_layers.size(); ++layer) {
            const auto& weights = plan.bindings.text_layers[layer];
            const auto prefix = "text/layers/" + std::to_string(layer) + "/";
            if (weights.is_full_attention) {
                check(std::get<FusedAttentionProjectionPlan>(weights.attention.projection).query_key_gate_value,
                      prefix + "attention/input_projection/input_scale_divisor");
                check(weights.attention.output, prefix + "attention/output_projection/input_scale_divisor");
            } else {
                check(std::get<FusedGdnInputProjectionPlan>(weights.gdn.input_projection).query_key_value_z,
                      prefix + "gdn/input_projection/input_scale_divisor");
                check(weights.gdn.output, prefix + "gdn/output_projection/input_scale_divisor");
                require(std::get<FusedGdnControlProjectionPlan>(weights.gdn.control_projection)
                            .a_b_projection.format == NumericFormat::BF16,
                        "QUASAR GDN controls lost the artifact's BF16 boundary");
            }
            check(weights.mlp.gate_up, prefix + "mlp/gate_up_projection/input_scale_divisor");
            check(weights.mlp.down, prefix + "mlp/down_projection/input_scale_divisor");
        }
        require(projections == 256 && plan.bindings.token_embedding.format == NumericFormat::W8G32_F16S &&
                    plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                    plan.bindings.mtp_format == NumericFormat::W8G32_F16S,
                "QUASAR storage profile is incomplete");
        int mtp_objects = 0;
        std::array<int, 2> dflash_objects{};
        for (const auto& placement : plan.materialization.device_objects) {
            const auto name = object_name(binder.descriptor(placement.object));
            require(!name.starts_with("vision/"),
                    "text-only QUASAR uploaded a Vision object to a GPU");
            if (name.starts_with("dflash/")) { ++dflash_objects.at(placement.device); }
            if (name.starts_with("mtp/")) { ++mtp_objects; }
        }
        require(backend == ninfer::SpeculativeBackend::Mtp ? mtp_objects > 0 : mtp_objects == 0,
                "MTP startup selection is incorrect");
        require(dflash_objects == (backend == ninfer::SpeculativeBackend::DFlash
                    ? std::array{66, 66} : std::array{0, 0}),
                "DFlash startup selection is incorrect");
        require(plan.bindings.dflash.projection_format == NumericFormat::W8G32_F16S,
                "QUASAR DFlash projections lost their W8 format");
    }

    // --vision at TP2 binds the container's complete Vision set. The default plans above upload
    // none of it; with the flag every Vision tensor must land on the primary device only, because
    // the Vision encoder runs on rank 0 and replicates composed embeddings to the peer.
    std::size_t stored_vision = 0;
    for (const auto& object : reader.objects()) {
        if (object_name(object).starts_with("vision/")) { ++stored_vision; }
    }
    require(stored_vision != 0, "QUASAR container has no Vision set to bind");
    for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp}) {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                                        {.vision = true, .speculative = backend}, 2);
        int vision_primary = 0;
        int vision_peer    = 0;
        for (const auto& placement : plan.materialization.device_objects) {
            const auto name = object_name(binder.descriptor(placement.object));
            if (!name.starts_with("vision/")) { continue; }
            if (placement.device == 0) {
                ++vision_primary;
            } else {
                ++vision_peer;
            }
        }
        require(vision_primary == static_cast<int>(stored_vision) && vision_peer == 0,
                "QUASAR Vision did not upload its complete set to the primary device only");
    }
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_QUASAR_ARTIFACT");
    if (!path || !*path) { std::cout << "SKIP: set NINFER_QUASAR_ARTIFACT\n"; return 77; }
    try {
        verify(path);
        std::cout << "OK QUASAR exact packed ranges, 256 scale pairs, TP2 None/MTP/DFlash placement "
                     "and zero GPU Vision objects\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
