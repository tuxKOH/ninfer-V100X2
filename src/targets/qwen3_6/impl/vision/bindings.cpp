#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::Q6G64_F16S,
                               {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
                                        {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
                                        {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", NumericFormat::W8G32_F16S,
                         {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm,
                                              bool host) {
    using artifact::NumericFormat;

    const auto weight = [&](artifact::ObjectHandle handle, NumericFormat format,
                            std::int32_t rows, std::int32_t columns) {
        return host ? artifact::host_materialized_weight(materialized, handle, format, rows,
                                                         columns)
                    : artifact::materialized_weight(materialized, handle, format, rows, columns);
    };
    const auto tensor = [&](artifact::ObjectHandle handle, NumericFormat format,
                            std::initializer_list<std::int32_t> shape) {
        return host ? artifact::host_materialized_tensor(materialized, handle, format, shape)
                    : artifact::materialized_tensor(materialized, handle, format, shape);
    };

    VisionCommonWeights out;
    out.patch_embedding = weight(backbone.patch_embedding, NumericFormat::Q6G64_F16S,
                                 VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim);
    out.patch_embedding_bias =
        tensor(backbone.patch_embedding_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        tensor(backbone.position_embedding, NumericFormat::BF16,
               {VisionBackboneConfig::hidden, VisionBackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv = weight(source.qkv, NumericFormat::Q4G64_F16S,
                            3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden);
        target.qkv_bias =
            tensor(source.qkv_bias, NumericFormat::BF16, {3 * VisionBackboneConfig::hidden});
        target.output = weight(source.output, NumericFormat::Q5G64_F16S,
                               VisionBackboneConfig::hidden, VisionBackboneConfig::hidden);
        target.output_bias =
            tensor(source.output_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.fc1 = weight(source.fc1, NumericFormat::Q4G64_F16S,
                            VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden);
        target.fc1_bias =
            tensor(source.fc1_bias, NumericFormat::BF16, {VisionBackboneConfig::intermediate});
        target.fc2 = weight(source.fc2, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
                            VisionBackboneConfig::intermediate);
        target.fc2_bias =
            tensor(source.fc2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            tensor(source.norm1_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            tensor(source.norm1_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            tensor(source.norm2_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            tensor(source.norm2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }

    out.merger_fc1 = weight(merger_input.fc1, NumericFormat::W8G32_F16S,
                            VisionBackboneConfig::merger_hidden,
                            VisionBackboneConfig::merger_hidden);
    out.merger_fc1_bias = tensor(merger_input.fc1_bias, NumericFormat::BF16,
                                 {VisionBackboneConfig::merger_hidden});
    out.merger_norm_weight =
        tensor(merger_norm.weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.merger_norm_bias =
        tensor(merger_norm.bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    return out;
}

} // namespace ninfer::targets::qwen3_6
