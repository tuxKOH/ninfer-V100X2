#pragma once
// Host-side helpers for the CPU Vision encoder: BF16/FP16 codecs, the quantized row-split
// dequantizers (Q4G64/Q5G64/Q6G64/W8G32), a small AVX2/FMA GEMM, and the 2D Vision rotary table.
// Deliberately free of CUDA and of the runtime's model types so it can be exercised on its own.

#include "core/tensor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

namespace cpu_vision_detail {

inline float bf16_to_f32(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out                = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest even, preserving NaN payloads.
    if ((bits & 0x7f800000U) != 0x7f800000U) {
        const std::uint32_t lsb  = (bits >> 16) & 1U;
        bits += 0x7fffU + lsb;
    }
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float f16_to_f32(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
    std::uint32_t exp        = (value >> 10) & 0x1fU;
    std::uint32_t mant       = value & 0x3ffU;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal: normalize.
            int shift = 0;
            while ((mant & 0x400U) == 0) {
                mant <<= 1;
                ++shift;
            }
            mant &= 0x3ffU;
            // A shifted f16 subnormal with fraction m (10 bits) is m*2^-24 = (1.f)*2^(-14-shift):
            // the unbiased exponent is -14-shift, not -15-shift.
            bits = sign | (static_cast<std::uint32_t>(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fU) {
        bits = sign | 0x7f800000U | (mant << 13);
    } else {
        bits = sign | ((exp - 15U + 127U) << 23) | (mant << 13);
    }
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline unsigned worker_count() {
    static const unsigned cached = [] {
        if (const char* env = std::getenv("NINFER_VISION_CPU_THREADS")) {
            const int value = std::atoi(env);
            if (value > 0) { return static_cast<unsigned>(std::min(value, 64)); }
        }
        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware == 0) { hardware = 4; }
        // Three quarters of the host, leaving room for the media preprocess pool and the engine's
        // own host work. The server's GPU loop is idle while the encoder runs, so this setting is
        // what decides image latency.
        return std::max(1U, std::min(hardware * 3U / 4U, 24U));
    }();
    return cached;
}

// The dequantized backbone is ~1.7 GB of host memory once warm. Set
// NINFER_VISION_CPU_CACHE=0 to dequantize per request instead, trading latency for memory.
inline bool weight_cache_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_VISION_CPU_CACHE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

template <class Function>
void parallel_for(std::size_t count, unsigned workers, Function&& function) {
    if (workers <= 1 || count <= 1) {
        for (std::size_t i = 0; i < count; ++i) { function(i); }
        return;
    }
    const unsigned width = static_cast<unsigned>(std::min<std::size_t>(workers, count));
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(width);
    for (unsigned worker = 0; worker < width; ++worker) {
        threads.emplace_back([&] {
            for (;;) {
                const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= count) { return; }
                function(index);
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
}

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    void resize(int new_rows, int new_cols) {
        rows = new_rows;
        cols = new_cols;
        data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    }
};

struct QuantSpec {
    int bits;
    int group;
};

inline QuantSpec quant_spec(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return {4, 64};
    case QType::Q5G64_F16S: return {5, 64};
    case QType::Q6G64_F16S: return {6, 64};
    case QType::W8G32_F16S: return {8, 32};
    default: throw std::invalid_argument("CPU Vision: unsupported weight format");
    }
}

inline int low_bytes_per_group(const QuantSpec& spec) {
    return spec.bits == 8 ? spec.group : spec.group / 2;
}

inline int high_bytes_per_group(const QuantSpec& spec) {
    if (spec.bits == 8) { return 0; }
    return spec.bits <= 4 ? 0 : spec.group * (spec.bits - 4) / 8;
}

inline int unpack_code(const std::uint8_t* low, const std::uint8_t* high, const QuantSpec& spec,
                       int index) {
    if (spec.bits == 8) { return static_cast<std::int8_t>(low[index]); }
    const std::uint8_t byte = low[index >> 1];
    std::uint32_t value = (index & 1) ? (byte >> 4) : (byte & 0x0fU);
    if (spec.bits == 5) {
        value |= static_cast<std::uint32_t>((high[index >> 3] >> (index & 7)) & 1U) << 4;
    } else if (spec.bits == 6) {
        const int bit = index * 2;
        value |= static_cast<std::uint32_t>((high[bit >> 3] >> (bit & 7)) & 3U) << 4;
    }
    const std::uint32_t sign = 1U << (spec.bits - 1);
    const std::uint32_t span = 1U << spec.bits;
    return (value & sign) ? static_cast<int>(value) - static_cast<int>(span)
                          : static_cast<int>(value);
}

// Fills rows [row_begin, row_end) of `out`, which must already be sized to the weight's shape.
// Splitting by rows keeps the decoder embarrassingly parallel, which matters because the first
// image pays for all 411M codes while later images hit the shared dequantized cache.
inline void dequantize_rows(const Weight& weight, Matrix& out, int row_begin, int row_end) {
    const auto rows = static_cast<int>(weight.shape[0]);
    const auto cols = static_cast<int>(weight.shape[1]);
    if (rows <= 0 || cols <= 0) { throw std::invalid_argument("CPU Vision: empty weight"); }
    if (out.rows != rows || out.cols != cols || row_begin < 0 || row_end > rows ||
        row_begin > row_end) {
        throw std::invalid_argument("CPU Vision: invalid dequantization range");
    }
    if (weight.layout == QuantLayout::Contiguous) {
        if (weight.qtype == QType::BF16_CTRL) {
            const auto* bits = static_cast<const std::uint16_t*>(weight.qdata);
            const std::size_t begin = static_cast<std::size_t>(row_begin) * cols;
            const std::size_t end   = static_cast<std::size_t>(row_end) * cols;
            for (std::size_t i = begin; i < end; ++i) { out.data[i] = bf16_to_f32(bits[i]); }
            return;
        }
        if (weight.qtype == QType::FP32_CTRL) {
            const auto* source = static_cast<const float*>(weight.qdata);
            std::memcpy(out.data.data() + static_cast<std::size_t>(row_begin) * cols,
                        source + static_cast<std::size_t>(row_begin) * cols,
                        static_cast<std::size_t>(row_end - row_begin) * cols * sizeof(float));
            return;
        }
        throw std::invalid_argument("CPU Vision: unsupported dense weight format");
    }
    if (weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("CPU Vision: unsupported weight layout");
    }
    const QuantSpec spec = quant_spec(weight.qtype);
    const int padded_cols = static_cast<int>(weight.padded_shape[1]);
    const int groups      = padded_cols / spec.group;
    const int low_bytes   = low_bytes_per_group(spec);
    const int high_bytes  = high_bytes_per_group(spec);
    const auto* low       = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* high      = high_bytes == 0 ? nullptr : static_cast<const std::uint8_t*>(weight.qhigh);
    const auto* scales    = static_cast<const std::uint8_t*>(weight.scales);
    for (int row = row_begin; row < row_end; ++row) {
        float* dst = out.data.data() + static_cast<std::size_t>(row) * cols;
        for (int group = 0; group < groups; ++group) {
            const std::size_t index = static_cast<std::size_t>(row) * groups + group;
            const auto scale_bits = static_cast<std::uint16_t>(
                scales[index * 2] | (static_cast<std::uint16_t>(scales[index * 2 + 1]) << 8));
            const float scale = f16_to_f32(scale_bits);
            const std::uint8_t* low_group  = low + index * low_bytes;
            const std::uint8_t* high_group = high == nullptr ? nullptr : high + index * high_bytes;
            const int begin = group * spec.group;
            const int end   = std::min(begin + spec.group, cols);
            for (int lane = begin; lane < end; ++lane) {
                dst[lane] = static_cast<float>(
                                unpack_code(low_group, high_group, spec, lane - begin)) *
                            scale;
            }
        }
    }
}

inline void dequantize(const Weight& weight, Matrix& out) {
    const auto rows = static_cast<int>(weight.shape[0]);
    const auto cols = static_cast<int>(weight.shape[1]);
    if (rows <= 0 || cols <= 0) { throw std::invalid_argument("CPU Vision: empty weight"); }
    out.resize(rows, cols);
    dequantize_rows(weight, out, 0, rows);
}

inline void to_float(const Tensor& tensor, std::vector<float>& out) {
    const std::size_t count = static_cast<std::size_t>(tensor.numel());
    out.resize(count);
    if (tensor.dtype == DType::BF16) {
        const auto* bits = static_cast<const std::uint16_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i) { out[i] = bf16_to_f32(bits[i]); }
        return;
    }
    if (tensor.dtype == DType::FP32) {
        std::memcpy(out.data(), tensor.data, count * sizeof(float));
        return;
    }
    throw std::invalid_argument("CPU Vision: unsupported dense tensor dtype");
}

// out[row + token * n] = sum_c weight[row * k + c] * input[c + token * k]
//
// Parallelism is over OUTPUT ROWS, not tokens, and each row pass consumes a block of tokens at a
// time. A worker therefore reads its slice of the weight matrix once per token block instead of
// once per token: with token-parallel stripes every worker streamed the whole projection for
// every token (tens of GB per image), while row stripes keep the slice resident in shared cache
// and make the projections compute-bound.
using GemmRowRange = void (*)(const float* weight, const float* input, int rows, int cols,
                              int tokens, float* out, int row_begin, int row_end);

constexpr int kGemmTokenBlock = 4;

inline void gemm_rows_generic(const float* weight, const float* input, int rows, int cols,
                              int tokens, float* out, int row_begin, int row_end) {
    for (int token_begin = 0; token_begin < tokens; token_begin += kGemmTokenBlock) {
        const int token_end = std::min(token_begin + kGemmTokenBlock, tokens);
        for (int row = row_begin; row < row_end; ++row) {
            const float* w = weight + static_cast<std::size_t>(row) * cols;
            for (int token = token_begin; token < token_end; ++token) {
                const float* x = input + static_cast<std::size_t>(token) * cols;
                float acc      = 0.0F;
                for (int c = 0; c < cols; ++c) { acc += w[c] * x[c]; }
                out[static_cast<std::size_t>(token) * rows + row] = acc;
            }
        }
    }
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma"))) inline float hsum256(__m256 value) {
    __m128 lo = _mm256_castps256_ps128(value);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Accumulates up to four token dot products against one weight row, so each 32-byte weight load
// feeds four FMAs.
__attribute__((target("avx2,fma"))) inline void dot4(const float* w, const float* x0,
                                                     const float* x1, const float* x2,
                                                     const float* x3, int cols, int count,
                                                     float* out0, float* out1, float* out2,
                                                     float* out3) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = a0;
    __m256 a2 = a0;
    __m256 a3 = a0;
    int c = 0;
    for (; c + 7 < cols; c += 8) {
        const __m256 wv = _mm256_loadu_ps(w + c);
        if (count > 0) { a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + c), a0); }
        if (count > 1) { a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + c), a1); }
        if (count > 2) { a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + c), a2); }
        if (count > 3) { a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + c), a3); }
    }
    float s0 = hsum256(a0);
    float s1 = hsum256(a1);
    float s2 = hsum256(a2);
    float s3 = hsum256(a3);
    for (; c < cols; ++c) {
        const float wv = w[c];
        if (count > 0) { s0 = std::fma(wv, x0[c], s0); }
        if (count > 1) { s1 = std::fma(wv, x1[c], s1); }
        if (count > 2) { s2 = std::fma(wv, x2[c], s2); }
        if (count > 3) { s3 = std::fma(wv, x3[c], s3); }
    }
    if (count > 0) { *out0 = s0; }
    if (count > 1) { *out1 = s1; }
    if (count > 2) { *out2 = s2; }
    if (count > 3) { *out3 = s3; }
}

__attribute__((target("avx2,fma"))) inline void gemm_rows_avx2(const float* weight,
                                                               const float* input, int rows,
                                                               int cols, int tokens, float* out,
                                                               int row_begin, int row_end) {
    for (int token_begin = 0; token_begin < tokens; token_begin += kGemmTokenBlock) {
        const int count    = std::min(kGemmTokenBlock, tokens - token_begin);
        const float* x0    = input + static_cast<std::size_t>(token_begin) * cols;
        const float* x1    = x0 + (count > 1 ? cols : 0);
        const float* x2    = x0 + (count > 2 ? 2 * cols : 0);
        const float* x3    = x0 + (count > 3 ? 3 * cols : 0);
        float* y0          = out + static_cast<std::size_t>(token_begin) * rows;
        float* y1          = y0 + (count > 1 ? rows : 0);
        float* y2          = y0 + (count > 2 ? 2 * rows : 0);
        float* y3          = y0 + (count > 3 ? 3 * rows : 0);
        for (int row = row_begin; row < row_end; ++row) {
            dot4(weight + static_cast<std::size_t>(row) * cols, x0, x1, x2, x3, cols, count,
                 y0 + row, y1 + row, y2 + row, y3 + row);
        }
    }
}
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma"))) inline float hsum_vec8(__m256 value) {
    __m128 lo       = _mm256_castps256_ps128(value);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Cephes exp over eight lanes (~1e-7 relative error after range reduction). The attention
// softmax and the GELU-tanh activation call scalar libm exp/tanh often enough that they dominate
// otherwise.
__attribute__((target("avx2,fma"))) inline __m256 exp8(__m256 x) {
    const __m256 l2e = _mm256_set1_ps(1.44269504088896341F);
    const __m256 c1  = _mm256_set1_ps(0.693359375F);
    const __m256 c2  = _mm256_set1_ps(-2.12194440e-4F);
    x                = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949F));
    x                = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949F));
    const __m256 fx =
        _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, l2e), _mm256_set1_ps(0.5F)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c1));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c2));
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y       = _mm256_set1_ps(1.9875691500e-4F);
    y              = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507e-3F));
    y              = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073e-3F));
    y              = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894e-2F));
    y              = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459e-1F));
    y              = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201e-1F));
    y              = _mm256_add_ps(_mm256_mul_ps(y, z), x);
    y              = _mm256_add_ps(y, _mm256_set1_ps(1.0F));
    // Scale by 2^fx: encode the rounded exponent with the IEEE bias before shifting it into the
    // exponent field.
    __m256i exponent = _mm256_cvtps_epi32(fx);
    exponent         = _mm256_add_epi32(exponent, _mm256_set1_epi32(0x7f));
    exponent         = _mm256_slli_epi32(exponent, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(exponent));
}

// head_dim 72 = nine 8-lane vectors, so the dot has no tail.
__attribute__((target("avx2,fma"))) inline float dot72(const float* a, const float* b) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = s0;
    __m256 s2 = s0;
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 0), _mm256_loadu_ps(b + 0), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), s2);
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), s2);
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64), s2);
    s0 = _mm256_add_ps(s0, _mm256_add_ps(s1, s2));
    return hsum_vec8(s0);
}

// Full (non-causal) attention for one head of one cu_seqlens segment, matching the layout the
// CUDA flash kernel consumes: qkv is [3H, P] with q at 0, k at H and v at 2H; the output is
// [H, P] with head-major head_dim blocks.
__attribute__((target("avx2,fma"))) inline void attention_head(const float* qkv, int begin,
                                                               int end, int hidden,
                                                               int head_dim, std::size_t head,
                                                               float* out) {
    const int length  = end - begin;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<std::size_t>(length));
    for (int p = begin; p < end; ++p) {
        const float* q  = qkv + static_cast<std::size_t>(p) * 3 * hidden + head * head_dim;
        float max_score = -INFINITY;
        for (int j = begin; j < end; ++j) {
            const float* k =
                qkv + static_cast<std::size_t>(j) * 3 * hidden + hidden + head * head_dim;
            const float dot = dot72(q, k) * scale;
            scores[static_cast<std::size_t>(j - begin)] = dot;
            max_score = std::max(max_score, dot);
        }
        const __m256 vmax = _mm256_set1_ps(max_score);
        float sum         = 0.0F;
        int j             = 0;
        for (; j + 7 < length; j += 8) {
            const __m256 shifted = _mm256_sub_ps(_mm256_loadu_ps(scores.data() + j), vmax);
            const __m256 e       = exp8(shifted);
            _mm256_storeu_ps(scores.data() + j, e);
            sum += hsum_vec8(e);
        }
        for (; j < length; ++j) {
            scores[static_cast<std::size_t>(j)] =
                std::exp(scores[static_cast<std::size_t>(j)] - max_score);
            sum += scores[static_cast<std::size_t>(j)];
        }
        const float inv = 1.0F / sum;
        float* dst = out + static_cast<std::size_t>(p) * hidden + head * head_dim;
        __m256 o0 = _mm256_setzero_ps();
        __m256 o1 = o0;
        __m256 o2 = o0;
        __m256 o3 = o0;
        __m256 o4 = o0;
        __m256 o5 = o0;
        __m256 o6 = o0;
        __m256 o7 = o0;
        __m256 o8 = o0;
        for (int j = begin; j < end; ++j) {
            const float w   = scores[static_cast<std::size_t>(j - begin)] * inv;
            const __m256 wv = _mm256_set1_ps(w);
            const float* v =
                qkv + static_cast<std::size_t>(j) * 3 * hidden + 2 * hidden + head * head_dim;
            o0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 0), o0);
            o1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 8), o1);
            o2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 16), o2);
            o3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 24), o3);
            o4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 32), o4);
            o5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 40), o5);
            o6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 48), o6);
            o7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 56), o7);
            o8 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + 64), o8);
        }
        _mm256_storeu_ps(dst + 0, o0);
        _mm256_storeu_ps(dst + 8, o1);
        _mm256_storeu_ps(dst + 16, o2);
        _mm256_storeu_ps(dst + 24, o3);
        _mm256_storeu_ps(dst + 32, o4);
        _mm256_storeu_ps(dst + 40, o5);
        _mm256_storeu_ps(dst + 48, o6);
        _mm256_storeu_ps(dst + 56, o7);
        _mm256_storeu_ps(dst + 64, o8);
    }
}
#endif

inline float gelu_tanh_scalar(float value) {
    return 0.5F * value *
           (1.0F + std::tanh(0.79788456080286535588F *
                             (value + 0.044715F * value * value * value)));
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma"))) inline void bias_gelu_tanh_avx2(float* data,
                                                                    const float* bias, int width,
                                                                    int tokens) {
    const __m256 k     = _mm256_set1_ps(0.79788456080286535588F);
    const __m256 cubic = _mm256_set1_ps(0.044715F);
    const __m256 half  = _mm256_set1_ps(0.5F);
    const __m256 one   = _mm256_set1_ps(1.0F);
    const __m256 two   = _mm256_set1_ps(2.0F);
    for (int t = 0; t < tokens; ++t) {
        float* row = data + static_cast<std::size_t>(t) * width;
        int c      = 0;
        for (; c + 7 < width; c += 8) {
            const __m256 x =
                _mm256_add_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(bias + c));
            const __m256 x3  = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            const __m256 arg = _mm256_mul_ps(k, _mm256_fmadd_ps(cubic, x3, x));
            // tanh(arg) = 1 - 2 / (exp(2*arg) + 1); exp8 clamps its exponent range.
            const __m256 e          = exp8(_mm256_add_ps(arg, arg));
            const __m256 tanh_value = _mm256_sub_ps(one, _mm256_div_ps(two,
                                                                       _mm256_add_ps(e, one)));
            _mm256_storeu_ps(row + c, _mm256_mul_ps(_mm256_mul_ps(half, x),
                                                    _mm256_add_ps(one, tanh_value)));
        }
        for (; c < width; ++c) { row[c] = gelu_tanh_scalar(row[c] + bias[c]); }
    }
}
#endif

// bias + GELU(tanh) over a [tokens, width] buffer: the hot tail of every MLP block.
inline void bias_gelu_tanh(float* data, const float* bias, int width, int tokens) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    if (width % 8 == 0 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        bias_gelu_tanh_avx2(data, bias, width, tokens);
        return;
    }
#endif
    for (int t = 0; t < tokens; ++t) {
        float* row = data + static_cast<std::size_t>(t) * width;
        for (int c = 0; c < width; ++c) {
            row[c] = gelu_tanh_scalar(row[c] + bias[c]);
        }
    }
}

inline GemmRowRange select_gemm_rows() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return gemm_rows_avx2;
    }
#endif
    return gemm_rows_generic;
}

// The 2D Vision rotary table (head_dim 72 = 2 axes x 18 pairs x 2). Values are the CUDA kernel's
// kVisionRopeInvFrequency table so both paths rotate with identical frequencies.
inline constexpr float kVisionInvFrequency[18] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F, 1.291549665e-01F,
    7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F, 1.668100537e-02F, 1.000000000e-02F,
    5.994842503e-03F, 3.593813664e-03F, 2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F,
    4.641588834e-04F, 2.782559402e-04F, 1.668100537e-04F,
};

} // namespace cpu_vision_detail
} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
