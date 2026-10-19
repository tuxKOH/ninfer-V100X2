#pragma once

// Minimal little-endian binary stream helpers shared by the host-image persistence code. The
// format is deliberately explicit (fixed widths, explicit lengths) so a file written by one build
// reads identically on another, and the reader can reject a truncated or malformed file before it
// touches any device state.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::image_stream {

template <typename T>
void put_scalar(std::ostream& out, T value) {
    static_assert(std::is_integral_v<T> && sizeof(T) <= 8);
    using U = std::make_unsigned_t<T>;
    U bits  = static_cast<U>(value);
    for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
        out.put(static_cast<char>((bits >> (8 * byte)) & static_cast<U>(0xff)));
    }
    if (!out) { throw std::runtime_error("sequence image write failed"); }
}

template <typename T>
T get_scalar(std::istream& in) {
    static_assert(std::is_integral_v<T> && sizeof(T) <= 8);
    using U      = std::make_unsigned_t<T>;
    U bits       = 0;
    for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
        const int value = in.get();
        if (value == std::char_traits<char>::eof()) {
            throw std::runtime_error("sequence image is truncated");
        }
        bits |= static_cast<U>(static_cast<unsigned char>(value)) << (8 * byte);
    }
    return static_cast<T>(bits);
}

inline void put_bytes(std::ostream& out, const void* data, std::size_t size) {
    if (size != 0) { out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)); }
    if (!out) { throw std::runtime_error("sequence image write failed"); }
}

inline void get_bytes(std::istream& in, void* data, std::size_t size) {
    if (size == 0) { return; }
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!in) { throw std::runtime_error("sequence image is truncated"); }
}

inline void put_size(std::ostream& out, std::size_t value) {
    put_scalar<std::uint64_t>(out, static_cast<std::uint64_t>(value));
}

inline std::size_t get_size(std::istream& in, std::size_t limit) {
    const std::uint64_t value = get_scalar<std::uint64_t>(in);
    if (value > limit) { throw std::runtime_error("sequence image length is out of range"); }
    return static_cast<std::size_t>(value);
}

template <typename T>
void put_vector(std::ostream& out, const std::vector<T>& values) {
    put_size(out, values.size());
    for (const T value : values) { put_scalar<T>(out, value); }
}

template <typename T>
std::vector<T> get_vector(std::istream& in, std::size_t limit) {
    std::vector<T> values(get_size(in, limit));
    for (T& value : values) { value = get_scalar<T>(in); }
    return values;
}

} // namespace ninfer::targets::qwen3_6::detail::image_stream
