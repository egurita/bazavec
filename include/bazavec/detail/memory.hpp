#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>

namespace bazavec {

// Apple Silicon cache line, measured on this machine via hw.cachelinesize.
// One row starts on a line boundary so a 16-byte NEON load never straddles
// two lines, and a later per-node lock can have a line to itself.
inline constexpr std::size_t kArenaAlignment = 128;

inline constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

namespace detail {

struct AlignedFree {
  // aligned_alloc is a malloc-family allocator. free() is the matching
  // deallocator; operator delete is not.
  void operator()(float* pointer) const noexcept { std::free(pointer); }
};

using AlignedPtr = std::unique_ptr<float, AlignedFree>;

inline std::size_t allocation_bytes(std::size_t n_floats) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  if (n_floats > kMax / sizeof(float)) {
    throw std::bad_alloc();
  }
  const std::size_t bytes = n_floats * sizeof(float);
  if (bytes > kMax - (kArenaAlignment - 1)) {
    throw std::bad_alloc();
  }
  return align_up(bytes, kArenaAlignment);
}

// Size is rounded up to a multiple of the alignment. The C standard requires
// that of aligned_alloc; a violation returns nullptr rather than a pointer.
inline AlignedPtr allocate_floats(std::size_t n_floats) {
  if (n_floats == 0) {
    return AlignedPtr{};
  }
  const std::size_t bytes = allocation_bytes(n_floats);
  void* raw = std::aligned_alloc(kArenaAlignment, bytes);
  if (raw == nullptr) {
    throw std::bad_alloc();
  }
  return AlignedPtr{static_cast<float*>(raw)};
}

}  // namespace detail
}  // namespace bazavec
