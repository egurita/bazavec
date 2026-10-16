#pragma once

#include "detail/memory.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace bazavec {

// One contiguous block of N x stride floats.
//
// stride is dim rounded up to 32 floats (128 bytes). Every row then starts on
// a cache line, the SIMD kernel's tail is zeros rather than a scalar cleanup
// on the hot path, and walking row i is an add, not a pointer chase.
class VectorArena {
 public:
  static constexpr std::size_t padded_stride(std::size_t dim) noexcept {
    constexpr std::size_t kFloatsPerLine = kArenaAlignment / sizeof(float);
    const std::size_t mask = ~(kFloatsPerLine - 1);
    return (dim + kFloatsPerLine - 1) & mask;
  }

  explicit VectorArena(std::size_t dim)
      : dim_(dim), stride_(padded_stride(dim)), size_(0), capacity_(0), block_() {
    if (dim_ == 0 || stride_ < dim_) {
      throw std::invalid_argument("VectorArena: dim must be positive");
    }
  }

  VectorArena(const VectorArena&) = delete;
  VectorArena& operator=(const VectorArena&) = delete;

  VectorArena(VectorArena&& other) noexcept
      : dim_(other.dim_),
        stride_(other.stride_),
        size_(other.size_),
        capacity_(other.capacity_),
        block_(std::move(other.block_)) {
    other.size_ = 0;
    other.capacity_ = 0;
  }

  VectorArena& operator=(VectorArena&& other) noexcept {
    if (this != &other) {
      dim_ = other.dim_;
      stride_ = other.stride_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      block_ = std::move(other.block_);
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  ~VectorArena() = default;

  void reserve(std::size_t n) {
    if (n <= capacity_) {
      return;
    }
    if (n > std::numeric_limits<std::size_t>::max() / stride_) {
      throw std::bad_alloc();
    }
    detail::AlignedPtr fresh = detail::allocate_floats(n * stride_);
    if (size_ > 0) {
      std::memcpy(fresh.get(), block_.get(), size_ * stride_ * sizeof(float));
    }
    block_ = std::move(fresh);
    capacity_ = n;
  }

  // `packed` is row-major, tightly packed, `count` rows of `dim` floats.
  // The pointer must not alias this arena.
  std::uint32_t add_many(const float* packed, std::size_t count) {
    if (count == 0) {
      return static_cast<std::uint32_t>(size_);
    }
    if (packed == nullptr) {
      throw std::invalid_argument("VectorArena::add_many: null data");
    }
    if (count > std::numeric_limits<std::uint32_t>::max() - size_) {
      throw std::length_error("VectorArena::add_many: id space exhausted");
    }
    const std::size_t need = size_ + count;
    if (need > capacity_) {
      std::size_t grown = capacity_ == 0 ? 8 : capacity_;
      while (grown < need) {
        if (grown > std::numeric_limits<std::size_t>::max() / 2) {
          grown = need;
          break;
        }
        grown *= 2;
      }
      reserve(grown);
    }

    float* dst = block_.get() + size_ * stride_;
    if (stride_ == dim_) {
      if (count > std::numeric_limits<std::size_t>::max() / dim_) {
        throw std::bad_alloc();
      }
      std::memcpy(dst, packed, count * dim_ * sizeof(float));
    } else {
      const std::size_t pad = stride_ - dim_;
      for (std::size_t row = 0; row < count; ++row) {
        float* out = dst + row * stride_;
        std::memcpy(out, packed + row * dim_, dim_ * sizeof(float));
        std::memset(out + dim_, 0, pad * sizeof(float));
      }
    }

    const auto first = static_cast<std::uint32_t>(size_);
    size_ += count;
    return first;
  }

  std::uint32_t add(std::span<const float> values) {
    if (values.size() != dim_ || values.data() == nullptr) {
      throw std::invalid_argument("VectorArena::add: expected dim floats");
    }
    return add_many(values.data(), 1);
  }

  const float* data() const noexcept { return block_.get(); }

  // Precondition: id < size().
  const float* vector(std::uint32_t id) const noexcept {
    assert(id < size_);
    const float* row = block_.get() + static_cast<std::size_t>(id) * stride_;
    assert(reinterpret_cast<std::uintptr_t>(row) % kArenaAlignment == 0);
    return row;
  }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t dim() const noexcept { return dim_; }
  std::size_t stride() const noexcept { return stride_; }

  std::size_t bytes() const noexcept { return size_ * stride_ * sizeof(float); }

  std::size_t allocated_bytes() const noexcept {
    if (capacity_ == 0) {
      return 0;
    }
    return detail::allocation_bytes(capacity_ * stride_);
  }

 private:
  std::size_t dim_;
  std::size_t stride_;
  std::size_t size_;
  std::size_t capacity_;
  detail::AlignedPtr block_;
};

static_assert(VectorArena::padded_stride(1) == 32);
static_assert(VectorArena::padded_stride(32) == 32);
static_assert(VectorArena::padded_stride(33) == 64);
static_assert(VectorArena::padded_stride(100) == 128);
static_assert(VectorArena::padded_stride(128) == 128);

}  // namespace bazavec
