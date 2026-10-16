#pragma once

#include "storage.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define BAZAVEC_NEON_I8 1
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define BAZAVEC_AVX2_I8 1
#endif

namespace bazavec {

// Symmetric scalar quantization, one scale per dimension. A float is stored as
// round(x / scale) in [-127, 127]. The integer L2 on those codes is what the
// SIMD kernels compute; it ranks the quantized space, not the original floats.
inline std::int32_t l2_squared_i8_scalar(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
  std::int32_t acc = 0;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t diff = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
    acc += diff * diff;
  }
  return acc;
}

inline std::int32_t l2_squared_i8_simd(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
#if defined(BAZAVEC_NEON_I8)
  // Diff of two int8 values fits in int16. The square fits in int32 (255^2),
  // so the product is widened before it is accumulated. Four chains hide the
  // multiply latency the same way the float kernel hides FMA latency.
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);
  std::size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    const auto squash = [](int32x4_t& acc, const std::int8_t* left, const std::int8_t* right) {
      const int8x16_t va = vld1q_s8(left);
      const int8x16_t vb = vld1q_s8(right);
      const int16x8_t low = vsubl_s8(vget_low_s8(va), vget_low_s8(vb));
      const int16x8_t high = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));
      acc = vaddq_s32(acc, vmull_s16(vget_low_s16(low), vget_low_s16(low)));
      acc = vaddq_s32(acc, vmull_s16(vget_high_s16(low), vget_high_s16(low)));
      acc = vaddq_s32(acc, vmull_s16(vget_low_s16(high), vget_low_s16(high)));
      acc = vaddq_s32(acc, vmull_s16(vget_high_s16(high), vget_high_s16(high)));
    };
    squash(acc0, a + i, b + i);
    squash(acc1, a + i + 16, b + i + 16);
    squash(acc2, a + i + 32, b + i + 32);
    squash(acc3, a + i + 48, b + i + 48);
  }
  for (; i + 16 <= n; i += 16) {
    const int8x16_t va = vld1q_s8(a + i);
    const int8x16_t vb = vld1q_s8(b + i);
    const int16x8_t low = vsubl_s8(vget_low_s8(va), vget_low_s8(vb));
    const int16x8_t high = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));
    acc0 = vaddq_s32(acc0, vmull_s16(vget_low_s16(low), vget_low_s16(low)));
    acc0 = vaddq_s32(acc0, vmull_s16(vget_high_s16(low), vget_high_s16(low)));
    acc0 = vaddq_s32(acc0, vmull_s16(vget_low_s16(high), vget_low_s16(high)));
    acc0 = vaddq_s32(acc0, vmull_s16(vget_high_s16(high), vget_high_s16(high)));
  }
  const int32x4_t sum4 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
  std::int32_t sum = vaddvq_s32(sum4);
  for (; i < n; ++i) {
    const std::int32_t diff = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
    sum += diff * diff;
  }
  return sum;
#elif defined(BAZAVEC_AVX2_I8)
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const auto squash = [](__m256i& acc, const std::int8_t* left, const std::int8_t* right) {
      const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(left));
      const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(right));
      const __m256i da = _mm256_cvtepi8_epi16(va);
      const __m256i db = _mm256_cvtepi8_epi16(vb);
      const __m256i diff = _mm256_sub_epi16(da, db);
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
    };
    squash(acc0, a + i, b + i);
    squash(acc1, a + i + 16, b + i + 16);
  }
  __m256i folded = _mm256_add_epi32(acc0, acc1);
  __m128i lo = _mm256_castsi256_si128(folded);
  __m128i hi = _mm256_extracti128_si256(folded, 1);
  __m128i sum128 = _mm_add_epi32(lo, hi);
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
  std::int32_t sum = _mm_cvtsi128_si32(sum128);
  for (; i < n; ++i) {
    const std::int32_t diff = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
    sum += diff * diff;
  }
  return sum;
#else
  return l2_squared_i8_scalar(a, b, n);
#endif
}

class ScalarQuantizer {
 public:
  static constexpr std::size_t padded_stride(std::size_t dim) noexcept { return (dim + 15) & ~std::size_t{15}; }

  void train(const VectorArena& arena) {
    if (arena.size() == 0) {
      throw std::invalid_argument("ScalarQuantizer::train: empty arena");
    }
    dim_ = arena.dim();
    stride_ = padded_stride(dim_);
    scale_.assign(dim_, 0.f);
    for (std::uint32_t id = 0; id < arena.size(); ++id) {
      const float* row = arena.vector(id);
      for (std::size_t d = 0; d < dim_; ++d) {
        const float magnitude = std::fabs(row[d]);
        if (magnitude > scale_[d]) {
          scale_[d] = magnitude;
        }
      }
    }
    for (float& scale : scale_) {
      scale = scale == 0.f ? 1.f : scale / 127.f;
    }
    trained_ = true;
  }

  void encode(const VectorArena& arena) {
    if (!trained_ || arena.dim() != dim_) {
      throw std::logic_error("ScalarQuantizer::encode: train() on this dimension first");
    }
    codes_.assign(arena.size() * stride_, 0);
    for (std::uint32_t id = 0; id < arena.size(); ++id) {
      encode_row(arena.vector(id), codes_.data() + static_cast<std::size_t>(id) * stride_);
    }
    rows_ = arena.size();
  }

  std::vector<std::int8_t> encode_query(std::span<const float> query) const {
    if (!trained_ || query.size() != dim_ || query.data() == nullptr) {
      throw std::invalid_argument("ScalarQuantizer::encode_query: expected dim floats");
    }
    std::vector<std::int8_t> out(stride_, 0);
    encode_row(query.data(), out.data());
    return out;
  }

  const std::int8_t* row(std::uint32_t id) const noexcept {
    return codes_.data() + static_cast<std::size_t>(id) * stride_;
  }

  float scale(std::size_t dim) const noexcept { return scale_[dim]; }
  std::size_t dim() const noexcept { return dim_; }
  std::size_t stride() const noexcept { return stride_; }
  std::size_t size() const noexcept { return rows_; }
  std::size_t bytes() const noexcept { return codes_.size(); }
  bool trained() const noexcept { return trained_; }

 private:
  void encode_row(const float* in, std::int8_t* out) const {
    for (std::size_t d = 0; d < dim_; ++d) {
      int rounded = static_cast<int>(std::lround(static_cast<double>(in[d]) / static_cast<double>(scale_[d])));
      if (rounded > 127) {
        rounded = 127;
      } else if (rounded < -127) {
        rounded = -127;
      }
      out[d] = static_cast<std::int8_t>(rounded);
    }
  }

  std::size_t dim_ = 0;
  std::size_t stride_ = 0;
  std::size_t rows_ = 0;
  bool trained_ = false;
  std::vector<float> scale_;
  std::vector<std::int8_t> codes_;
};

}  // namespace bazavec
