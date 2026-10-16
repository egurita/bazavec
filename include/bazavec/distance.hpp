#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define BAZAVEC_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define BAZAVEC_AVX2 1
#endif

namespace bazavec {

enum class Metric : std::uint8_t {
  L2Squared = 0,
  InnerProduct = 1,
  Cosine = 2,
};

enum class DistanceKernel : std::uint8_t {
  Scalar = 0,
  Simd = 1,
};

inline const char* to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::L2Squared:
      return "L2Squared";
    case Metric::InnerProduct:
      return "InnerProduct";
    case Metric::Cosine:
      return "Cosine";
  }
  return "unknown";
}

inline const char* to_string(DistanceKernel kernel) noexcept {
  switch (kernel) {
    case DistanceKernel::Scalar:
      return "scalar";
    case DistanceKernel::Simd:
      return "simd";
  }
  return "unknown";
}

inline const char* simd_isa() noexcept {
#if defined(BAZAVEC_NEON)
  return "arm-neon-128";
#elif defined(BAZAVEC_AVX2)
  return "avx2-fma";
#else
  return "scalar";
#endif
}

// One accumulator, no manual unroll. Clang is told not to rewrite this into
// NEON: this function is the baseline the SIMD kernel is measured against.
inline float l2_squared_scalar(const float* a, const float* b, std::size_t n) noexcept {
  float acc = 0.f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    const float diff = a[i] - b[i];
    acc += diff * diff;
  }
  return acc;
}

inline float inner_product_scalar(const float* a, const float* b, std::size_t n) noexcept {
  float acc = 0.f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

// Not __restrict. cosine() dots a vector with itself to get its norm, and
// restrict on two equal pointers would make that call undefined.
inline float cosine_scalar(const float* a, const float* b, std::size_t n) noexcept {
  const float dot = inner_product_scalar(a, b, n);
  const float norm_a = std::sqrt(inner_product_scalar(a, a, n));
  const float norm_b = std::sqrt(inner_product_scalar(b, b, n));
  if (norm_a == 0.f || norm_b == 0.f) {
    return 0.f;
  }
  return dot / (norm_a * norm_b);
}

namespace detail {

enum class Op : std::uint8_t { L2Squared, InnerProduct };

#if defined(BAZAVEC_NEON)

template <Op operation>
inline void accumulate_neon(float32x4_t& acc, const float* a, const float* b) noexcept {
  const float32x4_t va = vld1q_f32(a);
  const float32x4_t vb = vld1q_f32(b);
  if constexpr (operation == Op::L2Squared) {
    const float32x4_t diff = vsubq_f32(va, vb);
    acc = vfmaq_f32(acc, diff, diff);
  } else {
    acc = vfmaq_f32(acc, va, vb);
  }
}

// 4 registers x 4 lanes. One FMA chain can retire a result only after the
// FMA latency, so a single accumulator leaves the pipes idle. Four independent
// chains give the scheduler work to issue while earlier FMAs are still in flight.
template <Op operation>
inline float reduce_neon(const float* a, const float* b, std::size_t n) noexcept {
  constexpr std::size_t kLanes = 4;
  constexpr std::size_t kUnroll = 4;
  constexpr std::size_t kStep = kLanes * kUnroll;

  float32x4_t acc0 = vdupq_n_f32(0.f);
  float32x4_t acc1 = vdupq_n_f32(0.f);
  float32x4_t acc2 = vdupq_n_f32(0.f);
  float32x4_t acc3 = vdupq_n_f32(0.f);

  std::size_t i = 0;
  for (; i + kStep <= n; i += kStep) {
    accumulate_neon<operation>(acc0, a + i, b + i);
    accumulate_neon<operation>(acc1, a + i + kLanes, b + i + kLanes);
    accumulate_neon<operation>(acc2, a + i + 2 * kLanes, b + i + 2 * kLanes);
    accumulate_neon<operation>(acc3, a + i + 3 * kLanes, b + i + 3 * kLanes);
  }

  const float32x4_t sum4 =
      vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  float sum = vaddvq_f32(sum4);

  for (; i < n; ++i) {
    if constexpr (operation == Op::L2Squared) {
      const float diff = a[i] - b[i];
      sum += diff * diff;
    } else {
      sum += a[i] * b[i];
    }
  }
  return sum;
}

#endif  // BAZAVEC_NEON

#if defined(BAZAVEC_AVX2)

inline float hsum_avx(__m256 v) noexcept {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  const __m128 swapped = _mm_movehdup_ps(sum);
  sum = _mm_add_ps(sum, swapped);
  const __m128 high = _mm_movehl_ps(swapped, sum);
  sum = _mm_add_ss(sum, high);
  return _mm_cvtss_f32(sum);
}

template <Op operation>
inline void accumulate_avx(__m256& acc, const float* a, const float* b) noexcept {
  const __m256 va = _mm256_loadu_ps(a);
  const __m256 vb = _mm256_loadu_ps(b);
  if constexpr (operation == Op::L2Squared) {
    const __m256 diff = _mm256_sub_ps(va, vb);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  } else {
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
}

// 4 registers x 8 lanes. loadu is correct for a public kernel: an aligned
// load faults on a misaligned pointer, and callers are not all arena rows.
template <Op operation>
inline float reduce_avx(const float* a, const float* b, std::size_t n) noexcept {
  constexpr std::size_t kLanes = 8;
  constexpr std::size_t kUnroll = 4;
  constexpr std::size_t kStep = kLanes * kUnroll;

  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  std::size_t i = 0;
  for (; i + kStep <= n; i += kStep) {
    accumulate_avx<operation>(acc0, a + i, b + i);
    accumulate_avx<operation>(acc1, a + i + kLanes, b + i + kLanes);
    accumulate_avx<operation>(acc2, a + i + 2 * kLanes, b + i + 2 * kLanes);
    accumulate_avx<operation>(acc3, a + i + 3 * kLanes, b + i + 3 * kLanes);
  }
  for (; i + kLanes <= n; i += kLanes) {
    accumulate_avx<operation>(acc0, a + i, b + i);
  }

  const __m256 folded = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  float sum = hsum_avx(folded);

  for (; i < n; ++i) {
    if constexpr (operation == Op::L2Squared) {
      const float diff = a[i] - b[i];
      sum += diff * diff;
    } else {
      sum += a[i] * b[i];
    }
  }
  return sum;
}

#endif  // BAZAVEC_AVX2

}  // namespace detail

inline float l2_squared_simd(const float* a, const float* b, std::size_t n) noexcept {
#if defined(BAZAVEC_NEON)
  return detail::reduce_neon<detail::Op::L2Squared>(a, b, n);
#elif defined(BAZAVEC_AVX2)
  return detail::reduce_avx<detail::Op::L2Squared>(a, b, n);
#else
  return l2_squared_scalar(a, b, n);
#endif
}

inline float inner_product_simd(const float* a, const float* b, std::size_t n) noexcept {
#if defined(BAZAVEC_NEON)
  return detail::reduce_neon<detail::Op::InnerProduct>(a, b, n);
#elif defined(BAZAVEC_AVX2)
  return detail::reduce_avx<detail::Op::InnerProduct>(a, b, n);
#else
  return inner_product_scalar(a, b, n);
#endif
}

inline float cosine_simd(const float* a, const float* b, std::size_t n) noexcept {
  const float dot = inner_product_simd(a, b, n);
  const float norm_a = std::sqrt(inner_product_simd(a, a, n));
  const float norm_b = std::sqrt(inner_product_simd(b, b, n));
  if (norm_a == 0.f || norm_b == 0.f) {
    return 0.f;
  }
  return dot / (norm_a * norm_b);
}

inline float l2_squared(const float* a, const float* b, std::size_t n,
                        DistanceKernel kernel = DistanceKernel::Simd) noexcept {
  return kernel == DistanceKernel::Scalar ? l2_squared_scalar(a, b, n)
                                           : l2_squared_simd(a, b, n);
}

inline float inner_product(const float* a, const float* b, std::size_t n,
                           DistanceKernel kernel = DistanceKernel::Simd) noexcept {
  return kernel == DistanceKernel::Scalar ? inner_product_scalar(a, b, n)
                                           : inner_product_simd(a, b, n);
}

inline float cosine(const float* a, const float* b, std::size_t n,
                    DistanceKernel kernel = DistanceKernel::Simd) noexcept {
  return kernel == DistanceKernel::Scalar ? cosine_scalar(a, b, n) : cosine_simd(a, b, n);
}

inline float distance(const float* a, const float* b, std::size_t n, Metric metric,
                      DistanceKernel kernel = DistanceKernel::Simd) noexcept {
  switch (metric) {
    case Metric::L2Squared:
      return l2_squared(a, b, n, kernel);
    case Metric::InnerProduct:
      return inner_product(a, b, n, kernel);
    case Metric::Cosine:
      return cosine(a, b, n, kernel);
  }
  return 0.f;
}

}  // namespace bazavec
