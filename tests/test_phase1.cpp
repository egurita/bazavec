#include "bazavec/bazavec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (condition) {
    ++g_passed;
    return;
  }
  ++g_failed;
  std::cerr << file << ":" << line << " failed: " << expression << "\n";
}

void check_close(float got, float expected, float abs_tol, const char* expression, const char* file,
                 int line) {
  if (std::isfinite(got) && std::isfinite(expected) && std::fabs(got - expected) <= abs_tol) {
    ++g_passed;
    return;
  }
  ++g_failed;
  std::cerr << file << ":" << line << " failed: " << expression << " got " << got << " expected "
            << expected << " tol " << abs_tol << "\n";
}

#define CHECK(condition) ::check(static_cast<bool>(condition), #condition, __FILE__, __LINE__)
#define CHECK_CLOSE(got, expected, tol) \
  ::check_close((got), (expected), (tol), #got " ~ " #expected, __FILE__, __LINE__)

bool throws_invalid(const auto& fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

float tol_for(float expected) { return 1e-5f + 1e-4f * std::fabs(expected); }

double l2_double(const float* a, const float* b, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    acc += diff * diff;
  }
  return acc;
}

double dot_double(const float* a, const float* b, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return acc;
}

void test_stride_and_alignment() {
  using bazavec::VectorArena;
  using bazavec::kArenaAlignment;

  VectorArena arena(100);
  CHECK(arena.dim() == 100);
  CHECK(arena.stride() == 128);
  CHECK(arena.size() == 0);
  CHECK(arena.data() == nullptr);

  std::vector<float> row(100);
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = static_cast<float>(i + 1);
  }
  arena.reserve(4);
  const std::uint32_t id = arena.add(row);
  CHECK(id == 0);
  CHECK(arena.size() == 1);
  CHECK(arena.capacity() >= 4);
  const float* first = arena.vector(0);
  CHECK(reinterpret_cast<std::uintptr_t>(arena.data()) % kArenaAlignment == 0);
  CHECK(reinterpret_cast<std::uintptr_t>(first) % kArenaAlignment == 0);
  CHECK(first == arena.data());
  for (std::size_t i = 0; i < 100; ++i) {
    CHECK(first[i] == row[i]);
  }
  for (std::size_t i = 100; i < arena.stride(); ++i) {
    CHECK(first[i] == 0.f);
  }

  std::vector<float> packed(3 * 100);
  for (std::size_t row_index = 0; row_index < 3; ++row_index) {
    for (std::size_t col = 0; col < 100; ++col) {
      packed[row_index * 100 + col] = static_cast<float>(row_index * 1000 + col);
    }
  }
  const float* stable = arena.vector(0);
  arena.add_many(packed.data(), 3);
  CHECK(arena.vector(0) == stable);
  CHECK(arena.size() == 4);
  for (std::uint32_t row_index = 0; row_index < 4; ++row_index) {
    const float* pointer = arena.vector(row_index);
    CHECK(pointer == arena.data() + static_cast<std::size_t>(row_index) * arena.stride());
    CHECK(reinterpret_cast<std::uintptr_t>(pointer) % kArenaAlignment == 0);
  }
  CHECK(arena.vector(2)[5] == 1005.f);
  CHECK(arena.vector(2)[100] == 0.f);

  VectorArena wide(128);
  std::vector<float> block(128 * 2, 0.5f);
  wide.reserve(2);
  wide.add_many(block.data(), 2);
  CHECK(wide.stride() == 128);
  CHECK(wide.vector(1)[127] == 0.5f);
  CHECK(wide.allocated_bytes() % kArenaAlignment == 0);
}

void test_move_and_rejects() {
  bazavec::VectorArena arena(4);
  const float values[4] = {1.f, 2.f, 3.f, 4.f};
  arena.add(std::span<const float>(values, 4));

  bazavec::VectorArena moved(std::move(arena));
  CHECK(arena.size() == 0);
  CHECK(arena.capacity() == 0);
  CHECK(moved.size() == 1);
  CHECK(moved.vector(0)[3] == 4.f);

  const float again[4] = {9.f, 8.f, 7.f, 6.f};
  arena.add(std::span<const float>(again, 4));
  CHECK(arena.size() == 1);
  CHECK(arena.vector(0)[0] == 9.f);

  CHECK(throws_invalid([] { bazavec::VectorArena bad(0); }));
  CHECK(throws_invalid([&] { moved.add(std::span<const float>(values, 3)); }));
  CHECK(throws_invalid([] {
    bazavec::VectorArena empty(4);
    const float row[4] = {1.f, 1.f, 1.f, 1.f};
    empty.add(std::span<const float>(row, 4));
    bazavec::brute_force_search(empty, std::span<const float>(row, 3), 1,
                                 bazavec::Metric::L2Squared);
  }));
}

void test_exact_kernels() {
  const std::size_t dims[] = {1, 4, 7, 8, 15, 16, 17, 31, 32, 33, 100, 128, 1024};
  for (std::size_t n : dims) {
    std::vector<float> a(n, 1.f);
    std::vector<float> z(n, 0.f);
    std::vector<float> half(n, 0.5f);
    std::vector<float> neg(n, -1.f);

    const float expect_n = static_cast<float>(n);
    CHECK(bazavec::l2_squared_scalar(a.data(), z.data(), n) == expect_n);
    CHECK(bazavec::l2_squared_simd(a.data(), z.data(), n) == expect_n);
    CHECK(bazavec::l2_squared_simd(half.data(), z.data(), n) == expect_n * 0.25f);
    CHECK(bazavec::l2_squared_simd(a.data(), a.data(), n) == 0.f);
    CHECK(bazavec::inner_product_simd(a.data(), a.data(), n) == expect_n);
    CHECK(bazavec::inner_product_simd(a.data(), neg.data(), n) == -expect_n);
    CHECK(bazavec::inner_product_scalar(a.data(), neg.data(), n) == -expect_n);
    // sqrt(n) * sqrt(n) is not always bit-exact, so cosine is checked tightly, not with ==.
    CHECK_CLOSE(bazavec::cosine_simd(a.data(), a.data(), n), 1.f, 1e-5f);
    CHECK_CLOSE(bazavec::cosine_simd(a.data(), neg.data(), n), -1.f, 1e-5f);
    CHECK(bazavec::cosine_simd(z.data(), a.data(), n) == 0.f);
    CHECK(bazavec::cosine_scalar(z.data(), a.data(), n) == 0.f);
  }

  const float x_axis[4] = {1.f, 0.f, 0.f, 0.f};
  const float y_axis[4] = {0.f, 1.f, 0.f, 0.f};
  CHECK(bazavec::cosine_simd(x_axis, x_axis, 4) == 1.f);
  CHECK(bazavec::cosine_simd(x_axis, y_axis, 4) == 0.f);
  CHECK(bazavec::cosine_scalar(x_axis, y_axis, 4) == 0.f);

  alignas(128) float storage_a[40];
  alignas(128) float storage_b[40];
  for (int i = 0; i < 40; ++i) {
    storage_a[i] = 1.f;
    storage_b[i] = 0.f;
  }
  float* mis_a = storage_a + 1;
  float* mis_b = storage_b + 1;
  CHECK(bazavec::l2_squared_simd(mis_a, mis_b, 17) == 17.f);
  CHECK(bazavec::inner_product_simd(mis_a, mis_a, 17) == 17.f);
}

void test_random_against_double() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const std::size_t dims[] = {1, 7, 16, 17, 32, 100, 128};
  for (std::size_t n : dims) {
    std::vector<float> a(n);
    std::vector<float> b(n);
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = dist(rng);
      b[i] = dist(rng);
    }
    const float l2_ref = static_cast<float>(l2_double(a.data(), b.data(), n));
    const float dot_ref = static_cast<float>(dot_double(a.data(), b.data(), n));
    CHECK_CLOSE(bazavec::l2_squared_scalar(a.data(), b.data(), n), l2_ref, tol_for(l2_ref));
    CHECK_CLOSE(bazavec::l2_squared_simd(a.data(), b.data(), n), l2_ref, tol_for(l2_ref));
    CHECK_CLOSE(bazavec::inner_product_scalar(a.data(), b.data(), n), dot_ref, tol_for(dot_ref));
    CHECK_CLOSE(bazavec::inner_product_simd(a.data(), b.data(), n), dot_ref, tol_for(dot_ref));

    const double na = std::sqrt(dot_double(a.data(), a.data(), n));
    const double nb = std::sqrt(dot_double(b.data(), b.data(), n));
    const float cosine_ref =
        (na == 0.0 || nb == 0.0)
            ? 0.f
            : static_cast<float>(dot_double(a.data(), b.data(), n) / (na * nb));
    CHECK_CLOSE(bazavec::cosine_scalar(a.data(), b.data(), n), cosine_ref, tol_for(cosine_ref));
    CHECK_CLOSE(bazavec::cosine_simd(a.data(), b.data(), n), cosine_ref, tol_for(cosine_ref));
  }
}

std::vector<bazavec::Hit> reference_topk(const bazavec::VectorArena& arena,
                                           std::span<const float> query, std::size_t k,
                                           bazavec::Metric metric, bazavec::DistanceKernel kernel) {
  std::vector<float> padded(arena.stride(), 0.f);
  std::memcpy(padded.data(), query.data(), arena.dim() * sizeof(float));
  std::vector<bazavec::Hit> hits;
  hits.reserve(arena.size());
  for (std::uint32_t i = 0; i < arena.size(); ++i) {
    hits.push_back(bazavec::Hit{
        i, bazavec::distance(padded.data(), arena.vector(i), arena.stride(), metric, kernel)});
  }
  const bool l2 = metric == bazavec::Metric::L2Squared;
  std::sort(hits.begin(), hits.end(), [l2](const bazavec::Hit& a, const bazavec::Hit& b) {
    if (a.score != b.score) {
      return l2 ? a.score < b.score : a.score > b.score;
    }
    return a.id < b.id;
  });
  if (hits.size() > k) {
    hits.resize(k);
  }
  return hits;
}

void expect_same_hits(const std::vector<bazavec::Hit>& got, const std::vector<bazavec::Hit>& expected,
                      const std::string& label) {
  CHECK(got.size() == expected.size());
  const std::size_t n = got.size() < expected.size() ? got.size() : expected.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i].id != expected[i].id || got[i].score != expected[i].score) {
      std::cerr << label << " rank " << i << " got id " << got[i].id << " score " << got[i].score
                << " expected id " << expected[i].id << " score " << expected[i].score << "\n";
      CHECK(false);
    } else {
      CHECK(true);
    }
  }
}

void test_brute_force_geometry() {
  constexpr std::size_t kDim = 32;
  bazavec::VectorArena arena(kDim);
  std::vector<float> e0(kDim, 0.f);
  std::vector<float> e1(kDim, 0.f);
  std::vector<float> far(kDim, 0.f);
  e0[0] = 1.f;
  e1[1] = 1.f;
  far[0] = 10.f;
  arena.add(e0);
  arena.add(e1);
  arena.add(far);

  std::vector<float> query(kDim, 0.f);
  query[0] = 1.f;
  const std::vector<bazavec::Hit> hits = bazavec::brute_force_search(
      arena, query, 2, bazavec::Metric::L2Squared, bazavec::DistanceKernel::Simd);
  CHECK(hits.size() == 2);
  CHECK(hits[0].id == 0);
  CHECK(hits[0].score == 0.f);
  CHECK(hits[1].id == 1);
  CHECK(hits[1].score == 2.f);

  const std::vector<bazavec::Hit> all = bazavec::brute_force_search(
      arena, query, 10, bazavec::Metric::L2Squared);
  CHECK(all.size() == 3);
  CHECK(all[2].id == 2);

  const std::vector<bazavec::Hit> none =
      bazavec::brute_force_search(arena, query, 0, bazavec::Metric::L2Squared);
  CHECK(none.empty());

  bazavec::VectorArena tied(1);
  const float rows[] = {0.f, 1.f, 1.f};
  tied.add(std::span<const float>(&rows[0], 1));
  tied.add(std::span<const float>(&rows[1], 1));
  tied.add(std::span<const float>(&rows[2], 1));
  const float origin = 1.f;
  const std::vector<bazavec::Hit> tie = bazavec::brute_force_search(
      tied, std::span<const float>(&origin, 1), 2, bazavec::Metric::L2Squared);
  CHECK(tie.size() == 2);
  CHECK(tie[0].id == 1);
  CHECK(tie[0].score == 0.f);
  CHECK(tie[1].id == 2);
  CHECK(tie[1].score == 0.f);

  bazavec::VectorArena angles(4);
  const float x[4] = {1.f, 0.f, 0.f, 0.f};
  const float y[4] = {0.f, 1.f, 0.f, 0.f};
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  angles.add(std::span<const float>(y, 4));
  angles.add(std::span<const float>(zero, 4));
  angles.add(std::span<const float>(x, 4));
  const std::vector<bazavec::Hit> cosine_hits = bazavec::brute_force_search(
      angles, std::span<const float>(x, 4), 3, bazavec::Metric::Cosine);
  CHECK(cosine_hits.size() == 3);
  CHECK(cosine_hits[0].id == 2);
  CHECK(cosine_hits[0].score == 1.f);
  CHECK(cosine_hits[1].id == 0);
  CHECK(cosine_hits[1].score == 0.f);
  CHECK(cosine_hits[2].id == 1);
  CHECK(cosine_hits[2].score == 0.f);
}

void test_brute_force_matches_sort() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const std::size_t dims[] = {7, 100, 128};
  const bazavec::Metric metrics[] = {bazavec::Metric::L2Squared, bazavec::Metric::InnerProduct,
                                      bazavec::Metric::Cosine};
  const bazavec::DistanceKernel kernels[] = {bazavec::DistanceKernel::Scalar,
                                              bazavec::DistanceKernel::Simd};
  for (std::size_t dim : dims) {
    bazavec::VectorArena arena(dim);
    std::vector<float> packed(40 * dim);
    for (float& value : packed) {
      value = dist(rng);
    }
    arena.add_many(packed.data(), 40);
    std::vector<float> query(dim);
    for (float& value : query) {
      value = dist(rng);
    }
    for (bazavec::Metric metric : metrics) {
      for (bazavec::DistanceKernel kernel : kernels) {
        const std::vector<bazavec::Hit> got =
            bazavec::brute_force_search(arena, query, 10, metric, kernel);
        const std::vector<bazavec::Hit> expected =
            reference_topk(arena, query, 10, metric, kernel);
        expect_same_hits(got, expected,
                         std::string(bazavec::to_string(metric)) + "/" + bazavec::to_string(kernel) +
                             " dim " + std::to_string(dim));
      }
    }
  }
}

}  // namespace

int main() {
  test_stride_and_alignment();
  test_move_and_rejects();
  test_exact_kernels();
  test_random_against_double();
  test_brute_force_geometry();
  test_brute_force_matches_sort();

  std::cout << "phase 1 tests: " << g_passed << " passed, " << g_failed << " failed\n";
  return g_failed == 0 ? 0 : 1;
}
