#include "bazavec/bazavec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
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

#define CHECK(condition) ::check(static_cast<bool>(condition), #condition, __FILE__, __LINE__)

bazavec::VectorArena make_arena(std::size_t n, std::size_t dim, std::uint32_t seed) {
  bazavec::VectorArena arena(dim);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> packed(n * dim);
  for (float& value : packed) {
    value = dist(rng);
  }
  arena.reserve(n);
  arena.add_many(packed.data(), n);
  return arena;
}

float recall_ids(const std::vector<bazavec::Hit>& got, const std::vector<std::uint32_t>& truth) {
  if (truth.empty()) {
    return 1.f;
  }
  std::size_t hit = 0;
  for (const bazavec::Hit& candidate : got) {
    for (std::uint32_t expected : truth) {
      if (candidate.id == expected) {
        ++hit;
        break;
      }
    }
  }
  return static_cast<float>(hit) / static_cast<float>(truth.size());
}

void test_quantizer_roundtrip_and_simd() {
  bazavec::VectorArena arena = make_arena(64, 128, 5);
  bazavec::ScalarQuantizer quant;
  quant.train(arena);
  quant.encode(arena);
  CHECK(quant.stride() == 128);
  CHECK(quant.bytes() * 4 == arena.size() * arena.dim() * sizeof(float));

  bool within_half_bin = true;
  for (std::uint32_t id = 0; id < arena.size() && within_half_bin; ++id) {
    const float* original = arena.vector(id);
    const std::int8_t* code = quant.row(id);
    for (std::size_t d = 0; d < arena.dim(); ++d) {
      const float restored = static_cast<float>(code[d]) * quant.scale(d);
      if (std::fabs(restored - original[d]) > quant.scale(d) * 0.5f + 1e-5f) {
        within_half_bin = false;
        break;
      }
    }
    CHECK(bazavec::l2_squared_i8_simd(code, code, quant.stride()) == 0);
  }
  CHECK(within_half_bin);

  bool kernels_match = true;
  for (std::uint32_t i = 0; i < arena.size() && kernels_match; ++i) {
    for (std::uint32_t j = 0; j < arena.size(); ++j) {
      const std::int32_t scalar =
          bazavec::l2_squared_i8_scalar(quant.row(i), quant.row(j), quant.stride());
      const std::int32_t simd = bazavec::l2_squared_i8_simd(quant.row(i), quant.row(j), quant.stride());
      if (scalar != simd) {
        kernels_match = false;
        break;
      }
    }
  }
  CHECK(kernels_match);

  const float odd[7] = {0.f, 0.5f, -0.5f, 1.f, -1.f, 0.25f, 3.f};
  bazavec::VectorArena skinny(7);
  skinny.add(std::span<const float>(odd, 7));
  bazavec::ScalarQuantizer narrow;
  narrow.train(skinny);
  narrow.encode(skinny);
  CHECK(narrow.stride() == 16);
  CHECK(bazavec::l2_squared_i8_scalar(narrow.row(0), narrow.row(0), narrow.stride()) ==
        bazavec::l2_squared_i8_simd(narrow.row(0), narrow.row(0), narrow.stride()));
}

void test_bitset_predicate() {
  bazavec::Bitset bits(130);
  bits.set(0);
  bits.set(63);
  bits.set(64);
  bits.set(129);
  CHECK(bits.test(0));
  CHECK(bits.test(63));
  CHECK(bits.test(64));
  CHECK(bits.test(129));
  CHECK(!bits.test(1));
  CHECK(!bits.test(62));
  CHECK(!bits.test(65));
  CHECK(bits.count() == 4);

  bazavec::Columns columns;
  columns.category = {1, 2, 2, 2, 3};
  columns.value = {10.f, 0.f, 5.f, 9.f, 100.f};
  const bazavec::Bitset keep = bazavec::where_eq_and_ge(columns, 2, 5.f);
  CHECK(!keep.test(0));
  CHECK(!keep.test(1));
  CHECK(keep.test(2));
  CHECK(keep.test(3));
  CHECK(!keep.test(4));
  CHECK(keep.count() == 2);
}

void test_filtered_hnsw_recall() {
  constexpr std::size_t kN = 1500;
  constexpr std::size_t kDim = 32;
  bazavec::VectorArena arena = make_arena(kN, kDim, 8);
  bazavec::Columns columns;
  columns.category.resize(kN);
  columns.value.resize(kN);
  for (std::uint32_t id = 0; id < kN; ++id) {
    columns.category[id] = id % 4;
    columns.value[id] = static_cast<float>(id % 50);
  }
  const bazavec::Bitset keep = bazavec::where_eq_and_ge(columns, 1, 20.f);
  CHECK(keep.count() > 50);

  bazavec::HnswIndex::Params params;
  params.threads = 1;
  params.ef_construction = 64;
  params.seed = 8;
  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, params);
  index.build();

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> query(kDim);
  double recall_acc = 0.0;
  constexpr int kQueries = 15;
  bool predicate_held = true;
  for (int q = 0; q < kQueries; ++q) {
    for (float& value : query) {
      value = dist(rng);
    }
    const std::vector<bazavec::Hit> got =
        index.search_knn(query, 10, 64, [&](std::uint32_t id) { return keep.test(id); });
    for (const bazavec::Hit& hit : got) {
      if (!keep.test(hit.id)) {
        predicate_held = false;
      }
    }
    std::vector<bazavec::Hit> pool;
    pool.reserve(keep.count());
    for (std::uint32_t id = 0; id < kN; ++id) {
      if (!keep.test(id)) {
        continue;
      }
      pool.push_back(bazavec::Hit{id, bazavec::l2_squared_simd(query.data(), arena.vector(id), arena.dim())});
    }
    std::sort(pool.begin(), pool.end(), [](const bazavec::Hit& a, const bazavec::Hit& b) {
      if (a.score < b.score) {
        return true;
      }
      if (b.score < a.score) {
        return false;
      }
      return a.id < b.id;
    });
    if (pool.size() > 10) {
      pool.resize(10);
    }
    std::vector<std::uint32_t> truth;
    truth.reserve(pool.size());
    for (const bazavec::Hit& hit : pool) {
      truth.push_back(hit.id);
    }
    recall_acc += static_cast<double>(recall_ids(got, truth));
  }
  const float recall = static_cast<float>(recall_acc / kQueries);
  std::cout << "filtered recall@10 " << recall << "\n";
  CHECK(predicate_held);
  CHECK(recall > 0.95f);
}

}  // namespace

int main() {
  test_quantizer_roundtrip_and_simd();
  test_bitset_predicate();
  test_filtered_hnsw_recall();
  std::cout << "phase 3 tests: " << g_passed << " passed, " << g_failed << " failed\n";
  return g_failed == 0 ? 0 : 1;
}
