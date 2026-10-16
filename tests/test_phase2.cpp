#include "bazavec/bazavec.hpp"

#include <algorithm>
#include <cstdint>
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

#define CHECK(condition) ::check(static_cast<bool>(condition), #condition, __FILE__, __LINE__)

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

bool throws_logic(const auto& fn) {
  try {
    fn();
  } catch (const std::logic_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

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

float recall_at(const std::vector<bazavec::Hit>& got, const std::vector<bazavec::Hit>& truth) {
  if (truth.empty()) {
    return 1.f;
  }
  std::size_t hit = 0;
  for (const bazavec::Hit& candidate : got) {
    for (const bazavec::Hit& expected : truth) {
      if (candidate.id == expected.id) {
        ++hit;
        break;
      }
    }
  }
  return static_cast<float>(hit) / static_cast<float>(truth.size());
}

float mean_recall(const bazavec::HnswIndex& index, const bazavec::VectorArena& arena, std::size_t queries,
                  std::size_t k, std::uint32_t seed, std::size_t ef) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  double acc = 0.0;
  std::vector<float> query(arena.dim());
  for (std::size_t q = 0; q < queries; ++q) {
    for (float& value : query) {
      value = dist(rng);
    }
    const std::vector<bazavec::Hit> got =
        index.search_knn(query, k, ef);
    const std::vector<bazavec::Hit> truth =
        bazavec::brute_force_search(arena, query, k, index.metric(), bazavec::DistanceKernel::Simd);
    acc += static_cast<double>(recall_at(got, truth));
  }
  return static_cast<float>(acc / static_cast<double>(queries));
}

void expect_graph_sane(const bazavec::HnswIndex& index) {
  const std::uint32_t n = static_cast<std::uint32_t>(index.size());
  const std::uint32_t stride = index.layer0_stride();
  CHECK(stride == index.params().m * 2);
  CHECK(index.entry_point() < n);
  CHECK(index.max_level() >= 0);
  CHECK(index.max_level() <= bazavec::HnswIndex::kMaxLevel);
  CHECK(index.layer0_data() != nullptr);
  CHECK(index.graph_bytes() >= static_cast<std::size_t>(n) * stride * sizeof(std::uint32_t));
  const std::uint32_t* base = index.layer0_data();
  bool neighbors_ok = true;
  for (std::uint32_t id = 0; id < n && neighbors_ok; ++id) {
    const std::uint16_t degree = index.layer0_degree(id);
    if (degree == 0 || degree > stride) {
      neighbors_ok = false;
      break;
    }
    for (std::uint16_t slot = 0; slot < degree; ++slot) {
      const std::uint32_t neighbor = base[static_cast<std::size_t>(id) * stride + slot];
      if (neighbor >= n || neighbor == id) {
        neighbors_ok = false;
        break;
      }
      for (std::uint16_t other = 0; other < slot; ++other) {
        if (base[static_cast<std::size_t>(id) * stride + other] == neighbor) {
          neighbors_ok = false;
          break;
        }
      }
    }
  }
  CHECK(neighbors_ok);
}

void test_heuristic_prefers_diverse_neighbors() {
  bazavec::VectorArena arena(1);
  const float rows[] = {-5.f, 1.f, 1.05f};
  arena.add(std::span<const float>(&rows[0], 1));
  arena.add(std::span<const float>(&rows[1], 1));
  arena.add(std::span<const float>(&rows[2], 1));

  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, {});
  const float query = 0.f;
  const std::uint32_t candidates[] = {0, 1, 2};
  const std::vector<std::uint32_t> chosen = index.select_neighbors_heuristic(
      std::span<const float>(&query, 1), candidates, 2, 0, false, true);
  CHECK(chosen.size() == 2);
  CHECK(chosen[0] == 1);
  CHECK(chosen[1] == 0);
}

void test_rejects() {
  bazavec::VectorArena arena = make_arena(4, 8, 1);
  CHECK(throws_invalid([&] {
    bazavec::HnswIndex::Params params;
    params.m = 1;
    bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, params);
    (void)index;
  }));
  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, {});
  std::vector<float> query(8, 0.f);
  CHECK(throws_logic([&] { (void)index.search_knn(query, 1); }));
  index.build();
  CHECK(throws_logic([&] { index.build(); }));
  CHECK(throws_invalid([&] { (void)index.search_knn(std::span<const float>(query.data(), 3), 1); }));
  CHECK(index.search_knn(query, 0).empty());
}

void test_singleton_and_exhaustive_tiny() {
  bazavec::VectorArena one = make_arena(1, 8, 3);
  bazavec::HnswIndex::Params params;
  params.threads = 1;
  params.seed = 7;
  bazavec::HnswIndex single(one, bazavec::Metric::L2Squared, params);
  single.build();
  std::vector<float> query(one.vector(0), one.vector(0) + one.dim());
  const std::vector<bazavec::Hit> hit = single.search_knn(query, 1, 8);
  CHECK(hit.size() == 1);
  CHECK(hit[0].id == 0);
  CHECK(hit[0].score == 0.f);

  constexpr std::size_t kN = 40;
  bazavec::VectorArena arena = make_arena(kN, 16, 11);
  params.ef_construction = 64;
  params.ef_search = 64;
  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, params);
  index.build();
  expect_graph_sane(index);

  std::vector<float> probe(arena.dim());
  bool self_ok = true;
  for (std::uint32_t id = 0; id < kN; ++id) {
    std::copy(arena.vector(id), arena.vector(id) + arena.dim(), probe.begin());
    const std::vector<bazavec::Hit> self = index.search_knn(probe, 1, kN);
    if (self.size() != 1 || self[0].id != id || self[0].score != 0.f) {
      std::cerr << "self search id " << id << "\n";
      self_ok = false;
      break;
    }
  }
  CHECK(self_ok);

  std::mt19937 rng(19);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (float& value : probe) {
    value = dist(rng);
  }
  const std::vector<bazavec::Hit> all = index.search_knn(probe, kN, kN);
  CHECK(all.size() == kN);
}

void test_recall(std::size_t n, std::size_t dim, bazavec::Metric metric, std::uint32_t threads, std::uint32_t seed,
                 const char* label) {
  bazavec::VectorArena arena = make_arena(n, dim, seed);
  bazavec::HnswIndex::Params params;
  params.m = 16;
  params.ef_construction = 128;
  params.ef_search = 64;
  params.threads = threads;
  params.seed = seed;
  bazavec::HnswIndex index(arena, metric, params);
  index.build();
  if (threads == 1) {
    expect_graph_sane(index);
  }
  const float recall = mean_recall(index, arena, 20, 10, seed + 99, 64);
  std::cout << label << " recall@10 " << recall << "\n";
  CHECK(recall > 0.95f);
}

void test_serial_matches_second_build() {
  bazavec::VectorArena arena = make_arena(250, 32, 21);
  bazavec::HnswIndex::Params params;
  params.threads = 1;
  params.ef_construction = 64;
  params.seed = 21;
  bazavec::HnswIndex first(arena, bazavec::Metric::L2Squared, params);
  bazavec::HnswIndex second(arena, bazavec::Metric::L2Squared, params);
  first.build();
  second.build();
  bool same = first.entry_point() == second.entry_point() && first.max_level() == second.max_level() &&
              first.layer0_stride() == second.layer0_stride() && first.size() == second.size();
  const std::uint32_t stride = first.layer0_stride();
  for (std::uint32_t id = 0; same && id < first.size(); ++id) {
    if (first.layer0_degree(id) != second.layer0_degree(id)) {
      same = false;
      break;
    }
    const std::uint16_t degree = first.layer0_degree(id);
    for (std::uint16_t slot = 0; slot < degree; ++slot) {
      const std::size_t offset = static_cast<std::size_t>(id) * stride + slot;
      if (first.layer0_data()[offset] != second.layer0_data()[offset]) {
        same = false;
        break;
      }
    }
  }
  CHECK(same);
}

void test_many_searches_keep_self() {
  bazavec::VectorArena arena = make_arena(300, 32, 33);
  bazavec::HnswIndex::Params params;
  params.threads = 1;
  params.ef_construction = 64;
  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, params);
  index.build();
  std::vector<float> query(arena.dim());
  bool stable = true;
  for (int round = 0; round < 300; ++round) {
    const std::uint32_t id = static_cast<std::uint32_t>(round % 300);
    std::copy(arena.vector(id), arena.vector(id) + arena.dim(), query.begin());
    const std::vector<bazavec::Hit> hit = index.search_knn(query, 1, 32);
    if (hit.size() != 1 || hit[0].id != id) {
      stable = false;
      break;
    }
  }
  CHECK(stable);
}

}  // namespace

int main() {
  test_heuristic_prefers_diverse_neighbors();
  test_rejects();
  test_singleton_and_exhaustive_tiny();
  test_recall(1200, 64, bazavec::Metric::L2Squared, 1, 42, "serial-l2");
  test_recall(1200, 64, bazavec::Metric::L2Squared, 4, 42, "parallel-l2");
  test_recall(600, 32, bazavec::Metric::Cosine, 1, 9, "serial-cosine");
  test_serial_matches_second_build();
  test_many_searches_keep_self();

  std::cout << "phase 2 tests: " << g_passed << " passed, " << g_failed << " failed\n";
  return g_failed == 0 ? 0 : 1;
}
