#include "bazavec/bazavec.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

bool throws_runtime(const auto& fn) {
  try {
    fn();
  } catch (const std::runtime_error&) {
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

}  // namespace

int main() {
  constexpr std::size_t kCount = 200;
  constexpr std::size_t kDim = 32;
  bazavec::VectorArena arena = make_arena(kCount, kDim, 7);
  bazavec::HnswIndex::Params params;
  params.m = 8;
  params.ef_construction = 64;
  params.ef_search = 32;
  params.threads = 1;
  params.seed = 11;
  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, params);
  index.build();

  std::mt19937 rng(99);
  std::uniform_int_distribution<int> pick(0, static_cast<int>(kCount - 1));
  std::vector<std::vector<bazavec::Hit>> expected;
  expected.reserve(5);
  for (int query = 0; query < 5; ++query) {
    const float* row = arena.vector(static_cast<std::uint32_t>(pick(rng)));
    expected.push_back(index.search_knn(std::span<const float>(row, kDim), 5));
  }

  const std::string path = "/tmp/bazavec-" + std::to_string(::getpid()) + ".bzv";
  bazavec::save_bzv(path, arena, index);

  {
    bazavec::MappedIndex mapped(path);
    CHECK(mapped.arena().mapped());
    CHECK(mapped.arena().size() == kCount);
    CHECK(mapped.arena().dim() == kDim);
    CHECK(mapped.contains(mapped.arena().data()));
    CHECK(mapped.contains(mapped.index().layer0_data()));
    const auto* base = static_cast<const std::uint8_t*>(mapped.mapping());
    CHECK(mapped.arena().data() == reinterpret_cast<const float*>(base + bazavec::kBzvPayloadAlign));

    bool degrees = true;
    for (std::uint32_t id = 0; id < kCount; ++id) {
      if (mapped.index().layer0_degree(id) != index.layer0_degree(id)) {
        degrees = false;
      }
    }
    CHECK(degrees);
    CHECK(mapped.index().entry_point() == index.entry_point());
    CHECK(mapped.index().max_level() == index.max_level());

    bool hits = true;
    std::mt19937 again(99);
    for (int query = 0; query < 5; ++query) {
      const float* row = arena.vector(static_cast<std::uint32_t>(pick(again)));
      const std::vector<bazavec::Hit> got = mapped.search_knn(std::span<const float>(row, kDim), 5);
      if (got.size() != expected[static_cast<std::size_t>(query)].size()) {
        hits = false;
        continue;
      }
      for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i].id != expected[static_cast<std::size_t>(query)][i].id ||
            got[i].score != expected[static_cast<std::size_t>(query)][i].score) {
          hits = false;
        }
      }
    }
    CHECK(hits);
  }

  {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::streamoff>(file.tellg());
    CHECK(size > static_cast<std::streamoff>(bazavec::kBzvPayloadAlign));
    file.seekg(size - 1);
    char byte = 0;
    file.read(&byte, 1);
    file.seekp(size - 1);
    const char flipped = static_cast<char>(static_cast<unsigned char>(byte) ^ 0x1u);
    file.write(&flipped, 1);
    file.close();
    CHECK(throws_runtime([&] { bazavec::MappedIndex broken(path); }));
  }

  const std::string junk = path + ".bad";
  {
    std::ofstream file(junk, std::ios::binary);
    std::vector<char> bytes(256, 0);
    bytes[0] = 'X';
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  CHECK(throws_runtime([&] { bazavec::MappedIndex broken(junk); }));

  ::unlink(path.c_str());
  ::unlink(junk.c_str());

  if (g_failed != 0) {
    std::cerr << g_failed << " failed, " << g_passed << " passed\n";
    return 1;
  }
  std::cout << g_passed << " passed\n";
  return 0;
}
