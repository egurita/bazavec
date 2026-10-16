#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bazavec {

// Packed bits, one per row. Test is a shift and a mask, so the HNSW beam can
// apply it to every candidate without leaving the search loop.
class Bitset {
 public:
  explicit Bitset(std::size_t n) : n_(n), words_((n + 63) / 64, 0) {}

  void set(std::uint32_t id) noexcept {
    words_[id >> 6] |= std::uint64_t{1} << (id & 63);
  }

  bool test(std::uint32_t id) const noexcept {
    return (words_[id >> 6] & (std::uint64_t{1} << (id & 63))) != 0;
  }

  std::size_t size() const noexcept { return n_; }

  std::size_t count() const noexcept {
    std::size_t total = 0;
    for (std::uint64_t word : words_) {
      total += static_cast<std::size_t>(__builtin_popcountll(word));
    }
    return total;
  }

 private:
  std::size_t n_ = 0;
  std::vector<std::uint64_t> words_;
};

// Columnar attributes kept beside the arena. The predicate is resolved once,
// into a bitset, and that bitset is what search consults.
struct Columns {
  std::vector<std::uint32_t> category;
  std::vector<float> value;
};

inline Bitset where_eq_and_ge(const Columns& columns, std::uint32_t category, float min_value) {
  const std::size_t n = columns.category.size();
  Bitset keep(n);
  for (std::uint32_t id = 0; id < n; ++id) {
    if (columns.category[id] == category && columns.value[id] >= min_value) {
      keep.set(id);
    }
  }
  return keep;
}

}  // namespace bazavec
