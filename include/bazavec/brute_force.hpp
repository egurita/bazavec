#pragma once

#include "detail/memory.hpp"
#include "distance.hpp"
#include "storage.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bazavec {

struct Hit {
  std::uint32_t id = 0;
  float score = 0.f;
};

// Exact top-k. Results are best-first.
//
// L2Squared: score is squared Euclidean distance, smaller is better.
// InnerProduct and Cosine: score is the similarity, larger is better.
// Squared L2 is enough to rank: sqrt is monotone, so it cannot change order.
//
// The heap is a max-heap of the k best rows, with the worst of those k on
// top. A new row replaces the top only when it is strictly better. That is
// O(N log K) score comparisons and O(K) memory, against O(N log N) for a
// full sort. Ties break toward the smaller id.
inline std::vector<Hit> brute_force_search(const VectorArena& arena, std::span<const float> query,
                                            std::size_t k, Metric metric,
                                            DistanceKernel kernel = DistanceKernel::Simd) {
  if (query.size() != arena.dim() || query.data() == nullptr) {
    throw std::invalid_argument("brute_force_search: query must contain dim floats");
  }
  if (k == 0 || arena.size() == 0) {
    return {};
  }

  const std::size_t stride = arena.stride();
  detail::AlignedPtr query_block = detail::allocate_floats(stride);
  std::memcpy(query_block.get(), query.data(), arena.dim() * sizeof(float));
  if (stride > arena.dim()) {
    std::memset(query_block.get() + arena.dim(), 0, (stride - arena.dim()) * sizeof(float));
  }
  const float* query_ptr = query_block.get();

  const std::size_t n = arena.size();
  const std::size_t keep = k < n ? k : n;

  struct Item {
    float rank;
    float score;
    std::uint32_t id;
  };
  struct WorseOnTop {
    bool operator()(const Item& a, const Item& b) const noexcept {
      if (a.rank < b.rank) {
        return true;
      }
      if (b.rank < a.rank) {
        return false;
      }
      return a.id < b.id;
    }
  };

  // Sequential scan. The hardware stride prefetcher already pulls the next
  // cache lines of a dense arena. __builtin_prefetch earns its place on the
  // HNSW walk, where the next address is data-dependent.
  std::vector<Item> backing;
  backing.reserve(keep);
  std::priority_queue<Item, std::vector<Item>, WorseOnTop> heap{WorseOnTop{}, std::move(backing)};

  for (std::size_t i = 0; i < n; ++i) {
    const float* row = arena.vector(static_cast<std::uint32_t>(i));
    const float score = distance(query_ptr, row, stride, metric, kernel);
    const float rank = metric == Metric::L2Squared ? score : -score;
    const Item item{rank, score, static_cast<std::uint32_t>(i)};
    if (heap.size() < keep) {
      heap.push(item);
    } else if (WorseOnTop{}(item, heap.top())) {
      heap.pop();
      heap.push(item);
    }
  }

  std::vector<Hit> hits(heap.size());
  for (std::size_t i = hits.size(); i-- > 0;) {
    const Item top = heap.top();
    hits[i] = Hit{top.id, top.score};
    heap.pop();
  }
  return hits;
}

}  // namespace bazavec
