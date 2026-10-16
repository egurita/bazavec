#pragma once

#include "brute_force.hpp"
#include "detail/memory.hpp"
#include "distance.hpp"
#include "storage.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace bazavec {

// Hierarchical Navigable Small World graph (Malkov & Yashunin, TPAMI 2018).
//
// Layer 0 is the graph every query walks. Node i owns a fixed slot
// [i * max_m0, i * max_m0 + degree[i]) inside one array, so the index does not
// pay a heap allocation per node. Upper layers hold about 1/M of the nodes and
// live in a second packed array.
//
// A search stamps nodes with a generation counter instead of clearing a bitset.
// The walk prefetches the next neighbor's row so the random edge's cache miss
// overlaps the distance currently in flight. build() inserts on a thread pool;
// each node has a spinlock on its own cache line, held only while its neighbor
// list is copied or replaced.
//
// The arena must outlive the index. search_knn() is safe once build() has
// returned. Concurrent searches share one visited table and one query pad, so
// the caller serializes them.
class HnswIndex {
 public:
  static constexpr int kMaxLevel = 16;
  static constexpr std::uint32_t kInvalid = 0xffffffffu;

  struct Params {
    std::uint32_t m = 16;
    std::uint32_t ef_construction = 200;
    std::uint32_t ef_search = 64;
    std::uint32_t threads = 0;  // 0 uses hardware_concurrency
    std::uint64_t seed = 42;
  };

  HnswIndex(const VectorArena& arena, Metric metric, Params params)
      : arena_(&arena), metric_(metric), params_(params) {
    if (params_.m < 2 || params_.m > 48) {
      throw std::invalid_argument("HnswIndex: m must be in [2, 48]");
    }
    if (params_.ef_construction == 0 || params_.ef_search == 0) {
      throw std::invalid_argument("HnswIndex: ef must be positive");
    }
    if (params_.ef_construction < params_.m) {
      params_.ef_construction = params_.m;
    }
    max_m_ = params_.m;
    max_m0_ = params_.m * 2;
    ml_ = 1.0 / std::log(static_cast<double>(params_.m));
    query_block_ = detail::allocate_floats(arena_->stride());
    std::memset(query_block_.get(), 0, arena_->stride() * sizeof(float));
  }

  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  void build() {
    if (built_) {
      throw std::logic_error("HnswIndex::build: already built");
    }
    const std::size_t n = arena_->size();
    if (n > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw std::length_error("HnswIndex::build: id space exhausted");
    }
    if (n == 0) {
      built_ = true;
      return;
    }
    allocate(n);
    {
      std::lock_guard<std::mutex> guard(ep_mu_);
      enter_ = 0;
      max_level_ = static_cast<int>(levels_[0]);
    }
    constructing_.store(true, std::memory_order_release);
    if (n > 1) {
      const unsigned workers = worker_count(n);
      if (workers == 1) {
        Scratch scratch;
        prepare_scratch(scratch, params_.ef_construction);
        for (std::uint32_t id = 1; id < node_count_; ++id) {
          insert(id, scratch);
        }
      } else {
        std::atomic<std::uint32_t> next{1};
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned worker = 0; worker < workers; ++worker) {
          pool.emplace_back([this, &next] {
            Scratch scratch;
            prepare_scratch(scratch, params_.ef_construction);
            for (;;) {
              const std::uint32_t id = next.fetch_add(1, std::memory_order_relaxed);
              if (id >= node_count_) {
                return;
              }
              insert(id, scratch);
            }
          });
        }
        for (std::thread& worker : pool) {
          worker.join();
        }
      }
    }
    constructing_.store(false, std::memory_order_release);
    prepare_scratch(query_scratch_, std::max(params_.ef_construction, params_.ef_search));
    built_ = true;
  }

  // `ef == 0` uses Params::ef_search. The width is raised to at least k.
  // `accept` is applied on layer 0 only: a rejected node is not a hit, but it
  // stays in the beam so the walk can cross it. Upper layers ignore it and
  // descend to the geometric neighborhood first.
  template <class Accept>
  std::vector<Hit> search_knn(std::span<const float> query, std::size_t k, std::size_t ef, Accept accept) const {
    if (!built_) {
      throw std::logic_error("HnswIndex::search_knn: build() was not called");
    }
    if (query.size() != arena_->dim() || query.data() == nullptr) {
      throw std::invalid_argument("HnswIndex::search_knn: query must contain dim floats");
    }
    if (k == 0 || node_count_ == 0) {
      return {};
    }

    std::memcpy(query_block_.get(), query.data(), arena_->dim() * sizeof(float));
    const float* padded = query_block_.get();

    std::size_t width = ef == 0 ? static_cast<std::size_t>(params_.ef_search) : ef;
    if (width < k) {
      width = k;
    }
    prepare_scratch(query_scratch_, static_cast<std::uint32_t>(width));

    std::uint32_t ep = 0;
    int top = 0;
    {
      std::lock_guard<std::mutex> guard(ep_mu_);
      ep = enter_;
      top = max_level_;
    }
    for (int layer = top; layer > 0; --layer) {
      search_layer(padded, ep, 1, layer, query_scratch_, accept_all);
      ep = closest_id(query_scratch_.results, ep);
    }
    search_layer(padded, ep, width, 0, query_scratch_, accept);

    std::vector<Scored> found = query_scratch_.results;
    std::sort(found.begin(), found.end(), closer_score);
    if (found.size() > k) {
      found.resize(k);
    }
    std::vector<Hit> hits;
    hits.reserve(found.size());
    for (const Scored& item : found) {
      hits.push_back(Hit{item.id, score_from_rank(item.rank)});
    }
    return hits;
  }

  std::vector<Hit> search_knn(std::span<const float> query, std::size_t k, std::size_t ef = 0) const {
    return search_knn(query, k, ef, accept_all);
  }

  // Algorithm 4. `query` has dim floats; it is padded internally. Candidate ids
  // must already live in the arena. extend_candidates reads the built graph.
  std::vector<std::uint32_t> select_neighbors_heuristic(std::span<const float> query,
                                                        std::span<const std::uint32_t> candidates,
                                                        std::size_t m, int layer, bool extend_candidates,
                                                        bool keep_pruned_connections) const {
    if (query.size() != arena_->dim() || query.data() == nullptr) {
      throw std::invalid_argument("HnswIndex::select_neighbors_heuristic: query must contain dim floats");
    }
    if (layer < 0 || layer > kMaxLevel) {
      throw std::invalid_argument("HnswIndex::select_neighbors_heuristic: layer out of range");
    }
    detail::AlignedPtr padded = detail::allocate_floats(arena_->stride());
    std::memset(padded.get(), 0, arena_->stride() * sizeof(float));
    std::memcpy(padded.get(), query.data(), arena_->dim() * sizeof(float));

    Scratch scratch;
    scratch.pool.reserve(candidates.size() + 1);
    for (std::uint32_t id : candidates) {
      scratch.pool.push_back(Scored{rank_between(padded.get(), arena_->vector(id)), id});
    }
    select_neighbors_heuristic(padded.get(), m, layer, extend_candidates, keep_pruned_connections, kInvalid,
                               scratch);
    return scratch.chosen;
  }

  const Params& params() const noexcept { return params_; }
  Metric metric() const noexcept { return metric_; }
  bool built() const noexcept { return built_; }
  std::size_t size() const noexcept { return node_count_; }
  std::uint32_t entry_point() const noexcept { return enter_; }
  int max_level() const noexcept { return max_level_; }
  std::uint32_t layer0_stride() const noexcept { return max_m0_; }
  const std::uint32_t* layer0_data() const noexcept { return adj0_.data(); }

  std::uint16_t layer0_degree(std::uint32_t id) const {
    assert(id < node_count_);
    return deg0_[id];
  }

  std::size_t graph_bytes() const noexcept {
    return adj0_.size() * sizeof(std::uint32_t) + deg0_.size() * sizeof(std::uint16_t) +
           upper_edges_.size() * sizeof(std::uint32_t) + upper_deg_.size() * sizeof(std::uint16_t) +
           upper_slot_.size() * sizeof(std::uint32_t) + levels_.size() * sizeof(std::uint8_t) +
           static_cast<std::size_t>(node_count_) * sizeof(SpinLock);
  }

 private:
  // One cache line. Adjacent inserters do not bounce the same flag.
  class alignas(kArenaAlignment) SpinLock {
   public:
    void lock() const noexcept {
      while (flag_.test_and_set(std::memory_order_acquire)) {
        while (flag_.test(std::memory_order_relaxed)) {
#if defined(__aarch64__)
          __asm__ __volatile__("yield");
#elif defined(__x86_64__)
          __asm__ __volatile__("pause");
#else
          std::this_thread::yield();
#endif
        }
      }
    }

    void unlock() const noexcept { flag_.clear(std::memory_order_release); }

   private:
    mutable std::atomic_flag flag_{};
  };

  struct Scored {
    float rank = 0.f;
    std::uint32_t id = 0;
  };

  // Max-heap comparator: largest rank (the worst neighbor) ends on top.
  struct FartherFirst {
    bool operator()(const Scored& a, const Scored& b) const noexcept {
      if (a.rank < b.rank) {
        return true;
      }
      if (b.rank < a.rank) {
        return false;
      }
      return a.id < b.id;
    }
  };

  // Max-heap comparator with the order flipped, so the closest neighbor is on top.
  struct CloserFirst {
    bool operator()(const Scored& a, const Scored& b) const noexcept {
      if (a.rank > b.rank) {
        return true;
      }
      if (b.rank > a.rank) {
        return false;
      }
      return a.id > b.id;
    }
  };

  struct VisitedTable {
    std::vector<std::uint16_t> stamp;
    std::uint16_t gen = 0;

    void next() {
      ++gen;
      if (gen == 0) {
        std::fill(stamp.begin(), stamp.end(), 0);
        gen = 1;
      }
    }

    bool try_visit(std::uint32_t id) {
      if (stamp[id] == gen) {
        return false;
      }
      stamp[id] = gen;
      return true;
    }
  };

  struct Scratch {
    VisitedTable visited;
    std::vector<Scored> candidates;
    std::vector<Scored> results;
    std::vector<Scored> pool;
    std::vector<std::uint32_t> chosen;
    std::vector<std::uint32_t> discarded;
    std::uint32_t neigh[128]{};
  };

  static bool closer_score(const Scored& a, const Scored& b) noexcept {
    if (a.rank < b.rank) {
      return true;
    }
    if (b.rank < a.rank) {
      return false;
    }
    return a.id < b.id;
  }

  static void push_closer(std::vector<Scored>& heap, Scored item) {
    heap.push_back(item);
    std::push_heap(heap.begin(), heap.end(), CloserFirst{});
  }

  static void push_farther(std::vector<Scored>& heap, Scored item) {
    heap.push_back(item);
    std::push_heap(heap.begin(), heap.end(), FartherFirst{});
  }

  static Scored pop_closer(std::vector<Scored>& heap) {
    std::pop_heap(heap.begin(), heap.end(), CloserFirst{});
    const Scored out = heap.back();
    heap.pop_back();
    return out;
  }

  static Scored pop_farther(std::vector<Scored>& heap) {
    std::pop_heap(heap.begin(), heap.end(), FartherFirst{});
    const Scored out = heap.back();
    heap.pop_back();
    return out;
  }

  static std::uint32_t closest_id(const std::vector<Scored>& heap, std::uint32_t fallback) noexcept {
    if (heap.empty()) {
      return fallback;
    }
    Scored best = heap.front();
    for (const Scored& item : heap) {
      if (closer_score(item, best)) {
        best = item;
      }
    }
    return best.id;
  }

  float rank_between(const float* a, const float* b) const noexcept {
    const float score = distance(a, b, arena_->stride(), metric_, DistanceKernel::Simd);
    return metric_ == Metric::L2Squared ? score : -score;
  }

  float score_from_rank(float rank) const noexcept {
    return metric_ == Metric::L2Squared ? rank : -rank;
  }

  unsigned worker_count(std::size_t n) const noexcept {
    unsigned workers = params_.threads;
    if (workers == 0) {
      const unsigned hw = std::thread::hardware_concurrency();
      workers = hw == 0 ? 1u : hw;
    }
    if (n <= 1) {
      return 1;
    }
    const unsigned limit = static_cast<unsigned>(n - 1);
    return workers < limit ? workers : limit;
  }

  void prepare_scratch(Scratch& scratch, std::uint32_t ef) const {
    if (scratch.visited.stamp.size() != node_count_) {
      scratch.visited.stamp.assign(node_count_, 0);
      scratch.visited.gen = 0;
    }
    const std::size_t width = static_cast<std::size_t>(ef);
    const std::size_t pool_need = width * static_cast<std::size_t>(max_m0_) + 8;
    if (scratch.candidates.capacity() < width * 2) {
      scratch.candidates.reserve(width * 2);
    }
    if (scratch.results.capacity() < width + 1) {
      scratch.results.reserve(width + 1);
    }
    if (scratch.pool.capacity() < pool_need) {
      scratch.pool.reserve(pool_need);
    }
    if (scratch.chosen.capacity() < 128) {
      scratch.chosen.reserve(128);
    }
    if (scratch.discarded.capacity() < pool_need) {
      scratch.discarded.reserve(pool_need);
    }
  }

  void allocate(std::size_t n) {
    node_count_ = static_cast<std::uint32_t>(n);
    levels_.assign(n, 0);
    std::mt19937_64 rng(params_.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::size_t upper_nodes = 0;
    for (std::size_t i = 0; i < n; ++i) {
      double draw = unit(rng);
      if (draw < 1e-12) {
        draw = 1e-12;
      }
      int level = static_cast<int>(-std::log(draw) * ml_);
      if (level < 0) {
        level = 0;
      }
      if (level > kMaxLevel) {
        level = kMaxLevel;
      }
      levels_[i] = static_cast<std::uint8_t>(level);
      if (level > 0) {
        ++upper_nodes;
      }
    }

    adj0_.assign(n * static_cast<std::size_t>(max_m0_), 0);
    deg0_.assign(n, 0);
    upper_slot_.assign(n, kInvalid);
    upper_edges_.assign(upper_nodes * static_cast<std::size_t>(kMaxLevel) * max_m_, 0);
    upper_deg_.assign(upper_nodes * static_cast<std::size_t>(kMaxLevel), 0);
    std::uint32_t slot = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (levels_[i] > 0) {
        upper_slot_[i] = slot;
        ++slot;
      }
    }
    locks_.reset(new SpinLock[n]);
  }

  void prefetch_row(std::uint32_t id) const noexcept {
    const char* bytes = reinterpret_cast<const char*>(arena_->vector(id));
    const std::size_t size = arena_->stride() * sizeof(float);
    __builtin_prefetch(bytes, 0, 3);
    if (size > 128) {
      __builtin_prefetch(bytes + 128, 0, 3);
    }
    if (size > 256) {
      __builtin_prefetch(bytes + 256, 0, 3);
    }
    if (size > 384) {
      __builtin_prefetch(bytes + 384, 0, 3);
    }
  }

  void load_neighbors_raw(std::uint32_t id, int layer, std::uint32_t* out, std::uint16_t& degree) const {
    degree = 0;
    if (layer == 0) {
      degree = deg0_[id];
      const std::size_t base = static_cast<std::size_t>(id) * max_m0_;
      for (std::uint16_t i = 0; i < degree; ++i) {
        out[i] = adj0_[base + i];
      }
      return;
    }
    const std::uint32_t slot = upper_slot_[id];
    if (slot == kInvalid || layer > static_cast<int>(levels_[id])) {
      return;
    }
    const std::size_t block = static_cast<std::size_t>(slot) * static_cast<std::size_t>(kMaxLevel) +
                              static_cast<std::size_t>(layer - 1);
    degree = upper_deg_[block];
    const std::size_t base = block * max_m_;
    for (std::uint16_t i = 0; i < degree; ++i) {
      out[i] = upper_edges_[base + i];
    }
  }

  void store_neighbors_raw(std::uint32_t id, int layer, const std::uint32_t* ids, std::uint16_t degree) {
    if (layer == 0) {
      const std::size_t base = static_cast<std::size_t>(id) * max_m0_;
      for (std::uint16_t i = 0; i < degree; ++i) {
        adj0_[base + i] = ids[i];
      }
      deg0_[id] = degree;
      return;
    }
    const std::uint32_t slot = upper_slot_[id];
    assert(slot != kInvalid);
    const std::size_t block = static_cast<std::size_t>(slot) * static_cast<std::size_t>(kMaxLevel) +
                              static_cast<std::size_t>(layer - 1);
    const std::size_t base = block * max_m_;
    for (std::uint16_t i = 0; i < degree; ++i) {
      upper_edges_[base + i] = ids[i];
    }
    upper_deg_[block] = degree;
  }

  void load_neighbors(std::uint32_t id, int layer, std::uint32_t* out, std::uint16_t& degree) const {
    const bool guard = constructing_.load(std::memory_order_acquire);
    if (guard) {
      locks_[id].lock();
    }
    load_neighbors_raw(id, layer, out, degree);
    if (guard) {
      locks_[id].unlock();
    }
  }

  void store_neighbors(std::uint32_t id, int layer, const std::uint32_t* ids, std::uint16_t degree) {
    locks_[id].lock();
    store_neighbors_raw(id, layer, ids, degree);
    locks_[id].unlock();
  }

  // Keep a neighbor when it is closer to the query than to anything already kept.
  // That is the relative-neighborhood test from the paper: it refuses edges that
  // all land in the same direction. keep_pruned_connections fills any leftover
  // slots with the rejected (closer, redundant) nodes so a sparse region still
  // reaches degree M.
  void select_neighbors_heuristic(const float* padded_query, std::size_t m, int layer, bool extend_candidates,
                                  bool keep_pruned_connections, std::uint32_t skip_id, Scratch& scratch) const {
    scratch.chosen.clear();
    if (m == 0 || scratch.pool.empty()) {
      scratch.discarded.clear();
      return;
    }
    if (extend_candidates) {
      if (!built_ && constructing_.load(std::memory_order_acquire) == false) {
        throw std::logic_error("HnswIndex::select_neighbors_heuristic: extend requires build()");
      }
      scratch.discarded.clear();
      scratch.discarded.reserve(scratch.pool.size());
      for (const Scored& item : scratch.pool) {
        scratch.discarded.push_back(item.id);
      }
      const std::size_t seeds = scratch.discarded.size();
      for (std::size_t seed = 0; seed < seeds; ++seed) {
        std::uint16_t degree = 0;
        load_neighbors(scratch.discarded[seed], layer, scratch.neigh, degree);
        for (std::uint16_t n = 0; n < degree; ++n) {
          const std::uint32_t other = scratch.neigh[n];
          if (other == skip_id) {
            continue;
          }
          bool seen = false;
          for (const Scored& item : scratch.pool) {
            if (item.id == other) {
              seen = true;
              break;
            }
          }
          if (seen) {
            continue;
          }
          scratch.pool.push_back(Scored{rank_between(padded_query, arena_->vector(other)), other});
        }
      }
    }

    std::sort(scratch.pool.begin(), scratch.pool.end(), closer_score);
    std::size_t unique = 0;
    for (std::size_t read = 0; read < scratch.pool.size(); ++read) {
      const Scored item = scratch.pool[read];
      if (item.id == skip_id) {
        continue;
      }
      bool dup = false;
      for (std::size_t prev = 0; prev < unique; ++prev) {
        if (scratch.pool[prev].id == item.id) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        scratch.pool[unique] = item;
        ++unique;
      }
    }
    scratch.pool.resize(unique);

    scratch.discarded.clear();
    for (const Scored& cand : scratch.pool) {
      if (scratch.chosen.size() >= m) {
        break;
      }
      const float* cand_vec = arena_->vector(cand.id);
      bool diverse = true;
      for (std::uint32_t kept : scratch.chosen) {
        if (rank_between(cand_vec, arena_->vector(kept)) < cand.rank) {
          diverse = false;
          break;
        }
      }
      if (diverse) {
        scratch.chosen.push_back(cand.id);
      } else {
        scratch.discarded.push_back(cand.id);
      }
    }
    if (keep_pruned_connections) {
      for (std::uint32_t id : scratch.discarded) {
        if (scratch.chosen.size() >= m) {
          break;
        }
        scratch.chosen.push_back(id);
      }
    }
  }

  static constexpr auto accept_all = [](std::uint32_t) noexcept { return true; };

  // Algorithm 2. `rank` is smaller-is-better for every metric. The candidate
  // heap yields the closest node; the result heap keeps the ef closest hits, worst
  // on top. Rejected ids are still expanded when they fall inside that beam.
  template <class Accept>
  void search_layer(const float* query, std::uint32_t ep, std::size_t ef, int layer, Scratch& scratch,
                    Accept accept) const {
    scratch.candidates.clear();
    scratch.results.clear();
    if (ef == 0) {
      return;
    }
    scratch.visited.next();
    scratch.visited.try_visit(ep);
    const Scored first{rank_between(query, arena_->vector(ep)), ep};
    push_closer(scratch.candidates, first);
    if (accept(ep)) {
      push_farther(scratch.results, first);
    }

    while (!scratch.candidates.empty()) {
      if (!scratch.results.empty() && scratch.results.size() >= ef &&
          scratch.candidates.front().rank > scratch.results.front().rank) {
        break;
      }
      const Scored current = pop_closer(scratch.candidates);
      std::uint16_t degree = 0;
      load_neighbors(current.id, layer, scratch.neigh, degree);
      if (degree > 0) {
        prefetch_row(scratch.neigh[0]);
      }
      for (std::uint16_t i = 0; i < degree; ++i) {
        const std::uint32_t candidate = scratch.neigh[i];
        // The next row is a random 128-byte-aligned access. Issuing the prefetch
        // before the current distance overlaps that L2 miss with arithmetic.
        if (i + 1 < degree) {
          prefetch_row(scratch.neigh[i + 1]);
        }
        if (!scratch.visited.try_visit(candidate)) {
          continue;
        }
        const float rank = rank_between(query, arena_->vector(candidate));
        const bool room = scratch.results.empty() || scratch.results.size() < ef ||
                          rank < scratch.results.front().rank;
        if (!room) {
          continue;
        }
        push_closer(scratch.candidates, Scored{rank, candidate});
        if (accept(candidate)) {
          push_farther(scratch.results, Scored{rank, candidate});
          if (scratch.results.size() > ef) {
            pop_farther(scratch.results);
          }
        }
      }
    }
  }

  void link_reverse(std::uint32_t node, int layer, std::uint32_t newbie, Scratch& scratch) {
    if (node == newbie) {
      return;
    }
    const std::uint32_t cap = layer == 0 ? max_m0_ : max_m_;
    locks_[node].lock();
    std::uint16_t degree = 0;
    std::uint32_t ids[128];
    load_neighbors_raw(node, layer, ids, degree);
    for (std::uint16_t i = 0; i < degree; ++i) {
      if (ids[i] == newbie) {
        locks_[node].unlock();
        return;
      }
    }
    if (static_cast<std::uint32_t>(degree) < cap) {
      ids[degree] = newbie;
      ++degree;
      store_neighbors_raw(node, layer, ids, degree);
      locks_[node].unlock();
      return;
    }

    // The list is at most 2M entries. Pruning it under the lock is a few dozen
    // distances, and it avoids a compare-and-swap retry on the published list.
    const float* origin = arena_->vector(node);
    scratch.pool.clear();
    for (std::uint16_t i = 0; i < degree; ++i) {
      scratch.pool.push_back(Scored{rank_between(origin, arena_->vector(ids[i])), ids[i]});
    }
    scratch.pool.push_back(Scored{rank_between(origin, arena_->vector(newbie)), newbie});
    select_neighbors_heuristic(origin, cap, layer, false, true, node, scratch);
    const auto kept = static_cast<std::uint16_t>(scratch.chosen.size());
    for (std::uint16_t i = 0; i < kept; ++i) {
      ids[i] = scratch.chosen[i];
    }
    store_neighbors_raw(node, layer, ids, kept);
    locks_[node].unlock();
  }

  void insert(std::uint32_t id, Scratch& scratch) {
    const float* query = arena_->vector(id);
    const int level = static_cast<int>(levels_[id]);
    std::uint32_t ep = 0;
    int top = 0;
    {
      std::lock_guard<std::mutex> guard(ep_mu_);
      ep = enter_;
      top = max_level_;
    }

    for (int layer = top; layer > level; --layer) {
      search_layer(query, ep, 1, layer, scratch, accept_all);
      ep = closest_id(scratch.results, ep);
    }

    const int linked_top = level < top ? level : top;
    for (int layer = linked_top; layer >= 0; --layer) {
      search_layer(query, ep, params_.ef_construction, layer, scratch, accept_all);
      scratch.pool.clear();
      scratch.pool.insert(scratch.pool.end(), scratch.results.begin(), scratch.results.end());
      select_neighbors_heuristic(query, max_m_, layer, false, true, id, scratch);

      std::uint32_t linked[128];
      const auto linked_n = static_cast<std::uint16_t>(scratch.chosen.size());
      for (std::uint16_t i = 0; i < linked_n; ++i) {
        linked[i] = scratch.chosen[i];
      }
      store_neighbors(id, layer, linked, linked_n);
      for (std::uint16_t i = 0; i < linked_n; ++i) {
        link_reverse(linked[i], layer, id, scratch);
      }
      ep = closest_id(scratch.results, ep);
    }

    if (level > top) {
      std::lock_guard<std::mutex> guard(ep_mu_);
      if (level > max_level_) {
        max_level_ = level;
        enter_ = id;
      }
    }
  }

  const VectorArena* arena_;
  Metric metric_;
  Params params_;
  std::uint32_t max_m_ = 0;
  std::uint32_t max_m0_ = 0;
  double ml_ = 1.0;
  std::uint32_t node_count_ = 0;
  std::uint32_t enter_ = 0;
  int max_level_ = -1;
  bool built_ = false;
  std::atomic<bool> constructing_{false};
  mutable std::mutex ep_mu_;
  std::vector<std::uint8_t> levels_;
  std::vector<std::uint32_t> adj0_;
  std::vector<std::uint16_t> deg0_;
  std::vector<std::uint32_t> upper_slot_;
  std::vector<std::uint32_t> upper_edges_;
  std::vector<std::uint16_t> upper_deg_;
  std::unique_ptr<SpinLock[]> locks_;
  mutable detail::AlignedPtr query_block_;
  mutable Scratch query_scratch_;
};

}  // namespace bazavec
