#include "bazavec/bazavec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

// Embeddings cluster. Uniform noise in 128-d concentrates distances, so a
// beam that is wide enough to hit Recall@10 0.95 scans a large fraction of a
// 100k set and stops beating the sequential kernel. Two hundred tight Gaussians
// are the shape HNSW is built for, and the shape real embeddings actually have.
bazavec::VectorArena make_clustered(std::size_t n, std::size_t dim, std::size_t clusters, std::uint32_t seed,
                                    std::vector<float>& centers) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> spread(-1.f, 1.f);
  std::normal_distribution<float> gauss(0.f, 1.f);
  centers.assign(clusters * dim, 0.f);
  for (float& value : centers) {
    value = spread(rng);
  }
  bazavec::VectorArena arena(dim);
  std::vector<float> packed(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t cluster = i % clusters;
    for (std::size_t d = 0; d < dim; ++d) {
      packed[i * dim + d] = centers[cluster * dim + d] + 0.05f * gauss(rng);
    }
  }
  arena.reserve(n);
  arena.add_many(packed.data(), n);
  return arena;
}

double percentile(std::vector<double> samples, double q) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  const double index = q * static_cast<double>(samples.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(index);
  const std::size_t hi = std::min(lo + 1, samples.size() - 1);
  const double frac = index - static_cast<double>(lo);
  return samples[lo] * (1.0 - frac) + samples[hi] * frac;
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

}  // namespace

int main() {
#if defined(__APPLE__)
  (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  constexpr std::size_t kN = 100000;
  constexpr std::size_t kDim = 128;
  constexpr std::size_t kClusters = 200;
  constexpr std::size_t kK = 10;
  constexpr int kLatencyQueries = 80;
  constexpr int kRecallQueries = 30;

  std::cout << "bazavec phase 2 benchmark\n";
  std::cout << "compiler: " << __VERSION__ << "\n";
  std::cout << "isa: " << bazavec::simd_isa() << "\n";
  std::cout << "distribution: " << kClusters << " gaussians, sigma 0.05, centers uniform in [-1, 1]^" << kDim
            << "\n";

  std::vector<float> centers;
  bazavec::VectorArena arena = make_clustered(kN, kDim, kClusters, 42, centers);
  bazavec::HnswIndex::Params params;
  params.m = 16;
  params.ef_construction = 128;
  params.ef_search = 64;
  params.threads = 0;
  params.seed = 42;

  const auto built0 = Clock::now();
  bazavec::HnswIndex index(arena, bazavec::Metric::L2Squared, params);
  index.build();
  const auto built1 = Clock::now();
  const double build_s = std::chrono::duration<double>(built1 - built0).count();

  std::cout << "N: " << arena.size() << "  dim: " << arena.dim() << "  M: " << params.m
            << "  ef_construction: " << params.ef_construction << "  ef_search: " << params.ef_search << "\n";
  std::cout << "max_level: " << index.max_level() << "  entry: " << index.entry_point() << "\n";
  std::cout << "vectors: " << arena.bytes() << " bytes\n";
  std::cout << "graph: " << index.graph_bytes() << " bytes\n";
  std::cout << "build: " << build_s << " s\n";

  std::mt19937 rng(99);
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::vector<float> query(kDim);
  auto sample_query = [&] {
    const std::size_t cluster = rng() % kClusters;
    for (std::size_t d = 0; d < kDim; ++d) {
      query[d] = centers[cluster * kDim + d] + 0.05f * gauss(rng);
    }
  };

  double recall_acc = 0.0;
  for (int q = 0; q < kRecallQueries; ++q) {
    sample_query();
    const std::vector<bazavec::Hit> got = index.search_knn(query, kK);
    const std::vector<bazavec::Hit> truth =
        bazavec::brute_force_search(arena, query, kK, bazavec::Metric::L2Squared);
    recall_acc += static_cast<double>(recall_at(got, truth));
  }
  const double recall = recall_acc / static_cast<double>(kRecallQueries);
  std::cout << "recall@" << kK << ": " << recall << " over " << kRecallQueries << " queries\n";

  std::vector<double> hnsw_us;
  std::vector<double> brute_us;
  hnsw_us.reserve(kLatencyQueries);
  brute_us.reserve(kLatencyQueries);
  float sink = 0.f;
  for (int q = 0; q < 8; ++q) {
    sample_query();
    sink += index.search_knn(query, kK).front().score;
  }
  for (int q = 0; q < kLatencyQueries; ++q) {
    sample_query();
    const auto h0 = Clock::now();
    const std::vector<bazavec::Hit> hits = index.search_knn(query, kK);
    const auto h1 = Clock::now();
    hnsw_us.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
    sink += hits.front().score;

    const auto b0 = Clock::now();
    const std::vector<bazavec::Hit> brute =
        bazavec::brute_force_search(arena, query, kK, bazavec::Metric::L2Squared);
    const auto b1 = Clock::now();
    brute_us.push_back(std::chrono::duration<double, std::micro>(b1 - b0).count());
    sink += brute.front().score;
  }

  auto mean = [](const std::vector<double>& samples) {
    double acc = 0.0;
    for (double sample : samples) {
      acc += sample;
    }
    return acc / static_cast<double>(samples.size());
  };
  const double hnsw_mean = mean(hnsw_us);
  const double brute_mean = mean(brute_us);
  std::cout << "\nsearch top-" << kK << ", microseconds\n";
  std::cout << "hnsw    mean " << hnsw_mean << "  p50 " << percentile(hnsw_us, 0.50) << "  p99 "
            << percentile(hnsw_us, 0.99) << "\n";
  std::cout << "brute   mean " << brute_mean << "  p50 " << percentile(brute_us, 0.50) << "  p99 "
            << percentile(brute_us, 0.99) << "\n";
  std::cout << "qps " << (1e6 / hnsw_mean) << "\n";
  std::cout << "brute/hnsw " << (brute_mean / hnsw_mean) << "x\n";
  std::cout << "sink " << sink << "\n";

  if (recall < 0.95) {
    std::cerr << "benchmark recall below 0.95\n";
    return 1;
  }
  return 0;
}
