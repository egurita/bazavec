#include "bazavec/bazavec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <sys/resource.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

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

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double percentile(std::vector<double> samples, double q) {
  std::sort(samples.begin(), samples.end());
  const double index = q * static_cast<double>(samples.size() - 1);
  const auto lo = static_cast<std::size_t>(index);
  const auto hi = std::min(lo + 1, samples.size() - 1);
  const double frac = index - static_cast<double>(lo);
  return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}

double mean_of(const std::vector<double>& samples) {
  double acc = 0.0;
  for (double sample : samples) {
    acc += sample;
  }
  return acc / static_cast<double>(samples.size());
}

float recall_at(const std::vector<bazavec::Hit>& got, const std::vector<bazavec::Hit>& truth) {
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

std::vector<bazavec::Hit> brute_int8(const bazavec::ScalarQuantizer& quant, const std::int8_t* query, std::size_t k,
                                     bool simd) {
  struct Item {
    float score;
    std::uint32_t id;
  };
  struct WorseOnTop {
    bool operator()(const Item& a, const Item& b) const noexcept {
      if (a.score < b.score) {
        return true;
      }
      if (b.score < a.score) {
        return false;
      }
      return a.id < b.id;
    }
  };
  const std::size_t n = quant.size();
  const std::size_t keep = k < n ? k : n;
  std::vector<Item> kept;
  kept.reserve(keep);
  for (std::uint32_t id = 0; id < n; ++id) {
    const std::int32_t raw = simd ? bazavec::l2_squared_i8_simd(query, quant.row(id), quant.stride())
                                  : bazavec::l2_squared_i8_scalar(query, quant.row(id), quant.stride());
    const Item item{static_cast<float>(raw), id};
    if (kept.size() < keep) {
      kept.push_back(item);
      std::push_heap(kept.begin(), kept.end(), WorseOnTop{});
      continue;
    }
    if (WorseOnTop{}(item, kept.front())) {
      std::pop_heap(kept.begin(), kept.end(), WorseOnTop{});
      kept.back() = item;
      std::push_heap(kept.begin(), kept.end(), WorseOnTop{});
    }
  }
  std::sort_heap(kept.begin(), kept.end(), WorseOnTop{});
  std::vector<bazavec::Hit> hits;
  hits.reserve(kept.size());
  for (const Item& item : kept) {
    hits.push_back(bazavec::Hit{item.id, item.score});
  }
  return hits;
}

long rss_bytes() {
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
}

struct Latency {
  double mean = 0;
  double p50 = 0;
  double p99 = 0;
  double qps = 0;
};

Latency summarize(const std::vector<double>& samples) {
  Latency out;
  out.mean = mean_of(samples);
  out.p50 = percentile(samples, 0.50);
  out.p99 = percentile(samples, 0.99);
  out.qps = out.mean > 0.0 ? 1e6 / out.mean : 0.0;
  return out;
}

void print_row(const char* name, double recall, double build_s, const Latency& latency) {
  std::cout << "| " << std::left << std::setw(22) << name << " | " << std::right << std::setw(7) << std::fixed
            << std::setprecision(3) << recall << " | " << std::setw(8) << std::setprecision(3) << build_s << " | "
            << std::setw(8) << std::setprecision(1) << latency.mean << " | " << std::setw(8) << latency.p50 << " | "
            << std::setw(8) << latency.p99 << " | " << std::setw(8) << std::setprecision(0) << latency.qps << " |\n";
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
  constexpr int kRecallQueries = 30;
  constexpr int kLatencyQueries = 80;

  std::cout << "bazavec phase 5 benchmark\n";
  std::cout << "compiler: " << __VERSION__ << "\n";
  std::cout << "isa: " << bazavec::simd_isa() << "\n";
  std::cout << "distribution: " << kClusters << " gaussians, sigma 0.05, centers uniform in [-1, 1]^" << kDim
            << "\n";
  std::cout << "N: " << kN << "  dim: " << kDim << "  M: 16  ef_construction: 128  ef_search: 64\n";

  std::vector<float> centers;
  bazavec::VectorArena arena = make_clustered(kN, kDim, kClusters, 42, centers);
  std::mt19937 rng(99);
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::vector<std::vector<float>> queries(kLatencyQueries, std::vector<float>(kDim));
  for (std::vector<float>& query : queries) {
    const std::size_t cluster = rng() % kClusters;
    for (std::size_t d = 0; d < kDim; ++d) {
      query[d] = centers[cluster * kDim + d] + 0.05f * gauss(rng);
    }
  }

  std::vector<float> padded(arena.stride(), 0.f);
  std::copy(queries[0].begin(), queries[0].end(), padded.begin());
  volatile float sink = 0.f;
  auto time_float_scan = [&](bazavec::DistanceKernel kernel, int passes) {
    for (int pass = 0; pass < 2; ++pass) {
      for (std::uint32_t id = 0; id < kN; ++id) {
        sink += bazavec::l2_squared(padded.data(), arena.vector(id), arena.stride(), kernel);
      }
    }
    const auto start = Clock::now();
    for (int pass = 0; pass < passes; ++pass) {
      for (std::uint32_t id = 0; id < kN; ++id) {
        sink += bazavec::l2_squared(padded.data(), arena.vector(id), arena.stride(), kernel);
      }
    }
    return seconds_since(start) / static_cast<double>(passes);
  };
  const double scalar_scan = time_float_scan(bazavec::DistanceKernel::Scalar, 4);
  const double neon_scan = time_float_scan(bazavec::DistanceKernel::Simd, 20);
  const double bytes = static_cast<double>(kN) * static_cast<double>(kDim) * static_cast<double>(sizeof(float));

  bazavec::HnswIndex::Params params;
  params.m = 16;
  params.ef_construction = 128;
  params.ef_search = 64;
  params.threads = 0;
  params.seed = 42;

  const auto build0 = Clock::now();
  bazavec::HnswIndex fp32(arena, bazavec::Metric::L2Squared, params);
  fp32.build();
  const double fp32_build = seconds_since(build0);

  std::vector<std::vector<bazavec::Hit>> truth(kRecallQueries);
  for (int q = 0; q < kRecallQueries; ++q) {
    truth[static_cast<std::size_t>(q)] =
        bazavec::brute_force_search(arena, queries[static_cast<std::size_t>(q)], kK, bazavec::Metric::L2Squared);
  }
  double fp32_recall = 0.0;
  for (int q = 0; q < kRecallQueries; ++q) {
    const std::vector<bazavec::Hit> got = fp32.search_knn(queries[static_cast<std::size_t>(q)], kK);
    fp32_recall += static_cast<double>(recall_at(got, truth[static_cast<std::size_t>(q)]));
    sink += got.front().score;
  }
  fp32_recall /= kRecallQueries;

  auto time_search = [&](const auto& search) {
    std::vector<double> samples;
    samples.reserve(kLatencyQueries);
    for (int q = 0; q < 8; ++q) {
      sink += search(queries[static_cast<std::size_t>(q)]).front().score;
    }
    for (int q = 0; q < kLatencyQueries; ++q) {
      const auto start = Clock::now();
      const std::vector<bazavec::Hit> hits = search(queries[static_cast<std::size_t>(q)]);
      samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
      sink += hits.front().score;
    }
    return summarize(samples);
  };

  const Latency brute_scalar = time_search([&](const std::vector<float>& query) {
    return bazavec::brute_force_search(arena, query, kK, bazavec::Metric::L2Squared, bazavec::DistanceKernel::Scalar);
  });
  const Latency brute_neon = time_search([&](const std::vector<float>& query) {
    return bazavec::brute_force_search(arena, query, kK, bazavec::Metric::L2Squared, bazavec::DistanceKernel::Simd);
  });
  const Latency hnsw_fp32 = time_search([&](const std::vector<float>& query) { return fp32.search_knn(query, kK); });

  const auto quant0 = Clock::now();
  bazavec::ScalarQuantizer quant;
  quant.train(arena);
  quant.encode(arena);
  const double quant_s = seconds_since(quant0);
  const auto ibuild0 = Clock::now();
  bazavec::HnswIndex i8(arena, bazavec::Metric::L2Squared, params);
  i8.bind_quantizer(quant);
  i8.build();
  const double i8_build = seconds_since(ibuild0);

  double i8_vs_float = 0.0;
  double i8_vs_codes = 0.0;
  double i8_ceiling = 0.0;
  for (int q = 0; q < kRecallQueries; ++q) {
    const std::vector<float>& query = queries[static_cast<std::size_t>(q)];
    const std::vector<bazavec::Hit> got = i8.search_knn(query, kK);
    i8_vs_float += static_cast<double>(recall_at(got, truth[static_cast<std::size_t>(q)]));
    const std::vector<std::int8_t> encoded = quant.encode_query(query);
    const std::vector<bazavec::Hit> exact = brute_int8(quant, encoded.data(), kK, true);
    i8_vs_codes += static_cast<double>(recall_at(got, exact));
    i8_ceiling += static_cast<double>(recall_at(exact, truth[static_cast<std::size_t>(q)]));
    sink += got.front().score;
  }
  i8_vs_float /= kRecallQueries;
  i8_vs_codes /= kRecallQueries;
  i8_ceiling /= kRecallQueries;

  std::vector<std::int8_t> encoded0 = quant.encode_query(queries[0]);
  auto time_i8_scan = [&](bool simd, int passes) {
    for (int pass = 0; pass < 2; ++pass) {
      for (std::uint32_t id = 0; id < kN; ++id) {
        sink += simd ? static_cast<float>(bazavec::l2_squared_i8_simd(encoded0.data(), quant.row(id), quant.stride()))
                     : static_cast<float>(bazavec::l2_squared_i8_scalar(encoded0.data(), quant.row(id), quant.stride()));
      }
    }
    const auto start = Clock::now();
    for (int pass = 0; pass < passes; ++pass) {
      for (std::uint32_t id = 0; id < kN; ++id) {
        sink += simd ? static_cast<float>(bazavec::l2_squared_i8_simd(encoded0.data(), quant.row(id), quant.stride()))
                     : static_cast<float>(bazavec::l2_squared_i8_scalar(encoded0.data(), quant.row(id), quant.stride()));
      }
    }
    return seconds_since(start) / static_cast<double>(passes);
  };
  const double i8_scalar_scan = time_i8_scan(false, 8);
  const double i8_neon_scan = time_i8_scan(true, 20);
  const double i8_bytes = static_cast<double>(kN) * static_cast<double>(kDim);

  const Latency brute_i8_scalar = time_search([&](const std::vector<float>& query) {
    const std::vector<std::int8_t> encoded = quant.encode_query(query);
    return brute_int8(quant, encoded.data(), kK, false);
  });
  const Latency brute_i8_neon = time_search([&](const std::vector<float>& query) {
    const std::vector<std::int8_t> encoded = quant.encode_query(query);
    return brute_int8(quant, encoded.data(), kK, true);
  });
  const Latency hnsw_i8 = time_search([&](const std::vector<float>& query) { return i8.search_knn(query, kK); });

  const std::string path = "/tmp/bazavec-bench-" + std::to_string(::getpid()) + ".bzv";
  bazavec::save_bzv(path, arena, fp32);
  const auto map0 = Clock::now();
  bazavec::MappedIndex mapped(path);
  const double map_s = seconds_since(map0);
  const Latency hnsw_mapped =
      time_search([&](const std::vector<float>& query) { return mapped.search_knn(query, kK); });
  const std::vector<bazavec::Hit> live = fp32.search_knn(queries[0], kK);
  const std::vector<bazavec::Hit> restored = mapped.search_knn(queries[0], kK);
  bool same = live.size() == restored.size();
  for (std::size_t i = 0; same && i < live.size(); ++i) {
    same = live[i].id == restored[i].id && live[i].score == restored[i].score;
  }
  ::unlink(path.c_str());

  const double gb = 1e9;
  std::cout << "\nkernel scan, one query over " << kN << " rows\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "fp32 scalar  ns/vec " << (scalar_scan * 1e9 / static_cast<double>(kN)) << "  GB/s "
            << (bytes / scalar_scan / gb) << "\n";
  std::cout << "fp32 neon    ns/vec " << (neon_scan * 1e9 / static_cast<double>(kN)) << "  GB/s "
            << (bytes / neon_scan / gb) << "\n";
  std::cout << "int8 scalar  ns/vec " << (i8_scalar_scan * 1e9 / static_cast<double>(kN)) << "  GB/s "
            << (i8_bytes / i8_scalar_scan / gb) << "\n";
  std::cout << "int8 neon    ns/vec " << (i8_neon_scan * 1e9 / static_cast<double>(kN)) << "  GB/s "
            << (i8_bytes / i8_neon_scan / gb) << "\n";

  std::cout << "\nfp32 build " << std::setprecision(3) << fp32_build << " s\n";
  std::cout << "int8 train+encode " << quant_s << " s\n";
  std::cout << "int8 hnsw build " << i8_build << " s\n";
  std::cout << "int8 hnsw recall@10 vs code-space exact " << i8_vs_codes << "\n";
  std::cout << "mmap+crc " << (map_s * 1e3) << " ms  file " << mapped.mapping_bytes() << " bytes\n";
  std::cout << "mapped ids match " << (same ? "yes" : "no") << "\n";
  std::cout << "vectors " << arena.bytes() << "  graph " << fp32.graph_bytes() << "  codes " << quant.bytes()
            << "  rss " << rss_bytes() << "\n";
  std::cout << "max_level " << fp32.max_level() << " entry " << fp32.entry_point() << "\n";
  std::cout << "sink " << sink << "\n";

  std::cout << "\n| path                   | recall | setup s | mean us | p50 us | p99 us | qps |\n";
  std::cout << "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  print_row("brute fp32 scalar", 1.0, 0.0, brute_scalar);
  print_row("brute fp32 neon", 1.0, 0.0, brute_neon);
  print_row("hnsw fp32", fp32_recall, fp32_build, hnsw_fp32);
  print_row("hnsw fp32 mmap", fp32_recall, map_s, hnsw_mapped);
  print_row("brute int8 scalar", i8_ceiling, 0.0, brute_i8_scalar);
  print_row("brute int8 neon", i8_ceiling, 0.0, brute_i8_neon);
  print_row("hnsw int8", i8_vs_float, i8_build, hnsw_i8);

  if (fp32_recall < 0.95 || !same) {
    std::cerr << "benchmark failed recall or mmap round trip\n";
    return 1;
  }
  return 0;
}
