#include "bazavec/bazavec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

template <class T>
void do_not_optimize(T& value) noexcept {
  asm volatile("" : "+m"(value) : : "memory");
}

// Timed entry points live outside the anonymous namespace so extern "C" is legal.
// The scalar scans disable vectorization on the outer loop as well as inside
// the kernel: otherwise clang can SIMD the scan across rows and the baseline
// stops being scalar.
extern "C" __attribute__((noinline)) float bazavec_scan_l2_scalar(const float* base,
                                                                    const float* query,
                                                                    std::size_t n,
                                                                    std::size_t stride) {
  float acc = 0.f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    acc += bazavec::l2_squared_scalar(query, base + i * stride, stride);
  }
  return acc;
}

extern "C" __attribute__((noinline)) float bazavec_scan_ip_scalar(const float* base,
                                                                    const float* query,
                                                                    std::size_t n,
                                                                    std::size_t stride) {
  float acc = 0.f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    acc += bazavec::inner_product_scalar(query, base + i * stride, stride);
  }
  return acc;
}

extern "C" __attribute__((noinline)) float bazavec_scan_l2_autovec(const float* base,
                                                                     const float* query,
                                                                     std::size_t n,
                                                                     std::size_t stride) {
  float acc = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = base + i * stride;
    float partial = 0.f;
    for (std::size_t d = 0; d < stride; ++d) {
      const float diff = query[d] - row[d];
      partial += diff * diff;
    }
    acc += partial;
  }
  return acc;
}

extern "C" __attribute__((noinline)) float bazavec_scan_ip_autovec(const float* base,
                                                                     const float* query,
                                                                     std::size_t n,
                                                                     std::size_t stride) {
  float acc = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = base + i * stride;
    float partial = 0.f;
    for (std::size_t d = 0; d < stride; ++d) {
      partial += query[d] * row[d];
    }
    acc += partial;
  }
  return acc;
}

extern "C" __attribute__((noinline)) float bazavec_scan_l2_simd(const float* base, const float* query,
                                                                  std::size_t n, std::size_t stride) {
  float acc = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    acc += bazavec::l2_squared_simd(query, base + i * stride, stride);
  }
  return acc;
}

extern "C" __attribute__((noinline)) float bazavec_scan_ip_simd(const float* base, const float* query,
                                                                  std::size_t n, std::size_t stride) {
  float acc = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    acc += bazavec::inner_product_simd(query, base + i * stride, stride);
  }
  return acc;
}

namespace {

using Clock = std::chrono::steady_clock;

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

struct Timed {
  double seconds = 0;
  int reps = 0;
  float checksum = 0.f;
};

template <class Fn>
Timed time_scan(Fn&& fn, double target_seconds) {
  float warm = fn();
  do_not_optimize(warm);

  auto t0 = Clock::now();
  float pilot = fn();
  auto t1 = Clock::now();
  do_not_optimize(pilot);
  const double pilot_s = std::chrono::duration<double>(t1 - t0).count();

  int reps = 1;
  if (pilot_s > 0.0 && pilot_s < target_seconds) {
    reps = static_cast<int>(target_seconds / pilot_s);
    if (reps < 1) {
      reps = 1;
    }
    if (reps > 200) {
      reps = 200;
    }
  }

  float sink = 0.f;
  t0 = Clock::now();
  for (int rep = 0; rep < reps; ++rep) {
    sink += fn();
    do_not_optimize(sink);
  }
  t1 = Clock::now();
  const double total = std::chrono::duration<double>(t1 - t0).count();
  return Timed{total / static_cast<double>(reps), reps, sink};
}

void print_row(const std::string& metric, const std::string& kernel, double ns_per_vec, double gbps) {
  std::cout << std::left << std::setw(16) << metric << std::setw(16) << kernel << std::right
            << std::fixed << std::setprecision(2) << std::setw(12) << ns_per_vec << std::setw(12)
            << gbps << "\n";
}

float max_abs_l2(const float* base, const float* query, std::size_t n, std::size_t stride) {
  float worst = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = base + i * stride;
    const float delta = std::fabs(bazavec::l2_squared_scalar(query, row, stride) -
                                   bazavec::l2_squared_simd(query, row, stride));
    worst = std::max(worst, delta);
  }
  return worst;
}

float max_abs_ip(const float* base, const float* query, std::size_t n, std::size_t stride) {
  float worst = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = base + i * stride;
    const float delta = std::fabs(bazavec::inner_product_scalar(query, row, stride) -
                                   bazavec::inner_product_simd(query, row, stride));
    worst = std::max(worst, delta);
  }
  return worst;
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

}  // namespace

int main() {
#if defined(__APPLE__)
  (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  constexpr std::size_t kN = 100000;
  constexpr std::size_t kDim = 128;
  constexpr std::size_t kQueries = 4;
  constexpr double kTargetSeconds = 0.35;

  std::cout << "bazavec phase 1 benchmark\n";
  std::cout << "compiler: " << __VERSION__ << "\n";
  std::cout << "isa: " << bazavec::simd_isa() << "\n";

  bazavec::VectorArena arena = make_arena(kN, kDim, 42);
  bazavec::VectorArena queries = make_arena(kQueries, kDim, 99);
  const float* base = arena.data();
  const std::size_t stride = arena.stride();
  const double bytes_per_scan =
      static_cast<double>(arena.size()) * static_cast<double>(stride) * sizeof(float);

  std::cout << "N: " << arena.size() << "  dim: " << arena.dim() << "  stride: " << stride << "\n";
  std::cout << "footprint: " << arena.bytes() << " bytes (" << std::fixed << std::setprecision(2)
            << static_cast<double>(arena.bytes()) / (1024.0 * 1024.0) << " MiB)\n";
  std::cout << "allocated: " << arena.allocated_bytes() << " bytes, alignment "
            << bazavec::kArenaAlignment << "\n";
  std::cout << "GB/s counts database reads only (query is L1-resident). 1 GB = 1e9 bytes.\n\n";

  const float* query = queries.vector(0);
  const float l2_gap = max_abs_l2(base, query, arena.size(), stride);
  const float ip_gap = max_abs_ip(base, query, arena.size(), stride);
  std::cout << std::scientific << std::setprecision(3);
  std::cout << "max |simd - scalar| over " << arena.size() << " L2 distances: " << l2_gap << "\n";
  std::cout << "max |simd - scalar| over " << arena.size() << " IP distances: " << ip_gap << "\n\n";
  std::cout << std::defaultfloat;

  std::cout << std::left << std::setw(16) << "metric" << std::setw(16) << "kernel" << std::right
            << std::setw(12) << "ns/vec" << std::setw(12) << "GB/s" << "\n";

  auto report = [&](const char* metric, const char* kernel, const Timed& timed) {
    const double ns = timed.seconds / static_cast<double>(arena.size()) * 1e9;
    const double gbps = bytes_per_scan / timed.seconds / 1e9;
    print_row(metric, kernel, ns, gbps);
    return timed.seconds;
  };

  std::size_t q = 0;
  auto next_query = [&]() -> const float* {
    const float* pointer = queries.vector(static_cast<std::uint32_t>(q % kQueries));
    ++q;
    return pointer;
  };

  const double l2_scalar = report(
      "L2Squared", "scalar", time_scan([&] { return bazavec_scan_l2_scalar(base, next_query(), kN, stride); },
                                        kTargetSeconds));
  const double l2_auto = report(
      "L2Squared", "autovec",
      time_scan([&] { return bazavec_scan_l2_autovec(base, next_query(), kN, stride); }, kTargetSeconds));
  const char* const isa = bazavec::simd_isa();
  const double l2_simd = report(
      "L2Squared", isa, time_scan([&] { return bazavec_scan_l2_simd(base, next_query(), kN, stride); },
                                   kTargetSeconds));
  const double ip_scalar = report(
      "InnerProduct", "scalar",
      time_scan([&] { return bazavec_scan_ip_scalar(base, next_query(), kN, stride); }, kTargetSeconds));
  const double ip_auto = report(
      "InnerProduct", "autovec",
      time_scan([&] { return bazavec_scan_ip_autovec(base, next_query(), kN, stride); }, kTargetSeconds));
  const double ip_simd = report(
      "InnerProduct", isa,
      time_scan([&] { return bazavec_scan_ip_simd(base, next_query(), kN, stride); }, kTargetSeconds));

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\nL2  simd/scalar   " << (l2_scalar / l2_simd) << "x\n";
  std::cout << "L2  simd/autovec  " << (l2_auto / l2_simd) << "x\n";
  std::cout << "IP  simd/scalar   " << (ip_scalar / ip_simd) << "x\n";
  std::cout << "IP  simd/autovec  " << (ip_auto / ip_simd) << "x\n";

  constexpr int kLatencyQueries = 80;
  constexpr std::size_t kK = 10;
  std::vector<double> scalar_us;
  std::vector<double> simd_us;
  scalar_us.reserve(kLatencyQueries);
  simd_us.reserve(kLatencyQueries);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> probe(kDim);
  float sink = 0.f;

  for (int i = 0; i < 8; ++i) {
    for (float& value : probe) {
      value = dist(rng);
    }
    auto warm_hits =
        bazavec::brute_force_search(arena, probe, kK, bazavec::Metric::L2Squared);
    sink += warm_hits.front().score;
  }

  for (int i = 0; i < kLatencyQueries; ++i) {
    for (float& value : probe) {
      value = dist(rng);
    }
    const auto t0 = Clock::now();
    const std::vector<bazavec::Hit> hits = bazavec::brute_force_search(
        arena, probe, kK, bazavec::Metric::L2Squared, bazavec::DistanceKernel::Simd);
    const auto t1 = Clock::now();
    simd_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    sink += hits.front().score;

    const auto s0 = Clock::now();
    const std::vector<bazavec::Hit> scalar_hits = bazavec::brute_force_search(
        arena, probe, kK, bazavec::Metric::L2Squared, bazavec::DistanceKernel::Scalar);
    const auto s1 = Clock::now();
    scalar_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
    sink += scalar_hits.front().score;
  }
  do_not_optimize(sink);

  auto mean = [](const std::vector<double>& samples) {
    double acc = 0.0;
    for (double sample : samples) {
      acc += sample;
    }
    return acc / static_cast<double>(samples.size());
  };

  std::cout << "\nbrute-force top-" << kK << " L2, " << kLatencyQueries
            << " queries, microseconds\n";
  std::cout << std::left << std::setw(12) << "kernel" << std::right << std::setw(12) << "mean"
            << std::setw(12) << "p50" << std::setw(12) << "p99" << "\n";
  std::cout << std::left << std::setw(12) << "scalar" << std::right << std::fixed << std::setprecision(1)
            << std::setw(12) << mean(scalar_us) << std::setw(12) << percentile(scalar_us, 0.50)
            << std::setw(12) << percentile(scalar_us, 0.99) << "\n";
  std::cout << std::left << std::setw(12) << "simd" << std::right << std::setw(12) << mean(simd_us)
            << std::setw(12) << percentile(simd_us, 0.50) << std::setw(12) << percentile(simd_us, 0.99)
            << "\n";
  std::cout << "top-10 simd/scalar " << std::setprecision(2) << (mean(scalar_us) / mean(simd_us))
            << "x\n";

  std::vector<float> check_query(kDim);
  for (std::size_t i = 0; i < kDim; ++i) {
    check_query[i] = probe[i];
  }
  const std::vector<bazavec::Hit> best = bazavec::brute_force_search(
      arena, check_query, 1, bazavec::Metric::L2Squared, bazavec::DistanceKernel::Simd);
  std::vector<float> padded(stride, 0.f);
  std::memcpy(padded.data(), check_query.data(), kDim * sizeof(float));
  float argmin = bazavec::l2_squared_simd(padded.data(), arena.vector(0), stride);
  std::uint32_t argmin_id = 0;
  for (std::uint32_t i = 1; i < arena.size(); ++i) {
    const float score = bazavec::l2_squared_simd(padded.data(), arena.vector(i), stride);
    if (score < argmin || (score == argmin && i < argmin_id)) {
      argmin = score;
      argmin_id = i;
    }
  }
  const bool top1_ok = !best.empty() && best[0].id == argmin_id && best[0].score == argmin;
  std::cout << "top-1 matches linear argmin: " << (top1_ok ? "yes" : "no") << "\n";

  if (!std::isfinite(l2_gap) || l2_gap > 1e-3f || ip_gap > 1e-3f || !top1_ok) {
    std::cerr << "benchmark correctness check failed\n";
    return 1;
  }
  return 0;
}
