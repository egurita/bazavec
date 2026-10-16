#pragma once

#include "hnsw.hpp"
#include "storage.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#define BAZAVEC_CRC32_HW 1
#endif

namespace bazavec {

inline constexpr char kBzvMagic[4] = {'B', 'Z', 'V', 'C'};
inline constexpr std::uint32_t kBzvVersion = 1;
inline constexpr std::size_t kBzvHeaderBytes = 64;
inline constexpr std::size_t kBzvPayloadAlign = 128;

namespace detail {

inline void put_u32(std::uint8_t* out, std::uint32_t value) noexcept {
  out[0] = static_cast<std::uint8_t>(value);
  out[1] = static_cast<std::uint8_t>(value >> 8);
  out[2] = static_cast<std::uint8_t>(value >> 16);
  out[3] = static_cast<std::uint8_t>(value >> 24);
}

inline std::uint32_t get_u32(const std::uint8_t* in) noexcept {
  return static_cast<std::uint32_t>(in[0]) | (static_cast<std::uint32_t>(in[1]) << 8) |
         (static_cast<std::uint32_t>(in[2]) << 16) | (static_cast<std::uint32_t>(in[3]) << 24);
}

inline std::uint32_t crc32(const std::uint8_t* data, std::size_t n) noexcept {
  std::uint32_t crc = 0xffffffffu;
#if defined(BAZAVEC_CRC32_HW)
  // Same ISO polynomial as the byte loop (0xEDB88320). A bit at a time is
  // about one byte per several cycles, which made a 70 MiB check take
  // hundreds of milliseconds. The M-series CRC instruction does a word per
  // instruction.
  while (n >= 8) {
    std::uint64_t word = 0;
    std::memcpy(&word, data, sizeof(word));
    crc = __crc32d(crc, word);
    data += 8;
    n -= 8;
  }
  if (n >= 4) {
    std::uint32_t word = 0;
    std::memcpy(&word, data, sizeof(word));
    crc = __crc32w(crc, word);
    data += 4;
    n -= 4;
  }
  if (n >= 2) {
    std::uint16_t word = 0;
    std::memcpy(&word, data, sizeof(word));
    crc = __crc32h(crc, word);
    data += 2;
    n -= 2;
  }
  if (n == 1) {
    crc = __crc32b(crc, data[0]);
  }
#else
  for (std::size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xedb88320u & mask);
    }
  }
#endif
  return ~crc;
}

inline void align_buffer(std::vector<std::uint8_t>& buffer, std::size_t alignment) {
  while (buffer.size() % alignment != 0) {
    buffer.push_back(0);
  }
}

inline void append_bytes(std::vector<std::uint8_t>& buffer, const void* data, std::size_t n) {
  if (n == 0) {
    return;
  }
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  buffer.insert(buffer.end(), bytes, bytes + n);
}

inline std::uint32_t as_u32(std::size_t n) {
  if (n > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("save_bzv: section does not fit in 32 bits");
  }
  return static_cast<std::uint32_t>(n);
}

}  // namespace detail

inline void save_bzv(const std::string& path, const VectorArena& arena, const HnswIndex& index) {
  if (!index.built() || arena.size() != index.size() || arena.size() == 0) {
    throw std::invalid_argument("save_bzv: index must be built on this arena");
  }
  const HnswIndex::GraphImage graph = index.graph_image();
  std::vector<std::uint8_t> file(kBzvPayloadAlign, 0);
  detail::append_bytes(file, arena.data(), arena.bytes());
  detail::append_bytes(file, graph.levels, graph.nodes);
  detail::align_buffer(file, alignof(std::uint16_t));
  detail::append_bytes(file, graph.degree0, graph.nodes * sizeof(std::uint16_t));
  detail::align_buffer(file, alignof(std::uint32_t));
  detail::append_bytes(file, graph.neighbors0, graph.neighbors0_len * sizeof(std::uint32_t));
  detail::append_bytes(file, graph.upper_slot, graph.nodes * sizeof(std::uint32_t));
  detail::append_bytes(file, graph.upper_degree, graph.upper_degree_len * sizeof(std::uint16_t));
  detail::align_buffer(file, alignof(std::uint32_t));
  detail::append_bytes(file, graph.upper_edges, graph.upper_edges_len * sizeof(std::uint32_t));

  std::uint8_t header[kBzvHeaderBytes] = {};
  std::memcpy(header, kBzvMagic, 4);
  detail::put_u32(header + 4, kBzvVersion);
  detail::put_u32(header + 8, detail::as_u32(arena.dim()));
  detail::put_u32(header + 12, detail::as_u32(arena.size()));
  detail::put_u32(header + 16, detail::as_u32(arena.stride()));
  detail::put_u32(header + 20, static_cast<std::uint32_t>(index.metric()));
  detail::put_u32(header + 24, index.params().m);
  detail::put_u32(header + 28, index.params().ef_construction);
  detail::put_u32(header + 32, index.params().ef_search);
  detail::put_u32(header + 36, index.layer0_stride());
  detail::put_u32(header + 40, static_cast<std::uint32_t>(index.max_level()));
  detail::put_u32(header + 44, index.entry_point());
  detail::put_u32(header + 48, detail::as_u32(graph.upper_degree_len));
  detail::put_u32(header + 52, detail::as_u32(graph.upper_edges_len));
  detail::put_u32(header + 56, detail::crc32(file.data() + kBzvPayloadAlign, file.size() - kBzvPayloadAlign));
  std::memcpy(file.data(), header, kBzvHeaderBytes);

  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) {
    throw std::runtime_error("save_bzv: open failed");
  }
  std::size_t written = 0;
  while (written < file.size()) {
    const ssize_t n = ::write(fd, file.data() + written, file.size() - written);
    if (n < 0) {
      ::close(fd);
      throw std::runtime_error("save_bzv: write failed");
    }
    written += static_cast<std::size_t>(n);
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("save_bzv: close failed");
  }
}

// The float matrix and the HNSW arrays are pointers into the mapping. Nothing
// in search copies them. The mapping outlives the arena and the index because
// it is declared first and destroyed last.
class MappedIndex {
 public:
  explicit MappedIndex(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("MappedIndex: open failed");
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
      ::close(fd);
      throw std::runtime_error("MappedIndex: stat failed");
    }
    const auto bytes = static_cast<std::size_t>(st.st_size);
    if (bytes < kBzvPayloadAlign) {
      ::close(fd);
      throw std::runtime_error("MappedIndex: file is shorter than the header");
    }
    void* address = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (address == MAP_FAILED) {
      throw std::runtime_error("MappedIndex: mmap failed");
    }
    mapping_.address = address;
    mapping_.bytes = bytes;

    const auto* base = static_cast<const std::uint8_t*>(address);
    if (std::memcmp(base, kBzvMagic, 4) != 0 || detail::get_u32(base + 4) != kBzvVersion) {
      throw std::runtime_error("MappedIndex: bad magic or version");
    }
    const std::uint32_t crc = detail::get_u32(base + 56);
    if (crc != detail::crc32(base + kBzvPayloadAlign, bytes - kBzvPayloadAlign)) {
      throw std::runtime_error("MappedIndex: checksum mismatch");
    }

    const std::uint32_t dim = detail::get_u32(base + 8);
    const std::uint32_t count = detail::get_u32(base + 12);
    const std::uint32_t stride = detail::get_u32(base + 16);
    const std::uint32_t metric_raw = detail::get_u32(base + 20);
    HnswIndex::Params params;
    params.m = detail::get_u32(base + 24);
    params.ef_construction = detail::get_u32(base + 28);
    params.ef_search = detail::get_u32(base + 32);
    params.threads = 1;
    const std::uint32_t max_m0 = detail::get_u32(base + 36);
    const int max_level = static_cast<int>(detail::get_u32(base + 40));
    const std::uint32_t entry = detail::get_u32(base + 44);
    const std::uint32_t upper_degree_len = detail::get_u32(base + 48);
    const std::uint32_t upper_edges_len = detail::get_u32(base + 52);
    if (dim == 0 || count == 0 || metric_raw > static_cast<std::uint32_t>(Metric::Cosine) ||
        stride != VectorArena::padded_stride(dim) || params.m < 2 || max_m0 != params.m * 2 || entry >= count ||
        max_level < 0 || max_level > HnswIndex::kMaxLevel) {
      throw std::runtime_error("MappedIndex: header does not describe a graph");
    }

    const auto* end = base + bytes;
    auto* cursor = const_cast<std::uint8_t*>(base + kBzvPayloadAlign);
    auto advance = [&](std::size_t n) {
      if (cursor > end || static_cast<std::size_t>(end - cursor) < n) {
        throw std::runtime_error("MappedIndex: section runs past the file");
      }
      cursor += n;
    };
    auto align_up = [&](std::size_t alignment) {
      const auto address = reinterpret_cast<std::uintptr_t>(cursor);
      const std::size_t pad = (alignment - (address % alignment)) % alignment;
      advance(pad);
    };
    auto span_bytes = [](std::uint64_t items, std::uint64_t width) {
      if (width != 0 && items > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::runtime_error("MappedIndex: section runs past the file");
      }
      return static_cast<std::size_t>(items * width);
    };

    auto* floats = reinterpret_cast<float*>(cursor);
    advance(span_bytes(count, static_cast<std::uint64_t>(stride) * sizeof(float)));
    auto* levels = cursor;
    advance(count);
    align_up(alignof(std::uint16_t));
    auto* degree0 = reinterpret_cast<std::uint16_t*>(cursor);
    advance(span_bytes(count, sizeof(std::uint16_t)));
    align_up(alignof(std::uint32_t));
    auto* neighbors0 = reinterpret_cast<std::uint32_t*>(cursor);
    advance(span_bytes(count, static_cast<std::uint64_t>(max_m0) * sizeof(std::uint32_t)));
    auto* upper_slot = reinterpret_cast<std::uint32_t*>(cursor);
    advance(span_bytes(count, sizeof(std::uint32_t)));
    auto* upper_degree = reinterpret_cast<std::uint16_t*>(cursor);
    advance(span_bytes(upper_degree_len, sizeof(std::uint16_t)));
    align_up(alignof(std::uint32_t));
    auto* upper_edges = reinterpret_cast<std::uint32_t*>(cursor);
    advance(span_bytes(upper_edges_len, sizeof(std::uint32_t)));
    if (cursor != end) {
      throw std::runtime_error("MappedIndex: trailing or missing bytes");
    }

    arena_.reset(new VectorArena(floats, count, dim));
    index_.reset(new HnswIndex(*arena_, static_cast<Metric>(metric_raw), params));
    index_->restore_mapped(levels, degree0, neighbors0, static_cast<std::size_t>(count) * max_m0, upper_slot,
                           upper_degree, upper_degree_len, upper_edges, upper_edges_len, entry, max_level);
  }

  MappedIndex(const MappedIndex&) = delete;
  MappedIndex& operator=(const MappedIndex&) = delete;

  const VectorArena& arena() const noexcept { return *arena_; }
  const HnswIndex& index() const noexcept { return *index_; }
  const void* mapping() const noexcept { return mapping_.address; }
  std::size_t mapping_bytes() const noexcept { return mapping_.bytes; }

  bool contains(const void* pointer) const noexcept {
    const auto* begin = static_cast<const std::uint8_t*>(mapping_.address);
    const auto* at = static_cast<const std::uint8_t*>(pointer);
    return at >= begin && at < begin + mapping_.bytes;
  }

  std::vector<Hit> search_knn(std::span<const float> query, std::size_t k, std::size_t ef = 0) const {
    return index_->search_knn(query, k, ef);
  }

 private:
  struct Mapping {
    void* address = nullptr;
    std::size_t bytes = 0;
    ~Mapping() {
      if (address != nullptr && address != MAP_FAILED) {
        ::munmap(address, bytes);
      }
    }
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
  };

  Mapping mapping_;
  std::unique_ptr<VectorArena> arena_;
  std::unique_ptr<HnswIndex> index_;
};

}  // namespace bazavec
