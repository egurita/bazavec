# bazavec

Header-only C++20 vector search. There are no library dependencies. An index is a 128-byte-aligned float matrix, an HNSW graph in a handful of flat arrays, and an optional int8 code matrix. A query is either an exact scan or a graph walk. A bitmap predicate is applied on the bottom layer. A `.bzv` file maps back in without copying the matrix or the graph.

Public headers live under `include/bazavec/`.

## Layout

| Path | Role |
| --- | --- |
| `include/bazavec/detail/memory.hpp` | `std::aligned_alloc` block, freed with `free` |
| `include/bazavec/storage.hpp` | `VectorArena`, stride padded to 128 bytes |
| `include/bazavec/distance.hpp` | Scalar, NEON, and AVX2 float kernels |
| `include/bazavec/brute_force.hpp` | Exact top-k |
| `include/bazavec/hnsw.hpp` | Flattened multi-layer HNSW |
| `include/bazavec/quantizer.hpp` | Symmetric int8 quantization and integer L2 |
| `include/bazavec/filter.hpp` | Column predicate compiled to a bitset |
| `include/bazavec/persistence.hpp` | `.bzv` header, CRC-32, `mmap` load |
| `tests/test_phase1.cpp` | Kernel identity, alignment, heap order |
| `tests/test_phase2.cpp` | Heuristic, graph shape, Recall@10 |
| `tests/test_phase3.cpp` | Int8 kernels, filtered and int8 HNSW recall |
| `tests/test_phase4.cpp` | Save, zero-copy load, checksum |
| `bench/bench_phase1.cpp` | Float kernel throughput and top-10 latency |
| `bench/bench_phase2.cpp` | HNSW build and latency on clustered data |
| `bench/bench_phase5.cpp` | Scalar, NEON, int8, HNSW, and `mmap` together |

`stride` is `dim` rounded up to 32 floats. Row `i` is `data + i * stride`, so every row starts on a cache line and a 128-wide row has no scalar tail.

## Distances

`L2Squared` returns squared Euclidean distance. Square root is strictly increasing on `[0, ∞)`, so it cannot change which neighbor wins. `InnerProduct` and `Cosine` return similarity. Larger is better. A zero vector has cosine 0.

The scalar loop is one `fmadd` into a single accumulator, so the next element waits on that add. Clang's auto-vectorizer emits NEON arithmetic, then folds the lanes back into one scalar register before the next chunk. The hand-written kernel keeps four `fmla.4s` accumulators for the whole row and horizontal-adds once. `-ffast-math` is off. Scalar and SIMD disagree only by reduction order.

The int8 kernel is a widening multiply. A difference of two int8 values fits in int16. The square of that difference does not (`255^2 = 65025`), so the product is widened to int32 before it is accumulated. This Mac reports `hw.optional.arm.FEAT_DotProd = 1`. The kernel does not use a dot-product instruction, for that reason. On x86_64 the float kernels are AVX2 and the int8 kernel is the same widening multiply.

## HNSW

The graph is the one in Malkov and Yashunin. Default `M` is 16. Layer 0 allows `2M` neighbors. Upper layers allow `M`. A node's layer is drawn from the geometric distribution in the paper, capped at 16.

Layer 0 is one array of `n * 2M` ids. The neighbor list of node `i` is the slice `[i * 2M, i * 2M + degree[i])`. Upper layers hold about `1/M` of the nodes and use a second packed array. There is no `vector<vector<int>>`.

A search stamps nodes with a `uint16_t` generation instead of clearing a bitset. On wrap the stamps reset. The walk issues `__builtin_prefetch` on the next neighbor's row before the distance in flight. That address is data-dependent, so the hardware stride prefetcher does not see it. The prefetch stays inside the row.

`build()` inserts on a thread pool, in id order. Each node has a spinlock on its own cache line (`alignas(128)`), held only while its neighbor list is copied or replaced. The heuristic prune of a full list is a few dozen distances and stays under that one lock. Two locks are never held. After the workers join, search does not take the locks. Concurrent searches are not safe: they share one visited table and one query pad. Parallel insert does not reproduce one edge set across runs. Tests that compare a saved graph use one thread.

Neighbor selection is the paper's relative-neighborhood test. A candidate is kept when it is closer to the query than to every neighbor already chosen. Leftover slots are filled with the rejected candidates so a sparse region still reaches degree `M`.

`bind_quantizer()` before `build()` ranks with the int8 kernel. Insert and search then agree: the edges are neighbors in code space.

A filter is a bitset built from `category == X` and `value >= Y`. Upper layers ignore it and descend geometrically. On layer 0 a rejected node is not a hit, but it stays in the beam so the walk can cross it.

## File format

The header is 64 bytes, little-endian: magic `BZVC`, version, dimension, count, stride, metric, HNSW parameters, entry point, and the CRC-32 of the payload. Another 64 zero bytes put the float matrix at offset 128. `mmap` on this Mac is 16 KiB-aligned, so a row that starts 128 bytes later is still 128-byte aligned, which is what `VectorArena` requires.

The payload is the float matrix, then levels, degrees, layer-0 adjacency, and the packed upper layers. `MappedIndex` points the arena and the graph at those bytes. Search does not allocate a second copy. Spinlocks are not in the file. The destructor `munmap`s.

CRC-32 is the ISO polynomial `0xEDB88320`, initial value `0xFFFFFFFF`, final invert. The check value of the ASCII string `123456789` is `0xCBF43926`. On arm64, when the compiler enables CRC instructions, the implementation is `__crc32d` / `__crc32w`. The byte-at-a-time loop is the same polynomial and is what a debug build uses. A bit loop over a 70 MiB file took about 435 ms. The instruction brings that check to about 11 ms while the file is still in the buffer cache.

## Build

Release is what the benchmark measures. On arm64 that is `-O3 -mcpu=native`. On x86_64 it is `-O3 -march=native -mavx2 -mfma`.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build/release -j
./build/release/bazavec_tests
./build/release/bazavec_phase2_tests
./build/release/bazavec_phase3_tests
./build/release/bazavec_phase4_tests
./build/release/bench_phase5
```

Debug turns on UBSan with `-fno-sanitize-recover=undefined`:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -G Ninja
cmake --build build/debug -j
./build/debug/bazavec_tests
```

AddressSanitizer belongs on that line too (`-fsanitize=address,undefined`), enabled with `-DBAZAVEC_ENABLE_ASAN=ON`. AppleClang 17's ASan runtime on macOS 27 deadlocks before `main`: `AsanInitFromRtl` allocates while its init spinlock is held. An empty `main` reproduces it. UBSan is what Debug runs here until that runtime is fixed.

## Benchmark

One Release run of `bench_phase5` on a MacBook Air, Apple M4 (4 performance cores, 6 efficiency cores, 16 GB), AppleClang 17. The bench thread sets `QOS_CLASS_USER_INTERACTIVE`.

The set is 100,000 vectors of dimension 128, drawn from 200 Gaussians (sigma 0.05, centers uniform in `[-1, 1]^128`). `M = 16`, `ef_construction = 128`, `ef_search = 64`, build across the hardware threads. Recall@10 uses 30 queries against exact float L2. Latency uses 80 queries and is warm. 1 GB = 1e9 bytes. The kernel scan counts database bytes only.

| Kernel | ns/vector | GB/s |
| --- | ---: | ---: |
| FP32 scalar | 63.39 | 8.08 |
| FP32 NEON | 7.62 | 67.18 |
| INT8 scalar | 41.97 | 3.05 |
| INT8 NEON | 3.58 | 35.78 |

NEON is 8.3x the scalar float loop. The int8 NEON loop moves 4x less data and finishes a vector in 3.58 ns against 7.62 ns for float NEON. Its GB/s is lower because the row is 128 bytes instead of 512.

| Path | Recall@10 | Setup | mean µs | p50 µs | p99 µs | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brute force, FP32 scalar | 1.000 | | 7036.7 | 7005.1 | 7534.2 | 142 |
| Brute force, FP32 NEON | 1.000 | | 790.3 | 762.8 | 1074.0 | 1265 |
| HNSW, FP32 | 1.000 | build 1.210 s | 39.6 | 41.5 | 56.2 | 25262 |
| HNSW, FP32, mapped | 1.000 | load 10.66 ms | 46.8 | 45.1 | 116.4 | 21345 |
| Brute force, INT8 scalar | 0.890 | | 4367.3 | 4345.8 | 4525.9 | 229 |
| Brute force, INT8 NEON | 0.890 | | 360.7 | 354.0 | 447.4 | 2773 |
| HNSW, INT8 | 0.890 | build 0.815 s | 25.7 | 26.1 | 35.6 | 38877 |

The float matrix is 51,200,000 bytes (48.83 MiB). The int8 codes are 12,800,000 bytes (12.21 MiB), four times smaller. Training and encoding took 0.014 s. The FP32 graph is 32,867,264 bytes (31.35 MiB) including the build-time spinlocks. Those locks are not written out. The `.bzv` file is 71,267,392 bytes (67.97 MiB).

HNSW FP32 is about 20x the NEON scan at Recall@10 of 1.00 on this set. HNSW INT8 recall against the exact int8 ranking was 1.000. The drop to 0.890 against float L2 is the quantizer: the integer distance is the sum of squared code deltas, and a dimension with a larger scale does not weigh more. The exact int8 scan has the same 0.890, so the graph is not where that loss comes from.

The 10.66 ms figure is `mmap` plus CRC-32 of the file just written, so the pages are still in the buffer cache. It is not a cold read from the SSD. About 71 MB in 10.66 ms is 6.7 GB/s. The mapped top-10 ids and scores matched the live index. Search after that walks the mapped pages.

Uniform noise in 128-d concentrates distances. The same `ef` does not hold 0.95 once N is large. The unit test covers that distribution at a few thousand vectors.

`getrusage` reported a max RSS of 380,583,936 bytes at the end of the process. That high-water mark holds the float matrix, both graphs, the codes, and the mapping at once.

## Tests

`bazavec_tests` is 570 checks: padded stride, 128-byte row alignment, zero padding, bit-exact distances on dyadic rationals, a double-precision reference, tie-break by smaller id, and brute-force order against a full sort for L2, inner product, and cosine. Phase 2 requires Recall@10 above 0.95 against brute force, including a parallel build. Phase 3 checks the int8 kernels against the scalar loop, filtered Recall@10, and int8 HNSW against an exact code-space scan. Phase 4 checks that a mapped search returns the same ids and scores, that the row pointer is inside the mapping, and that a flipped byte or a bad magic value is rejected. Release and Debug/UBSan both pass. The AVX2 kernels were syntax-checked with `-target x86_64-apple-macos14 -mavx2 -mfma`.
