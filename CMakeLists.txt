cmake_minimum_required(VERSION 3.20)

project(bazavec
  VERSION 0.2.0
  DESCRIPTION "High-performance vector search and hybrid query engine"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release" FORCE)
endif()

# -ffast-math is intentionally absent. It reassociates reductions, so the
# scalar baseline and the SIMD kernel would no longer agree within rounding error.

add_library(bazavec INTERFACE)
add_library(bazavec::bazavec ALIAS bazavec)
target_include_directories(bazavec INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(bazavec INTERFACE cxx_std_20)

# Debug carries UBSan with recovery disabled, so a real bug aborts the test.
# AddressSanitizer belongs on that line too (`-fsanitize=address,undefined`).
# AppleClang 17's ASan runtime on macOS 27 deadlocks before main: AsanInitFromRtl
# mallocs while holding its init spinlock, and an empty main reproduces it.
# -DBAZAVEC_ENABLE_ASAN=ON is the full Debug preset once that runtime is fixed.
option(BAZAVEC_ENABLE_ASAN "AddressSanitizer in Debug (hangs on AppleClang 17 / macOS 27)" OFF)
set(BAZAVEC_DEBUG_SANITIZERS "undefined")
if(BAZAVEC_ENABLE_ASAN)
  set(BAZAVEC_DEBUG_SANITIZERS "address,undefined")
endif()
message(STATUS "bazavec Debug sanitizers: ${BAZAVEC_DEBUG_SANITIZERS}")

find_package(Threads REQUIRED)

function(bazavec_configure target)
  target_link_libraries(${target} PRIVATE bazavec Threads::Threads)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    "$<$<CONFIG:Debug>:-O0;-g;-fno-omit-frame-pointer;-fsanitize=${BAZAVEC_DEBUG_SANITIZERS};-fno-sanitize-recover=undefined>")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    target_compile_options(${target} PRIVATE
      "$<$<NOT:$<CONFIG:Debug>>:-O3;-mcpu=native>")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_compile_options(${target} PRIVATE
      "$<$<NOT:$<CONFIG:Debug>>:-O3;-march=native;-mavx2;-mfma>")
  else()
    target_compile_options(${target} PRIVATE
      "$<$<NOT:$<CONFIG:Debug>>:-O3>")
  endif()
  target_link_options(${target} PRIVATE
    "$<$<CONFIG:Debug>:-fsanitize=${BAZAVEC_DEBUG_SANITIZERS}>")
endfunction()

add_executable(bazavec_tests tests/test_phase1.cpp)
bazavec_configure(bazavec_tests)

add_executable(bench_bazavec bench/bench_phase1.cpp)
bazavec_configure(bench_bazavec)


enable_testing()
add_test(NAME phase1 COMMAND bazavec_tests)

message(STATUS "bazavec processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "bazavec build type: ${CMAKE_BUILD_TYPE}")
