cmake_minimum_required(VERSION 3.21)
project(neuralop_mlx_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Locate MLX ───────────────────────────────────────────────────────────────
# If MLX is installed inside a conda/venv, pass its cmake dir with:
#   cmake -DMLX_DIR=<path/to/site-packages/mlx/share/cmake/MLX> ..
# or
#   cmake -DCMAKE_PREFIX_PATH=<path/to/site-packages/mlx> ..
find_package(MLX REQUIRED)

message(STATUS "MLX include dirs : ${MLX_INCLUDE_DIRS}")
message(STATUS "MLX build metal  : ${MLX_BUILD_METAL}")

# ── nlohmann/json (single header, downloaded once at configure time) ─────────
set(JSON_HPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(JSON_HPP     ${JSON_HPP_DIR}/json.hpp)
if(NOT EXISTS ${JSON_HPP})
  message(STATUS "Downloading nlohmann/json.hpp …")
  file(MAKE_DIRECTORY ${JSON_HPP_DIR})
  file(DOWNLOAD
    "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
    ${JSON_HPP}
    STATUS _dl_status)
  list(GET _dl_status 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    message(FATAL_ERROR
      "Failed to download json.hpp. Either provide it manually at\n"
      "  ${JSON_HPP}\n"
      "or set -DJSON_HPP_DIR=<dir containing json.hpp>")
  endif()
endif()

# ── Library sources (always compiled) ───────────────────────────────────────
add_library(neuralop_mlx_cpp STATIC
  src/config.cpp
  src/ops.cpp
  src/spectral_conv.cpp
  src/fno_blocks.cpp
  src/fno.cpp
  src/weight_loader.cpp
)

target_include_directories(neuralop_mlx_cpp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS}
    ${JSON_HPP_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(neuralop_mlx_cpp PRIVATE ${MLX_CXX_FLAGS})
target_link_libraries(neuralop_mlx_cpp PUBLIC mlx)

# ── Spectral contraction: Metal kernel (default) or matmul fallback ──────────
option(NEURALOP_USE_METAL
  "Build custom Metal kernel for the spectral contraction (requires MLX Metal)"
  ON)

if(NEURALOP_USE_METAL AND MLX_BUILD_METAL)
  # Check if the Metal compiler actually works (xcrun metal --version)
  execute_process(
    COMMAND xcrun metal --version
    RESULT_VARIABLE _metal_ok
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT _metal_ok EQUAL 0)
    message(STATUS "NEURALOP_USE_METAL=ON but Metal compiler not found; "
                   "falling back to matmul contraction. "
                   "Install Metal toolchain with: "
                   "xcodebuild -downloadComponent MetalToolchain")
    set(NEURALOP_USE_METAL OFF)
  endif()
endif()

if(NEURALOP_USE_METAL AND MLX_BUILD_METAL)
  message(STATUS "Spectral contraction: custom Metal kernel")

  # Output the .metallib next to the library
  set(METALLIB_OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib)
  file(MAKE_DIRECTORY ${METALLIB_OUTPUT_DIR})

  # mlx_build_metallib is defined in extension.cmake, included by find_package(MLX)
  mlx_build_metallib(
    TARGET   neuralop_metallib
    TITLE    neuralop_metal
    SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/spectral_contract.metal
    INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${METALLIB_OUTPUT_DIR}
  )

  target_compile_definitions(neuralop_mlx_cpp PRIVATE
    NEURALOP_USE_METAL=1
    NEURALOP_DEFAULT_METALLIB_PATH="${METALLIB_OUTPUT_DIR}/neuralop_metal.metallib"
  )

  target_sources(neuralop_mlx_cpp PRIVATE src/spectral_contract_metal.cpp)
  add_dependencies(neuralop_mlx_cpp neuralop_metallib)

else()
  if(NEURALOP_USE_METAL AND NOT MLX_BUILD_METAL)
    message(STATUS "NEURALOP_USE_METAL=ON but MLX has no Metal support; "
                   "falling back to matmul contraction")
  else()
    message(STATUS "Spectral contraction: reshape+batched-matmul fallback")
  endif()
  target_sources(neuralop_mlx_cpp PRIVATE src/spectral_contract_cpu.cpp)
endif()

# ── Example inference binary ─────────────────────────────────────────────────
add_executable(fno_infer examples/infer.cpp)
target_link_libraries(fno_infer PRIVATE neuralop_mlx_cpp)
target_compile_options(fno_infer PRIVATE ${MLX_CXX_FLAGS})

# ── Quick build instructions printed at configure time ───────────────────────
message(STATUS "")
message(STATUS "Build: cmake --build . --target fno_infer")
message(STATUS "Usage: ./fno_infer <weights.safetensors> <config.json> <B> <d0> [d1 ...]")
