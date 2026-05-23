// SpectralConvND — ports neuralop_mlx/layers/spectral_convolution.py
#pragma once

#include <string>
#include <vector>
#include <mlx/mlx.h>

namespace neuralop {

namespace mx = mlx::core;

// N-dimensional real FFT spectral convolution layer.
// Stores weights as (weight_r, weight_i): float32 (in_ch, out_ch, m0,…,mN-1)
//
// Forward:
//   1. N-D rFFT of input
//   2. fftshift on non-last FFT axes (if order > 1)
//   3. Extract mode sub-region and contract with weight
//   4. Pad back to full FFT size
//   5. ifftshift (if order > 1)
//   6. Hermitian-symmetric inverse FFT
//   7. Add bias
struct SpectralConvND {
  int in_channels;
  int out_channels;
  int order;            // number of spatial dims
  std::vector<int> n_modes;        // per-dim mode counts; last entry halved for rFFT
  std::vector<int> _n_modes;       // effective: last = n_modes.back()/2 + 1

  mx::array weight_r;  // (in_ch, out_ch, m0,...,mN-1)
  mx::array weight_i;
  mx::array bias;      // (out_ch,) — may be empty
  bool      has_bias   = false;

  bool enforce_hermitian_symmetry = true;
  std::string fft_norm;           // "forward" | "backward" | "ortho"
  std::string metallib_path;

  SpectralConvND() = delete;
  SpectralConvND(int in_ch, int out_ch,
                 std::vector<int> n_modes,
                 bool use_bias = true,
                 bool hermitian = true,
                 std::string fft_norm = "forward",
                 std::string metallib_path = "");

  mx::array forward(const mx::array& x) const;
};

} // namespace neuralop
