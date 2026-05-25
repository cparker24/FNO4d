// FNOTauEta — FNO extended with sinusoidal τ/η positional embeddings.
// Ports neuralop_mlx/models/fno_tau_eta.py to C++/MLX for inference.
//
// Tensor layout: (B, C, X, Y, Z, T)
//   Z — spacetime rapidity η_s  (axis 4)
//   T — proper time τ           (axis 5)
#pragma once

#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "fno.h"
#include "ops.h"

namespace neuralop {

namespace mx = mlx::core;

struct FNOTauEta {
  FNO                  fno;         // base model (in_channels+4 as in_channels)
  std::vector<TauBias> tau_biases;  // one per FNO layer
  std::vector<EtaBias> eta_biases;  // one per FNO layer
  float                tau_max;
  float                eta_max;

  FNOTauEta() = delete;

  explicit FNOTauEta(
      int in_channels,
      int out_channels,
      int hidden_channels,
      int n_layers,
      const std::vector<int>& n_modes,
      float tau_max,
      float eta_max,
      bool use_positional_embedding  = false,
      bool use_channel_mlp           = true,
      float lifting_channel_ratio    = 2.0f,
      float projection_channel_ratio = 2.0f,
      float channel_mlp_expansion    = 0.5f,
      const std::string& fno_skip         = "linear",
      const std::string& channel_mlp_skip = "soft-gating",
      const std::string& fft_norm         = "forward",
      const std::string& metallib_path    = "");

  // tau_vals : physical τ coordinates, length T  [fm/c]
  // eta_vals : physical η_s coordinates, length Z
  // x        : (B, C_in, X, Y, Z, T)
  // Returns  : (B, C_out, X, Y, Z, T)
  mx::array forward(const mx::array& x,
                    const std::vector<float>& tau_vals,
                    const std::vector<float>& eta_vals) const;

  // Prevent copy (FNO members hold unique_ptrs)
  FNOTauEta(const FNOTauEta&) = delete;
  FNOTauEta& operator=(const FNOTauEta&) = delete;
};

} // namespace neuralop
