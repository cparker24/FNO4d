// FNOBlocks — ports neuralop_mlx/layers/fno_block.py FNOBlocks.
// One block = SpectralConvND + skip + optional ChannelMLP + skip
#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <mlx/mlx.h>
#include "ops.h"
#include "spectral_conv.h"

namespace neuralop {

namespace mx = mlx::core;

struct FNOBlocks {
  int n_layers;
  int hidden_channels;
  bool use_channel_mlp;

  std::vector<SpectralConvND> convs;        // one per layer
  std::vector<Skip>           fno_skips;    // one per layer
  std::vector<ChannelMLP>     channel_mlps; // one per layer (empty if !use_channel_mlp)
  std::vector<Skip>           mlp_skips;    // one per layer (empty if !use_channel_mlp)

  FNOBlocks() = default;
  FNOBlocks(int in_channels, int out_channels, int hidden_channels,
            int n_layers, const std::vector<int>& n_modes,
            bool use_channel_mlp, float channel_mlp_expansion,
            const std::string& fno_skip_type,
            const std::string& mlp_skip_type,
            const std::string& fft_norm,
            const std::string& metallib_path);

  // Apply the i-th block; returns updated x.
  // If i < n_layers-1, applies gelu after each sub-block.
  mx::array forward(const mx::array& x, int index) const;
};

} // namespace neuralop
