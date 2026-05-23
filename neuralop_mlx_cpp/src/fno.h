// FNO — ports neuralop_mlx/models/fno.py FNO
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "ops.h"
#include "fno_blocks.h"

namespace neuralop {

namespace mx = mlx::core;

struct FNO {
  // Hyper-parameters stored for weight loading / inspection
  int in_channels;
  int out_channels;
  int hidden_channels;
  int n_layers;
  bool use_positional_embedding;
  bool use_channel_mlp;
  std::string fno_skip;
  std::string channel_mlp_skip;
  float lifting_channel_ratio;
  float projection_channel_ratio;
  float channel_mlp_expansion;
  std::string metallib_path;

  std::optional<GridEmbeddingND> positional_embedding;
  std::unique_ptr<ChannelMLP>    lifting;
  std::unique_ptr<FNOBlocks>     blocks;
  std::unique_ptr<ChannelMLP>    projection;

  explicit FNO(
    int in_channels,
    int out_channels,
    int hidden_channels,
    int n_layers,
    const std::vector<int>& n_modes,
    bool use_positional_embedding = true,
    bool use_channel_mlp = true,
    float lifting_channel_ratio    = 2.0f,
    float projection_channel_ratio = 2.0f,
    float channel_mlp_expansion    = 0.5f,
    const std::string& fno_skip         = "linear",
    const std::string& channel_mlp_skip = "soft-gating",
    const std::string& fft_norm         = "forward",
    const std::string& metallib_path    = "");

  mx::array forward(const mx::array& x) const;

  // Prevent copy
  FNO(const FNO&) = delete;
  FNO& operator=(const FNO&) = delete;
};

} // namespace neuralop
