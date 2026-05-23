// neuralop_mlx_cpp — public umbrella header
// Inference-only C++ port of neuralop_mlx using Apple's MLX C++ backend.
//
// Typical usage:
//
//   #include "neuralop_mlx_cpp/fno_inference.h"
//   namespace no = neuralop;
//   namespace mx = mlx::core;
//
//   // 1. Load config produced by tools/export_weights.py
//   no::FNOConfig cfg = no::FNOConfig::from_json("model_config.json");
//
//   // 2. Build model
//   no::FNO model(cfg.in_channels, cfg.out_channels, cfg.hidden_channels,
//                 cfg.n_layers, cfg.n_modes,
//                 cfg.positional_embedding, cfg.use_channel_mlp,
//                 cfg.lifting_channel_ratio, cfg.projection_channel_ratio,
//                 cfg.channel_mlp_expansion,
//                 cfg.fno_skip, cfg.channel_mlp_skip, cfg.fft_norm,
//                 cfg.metallib_path);
//
//   // 3. Load weights (MLX-format or PyTorch-format safetensors)
//   no::load_weights(model, "weights.safetensors");
//
//   // 4. Run inference — input shape (B, in_ch, d0, d1, ..., dN-1)
//   mx::array input = mx::random::normal({1, 4, 50, 50, 50, 10});
//   mx::array output = model.forward(input);
//   mx::eval(output);
//
// Weight file formats:
//   MLX-format safetensors  — produced by tools/export_weights.py
//   PyTorch-format safetensors — from safetensors.torch.save_file

#pragma once

#include <string>
#include <vector>
#include <mlx/mlx.h>

namespace neuralop {

// ── FNO hyperparameter configuration ────────────────────────────────────────
// Parsed from the JSON config file written by tools/export_weights.py.
struct FNOConfig {
  std::vector<int> n_modes;         // Fourier modes per spatial dim
  int in_channels          = 4;
  int out_channels         = 4;
  int hidden_channels      = 64;
  int n_layers             = 4;
  float lifting_channel_ratio    = 2.0f;
  float projection_channel_ratio = 2.0f;
  bool  positional_embedding     = true;
  bool  use_channel_mlp          = true;
  float channel_mlp_expansion    = 0.5f;
  std::string fno_skip         = "linear";
  std::string channel_mlp_skip = "soft-gating";
  std::string fft_norm         = "forward";
  // Path to neuralop_metal.metallib; empty = use default build-time path.
  std::string metallib_path    = "";

  // Deserialise from JSON (see tools/export_weights.py)
  static FNOConfig from_json(const std::string& path);

  // Deserialise from a training YAML (e.g. config_test_train.yaml).
  // FNO-specific fields not present in the YAML use the defaults above.
  // Only 'n_modes' is required; 'hidden_channels' and any other recognised
  // top-level keys are picked up automatically; unknown keys are ignored.
  static FNOConfig from_yaml(const std::string& path);

  // Convenience: dispatch on file extension (.json → from_json, else from_yaml)
  static FNOConfig from_file(const std::string& path);
};

} // namespace neuralop

// Include the full implementation headers so callers get FNO and load_weights
#include "../src/fno.h"
#include "../src/weight_loader.h"
