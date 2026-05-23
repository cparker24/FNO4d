// examples/infer.cpp — minimal inference demo for neuralop_mlx_cpp
//
// Usage:
//   ./fno_infer <weights.safetensors> <config.(json|yaml)> <B> <d0> [d1 d2 …]
//
// Examples:
//   ./fno_infer weights.safetensors model_config.json 1 50 50 50 10
//   ./fno_infer weights.safetensors config_test_train.yaml 1 50 50 50 10

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "neuralop_mlx_cpp/fno_inference.h"
#include "../src/fno.h"
#include "../src/weight_loader.h"

namespace mx = mlx::core;
using namespace neuralop;

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <weights.safetensors> <config.(json|yaml)>"
                 " <batch> <d0> [d1 d2 …]\n";
    return 1;
  }

  const std::string weights_path = argv[1];
  const std::string config_path  = argv[2];
  const int         batch        = std::atoi(argv[3]);

  std::vector<int> spatial;
  for (int i = 4; i < argc; ++i) spatial.push_back(std::atoi(argv[i]));

  if (spatial.empty()) {
    std::cerr << "Error: provide at least one spatial dimension\n";
    return 1;
  }

  // ── Load config (JSON or YAML, detected by extension) ────────────────────
  FNOConfig cfg;
  try {
    cfg = FNOConfig::from_file(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load config: " << e.what() << "\n";
    return 1;
  }

  if ((int)spatial.size() != (int)cfg.n_modes.size()) {
    std::cerr << "Error: config has " << cfg.n_modes.size()
              << " spatial dims but " << spatial.size()
              << " were specified on the command line.\n";
    return 1;
  }

  std::cout << "Model: in_ch=" << cfg.in_channels
            << "  out_ch=" << cfg.out_channels
            << "  hidden=" << cfg.hidden_channels
            << "  layers=" << cfg.n_layers << "\n";

  // ── Build model ───────────────────────────────────────────────────────────
  FNO model(cfg.in_channels, cfg.out_channels, cfg.hidden_channels,
             cfg.n_layers, cfg.n_modes,
             cfg.positional_embedding, cfg.use_channel_mlp,
             cfg.lifting_channel_ratio, cfg.projection_channel_ratio,
             cfg.channel_mlp_expansion,
             cfg.fno_skip, cfg.channel_mlp_skip,
             cfg.fft_norm, cfg.metallib_path);

  // ── Load weights ──────────────────────────────────────────────────────────
  std::cout << "Loading weights from " << weights_path << " …\n";
  try {
    load_weights(model, weights_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load weights: " << e.what() << "\n";
    return 1;
  }

  // ── Build random input ────────────────────────────────────────────────────
  mx::Shape input_shape = {batch, cfg.in_channels};
  for (int d : spatial) input_shape.push_back(d);

  std::cout << "Input shape: [";
  for (int i = 0; i < (int)input_shape.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << input_shape[i];
  }
  std::cout << "]\n";

  auto input = mx::random::normal(input_shape);
  mx::eval(input);  // realise input before timing

  // ── Warm-up + inference ────────────────────────────────────────────────────
  {
    // Warm-up run (compiles Metal kernels / JIT)
    std::cout << "Warm-up pass …\n";
    auto out_warmup = model.forward(input);
    mx::eval(out_warmup);
  }

  // Timed run
  auto t0 = std::chrono::high_resolution_clock::now();
  auto output = model.forward(input);
  mx::eval(output);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // ── Print results ─────────────────────────────────────────────────────────
  std::cout << "Output shape: [";
  for (int i = 0; i < output.ndim(); ++i) {
    if (i) std::cout << ", ";
    std::cout << output.shape(i);
  }
  std::cout << "]\n";
  std::cout << "Inference time: " << ms << " ms\n";

  // Basic sanity: print a few output values
  auto flat = mx::flatten(output);
  mx::eval(flat);
  std::cout << "First 4 output values: ";
  for (int i = 0; i < std::min(4, (int)flat.size()); ++i) {
    auto val = mx::take(flat, i).item<float>();
    std::cout << val << "  ";
  }
  std::cout << "\n";

  return 0;
}
