// FNOConfig::from_json — parses the JSON file produced by tools/export_weights.py
#include "neuralop_mlx_cpp/fno_inference.h"

#include <fstream>
#include <stdexcept>
#include <json.hpp>

namespace neuralop {

FNOConfig FNOConfig::from_json(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config file: " + path);

  auto j = nlohmann::json::parse(f);

  FNOConfig cfg;

  if (!j.contains("n_modes"))
    throw std::runtime_error("Config missing required field: n_modes");
  cfg.n_modes = j["n_modes"].get<std::vector<int>>();

  if (j.contains("in_channels"))          cfg.in_channels          = j["in_channels"].get<int>();
  if (j.contains("out_channels"))         cfg.out_channels         = j["out_channels"].get<int>();
  if (j.contains("hidden_channels"))      cfg.hidden_channels      = j["hidden_channels"].get<int>();
  if (j.contains("n_layers"))             cfg.n_layers             = j["n_layers"].get<int>();
  if (j.contains("lifting_channel_ratio"))    cfg.lifting_channel_ratio    = j["lifting_channel_ratio"].get<float>();
  if (j.contains("projection_channel_ratio")) cfg.projection_channel_ratio = j["projection_channel_ratio"].get<float>();
  if (j.contains("positional_embedding"))     cfg.positional_embedding     = j["positional_embedding"].get<bool>();
  if (j.contains("use_channel_mlp"))          cfg.use_channel_mlp          = j["use_channel_mlp"].get<bool>();
  if (j.contains("channel_mlp_expansion"))    cfg.channel_mlp_expansion    = j["channel_mlp_expansion"].get<float>();
  if (j.contains("fno_skip"))                 cfg.fno_skip                 = j["fno_skip"].get<std::string>();
  if (j.contains("channel_mlp_skip"))         cfg.channel_mlp_skip         = j["channel_mlp_skip"].get<std::string>();
  if (j.contains("fft_norm"))                 cfg.fft_norm                 = j["fft_norm"].get<std::string>();
  if (j.contains("metallib_path"))            cfg.metallib_path            = j["metallib_path"].get<std::string>();

  return cfg;
}

} // namespace neuralop
