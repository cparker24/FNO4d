// FNOConfig::from_json — parses the JSON file produced by tools/export_weights.py
#include "neuralop_mlx_cpp/fno_inference.h"

#include <cctype>
#include <fstream>
#include <sstream>
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

// ── from_yaml ───────────────────────────────────────────────────────────────
// Minimal flat-YAML parser — handles the key: value and key: [a, b, c] forms
// used by the training configs.  No external library required.

static std::string _trim(const std::string& s) {
  const char* ws = " \t\r\n";
  auto b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  auto e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

static std::vector<int> _parse_int_list(const std::string& s) {
  // Expects inline YAML list:  [32, 32, 16, 16]
  std::vector<int> result;
  auto lb = s.find('[');
  auto rb = s.find(']');
  if (lb == std::string::npos || rb == std::string::npos) return result;
  std::istringstream ss(s.substr(lb + 1, rb - lb - 1));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = _trim(tok);
    if (!tok.empty()) result.push_back(std::stoi(tok));
  }
  return result;
}

FNOConfig FNOConfig::from_yaml(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config file: " + path);

  FNOConfig cfg;  // all fields initialised to defaults

  std::string line;
  while (std::getline(f, line)) {
    // Strip inline comments
    auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);

    auto colon = line.find(':');
    if (colon == std::string::npos) continue;

    std::string key = _trim(line.substr(0, colon));
    std::string val = _trim(line.substr(colon + 1));
    if (key.empty() || val.empty()) continue;

    try {
      if      (key == "n_modes")                  cfg.n_modes                  = _parse_int_list(val);
      else if (key == "in_channels")              cfg.in_channels              = std::stoi(val);
      else if (key == "out_channels")             cfg.out_channels             = std::stoi(val);
      else if (key == "hidden_channels")          cfg.hidden_channels          = std::stoi(val);
      else if (key == "n_layers")                 cfg.n_layers                 = std::stoi(val);
      else if (key == "lifting_channel_ratio")    cfg.lifting_channel_ratio    = std::stof(val);
      else if (key == "projection_channel_ratio") cfg.projection_channel_ratio = std::stof(val);
      else if (key == "channel_mlp_expansion")    cfg.channel_mlp_expansion    = std::stof(val);
      else if (key == "fno_skip")                 cfg.fno_skip                 = val;
      else if (key == "channel_mlp_skip")         cfg.channel_mlp_skip         = val;
      else if (key == "fft_norm")                 cfg.fft_norm                 = val;
      else if (key == "metallib_path")            cfg.metallib_path            = val;
      else if (key == "positional_embedding")     cfg.positional_embedding     = (val == "true" || val == "True" || val == "1" || val == "yes");
      else if (key == "use_channel_mlp")          cfg.use_channel_mlp          = (val == "true" || val == "True" || val == "1" || val == "yes");
      // Unknown keys are silently ignored
    } catch (const std::exception&) {
      throw std::runtime_error("YAML parse error on key '" + key + "' value '" + val + "'");
    }
  }

  if (cfg.n_modes.empty())
    throw std::runtime_error("YAML config missing required key: n_modes");

  return cfg;
}

// ── from_file ────────────────────────────────────────────────────────────────
FNOConfig FNOConfig::from_file(const std::string& path) {
  // Dispatch on extension: .json → from_json, anything else → from_yaml
  auto dot = path.rfind('.');
  if (dot != std::string::npos) {
    std::string ext = path.substr(dot + 1);
    // lowercase
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == "json") return from_json(path);
  }
  return from_yaml(path);
}

} // namespace neuralop
