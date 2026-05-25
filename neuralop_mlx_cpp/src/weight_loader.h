// weight_loader.h — loads safetensors weights into FNO structs.
// Handles two formats:
//   MLX-format: produced by tools/export_weights.py
//   PyTorch-format: saved by safetensors.torch.save_file (or torch.save)
#pragma once

#include <string>
#include <unordered_map>
#include <mlx/mlx.h>

#include "fno.h"
#include "fno_tau_eta.h"

namespace neuralop {

namespace mx = mlx::core;

using WeightMap = std::unordered_map<std::string, mx::array>;

// Auto-detects format and loads all weights into model
void load_weights(FNO& model, const std::string& path);

// Lower-level: load from a pre-parsed weight map
void load_weights_from_map(FNO& model, const WeightMap& W, bool is_pt_format);

// FNOTauEta variants — same format detection, also reads tau_max/eta_max scalars
void load_weights(FNOTauEta& model, const std::string& path);
void load_weights_from_map(FNOTauEta& model, const WeightMap& W, bool is_pt_format);

} // namespace neuralop
