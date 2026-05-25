// Weight loading for FNO from safetensors files.
//
// Key name mappings (PyTorch safetensors → MLX internal):
//
//   PyTorch key                              │ MLX key / treatment
//  ─────────────────────────────────────────┼──────────────────────────────────
//  fno_blocks.convs.i.weight.tensor         │ split: real→weight_r, imag→weight_i
//  fno_blocks.fno_skips.i.conv.weight       │ squeeze last dim → fno_skips[i].linear.weight
//  fno_blocks.channel_mlp.i.fcs.j.weight   │ squeeze last dim → channel_mlps[i].fcs[j].weight
//  fno_blocks.channel_mlp.i.fcs.j.bias     │ → channel_mlps[i].fcs[j].bias
//  fno_blocks.channel_mlp_skips.i.weight    │ flatten → mlp_skips[i].alpha
//  lifting.fcs.j.weight                     │ squeeze last dim → lifting.fcs[j].weight
//  lifting.fcs.j.bias                       │ → lifting.fcs[j].bias
//  projection.fcs.j.weight                  │ squeeze last dim → projection.fcs[j].weight
//  projection.fcs.j.bias                    │ → projection.fcs[j].bias
//
// MLX safetensors key names (produced by export_weights.py):
//  fno_blocks.convs.i.weight_r / .weight_i / .bias
//  fno_blocks.fno_skips.i.linear.weight / .bias
//  fno_blocks.channel_mlps.i.fcs.j.weight / .bias
//  fno_blocks.mlp_skips.i.alpha
//  lifting.fcs.j.weight / .bias
//  projection.fcs.j.weight / .bias

#include "weight_loader.h"
#include "fno_tau_eta.h"

#include <stdexcept>
#include <string>

namespace neuralop {

// ── Helpers ─────────────────────────────────────────────────────────────────

static const mx::array& require(const WeightMap& W, const std::string& key) {
  auto it = W.find(key);
  if (it == W.end())
    throw std::runtime_error("Missing weight key: " + key);
  return it->second;
}

static const mx::array* find(const WeightMap& W, const std::string& key) {
  auto it = W.find(key);
  return (it != W.end()) ? &it->second : nullptr;
}

// Squeeze last dimension (e.g. (out,in,1) → (out,in))
static mx::array squeeze_last(const mx::array& a) {
  int n = a.ndim();
  if (n > 0 && a.shape(n - 1) == 1)
    return mx::squeeze(a, n - 1);
  return a;
}

// Assign weight into Linear, squeezing trailing size-1 dim if needed
static void assign_linear(Linear& fc, const WeightMap& W,
                           const std::string& w_key,
                           const std::string& b_key) {
  fc.weight = squeeze_last(require(W, w_key));
  if (auto* b = find(W, b_key)) {
    fc.bias     = *b;
    fc.has_bias = true;
  }
}

// ── Load MLX-format weights ──────────────────────────────────────────────────

static void load_mlx(FNO& model, const WeightMap& W) {
  int n_layers = model.n_layers;

  // lifting
  for (int j = 0; j < (int)model.lifting->fcs.size(); ++j) {
    std::string prefix = "lifting.fcs." + std::to_string(j) + ".";
    assign_linear(model.lifting->fcs[j], W,
                  prefix + "weight", prefix + "bias");
  }

  // projection
  for (int j = 0; j < (int)model.projection->fcs.size(); ++j) {
    std::string prefix = "projection.fcs." + std::to_string(j) + ".";
    assign_linear(model.projection->fcs[j], W,
                  prefix + "weight", prefix + "bias");
  }

  // fno_blocks
  for (int i = 0; i < n_layers; ++i) {
    std::string ci = "fno_blocks.convs." + std::to_string(i) + ".";
    model.blocks->convs[i].weight_r = require(W, ci + "weight_r");
    model.blocks->convs[i].weight_i = require(W, ci + "weight_i");
    if (auto* b = find(W, ci + "bias")) {
      model.blocks->convs[i].bias     = *b;
      model.blocks->convs[i].has_bias = true;
    }

    // fno_skip (linear type expected)
    std::string si = "fno_blocks.fno_skips." + std::to_string(i) + ".linear.";
    if (auto* sk = std::get_if<LinearSkip>(&model.blocks->fno_skips[i])) {
      assign_linear(sk->linear, W, si + "weight", si + "bias");
    }

    // channel_mlp and mlp_skip
    if (model.use_channel_mlp) {
      for (int j = 0; j < (int)model.blocks->channel_mlps[i].fcs.size(); ++j) {
        std::string mp = "fno_blocks.channel_mlps." + std::to_string(i)
                       + ".fcs." + std::to_string(j) + ".";
        assign_linear(model.blocks->channel_mlps[i].fcs[j], W,
                      mp + "weight", mp + "bias");
      }
      std::string ap = "fno_blocks.mlp_skips." + std::to_string(i) + ".alpha";
      if (auto* sg = std::get_if<SoftGatingSkip>(&model.blocks->mlp_skips[i])) {
        sg->alpha = require(W, ap);
      }
    }
  }
}

// ── Load PyTorch-format weights ──────────────────────────────────────────────

static void load_pytorch(FNO& model, const WeightMap& W) {
  int n_layers = model.n_layers;

  // lifting
  for (int j = 0; j < (int)model.lifting->fcs.size(); ++j) {
    std::string prefix = "lifting.fcs." + std::to_string(j) + ".";
    assign_linear(model.lifting->fcs[j], W,
                  prefix + "weight", prefix + "bias");
  }

  // projection
  for (int j = 0; j < (int)model.projection->fcs.size(); ++j) {
    std::string prefix = "projection.fcs." + std::to_string(j) + ".";
    assign_linear(model.projection->fcs[j], W,
                  prefix + "weight", prefix + "bias");
  }

  // fno_blocks
  for (int i = 0; i < n_layers; ++i) {
    // Spectral weights: complex64 → split real/imag
    std::string ci = "fno_blocks.convs." + std::to_string(i)
                   + ".weight.tensor";
    const auto& cw = require(W, ci);
    model.blocks->convs[i].weight_r = mx::real(cw);
    model.blocks->convs[i].weight_i = mx::imag(cw);
    if (auto* b = find(W, "fno_blocks.convs." + std::to_string(i) + ".bias")) {
      model.blocks->convs[i].bias     = *b;
      model.blocks->convs[i].has_bias = true;
    }

    // fno_skip: PyTorch uses .conv.weight (out,in,1) for 1-D conv
    std::string fsi = "fno_blocks.fno_skips." + std::to_string(i) + ".";
    // try .conv.weight first, fall back to .linear.weight
    auto* fsw = find(W, fsi + "conv.weight");
    if (!fsw) fsw = find(W, fsi + "linear.weight");
    if (fsw) {
      if (auto* sk = std::get_if<LinearSkip>(&model.blocks->fno_skips[i])) {
        sk->linear.weight  = squeeze_last(*fsw);
        sk->linear.has_bias = false;
        if (auto* b = find(W, fsi + "conv.bias"))
          { sk->linear.bias = *b; sk->linear.has_bias = true; }
        else if (auto* b = find(W, fsi + "linear.bias"))
          { sk->linear.bias = *b; sk->linear.has_bias = true; }
      }
    }

    // channel_mlp: PyTorch uses "channel_mlp" key (not "channel_mlps")
    if (model.use_channel_mlp) {
      for (int j = 0; j < (int)model.blocks->channel_mlps[i].fcs.size(); ++j) {
        std::string mp = "fno_blocks.channel_mlp." + std::to_string(i)
                       + ".fcs." + std::to_string(j) + ".";
        auto& fc = model.blocks->channel_mlps[i].fcs[j];
        if (auto* w = find(W, mp + "weight")) fc.weight  = squeeze_last(*w);
        if (auto* b = find(W, mp + "bias"))   { fc.bias = *b; fc.has_bias = true; }
      }

      // mlp_skip: PyTorch uses channel_mlp_skips.i.weight (1,C,1,...,1)
      std::string sp = "fno_blocks.channel_mlp_skips." + std::to_string(i) + ".weight";
      if (auto* sg = std::get_if<SoftGatingSkip>(&model.blocks->mlp_skips[i])) {
        if (auto* w = find(W, sp)) {
          sg->alpha = mx::flatten(*w);
        }
      }
    }
  }
}

// ── Public API ───────────────────────────────────────────────────────────────

void load_weights_from_map(FNO& model, const WeightMap& W, bool is_pt_format) {
  if (is_pt_format)
    load_pytorch(model, W);
  else
    load_mlx(model, W);
}

void load_weights(FNO& model, const std::string& path) {
  auto [W, meta] = mx::load_safetensors(path);

  // Auto-detect format: MLX format has weight_r / weight_i keys
  bool is_mlx = (W.count("fno_blocks.convs.0.weight_r") > 0);
  bool is_pt  = (W.count("fno_blocks.convs.0.weight.tensor") > 0);

  if (!is_mlx && !is_pt) {
    throw std::runtime_error(
        "Cannot detect weight format in '" + path +
        "'. Expected key 'fno_blocks.convs.0.weight_r' (MLX) or "
        "'fno_blocks.convs.0.weight.tensor' (PyTorch).");
  }

  load_weights_from_map(model, W, /*is_pt_format=*/is_pt);
}

// ── FNOTauEta weight loading ─────────────────────────────────────────────────
//
// Key layout written by the MLX trainer checkpoint():
//
//   MLX keys for base FNO internals (produced by mlx.utils.tree_flatten):
//     _fno.lifting.fcs.{j}.weight / .bias
//     _fno.fno_blocks.convs.{i}.weight_r / .weight_i / .bias
//     ... (same pattern as FNO, prefixed with "_fno.")
//
//   Per-block bias keys:
//     tau_biases.{i}.proj.weight
//     eta_biases.{i}.proj.weight
//
//   Scalars:
//     _tau_max   (rank-0 float32)
//     _eta_max   (rank-0 float32)
//
// PyTorch safetensors keys use "fno." prefix instead of "_fno.".

static WeightMap strip_prefix(const WeightMap& W, const std::string& prefix) {
  WeightMap out;
  for (const auto& [k, v] : W) {
    if (k.size() > prefix.size() && k.substr(0, prefix.size()) == prefix)
      out[k.substr(prefix.size())] = v;
  }
  return out;
}

void load_weights_from_map(FNOTauEta& model, const WeightMap& W,
                            bool is_pt_format) {
  // Load tau_max / eta_max scalars if present.
  if (auto it = W.find("_tau_max"); it != W.end())
    model.tau_max = mx::item<float>(it->second);
  if (auto it = W.find("_eta_max"); it != W.end())
    model.eta_max = mx::item<float>(it->second);

  // Strip the FNO sub-model prefix and delegate to the existing FNO loader.
  // MLX format uses "_fno." prefix; PyTorch format uses "fno." prefix.
  const std::string fno_prefix = is_pt_format ? "fno." : "_fno.";
  auto fno_W = strip_prefix(W, fno_prefix);
  if (!fno_W.empty()) {
    load_weights_from_map(model.fno, fno_W, is_pt_format);
  }

  // Load per-block bias weights.
  for (int i = 0; i < model.fno.n_layers; ++i) {
    std::string tp = "tau_biases." + std::to_string(i) + ".proj.";
    if (auto* w = find(W, tp + "weight"))
      model.tau_biases[i].proj.weight = *w;

    std::string ep = "eta_biases." + std::to_string(i) + ".proj.";
    if (auto* w = find(W, ep + "weight"))
      model.eta_biases[i].proj.weight = *w;
  }
}

void load_weights(FNOTauEta& model, const std::string& path) {
  auto [W, meta] = mx::load_safetensors(path);

  // Detect format by looking for FNO sub-model keys.
  bool is_mlx = (W.count("_fno.fno_blocks.convs.0.weight_r") > 0
              || W.count("fno_blocks.convs.0.weight_r") > 0);
  bool is_pt  = (W.count("fno.fno_blocks.convs.0.weight.tensor") > 0
              || W.count("fno_blocks.convs.0.weight.tensor") > 0);

  if (!is_mlx && !is_pt) {
    throw std::runtime_error(
        "Cannot detect FNOTauEta weight format in '" + path + "'.");
  }

  load_weights_from_map(model, W, /*is_pt_format=*/is_pt);
}

} // namespace neuralop
