// Internal layer types shared across the library.
// Not part of the public API.
#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace neuralop {

namespace mx = mlx::core;

// ── Inline activation ─────────────────────────────────────────────────────
// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
inline mx::array gelu(const mx::array& x) {
  constexpr float sqrt2_inv = 0.70710678118f;  // 1/sqrt(2)
  return mx::multiply(
    x,
    mx::multiply(
      mx::array(0.5f),
      mx::add(
        mx::ones_like(x),
        mx::erf(mx::multiply(x, mx::array(sqrt2_inv)))
      )
    )
  );
}

// ── Linear (channels-last matmul) ─────────────────────────────────────────
// Equivalent to nn.Linear: y = x @ weight.T + bias
// weight: (out_features, in_features)
// bias:   (out_features,)
// x:      (..., in_features)   →   y: (..., out_features)
struct Linear {
  mx::array weight;    // (out, in)
  mx::array bias;      // (out,) — may be uninitialized (check with bias.size()==0)
  bool      has_bias = false;

  Linear() = delete;
  Linear(int in_features, int out_features, bool use_bias = true);

  mx::array forward(const mx::array& x) const;
};

// ── ChannelMLP ────────────────────────────────────────────────────────────
// Point-wise MLP along the channel axis.
// Input/output shape: (batch, channels, d1, ..., dN)
// Internally flattens spatial dims, transposes to channels-last, applies
// Linear+gelu chain, transposes back, restores spatial shape.
struct ChannelMLP {
  std::vector<Linear> fcs;
  int n_layers    = 0;
  int out_channels = 0;

  ChannelMLP() = default;
  ChannelMLP(int in_channels, int out_channels, int hidden_channels,
             int n_layers, bool use_bias = true);

  mx::array forward(const mx::array& x) const;
};

// ── Skip connections ──────────────────────────────────────────────────────
struct LinearSkip {
  Linear linear;
  LinearSkip() = delete;
  explicit LinearSkip(int in_ch, int out_ch);
  mx::array forward(const mx::array& x) const;
};

struct SoftGatingSkip {
  mx::array alpha;  // (channels,)  — element-wise per-channel gate
  int channels = 0;
  SoftGatingSkip() = delete;
  explicit SoftGatingSkip(int channels);
  mx::array forward(const mx::array& x) const;
};

struct IdentitySkip {
  mx::array forward(const mx::array& x) const { return x; }
};

using Skip = std::variant<LinearSkip, SoftGatingSkip, IdentitySkip>;

// Apply whichever skip type is active
mx::array apply_skip(const Skip& s, const mx::array& x);

// Factory matching Python's build_skip()
Skip build_skip(const std::string& skip_type, int in_ch, int out_ch);

// ── GridEmbeddingND ────────────────────────────────────────────────────────
// Appends N coordinate grids to the input along the channel axis.
// Input:  (batch, in_channels, d0, ..., dN-1)
// Output: (batch, in_channels + dim, d0, ..., dN-1)
struct GridEmbeddingND {
  int dim = 0;
  std::vector<std::pair<float, float>> boundaries;

  // Cache
  mutable std::vector<int> cached_spatial;
  mutable mx::array        cached_grid;   // (batch=1, dim, d0, ..., dN-1)

  GridEmbeddingND() = delete;
  GridEmbeddingND(int dim,
                  std::vector<std::pair<float, float>> bounds = {});

  mx::array forward(const mx::array& data) const;

 private:
  void build_cache(const std::vector<int>& spatial_dims) const;
};

// ── τ/η sinusoidal coordinate embeddings ──────────────────────────────────
// Designed for Milne-coordinate FNO surrogates where NormFn3d normalisation
// removes the dominant amplitude cues along proper-time and rapidity axes.

// 2-channel sinusoidal embedding of the proper-time axis.
// tau_vals : physical τ values, length T  [fm/c]
// tau_max  : reference scale             [fm/c]
// Returns  : (T, 2) array, columns [sin(π·τ/τ_max), cos(π·τ/τ_max)]
mx::array sinusoidal_tau_embedding(const std::vector<float>& tau_vals,
                                    float tau_max);

// 2-channel sinusoidal embedding of the spacetime-rapidity axis.
// eta_vals : physical η_s values, length Z
// eta_max  : |η| range used during training
// Returns  : (Z, 2) array, columns [sin(π·η/η_max), cos(π·η/η_max)]
mx::array sinusoidal_eta_embedding(const std::vector<float>& eta_vals,
                                    float eta_max);

// Per-FNO-block learned τ-embedding injection.
// Projects (T, 2) → (1, C, 1, 1, 1, T) and adds to (B, C, X, Y, Z, T).
struct TauBias {
  Linear proj;  // (C, 2)  no bias

  TauBias() = delete;
  explicit TauBias(int channels);

  // x       : (B, C, X, Y, Z, T)
  // tau_emb : (T, 2)
  // Returns : (B, C, X, Y, Z, T)
  mx::array forward(const mx::array& x, const mx::array& tau_emb) const;
};

// Per-FNO-block learned η-embedding injection.
// Projects (Z, 2) → (1, C, 1, 1, Z, 1) and adds to (B, C, X, Y, Z, T).
struct EtaBias {
  Linear proj;  // (C, 2)  no bias

  EtaBias() = delete;
  explicit EtaBias(int channels);

  // x       : (B, C, X, Y, Z, T)
  // eta_emb : (Z, 2)
  // Returns : (B, C, X, Y, Z, T)
  mx::array forward(const mx::array& x, const mx::array& eta_emb) const;
};

} // namespace neuralop
