#include "fno_tau_eta.h"

namespace neuralop {

FNOTauEta::FNOTauEta(
    int in_channels,
    int out_channels,
    int hidden_channels,
    int n_layers,
    const std::vector<int>& n_modes,
    float tau_max_,
    float eta_max_,
    bool use_pos_emb,
    bool use_ch_mlp,
    float lift_ratio,
    float proj_ratio,
    float mlp_exp,
    const std::string& fno_skip_,
    const std::string& ch_mlp_skip_,
    const std::string& fft_norm_,
    const std::string& metallib_path_)
    : fno(in_channels + 4,   // +4 for 2 τ channels + 2 η channels
          out_channels,
          hidden_channels,
          n_layers,
          n_modes,
          use_pos_emb,
          use_ch_mlp,
          lift_ratio,
          proj_ratio,
          mlp_exp,
          fno_skip_,
          ch_mlp_skip_,
          fft_norm_,
          metallib_path_),
      tau_max(tau_max_),
      eta_max(eta_max_)
{
  tau_biases.reserve(n_layers);
  eta_biases.reserve(n_layers);
  for (int i = 0; i < n_layers; ++i) {
    tau_biases.emplace_back(hidden_channels);
    eta_biases.emplace_back(hidden_channels);
  }
}

mx::array FNOTauEta::forward(const mx::array& x,
                              const std::vector<float>& tau_vals,
                              const std::vector<float>& eta_vals) const {
  // x: (B, C_in, X, Y, Z, T)
  auto shape = x.shape();
  int B = shape[0];
  int C = shape[1];
  int X = shape[2];
  int Y = shape[3];
  int Z = shape[4];
  int T = shape[5];

  // Build sinusoidal embeddings.
  auto tau_emb = sinusoidal_tau_embedding(tau_vals, tau_max);  // (T, 2)
  auto eta_emb = sinusoidal_eta_embedding(eta_vals, eta_max);  // (Z, 2)

  // Broadcast τ embedding channels: (T, 2) → (B, 2, X, Y, Z, T)
  // tau_emb.T → (2, T) → reshape (1, 2, 1, 1, 1, T) → broadcast
  auto tau_ch = mx::transpose(tau_emb);               // (2, T)
  tau_ch = mx::reshape(tau_ch, {1, 2, 1, 1, 1, T});
  tau_ch = mx::broadcast_to(tau_ch, {B, 2, X, Y, Z, T});

  // Broadcast η embedding channels: (Z, 2) → (B, 2, X, Y, Z, T)
  auto eta_ch = mx::transpose(eta_emb);               // (2, Z)
  eta_ch = mx::reshape(eta_ch, {1, 2, 1, 1, Z, 1});
  eta_ch = mx::broadcast_to(eta_ch, {B, 2, X, Y, Z, T});

  // Concatenate along channel axis → (B, C_in+4, X, Y, Z, T)
  auto x_aug = mx::concatenate({x, tau_ch, eta_ch}, /*axis=*/1);

  // Run FNO internals manually with per-block bias injection.
  auto h = x_aug;

  if (fno.positional_embedding.has_value()) {
    h = fno.positional_embedding->forward(h);
  }

  h = fno.lifting->forward(h);

  for (int i = 0; i < fno.n_layers; ++i) {
    h = fno.blocks->forward(h, i);
    h = tau_biases[i].forward(h, tau_emb);
    h = eta_biases[i].forward(h, eta_emb);
  }

  return fno.projection->forward(h);
}

} // namespace neuralop
