#include "fno.h"

#include <cmath>
#include <stdexcept>

namespace neuralop {

FNO::FNO(int in_channels_, int out_channels_, int hidden_channels_,
         int n_layers_, const std::vector<int>& n_modes,
         bool use_pos_emb, bool use_ch_mlp,
         float lifting_ratio, float proj_ratio, float mlp_exp,
         const std::string& fno_skip_, const std::string& ch_mlp_skip_,
         const std::string& fft_norm_, const std::string& metallib_path_)
    : in_channels(in_channels_), out_channels(out_channels_),
      hidden_channels(hidden_channels_), n_layers(n_layers_),
      use_positional_embedding(use_pos_emb), use_channel_mlp(use_ch_mlp),
      fno_skip(fno_skip_), channel_mlp_skip(ch_mlp_skip_),
      lifting_channel_ratio(lifting_ratio),
      projection_channel_ratio(proj_ratio),
      channel_mlp_expansion(mlp_exp),
      metallib_path(metallib_path_)
{
  int n_dim = static_cast<int>(n_modes.size());

  // Positional embedding appends coordinate grids, increasing in_channels
  int lift_in = in_channels;
  if (use_pos_emb) {
    // Default boundaries [0,1] for each spatial dim
    positional_embedding = GridEmbeddingND(n_dim);
    lift_in += n_dim;
  }

  // Lifting MLP: lift_in → lift_hidden → hidden
  int lift_hidden = static_cast<int>(std::round(hidden_channels * lifting_ratio));
  lifting = std::make_unique<ChannelMLP>(lift_in, hidden_channels,
                                          lift_hidden, /*n_layers=*/2);

  // FNO blocks
  blocks = std::make_unique<FNOBlocks>(
      hidden_channels, hidden_channels, hidden_channels,
      n_layers, n_modes,
      use_ch_mlp, mlp_exp,
      fno_skip_, ch_mlp_skip_,
      fft_norm_, metallib_path_);

  // Projection MLP: hidden → proj_hidden → out_channels
  int proj_hidden = static_cast<int>(std::round(out_channels * proj_ratio));
  projection = std::make_unique<ChannelMLP>(hidden_channels, out_channels,
                                             proj_hidden, /*n_layers=*/2);
}

mx::array FNO::forward(const mx::array& x) const {
  auto h = x;

  // 1. Positional embedding
  if (positional_embedding.has_value()) {
    h = positional_embedding->forward(h);
  }

  // 2. Lifting
  h = lifting->forward(h);

  // 3. FNO blocks
  for (int i = 0; i < n_layers; ++i) {
    h = blocks->forward(h, i);
  }

  // 4. Projection
  h = projection->forward(h);

  return h;
}

} // namespace neuralop
