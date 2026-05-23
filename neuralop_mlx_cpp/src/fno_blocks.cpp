#include "fno_blocks.h"
#include <cmath>

namespace neuralop {

FNOBlocks::FNOBlocks(int in_channels, int out_channels, int hidden_channels_,
                     int n_layers_, const std::vector<int>& n_modes,
                     bool use_channel_mlp_, float channel_mlp_expansion,
                     const std::string& fno_skip_type,
                     const std::string& mlp_skip_type,
                     const std::string& fft_norm,
                     const std::string& metallib_path)
    : n_layers(n_layers_), hidden_channels(hidden_channels_),
      use_channel_mlp(use_channel_mlp_)
{
  convs.reserve(n_layers);
  fno_skips.reserve(n_layers);
  if (use_channel_mlp) {
    channel_mlps.reserve(n_layers);
    mlp_skips.reserve(n_layers);
  }

  for (int i = 0; i < n_layers; ++i) {
    convs.emplace_back(in_channels, out_channels, n_modes,
                       /*use_bias=*/true, /*hermitian=*/true,
                       fft_norm, metallib_path);
    fno_skips.push_back(build_skip(fno_skip_type, in_channels, out_channels));

    if (use_channel_mlp) {
      int mlp_hidden = static_cast<int>(std::round(out_channels * channel_mlp_expansion));
      channel_mlps.emplace_back(out_channels, out_channels, mlp_hidden,
                                 /*n_layers=*/2);
      mlp_skips.push_back(build_skip(mlp_skip_type, in_channels, out_channels));
    }
  }
}

mx::array FNOBlocks::forward(const mx::array& x, int index) const {
  bool is_last = (index == n_layers - 1);

  // FNO sub-block: spectral conv + skip
  auto x_fno = mx::add(convs[index].forward(x),
                        apply_skip(fno_skips[index], x));
  auto h = is_last ? x_fno : gelu(x_fno);

  // Optional MLP sub-block
  if (use_channel_mlp) {
    // Note: mlp_skip uses original x (before fno sub-block), matching Python
    auto x_mlp = mx::add(channel_mlps[index].forward(h),
                          apply_skip(mlp_skips[index], x));
    h = is_last ? x_mlp : gelu(x_mlp);
  }

  return h;
}

} // namespace neuralop
