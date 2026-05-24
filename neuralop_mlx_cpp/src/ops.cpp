#include "ops.h"

#include <cmath>
#include <stdexcept>
#include <numeric>

namespace neuralop {

// ── Linear ─────────────────────────────────────────────────────────────────

Linear::Linear(int in_features, int out_features, bool use_bias)
    : has_bias(use_bias),
      weight(mx::zeros({out_features, in_features})),
      bias(use_bias ? mx::zeros({out_features}) : mx::array(0.0f))
{}

mx::array Linear::forward(const mx::array& x) const {
  // weight: (out, in)  →  matmul(x, weight.T)  →  (..., out)
  auto wT = mx::transpose(weight);   // (in, out)
  auto y  = mx::matmul(x, wT);
  if (has_bias) {
    y = mx::add(y, bias);
  }
  return y;
}

// ── ChannelMLP ─────────────────────────────────────────────────────────────

ChannelMLP::ChannelMLP(int in_channels, int out_channels_,
                       int hidden_channels, int n_layers_, bool use_bias)
    : n_layers(n_layers_), out_channels(out_channels_)
{
  // dims: in_channels → hidden … → out_channels
  std::vector<int> dims;
  dims.push_back(in_channels);
  for (int i = 0; i < n_layers - 1; ++i) dims.push_back(hidden_channels);
  dims.push_back(out_channels_);

  fcs.reserve(dims.size() - 1);
  for (int i = 0; i < (int)dims.size() - 1; ++i) {
    fcs.emplace_back(dims[i], dims[i + 1], use_bias);
  }
}

mx::array ChannelMLP::forward(const mx::array& x) const {
  // x shape: (B, C, d0, ..., dN-1)
  int ndim = x.ndim();
  auto orig_shape = x.shape();

  mx::array h = x;

  // Flatten spatial dims → (B, C, prod_spatial)
  if (ndim > 3) {
    int B = orig_shape[0];
    int C = orig_shape[1];
    int prod_spatial = 1;
    for (int i = 2; i < ndim; ++i) prod_spatial *= orig_shape[i];
    h = mx::reshape(h, {B, C, prod_spatial});
  }

  // Transpose to channels-last: (B, prod_spatial, C)
  h = mx::transpose(h, {0, 2, 1});

  // Apply fc layers
  for (int i = 0; i < (int)fcs.size(); ++i) {
    h = fcs[i].forward(h);
    if (i < (int)fcs.size() - 1) {
      h = gelu(h);
    }
  }

  // Transpose back: (B, C_out, prod_spatial)
  h = mx::transpose(h, {0, 2, 1});

  // Restore spatial shape if needed
  if (ndim > 3) {
    mx::Shape out_shape = {orig_shape[0], (mx::ShapeElem)out_channels};
    for (int i = 2; i < ndim; ++i) out_shape.push_back(orig_shape[i]);
    h = mx::reshape(h, out_shape);
  }

  return h;
}

// ── Skip connections ───────────────────────────────────────────────────────

LinearSkip::LinearSkip(int in_ch, int out_ch) : linear(in_ch, out_ch, /*bias=*/true) {}

mx::array LinearSkip::forward(const mx::array& x) const {
  // x: (B, C, d0, ..., dN-1)  — apply channel-wise linear
  int ndim = x.ndim();
  auto orig_shape = x.shape();
  int B  = orig_shape[0];
  int Ci = orig_shape[1];
  int prod_spatial = 1;
  for (int i = 2; i < ndim; ++i) prod_spatial *= orig_shape[i];

  auto h = mx::reshape(x, {B, Ci, prod_spatial});
  h = mx::transpose(h, {0, 2, 1});      // (B, prod_s, Ci)
  h = linear.forward(h);                 // (B, prod_s, Co)
  h = mx::transpose(h, {0, 2, 1});      // (B, Co, prod_s)

  mx::Shape out_shape = {(mx::ShapeElem)B, linear.weight.shape()[0]};
  for (int i = 2; i < ndim; ++i) out_shape.push_back(orig_shape[i]);
  return mx::reshape(h, out_shape);
}

SoftGatingSkip::SoftGatingSkip(int channels_)
    : channels(channels_),
      alpha(mx::zeros({channels_}))
{}

mx::array SoftGatingSkip::forward(const mx::array& x) const {
  // alpha: (C,) → reshape to (1, C, 1, ..., 1) and multiply
  int ndim = x.ndim();
  mx::Shape a_shape(ndim, 1);
  a_shape[1] = channels;
  auto a = mx::reshape(alpha, a_shape);
  return mx::multiply(a, x);
}

mx::array apply_skip(const Skip& s, const mx::array& x) {
  return std::visit([&x](const auto& sk) { return sk.forward(x); }, s);
}

Skip build_skip(const std::string& skip_type, int in_ch, int out_ch) {
  if (skip_type == "linear")       return LinearSkip(in_ch, out_ch);
  if (skip_type == "soft-gating")  return SoftGatingSkip(in_ch);
  if (skip_type == "identity")     return IdentitySkip{};
  throw std::invalid_argument("Unknown skip type: " + skip_type);
}

// ── GridEmbeddingND ────────────────────────────────────────────────────────

GridEmbeddingND::GridEmbeddingND(int dim_,
                                  std::vector<std::pair<float,float>> bounds)
    : dim(dim_), boundaries(std::move(bounds)),
      cached_grid(mx::array(0.0f))  // placeholder; rebuilt on first forward()
{
  // If boundaries not specified, default to [0, 1] for every dim
  while ((int)boundaries.size() < dim) {
    boundaries.push_back({0.f, 1.f});
  }
}

void GridEmbeddingND::build_cache(const std::vector<int>& spatial_dims) const {
  // Build grid of shape (dim, d0, d1, ..., dN-1) then expand batch dim
  std::vector<mx::array> grids;
  grids.reserve(dim);

  for (int i = 0; i < dim; ++i) {
    int n    = spatial_dims[i];
    float lo = boundaries[i].first;
    float hi = boundaries[i].second;

    // linspace [lo, hi] with n points
    std::vector<float> vals(n);
    if (n == 1) {
      vals[0] = lo;
    } else {
      float step = (hi - lo) / (n - 1);
      for (int j = 0; j < n; ++j) vals[j] = lo + step * j;
    }
    auto g = mx::array(vals.data(), {n}, mx::float32);

    // Reshape so it broadcasts across all spatial dimensions
    // For dim i, shape should be (1, 1, ..., n_i, ..., 1) with n_i at position i
    mx::Shape gshape(dim, 1);
    gshape[i] = n;
    g = mx::reshape(g, gshape);

    // Broadcast to full spatial shape
    mx::Shape full_shape(spatial_dims.begin(), spatial_dims.end());
    g = mx::broadcast_to(g, full_shape);

    // Add batch and channel dims: (1, 1, d0, ..., dN-1)
    g = mx::expand_dims(g, 0);  // (1, d0, ..., dN-1)
    g = mx::expand_dims(g, 0);  // (1, 1, d0, ..., dN-1)
    grids.push_back(g);
  }

  // Concatenate along channel axis → (1, dim, d0, ..., dN-1)
  cached_grid    = mx::concatenate(grids, 1);
  cached_spatial = spatial_dims;
}

mx::array GridEmbeddingND::forward(const mx::array& data) const {
  // data: (B, C, d0, ..., dN-1)
  int ndim = data.ndim();
  int B    = data.shape()[0];
  std::vector<int> spatial_dims(data.shape().begin() + 2, data.shape().end());

  if (cached_spatial != spatial_dims) {
    build_cache(spatial_dims);
  }

  // Expand batch dimension of cached grid
  mx::Shape bcast_shape = {B};
  for (int i = 1; i < cached_grid.ndim(); ++i)
    bcast_shape.push_back(cached_grid.shape()[i]);
  auto grid = mx::broadcast_to(cached_grid, bcast_shape);

  return mx::concatenate({data, grid}, 1);
}

// ── τ/η sinusoidal coordinate embeddings ──────────────────────────────────

mx::array sinusoidal_tau_embedding(const std::vector<float>& tau_vals,
                                    float tau_max) {
  int T = static_cast<int>(tau_vals.size());
  std::vector<float> sin_v(T), cos_v(T);
  constexpr float pi = 3.14159265358979323846f;
  for (int i = 0; i < T; ++i) {
    float t = tau_vals[i] / tau_max;
    sin_v[i] = std::sin(pi * t);
    cos_v[i] = std::cos(pi * t);
  }
  auto s = mx::array(sin_v.data(), {T}, mx::float32);
  auto c = mx::array(cos_v.data(), {T}, mx::float32);
  // Stack along last axis → (T, 2)
  return mx::stack({s, c}, /*axis=*/1);
}

mx::array sinusoidal_eta_embedding(const std::vector<float>& eta_vals,
                                    float eta_max) {
  int Z = static_cast<int>(eta_vals.size());
  std::vector<float> sin_v(Z), cos_v(Z);
  constexpr float pi = 3.14159265358979323846f;
  for (int i = 0; i < Z; ++i) {
    float e = eta_vals[i] / eta_max;
    sin_v[i] = std::sin(pi * e);
    cos_v[i] = std::cos(pi * e);
  }
  auto s = mx::array(sin_v.data(), {Z}, mx::float32);
  auto c = mx::array(cos_v.data(), {Z}, mx::float32);
  return mx::stack({s, c}, /*axis=*/1);
}

// ── TauBias ────────────────────────────────────────────────────────────────

TauBias::TauBias(int channels)
    : proj(2, channels, /*use_bias=*/false)
{}

mx::array TauBias::forward(const mx::array& x, const mx::array& tau_emb) const {
  // tau_emb: (T, 2)
  int T = tau_emb.shape()[0];
  int C = x.shape()[1];
  // Project: (T, 2) × weight.T(2, C) → (T, C)
  auto bias = proj.forward(tau_emb);          // (T, C)
  bias = mx::transpose(bias);                  // (C, T)
  bias = mx::reshape(bias, {1, C, 1, 1, 1, T});
  return mx::add(x, bias);
}

// ── EtaBias ────────────────────────────────────────────────────────────────

EtaBias::EtaBias(int channels)
    : proj(2, channels, /*use_bias=*/false)
{}

mx::array EtaBias::forward(const mx::array& x, const mx::array& eta_emb) const {
  // eta_emb: (Z, 2)
  int Z = eta_emb.shape()[0];
  int C = x.shape()[1];
  auto bias = proj.forward(eta_emb);          // (Z, C)
  bias = mx::transpose(bias);                  // (C, Z)
  bias = mx::reshape(bias, {1, C, 1, 1, Z, 1});
  return mx::add(x, bias);
}

} // namespace neuralop
