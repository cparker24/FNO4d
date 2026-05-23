// SpectralConvND forward pass.
// See spectral_conv.h for an overview.  This closely follows
// neuralop_mlx/layers/spectral_convolution.py SpectralConvolution.forward().

#include "spectral_conv.h"
#include "spectral_contract.h"
#include "ops.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace neuralop {

// ── Constructor ─────────────────────────────────────────────────────────────

// Helper: compute weight shape from n_modes (last dim halved for rFFT)
static mx::Shape _wshape(int in_ch, int out_ch, const std::vector<int>& nm) {
  mx::Shape s = {in_ch, out_ch};
  for (int i = 0; i + 1 < (int)nm.size(); ++i) s.push_back(nm[i]);
  s.push_back(nm.back() / 2 + 1);
  return s;
}

SpectralConvND::SpectralConvND(int in_ch, int out_ch,
                                std::vector<int> n_modes_,
                                bool use_bias,
                                bool hermitian,
                                std::string fft_norm_,
                                std::string metallib_path_)
    : in_channels(in_ch), out_channels(out_ch),
      order(static_cast<int>(n_modes_.size())),
      n_modes(n_modes_),
      _n_modes([&]{ auto v = n_modes_; v.back() = v.back()/2+1; return v; }()),
      weight_r(mx::zeros(_wshape(in_ch, out_ch, n_modes_))),
      weight_i(mx::zeros(_wshape(in_ch, out_ch, n_modes_))),
      bias(use_bias ? mx::zeros({out_ch}) : mx::array(0.0f)),
      has_bias(use_bias),
      enforce_hermitian_symmetry(hermitian),
      fft_norm(std::move(fft_norm_)),
      metallib_path(std::move(metallib_path_))
{}

// ── Forward ──────────────────────────────────────────────────────────────────

mx::array SpectralConvND::forward(const mx::array& x) const {
  // x: (B, Ci, d0, ..., dN-1)
  namespace fft = mx::fft;

  int ndim = x.ndim();  // order + 2
  int B    = x.shape()[0];

  // Spatial shape and FFT output shape
  std::vector<int> spatial(x.shape().begin() + 2, x.shape().end());

  // Build axes vectors
  std::vector<int> fft_axes;
  for (int i = 0; i < order; ++i) fft_axes.push_back(2 + i);

  std::vector<int> shift_axes(fft_axes.begin(), fft_axes.end() - 1);

  // ── Step 1: N-D real FFT ────────────────────────────────────────────────
  // fft_norm: "forward" maps to FFTNorm::Forward, etc.
  fft::FFTNorm norm = fft::FFTNorm::Backward;
  if (fft_norm == "forward") norm = fft::FFTNorm::Forward;
  else if (fft_norm == "ortho") norm = fft::FFTNorm::Ortho;

  auto x_ft = fft::rfftn(x, fft_axes, norm);
  // shape: (B, Ci, d0, ..., d_{N-2}, d_{N-1}//2+1)

  // FFT output spatial sizes
  std::vector<int> fft_size;
  for (int i = 0; i < order - 1; ++i) fft_size.push_back(spatial[i]);
  fft_size.push_back(spatial[order - 1] / 2 + 1);

  // ── Step 2: fftshift non-last dims ─────────────────────────────────────
  if (order > 1 && !shift_axes.empty()) {
    x_ft = fft::fftshift(x_ft, shift_axes);
  }

  // ── Step 3: compute weight-slicing starts ───────────────────────────────
  // starts[i] = max(0, nm_i - min(fft_size_i, nm_i))
  // If _n_modes[i] <= fft_size[i], start = 0 (take full weight as-is).
  // If _n_modes[i] >  fft_size[i], start > 0 (trim center of weight).
  std::vector<int> starts(order);
  for (int i = 0; i < order; ++i) {
    int nm = _n_modes[i];
    int fs = fft_size[i];
    starts[i] = std::max(0, nm - std::min(fs, nm));
  }

  // ── Step 4: slice weight ─────────────────────────────────────────────────
  // Weight shape: (Ci, Co, m0, m1, ..., mN-1)
  mx::Shape sw_start(order + 2, 0);
  mx::Shape sw_stop (weight_r.shape().begin(), weight_r.shape().end());

  for (int i = 0; i < order - 1; ++i) {
    int s = starts[i];
    if (s > 0) {
      sw_start[2 + i] = s / 2;
      sw_stop [2 + i] = weight_r.shape()[2 + i] - s / 2;
    }
  }
  // Last dim
  {
    int s = starts[order - 1];
    if (s > 0) {
      sw_stop[2 + order - 1] = weight_r.shape()[2 + order - 1] - s;
    }
  }
  auto wr_sl = mx::slice(weight_r, sw_start, sw_stop);
  auto wi_sl = mx::slice(weight_i, sw_start, sw_stop);

  // Effective weight spatial shape after slicing
  std::vector<int> km;
  for (int i = 0; i < order; ++i)
    km.push_back(static_cast<int>(wr_sl.shape()[2 + i]));

  // ── Step 5: slice x_ft ─────────────────────────────────────────────────
  // Non-last dims: center-based slice
  // Last dim: from-zero slice
  mx::Shape sx_start(ndim, 0);
  mx::Shape sx_stop (x_ft.shape().begin(), x_ft.shape().end());

  for (int i = 0; i < order - 1; ++i) {
    int fs  = fft_size[i];
    int km_i = km[i];
    int center = fs / 2;
    int neg    = km_i / 2;
    int pos    = km_i / 2 + km_i % 2;
    sx_start[2 + i] = center - neg;
    sx_stop [2 + i] = center + pos;
  }
  // Last dim
  {
    int km_last = km[order - 1];
    int fs_last = fft_size[order - 1];
    if (km_last < fs_last) {
      sx_stop[2 + order - 1] = km_last;
    }
  }
  auto x_sl = mx::slice(x_ft, sx_start, sx_stop);
  // x_sl: (B, Ci, km_0, ..., km_{N-1})

  // ── Step 6: complex contraction ─────────────────────────────────────────
  int prod_s = 1;
  for (int k : km) prod_s *= k;

  // Flatten spatial dims
  auto x_flat  = mx::reshape(mx::contiguous(x_sl), {B, in_channels, prod_s});
  auto wr_flat = mx::reshape(mx::contiguous(wr_sl), {in_channels, out_channels, prod_s});
  auto wi_flat = mx::reshape(mx::contiguous(wi_sl), {in_channels, out_channels, prod_s});

  auto contracted_flat = spectral_contract(
      x_flat, wr_flat, wi_flat,
      B, in_channels, out_channels, prod_s,
      metallib_path);
  // contracted_flat: complex64 (B, Co, prod_s)

  // Restore spatial shape
  mx::Shape c_shape = {B, out_channels};
  for (int k : km) c_shape.push_back(k);
  auto contracted = mx::reshape(contracted_flat, c_shape);

  // ── Step 7: pad contracted back to full FFT size ────────────────────────
  std::vector<std::pair<int, int>> pw;
  pw.push_back({0, 0});  // batch
  pw.push_back({0, 0});  // out_ch

  for (int i = 0; i < order - 1; ++i) {
    int fs  = fft_size[i];
    int km_i = km[i];
    int center = fs / 2;
    int neg    = km_i / 2;
    int pos    = km_i / 2 + km_i % 2;
    pw.push_back({center - neg, fs - center - pos});
  }
  // Last dim
  {
    int km_last = km[order - 1];
    int fs_last = fft_size[order - 1];
    pw.push_back({0, std::max(0, fs_last - km_last)});
  }

  auto zero_c = mx::array(std::complex<float>(0.f, 0.f));
  auto out_ft = mx::pad(contracted, pw, zero_c);
  // out_ft: complex64 (B, Co, fft_size[0], ..., fft_size[N-1])

  // ── Step 8: ifftshift non-last dims ────────────────────────────────────
  if (order > 1 && !shift_axes.empty()) {
    out_ft = fft::ifftshift(out_ft, shift_axes);
  }

  // ── Step 9: inverse FFT ─────────────────────────────────────────────────
  int last_axis = 2 + order - 1;

  if (enforce_hermitian_symmetry) {
    // ifftn on all but last spatial dim
    if (order > 1) {
      // n = original spatial sizes for non-last dims
      std::vector<int> non_last_sizes(spatial.begin(), spatial.end() - 1);
      mx::Shape shape_nl(non_last_sizes.begin(), non_last_sizes.end());
      out_ft = fft::ifftn(out_ft, shape_nl, shift_axes, norm);
    }

    // Zero DC imaginary (index 0 along last axis)
    {
      mx::Shape dc_start(out_ft.ndim(), 0);
      mx::Shape dc_stop (out_ft.shape().begin(), out_ft.shape().end());
      dc_stop[last_axis] = 1;
      auto dc   = mx::slice(out_ft, dc_start, dc_stop);
      auto dc_r = mx::astype(mx::real(dc), mx::complex64);
      out_ft = mx::slice_update(out_ft, dc_r, dc_start, dc_stop);
    }

    // Zero Nyquist imaginary (index -1 along last axis, if even)
    if (spatial.back() % 2 == 0) {
      int last_sz = out_ft.shape()[last_axis];
      mx::Shape ny_start(out_ft.ndim(), 0);
      mx::Shape ny_stop (out_ft.shape().begin(), out_ft.shape().end());
      ny_start[last_axis] = last_sz - 1;
      auto ny   = mx::slice(out_ft, ny_start, ny_stop);
      auto ny_r = mx::astype(mx::real(ny), mx::complex64);
      out_ft = mx::slice_update(out_ft, ny_r, ny_start, ny_stop);
    }

    // irfft on last dim
    auto r = fft::irfft(out_ft, spatial.back(), last_axis, norm);
    if (has_bias) {
      mx::Shape bias_shape(r.ndim(), 1);
      bias_shape[1] = out_channels;
      r = mx::add(r, mx::reshape(bias, bias_shape));
    }
    return r;
  } else {
    // Full irfftn
    mx::Shape n_shape(spatial.begin(), spatial.end());
    auto r = fft::irfftn(out_ft, n_shape, fft_axes, norm);
    if (has_bias) {
      mx::Shape bias_shape(r.ndim(), 1);
      bias_shape[1] = out_channels;
      r = mx::add(r, mx::reshape(bias, bias_shape));
    }
    return r;
  }
}

} // namespace neuralop
