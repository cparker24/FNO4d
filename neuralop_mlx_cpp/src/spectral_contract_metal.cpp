// Custom Metal kernel implementation of the spectral contraction.
// Compiled when NEURALOP_USE_METAL=1.
//
// The kernel caches a tile of x_ft in threadgroup memory so that all
// out_ch threads that share the same (batch, spatial_idx) position reuse
// those values instead of re-loading from device memory for each e.
//
// Grid:        (prod_s, out_ch, batch)
// Threadgroup: (1, TILE_C, 1)          — TILE_C = 8 or 16
// TG memory:   2 * TILE_C * sizeof(float) for tg_xr / tg_xi

#include "spectral_contract.h"

#include <memory>
#include <string>
#include <stdexcept>

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>

// The CPU path (reshape+matmul) reused for eval_cpu inside this primitive.
// Forward-declare the CPU version which lives in spectral_contract_cpu.cpp.
// We provide the CPU fallback here inline to avoid a circular dependency.

namespace neuralop {

// ── CPU fallback (used inside eval_cpu) ─────────────────────────────────
static mx::array bmm_cpu(const mx::array& a, const mx::array& b) {
  auto aT = mx::transpose(a, {2, 0, 1});
  auto bT = mx::transpose(b, {2, 0, 1});
  return mx::transpose(mx::matmul(aT, bT), {1, 2, 0});
}

static mx::array contract_cpu(const mx::array& x_ft,
                               int batch, int out_ch, int prod_s,
                               const mx::array& wr, const mx::array& wi) {
  auto xr = mx::real(x_ft);
  auto xi = mx::imag(x_ft);
  auto out_r = mx::subtract(bmm_cpu(xr, wr), bmm_cpu(xi, wi));
  auto out_i = mx::add     (bmm_cpu(xr, wi), bmm_cpu(xi, wr));
  auto stacked = mx::stack({out_r, out_i}, -1);
  auto contig  = mx::contiguous(stacked);
  int  total   = batch * out_ch * prod_s;
  auto flat    = mx::flatten(contig);
  auto cx_flat = mx::view(flat, mx::complex64);
  return mx::reshape(cx_flat, {batch, out_ch, prod_s});
}

// ── Primitive ─────────────────────────────────────────────────────────────

class SpectralContractPrim : public mx::Primitive {
 public:
  SpectralContractPrim(mx::Stream stream,
                        int batch, int in_ch, int out_ch, int prod_s,
                        std::string metallib_path)
      : Primitive(stream),
        batch_(batch), in_ch_(in_ch), out_ch_(out_ch), prod_s_(prod_s),
        metallib_path_(std::move(metallib_path)) {}

  // CPU fallback — runs reshape+matmul on CPU stream
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    auto result = contract_cpu(inputs[0], batch_, out_ch_, prod_s_,
                               inputs[1], inputs[2]);
    mx::eval(result);
    outputs[0] = result;
  }

  // GPU — dispatch custom Metal kernel
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    using namespace mlx::core::metal;

    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    const auto& s = stream();
    auto& d = device(s.device);

    const std::string& lib_path =
        metallib_path_.empty() ? NEURALOP_DEFAULT_METALLIB_PATH : metallib_path_;

    auto* lib    = d.get_library("neuralop_metallib", lib_path);
    int   tile_c = (in_ch_ >= 16) ? 16 : 8;
    std::string kname = "spectral_contract_fwd_" + std::to_string(tile_c);
    auto* kernel = d.get_kernel(kname, lib);

    auto& enc = get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array (inputs[0], 0);  // x_ft  complex64
    enc.set_input_array (inputs[1], 1);  // wr     float32
    enc.set_input_array (inputs[2], 2);  // wi     float32
    enc.set_output_array(out,        3);  // out   complex64

    enc.set_bytes(batch_,   4);
    enc.set_bytes(in_ch_,   5);
    enc.set_bytes(out_ch_,  6);
    enc.set_bytes(prod_s_,  7);

    // Threadgroup memory: tg_xr (index 0) and tg_xi (index 1)
    enc.set_threadgroup_memory_length(tile_c * sizeof(float), 0);
    enc.set_threadgroup_memory_length(tile_c * sizeof(float), 1);

    MTL::Size grid = MTL::Size(prod_s_, out_ch_, batch_);
    MTL::Size tg   = MTL::Size(1, tile_c, 1);
    enc.dispatch_threads(grid, tg);
  }

  const char* name() const override { return "SpectralContract"; }

 private:
  int         batch_, in_ch_, out_ch_, prod_s_;
  std::string metallib_path_;
};

// ── Public entry point ────────────────────────────────────────────────────

mx::array spectral_contract(const mx::array& x_ft,
                             const mx::array& wr,
                             const mx::array& wi,
                             int batch, int in_ch, int out_ch, int prod_s,
                             const std::string& metallib_path)
{
  auto stream = mx::default_stream(mx::gpu);
  return mx::array(
      {batch, out_ch, prod_s},
      mx::complex64,
      std::make_shared<SpectralContractPrim>(
          stream, batch, in_ch, out_ch, prod_s, metallib_path),
      {x_ft, wr, wi}
  );
}

} // namespace neuralop
