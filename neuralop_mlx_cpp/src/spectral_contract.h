// spectral_contract.h — interface for the core complex einsum:
//   out[b, e, s] = sum_c  x_ft[b,c,s] * (wr[c,e,s] + i*wi[c,e,s])
//
// Two implementations are compiled; exactly one is linked:
//   spectral_contract_cpu.cpp   — reshape + batched matmul (always available)
//   spectral_contract_metal.cpp — custom Metal kernel (when NEURALOP_USE_METAL=1)
//
// Callers go through SpectralConvND which calls this function with already-sliced
// and flattened tensors. The public API only needs this single function.
#pragma once

#include <string>
#include <mlx/mlx.h>

namespace neuralop {

namespace mx = mlx::core;

// Arguments:
//   x_ft          complex64  (batch, in_ch, prod_s)   — flattened FFT modes
//   wr            float32    (in_ch, out_ch, prod_s)   — weight real part
//   wi            float32    (in_ch, out_ch, prod_s)   — weight imaginary part
//   batch, in_ch, out_ch, prod_s  — dimension sizes
//   metallib_path — path to neuralop_metal.metallib; ignored by CPU version
// Returns:
//   complex64  (batch, out_ch, prod_s)
mx::array spectral_contract(const mx::array& x_ft,
                             const mx::array& wr,
                             const mx::array& wi,
                             int batch, int in_ch, int out_ch, int prod_s,
                             const std::string& metallib_path = "");

} // namespace neuralop
