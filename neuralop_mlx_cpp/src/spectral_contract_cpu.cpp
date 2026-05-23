// Spectral contraction via reshape + batched matrix multiply (always available,
// no Metal required).
//
// Implements: out[b,e,s] = sum_c  x_ft[b,c,s] * (wr[c,e,s] + i*wi[c,e,s])
//
// x_ft : complex64  (batch, in_ch, prod_s)
// wr   : float32    (in_ch, out_ch, prod_s)
// wi   : float32    (in_ch, out_ch, prod_s)
// out  : complex64  (batch, out_ch, prod_s)
//
// Implementation:
//   Let xr = real(x_ft), xi = imag(x_ft)   — float32 (batch, in_ch, prod_s)
//   out_r = bmm(xr, wr) - bmm(xi, wi)
//   out_i = bmm(xr, wi) + bmm(xi, wr)
//
//   bmm(a, b): (B, Ci, Ps) × (Ci, Co, Ps) → (B, Co, Ps)
//     reshape  a → (Ps, B, Ci)
//     reshape  b → (Ps, Ci, Co)
//     matmul   → (Ps, B, Co)
//     transpose → (B, Co, Ps)
//
//   Combine: stack([out_r, out_i], -1) → (B, Co, Ps, 2) float32
//            flatten → (B*Co*Ps*2,)
//            view(complex64) → (B*Co*Ps,)
//            reshape → (B, Co, Ps)

#include "spectral_contract.h"
#include <mlx/mlx.h>

namespace neuralop {

static mx::array bmm(const mx::array& a, const mx::array& b) {
  // a: (B, Ci, Ps), b: (Ci, Co, Ps)
  // → transpose both, matmul, transpose back
  auto aT = mx::transpose(a, {2, 0, 1});  // (Ps, B, Ci)
  auto bT = mx::transpose(b, {2, 0, 1});  // (Ps, Ci, Co)
  return mx::transpose(mx::matmul(aT, bT), {1, 2, 0});  // (B, Co, Ps)
}

mx::array spectral_contract(const mx::array& x_ft,
                             const mx::array& wr,
                             const mx::array& wi,
                             int batch, int in_ch, int out_ch, int prod_s,
                             const std::string& /*metallib_path*/)
{
  auto xr = mx::real(x_ft);  // (B, Ci, Ps) float32
  auto xi = mx::imag(x_ft);  // (B, Ci, Ps) float32

  auto out_r = mx::subtract(bmm(xr, wr), bmm(xi, wi));  // (B, Co, Ps)
  auto out_i = mx::add     (bmm(xr, wi), bmm(xi, wr));  // (B, Co, Ps)

  // Pack real+imag into complex64:
  // stack along last dim → (B, Co, Ps, 2) float32 with layout [r0,i0,r1,i1,...]
  auto stacked = mx::stack({out_r, out_i}, -1);  // (B, Co, Ps, 2)
  // Ensure contiguous memory (required for view)
  auto contig  = mx::contiguous(stacked);
  // Flatten → view as complex64 → reshape to (B, Co, Ps)
  int  total   = batch * out_ch * prod_s;
  auto flat    = mx::flatten(contig);               // (B*Co*Ps*2,) float32
  auto cx_flat = mx::view(flat, mx::complex64);     // (B*Co*Ps,)   complex64
  return mx::reshape(cx_flat, {batch, out_ch, prod_s});
}

// No-op: CPU build always uses the matmul path.
void spectral_contract_force_cpu(bool) {}

} // namespace neuralop
