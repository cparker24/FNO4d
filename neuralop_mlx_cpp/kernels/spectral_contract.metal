// Metal kernel for the spectral contraction:
//   out[b,e,s] = sum_c  x_ft[b,c,s] * (wr[c,e,s] + i*wi[c,e,s])
//
// Memory layouts (all row-major / C-contiguous):
//   x_ft  : complex64  indexed as  a * in_ch * prod_s + c * prod_s + s
//   wr/wi : float32    indexed as  c * out_ch * prod_s + e * prod_s + s
//   out   : complex64  indexed as  a * out_ch * prod_s + e * prod_s + s
//
// Threadgroup tiling:
//   Grid        = (prod_s, out_ch, batch)  — one thread per output element
//   Threadgroup = (1, TILE_C, 1)
//   All TILE_C threads in a group share the same (s, batch) but differ in e.
//   They cooperate to preload TILE_C values of x_ft into threadgroup memory,
//   then each independently reads weights for its own e.

#include <metal_stdlib>
// Avoid pulling in MLX headers: the MLX complex.h references bfloat16_t which
// may not exist in older Metal toolchains.  We only need complex64_t and the
// instantiate_kernel macro — define them here directly.

struct complex64_t {
    float real;
    float imag;
    complex64_t() = default;
    complex64_t(float r, float i) : real(r), imag(i) {}
};

// Equivalent to the macro in mlx/backend/metal/kernels/defines.h
#define instantiate_kernel(name, func, ...)                              \
    template [[host_name(name)]] [[kernel]]                              \
    decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

using namespace metal;

template<int TILE_C>
[[kernel]] void spectral_contract_fwd(
    device const complex64_t* x_ft  [[buffer(0)]],
    device const float*       wr    [[buffer(1)]],
    device const float*       wi    [[buffer(2)]],
    device       complex64_t* out   [[buffer(3)]],
    constant     int&         batch   [[buffer(4)]],
    constant     int&         in_ch   [[buffer(5)]],
    constant     int&         out_ch  [[buffer(6)]],
    constant     int&         prod_s  [[buffer(7)]],
    threadgroup  float*       tg_xr [[threadgroup(0)]],
    threadgroup  float*       tg_xi [[threadgroup(1)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgsz [[threads_per_threadgroup]])
{
    int s = (int)gid.x;
    int e = (int)gid.y;
    int a = (int)gid.z;

    if (s >= prod_s || e >= out_ch || a >= batch) return;

    float acc_r = 0.f;
    float acc_i = 0.f;

    for (int c0 = 0; c0 < in_ch; c0 += TILE_C) {
        // Cooperative load: thread tid.y loads x_ft[a][c0 + tid.y][s]
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)tid.y < TILE_C && (c0 + (int)tid.y) < in_ch) {
            int c   = c0 + (int)tid.y;
            int idx = a * in_ch * prod_s + c * prod_s + s;
            tg_xr[tid.y] = x_ft[idx].real;
            tg_xi[tid.y] = x_ft[idx].imag;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over the tile
        int tc_end = min(TILE_C, in_ch - c0);
        for (int tc = 0; tc < tc_end; ++tc) {
            int c      = c0 + tc;
            int w_idx  = c * out_ch * prod_s + e * prod_s + s;
            float w_r  = wr[w_idx];
            float w_i  = wi[w_idx];
            float xr_c = tg_xr[tc];
            float xi_c = tg_xi[tc];
            acc_r += xr_c * w_r - xi_c * w_i;
            acc_i += xr_c * w_i + xi_c * w_r;
        }
    }

    out[a * out_ch * prod_s + e * prod_s + s] = complex64_t(acc_r, acc_i);
}

instantiate_kernel("spectral_contract_fwd_8",  spectral_contract_fwd, 8)
instantiate_kernel("spectral_contract_fwd_16", spectral_contract_fwd, 16)
