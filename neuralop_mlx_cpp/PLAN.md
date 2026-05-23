# neuralop_mlx_cpp — Implementation Plan

## Goal

Provide a self-contained C++ inference library that:
1. Loads trained FNO model weights (PyTorch `.safetensors` or MLX `.safetensors`)
2. Runs FNO forward inference on Apple Silicon via Apple's MLX C++ backend
3. Uses a custom Metal kernel for the inner spectral contraction (default), with a fallback to reshape + batched matmul

---

## Why MLX C++ (not LibTorch)

PyTorch's MPS backend has a broken complex FFT implementation. MLX on Apple Silicon has a correct N-D real FFT. The Python `neuralop_mlx` library already uses MLX for this reason. This C++ library ports the same logic without requiring a Python runtime at inference time.

---

## Architecture

```
neuralop_mlx_cpp/
├── CMakeLists.txt
├── include/neuralop_mlx_cpp/fno_inference.h   ← public umbrella header
├── src/
│   ├── config.cpp                             ← FNOConfig::from_json
│   ├── ops.h / ops.cpp                        ← Linear, ChannelMLP, skips, GridEmbeddingND
│   ├── spectral_contract.h                    ← shared interface
│   ├── spectral_contract_cpu.cpp              ← matmul fallback
│   ├── spectral_contract_metal.cpp            ← Metal primitive (NEURALOP_USE_METAL=1)
│   ├── spectral_conv.h / spectral_conv.cpp    ← SpectralConvND forward
│   ├── fno_blocks.h / fno_blocks.cpp          ← FNOBlocks forward
│   ├── fno.h / fno.cpp                        ← FNO model
│   └── weight_loader.h / weight_loader.cpp    ← safetensors weight loading
├── kernels/
│   └── spectral_contract.metal                ← Metal kernel
├── tools/
│   └── export_weights.py                      ← one-time weight conversion
└── examples/
    └── infer.cpp                              ← inference demo
```

---

## Layer Porting (Python → C++)

### SpectralConvND (`spectral_convolution.py`)

The most complex layer. Key transformations:

1. `rfftn(x, fft_dims, norm)` — N-D real FFT
2. `fftshift(x_ft, shift_dims)` — shift non-last FFT dims (if order > 1)
3. Compute weight-trim starts: `start_i = max(0, n_mode_i - min(fft_size_i, n_mode_i))`
4. Slice weight and x_ft to the active mode region
5. Flatten spatial → contract: `out[b,e,s] = Σ_c x_ft[b,c,s] * (wr[c,e,s] + i·wi[c,e,s])`
6. Pad back to full FFT size
7. `ifftshift` (if order > 1)
8. Hermitian inverse FFT: `ifftn(non-last) → zero DC/Nyquist imag → irfft(last)`
9. Add bias

**Weight storage**: `weight_r`, `weight_i` as float32 `(in_ch, out_ch, m0,…,mN-1)`.
`_n_modes[-1] = n_modes[-1]//2 + 1` (halved for real FFT last axis).

### ChannelMLP (`channel_mlp.py`)

Point-wise MLP over channels:
- reshape spatial dims → `(B, C, prod_s)`
- transpose → `(B, prod_s, C)`
- linear + gelu chain
- transpose back, restore spatial shape

**Note**: `gelu` is not in MLX C++ ops; implemented as `0.5·x·(1 + erf(x/√2))`.

### FNOBlocks (`fno_block.py`)

```
for index in range(n_layers):
    x_fno = conv(x) + skip_fno(x)
    x     = gelu(x_fno)   [or x_fno at last layer]
    x_mlp = mlp(x) + skip_mlp(x_in)   ← x_in is original input to block
    x     = gelu(x_mlp)   [or x_mlp at last layer]
```

### FNO model (`fno.py`)

```
x → positional_embedding? → lifting_mlp → fno_blocks × n_layers → projection_mlp
```

---

## Spectral Contraction: Two Implementations

### Metal kernel (`kernels/spectral_contract.metal`)

- Template parameter `TILE_C ∈ {8, 16}`
- Grid: `(prod_s, out_ch, batch)` — one thread per output element
- Threadgroup: `(1, TILE_C, 1)` — TILE_C threads share same `(s, batch)`
- Threadgroup memory: `tg_xr[TILE_C]`, `tg_xi[TILE_C]` — cooperative load of `x_ft` tile
- Each thread reads its own weight column independently

**Memory benefit**: `x_ft` is loaded once per tile and reused by `TILE_C` output channels, reducing device memory bandwidth by `TILE_C×` for the x_ft reads.

### CPU/fallback (`src/spectral_contract_cpu.cpp`)

Batched matrix multiply using transposed layout:
```
bmm(a, b): (B, Ci, Ps) × (Ci, Co, Ps) → (B, Co, Ps)
  aT = transpose(a, [2,0,1])  → (Ps, B, Ci)
  bT = transpose(b, [2,0,1])  → (Ps, Ci, Co)
  mm = matmul(aT, bT)         → (Ps, B, Co)
  return transpose(mm, [1,2,0])

out_r = bmm(xr, wr) - bmm(xi, wi)
out_i = bmm(xr, wi) + bmm(xi, wr)
result = view(flatten(contiguous(stack([out_r, out_i], -1))), complex64)
         .reshape(B, Co, Ps)
```

---

## Weight Loading

### Format detection

| Key present | Format |
|---|---|
| `fno_blocks.convs.0.weight_r` | MLX-format (from `export_weights.py`) |
| `fno_blocks.convs.0.weight.tensor` | PyTorch-format safetensors |

### Key mappings (PyTorch → C++ struct)

| PyTorch safetensors key | C++ struct field | Transform |
|---|---|---|
| `fno_blocks.convs.i.weight.tensor` (complex64) | `convs[i].weight_r/i` | `real()` / `imag()` |
| `fno_blocks.fno_skips.i.conv.weight` (out,in,1) | `fno_skips[i].linear.weight` | `squeeze(-1)` |
| `fno_blocks.channel_mlp.i.fcs.j.weight` (out,in,1) | `channel_mlps[i].fcs[j].weight` | `squeeze(-1)` |
| `fno_blocks.channel_mlp_skips.i.weight` (1,C,1,…) | `mlp_skips[i].alpha` | `flatten()` |
| `lifting.fcs.j.weight` (out,in,1) | `lifting.fcs[j].weight` | `squeeze(-1)` |

---

## Build System

```
cmake -DMLX_DIR=<path/to/site-packages/mlx/share/cmake/MLX> \
      -DNEURALOP_USE_METAL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -S neuralop_mlx_cpp -B build
cmake --build build --target fno_infer -j
```

When `NEURALOP_USE_METAL=ON` (default) and `MLX_BUILD_METAL=ON`:
- Compiles `kernels/spectral_contract.metal` → `neuralop_metal.metallib`
- Links `spectral_contract_metal.cpp`
- Defines `NEURALOP_DEFAULT_METALLIB_PATH`

When `NEURALOP_USE_METAL=OFF` or Metal unavailable:
- Links `spectral_contract_cpu.cpp` instead

---

## Weight Export Workflow

```
# Step 1: Export PyTorch → MLX safetensors + config JSON
conda activate fno_env_mlx
python neuralop_mlx_cpp/tools/export_weights.py \
    --pt path/to/best_model.pt \
    --train-config path/to/config.yaml \
    --out weights.safetensors \
    --config-out model_config.json

# Step 2: Run C++ inference
./build/fno_infer weights.safetensors model_config.json 1 50 50 50 10
```

PyTorch-format safetensors can be loaded directly (auto-detected) — no conversion needed if the training code uses `safetensors.torch.save_file`.

---

## Dependencies

| Library | Version | How |
|---|---|---|
| MLX C++ | ≥ 0.21 | `find_package(MLX)` |
| nlohmann/json | 3.11.3 | Single-header, auto-downloaded at cmake configure |
| metal-cpp | bundled with MLX | via `MLX_INCLUDE_DIRS` |

---

## Key Design Decisions

1. **Inference-only**: No backprop. `Primitive::vjp`/`jvp`/`vmap` not implemented.
2. **No Python bindings**: Pure C++ for embedding in downstream apps.
3. **JSON config**: Simple, JSON-portable, independent of YAML/Python runtime.
4. **Auto-detect weight format**: No user configuration needed for format.
5. **Hermitian FFT default**: Matches `enforce_hermitian_symmetry=True` default in Python.
6. **FFT norm "forward"**: Matches training default; configurable per model.
