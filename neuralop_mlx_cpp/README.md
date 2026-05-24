# neuralop_mlx_cpp

Inference-only C++ library for the **Fourier Neural Operator (FNO)** using
[Apple MLX](https://ml-explore.github.io/mlx) as the compute backend.

Designed for RHIC heavy-ion physics simulation inference on Apple Silicon Macs,
where the standard PyTorch/MPS backend has a broken complex FFT.

---

## Features

- **Correct complex N-D FFT** via MLX (fixes the PyTorch/MPS bug)
- **Custom Metal kernel** for the spectral contraction (default)
  — falls back to reshape + batched matmul when Metal is unavailable
- **Direct weight loading** from PyTorch `.safetensors` or MLX `.safetensors`
- **No Python runtime** required at inference time
- Single-pass `cmake` + `make` build

---

## Quick Start

### 1. Prerequisites

- macOS 13+ with Apple Silicon
- Xcode command-line tools:
  ```bash
  xcode-select --install
  ```
- MLX installed in a conda environment (the same env used for training):
  ```bash
  conda activate fno_env_mlx
  ```
- **(Optional, for Metal kernel)** The Metal compiler toolchain. It is separate
  from the standard Xcode CLI tools and must be downloaded explicitly:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
  Without this, CMake detects the missing compiler at configure time and
  automatically falls back to the matmul implementation — the build will still
  succeed, just without the GPU-optimised kernel.

### 2. Build

```bash
cd /path/to/FNO4d

cmake \
  -DMLX_DIR=$(python -c "import importlib.util, os; sp=importlib.util.find_spec('mlx'); print(os.path.join(list(sp.submodule_search_locations)[0],'share','cmake','MLX'))") \
  -DCMAKE_BUILD_TYPE=Release \
  -S neuralop_mlx_cpp \
  -B neuralop_mlx_cpp/build

cmake --build neuralop_mlx_cpp/build --target fno_infer -j
```

> **If the `python -c` snippet above fails** (e.g. `mlx.__file__` is `None`),
> pass the path explicitly instead:
> ```bash
> -DMLX_DIR=/opt/homebrew/Caskroom/miniconda/base/envs/fno_env_mlx/lib/python3.13/site-packages/mlx/share/cmake/MLX
> ```
> Adjust the conda env name / Python version to match your installation.

The first configure step will auto-download `nlohmann/json.hpp` into
`neuralop_mlx_cpp/third_party/`.

### 3. Export weights

Convert a PyTorch checkpoint to the MLX safetensors format:

```bash
python neuralop_mlx_cpp/tools/export_weights.py \
    --pt path/to/best_model.pt \
    --train-config path/to/config.yaml \
    --out weights.safetensors \
    --config-out model_config.json
```

Or if you already have a PyTorch safetensors file, it can be loaded directly
(the weight loader auto-detects the format).

### 4. Run inference

```bash
./neuralop_mlx_cpp/build/fno_infer \
    weights.safetensors \
    model_config.json \
    1 50 50 50 10     # batch=1, spatial=(50,50,50,10)
```

---

## Using in Your Own Code

```cpp
#include "neuralop_mlx_cpp/fno_inference.h"
namespace no = neuralop;
namespace mx = mlx::core;

// Load config
no::FNOConfig cfg = no::FNOConfig::from_json("model_config.json");

// Build model
no::FNO model(
    cfg.in_channels, cfg.out_channels, cfg.hidden_channels,
    cfg.n_layers, cfg.n_modes,
    cfg.positional_embedding, cfg.use_channel_mlp,
    cfg.lifting_channel_ratio, cfg.projection_channel_ratio,
    cfg.channel_mlp_expansion,
    cfg.fno_skip, cfg.channel_mlp_skip, cfg.fft_norm,
    cfg.metallib_path);

// Load weights (MLX-format or PyTorch-format safetensors)
no::load_weights(model, "weights.safetensors");

// Run inference; input shape = (batch, in_channels, d0, ..., dN-1)
mx::array input = mx::random::normal({1, 4, 50, 50, 50, 10});
mx::array output = model.forward(input);
mx::eval(output);   // triggers computation
```

Link your target with `neuralop_mlx_cpp` in your `CMakeLists.txt`:

```cmake
target_link_libraries(my_app PRIVATE neuralop_mlx_cpp)
```

---

## Build Options

| CMake Option | Default | Description |
|---|---|---|
| `NEURALOP_USE_METAL` | `ON` | Build custom Metal kernel for spectral contraction |
| `MLX_DIR` | — | Path to `MLX/share/cmake/MLX` if not on `CMAKE_PREFIX_PATH` |

When `NEURALOP_USE_METAL=ON` and the MLX installation includes Metal support,
the Metal kernel `kernels/spectral_contract.metal` is compiled into
`neuralop_metal.metallib` alongside the library.

To use the matmul fallback regardless:
```bash
cmake -DNEURALOP_USE_METAL=OFF ...
```

---

## Weight File Formats

The `load_weights` function auto-detects the format from the key names:

### MLX-format (produced by `tools/export_weights.py`)

Weights stored as float32 pairs:
- `fno_blocks.convs.i.weight_r` / `.weight_i`
- `fno_blocks.fno_skips.i.linear.weight`
- `fno_blocks.channel_mlps.i.fcs.j.weight`
- `fno_blocks.mlp_skips.i.alpha`
- `lifting.fcs.j.weight` / `projection.fcs.j.weight`

### PyTorch-format (`safetensors.torch.save_file`)

Complex weights automatically split:
- `fno_blocks.convs.i.weight.tensor` (complex64) → `weight_r` + `weight_i`
- `fno_blocks.fno_skips.i.conv.weight` (out,in,1) → squeezed linear weight
- `fno_blocks.channel_mlp.i.fcs.j.weight` → channel MLP weights
- `fno_blocks.channel_mlp_skips.i.weight` → soft-gating alpha

---

## Project Structure

```
neuralop_mlx_cpp/
├── CMakeLists.txt
├── PLAN.md                            ← design & implementation notes
├── include/neuralop_mlx_cpp/
│   └── fno_inference.h               ← public umbrella header
├── src/
│   ├── config.cpp                    ← FNOConfig::from_json
│   ├── ops.h / ops.cpp               ← Linear, ChannelMLP, skips, GridEmbeddingND
│   ├── spectral_contract.h           ← shared spectral contraction interface
│   ├── spectral_contract_cpu.cpp     ← matmul fallback
│   ├── spectral_contract_metal.cpp   ← Metal custom primitive
│   ├── spectral_conv.h / .cpp        ← SpectralConvND
│   ├── fno_blocks.h / .cpp           ← FNOBlocks
│   ├── fno.h / .cpp                  ← FNO top-level model
│   └── weight_loader.h / .cpp        ← safetensors weight loading
├── kernels/
│   └── spectral_contract.metal       ← Metal compute kernel
├── tools/
│   └── export_weights.py             ← PyTorch → MLX weight conversion
└── examples/
    └── infer.cpp                     ← CLI inference demo
```

---

## Metal Kernel Details

The spectral contraction `out[b,e,s] = Σ_c x_ft[b,c,s] * W[c,e,s]` is
implemented as a tiled threadgroup kernel:

- **Grid**: `(prod_s, out_ch, batch)` — one thread per output element
- **Threadgroup**: `(1, TILE_C, 1)` where `TILE_C ∈ {8, 16}`
- **Threadgroup memory**: `2 × TILE_C × sizeof(float)` for cached `x_ft` tile

All `TILE_C` threads in a threadgroup share the same `(s, batch)` position.
They cooperatively load `TILE_C` values of `x_ft` into threadgroup memory,
then each independently accumulates over its unique weight column `e`.

This reuses each `x_ft` value `TILE_C` times from fast threadgroup memory
instead of reading it from device memory once per output channel.

---

## Related Python Library

The Python implementation lives in `neuralop_mlx/` in the same repository.
The `tools/export_weights.py` script uses the Python library to load PyTorch
weights and convert them to the MLX format readable by this C++ library.

---

## Dependencies

| Library | Purpose | Provided by |
|---|---|---|
| [MLX](https://github.com/ml-explore/mlx) | Array ops + Metal FFT | conda `fno_env_mlx` |
| [metal-cpp](https://developer.apple.com/metal/cpp/) | Metal C++ bindings | bundled with MLX |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON config parsing | auto-downloaded |

---

## τ/η Positional Embedding — `FNOTauEta` (C++/MLX)

See the root `README.md` for the physics motivation and design rationale.

### New source files

| File | Purpose |
|------|---------|
| `src/fno_tau_eta.h` | `FNOTauEta` struct declaration |
| `src/fno_tau_eta.cpp` | `FNOTauEta` implementation |
| `src/ops.h` | `TauBias`, `EtaBias` structs + `sinusoidal_tau_embedding` / `sinusoidal_eta_embedding` |
| `src/ops.cpp` | Implementations of the above |
| `src/weight_loader.h/.cpp` | `load_weights(FNOTauEta&, path)` overload |

Add `fno_tau_eta.cpp` to your `CMakeLists.txt` source list alongside `fno.cpp`.

### Inference example

```cpp
#include "fno_tau_eta.h"
#include "weight_loader.h"

// Build coordinate vectors from NormFn3d metadata
std::vector<float> tau_vals(128), eta_vals(32);
for (int i = 0; i < 128; ++i) tau_vals[i] = 0.4f + i * 0.076f;
for (int i = 0; i < 32;  ++i) eta_vals[i] = -4.0f + i * (8.0f / 31.f);

// Construct model (must match checkpoint architecture)
neuralop::FNOTauEta model(
    /*in_channels=*/4, /*out_channels=*/4, /*hidden_channels=*/32,
    /*n_layers=*/6,
    /*n_modes=*/{8, 8, 4, 6},
    /*tau_max=*/12.4f, /*eta_max=*/4.0f
);

// Load weights — tau_max / eta_max scalars are also read from the file
neuralop::load_weights(model, "ckpt/model.safetensors");

// Input: (B, C, X, Y, Z, T)
auto x = mlx::core::zeros({1, 4, 64, 64, 32, 128});

auto pred = model.forward(x, tau_vals, eta_vals);
mlx::core::eval(pred);
// pred shape: (1, 4, 64, 64, 32, 128)
```

### Weight key layout (safetensors)

The C++ weight loader expects the same safetensors file produced by the MLX
Python `Trainer.checkpoint()`:

| Key pattern | Loaded into |
|-------------|-------------|
| `_fno.lifting.fcs.{j}.weight` | `fno.lifting.fcs[j].weight` |
| `_fno.fno_blocks.convs.{i}.weight_r / .weight_i` | `fno.blocks.convs[i]` |
| `_fno.fno_blocks.{fno/mlp}_skips.{i}.*` | `fno.blocks.{fno/mlp}_skips[i]` |
| `_fno.projection.fcs.{j}.weight` | `fno.projection.fcs[j].weight` |
| `tau_biases.{i}.proj.weight` | `tau_biases[i].proj.weight` |
| `eta_biases.{i}.proj.weight` | `eta_biases[i].proj.weight` |
| `_tau_max` | `model.tau_max` |
| `_eta_max` | `model.eta_max` |

PyTorch safetensors (exported via `safetensors.torch.save_file`) use `fno.*`
instead of `_fno.*`; the loader detects the format automatically.
