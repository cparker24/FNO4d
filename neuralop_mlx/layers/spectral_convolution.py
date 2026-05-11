"""MLX implementation of SpectralConvND (dense weights only, real FFT path).

Matches the logic of neuralop/layers/spectral_convolution.py for the default
non-separable, non-factorized, real-data forward pass (enforce_hermitian_symmetry=True).

Weight storage: real and imaginary parts are stored as separate float32 arrays
(weight_r, weight_i) so that any real-valued optimizer (AdamW, etc.) can update
them without needing complex-number support in the optimizer.
"""

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn


def _build_einsum_eq(order: int) -> str:
    """Build einsum subscript for dense non-separable contraction.

    Matches _contract_dense in neuralop spectral_convolution.py.
    For N spatial modes the tensors are 2+N dimensional:
      x:      (batch=a, in=b,  s0=c, ...)
      weight: (in=b,  out=e,   s0=c, ...)   where out=syms[2+order]
      result: (batch=a, out=e, s0=c, ...)
    All spatial symbols are shared so they are kept element-wise (not contracted).
    """
    syms = "abcdefghijklmnopqrstuvwxyz"
    b_sym = syms[0]                        # batch: 'a'
    i_sym = syms[1]                        # in_ch: 'b'
    o_sym = syms[2 + order]                # out_ch: first unused after spatial
    spatial_syms = list(syms[2:2 + order]) # spatial: 'c','d',... (same in x, w, out)
    x_eq   = b_sym + i_sym + "".join(spatial_syms)
    w_eq   = i_sym + o_sym + "".join(spatial_syms)
    out_eq = b_sym + o_sym + "".join(spatial_syms)
    return f"{x_eq},{w_eq}->{out_eq}"


def _complex_einsum(eq: str, x_c: mx.array, wr: mx.array, wi: mx.array) -> mx.array:
    """Complex matrix-multiply via real einsum: (x_r + i*x_i) @ (wr + i*wi).

    Avoids the need for a complex-aware optimizer by storing weights as two real arrays.
    """
    x_r = mx.real(x_c)
    x_i = mx.imag(x_c)
    real_part = mx.einsum(eq, x_r, wr) - mx.einsum(eq, x_i, wi)
    imag_part = mx.einsum(eq, x_r, wi) + mx.einsum(eq, x_i, wr)
    # Reconstruct complex output: cast each to complex64, then add imaginary part
    return real_part.astype(mx.complex64) + (imag_part * 1j).astype(mx.complex64)


class SpectralConvND(nn.Module):
    """N-dimensional spectral convolution using real FFT.

    Implements the core FNO layer: truncated Fourier-space weight multiplication.

    Weights are stored as two float32 arrays (weight_r, weight_i) so that
    standard real-valued optimizers (e.g. AdamW) work correctly.

    Parameters
    ----------
    in_channels, out_channels : int
    n_modes : list[int]
        Number of Fourier modes per spatial dimension. The last entry is
        halved internally (n_modes[-1] // 2 + 1) to account for the real FFT.
    bias : bool
    fft_norm : str
        Normalization for FFT/IFFT, default "forward" matching neuralop.
    enforce_hermitian_symmetry : bool
        Zero the imaginary parts of DC and Nyquist before irfft to avoid
        line artifacts (matches PyTorch version, default True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: List[int],
        bias: bool = True,
        fft_norm: str = "forward",
        enforce_hermitian_symmetry: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = len(n_modes)
        self.fft_norm = fft_norm
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry

        # Internally store the reduced last-dim mode count (real FFT redundancy)
        n_modes = list(n_modes)
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

        # Real and imaginary parts stored separately as float32 (optimizer-friendly)
        weight_shape = [in_channels, out_channels] + n_modes
        fan_in = in_channels * math.prod(n_modes)
        init_std = math.sqrt(2.0 / fan_in)
        self.weight_r = mx.random.normal(weight_shape) * init_std
        self.weight_i = mx.random.normal(weight_shape) * init_std

        if bias:
            bias_shape = [1, out_channels] + [1] * self.order
            self.bias = mx.zeros(bias_shape)
        else:
            self.bias = None

        self._einsum_eq = _build_einsum_eq(self.order)

    @property
    def weight(self) -> mx.array:
        """Complex weight (read-only view, for weight transfer convenience)."""
        return self.weight_r.astype(mx.complex64) + (self.weight_i * 1j).astype(mx.complex64)

    @weight.setter
    def weight(self, value: mx.array):
        """Set weight from a complex64 array (used in weight transfer tests)."""
        self.weight_r = mx.real(value).astype(mx.float32)
        self.weight_i = mx.imag(value).astype(mx.float32)

    def __call__(self, x):
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape (batch, in_channels, d0, d1, ..., d_{N-1})

        Returns
        -------
        mx.array, same shape with out_channels replacing in_channels
        """
        batchsize = x.shape[0]
        mode_sizes = list(x.shape[2:])  # spatial dims (before FFT)

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # real FFT last-dim size
        fft_dims = list(range(-self.order, 0))
        shift_dims = fft_dims[:-1]  # all except last (real FFT symmetric)

        # Forward FFT
        x_ft = mx.fft.rfftn(x, axes=fft_dims, norm=self.fft_norm)
        if self.order > 1:
            x_ft = mx.fft.fftshift(x_ft, axes=shift_dims)

        # Weight slice: handle case where n_modes > fft_size (pad to max)
        starts = [
            max(0, nm - min(fs, nm))
            for fs, nm in zip(fft_size, self._n_modes)
        ]

        slices_w = [slice(None), slice(None)]  # in_ch, out_ch
        for start in starts[:-1]:
            slices_w.append(slice(start // 2, -start // 2) if start else slice(None))
        # Last dim: real FFT — weight already halved, slice from start
        slices_w.append(slice(None, -starts[-1]) if starts[-1] else slice(None))
        slices_w_t = tuple(slices_w)
        wr = self.weight_r[slices_w_t]
        wi = self.weight_i[slices_w_t]

        # Input mode selection (center-based for non-last dims, from-zero for last)
        slices_x = [slice(None), slice(None)]  # batch, in_ch
        weight_spatial_shape = list(wr.shape[2:])
        for fs, km in zip(fft_size[:-1], weight_spatial_shape[:-1]):
            center = fs // 2
            neg = km // 2
            pos = km // 2 + km % 2
            slices_x.append(slice(center - neg, center + pos))
        # Last dim: DC at index 0, take from zero
        km_last = weight_spatial_shape[-1]
        if km_last < fft_size[-1]:
            slices_x.append(slice(None, km_last))
        else:
            slices_x.append(slice(None))

        slices_x = tuple(slices_x)

        # Spectral contraction using real-split complex multiply
        contracted = _complex_einsum(self._einsum_eq, x_ft[slices_x], wr, wi)

        # Pad contracted modes into full FFT buffer.
        # mx.at[multi-dim-slice].add has a stride bug in MLX 0.31; mx.pad is correct.
        pad_widths = [(0, 0), (0, 0)]  # batch, out_ch
        for fs, km in zip(fft_size[:-1], weight_spatial_shape[:-1]):
            center = fs // 2
            neg = km // 2
            pos = km // 2 + km % 2
            pad_widths.append((center - neg, fs - center - pos))
        km_last = weight_spatial_shape[-1]
        pad_widths.append((0, max(0, fft_size[-1] - km_last)))
        out_ft = mx.pad(contracted, pad_widths)

        # Inverse FFT
        if self.order > 1:
            out_ft = mx.fft.ifftshift(out_ft, axes=shift_dims)

        if self.enforce_hermitian_symmetry:
            # ifftn on all dims except last
            out_ft = mx.fft.ifftn(out_ft, s=mode_sizes[:-1], axes=fft_dims[:-1], norm=self.fft_norm)
            # Zero imaginary at DC
            dc = out_ft[..., 0]
            out_ft = out_ft.at[..., 0].add(
                (mx.real(dc) - dc).astype(mx.complex64)
            )
            # Zero imaginary at Nyquist if last spatial dim is even
            if mode_sizes[-1] % 2 == 0:
                ny = out_ft[..., -1]
                out_ft = out_ft.at[..., -1].add(
                    (mx.real(ny) - ny).astype(mx.complex64)
                )
            x = mx.fft.irfft(out_ft, n=mode_sizes[-1], axis=fft_dims[-1], norm=self.fft_norm)
        else:
            x = mx.fft.irfftn(out_ft, s=mode_sizes, axes=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias

        return x
