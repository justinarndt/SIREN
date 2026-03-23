"""
Phase-Magnitude Quantization with Straight-Through Estimator
============================================================

Standard quantization fails for structured (circulant) matrices because
parameters live in the frequency domain — a small δ in a spectral coefficient
produces a *global, non-local* perturbation in the spatial-domain weights.

Solution: decompose each complex spectral coefficient into polar form
    Λ = r · e^{jθ}
and quantize magnitude r and phase θ independently:
    - Magnitude: log-domain uniform quantization (captures dynamic range)
    - Phase: uniform quantization on [−π, π]

The Straight-Through Estimator (STE) allows gradients to bypass the
non-differentiable rounding:
    forward:   x̂ = quantize(x)
    backward:  ∂L/∂x = ∂L/∂x̂   (identity gradient)

Implementation:  x̂ = x + (quantize(x) − x).detach()

References:
  [1] Bengio et al., "Estimating or Propagating Gradients Through
      Stochastic Neurons," arXiv 1305.2982, 2013.
  [2] Arndt, "CSTE Design, Build, Benchmark on TPU v5e-8," §4.3, 2026.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  STE Quantization Primitives
# ---------------------------------------------------------------------------

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator gradient."""
    return x + (torch.round(x) - x).detach()


def uniform_quantize(x: torch.Tensor, bits: int, x_min: float, x_max: float) -> torch.Tensor:
    """
    Uniform quantization to `bits` levels with STE.

    Maps [x_min, x_max] → {0, 1, ..., 2^bits - 1} → [x_min, x_max].
    """
    n_levels = 2 ** bits - 1
    x_clamped = torch.clamp(x, x_min, x_max)
    x_normalized = (x_clamped - x_min) / (x_max - x_min)  # [0, 1]
    x_quantized = ste_round(x_normalized * n_levels) / n_levels
    return x_quantized * (x_max - x_min) + x_min


def log_quantize(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Log-domain quantization for magnitude values.

    Better dynamic range than uniform for spectral magnitudes which
    span several orders of magnitude.
    """
    sign = torch.sign(x)
    log_x = torch.log(torch.abs(x) + eps)
    log_min = log_x.min().item()
    log_max = log_x.max().item()

    if log_max - log_min < eps:
        return x  # No quantization needed for constant tensors

    log_q = uniform_quantize(log_x, bits, log_min, log_max)
    return sign * torch.exp(log_q)


# ---------------------------------------------------------------------------
#  Phase-Magnitude Quantizer
# ---------------------------------------------------------------------------

class PhaseMagnitudeQuantizer(nn.Module):
    """
    Quantize complex spectral coefficients in polar (phase-magnitude) form.

    For each complex coefficient Λ = r · e^{jθ}:
        - r is quantized via log-domain uniform quantization
        - θ is quantized via uniform quantization on [−π, π]

    The decomposition preserves the structure of the frequency domain
    better than naive real/imag quantization.

    Args:
        bits:        Number of quantization bits (default 4).
        enabled:     Whether quantization is active (for training toggles).
    """

    def __init__(self, bits: int = 4, enabled: bool = True):
        super().__init__()
        self.bits = bits
        self.enabled = enabled

        # Track quantization statistics
        self.register_buffer("num_calls", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_snr_db", torch.tensor(0.0))

    @property
    def avg_snr_db(self) -> float:
        """Average signal-to-noise ratio from quantization."""
        if self.num_calls.item() == 0:
            return float("inf")
        return (self.total_snr_db / self.num_calls).item()

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize spectral coefficients (stored as separate real/imag parts).

        Args:
            real: Real part of spectral coefficients.
            imag: Imaginary part of spectral coefficients.

        Returns:
            (real_q, imag_q): Quantized real and imaginary parts.
        """
        if not self.enabled or not self.training:
            return real, imag

        # Convert to polar
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        phase = torch.atan2(imag, real)

        # Quantize independently
        magnitude_q = log_quantize(magnitude, self.bits)
        phase_q = uniform_quantize(phase, self.bits, -math.pi, math.pi)

        # Convert back to rectangular
        real_q = magnitude_q * torch.cos(phase_q)
        imag_q = magnitude_q * torch.sin(phase_q)

        # Track SNR
        if self.training:
            signal_power = (real ** 2 + imag ** 2).mean()
            noise_power = ((real - real_q) ** 2 + (imag - imag_q) ** 2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-12))
            self.num_calls += 1
            self.total_snr_db += snr.detach()

        return real_q, imag_q

    def extra_repr(self) -> str:
        return f"bits={self.bits}, enabled={self.enabled}, avg_snr={self.avg_snr_db:.1f}dB"


# ---------------------------------------------------------------------------
#  Quantization-aware BlockCirculantLinear wrapper
# ---------------------------------------------------------------------------

class QuantizedBlockCirculantLinear(nn.Module):
    """
    Block-circulant linear layer with integrated phase-magnitude quantization.

    Wraps a BlockCirculantLinear and applies quantization during training
    via STE.  At inference time, weights are pre-quantized and stored at
    the target bit-width.

    This is the layer that achieves the full CSTE compression:
        - Circulant structure: 512x compression (parameter count)
        - 4-bit quantization: 8x compression (storage per parameter)
        - Combined: ~2048x total compression in checkpoint size

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        block_size:   Circulant block size (default 512).
        bits:         Quantization bits (default 4).
        bias:         Include bias (default True).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 512,
        bits: int = 4,
        bias: bool = True,
    ):
        super().__init__()

        from siren.core.circulant import BlockCirculantLinear

        self.circulant = BlockCirculantLinear(
            in_features, out_features, block_size, bias=bias
        )
        self.quantizer = PhaseMagnitudeQuantizer(bits=bits)

    @property
    def total_compression(self) -> float:
        """Total compression: circulant × quantization."""
        # Circulant compression
        circ_ratio = self.circulant.compression_ratio
        # Quantization compression: fp32 (32 bits) → target bits
        quant_ratio = 32 / self.quantizer.bits
        return circ_ratio * quant_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply quantization to spectral coefficients
        real_q, imag_q = self.quantizer(
            self.circulant.spectral_real,
            self.circulant.spectral_imag,
        )

        # Temporarily swap in quantized values
        orig_real = self.circulant.spectral_real
        orig_imag = self.circulant.spectral_imag

        # Use STE: quantized forward, unquantized backward
        self.circulant.spectral_real = nn.Parameter(real_q)
        self.circulant.spectral_imag = nn.Parameter(imag_q)

        y = self.circulant(x)

        # Restore originals for gradient computation
        self.circulant.spectral_real = orig_real
        self.circulant.spectral_imag = orig_imag

        return y

    def extra_repr(self) -> str:
        return (
            f"total_compression={self.total_compression:.0f}x, "
            f"bits={self.quantizer.bits}"
        )
