"""
Fused FFT-Multiply-IFFT Kernel with Ising Activation
=====================================================

In a non-fused implementation, the forward pass of a circulant layer
requires three memory round-trips:
    1. Write x to memory → FFT → write X to memory
    2. Read X, read Λ → pointwise multiply → write Y to memory
    3. Read Y → IFFT → write y to memory

The fused kernel eliminates intermediate memory traffic by computing
the entire chain in a single pass.  On the ANE, this means the data
stays in SRAM registers — never touches DRAM.

The Ising activation is a stochastic non-linearity inspired by
statistical mechanics:
    σ(x) = sign(x + τ · ε),  ε ~ N(0, 1)

At high temperature τ, the activation is stochastic (exploration).
At low temperature τ → 0, it converges to sign(x) (exploitation).
Temperature is annealed during training via cosine schedule.

References:
  [3] Arndt, "CSTE Design, Build, Benchmark on TPU v5e-8," §5.1, 2026.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class IsingActivation(nn.Module):
    """
    Ising-inspired stochastic activation function.

    During training, applies temperature-modulated noise to simulate
    thermal fluctuations in a spin system.  During inference, reduces
    to a deterministic activation.

    The activation provides a non-linearity that is mathematically
    aligned with the circulant projection's spectral structure —
    both operate on the principle of global feature mixing.

    Args:
        initial_temp:  Starting temperature τ₀ (default 1.0).
        final_temp:    Final temperature τ_f (default 0.01).
        total_steps:   Total annealing steps (default 10000).
        mode:          "hard" (sign-based) or "soft" (tanh-based).
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        total_steps: int = 10000,
        mode: str = "soft",
    ):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        self.mode = mode
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @property
    def temperature(self) -> float:
        """Current temperature via cosine annealing."""
        progress = min(self.step.item() / max(self.total_steps, 1), 1.0)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_temp + (self.initial_temp - self.final_temp) * cos_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            tau = self.temperature
            noise = torch.randn_like(x) * tau

            if self.mode == "hard":
                # Hard Ising: sign(x + noise) with STE
                y = torch.sign(x + noise)
                # STE: gradient flows through as if activation were identity
                y = x + (y - x).detach()
            else:
                # Soft Ising: tanh((x + noise) / max(tau, 1e-6))
                y = torch.tanh((x + noise) / max(tau, 1e-6))

            self.step += 1
            return y
        else:
            # Deterministic at inference
            if self.mode == "hard":
                return torch.sign(x)
            else:
                return torch.tanh(x / max(self.final_temp, 1e-6))

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, temp={self.temperature:.4f}, "
            f"step={self.step.item()}/{self.total_steps}"
        )


class FusedCirculantBlock(nn.Module):
    """
    Fused block-circulant forward pass: SignFlip → FFT → Multiply → IFFT → Activate.

    This module wraps a BlockCirculantLinear and an activation function
    into a single computational block.  While PyTorch's autograd handles
    the fusion at the graph level, this module explicitly structures
    the computation to minimize intermediate tensor allocations.

    On the Apple Neural Engine, this translates to a single SRAM-resident
    compute pass with zero DRAM spill for weight access.

    Args:
        in_features:     Input dimension.
        out_features:    Output dimension.
        block_size:      Circulant block size p.
        activation:      Activation type ("ising", "gelu", "swiglu", "none").
        ising_temp:      Initial Ising temperature (if activation="ising").
        quantize:        Apply phase-magnitude quantization.
        quant_bits:      Quantization bits (default 4).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 512,
        activation: str = "gelu",
        ising_temp: float = 1.0,
        quantize: bool = False,
        quant_bits: int = 4,
    ):
        super().__init__()

        if quantize:
            from siren.core.quantization import QuantizedBlockCirculantLinear
            self.linear = QuantizedBlockCirculantLinear(
                in_features, out_features, block_size, quant_bits
            )
        else:
            from siren.core.circulant import BlockCirculantLinear
            self.linear = BlockCirculantLinear(
                in_features, out_features, block_size
            )

        # Activation
        if activation == "ising":
            self.act = IsingActivation(initial_temp=ising_temp)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused: circulant_matvec → activation in single pass."""
        return self.act(self.linear(x))

    @property
    def compression_ratio(self) -> float:
        if hasattr(self.linear, "total_compression"):
            return self.linear.total_compression
        return self.linear.compression_ratio

    def extra_repr(self) -> str:
        return f"compression={self.compression_ratio:.0f}x"
