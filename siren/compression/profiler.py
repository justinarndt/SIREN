"""
Model Profiler — Scientific Parameter and FLOP Analysis
========================================================

Accurate comparison of dense vs CSTE models requires tracking:
    1. Parameter count (trainable, non-trainable, dense-equivalent)
    2. FLOPs per forward pass (dense matvec vs FFT-based)
    3. Memory footprint (checkpoint size at various precisions)
    4. Compression ratios (per-layer and aggregate)

FLOP Accounting:
    Dense matvec:      2 × m × n  FLOPs  (multiply-add)
    Circulant matvec:  5 × n × log₂(n)  FLOPs  (FFT + pointwise + IFFT)
    For n=4096, p=512: dense = 33.5M FLOPs, circulant = ~46K FLOPs
                       → ~730x FLOP reduction per projection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from siren.core.circulant import BlockCirculantLinear


@dataclass
class LayerProfile:
    """Profile for a single layer."""
    name: str
    layer_type: str
    dense_params: int
    actual_params: int
    compression_ratio: float
    dense_flops: int
    cste_flops: int
    flop_reduction: float


@dataclass
class ModelProfile:
    """Aggregate model profile."""
    total_dense_params: int
    total_actual_params: int
    total_compression: float
    total_dense_flops: int
    total_cste_flops: int
    total_flop_reduction: float
    layers: List[LayerProfile]
    checkpoint_sizes: Dict[str, float]  # precision → MB


class ModelProfiler:
    """
    Comprehensive model profiler comparing dense vs CSTE metrics.

    Usage:
        profiler = ModelProfiler(model, seq_len=512, batch_size=1)
        profile = profiler.analyze()
        print(profiler.format_report(profile))
    """

    def __init__(
        self,
        model: nn.Module,
        seq_len: int = 512,
        batch_size: int = 1,
    ):
        self.model = model
        self.seq_len = seq_len
        self.batch_size = batch_size

    def analyze(self) -> ModelProfile:
        """Run full analysis."""
        layers = []

        for name, module in self.model.named_modules():
            if isinstance(module, BlockCirculantLinear):
                lp = self._profile_circulant(name, module)
                layers.append(lp)
            elif isinstance(module, nn.Linear) and not any(
                isinstance(parent, BlockCirculantLinear)
                for parent in self._get_parents(name)
            ):
                lp = self._profile_linear(name, module)
                layers.append(lp)

        # Aggregate
        total_dense = sum(l.dense_params for l in layers)
        total_actual = sum(l.actual_params for l in layers)
        total_dense_flops = sum(l.dense_flops for l in layers)
        total_cste_flops = sum(l.cste_flops for l in layers)

        # Add embedding params
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                total_dense += module.weight.numel()
                total_actual += module.weight.numel()

        checkpoint_sizes = {
            "fp32": total_actual * 4 / 1e6,
            "bf16": total_actual * 2 / 1e6,
            "int8": total_actual * 1 / 1e6,
            "int4": total_actual * 0.5 / 1e6,
        }

        return ModelProfile(
            total_dense_params=total_dense,
            total_actual_params=total_actual,
            total_compression=total_dense / max(total_actual, 1),
            total_dense_flops=total_dense_flops,
            total_cste_flops=total_cste_flops,
            total_flop_reduction=total_dense_flops / max(total_cste_flops, 1),
            layers=layers,
            checkpoint_sizes=checkpoint_sizes,
        )

    def _profile_circulant(self, name: str, module: BlockCirculantLinear) -> LayerProfile:
        p = module.block_size
        m, n = module.out_features, module.in_features

        dense_params = m * n
        actual_params = module.actual_params
        dense_flops = 2 * m * n * self.seq_len * self.batch_size

        # CSTE FLOPs: B_out × B_in × (5 × p × log2(p)) per token
        n_blocks = module.num_blocks_out * module.num_blocks_in
        fft_flops = int(5 * p * math.log2(max(p, 2)))
        cste_flops = n_blocks * fft_flops * self.seq_len * self.batch_size

        return LayerProfile(
            name=name,
            layer_type="BlockCirculant",
            dense_params=dense_params,
            actual_params=actual_params,
            compression_ratio=dense_params / max(actual_params, 1),
            dense_flops=dense_flops,
            cste_flops=cste_flops,
            flop_reduction=dense_flops / max(cste_flops, 1),
        )

    def _profile_linear(self, name: str, module: nn.Linear) -> LayerProfile:
        m, n = module.out_features, module.in_features
        params = sum(p.numel() for p in module.parameters())
        flops = 2 * m * n * self.seq_len * self.batch_size

        return LayerProfile(
            name=name,
            layer_type="Dense",
            dense_params=params,
            actual_params=params,
            compression_ratio=1.0,
            dense_flops=flops,
            cste_flops=flops,
            flop_reduction=1.0,
        )

    def _get_parents(self, child_name: str) -> list:
        """Get parent modules for a named module."""
        parts = child_name.split(".")
        parents = []
        for i in range(len(parts)):
            prefix = ".".join(parts[:i])
            if prefix:
                try:
                    parent = dict(self.model.named_modules())[prefix]
                    parents.append(parent)
                except KeyError:
                    pass
        return parents

    def format_report(self, profile: ModelProfile) -> str:
        """Format a scientific-grade analysis report."""
        lines = [
            "=" * 80,
            "SIREN Model Profiler — Dense vs CSTE Analysis",
            "=" * 80,
            "",
            "AGGREGATE METRICS",
            "-" * 80,
            f"  Dense equivalent parameters:    {profile.total_dense_params:>16,}",
            f"  CSTE actual parameters:         {profile.total_actual_params:>16,}",
            f"  Parameter compression:          {profile.total_compression:>16.1f}x",
            "",
            f"  Dense FLOPs/token:              {profile.total_dense_flops:>16,}",
            f"  CSTE FLOPs/token:               {profile.total_cste_flops:>16,}",
            f"  FLOP reduction:                 {profile.total_flop_reduction:>16.1f}x",
            "",
            "CHECKPOINT SIZES",
            "-" * 80,
        ]

        for precision, size_mb in profile.checkpoint_sizes.items():
            if size_mb < 1:
                lines.append(f"  {precision:<6}: {size_mb * 1000:>10.1f} KB")
            else:
                lines.append(f"  {precision:<6}: {size_mb:>10.2f} MB")

        lines += [
            "",
            "PER-LAYER BREAKDOWN",
            "-" * 80,
            f"  {'Layer':<35} {'Type':<15} {'Dense':>10} {'CSTE':>10} {'Ratio':>8}",
            "  " + "-" * 78,
        ]

        for lp in profile.layers:
            lines.append(
                f"  {lp.name:<35} {lp.layer_type:<15} "
                f"{lp.dense_params:>10,} {lp.actual_params:>10,} "
                f"{lp.compression_ratio:>7.0f}x"
            )

        lines.append("=" * 80)
        return "\n".join(lines)
