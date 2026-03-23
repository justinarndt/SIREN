"""
ANE Latency Model — Roofline Analysis
======================================

Uses the roofline model to determine whether each layer is
compute-bound or memory-bound on the target ANE:

    Latency = max(T_compute, T_memory)

Where:
    T_compute = total_ops / peak_throughput
    T_memory  = total_bytes_accessed / memory_bandwidth

The operational intensity (OI) determines the regime:
    OI = total_ops / total_bytes_accessed

If OI > (peak_throughput / bandwidth):  compute-bound
If OI < (peak_throughput / bandwidth):  memory-bound

CSTE shifts workloads toward compute-bound (high OI) because:
    1. Fewer bytes accessed (weights are 2000x smaller)
    2. FFT ops maintain reasonable compute per output element
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import math
import torch.nn as nn

from siren.core.circulant import BlockCirculantLinear
from siren.ane.power_model import ANEChip, ANE_SPECS


@dataclass
class LayerLatency:
    """Latency analysis for a single layer."""
    name: str
    compute_us: float        # Compute time in microseconds
    memory_us: float         # Memory access time in microseconds
    total_us: float          # max(compute, memory)
    regime: str              # "compute-bound" or "memory-bound"
    operational_intensity: float  # FLOPs / bytes


@dataclass
class ModelLatency:
    """Aggregate latency analysis."""
    total_us: float
    total_ms: float
    layers: List[LayerLatency]
    compute_bound_layers: int
    memory_bound_layers: int
    bottleneck: str          # "compute" or "memory"


class ANELatencyModel:
    """
    Roofline-based latency estimator for Apple Neural Engine.

    Args:
        chip:     Target ANE chip.
        seq_len:  Sequence length for analysis.
    """

    def __init__(self, chip: ANEChip = ANEChip.M5_PRO, seq_len: int = 512):
        self.chip = chip
        self.spec = ANE_SPECS[chip]
        self.seq_len = seq_len

        # Derived specs
        self.peak_tops = self.spec.peak_tops * 1e12  # ops/sec
        self.bandwidth = self.spec.dram_bandwidth_gbps * 1e9  # bytes/sec
        self.sram_bandwidth = self.bandwidth * 10  # SRAM is ~10x DRAM bandwidth
        self.ridge_point = self.peak_tops / self.bandwidth  # OI threshold

    def analyze(self, model: nn.Module, precision_bits: int = 16) -> ModelLatency:
        """
        Run roofline analysis on the model.

        Args:
            model:          Model to analyze.
            precision_bits: Weight precision (4, 8, 16 bits).
        """
        bytes_per_param = precision_bits / 8
        total_sram = self.spec.total_sram_kb * 1024
        layers = []

        for name, module in model.named_modules():
            if isinstance(module, BlockCirculantLinear):
                lp = self._analyze_circulant(name, module, bytes_per_param, total_sram)
                layers.append(lp)

        total_us = sum(l.total_us for l in layers)
        compute_bound = sum(1 for l in layers if l.regime == "compute-bound")
        memory_bound = sum(1 for l in layers if l.regime == "memory-bound")

        return ModelLatency(
            total_us=total_us,
            total_ms=total_us / 1000,
            layers=layers,
            compute_bound_layers=compute_bound,
            memory_bound_layers=memory_bound,
            bottleneck="compute" if compute_bound > memory_bound else "memory",
        )

    def _analyze_circulant(
        self,
        name: str,
        module: BlockCirculantLinear,
        bytes_per_param: float,
        total_sram: float,
    ) -> LayerLatency:
        p = module.block_size
        n_blocks = module.num_blocks_out * module.num_blocks_in

        # FLOPs: FFT + pointwise multiply + IFFT per block, per token
        fft_ops = 5 * p * math.log2(max(p, 2))
        total_ops = n_blocks * fft_ops * self.seq_len

        # Bytes: weight reads + input reads + output writes
        weight_bytes = module.actual_params * bytes_per_param
        io_bytes = (module.in_features + module.out_features) * self.seq_len * bytes_per_param

        # Use SRAM bandwidth if weights fit
        uses_sram = weight_bytes <= total_sram
        bw = self.sram_bandwidth if uses_sram else self.bandwidth

        total_bytes = weight_bytes + io_bytes
        oi = total_ops / max(total_bytes, 1)

        compute_us = (total_ops / self.peak_tops) * 1e6
        memory_us = (total_bytes / bw) * 1e6
        total_us = max(compute_us, memory_us)

        regime = "compute-bound" if compute_us >= memory_us else "memory-bound"

        return LayerLatency(
            name=name,
            compute_us=compute_us,
            memory_us=memory_us,
            total_us=total_us,
            regime=regime,
            operational_intensity=oi,
        )

    def format_report(self, result: ModelLatency) -> str:
        """Format roofline analysis as a report."""
        lines = [
            "=" * 75,
            f"ANE Roofline Latency Analysis — {self.spec.name}",
            f"Peak: {self.spec.peak_tops} TOPS | "
            f"BW: {self.spec.dram_bandwidth_gbps} GB/s | "
            f"Ridge: {self.ridge_point:.1f} ops/byte",
            "=" * 75,
            "",
            f"  Total latency:          {result.total_us:>10.1f} μs "
            f"({result.total_ms:.3f} ms)",
            f"  Compute-bound layers:   {result.compute_bound_layers:>10d}",
            f"  Memory-bound layers:    {result.memory_bound_layers:>10d}",
            f"  Overall bottleneck:     {result.bottleneck:>10s}",
            "",
            f"  {'Layer':<35} {'Compute':>8} {'Memory':>8} "
            f"{'Total':>8} {'OI':>8} {'Regime':<15}",
            "  " + "-" * 82,
        ]

        for lp in result.layers:
            regime_symbol = "⚙️" if lp.regime == "compute-bound" else "📦"
            lines.append(
                f"  {lp.name:<35} "
                f"{lp.compute_us:>7.1f}μ "
                f"{lp.memory_us:>7.1f}μ "
                f"{lp.total_us:>7.1f}μ "
                f"{lp.operational_intensity:>7.1f} "
                f"{regime_symbol} {lp.regime}"
            )

        lines.append("=" * 75)
        return "\n".join(lines)
