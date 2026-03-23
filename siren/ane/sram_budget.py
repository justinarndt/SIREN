"""
ANE SRAM Budget Analyzer
========================

Maps model weight tensors to the ANE's SRAM hierarchy to determine
which layers fit in which tier and whether any weights spill to DRAM.

SRAM Hierarchy (typical Apple Silicon):
    ┌────────────────────────────────┐
    │  L1 SRAM (per core):  48-64KB │  ← Fastest, per-MAC-unit
    │  × 16 cores = 768KB-1MB       │
    ├────────────────────────────────┤
    │  L2 SRAM (shared):   4-6 MB   │  ← Shared across all cores
    ├────────────────────────────────┤
    │  DRAM (LPDDR5):      6-192 GB │  ← 20-30x more energy per byte
    └────────────────────────────────┘

CSTE target: entire model in L2 SRAM (4-6 MB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from siren.core.circulant import BlockCirculantLinear
from siren.ane.power_model import ANEChip, ANE_SPECS


@dataclass
class LayerSRAMAllocation:
    """SRAM allocation for a single layer."""
    name: str
    size_bytes: int
    tier: str           # "L1", "L2", or "DRAM_SPILL"
    fits_l1: bool
    fits_l2: bool


@dataclass
class SRAMBudgetReport:
    """Full SRAM budget analysis."""
    chip_name: str
    l1_capacity_kb: float
    l2_capacity_kb: float
    total_sram_kb: float
    model_size_bytes: int
    fits_entirely_in_sram: bool
    layers: List[LayerSRAMAllocation]
    l1_utilized_bytes: int
    l2_utilized_bytes: int
    dram_spill_bytes: int
    utilization_pct: float


class SRAMBudgetAnalyzer:
    """
    Analyze how model weights map to the ANE SRAM hierarchy.

    Determines:
        1. Which layers fit in L1 (per-core SRAM)
        2. Which layers fit in L2 (shared SRAM)
        3. Which layers spill to DRAM (bad for power)
        4. Total SRAM utilization percentage

    Args:
        chip: Target ANE chip configuration.
    """

    def __init__(self, chip: ANEChip = ANEChip.M5_PRO):
        self.chip = chip
        self.spec = ANE_SPECS[chip]

    def analyze(
        self,
        model: nn.Module,
        precision_bits: int = 16,
    ) -> SRAMBudgetReport:
        """
        Run SRAM budget analysis.

        Args:
            model:          Model to analyze.
            precision_bits: Storage precision (4, 8, or 16 bits).
        """
        bytes_per_param = precision_bits / 8
        l1_cap = self.spec.sram_l1_kb * 1024  # per core, in bytes
        l2_cap = self.spec.sram_l2_kb * 1024

        layers = []
        l1_used = 0
        l2_used = 0
        dram_spill = 0
        total_size = 0

        for name, module in model.named_modules():
            if isinstance(module, BlockCirculantLinear):
                n_params = module.actual_params
                size = int(n_params * bytes_per_param)
                total_size += size

                fits_l1 = size <= l1_cap
                fits_l2 = size <= l2_cap

                if fits_l1:
                    tier = "L1"
                    l1_used += size
                elif fits_l2:
                    tier = "L2"
                    l2_used += size
                else:
                    tier = "DRAM_SPILL"
                    dram_spill += size

                layers.append(LayerSRAMAllocation(
                    name=name, size_bytes=size, tier=tier,
                    fits_l1=fits_l1, fits_l2=fits_l2,
                ))

            elif isinstance(module, (nn.Embedding, nn.Linear)):
                n_params = sum(p.numel() for p in module.parameters(recurse=False))
                if n_params == 0:
                    continue
                size = int(n_params * bytes_per_param)
                total_size += size

                fits_l1 = size <= l1_cap
                fits_l2 = size <= l2_cap

                if fits_l2:
                    tier = "L2"
                    l2_used += size
                else:
                    tier = "DRAM_SPILL"
                    dram_spill += size

                layers.append(LayerSRAMAllocation(
                    name=name, size_bytes=size, tier=tier,
                    fits_l1=fits_l1, fits_l2=fits_l2,
                ))

        total_sram = (self.spec.sram_l1_kb * self.spec.num_cores +
                      self.spec.sram_l2_kb) * 1024

        return SRAMBudgetReport(
            chip_name=self.spec.name,
            l1_capacity_kb=self.spec.sram_l1_kb * self.spec.num_cores,
            l2_capacity_kb=self.spec.sram_l2_kb,
            total_sram_kb=total_sram / 1024,
            model_size_bytes=total_size,
            fits_entirely_in_sram=total_size <= total_sram,
            layers=layers,
            l1_utilized_bytes=l1_used,
            l2_utilized_bytes=l2_used,
            dram_spill_bytes=dram_spill,
            utilization_pct=min(100.0, (total_size / total_sram) * 100),
        )

    def format_report(self, report: SRAMBudgetReport) -> str:
        """Format SRAM budget analysis as a report."""
        status = "✓ WEIGHT-IN-SRAM MODE" if report.fits_entirely_in_sram else "✗ DRAM SPILL"

        lines = [
            "=" * 70,
            f"SRAM Budget Analysis — {report.chip_name}",
            "=" * 70,
            f"  Status: {status}",
            "",
            f"  L1 SRAM (per-core): {report.l1_capacity_kb:.0f} KB",
            f"  L2 SRAM (shared):   {report.l2_capacity_kb:.0f} KB",
            f"  Total SRAM:         {report.total_sram_kb:.0f} KB "
            f"({report.total_sram_kb / 1024:.1f} MB)",
            "",
            f"  Model size:         {report.model_size_bytes:,} bytes "
            f"({report.model_size_bytes / 1e6:.2f} MB)",
            f"  SRAM utilization:   {report.utilization_pct:.1f}%",
            "",
            f"  L1 utilized:        {report.l1_utilized_bytes:,} bytes",
            f"  L2 utilized:        {report.l2_utilized_bytes:,} bytes",
            f"  DRAM spill:         {report.dram_spill_bytes:,} bytes",
            "",
            "LAYER ALLOCATION",
            "-" * 70,
            f"  {'Layer':<40} {'Size':>10} {'Tier':>10}",
            "  " + "-" * 60,
        ]

        for la in report.layers:
            size_str = f"{la.size_bytes:,}"
            tier_emoji = {"L1": "🟢", "L2": "🟡", "DRAM_SPILL": "🔴"}
            lines.append(
                f"  {la.name:<40} {size_str:>10} "
                f"{tier_emoji.get(la.tier, '')} {la.tier}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)
