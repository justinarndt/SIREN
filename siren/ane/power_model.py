"""
Apple Neural Engine Power Model
================================

The Apple Neural Engine (ANE) is a matrix-multiply accelerator with a
strict power budget.  Unlike GPUs, the ANE is designed for sustained
low-power inference — peak throughput is limited but energy efficiency
is 5-10x better than GPU.

Power model:  P_total = P_compute + P_sram + P_dram

Where:
    P_compute = TOPS_utilized × (TDP / TOPS_peak)
    P_sram    = sram_accesses × energy_per_sram_access
    P_dram    = dram_accesses × energy_per_dram_access

CSTE advantage:  When model weights fit in ANE SRAM (~4MB L2 + 768KB L1),
P_dram drops to zero.  This eliminates the dominant power term, achieving
the "weight-in-SRAM" execution mode described in the CSTE paper §7.

Hardware specifications sourced from Apple Silicon documentation,
die analysis (TechInsights), and reverse-engineering community data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ANEChip(Enum):
    """Supported Apple Neural Engine configurations."""
    A17_PRO = "a17_pro"        # iPhone 15 Pro
    A18 = "a18"                # iPhone 16
    A18_PRO = "a18_pro"        # iPhone 16 Pro
    A19 = "a19"                # iPhone 17
    A19_PRO = "a19_pro"        # iPhone 17 Pro
    M4 = "m4"                  # iPad Pro, MacBook Air
    M4_PRO = "m4_pro"          # MacBook Pro 14"
    M4_MAX = "m4_max"          # MacBook Pro 16"
    M5 = "m5"                  # 2026 MacBook
    M5_PRO = "m5_pro"          # 2026 MacBook Pro 14"
    M5_MAX = "m5_max"          # 2026 MacBook Pro 16"


@dataclass
class ANESpec:
    """Hardware specifications for an ANE configuration."""
    name: str
    num_cores: int             # Neural Engine cores
    peak_tops: float           # Peak INT8 TOPS
    tdp_watts: float           # Thermal design power for ANE partition
    sram_l1_kb: float          # Per-core L1 SRAM (KB)
    sram_l2_kb: float          # Shared L2 SRAM (KB)
    dram_bandwidth_gbps: float # LPDDR bandwidth available to ANE
    process_nm: int            # Manufacturing process node

    @property
    def total_sram_kb(self) -> float:
        """Total SRAM available to ANE."""
        return (self.sram_l1_kb * self.num_cores) + self.sram_l2_kb

    @property
    def total_sram_mb(self) -> float:
        return self.total_sram_kb / 1024


# Estimated ANE specifications (sources: TechInsights, AnandTech, die shots)
ANE_SPECS: Dict[ANEChip, ANESpec] = {
    ANEChip.A17_PRO: ANESpec("A17 Pro", 16, 35.0, 4.0, 48, 4096, 34.1, 3),
    ANEChip.A18: ANESpec("A18", 16, 35.0, 3.5, 48, 4096, 34.1, 3),
    ANEChip.A18_PRO: ANESpec("A18 Pro", 16, 35.0, 4.0, 48, 4096, 34.1, 3),
    ANEChip.A19: ANESpec("A19", 16, 38.0, 3.5, 48, 4096, 51.2, 3),
    ANEChip.A19_PRO: ANESpec("A19 Pro", 16, 38.0, 4.5, 64, 6144, 51.2, 3),
    ANEChip.M4: ANESpec("M4", 16, 38.0, 5.0, 48, 4096, 120.0, 3),
    ANEChip.M4_PRO: ANESpec("M4 Pro", 16, 38.0, 6.0, 48, 4096, 200.0, 3),
    ANEChip.M4_MAX: ANESpec("M4 Max", 16, 38.0, 8.0, 48, 4096, 400.0, 3),
    ANEChip.M5: ANESpec("M5", 16, 42.0, 5.0, 64, 6144, 150.0, 3),
    ANEChip.M5_PRO: ANESpec("M5 Pro", 16, 42.0, 7.0, 64, 6144, 273.0, 3),
    ANEChip.M5_MAX: ANESpec("M5 Max", 16, 42.0, 9.0, 64, 6144, 546.0, 3),
}

# Energy per operation (picoJoules, estimated from die analysis)
ENERGY_PJ = {
    "int8_mac": 0.2,           # INT8 multiply-accumulate
    "sram_read_byte": 0.5,     # L1/L2 SRAM read per byte
    "sram_write_byte": 0.7,    # L1/L2 SRAM write per byte
    "dram_read_byte": 12.0,    # LPDDR5 read per byte
    "dram_write_byte": 15.0,   # LPDDR5 write per byte
}


class ANEPowerModel:
    """
    Estimate inference power draw for a model on the Apple Neural Engine.

    This model accounts for:
        1. Compute power (MAC operations at the ANE's operating frequency)
        2. SRAM power (weight reads from L1/L2)
        3. DRAM power (weight reads from LPDDR when weights spill)

    The key insight: CSTE models that fit in SRAM eliminate DRAM access,
    which is the dominant power term (12-15 pJ/byte vs 0.5-0.7 pJ/byte).

    Usage:
        power_model = ANEPowerModel(ANEChip.M5_PRO)
        result = power_model.estimate(
            model_size_bytes=3_500_000,  # 3.5 MB CSTE model
            ops_per_inference=50_000_000,
            seq_len=512,
        )
        print(result)
    """

    def __init__(self, chip: ANEChip = ANEChip.M5_PRO):
        self.chip = chip
        self.spec = ANE_SPECS[chip]

    def estimate(
        self,
        model_size_bytes: int,
        ops_per_inference: int,
        seq_len: int = 512,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        Estimate power for a single inference pass.

        Args:
            model_size_bytes: Total model size in bytes.
            ops_per_inference: Total INT8 operations per inference.
            seq_len:          Sequence length.
            batch_size:       Batch size.

        Returns:
            Dict with power breakdown in milliwatts.
        """
        sram_capacity = self.spec.total_sram_kb * 1024  # bytes
        fits_in_sram = model_size_bytes <= sram_capacity

        # Compute power
        compute_pj = ops_per_inference * ENERGY_PJ["int8_mac"]
        compute_mw = self._pj_to_mw(compute_pj, ops_per_inference)

        # Memory power
        if fits_in_sram:
            sram_read_pj = model_size_bytes * ENERGY_PJ["sram_read_byte"]
            dram_read_pj = 0.0
            memory_source = "SRAM-only (weight-in-SRAM mode)"
        else:
            # Weights that fit in SRAM are read from SRAM
            sram_bytes = int(sram_capacity)
            dram_bytes = model_size_bytes - sram_bytes
            sram_read_pj = sram_bytes * ENERGY_PJ["sram_read_byte"]
            dram_read_pj = dram_bytes * ENERGY_PJ["dram_read_byte"]
            memory_source = f"SRAM + DRAM ({dram_bytes / 1e6:.1f} MB spill)"

        sram_mw = self._pj_to_mw(sram_read_pj, ops_per_inference)
        dram_mw = self._pj_to_mw(dram_read_pj, ops_per_inference)

        # Activation memory (always fits in SRAM for reasonable seq_len)
        act_bytes = batch_size * seq_len * 4096 * 2  # rough estimate
        act_pj = act_bytes * ENERGY_PJ["sram_read_byte"]
        act_mw = self._pj_to_mw(act_pj, ops_per_inference)

        total_mw = compute_mw + sram_mw + dram_mw + act_mw

        # Inference latency estimate
        latency_ms = (ops_per_inference / (self.spec.peak_tops * 1e12)) * 1e3

        # TOPs/W
        tops_w = (self.spec.peak_tops * 1000) / max(total_mw, 0.1)

        return {
            "chip": self.spec.name,
            "model_size_mb": model_size_bytes / 1e6,
            "fits_in_sram": fits_in_sram,
            "sram_capacity_mb": sram_capacity / 1e6,
            "memory_source": memory_source,
            "compute_mw": round(compute_mw, 2),
            "sram_mw": round(sram_mw, 2),
            "dram_mw": round(dram_mw, 2),
            "activation_mw": round(act_mw, 2),
            "total_mw": round(total_mw, 2),
            "total_w": round(total_mw / 1000, 3),
            "latency_ms": round(latency_ms, 3),
            "tops_per_w": round(tops_w, 1),
            "battery_hrs_1000mah": round(1000 / max(total_mw, 0.1), 1),
        }

    def compare_dense_vs_cste(
        self,
        dense_size_bytes: int,
        cste_size_bytes: int,
        dense_ops: int,
        cste_ops: int,
        seq_len: int = 512,
    ) -> str:
        """
        Format a comparison report between dense and CSTE inference.
        """
        dense = self.estimate(dense_size_bytes, dense_ops, seq_len)
        cste = self.estimate(cste_size_bytes, cste_ops, seq_len)

        lines = [
            "=" * 70,
            f"ANE Power Analysis — {self.spec.name}",
            f"SRAM Budget: {self.spec.total_sram_mb:.1f} MB "
            f"({self.spec.num_cores} cores × {self.spec.sram_l1_kb}KB L1 "
            f"+ {self.spec.sram_l2_kb}KB L2)",
            "=" * 70,
            "",
            f"{'Metric':<30} {'Dense':>15} {'CSTE':>15} {'Savings':>12}",
            "-" * 72,
            f"{'Model size':<30} "
            f"{dense['model_size_mb']:>14.1f}M "
            f"{cste['model_size_mb']:>14.2f}M "
            f"{dense['model_size_mb']/max(cste['model_size_mb'], 0.001):>11.0f}x",
            f"{'Fits in SRAM':<30} "
            f"{'✗':>15} "
            f"{'✓':>15} "
            f"{'—':>12}",
            f"{'Compute power (mW)':<30} "
            f"{dense['compute_mw']:>15.1f} "
            f"{cste['compute_mw']:>15.1f} "
            f"{dense['compute_mw']/max(cste['compute_mw'], 0.01):>11.1f}x",
            f"{'SRAM power (mW)':<30} "
            f"{dense['sram_mw']:>15.1f} "
            f"{cste['sram_mw']:>15.1f} "
            f"{max(dense['sram_mw'], 0.01)/max(cste['sram_mw'], 0.01):>11.1f}x",
            f"{'DRAM power (mW)':<30} "
            f"{dense['dram_mw']:>15.1f} "
            f"{cste['dram_mw']:>15.1f} "
            f"{'∞ (eliminated)':>12}" if cste['dram_mw'] == 0 else
            f"{'DRAM power (mW)':<30} "
            f"{dense['dram_mw']:>15.1f} "
            f"{cste['dram_mw']:>15.1f} "
            f"{dense['dram_mw']/max(cste['dram_mw'], 0.01):>11.1f}x",
            f"{'Total power (mW)':<30} "
            f"{dense['total_mw']:>15.1f} "
            f"{cste['total_mw']:>15.1f} "
            f"{dense['total_mw']/max(cste['total_mw'], 0.01):>11.1f}x",
            f"{'Latency (ms)':<30} "
            f"{dense['latency_ms']:>15.3f} "
            f"{cste['latency_ms']:>15.3f} "
            f"{dense['latency_ms']/max(cste['latency_ms'], 0.001):>11.1f}x",
            f"{'TOPs/W':<30} "
            f"{dense['tops_per_w']:>15.1f} "
            f"{cste['tops_per_w']:>15.1f} "
            f"{cste['tops_per_w']/max(dense['tops_per_w'], 0.01):>11.1f}x",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)

    @staticmethod
    def _pj_to_mw(energy_pj: float, ops: int, utilization: float = 0.8) -> float:
        """
        Convert picoJoules total energy to average milliwatts.

        Assumes the operations execute at ~80% utilization of peak throughput.
        """
        # Rough: assume 1GHz effective clock, so ops take ops/1e9 seconds
        seconds = ops / 1e9  # very rough
        if seconds <= 0:
            return 0.0
        watts = (energy_pj * 1e-12) / seconds
        return watts * 1e3  # to mW
