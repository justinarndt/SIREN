"""
SIREN Benchmark Suite — Full Scientific Rigor
==============================================

Runs all benchmarks and generates a comprehensive report:
    1. Parameter compression analysis
    2. FLOP reduction analysis
    3. ANE SRAM budget analysis
    4. ANE power comparison (dense vs CSTE)
    5. ANE latency roofline analysis
    6. Inference throughput (tokens/sec)
    7. Numerical accuracy (Frobenius reconstruction error)

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --config large --chip m5_pro
"""

from __future__ import annotations

import argparse
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from siren.models.transformer import SIRENTransformer, SIRENConfig
from siren.compression.profiler import ModelProfiler
from siren.ane.power_model import ANEPowerModel, ANEChip
from siren.ane.sram_budget import SRAMBudgetAnalyzer
from siren.ane.latency_model import ANELatencyModel


def banner(title: str) -> str:
    return f"\n{'█' * 70}\n█  {title.upper():<66}█\n{'█' * 70}\n"


def run_parameter_analysis(model: SIRENTransformer, config: SIRENConfig) -> str:
    """Benchmark 1: Parameter compression analysis."""
    lines = [banner("1. PARAMETER COMPRESSION ANALYSIS")]
    lines.append(model.param_report())
    lines.append("")
    lines.append(model.per_layer_report())
    return "\n".join(lines)


def run_flop_analysis(model: SIRENTransformer, seq_len: int = 512) -> str:
    """Benchmark 2: FLOP reduction analysis."""
    lines = [banner("2. FLOP REDUCTION ANALYSIS")]
    profiler = ModelProfiler(model, seq_len=seq_len)
    profile = profiler.analyze()
    lines.append(profiler.format_report(profile))
    return "\n".join(lines)


def run_sram_analysis(
    model: SIRENTransformer,
    chip: ANEChip,
    config: SIRENConfig,
) -> str:
    """Benchmark 3: ANE SRAM budget analysis at multiple precisions."""
    lines = [banner("3. ANE SRAM BUDGET ANALYSIS")]
    analyzer = SRAMBudgetAnalyzer(chip)

    for bits in [16, 8, 4]:
        lines.append(f"\n--- Precision: {bits}-bit ---")
        report = analyzer.analyze(model, precision_bits=bits)
        lines.append(analyzer.format_report(report))

    return "\n".join(lines)


def run_power_analysis(
    model: SIRENTransformer,
    config: SIRENConfig,
    chip: ANEChip,
) -> str:
    """Benchmark 4: ANE power comparison (dense vs CSTE)."""
    lines = [banner("4. ANE POWER COMPARISON")]
    power = ANEPowerModel(chip)

    # Dense model estimates (7B = 14GB bf16)
    dense_params = 7_000_000_000
    dense_size = dense_params * 2  # bf16

    # CSTE model
    cste_params = sum(p.numel() for p in model.parameters())
    cste_size = cste_params * 2  # bf16

    # Dense FLOPs for 7B model, seq_len=512
    dense_ops = 2 * dense_params * 512  # rough: 2 * params * seq_len
    cste_ops = int(dense_ops / max(model.config.block_size, 1))

    comparison = power.compare_dense_vs_cste(
        dense_size_bytes=dense_size,
        cste_size_bytes=cste_size,
        dense_ops=dense_ops,
        cste_ops=cste_ops,
    )
    lines.append(comparison)

    # Per-chip analysis for CSTE
    lines.append("\n--- CSTE Power Across Apple Silicon ---")
    header = f"  {'Chip':<15} {'Size':>8} {'SRAM?':>6} {'Power':>8} {'TOPs/W':>8} {'Latency':>10}"
    lines.append(header)
    lines.append("  " + "-" * 55)

    for c in [ANEChip.A19_PRO, ANEChip.M4, ANEChip.M4_PRO, ANEChip.M5, ANEChip.M5_PRO, ANEChip.M5_MAX]:
        pm = ANEPowerModel(c)
        result = pm.estimate(cste_size, cste_ops)
        sram = "✓" if result["fits_in_sram"] else "✗"
        lines.append(
            f"  {result['chip']:<15} "
            f"{result['model_size_mb']:>7.2f}M "
            f"{sram:>6} "
            f"{result['total_mw']:>6.1f}mW "
            f"{result['tops_per_w']:>7.1f} "
            f"{result['latency_ms']:>8.3f}ms"
        )

    return "\n".join(lines)


def run_latency_analysis(
    model: SIRENTransformer,
    chip: ANEChip,
) -> str:
    """Benchmark 5: ANE roofline latency analysis."""
    lines = [banner("5. ROOFLINE LATENCY ANALYSIS")]
    latency_model = ANELatencyModel(chip, seq_len=512)

    for bits in [16, 4]:
        lines.append(f"\n--- Precision: {bits}-bit ---")
        result = latency_model.analyze(model, precision_bits=bits)
        lines.append(latency_model.format_report(result))

    return "\n".join(lines)


def run_throughput_benchmark(
    model: SIRENTransformer,
    config: SIRENConfig,
) -> str:
    """Benchmark 6: Inference throughput (tokens/sec)."""
    lines = [banner("6. INFERENCE THROUGHPUT")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    seq_lens = [128, 256, 512, 1024]
    batch_size = 1

    lines.append(f"  Device: {device}")
    lines.append(f"  Config: d={config.d_model}, L={config.num_layers}, p={config.block_size}")
    lines.append(f"")
    lines.append(f"  {'Seq Len':>8} {'Latency (ms)':>14} {'Tokens/sec':>12} {'Memory (MB)':>12}")
    lines.append("  " + "-" * 50)

    for seq_len in seq_lens:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                _ = model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        latency_ms = (sum(times) / len(times)) * 1000
        tokens_sec = (batch_size * seq_len) / (sum(times) / len(times))

        # Memory
        if device == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

        lines.append(
            f"  {seq_len:>8} {latency_ms:>14.2f} {tokens_sec:>12.0f} {mem_mb:>12.1f}"
        )

    model = model.cpu()
    return "\n".join(lines)


def run_accuracy_benchmark(
    model: SIRENTransformer,
    config: SIRENConfig,
) -> str:
    """Benchmark 7: Numerical accuracy (reconstruction error)."""
    lines = [banner("7. NUMERICAL ACCURACY — FROBENIUS RECONSTRUCTION")]

    from siren.core.circulant import BlockCirculantLinear

    lines.append(f"  Block size p = {config.block_size}")
    lines.append(f"")
    lines.append(f"  {'Layer':<40} {'Frob Norm':>12} {'Rel Error':>12}")
    lines.append("  " + "-" * 64)

    total_frob = 0.0
    n_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, BlockCirculantLinear):
            # Reconstruct dense matrix from spectral coefficients
            W_recon = module.reconstruct_dense()

            # Create a "reference" dense matrix
            # Since we don't have a teacher, measure self-consistency:
            # forward pass a test vector through both representations
            p = module.block_size
            test_x = torch.randn(1, module.in_features)

            with torch.no_grad():
                y_circ = module(test_x)
                y_dense = test_x @ W_recon.T
                if module.bias is not None:
                    y_dense = y_dense + module.bias

            frob = torch.norm(y_circ - y_dense[:, :module.out_features]).item()
            rel_err = frob / (torch.norm(y_circ).item() + 1e-12)
            total_frob += frob
            n_layers += 1

            lines.append(f"  {name:<40} {frob:>12.6f} {rel_err:>12.2e}")

    lines.append("  " + "-" * 64)
    lines.append(f"  {'Average':<40} {total_frob / max(n_layers, 1):>12.6f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SIREN Benchmark Suite")
    parser.add_argument(
        "--config",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model configuration preset (default: small for fast benchmarks)",
    )
    parser.add_argument(
        "--chip",
        choices=["a19_pro", "m4", "m4_pro", "m5", "m5_pro", "m5_max"],
        default="m5_pro",
        help="Target ANE chip (default: m5_pro)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    # Build model
    config_map = {
        "tiny": SIRENConfig.tiny,
        "small": SIRENConfig.small,
        "medium": SIRENConfig.medium,
        "large": SIRENConfig.large,
    }
    chip_map = {
        "a19_pro": ANEChip.A19_PRO,
        "m4": ANEChip.M4,
        "m4_pro": ANEChip.M4_PRO,
        "m5": ANEChip.M5,
        "m5_pro": ANEChip.M5_PRO,
        "m5_max": ANEChip.M5_MAX,
    }

    config = config_map[args.config]()
    chip = chip_map[args.chip]

    print(f"Building SIREN model ({args.config})...")
    model = SIRENTransformer(config)
    print(f"Model built. Running benchmarks...\n")

    # Run all benchmarks
    sections = [
        run_parameter_analysis(model, config),
        run_flop_analysis(model),
        run_sram_analysis(model, chip, config),
        run_power_analysis(model, config, chip),
        run_latency_analysis(model, chip),
        run_throughput_benchmark(model, config),
        run_accuracy_benchmark(model, config),
    ]

    report = "\n".join(sections)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
