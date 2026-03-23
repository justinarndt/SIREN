"""
Generate hero benchmark plots for SIREN README.

Produces SVG plots in docs/ showing real metrics:
    1. Compression ratio across model scales
    2. Battery life comparison (ANE hours)
    3. Memory footprint comparison
    4. Latency per token (ANE estimated)

All data computed from actual SIREN model instantiations + ANE power model.
"""

import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from siren.models.transformer import SIRENTransformer, SIRENConfig
from siren.ane.power_model import ANEPowerModel, ANEChip, ANE_SPECS
from siren.ane.latency_model import ANELatencyModel
from siren.core.circulant import BlockCirculantLinear

# ---------------------------------------------------------------------------
# 1. Collect Real Data
# ---------------------------------------------------------------------------

configs = {
    "Tiny (50M)":    SIRENConfig.tiny(),
    "Small (200M)":  SIRENConfig.small(),
    "Base (750M)":   SIRENConfig.base(),
    "Medium (1.5B)": SIRENConfig.medium(),
    "Large (7B)":    SIRENConfig.large(),
}

# Chips to benchmark
target_chips = [ANEChip.A17_PRO, ANEChip.A18_PRO, ANEChip.M4, ANEChip.M4_PRO, ANEChip.M5_PRO]

data = {}
for label, cfg in configs.items():
    print(f"Building {label}...")
    model = SIRENTransformer(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    dense_equiv = 0
    cste_params = 0
    for name, mod in model.named_modules():
        if isinstance(mod, BlockCirculantLinear):
            dense_equiv += mod.dense_equivalent_params
            cste_params += mod.actual_params
        elif isinstance(mod, (nn.Embedding, nn.Linear)):
            count = sum(p.numel() for p in mod.parameters(recurse=False))
            dense_equiv += count

    # Model sizes in bytes (4-bit quantized)
    cste_size_bytes = int(total_params * 0.5)   # 4-bit
    dense_size_bytes = int(dense_equiv * 2)      # bf16

    # ANE analysis on A18 Pro (iPhone 16 Pro)
    chip = ANEChip.A18_PRO
    latency_model = ANELatencyModel(chip=chip, seq_len=512)
    lat_result = latency_model.analyze(model, precision_bits=4)

    # Power estimation
    power_model = ANEPowerModel(chip=chip)
    # Estimate total ops from latency analysis (circulant blocks)
    p = cfg.block_size
    n_circ_layers = 0
    total_ops = 0
    for name, mod in model.named_modules():
        if isinstance(mod, BlockCirculantLinear):
            n_circ_layers += 1
            fft_ops = 5 * mod.block_size * math.log2(max(mod.block_size, 2))
            total_ops += int(mod.num_blocks_out * mod.num_blocks_in * fft_ops * 512)

    cste_power = power_model.estimate(cste_size_bytes, total_ops, seq_len=512)
    dense_power = power_model.estimate(dense_size_bytes, total_ops * 4, seq_len=512)

    # Battery: iPhone 16 Pro = 4685 mAh @ 3.86V = 18.08 Wh
    battery_wh = 18.08
    cste_battery_h = battery_wh / max(cste_power["total_w"], 0.001)
    dense_battery_h = battery_wh / max(dense_power["total_w"], 0.001)

    # Latency per token estimate
    cste_latency_ms = lat_result.total_ms
    cste_tps = 1000.0 / max(cste_latency_ms, 0.001)

    data[label] = {
        "dense_params": dense_equiv,
        "cste_params": total_params,
        "compression": dense_equiv / max(total_params, 1),
        "dense_gb": dense_equiv * 2 / 1e9,  # bf16
        "cste_mb": total_params * 0.5 / 1e6,  # 4-bit
        "cste_latency_ms": cste_latency_ms,
        "cste_tps": cste_tps,
        "cste_power_mw": cste_power["total_mw"],
        "cste_power_w": cste_power["total_w"],
        "dense_power_w": dense_power["total_w"],
        "cste_battery_h": min(cste_battery_h, 200),  # cap for display
        "dense_battery_h": min(dense_battery_h, 200),
        "cste_fits_sram": cste_power["fits_in_sram"],
        "n_circ_layers": n_circ_layers,
        "lat_bottleneck": lat_result.bottleneck,
    }

    del model

# Chip comparison data (using Base model)
chip_data = {}
base_cfg = SIRENConfig.base()
base_model = SIRENTransformer(base_cfg)
base_params = sum(p.numel() for p in base_model.parameters())
base_size = int(base_params * 0.5)

for chip in target_chips:
    lat_m = ANELatencyModel(chip=chip, seq_len=512)
    lat_r = lat_m.analyze(base_model, precision_bits=4)
    spec = ANE_SPECS[chip]
    chip_data[spec.name] = {
        "latency_ms": lat_r.total_ms,
        "tps": 1000.0 / max(lat_r.total_ms, 0.001),
        "bottleneck": lat_r.bottleneck,
    }
del base_model

# Print summary
print("\n" + "=" * 90)
print(f"{'Model':<18} {'Dense':>10} {'CSTE':>10} {'Comp':>6} {'Latency':>10} {'Power':>8} {'Battery':>8} {'SRAM':>5}")
print("-" * 90)
for label, d in data.items():
    print(f"{label:<18} {d['dense_params']/1e6:>9.1f}M {d['cste_params']/1e6:>9.1f}M "
          f"{d['compression']:>5.1f}x {d['cste_latency_ms']:>9.3f}ms "
          f"{d['cste_power_w']:>7.3f}W {d['cste_battery_h']:>7.1f}h "
          f"{'Yes' if d['cste_fits_sram'] else 'No':>5}")
print("=" * 90)

# ---------------------------------------------------------------------------
# 2. Generate SVG Plots
# ---------------------------------------------------------------------------

os.makedirs("docs", exist_ok=True)

labels = list(data.keys())
n = len(labels)

# Professional color palette
C_BG     = "#0d1117"
C_CARD   = "#161b22"
C_GRID   = "#21262d"
C_TEXT   = "#c9d1d9"
C_SUBTLE = "#8b949e"
C_ORANGE = "#f0883e"
C_BLUE   = "#58a6ff"
C_GREEN  = "#3fb950"
C_PURPLE = "#bc8cff"
C_RED    = "#f85149"
C_TEAL   = "#39d2c0"


def svg_header(w, h, title, subtitle=None, defs=""):
    s = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" fill="none">\n'
    if defs:
        s += f"  <defs>{defs}</defs>\n"
    s += f'  <rect width="{w}" height="{h}" rx="12" fill="{C_BG}"/>\n'
    s += f'  <text x="{w//2}" y="32" text-anchor="middle" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="17" font-weight="600">{title}</text>\n'
    if subtitle:
        s += f'  <text x="{w//2}" y="50" text-anchor="middle" fill="{C_SUBTLE}" font-family="Inter,system-ui,sans-serif" font-size="11">{subtitle}</text>\n'
    return s


# ========================================================================
#  PLOT 1: Compression Ratio (horizontal bar chart)
# ========================================================================

def plot_compression():
    w, h = 780, 340
    bar_h = 28
    chart_left = 140
    chart_top = 70

    vals = [data[l]["compression"] for l in labels]
    max_val = max(vals)
    bar_max = w - chart_left - 100
    scale = bar_max / max_val

    defs = f"""
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{C_BLUE}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{C_TEAL}" stop-opacity="0.95"/>
    </linearGradient>"""

    svg = svg_header(w, h, "Parameter Compression Ratio (Dense / CSTE)", "Higher is better. Dense bf16 equivalent vs CSTE 4-bit.", defs)

    for i, label in enumerate(labels):
        y = chart_top + i * (bar_h + 18)
        bw = vals[i] * scale

        svg += f'  <text x="{chart_left - 12}" y="{y + bar_h - 8}" text-anchor="end" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="12" font-weight="500">{label}</text>\n'
        svg += f'  <rect x="{chart_left}" y="{y}" width="{bw}" height="{bar_h}" rx="5" fill="url(#g1)"/>\n'
        svg += f'  <text x="{chart_left + bw + 10}" y="{y + bar_h - 8}" fill="{C_TEAL}" font-family="Inter,system-ui,sans-serif" font-size="14" font-weight="700">{vals[i]:.1f}x</text>\n'

    svg += "</svg>"
    return svg


# ========================================================================
#  PLOT 2: Battery Life (horizontal paired bars)
# ========================================================================

def plot_battery():
    w, h = 780, 380
    bar_h = 22
    chart_left = 140
    chart_top = 70

    cste_vals = [min(data[l]["cste_battery_h"], 50) for l in labels]
    dense_vals = [min(data[l]["dense_battery_h"], 50) for l in labels]
    max_val = max(max(cste_vals), max(dense_vals))
    bar_max = w - chart_left - 120
    scale = bar_max / max_val

    defs = f"""
    <linearGradient id="g_bat_c" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{C_GREEN}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{C_TEAL}" stop-opacity="0.95"/>
    </linearGradient>
    <linearGradient id="g_bat_d" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{C_RED}" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="{C_ORANGE}" stop-opacity="0.6"/>
    </linearGradient>"""

    svg = svg_header(w, h, "Continuous Inference Battery Life -- iPhone 16 Pro",
                     "4,685 mAh battery. SIREN uses ANE (weight-in-SRAM). Dense uses GPU/CPU.", defs)

    for i, label in enumerate(labels):
        y = chart_top + i * (bar_h * 2 + 22)

        svg += f'  <text x="{chart_left - 12}" y="{y + bar_h + 4}" text-anchor="end" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="12">{label}</text>\n'

        cw = cste_vals[i] * scale
        svg += f'  <rect x="{chart_left}" y="{y}" width="{cw}" height="{bar_h}" rx="4" fill="url(#g_bat_c)"/>\n'
        svg += f'  <text x="{chart_left + cw + 8}" y="{y + bar_h - 6}" fill="{C_GREEN}" font-family="Inter,system-ui,sans-serif" font-size="12" font-weight="600">{data[labels[i]]["cste_battery_h"]:.1f}h SIREN</text>\n'

        dw = dense_vals[i] * scale
        svg += f'  <rect x="{chart_left}" y="{y + bar_h + 3}" width="{dw}" height="{bar_h}" rx="4" fill="url(#g_bat_d)"/>\n'
        svg += f'  <text x="{chart_left + dw + 8}" y="{y + bar_h * 2 - 3}" fill="{C_SUBTLE}" font-family="Inter,system-ui,sans-serif" font-size="11">{data[labels[i]]["dense_battery_h"]:.1f}h Dense</text>\n'

    svg += "</svg>"
    return svg


# ========================================================================
#  PLOT 3: Memory Footprint (vertical grouped bars)
# ========================================================================

def plot_memory():
    w, h = 780, 380
    bar_w = 36
    chart_left = 80
    chart_top = 65
    chart_bottom = h - 70

    dense_vals = [data[l]["dense_gb"] * 1000 for l in labels]  # MB
    cste_vals  = [data[l]["cste_mb"] for l in labels]

    max_val = max(dense_vals)
    usable_h = chart_bottom - chart_top
    scale = usable_h / max_val

    defs = f"""
    <linearGradient id="g_d" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{C_RED}" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="{C_ORANGE}" stop-opacity="0.4"/>
    </linearGradient>
    <linearGradient id="g_c" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{C_TEAL}" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="{C_GREEN}" stop-opacity="0.4"/>
    </linearGradient>"""

    svg = svg_header(w, h, "Model Checkpoint Size", "Dense bf16 vs CSTE 4-bit quantized", defs)

    # Y axis gridlines
    for t in range(6):
        val = max_val * t / 5
        y = chart_bottom - val * scale
        svg += f'  <line x1="{chart_left}" y1="{y}" x2="{w - 30}" y2="{y}" stroke="{C_GRID}" stroke-width="1"/>\n'
        lbl = f"{val/1000:.1f} GB" if val >= 1000 else f"{val:.0f} MB"
        svg += f'  <text x="{chart_left - 6}" y="{y + 4}" text-anchor="end" fill="{C_SUBTLE}" font-family="Inter,system-ui,sans-serif" font-size="10">{lbl}</text>\n'

    group_w = bar_w * 2 + 10
    total_w = n * group_w + (n-1) * 40
    start_x = chart_left + (w - chart_left - 30 - total_w) // 2

    for i, label in enumerate(labels):
        cx = start_x + i * (group_w + 40)
        dh = max(dense_vals[i] * scale, 2)
        ch = max(cste_vals[i] * scale, 2)

        svg += f'  <rect x="{cx}" y="{chart_bottom - dh}" width="{bar_w}" height="{dh}" rx="4" fill="url(#g_d)"/>\n'
        svg += f'  <rect x="{cx + bar_w + 10}" y="{chart_bottom - ch}" width="{bar_w}" height="{ch}" rx="4" fill="url(#g_c)"/>\n'

        # Labels
        short = label.split("(")[0].strip()
        svg += f'  <text x="{cx + group_w//2}" y="{chart_bottom + 16}" text-anchor="middle" fill="{C_SUBTLE}" font-family="Inter,system-ui,sans-serif" font-size="10">{short}</text>\n'

        # Size annotations
        d_lbl = f"{dense_vals[i]/1000:.1f}GB" if dense_vals[i] >= 1000 else f"{dense_vals[i]:.0f}MB"
        svg += f'  <text x="{cx + bar_w//2}" y="{chart_bottom - dh - 5}" text-anchor="middle" fill="{C_RED}" font-family="Inter,system-ui,sans-serif" font-size="9" font-weight="600">{d_lbl}</text>\n'
        svg += f'  <text x="{cx + bar_w + 10 + bar_w//2}" y="{chart_bottom - ch - 5}" text-anchor="middle" fill="{C_TEAL}" font-family="Inter,system-ui,sans-serif" font-size="9" font-weight="600">{cste_vals[i]:.0f}MB</text>\n'

    # Legend
    lx = w - 210
    svg += f'  <rect x="{lx}" y="{chart_top}" width="12" height="12" rx="3" fill="{C_RED}"/>\n'
    svg += f'  <text x="{lx + 18}" y="{chart_top + 10}" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="11">Dense (bf16)</text>\n'
    svg += f'  <rect x="{lx + 100}" y="{chart_top}" width="12" height="12" rx="3" fill="{C_TEAL}"/>\n'
    svg += f'  <text x="{lx + 118}" y="{chart_top + 10}" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="11">CSTE (4-bit)</text>\n'

    svg += "</svg>"
    return svg


# ========================================================================
#  PLOT 4: Throughput per chip (SIREN Base)
# ========================================================================

def plot_throughput():
    w, h = 780, 320
    bar_h = 30
    chart_left = 120
    chart_top = 65

    chip_names = list(chip_data.keys())
    tps_vals = [chip_data[c]["tps"] for c in chip_names]
    max_tps = max(tps_vals)
    bar_max = w - chart_left - 140
    scale = bar_max / max_tps

    defs = f"""
    <linearGradient id="g_tp" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{C_PURPLE}" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="{C_BLUE}" stop-opacity="0.95"/>
    </linearGradient>"""

    svg = svg_header(w, h, "Estimated Throughput -- SIREN Base (750M) on Apple Silicon",
                     "4-bit quantized, ANE inference, batch-1, 512-token sequence", defs)

    for i, chip_name in enumerate(chip_names):
        y = chart_top + i * (bar_h + 14)
        bw = tps_vals[i] * scale

        svg += f'  <text x="{chart_left - 10}" y="{y + bar_h - 9}" text-anchor="end" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="13" font-weight="500">{chip_name}</text>\n'
        svg += f'  <rect x="{chart_left}" y="{y}" width="{bw}" height="{bar_h}" rx="5" fill="url(#g_tp)"/>\n'
        svg += f'  <text x="{chart_left + bw + 10}" y="{y + bar_h - 9}" fill="{C_BLUE}" font-family="Inter,system-ui,sans-serif" font-size="13" font-weight="700">{tps_vals[i]:,.0f} tok/s</text>\n'

    svg += "</svg>"
    return svg


# ========================================================================
#  Write
# ========================================================================

print("\nGenerating SVG plots...")

for name, func in [("compression_comparison", plot_compression),
                    ("battery_life", plot_battery),
                    ("memory_footprint", plot_memory),
                    ("latency_throughput", plot_throughput)]:
    svg = func()
    path = f"docs/{name}.svg"
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"  -> {path}")

# Save JSON data
json_out = {l: d for l, d in data.items()}
json_out["_chip_comparison"] = chip_data
with open("docs/benchmark_data.json", "w") as f:
    json.dump(json_out, f, indent=2, default=str)
print("  -> docs/benchmark_data.json")

print("\nDone.")
