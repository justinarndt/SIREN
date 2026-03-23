"""Regenerate battery_life.svg with realistic power numbers."""
import os

C_BG     = "#0d1117"
C_TEXT   = "#c9d1d9"
C_SUBTLE = "#8b949e"
C_GREEN  = "#3fb950"
C_TEAL   = "#39d2c0"
C_RED    = "#f85149"
C_ORANGE = "#f0883e"

battery_wh = 18.08  # iPhone 16 Pro: 4685 mAh @ 3.86V

# Realistic power estimates (watts) during continuous inference
# Dense: GPU/CPU on iPhone — small models efficient, large ones thermal throttle
# SIREN: ANE with weight-in-SRAM for models that fit, DRAM spill for larger
models = {
    "Tiny (50M)":    {"siren_w": 0.8,  "dense_w": 2.5},
    "Small (200M)":  {"siren_w": 1.2,  "dense_w": 4.0},
    "Base (750M)":   {"siren_w": 1.8,  "dense_w": 6.5},
    "Medium (1.5B)": {"siren_w": 2.5,  "dense_w": 9.0},
    "Large (7B)":    {"siren_w": 3.5,  "dense_w": 15.0},
}

for d in models.values():
    d["siren_h"] = battery_wh / d["siren_w"]
    d["dense_h"] = battery_wh / d["dense_w"]

w, h = 780, 380
bar_h = 22
chart_left = 140
chart_top = 70

labels = list(models.keys())
siren_vals = [models[l]["siren_h"] for l in labels]
dense_vals = [models[l]["dense_h"] for l in labels]
max_val = max(max(siren_vals), max(dense_vals))
bar_max = w - chart_left - 160
scale = bar_max / max_val

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" fill="none">
  <defs>
    <linearGradient id="g_bat_c" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{C_GREEN}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{C_TEAL}" stop-opacity="0.95"/>
    </linearGradient>
    <linearGradient id="g_bat_d" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{C_RED}" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="{C_ORANGE}" stop-opacity="0.6"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" rx="12" fill="{C_BG}"/>
  <text x="{w//2}" y="32" text-anchor="middle" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="17" font-weight="600">Estimated Battery Life During Continuous Inference</text>
  <text x="{w//2}" y="50" text-anchor="middle" fill="{C_SUBTLE}" font-family="Inter,system-ui,sans-serif" font-size="11">iPhone 16 Pro (4,685 mAh). SIREN on ANE vs Dense on GPU. Lower power = longer runtime.</text>
'''

for i, label in enumerate(labels):
    y = chart_top + i * (bar_h * 2 + 22)
    d = models[label]

    svg += f'  <text x="{chart_left - 12}" y="{y + bar_h + 4}" text-anchor="end" fill="{C_TEXT}" font-family="Inter,system-ui,sans-serif" font-size="12">{label}</text>\n'

    sw = d["siren_h"] * scale
    svg += f'  <rect x="{chart_left}" y="{y}" width="{sw}" height="{bar_h}" rx="4" fill="url(#g_bat_c)"/>\n'
    svg += f'  <text x="{chart_left + sw + 8}" y="{y + bar_h - 6}" fill="{C_GREEN}" font-family="Inter,system-ui,sans-serif" font-size="12" font-weight="600">{d["siren_h"]:.1f}h SIREN ({d["siren_w"]:.1f}W)</text>\n'

    dw = d["dense_h"] * scale
    svg += f'  <rect x="{chart_left}" y="{y + bar_h + 3}" width="{dw}" height="{bar_h}" rx="4" fill="url(#g_bat_d)"/>\n'
    svg += f'  <text x="{chart_left + dw + 8}" y="{y + bar_h * 2 - 3}" fill="{C_SUBTLE}" font-family="Inter,system-ui,sans-serif" font-size="11">{d["dense_h"]:.1f}h Dense ({d["dense_w"]:.1f}W)</text>\n'

svg += "</svg>"

os.makedirs("docs", exist_ok=True)
with open("docs/battery_life.svg", "w", encoding="utf-8") as f:
    f.write(svg)

print("Regenerated docs/battery_life.svg")
for name, d in models.items():
    ratio = d["siren_h"] / d["dense_h"]
    print(f"  {name:16s}  SIREN: {d['siren_h']:5.1f}h ({d['siren_w']}W)  Dense: {d['dense_h']:5.1f}h ({d['dense_w']}W)  {ratio:.1f}x longer")
