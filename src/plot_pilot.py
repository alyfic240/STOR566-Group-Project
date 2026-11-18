"""Figures from the level-setting pilot (validation images only).
Usage: python3 -I plot_pilot.py <data.npz> <pilot_dir> [val_index]
Writes fig_levels_examples.png always, and fig_pilot_validation_sweep.png once pilot_results.json exists.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import ferlib as fl
import analyze as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

data = fl.load_data(sys.argv[1])
out = sys.argv[2]
idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# ---- what each primary severity level looks like (one validation image, the exact evaluation noise) ----
xv = fl.to_float(data["val"][0])
fig, axes = plt.subplots(2, 6, figsize=(10.5, 4.5))
row_noise = [("original", xv)] + [(f"σ = {s:g}", fl.make_condition(xv, "val", "noise", s)) for s in fl.NOISE_PRIMARY]
row_res = [("48 px (original)", xv)] + [(f"{s} px", fl.make_condition(xv, "val", "res", s)) for s in fl.RES_PRIMARY]
for r, row in enumerate((row_noise, row_res)):
    for c, (title, x) in enumerate(row):
        ax = axes[r, c]
        ax.imshow(x[idx, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, color=az.INK2, fontweight="normal", loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
axes[0, 0].set_ylabel("Gaussian noise", color=az.INK2)
axes[1, 0].set_ylabel("Reduced resolution", color=az.INK2)
fig.suptitle(f"Primary severity levels on one validation image (label: {fl.FER_LABELS[int(data['val'][1][idx])]})",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=az.INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.subplots_adjust(hspace=0.32)
fig.savefig(os.path.join(out, "fig_levels_examples.png"))
plt.close(fig)
print("wrote fig_levels_examples.png")

p = os.path.join(out, "pilot_results.json")
if os.path.exists(p):
    R = json.load(open(p))
    clean, maj = R["val"]["clean"]["acc"], R["majority_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind, grid, primary in ((axes[0], "noise", R["noise_grid"], fl.NOISE_PRIMARY),
                                    (axes[1], "res", R["res_grid"], fl.RES_PRIMARY)):
        xs = [0.0 if kind == "noise" else 48] + list(grid)
        ys = [clean] + [R["val"][fl.cond_name(kind, g)]["acc"] for g in grid]
        c = az.ARCH_COLOR["cnn"]
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=5, markerfacecolor=az.SURFACE,
                markeredgecolor=c, markeredgewidth=1.5)
        px = [x for x in xs if x in primary]
        ax.plot(px, [ys[xs.index(x)] for x in px], linestyle="none", marker="o", markersize=8, markerfacecolor=c,
                markeredgecolor=az.SURFACE, markeredgewidth=1.5, label="primary level")
        ax.plot([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=az.SURFACE, markeredgecolor=c,
                markeredgewidth=1.5, label="other tested level")
        for yy, txt in ((0.9 * clean, f"90% of original accuracy ({0.9 * clean * 100:.1f}%)"),
                        (maj, f"majority-class rate ({maj * 100:.1f}%)")):
            ax.axhline(yy, color=az.MUTED, linewidth=1, linestyle=(0, (2, 2)))
            ax.annotate(txt, xy=(xs[-1], yy), xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=8.5, color=az.INK2)
        az.style(ax)
        if kind == "res":
            ax.invert_xaxis()
            ax.set_xlabel("Intermediate image size (pixels per side; 48 = original)")
            ax.set_title("Reduced resolution")
        else:
            ax.set_xlabel("Gaussian noise standard deviation σ (pixels scaled to [0, 1])")
            ax.set_title("Gaussian noise")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Validation accuracy (pilot baseline CNN)")
    az.legend(axes[0], loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.suptitle("Level-setting pilot: validation accuracy of the baseline CNN across severities",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=az.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "fig_pilot_validation_sweep.png"))
    plt.close(fig)
    print("wrote fig_pilot_validation_sweep.png")
