"""Slide-ready figures (large type, one message each) from the final runs.

Usage: python3 slide_figs.py <out_dir> <slides_fig_dir>
Hue = architecture (blue CNN, orange ResNet-18, aqua ViT-B/16); filled/solid = standard training,
hollow/dashed = noise augmentation. Same encodings as the paper figures.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import analyze_extra as X
import ferlib as fl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

SPLIT = "test"
BASE = 15
SHORT = {"cnn": "Baseline CNN", "resnet18": "ResNet-18", "vit_b16": "ViT-B/16"}
RC = {"font.size": BASE, "axes.titlesize": BASE + 2, "axes.labelsize": BASE, "xtick.labelsize": BASE - 1,
      "ytick.labelsize": BASE - 1, "legend.fontsize": BASE - 1, "savefig.dpi": 200}


def pct(ax, axis="y", decimals=0):
    f = mticker.FuncFormatter(lambda t, _: f"{t * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def archs_with(runs, variant="std"):
    return [a for a in A.ARCHS if A.seeds_of(runs, a, variant)]


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"))
    fig.savefig(os.path.join(out, name + ".svg"))
    plt.close(fig)


# ---- RQ1: original images -------------------------------------------------------------------------
def s_rq1(runs, out):
    archs = archs_with(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    maj = X.true_share(runs).max()
    for ax, metric, title in zip(axes, ("acc", "macro_f1"), ("Accuracy", "Macro-F1")):
        for i, a in enumerate(archs):
            v = A.vals(runs, a, "std", SPLIT, "clean", metric)
            m, sd = A.mean_sd(v)
            ax.bar(i, m, width=0.62, color=A.ARCH_COLOR[a], edgecolor=A.SURFACE, linewidth=2, zorder=3)
            if not np.isnan(sd):
                ax.errorbar(i, m, yerr=sd, color=A.INK, linewidth=1.6, capsize=5, zorder=4)
            ax.text(i, m + (0 if np.isnan(sd) else sd) + 0.015, f"{m * 100:.1f}%", ha="center", va="bottom",
                    color=A.INK, fontsize=BASE + 2, fontweight="bold")
        if metric == "acc":
            ax.axhline(maj, color=A.INK, linestyle=(0, (1, 2)), linewidth=1.8, zorder=6)
            ax.text(1.02, maj, f"always “happy”\n{maj * 100:.1f}%", ha="left", va="center", color=A.INK2,
                    fontsize=BASE - 3, transform=ax.get_yaxis_transform(), linespacing=1.3)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([SHORT[a] for a in archs])
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        pct(ax)
        A.style(ax)
    fig.tight_layout(w_pad=7)
    save(fig, out, "s_rq1_clean")


# ---- RQ2: accuracy vs severity ---------------------------------------------------------------------
def s_curves(runs, out, variants=("std",), name="s_rq2_curves", relative=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    maj = X.true_share(runs).max()
    for ax, kind in zip(axes, ("noise", "res")):
        lv = A.conds(kind)
        x = np.arange(len(lv) + 1)
        ends = []
        for a in A.ARCHS:
            for v in variants:
                if not A.seeds_of(runs, a, v):
                    continue
                ys = [A.vals(runs, a, v, SPLIT, c) for c in ["clean"] + [c for _, c in lv]]
                if relative:
                    ys = [y / ys[0] for y in ys]
                m = np.array([y.mean() for y in ys])
                sd = np.array([y.std(ddof=1) if len(y) > 1 else 0.0 for y in ys])
                kw = A.series_kw(a, v)
                kw.update(linewidth=3, markersize=kw["markersize"] + 3)
                ax.fill_between(x, m - sd, m + sd, color=kw["color"], alpha=0.12, linewidth=0)
                ax.plot(x, m, **kw)
                ends.append((m[-1], a, v))
        if not relative:
            ax.axhline(maj, color=A.MUTED, linestyle=(0, (1, 2)), linewidth=1.5)
            ax.text(x[0], maj - 0.012, f"majority class {maj * 100:.1f}%", color=A.INK2, fontsize=BASE - 3, va="top")
        else:
            ax.axhline(0.9, color=A.MUTED, linestyle=(0, (1, 2)), linewidth=1.5)
            ax.text(x[-1], 0.905, "90% of original accuracy", color=A.INK2, fontsize=BASE - 3, va="bottom", ha="right")
        ax.set_xticks(x)
        if kind == "noise":
            ax.set_xticklabels(["0"] + [f"{l:g}" for l, _ in lv])
            ax.set_xlabel("Gaussian noise σ  (pixels in [0, 1])")
            ax.set_title("More noise →")
        else:
            ax.set_xticklabels(["48"] + [str(l) for l, _ in lv])
            ax.set_xlabel("Image down-sampled to s × s pixels")
            ax.set_title("Lower resolution →")
        pct(ax)
        A.style(ax)
    axes[0].set_ylabel("Relative accuracy" if relative else "Test accuracy")
    handles = []
    for a in A.ARCHS:
        for v in variants:
            if A.seeds_of(runs, a, v):
                kw = A.series_kw(a, v)
                kw.update(linewidth=3, markersize=kw["markersize"] + 3)
                lab = SHORT[a] + ("" if len(variants) == 1 else (", standard" if v == "std" else ", noise aug."))
                handles.append(Line2D([], [], label=lab, **kw))
    ncol = len(handles) if len(variants) == 1 else len(variants)
    if len(variants) > 1:   # column-major fill: keep each architecture in its own row
        order = [handles[i] for v in range(len(variants)) for i in range(v, len(handles), len(variants))]
        handles = order
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, labelcolor=A.INK2, handlelength=3.2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.5)
    rows = int(np.ceil(len(handles) / ncol))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.07 * rows - 0.01))
    save(fig, out, name)


# ---- RQ3: dumbbells --------------------------------------------------------------------------------
def s_dumbbell(runs, out):
    archs = [a for a in A.ARCHS if A.seeds_of(runs, a, "std") and A.seeds_of(runs, a, "aug")]
    groups = [("Original images", ["clean"]), ("Gaussian noise (avg. of 5 levels)", X.primary_conds("noise")),
              ("Reduced resolution (avg. of 5 levels)", X.primary_conds("res"))]
    fig, ax = plt.subplots(figsize=(11.5, 1.2 + 0.62 * len(archs) * len(groups) + 0.5 * len(groups)))
    y, yt, yl = 0.0, [], []
    for gi, (gname, cl) in enumerate(groups):
        ax.text(0.0, y - 0.55, gname, color=A.INK, fontsize=BASE, fontweight="bold", va="center", ha="left",
                transform=ax.get_yaxis_transform())
        y += 0.35
        for a in archs:
            s0 = X.avg_over(runs, a, "std", cl).mean()
            s1 = X.avg_over(runs, a, "aug", cl).mean()
            c = A.ARCH_COLOR[a]
            ax.plot([s0, s1], [y, y], color=c, linewidth=4, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.plot([s0], [y], marker="o", markersize=13, markerfacecolor=c, markeredgecolor=A.SURFACE, markeredgewidth=2, zorder=3)
            ax.plot([s1], [y], marker="s", markersize=12, markerfacecolor=A.SURFACE, markeredgecolor=c, markeredgewidth=3, zorder=3)
            d = (s1 - s0) * 100
            ax.text(max(s0, s1) + 0.012, y, f"{d:+.1f} pts", va="center", ha="left", color=A.INK, fontsize=BASE - 1,
                    fontweight="bold" if abs(d) >= 5 else "normal")
            yt.append(y)
            yl.append(SHORT[a])
            y += 0.62
        y += 0.75
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_ylim(y - 0.9, -1.0)
    ax.set_xlim(0.2, 0.8)
    ax.set_xlabel("Test accuracy")
    pct(ax, "x")
    A.style(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", markersize=13, markerfacecolor=A.INK2, markeredgecolor=A.SURFACE, linestyle="none",
                      label="standard training"),
               Line2D([], [], marker="s", markersize=12, markerfacecolor=A.SURFACE, markeredgecolor=A.INK2, markeredgewidth=3,
                      linestyle="none", label="noise augmentation")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, labelcolor=A.INK2, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "s_rq3_dumbbell")


# ---- prediction collapse ---------------------------------------------------------------------------
def s_collapse(runs, out, variants=("std", "aug")):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    ts = X.true_share(runs).max()
    for ax, kind in zip(axes, ("noise", "res")):
        lv = A.conds(kind)
        x = np.arange(len(lv) + 1)
        for a in A.ARCHS:
            for v in variants:
                if not A.seeds_of(runs, a, v):
                    continue
                ys = [X.pred_share(runs, a, v, c).max(axis=1) for c in ["clean"] + [c for _, c in lv]]
                m = np.array([y.mean() for y in ys])
                kw = A.series_kw(a, v)
                kw.update(linewidth=3, markersize=kw["markersize"] + 3)
                ax.plot(x, m, **kw)
        ax.axhline(ts, color=A.MUTED, linestyle=(0, (1, 2)), linewidth=1.5)
        ax.text(x[0], ts - 0.015, f"largest true class {ts * 100:.1f}%", color=A.INK2, fontsize=BASE - 3, va="top")
        ax.set_xticks(x)
        if kind == "noise":
            ax.set_xticklabels(["0"] + [f"{l:g}" for l, _ in lv])
            ax.set_xlabel("Gaussian noise σ")
            ax.set_title("More noise →")
        else:
            ax.set_xticklabels(["48"] + [str(l) for l, _ in lv])
            ax.set_xlabel("Image down-sampled to s × s pixels")
            ax.set_title("Lower resolution →")
        ax.set_ylim(0.15, 1.0)
        pct(ax)
        A.style(ax)
    axes[0].set_ylabel("Share of test images put\ninto the most-predicted class")
    handles = []
    for v in variants:
        for a in A.ARCHS:
            if A.seeds_of(runs, a, v):
                kw = A.series_kw(a, v)
                kw.update(linewidth=3, markersize=kw["markersize"] + 3)
                handles.append(Line2D([], [], label=SHORT[a] + (", standard" if v == "std" else ", noise aug."), **kw))
    n_arch = len([a for a in A.ARCHS if A.seeds_of(runs, a, variants[0])])
    fig.legend(handles=handles, loc="upper center", ncol=n_arch, frameon=False, labelcolor=A.INK2, handlelength=3.2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.07 * len(variants) - 0.01))
    save(fig, out, "s_collapse")


# ---- per-class recall: original vs mid-level degradation ---------------------------------------------
def s_recall(runs, out, variant="std", noise_level=0.1, res_level=24):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", A.SEQ_BLUES)
    archs = archs_with(runs, variant)
    conds = [("original", "clean"), (f"σ = {noise_level:g}", fl.cond_name("noise", noise_level)),
             (f"{res_level} px", fl.cond_name("res", res_level))]
    fig, axes = plt.subplots(1, len(archs), figsize=(3.9 * len(archs) + 0.9, 5.4), squeeze=False)
    for ax, a in zip(axes[0], archs):
        M = np.array([[np.mean([runs[(a, variant, s)]["metrics"][SPLIT][c]["recall"][k] for s in A.seeds_of(runs, a, variant)])
                       for _, c in conds] for k in range(7)])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(7):
            for j in range(len(conds)):
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=BASE,
                        color="#ffffff" if M[i, j] > 0.55 else A.INK)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([n for n, _ in conds])
        ax.set_yticks(range(7))
        ax.set_yticklabels(fl.FER_LABELS if a == archs[0] else [""] * 7)
        ax.set_title(SHORT[a], color=A.INK)
        ax.tick_params(length=0, labelcolor=A.INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        for j in range(1, len(conds)):
            ax.axvline(j - 0.5, color=A.SURFACE, linewidth=3)
        for i in range(1, 7):
            ax.axhline(i - 0.5, color=A.SURFACE, linewidth=2)
    fig.tight_layout()
    save(fig, out, f"s_recall_{variant}")


# ---- selection rule --------------------------------------------------------------------------------
def s_selection(runs, out, arch="resnet18"):
    if not A.seeds_of(runs, arch, "std"):
        return
    c = A.ARCH_COLOR[arch]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for s_ in A.seeds_of(runs, arch, "std"):
        r = runs[(arch, "std", s_)]
        ep = np.array([h["epoch"] for h in r["history"]])
        vc = np.array([h["val_acc_clean"] for h in r["history"]])
        vn = np.array([h["val_acc_noisy"] for h in r["history"]])
        ax.plot(ep, vc, color=c, linewidth=2.6)
        ax.plot(ep, vn, color=c, linewidth=2.6, linestyle=(0, (4, 2)))
        b = r["best_epoch"] - 1
        ax.plot([ep[b]], [vc[b]], marker="o", markersize=13, markerfacecolor=c, markeredgecolor=A.SURFACE, markeredgewidth=2,
                linestyle="none", zorder=5)
        ax.plot([ep[b]], [vn[b]], marker="o", markersize=13, markerfacecolor=A.SURFACE, markeredgecolor=c, markeredgewidth=3,
                linestyle="none", zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    pct(ax)
    A.style(ax)
    handles = [Line2D([], [], color=c, linewidth=2.6, label="original images"),
               Line2D([], [], color=c, linewidth=2.6, linestyle=(0, (4, 2)), label="noisy images"),
               Line2D([], [], marker="o", markersize=12, markerfacecolor=c, markeredgecolor=A.SURFACE, linestyle="none",
                      label="checkpoint picked by the score")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, labelcolor=A.INK2, bbox_to_anchor=(0.5, 1.0),
               fontsize=BASE - 2, columnspacing=1.5, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out, f"s_selection_{arch}")


def main(out_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    runs = A.load_runs(out_dir)
    with plt.rc_context(RC):
        s_rq1(runs, fig_dir)
        s_curves(runs, fig_dir, ("std",), "s_rq2_curves")
        s_curves(runs, fig_dir, ("std",), "s_rq2_relative", relative=True)
        s_curves(runs, fig_dir, ("std", "aug"), "s_rq3_curves")
        s_dumbbell(runs, fig_dir)
        s_collapse(runs, fig_dir)
        s_recall(runs, fig_dir, "std")
        s_recall(runs, fig_dir, "aug")
        for a in A.ARCHS:
            s_selection(runs, fig_dir, a)
    print("slide figures written to", fig_dir, "->", sorted(f for f in os.listdir(fig_dir) if f.endswith(".png")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
