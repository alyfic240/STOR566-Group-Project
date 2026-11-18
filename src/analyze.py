"""Aggregate final runs into tables and figures.

Usage: python3 analyze.py <out_dir>          (reads <out_dir>/final/*.json, writes <out_dir>/analysis/)
Works on partial results (one architecture, fewer seeds).
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ferlib as fl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- chart style (light surface; hue = architecture, line/marker style = training variant) ----
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ARCH_COLOR = {"cnn": "#2a78d6", "resnet18": "#eb6834", "vit_b16": "#1baf7a"}
ARCH_LABEL = {"cnn": "Baseline CNN", "resnet18": "ResNet-18 (pretrained)", "vit_b16": "ViT-B/16 (pretrained)"}
VAR_LABEL = {"std": "standard training", "aug": "noise augmentation",
             "ref": "standard, clean-selected checkpoint"}
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ARCHS, VARIANTS = ("cnn", "resnet18", "vit_b16"), ("std", "aug")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 100,
                     "savefig.dpi": 200, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def series_kw(arch, variant):
    c = ARCH_COLOR[arch]
    if variant == "std":
        return dict(color=c, linestyle="-", linewidth=2, marker="o", markersize=7,
                    markerfacecolor=c, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if variant == "ref":
        return dict(color=c, linestyle=(0, (1, 1.5)), linewidth=2, marker="^", markersize=7,
                    markerfacecolor=c, markeredgecolor=SURFACE, markeredgewidth=1.5)
    return dict(color=c, linestyle=(0, (4, 2)), linewidth=2, marker="s", markersize=6.5,
                markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=2)


def legend(ax, fontsize=9, **kw):
    return ax.legend(frameon=False, fontsize=fontsize, labelcolor=INK2, handlelength=3, **kw)


# ---- loading and aggregation ----
def load_runs(out_dir):
    runs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "final", "*.json"))):
        r = json.load(open(p))
        runs[(r["config"]["arch"], r["variant"], r["config"]["seed"])] = r
    return runs


def seeds_of(runs, arch, variant):
    return sorted(s for (a, v, s) in runs if a == arch and v == variant)


def vals(runs, arch, variant, split, cond, metric="acc"):
    return np.array([runs[(arch, variant, s)]["metrics"][split][cond][metric] for s in seeds_of(runs, arch, variant)])


def mean_sd(x):
    x = np.asarray(x, float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")) if len(x) else (float("nan"),) * 2


def fmt(x, pct=True, digits=1):
    m, s = mean_sd(x)
    k = 100 if pct else 1
    if np.isnan(m):
        return "n/a"
    return f"{m * k:.{digits}f}" + (f" ± {s * k:.{digits}f}" if not np.isnan(s) else "")


def conds(kind, extended=False):
    if kind == "noise":
        lv = fl.NOISE_PRIMARY + (fl.NOISE_EXTENDED if extended else [])
    else:
        lv = fl.RES_PRIMARY + (fl.RES_EXTENDED if extended else [])
    return [(lvl, fl.cond_name(kind, lvl)) for lvl in lv]


def configs(runs):
    return [(a, v) for a in ARCHS for v in VARIANTS if seeds_of(runs, a, v)]


# ---- tables ----
def write_tables(runs, out):
    rows = ["arch,variant,seed,split,condition,acc,macro_f1," + ",".join("recall_" + c for c in fl.FER_LABELS)]
    for (a, v, s), r in sorted(runs.items()):
        for split in ("val", "test"):
            for cname, m in r["metrics"][split].items():
                rows.append(f"{a},{v},{s},{split},{cname},{m['acc']:.5f},{m['macro_f1']:.5f}," +
                            ",".join(f"{x:.4f}" for x in m["recall"]))
    open(os.path.join(out, "summary_long.csv"), "w").write("\n".join(rows) + "\n")

    md = ["# Results summary (test split unless stated; mean ± sample SD over seeds, in %)", ""]
    md += ["## RQ1 - original images", "",
           "| Model | Training | Seeds | Val acc | Test acc | Test macro-F1 | Best epoch |", "|---|---|---|---|---|---|---|"]
    for a, v in configs(runs):
        be = [runs[(a, v, s)]["best_epoch"] for s in seeds_of(runs, a, v)]
        md.append(f"| {ARCH_LABEL[a]} | {VAR_LABEL[v]} | {len(be)} | {fmt(vals(runs, a, v, 'val', 'clean'))} | "
                  f"{fmt(vals(runs, a, v, 'test', 'clean'))} | {fmt(vals(runs, a, v, 'test', 'clean', 'macro_f1'))} | "
                  f"{', '.join(map(str, be))} |")
    for kind, title, unit in (("noise", "Gaussian noise", "sigma"), ("res", "reduced resolution", "size")):
        for metric, mname in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
            cl = conds(kind, extended=True)
            md += ["", f"## RQ2 - test {mname} under {title} ({unit}; primary levels first, then extended)", "",
                   "| Model | Training | clean | " + " | ".join(f"{l:g}" for l, _ in cl) + " |",
                   "|---|---|---|" + "---|" * len(cl)]
            for a, v in configs(runs):
                md.append(f"| {ARCH_LABEL[a]} | {VAR_LABEL[v]} | {fmt(vals(runs, a, v, 'test', 'clean', metric))} | " +
                          " | ".join(fmt(vals(runs, a, v, "test", c, metric)) for _, c in cl) + " |")
        cl = conds(kind)
        md += ["", f"### Relative accuracy under {title} (degraded / original, per seed) and 90% threshold", "",
               "| Model | Training | " + " | ".join(f"{l:g}" for l, _ in cl) + " | first primary level below 90% |",
               "|---|---|" + "---|" * (len(cl) + 1)]
        for a, v in configs(runs):
            base = vals(runs, a, v, "test", "clean")
            rel = [vals(runs, a, v, "test", c) / base for _, c in cl]
            first = next((f"{l:g}" for (l, _), r in zip(cl, rel) if r.mean() < 0.90), "none in tested range")
            md.append(f"| {ARCH_LABEL[a]} | {VAR_LABEL[v]} | " + " | ".join(fmt(r) for r in rel) + f" | {first} |")
    md += ["", "## RQ3 - effect of noise augmentation (augmented minus standard, same seed; percentage points)", ""]
    cl = [(None, "clean")] + conds("noise") + conds("res")
    md += ["| Model | clean | " + " | ".join(c for _, c in cl[1:]) + " |", "|---|" + "---|" * len(cl)]
    for a in ARCHS:
        common = sorted(set(seeds_of(runs, a, "std")) & set(seeds_of(runs, a, "aug")))
        if not common:
            continue
        cells = []
        for _, c in cl:
            d = [runs[(a, "aug", s)]["metrics"]["test"][c]["acc"] - runs[(a, "std", s)]["metrics"]["test"][c]["acc"]
                 for s in common]
            m, sd = mean_sd(d)
            cells.append(f"{m * 100:+.1f}" + (f" ± {sd * 100:.1f}" if not np.isnan(sd) else ""))
        md.append(f"| {ARCH_LABEL[a]} | " + " | ".join(cells) + " |")
    ref_archs = [a for a in ARCHS if seeds_of(runs, a, "ref")]
    if ref_archs:
        cl = [(None, "clean")] + conds("noise") + conds("res")
        md += ["", "## Reference - standard training with the checkpoint chosen on clean validation accuracy only", "",
               "Same training and learning rate as 'standard training'; only the stopping rule differs.", "",
               "| Model | Seeds | Val acc | Test acc | Test macro-F1 | Best epoch | " + " | ".join(c for _, c in cl[1:]) + " |",
               "|---|---|---|---|---|---|" + "---|" * (len(cl) - 1)]
        for a in ref_archs:
            be = [runs[(a, "ref", s)]["best_epoch"] for s in seeds_of(runs, a, "ref")]
            md.append(f"| {ARCH_LABEL[a]} | {len(be)} | {fmt(vals(runs, a, 'ref', 'val', 'clean'))} | "
                      f"{fmt(vals(runs, a, 'ref', 'test', 'clean'))} | {fmt(vals(runs, a, 'ref', 'test', 'clean', 'macro_f1'))} | "
                      f"{', '.join(map(str, be))} | " + " | ".join(fmt(vals(runs, a, "ref", "test", c)) for _, c in cl[1:]) + " |")
        md += ["", "### Effect of the selection rule (clean + noisy selection minus clean-only selection, same seed; "
                   "percentage points of test accuracy)", "",
               "| Model | clean | " + " | ".join(c for _, c in cl[1:]) + " |", "|---|" + "---|" * len(cl)]
        for a in ref_archs:
            common = sorted(set(seeds_of(runs, a, "std")) & set(seeds_of(runs, a, "ref")))
            if not common:
                continue
            cells = []
            for _, c in cl:
                d = [runs[(a, "std", s)]["metrics"]["test"][c]["acc"] - runs[(a, "ref", s)]["metrics"]["test"][c]["acc"]
                     for s in common]
                m, sd = mean_sd(d)
                cells.append(f"{m * 100:+.1f}" + (f" ± {sd * 100:.1f}" if not np.isnan(sd) else ""))
            md.append(f"| {ARCH_LABEL[a]} | " + " | ".join(cells) + " |")
    md += ["", "## Per-class recall on original test images (%)", "",
           "| Model | Training | " + " | ".join(fl.FER_LABELS) + " |", "|---|---|" + "---|" * 7]
    for a, v in configs(runs):
        rec = np.array([runs[(a, v, s)]["metrics"]["test"]["clean"]["recall"] for s in seeds_of(runs, a, v)])
        md.append(f"| {ARCH_LABEL[a]} | {VAR_LABEL[v]} | " + " | ".join(f"{x * 100:.1f}" for x in rec.mean(0)) + " |")
    sel = {}
    for a in ARCHS:
        p = os.path.join(os.path.dirname(out), f"selection_{a}.json")
        if os.path.exists(p):
            sel[a] = json.load(open(p))
    if sel:
        md += ["", "## Selected hyperparameters (validation selection score only)", ""]
        for a, s in sel.items():
            md.append(f"- {ARCH_LABEL[a]}: lr {s['lr']:g} (scores {s['lr_scores']}); augmentation p={s['aug_p']:g}, "
                      f"sigma_max={s['aug_sigma_max']:g} (scores {s['aug_scores']}); frozen {s['frozen_at']}")
    md += ["", "SD values describe variation across training seeds; they are not confidence intervals."]
    open(os.path.join(out, "results_summary.md"), "w").write("\n".join(md) + "\n")
    return "\n".join(md)


# ---- figures ----
def majority_rate(runs, split="test"):
    r = next(iter(runs.values()))
    cm = np.array(r["metrics"][split]["clean"]["cm"])
    return float(cm.sum(1).max() / cm.sum())


def fig_degradation(runs, out, metric="acc", relative=False, extended=False, name="fig_degradation_acc"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, kind in zip(axes, ("noise", "res")):
        cl = conds(kind, extended)
        xs = [0.0 if kind == "noise" else 48] + [l for l, _ in cl]
        for a, v in configs(runs):
            base = vals(runs, a, v, "test", "clean", metric)
            ys = [base] + [vals(runs, a, v, "test", c, metric) for _, c in cl]
            if relative:
                ys = [y / base for y in ys]
            m = np.array([y.mean() for y in ys])
            sd = np.array([y.std(ddof=1) if len(y) > 1 else 0.0 for y in ys])
            ax.fill_between(xs, m - sd, m + sd, color=ARCH_COLOR[a], alpha=0.10, linewidth=0)
            ax.plot(xs, m, label=f"{ARCH_LABEL[a]}, {VAR_LABEL[v]}", **series_kw(a, v))
        style(ax)
        if kind == "res":
            ax.invert_xaxis()
            ax.set_xticks(xs)
            ax.set_xlabel("Intermediate image size (pixels per side; 48 = original)")
            ax.set_title("Reduced resolution")
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs], rotation=0 if not extended else 45)
            ax.set_xlabel("Gaussian noise standard deviation σ (pixels scaled to [0, 1])")
            ax.set_title("Gaussian noise")
        if relative:
            ax.axhline(0.90, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
            ax.annotate("90% of original accuracy", xy=(xs[-1], 0.90), xytext=(0, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8.5, color=INK2)
        elif metric == "acc":
            mr = majority_rate(runs)
            ax.axhline(mr, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
            ax.annotate(f"majority-class rate {mr * 100:.1f}%", xy=(xs[0], mr), xytext=(0, 4),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ylab = {"acc": "Test accuracy", "macro_f1": "Test macro-F1"}[metric]
    axes[0].set_ylabel("Relative accuracy (degraded / original)" if relative else ylab)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    h, l = axes[0].get_legend_handles_labels()
    rows = math.ceil(len(l) / 2)
    order = sorted(range(len(l)), key=lambda i: (i % 2, i))   # legend fills by column: standard runs left, augmented right
    fig.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=9, labelcolor=INK2, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=3, columnspacing=3)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 * rows - 0.03))
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


def fig_by_architecture(runs, out, metric="acc"):
    archs = [a for a in ARCHS if any(seeds_of(runs, a, v) for v in ("ref", "std", "aug"))]
    if not archs:
        return
    fig, axes = plt.subplots(2, len(archs), figsize=(4.4 * len(archs), 7.2), sharey=True, squeeze=False)
    for j, a in enumerate(archs):
        for i, kind in enumerate(("noise", "res")):
            ax = axes[i, j]
            cl = conds(kind)
            xs = [0.0 if kind == "noise" else 48] + [l for l, _ in cl]
            for v in ("ref", "std", "aug"):
                if not seeds_of(runs, a, v):
                    continue
                ys = [vals(runs, a, v, "test", "clean", metric)] + [vals(runs, a, v, "test", c, metric) for _, c in cl]
                m = np.array([y.mean() for y in ys])
                sd = np.array([y.std(ddof=1) if len(y) > 1 else 0.0 for y in ys])
                ax.fill_between(xs, m - sd, m + sd, color=ARCH_COLOR[a], alpha=0.10, linewidth=0)
                ax.plot(xs, m, label=VAR_LABEL[v], **series_kw(a, v))
            style(ax)
            ax.set_xticks(xs)
            if kind == "res":
                ax.invert_xaxis()
                ax.set_xlabel("Intermediate image size (px)")
            else:
                ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8.5)
                ax.set_xlabel("Gaussian noise σ")
                ax.set_title(ARCH_LABEL[a])
            mr = majority_rate(runs)
            ax.axhline(mr, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
            if j == 0:
                ax.set_ylabel("Test accuracy")
                ax.annotate(f"majority-class rate {mr * 100:.1f}%", xy=(xs[0], mr), xytext=(0, 4),
                            textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK2)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    from matplotlib.lines import Line2D
    present = [v for v in ("ref", "std", "aug") if any(seeds_of(runs, a, v) for a in archs)]
    proxies = []
    for v in present:
        kw = series_kw(archs[0], v)
        kw.update(color=INK2)
        kw["markerfacecolor"] = SURFACE if v == "aug" else INK2
        kw["markeredgecolor"] = INK2 if v == "aug" else SURFACE
        proxies.append(Line2D([], [], **kw))
    fig.legend(proxies, [VAR_LABEL[v] for v in present], frameon=False, fontsize=9, labelcolor=INK2, ncol=len(present),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=3.4, columnspacing=3)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(out, "fig_by_architecture.png"))
    plt.close(fig)


def fig_aug_delta(runs, out):
    cl = [(None, "clean")] + conds("noise") + conds("res")
    labels = ["original"] + [f"σ={l:g}" for l, _ in conds("noise")] + [f"{l}px" for l, _ in conds("res")]
    archs = [a for a in ARCHS if set(seeds_of(runs, a, "std")) & set(seeds_of(runs, a, "aug"))]
    if not archs:
        return
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.78 / len(archs) - 0.04
    for k, a in enumerate(archs):
        common = sorted(set(seeds_of(runs, a, "std")) & set(seeds_of(runs, a, "aug")))
        d = np.array([[runs[(a, "aug", s)]["metrics"]["test"][c]["acc"] - runs[(a, "std", s)]["metrics"]["test"][c]["acc"]
                       for _, c in cl] for s in common]) * 100
        m = d.mean(0)
        sd = d.std(0, ddof=1) if len(common) > 1 else np.zeros_like(m)
        x = np.arange(len(cl)) + (k - (len(archs) - 1) / 2) * (w + 0.04)
        ax.bar(x, m, width=w, color=ARCH_COLOR[a], label=ARCH_LABEL[a], linewidth=0)
        ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    style(ax)
    ax.axhline(0, color=AXIS, linewidth=1)
    for xsep in (0.5, 0.5 + len(conds("noise"))):
        ax.axvline(xsep, color=GRID, linewidth=1)
    ax.set_xticks(np.arange(len(cl)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Change in test accuracy (percentage points)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.14 * (hi - lo))
    ymax = ax.get_ylim()[1]
    ax.annotate("Gaussian noise (corruption seen in training)", xy=(3, ymax), xytext=(0, -3), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=INK2)
    ax.annotate("Reduced resolution (unseen corruption)", xy=(3 + len(conds("noise")), ymax), xytext=(0, -3),
                textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK2)
    fig.suptitle("Effect of noise augmentation: augmented minus standard training (same seed)", x=0.01, y=0.985,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left", bbox_to_anchor=(0.045, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(out, "fig_augmentation_effect.png"))
    plt.close(fig)


def _heat(ax, M, xt, yt, title, vmax=1.0):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUES)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_yticks(range(len(yt)))
    ax.set_yticklabels(yt, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if M[i, j] > 0.55 * vmax else INK)
    ax.set_title(title, fontsize=10)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_recall_heatmaps(runs, out):
    cfgs = configs(runs)
    if not cfgs:
        return
    cl = [(None, "clean")] + conds("noise") + conds("res")
    xt = ["orig."] + [f"σ{l:g}" for l, _ in conds("noise")] + [f"{l}px" for l, _ in conds("res")]
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(9.5, 2.5 * len(cfgs)), squeeze=False)
    for ax, (a, v) in zip(axes[:, 0], cfgs):
        M = np.array([np.mean([runs[(a, v, s)]["metrics"]["test"][c]["recall"] for s in seeds_of(runs, a, v)], 0)
                      for _, c in cl]).T
        _heat(ax, M, xt, fl.FER_LABELS, f"{ARCH_LABEL[a]}, {VAR_LABEL[v]} - per-class recall on test (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_per_class_recall.png"))
    plt.close(fig)


def fig_confusion(runs, out, variant="std"):
    archs = [a for a in ARCHS if seeds_of(runs, a, variant)]
    if not archs:
        return
    fig, axes = plt.subplots(1, len(archs), figsize=(5.6 * len(archs), 4.8), squeeze=False)
    for ax, a in zip(axes[0], archs):
        cm = sum(np.array(runs[(a, variant, s)]["metrics"]["test"]["clean"]["cm"]) for s in seeds_of(runs, a, variant))
        _heat(ax, cm / cm.sum(1, keepdims=True), fl.FER_LABELS, fl.FER_LABELS,
              f"{ARCH_LABEL[a]}, {VAR_LABEL[variant]} - original test images\nrow-normalised confusion (%), seeds pooled")
        ax.set_xlabel("Predicted", color=INK2)
        ax.set_ylabel("True", color=INK2)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_confusion_clean_{variant}.png"))
    plt.close(fig)


def fig_learning_curves(runs, out, variant="std"):
    """Validation accuracy per epoch on original and on noisy images, one line per seed, with the selected
    checkpoint marked. Shows how the selection score behaves for models trained without noise."""
    archs = [a for a in ARCHS if seeds_of(runs, a, variant)]
    if not archs:
        return
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(archs), figsize=(4.4 * len(archs), 4.1), squeeze=False)
    for ax, a in zip(axes[0], archs):
        c = ARCH_COLOR[a]
        for s_ in seeds_of(runs, a, variant):
            r = runs[(a, variant, s_)]
            ep = np.array([h["epoch"] for h in r["history"]])
            vc = np.array([h["val_acc_clean"] for h in r["history"]])
            vn = np.array([h["val_acc_noisy"] for h in r["history"]])
            ax.plot(ep, vc, color=c, linewidth=1.6, alpha=0.9)
            ax.plot(ep, vn, color=c, linewidth=1.6, alpha=0.9, linestyle=(0, (4, 2)))
            b = r["best_epoch"] - 1
            ax.plot([ep[b]], [vc[b]], marker="o", markersize=8, markerfacecolor=c, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none", zorder=5)
            ax.plot([ep[b]], [vn[b]], marker="o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=c,
                    markeredgewidth=2, linestyle="none", zorder=5)
        style(ax)
        ax.set_title(ARCH_LABEL[a])
        ax.set_xlabel("Epoch")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=8))
    axes[0][0].set_ylabel("Validation accuracy")
    handles = [Line2D([], [], color=INK2, linewidth=1.8, label="original validation images"),
               Line2D([], [], color=INK2, linewidth=1.8, linestyle=(0, (4, 2)), label="noisy validation images"),
               Line2D([], [], color=INK2, marker="o", markersize=8, markerfacecolor=INK2, markeredgecolor=SURFACE,
                      linestyle="none", label="selected checkpoint (one line per seed)")]
    fig.suptitle(f"Validation accuracy during {VAR_LABEL[variant]}", x=0.01, y=0.985, ha="left", va="top",
                 fontsize=11, fontweight="bold", color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.0, 0.925), handlelength=3)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(os.path.join(out, "fig_learning_curves.png" if variant == "std" else f"fig_learning_curves_{variant}.png"))
    plt.close(fig)


def main(out_dir):
    import matplotlib.ticker  # noqa: F401
    import matplotlib.colors  # noqa: F401

    runs = load_runs(out_dir)
    if not runs:
        print("no final runs found in", out_dir)
        return
    out = os.path.join(out_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    print(write_tables(runs, out))
    jobs = [("degradation accuracy", lambda: fig_degradation(runs, out, "acc", name="fig_degradation_acc")),
            ("degradation macro-F1", lambda: fig_degradation(runs, out, "macro_f1", name="fig_degradation_macro_f1")),
            ("relative accuracy", lambda: fig_degradation(runs, out, "acc", relative=True, name="fig_relative_acc")),
            ("extended sweep", lambda: fig_degradation(runs, out, "acc", extended=True, name="fig_degradation_acc_extended")),
            ("by architecture", lambda: fig_by_architecture(runs, out)),
            ("augmentation effect", lambda: fig_aug_delta(runs, out)),
            ("per-class recall", lambda: fig_recall_heatmaps(runs, out)),
            ("confusion std", lambda: fig_confusion(runs, out, "std")),
            ("confusion aug", lambda: fig_confusion(runs, out, "aug")),
            ("learning curves", lambda: fig_learning_curves(runs, out)),
            ("learning curves aug", lambda: fig_learning_curves(runs, out, "aug"))]
    for name, job in jobs:
        try:
            job()
        except Exception as e:  # one failed figure must not block the rest
            print(f"figure '{name}' failed: {e!r}")
    print("analysis written to", out)


if __name__ == "__main__":
    main(sys.argv[1])
