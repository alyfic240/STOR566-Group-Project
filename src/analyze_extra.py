"""Deeper analyses on the final runs (complements analyze.py).

Usage: python3 analyze_extra.py <out_dir> [data_npz]
Reads <out_dir>/final/*.json and *_preds.npz, writes <out_dir>/analysis_extra/.

1. Headline table: clean accuracy, mean accuracy over the five primary noise levels, mean accuracy over the
   five primary resolution levels (mean and sample SD over seeds).
2. Prediction collapse: share of test images assigned to the most-predicted class, per severity level.
3. Predicted-class distribution heatmaps.
4. Test-set sampling uncertainty: bootstrap over test images (models held fixed) for seed-averaged accuracy
   and for paired differences (augmentation vs. standard; architecture vs. architecture).
5. Per-class recall retention at a mid severity level.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import ferlib as fl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT = "test"
ALL_VARIANTS = ("std", "aug", "ref")
BOOT_B = 5000
BOOT_SEED = 566


def configs(runs, variants=ALL_VARIANTS):
    return [(a, v) for a in A.ARCHS for v in variants if A.seeds_of(runs, a, v)]


def cms(runs, arch, variant, cond, split=SPLIT):
    return np.array([runs[(arch, variant, s)]["metrics"][split][cond]["cm"] for s in A.seeds_of(runs, arch, variant)], float)


def pred_share(runs, arch, variant, cond, split=SPLIT):
    """Per seed: share of images assigned to each class. Shape (n_seeds, 7)."""
    c = cms(runs, arch, variant, cond, split)
    return c.sum(axis=1) / c.sum(axis=(1, 2))[:, None]


def true_share(runs, split=SPLIT):
    r = next(iter(runs.values()))
    c = np.array(r["metrics"][split]["clean"]["cm"], float)
    return c.sum(axis=1) / c.sum()


def primary_conds(kind):
    return [c for _, c in A.conds(kind)]


def avg_over(runs, arch, variant, conds, metric="acc", split=SPLIT):
    """Per seed: metric averaged over the given conditions. Shape (n_seeds,)."""
    return np.mean([A.vals(runs, arch, variant, split, c, metric) for c in conds], axis=0)


# ---------- 1. headline table ----------
def headline(runs):
    rows, md = [], []
    md.append("| Model | Training | Clean acc | Noise avg acc | Resolution avg acc | Clean macro-F1 | Noise avg macro-F1 | Resolution avg macro-F1 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for a, v in configs(runs):
        cells = []
        for metric in ("acc", "macro_f1"):
            clean = A.vals(runs, a, v, SPLIT, "clean", metric)
            nz = avg_over(runs, a, v, primary_conds("noise"), metric)
            rs = avg_over(runs, a, v, primary_conds("res"), metric)
            cells += [A.fmt(clean), A.fmt(nz), A.fmt(rs)]
            for name, x in (("clean", clean), ("noise_avg", nz), ("res_avg", rs)):
                m, s = A.mean_sd(x)
                rows.append(dict(arch=a, variant=v, metric=metric, quantity=name, mean=m, sd=s, n_seeds=len(x)))
        md.append(f"| {A.ARCH_LABEL[a]} | {A.VAR_LABEL[v]} | " + " | ".join(cells) + " |")
    return rows, md


# ---------- 2. prediction collapse ----------
def collapse_table(runs):
    md = ["| Model | Training | Condition | Share assigned to most-predicted class | Most-predicted class by seed |", "|---|---|---|---|---|"]
    rows = []
    for a, v in configs(runs):
        for kind in ("clean", "noise", "res"):
            cl = ["clean"] if kind == "clean" else primary_conds(kind)
            for c in cl:
                p = pred_share(runs, a, v, c)
                top = p.max(axis=1)
                names = [fl.FER_LABELS[i] for i in p.argmax(axis=1)]
                m, s = A.mean_sd(top)
                rows.append(dict(arch=a, variant=v, condition=c, top_share_mean=m, top_share_sd=s, top_classes="/".join(names)))
                md.append(f"| {A.ARCH_LABEL[a]} | {A.VAR_LABEL[v]} | {c} | {A.fmt(top)} | {', '.join(names)} |")
    return rows, md


def fig_collapse(runs, out, variants=("std", "aug")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    ts = true_share(runs).max()
    for ax, kind in zip(axes, ("noise", "res")):
        lv = [l for l, _ in A.conds(kind)]
        x = np.arange(len(lv) + 1)
        for a, v in configs(runs, variants):
            ys = [pred_share(runs, a, v, c).max(axis=1) for c in ["clean"] + primary_conds(kind)]
            m = np.array([y.mean() for y in ys])
            sd = np.array([y.std(ddof=1) if len(y) > 1 else 0.0 for y in ys])
            kw = A.series_kw(a, v)
            ax.fill_between(x, m - sd, m + sd, color=kw["color"], alpha=0.10, linewidth=0)
            ax.plot(x, m, label=f"{A.ARCH_LABEL[a]}, {A.VAR_LABEL[v]}", **kw)
        ax.axhline(ts, color=A.MUTED, linestyle=(0, (1, 2)), linewidth=1.2)
        ax.text(x[0], ts - 0.012, f"largest true class: {ts * 100:.1f}%", color=A.INK2, fontsize=8.5, va="top")
        ax.set_xticks(x)
        if kind == "noise":
            ax.set_xticklabels(["0"] + [f"{l:g}" for l in lv])
            ax.set_xlabel("Gaussian noise standard deviation σ (pixels scaled to [0, 1])")
            ax.set_title("Gaussian noise")
        else:
            ax.set_xticklabels(["48"] + [str(l) for l in lv])
            ax.set_xlabel("Intermediate image size (pixels per side; 48 = original)")
            ax.set_title("Reduced resolution")
        ax.yaxis.set_major_formatter(lambda t, _: f"{t * 100:.0f}%")
        A.style(ax)
    axes[0].set_ylabel("Share of test images assigned\nto the most-predicted class")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2 if len(l) <= 6 else 3, frameon=False, fontsize=9, labelcolor=A.INK2,
               handlelength=3, bbox_to_anchor=(0.5, 1.0))
    nrows = int(np.ceil(len(l) / (2 if len(l) <= 6 else 3)))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.065 * nrows - 0.02))
    fig.savefig(os.path.join(out, "fig_prediction_collapse.png"))
    plt.close(fig)


# ---------- 3. predicted-class distribution heatmaps ----------
def fig_pred_heatmaps(runs, out, variant="std"):
    archs = [a for a in A.ARCHS if A.seeds_of(runs, a, variant)]
    if not archs:
        return
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", A.SEQ_BLUES)
    conds = ["clean"] + primary_conds("noise") + primary_conds("res")
    ylab = ["original"] + [f"σ = {l:g}" for l, _ in A.conds("noise")] + [f"{l} px" for l, _ in A.conds("res")]
    fig, axes = plt.subplots(1, len(archs) + 1, figsize=(3.6 * len(archs) + 1.5, 5.2),
                             gridspec_kw={"width_ratios": [7] * len(archs) + [0.0001]})
    axes = np.atleast_1d(axes)
    axes[-1].axis("off")
    ts = true_share(runs)
    for ax, a in zip(axes, archs):
        M = np.array([pred_share(runs, a, variant, c).mean(axis=0) for c in conds])
        M = np.vstack([ts[None, :], M])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if M[i, j] > 0.5 else A.INK)
        ax.set_xticks(range(7))
        ax.set_xticklabels(fl.FER_LABELS, rotation=40, ha="right")
        ax.set_yticks(range(len(ylab) + 1))
        ax.set_yticklabels(["true labels"] + ylab if a == archs[0] else [""] * (len(ylab) + 1))
        ax.set_title(A.ARCH_LABEL[a])
        ax.axhline(0.5, color=A.SURFACE, linewidth=3)
        ax.axhline(1.5, color=A.SURFACE, linewidth=2)
        ax.axhline(6.5, color=A.SURFACE, linewidth=2)
        ax.tick_params(length=0, labelcolor=A.INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.title.set_color(A.INK)
    fig.suptitle(f"Where the predictions go: share of test images assigned to each class (%), {A.VAR_LABEL[variant]}",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=A.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, f"fig_predicted_distribution_{variant}.png"))
    plt.close(fig)


# ---------- 4. test-set bootstrap ----------
def load_correct(out_dir, runs, y):
    """correct[(arch, variant)][cond] -> array (n_seeds, n_images) of 0/1."""
    out = {}
    for a, v in configs(runs):
        per = {}
        zs = []
        for s in A.seeds_of(runs, a, v):
            name = runs[(a, v, s)]["name"]
            p = os.path.join(out_dir, "final", name + "_preds.npz")
            if not os.path.exists(p):
                zs = None
                break
            zs.append(np.load(p))
        if zs is None:
            continue
        for c in ["clean"] + primary_conds("noise") + primary_conds("res"):
            per[c] = np.stack([(z[f"{SPLIT}__{c}"].astype(np.int64) == y).astype(np.float64) for z in zs])
        out[(a, v)] = per
    return out


def boot_ci(per_image, idx):
    """per_image: (n,) vector; idx: (B, n) resampling indices. Returns mean, lo, hi (percentile 95%)."""
    bs = per_image[idx].mean(axis=1)
    return float(per_image.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def bootstrap_tables(correct, n):
    rng = np.random.RandomState(BOOT_SEED)
    idx = rng.randint(0, n, size=(BOOT_B, n))
    rows, md = [], []
    groups = {"clean": ["clean"], "noise_avg": primary_conds("noise"), "res_avg": primary_conds("res")}

    def per_image(key, conds):
        return np.mean([correct[key][c].mean(axis=0) for c in conds], axis=0)

    md.append("### Seed-averaged accuracy with a 95% bootstrap interval over test images (models held fixed)")
    md.append("| Model | Training | Clean | Noise avg | Resolution avg |")
    md.append("|---|---|---|---|---|")
    for key in correct:
        cells = []
        for g, cl in groups.items():
            m, lo, hi = boot_ci(per_image(key, cl), idx)
            rows.append(dict(kind="level", a=key[0], va=key[1], b="", vb="", quantity=g, mean=m, lo=lo, hi=hi))
            cells.append(f"{m * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]")
        md.append(f"| {A.ARCH_LABEL[key[0]]} | {A.VAR_LABEL[key[1]]} | " + " | ".join(cells) + " |")

    def diff_block(title, pairs):
        if not pairs:
            return
        md.append("")
        md.append(f"### {title}")
        md.append("Difference in seed-averaged accuracy, percentage points, with a 95% bootstrap interval over test images.")
        md.append("| Comparison | Clean | Noise avg | Resolution avg |")
        md.append("|---|---|---|---|")
        for (k1, k2, label) in pairs:
            cells = []
            for g, cl in groups.items():
                m, lo, hi = boot_ci(per_image(k1, cl) - per_image(k2, cl), idx)
                rows.append(dict(kind="diff", a=k1[0], va=k1[1], b=k2[0], vb=k2[1], quantity=g, mean=m, lo=lo, hi=hi))
                star = "" if lo <= 0 <= hi else " *"
                cells.append(f"{m * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]{star}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")

    archs = [a for a in A.ARCHS if (a, "std") in correct]
    diff_block("Noise augmentation minus standard training (same architecture)",
               [((a, "aug"), (a, "std"), A.ARCH_LABEL[a]) for a in archs if (a, "aug") in correct])
    diff_block("Clean-selected checkpoint minus proposal-rule checkpoint (standard training)",
               [((a, "ref"), (a, "std"), A.ARCH_LABEL[a]) for a in archs if (a, "ref") in correct])
    for v in ALL_VARIANTS:
        av = [a for a in A.ARCHS if (a, v) in correct]
        pairs = [((av[j], v), (av[i], v), f"{A.ARCH_LABEL[av[j]]} minus {A.ARCH_LABEL[av[i]]}")
                 for i in range(len(av)) for j in range(i + 1, len(av))]
        diff_block(f"Architecture comparisons, {A.VAR_LABEL[v]}", pairs)
    md.append("")
    md.append("`*` marks intervals that exclude zero. These intervals reflect test-set sampling only; "
              "seed-to-seed variation is reported separately as the SD over three seeds.")
    return rows, md


# ---------- 5. per-class recall retention ----------
def recall_retention(runs, noise_level=0.1, res_level=24):
    cn, cr = fl.cond_name("noise", noise_level), fl.cond_name("res", res_level)
    md = [f"Recall on original images -> recall at σ = {noise_level:g} / at {res_level} px (seed mean, %).",
          "| Model | Training | " + " | ".join(fl.FER_LABELS) + " |", "|---|---|" + "---|" * 7]
    rows = []
    for a, v in configs(runs):
        cells = []
        for k in range(7):
            r0 = np.mean([runs[(a, v, s)]["metrics"][SPLIT]["clean"]["recall"][k] for s in A.seeds_of(runs, a, v)])
            rn = np.mean([runs[(a, v, s)]["metrics"][SPLIT][cn]["recall"][k] for s in A.seeds_of(runs, a, v)])
            rr = np.mean([runs[(a, v, s)]["metrics"][SPLIT][cr]["recall"][k] for s in A.seeds_of(runs, a, v)])
            cells.append(f"{r0 * 100:.0f} → {rn * 100:.0f} / {rr * 100:.0f}")
            rows.append(dict(arch=a, variant=v, cls=fl.FER_LABELS[k], clean=r0, noise=rn, res=rr))
        md.append(f"| {A.ARCH_LABEL[a]} | {A.VAR_LABEL[v]} | " + " | ".join(cells) + " |")
    return rows, md


def main(out_dir, data_npz=None):
    runs = A.load_runs(out_dir)
    if not runs:
        print("no final runs found")
        return
    out = os.path.join(out_dir, "analysis_extra")
    os.makedirs(out, exist_ok=True)
    doc = ["# Extra analyses (test split)", "",
           f"Final runs found: {len(runs)}. Mean ± sample SD over seeds unless stated otherwise.", ""]
    payload = {}

    rows, md = headline(runs)
    payload["headline"] = rows
    doc += ["## 1. Headline numbers", "",
            "Noise avg / Resolution avg = accuracy averaged over the five primary severity levels "
            f"(σ ∈ {fl.NOISE_PRIMARY}; sizes {fl.RES_PRIMARY}).", ""] + md + [""]

    rows, md = collapse_table(runs)
    payload["collapse"] = rows
    ts = true_share(runs)
    doc += ["## 2. Prediction collapse", "",
            "True class shares in the test split: " + ", ".join(f"{n} {s * 100:.1f}%" for n, s in zip(fl.FER_LABELS, ts)) + ".", ""] + md + [""]
    fig_collapse(runs, out)
    for v in ALL_VARIANTS:
        fig_pred_heatmaps(runs, out, v)

    data_npz = data_npz or os.path.join(out_dir, "fer_dedup.npz")
    if os.path.exists(data_npz):
        y = fl.load_data(data_npz)[SPLIT][1].astype(np.int64)
        correct = load_correct(out_dir, runs, y)
        # consistency check against the stored accuracies
        for (a, v), per in correct.items():
            stored = A.vals(runs, a, v, SPLIT, "clean")
            assert np.allclose(per["clean"].mean(axis=1), stored, atol=1e-9), (a, v)
        rows, md = bootstrap_tables(correct, len(y))
        payload["bootstrap"] = rows
        doc += ["## 3. Test-set sampling uncertainty", "",
                f"Bootstrap with {BOOT_B} resamples of the {len(y)} test images; trained models are held fixed.", ""] + md + [""]
    else:
        doc += ["## 3. Test-set sampling uncertainty", "", f"skipped: {data_npz} not found", ""]

    rows, md = recall_retention(runs)
    payload["recall_retention"] = rows
    doc += ["## 4. Per-class recall under a mid-level degradation", ""] + md + [""]

    with open(os.path.join(out, "extra_summary.md"), "w") as f:
        f.write("\n".join(doc))
    with open(os.path.join(out, "extra_summary.json"), "w") as f:
        json.dump(payload, f, indent=1)
    print("\n".join(doc))
    print(f"extra analysis written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
