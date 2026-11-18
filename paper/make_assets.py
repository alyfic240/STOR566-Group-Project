#!/usr/bin/env python3
"""Build every generated asset of the paper (figures, LaTeX tables, number macros) from an experiment folder.

Usage: python3 make_assets.py <out_dir> <data_npz> [paper_dir]

<out_dir> holds final/*.json, final/*_preds.npz, tune/*.json, selection_*.json (the Colab results folder
or an extracted results_bundle.zip). Nothing in the paper is typed by hand from these numbers: tables are
written to <paper_dir>/generated/*.tex and single numbers are exposed as macros in generated/numbers.tex.
"""
import glob
import json
import re
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the pipeline code lives next to this folder: "src" in the repository, "fer566" in the original workspace
for _d in ("src", "fer566"):
    if os.path.isdir(os.path.join(os.path.dirname(HERE), _d)):
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), _d))
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure

matplotlib.rcParams["pdf.fonttype"] = 42      # embed TrueType, no Type 3 fonts
_orig_savefig = matplotlib.figure.Figure.savefig


def _savefig_both(self, fname, *a, **k):
    _orig_savefig(self, fname, *a, **k)
    if isinstance(fname, str) and fname.endswith(".png"):
        _orig_savefig(self, fname[:-4] + ".pdf", *a, **k)


matplotlib.figure.Figure.savefig = _savefig_both

import analyze as A
import analyze_extra as X
import ferlib as fl

SPLIT = "test"
ARCH_TEX = {"cnn": "Baseline CNN", "resnet18": "ResNet-18", "vit_b16": "ViT-B/16"}
VAR_TEX = {"std": "standard", "aug": "noise aug.", "ref": "standard (clean-sel.)"}
VAR_TEX_SHORT = {"std": "standard", "aug": "noise aug.", "ref": "clean-sel."}


def tex_minus(s):
    """A hyphen in front of a number is typeset as a minus sign."""
    return s.replace("-", "$-$")


def pm(x, digits=1, bold=False, signed=False):
    m, s = A.mean_sd(x)
    if np.isnan(m):
        return "--"
    body = tex_minus(f"{m * 100:+.{digits}f}") if signed else f"{m * 100:.{digits}f}"
    if bold:
        body = r"\textbf{" + body + "}"
    if np.isnan(s):
        return body
    return body + r"{\scriptsize$\,\pm\,$" + f"{s * 100:.{digits}f}" + "}"


def configs(runs, variants=("std", "aug")):
    return [(a, v) for a in A.ARCHS for v in variants if A.seeds_of(runs, a, v)]


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def quantities(runs, a, v):
    q = {}
    for metric in ("acc", "macro_f1"):
        q[("clean", metric)] = A.vals(runs, a, v, SPLIT, "clean", metric)
        q[("navg", metric)] = X.avg_over(runs, a, v, X.primary_conds("noise"), metric)
        q[("ravg", metric)] = X.avg_over(runs, a, v, X.primary_conds("res"), metric)
    return q


# ---------------------------------------------------------------- tables
def tab_main(runs, out):
    cfgs = configs(runs)
    Q = {c: quantities(runs, *c) for c in cfgs}
    cols = [("clean", "acc"), ("clean", "macro_f1"), ("navg", "acc"), ("navg", "macro_f1"), ("ravg", "acc"), ("ravg", "macro_f1")]
    best = {c: max(cfgs, key=lambda k: Q[k][c].mean()) for c in cols}
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Original images} & \multicolumn{2}{c}{Gaussian noise (avg.)} & \multicolumn{2}{c}{Reduced resolution (avg.)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Model & Training & Acc. & Macro-F1 & Acc. & Macro-F1 & Acc. & Macro-F1 \\", r"\midrule"]
    prev = None
    for (a, v) in cfgs:
        if prev is not None and a != prev:
            L.append(r"\addlinespace")
        name = ARCH_TEX[a] if a != prev else ""
        cells = [pm(Q[(a, v)][c], bold=(best[c] == (a, v))) for c in cols]
        L.append(f"{name} & {VAR_TEX[v]} & " + " & ".join(cells) + r" \\")
        prev = a
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_main.tex"), L)


def tab_levels(runs, out, kind, variants=("std", "aug"), name=None, extended=False, metric="acc"):
    lv = A.conds(kind, extended=extended)
    cfgs = configs(runs, variants)
    head = [f"{l:g}" if kind == "noise" else f"{l}" for l, _ in lv]
    L = [r"\begin{tabular}{ll" + "c" * (len(lv) + 1) + "}", r"\toprule",
         r" & & & \multicolumn{" + str(len(lv)) + "}{c}{" + (r"Noise standard deviation $\sigma$" if kind == "noise" else "Intermediate size $s$ (pixels)") + r"} \\",
         r"\cmidrule(lr){4-" + str(len(lv) + 3) + "}",
         "Model & Training & Original & " + " & ".join(head) + r" \\", r"\midrule"]
    prev = None
    for (a, v) in cfgs:
        if prev is not None and a != prev:
            L.append(r"\addlinespace")
        cells = [pm(A.vals(runs, a, v, SPLIT, "clean", metric))] + [pm(A.vals(runs, a, v, SPLIT, c, metric)) for _, c in lv]
        L.append(f"{ARCH_TEX[a] if a != prev else ''} & {VAR_TEX[v]} & " + " & ".join(cells) + r" \\")
        prev = a
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, name or f"tab_{kind}_{metric}.tex"), L)


def relative(runs, a, v, cond):
    return A.vals(runs, a, v, SPLIT, cond) / A.vals(runs, a, v, SPLIT, "clean")


def first_below(runs, a, v, kind, thr=0.9):
    for l, c in A.conds(kind):
        if relative(runs, a, v, c).mean() < thr:
            return l
    return None


def tab_relative(runs, out):
    nz, rs = A.conds("noise"), A.conds("res")
    L = [r"\begin{tabular}{ll" + "c" * len(nz) + "c" + "c" * len(rs) + "c}", r"\toprule",
         r" & & \multicolumn{" + str(len(nz) + 1) + r"}{c}{Gaussian noise $\sigma$} & \multicolumn{" + str(len(rs) + 1) + r"}{c}{Intermediate size $s$} \\",
         r"\cmidrule(lr){3-" + str(len(nz) + 3) + r"}\cmidrule(lr){" + str(len(nz) + 4) + "-" + str(len(nz) + len(rs) + 4) + "}",
         "Model & Training & " + " & ".join(f"{l:g}" for l, _ in nz) + r" & $<$90\% at & " + " & ".join(str(l) for l, _ in rs) + r" & $<$90\% at \\",
         r"\midrule"]
    prev = None
    for (a, v) in configs(runs):
        if prev is not None and a != prev:
            L.append(r"\addlinespace")
        cells = []
        for kind, lv in (("noise", nz), ("res", rs)):
            cells += [f"{relative(runs, a, v, c).mean() * 100:.0f}" for _, c in lv]
            fb = first_below(runs, a, v, kind)
            cells.append("--" if fb is None else (f"$\\sigma{{=}}{fb:g}$" if kind == "noise" else f"{fb}\\,px"))
        L.append(f"{ARCH_TEX[a] if a != prev else ''} & {VAR_TEX[v]} & " + " & ".join(cells) + r" \\")
        prev = a
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_relative.tex"), L)


def paired_delta(runs, a, cond_list, v1="aug", v0="std", metric="acc"):
    s1, s0 = A.seeds_of(runs, a, v1), A.seeds_of(runs, a, v0)
    seeds = [s for s in s1 if s in s0]
    def per_seed(v, s):
        return np.mean([runs[(a, v, s)]["metrics"][SPLIT][c][metric] for c in cond_list])
    return np.array([per_seed(v1, s) - per_seed(v0, s) for s in seeds])


def tab_aug_delta(runs, out):
    nz, rs = A.conds("noise"), A.conds("res")
    archs = [a for a in A.ARCHS if A.seeds_of(runs, a, "std") and A.seeds_of(runs, a, "aug")]
    L = [r"\begin{tabular}{l" + "c" * (1 + len(nz) + len(rs)) + "}", r"\toprule",
         r" & & \multicolumn{" + str(len(nz)) + r"}{c}{Gaussian noise $\sigma$ (seen in training)} & \multicolumn{" + str(len(rs)) + r"}{c}{Intermediate size $s$ (never seen)} \\",
         r"\cmidrule(lr){3-" + str(len(nz) + 2) + r"}\cmidrule(lr){" + str(len(nz) + 3) + "-" + str(len(nz) + len(rs) + 2) + "}",
         "Model & Original & " + " & ".join(f"{l:g}" for l, _ in nz) + " & " + " & ".join(str(l) for l, _ in rs) + r" \\", r"\midrule"]
    for a in archs:
        cells = [pm(paired_delta(runs, a, ["clean"]), signed=True)]
        cells += [pm(paired_delta(runs, a, [c]), signed=True) for _, c in nz + rs]
        L.append(f"{ARCH_TEX[a]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_aug_delta.tex"), L)


def tab_ref(runs, out):
    archs = [a for a in A.ARCHS if A.seeds_of(runs, a, "ref")]
    if not archs:
        write(os.path.join(out, "tab_ref.tex"), [r"\emph{Reference runs are not available yet.}"])
        return False
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Model & Checkpoint rule & Epochs & Original & Noise avg. & Resolution avg. \\", r"\midrule"]
    for i, a in enumerate(archs):
        if i:
            L.append(r"\addlinespace")
        for v, lab in (("std", r"score $S$"), ("ref", "original only")):
            if not A.seeds_of(runs, a, v):
                continue
            ep = ", ".join(str(runs[(a, v, s)]["best_epoch"]) for s in A.seeds_of(runs, a, v))
            q = quantities(runs, a, v)
            L.append(f"{ARCH_TEX[a] if v == 'std' else ''} & {lab} & {ep} & {pm(q[('clean', 'acc')])} & {pm(q[('navg', 'acc')])} & {pm(q[('ravg', 'acc')])}" + r" \\")
        if A.seeds_of(runs, a, "std"):
            d = [pm(paired_delta(runs, a, cl, v1="ref", v0="std"), signed=True)
                 for cl in (["clean"], X.primary_conds("noise"), X.primary_conds("res"))]
            L.append(r" & \emph{difference} & & " + " & ".join(d) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_ref.tex"), L)
    return True


def tab_recall(runs, out, variants=("std", "aug")):
    L = [r"\begin{tabular}{ll" + "c" * 7 + "}", r"\toprule",
         "Model & Training & " + " & ".join(c.capitalize() for c in fl.FER_LABELS) + r" \\", r"\midrule"]
    prev = None
    for (a, v) in configs(runs, variants):
        if prev is not None and a != prev:
            L.append(r"\addlinespace")
        rec = np.mean([runs[(a, v, s)]["metrics"][SPLIT]["clean"]["recall"] for s in A.seeds_of(runs, a, v)], axis=0)
        L.append(f"{ARCH_TEX[a] if a != prev else ''} & {VAR_TEX[v]} & " + " & ".join(f"{r * 100:.1f}" for r in rec) + r" \\")
        prev = a
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_recall.tex"), L)


def tab_selection(out_dir, out):
    L = [r"\begin{tabular}{lp{0.36\linewidth}p{0.40\linewidth}}", r"\toprule",
         r"Model & Learning rate: score $S$ & Noise augmentation $(p, \sigma_{\max})$: score $S$ \\", r"\midrule"]
    sel = {}
    for a in A.ARCHS:
        p = os.path.join(out_dir, f"selection_{a}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        sel[a] = s
        lr = ", ".join((r"\textbf{%s: %.3f}" if float(k) == s["lr"] else "%s: %.3f") % (sci(float(k)), v) for k, v in s["lr_scores"].items())
        def is_best(k):
            pp, ss = k.split("|")
            return float(pp) == s["aug_p"] and float(ss) == s["aug_sigma_max"]
        au = ", ".join((r"\textbf{(%s, %s): %.3f}" if is_best(k) else "(%s, %s): %.3f") % (*k.split("|"), v) for k, v in s["aug_scores"].items())
        L.append(f"{ARCH_TEX[a]} & {lr} & {au}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_selection.tex"), L)
    return sel


def tab_bootstrap(out_dir, out):
    pth = os.path.join(out_dir, "analysis_extra", "extra_summary.json")
    rows = json.load(open(pth)).get("bootstrap") if os.path.exists(pth) else None
    if not rows:
        write(os.path.join(out, "tab_bootstrap.tex"), [r"\emph{Bootstrap intervals are not available.}"])
        return
    def cell(r, signed=False):
        f = "{:+.1f}" if signed else "{:.1f}"
        star = "" if (not signed or r["lo"] <= 0 <= r["hi"]) else "$^{*}$"
        return tex_minus(f.format(r["mean"] * 100) + r"{\scriptsize\,[" + f.format(r["lo"] * 100) + ", " + f.format(r["hi"] * 100) + "]}") + star
    L = [r"\begin{tabular}{lccc}", r"\toprule", r" & Original & Noise avg. & Resolution avg. \\", r"\midrule",
         r"\multicolumn{4}{l}{\emph{Seed-averaged accuracy (\%)}} \\"]
    lev = {}
    for r in rows:
        if r["kind"] == "level":
            lev.setdefault((r["a"], r["va"]), {})[r["quantity"]] = r
    for (a, v), q in lev.items():
        L.append(f"{ARCH_TEX[a]}, {VAR_TEX_SHORT[v]} & " + " & ".join(cell(q[k]) for k in ("clean", "noise_avg", "res_avg")) + r" \\")
    dif = {}
    for r in rows:
        if r["kind"] == "diff":
            dif.setdefault((r["a"], r["va"], r["b"], r["vb"]), {})[r["quantity"]] = r
    groups = [(r"\emph{Noise augmentation minus standard training (points)}", lambda k: k[0] == k[2] and k[1] == "aug" and k[3] == "std"),
              (r"\emph{Clean-selected minus score-selected checkpoint (points)}", lambda k: k[0] == k[2] and k[1] == "ref" and k[3] == "std"),
              (r"\emph{Between architectures, standard training (points)}", lambda k: k[0] != k[2] and k[1] == "std"),
              (r"\emph{Between architectures, noise augmentation (points)}", lambda k: k[0] != k[2] and k[1] == "aug")]
    for title, pred in groups:
        ks = [k for k in dif if pred(k)]
        if not ks:
            continue
        L += [r"\addlinespace", r"\multicolumn{4}{l}{" + title + r"} \\"]
        for k in ks:
            lab = ARCH_TEX[k[0]] if k[0] == k[2] else f"{ARCH_TEX[k[0]]} $-$ {ARCH_TEX[k[2]]}"
            L.append(f"{lab} & " + " & ".join(cell(dif[k][q], signed=True) for q in ("clean", "noise_avg", "res_avg")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tab_bootstrap.tex"), L)


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


# ---------------------------------------------------------------- number macros
def numbers(runs, out_dir, sel, out, audit_path=None):
    D = {}

    def put(key, val):
        D[key] = val

    for (a, v) in configs(runs, ("std", "aug", "ref")):
        q = quantities(runs, a, v)
        for (k, metric), x in q.items():
            m, s = A.mean_sd(x)
            put(f"{metric}@{a}@{v}@{k}", f"{m * 100:.1f}")
            put(f"{metric}sd@{a}@{v}@{k}", f"{s * 100:.1f}")
        for kind in ("noise", "res"):
            for l, c in A.conds(kind, extended=True):
                key = (f"n{l:g}" if kind == "noise" else f"r{l}")
                put(f"acc@{a}@{v}@{key}", f"{A.vals(runs, a, v, SPLIT, c).mean() * 100:.1f}")
                put(f"macro_f1@{a}@{v}@{key}", f"{A.vals(runs, a, v, SPLIT, c, 'macro_f1').mean() * 100:.1f}")
                put(f"rel@{a}@{v}@{key}", f"{relative(runs, a, v, c).mean() * 100:.0f}")
                p = X.pred_share(runs, a, v, c).max(axis=1)
                put(f"top@{a}@{v}@{key}", f"{p.mean() * 100:.0f}")
            fb = first_below(runs, a, v, kind)
            put(f"below@{a}@{v}@{kind}", "none" if fb is None else f"{fb:g}")
        put(f"top@{a}@{v}@clean", f"{X.pred_share(runs, a, v, 'clean').max(axis=1).mean() * 100:.0f}")
        for ckey, cname in (("clean", "clean"), ("n0.1", fl.cond_name("noise", 0.1)), ("r24", fl.cond_name("res", 24))):
            rec = np.mean([runs[(a, v, s)]["metrics"][SPLIT][cname]["recall"] for s in A.seeds_of(runs, a, v)], axis=0)
            for k, cls in enumerate(fl.FER_LABELS):
                put(f"recall@{a}@{v}@{cls}@{ckey}", f"{rec[k] * 100:.0f}")
        put(f"epochs@{a}@{v}", ", ".join(str(runs[(a, v, s)]["best_epoch"]) for s in A.seeds_of(runs, a, v)))
        put(f"nseeds@{a}@{v}", str(len(A.seeds_of(runs, a, v))))
    for a in A.ARCHS:
        if A.seeds_of(runs, a, "std") and A.seeds_of(runs, a, "aug"):
            for k, cl in (("clean", ["clean"]), ("navg", X.primary_conds("noise")), ("ravg", X.primary_conds("res"))):
                d = paired_delta(runs, a, cl)
                put(f"delta@{a}@{k}", f"{d.mean() * 100:+.1f}")
                put(f"deltasd@{a}@{k}", f"{d.std(ddof=1) * 100:.1f}" if len(d) > 1 else "n/a")
            for kind in ("noise", "res"):
                for l, c in A.conds(kind):
                    key = (f"n{l:g}" if kind == "noise" else f"r{l}")
                    put(f"delta@{a}@{key}", f"{paired_delta(runs, a, [c]).mean() * 100:+.1f}")
        if A.seeds_of(runs, a, "std") and A.seeds_of(runs, a, "ref"):
            for k, cl in (("clean", ["clean"]), ("navg", X.primary_conds("noise")), ("ravg", X.primary_conds("res"))):
                d = paired_delta(runs, a, cl, v1="ref", v0="std")
                put(f"refdelta@{a}@{k}", f"{d.mean() * 100:+.1f}")
                put(f"refdeltasd@{a}@{k}", f"{d.std(ddof=1) * 100:.1f}" if len(d) > 1 else "n/a")
                put(f"refabs@{a}@{k}", f"{abs(d.mean()) * 100:.1f}")
        if A.seeds_of(runs, a, "aug") and A.seeds_of(runs, a, "ref"):
            # augmentation measured against the clean-selected reference checkpoints
            for k, cl in (("clean", ["clean"]), ("navg", X.primary_conds("noise")), ("ravg", X.primary_conds("res"))):
                d = paired_delta(runs, a, cl, v1="aug", v0="ref")
                put(f"augref@{a}@{k}", f"{d.mean() * 100:+.1f}")
                put(f"augrefabs@{a}@{k}", f"{abs(d.mean()) * 100:.1f}")
        if a in sel:
            put(f"lr@{a}", sci(sel[a]["lr"]))
            put(f"augp@{a}", f"{sel[a]['aug_p']:g}")
            put(f"augsigma@{a}", f"{sel[a]['aug_sigma_max']:g}")
        ps = [runs[k]["n_params"] for k in runs if k[0] == a]
        if ps:
            put(f"params@{a}", f"{ps[0]:,}".replace(",", "{,}"))
    for v in ("std", "aug", "ref"):
        av = [a for a in A.ARCHS if A.seeds_of(runs, a, v)]
        for i in range(len(av)):
            for j in range(i + 1, len(av)):
                qa, qb = quantities(runs, av[j], v), quantities(runs, av[i], v)
                for k in ("clean", "navg", "ravg"):
                    put(f"archdiff@{av[j]}@{av[i]}@{v}@{k}", f"{(qa[(k, 'acc')].mean() - qb[(k, 'acc')].mean()) * 100:.1f}")
    ts = X.true_share(runs)
    put("majority", f"{ts.max() * 100:.1f}")
    put("nfinal", str(sum(1 for k in runs if k[1] in ("std", "aug"))))
    put("nref", str(sum(1 for k in runs if k[1] == "ref")))
    lrc_path = os.path.join(out_dir, "posthoc", "lr_check_summary.json")
    put("haslrcheck", "1" if os.path.exists(lrc_path) else "0")
    if os.path.exists(lrc_path):
        for a, r in json.load(open(lrc_path)).items():
            put(f"lrcheck@{a}@lr", sci(r["lr"]))
            put(f"lrcheck@{a}@score", f"{r['score']:.3f}")
            put(f"lrcheck@{a}@selscore", f"{r['selected_score']:.3f}")
            put(f"lrcheck@{a}@diff", f"{r['score'] - r['selected_score']:+.3f}")
            put(f"lrcheck@{a}@clean", f"{r['val_acc_clean'] * 100:.1f}")
            put(f"lrcheck@{a}@noisy", f"{r['val_acc_noisy'] * 100:.1f}")
            put(f"lrcheck@{a}@epoch", str(r["best_epoch"]))
            put(f"lrcheck@{a}@absdiff", f"{abs(r['score'] - r['selected_score']):.3f}")
    # seed-to-seed spread of the selection score S for standard training at the selected learning rate
    # (std runs keep the epoch with the highest S, so best_score is S at the kept checkpoint)
    for a in A.ARCHS:
        ss = [runs[(a, "std", s)]["best_score"] for s in A.seeds_of(runs, a, "std")]
        if len(ss) > 1:
            put(f"scoremean@{a}", f"{np.mean(ss):.3f}")
            put(f"scoresd@{a}", f"{np.std(ss, ddof=1):.3f}")
    # total training time of the final runs (GPU hours), by variant group
    tot = {"protocol": 0.0, "ref": 0.0}
    for k, r in runs.items():
        tot["ref" if k[1] == "ref" else "protocol"] += float(r.get("train_seconds", 0.0))
    put("gpuhours@protocol", f"{tot['protocol'] / 3600:.1f}")
    put("gpuhours@ref", f"{tot['ref'] / 3600:.1f}")
    put("gpuhours@final", f"{(tot['protocol'] + tot['ref']) / 3600:.1f}")
    n_test = int(np.array(next(iter(runs.values()))["metrics"][SPLIT]["clean"]["cm"]).sum())
    put("ntest", f"{n_test:,}".replace(",", "{,}"))
    if audit_path and os.path.exists(audit_path):
        au = json.load(open(audit_path))
        for k, v in au.items():
            if isinstance(v, (int, float)):
                put(f"audit@{k}", f"{v:,}".replace(",", "{,}") if isinstance(v, int) else f"{v:g}")
    L = ["% generated by make_assets.py -- do not edit",
         r"\makeatletter",
         r"\newcommand{\num}[1]{\@ifundefined{num@#1}{\textbf{??}}{\csname num@#1\endcsname}}",
         r"\makeatother"]
    for k in sorted(D):
        v = tex_minus(D[k]) if re.fullmatch(r"-\d+(\.\d+)?", D[k]) else D[k]
        L.append(r"\expandafter\def\csname num@" + k + r"\endcsname{" + v + "}")
    write(os.path.join(out, "numbers.tex"), L)
    json.dump(D, open(os.path.join(out, "numbers.json"), "w"), indent=1, sort_keys=True)
    return D


FIGS = ["analysis/fig_degradation_acc", "analysis/fig_degradation_macro_f1", "analysis/fig_relative_acc",
        "analysis/fig_degradation_acc_extended", "analysis/fig_by_architecture", "analysis/fig_augmentation_effect",
        "analysis/fig_per_class_recall", "analysis/fig_confusion_clean_std", "analysis/fig_confusion_clean_aug",
        "analysis/fig_learning_curves", "analysis_extra/fig_prediction_collapse",
        "analysis_extra/fig_predicted_distribution_std", "analysis_extra/fig_predicted_distribution_aug"]


def main(out_dir, data_npz, paper_dir=HERE):
    gen = os.path.join(paper_dir, "generated")
    figs = os.path.join(paper_dir, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    A.main(out_dir)
    X.main(out_dir, data_npz)
    for f in FIGS:
        for ext in (".pdf", ".png"):
            src = os.path.join(out_dir, f + ext)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(figs, os.path.basename(src)))
    runs = A.load_runs(out_dir)
    tab_main(runs, gen)
    tab_levels(runs, gen, "noise", extended=True, name="tab_noise_acc.tex")
    tab_levels(runs, gen, "res", extended=True, name="tab_res_acc.tex")
    tab_levels(runs, gen, "noise", extended=False, metric="macro_f1", name="tab_noise_f1.tex")
    tab_levels(runs, gen, "res", extended=False, metric="macro_f1", name="tab_res_f1.tex")
    tab_relative(runs, gen)
    tab_aug_delta(runs, gen)
    has_ref = tab_ref(runs, gen)
    tab_recall(runs, gen)
    sel = tab_selection(out_dir, gen)
    tab_bootstrap(out_dir, gen)
    audit = None
    for cand in (os.path.join(out_dir, "data_audit.json"), os.path.join(os.path.dirname(data_npz), "fer_dedup_audit.json")):
        if os.path.exists(cand):
            audit = cand
            break
    D = numbers(runs, out_dir, sel, gen, audit)
    print(f"assets written: {len(glob.glob(os.path.join(gen, '*.tex')))} tex files, {len(D)} number macros, "
          f"{len(glob.glob(os.path.join(figs, '*.pdf')))} pdf figures; reference runs present: {has_ref}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else HERE)
