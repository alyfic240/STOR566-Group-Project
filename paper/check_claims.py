"""Check the qualitative statements that are written out in words in main.tex.

Numbers in the paper come from generated/numbers.tex, so they follow the results automatically.
Statements such as "the sign is the same in all three seeds" or "less than one point" do not; this
script re-derives each of them from the run files and fails loudly if one no longer holds.

usage: python3 -I paper/check_claims.py <results_dir> [paper_dir]
"""
import glob
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
PAPER = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(PAPER, "generated", "numbers.json")))

NOISE = [0.025, 0.05, 0.1, 0.15, 0.2]
RES = [40, 32, 24, 16, 8]
ARCHS = ("cnn", "resnet18", "vit_b16")

runs = {}
for f in sorted(glob.glob(os.path.join(OUT, "final", "*.json"))):
    d = json.load(open(f))
    arch = d["name"].split("-")[0]
    seed = int(d["name"].rsplit("seed", 1)[1])
    runs[(arch, d["variant"], seed)] = d


def acc(a, v, s, what):
    m = runs[(a, v, s)]["metrics"]["test"]
    if what == "clean":
        return m["clean"]["acc"] * 100
    if what == "navg":
        return float(np.mean([m[f"noise_{l:g}"]["acc"] for l in NOISE])) * 100
    if what == "ravg":
        return float(np.mean([m[f"res_{l}"]["acc"] for l in RES])) * 100
    raise KeyError(what)


def mean(a, v, what):
    return float(np.mean([acc(a, v, s, what) for s in range(3)]))


problems = []


def check(label, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + label + (f"   [{detail}]" if detail else ""))
    if not ok:
        problems.append(label)


print("run counts")
check("18 protocol runs", sum(1 for k in runs if k[1] in ("std", "aug")) == 18 and N["nfinal"] == "18")
check("9 reference runs", sum(1 for k in runs if k[1] == "ref") == 9 and N["nref"] == "9")

print("references")
nbib = open(os.path.join(PAPER, "main.bbl")).read().count("\\bibitem")
check("the bibliography has 40 entries (the slides say 'From 5 to 40 references')", nbib == 40, str(nbib))

print("Section 5.1")
sds = {a: float(np.std([acc(a, "std", s, "clean") for s in range(3)], ddof=1)) for a in ARCHS}
check("seed SD on original images is at most 1.4 points (standard training)",
      max(sds.values()) < 1.45, ", ".join(f"{a} {v:.2f}" for a, v in sds.items()))

print("Section 5.5: reference checkpoints")
ep = {(a, v): [runs[(a, v, s)]["best_epoch"] for s in range(3)] for a in ARCHS for v in ("std", "ref")}
for a in ("resnet18", "cnn"):
    check(f"{a}: reference runs stop later or at the same epoch",
          all(r >= s for r, s in zip(ep[(a, "ref")], ep[(a, "std")])), f"ref {ep[(a, 'ref')]} std {ep[(a, 'std')]}")
check("vit_b16: the two rules stop at most one epoch apart",
      all(abs(r - s) <= 1 for r, s in zip(ep[("vit_b16", "ref")], ep[("vit_b16", "std")])),
      f"ref {ep[('vit_b16', 'ref')]} std {ep[('vit_b16', 'std')]}")
d = {(a, w): [acc(a, "ref", s, w) - acc(a, "std", s, w) for s in range(3)] for a in ARCHS for w in ("clean", "navg")}
fmt = lambda xs: ", ".join(f"{x:+.1f}" for x in xs)
check("resnet18: reference more accurate on original images in all three seeds",
      all(x > 0 for x in d[("resnet18", "clean")]), fmt(d[("resnet18", "clean")]))
check("resnet18: reference less accurate under noise in all three seeds",
      all(x < 0 for x in d[("resnet18", "navg")]), fmt(d[("resnet18", "navg")]))
check("cnn: sign not the same in all three seeds (original images)",
      not (all(x > 0 for x in d[("cnn", "clean")]) or all(x < 0 for x in d[("cnn", "clean")])), fmt(d[("cnn", "clean")]))
check("cnn: sign not the same in all three seeds (noise)",
      not (all(x > 0 for x in d[("cnn", "navg")]) or all(x < 0 for x in d[("cnn", "navg")])), fmt(d[("cnn", "navg")]))
for a in ("resnet18", "cnn"):
    check(f"{a}: reference more accurate on original images and less accurate under noise on average",
          np.mean(d[(a, "clean")]) > 0 and np.mean(d[(a, "navg")]) < 0,
          f"{np.mean(d[(a, 'clean')]):+.1f} / {np.mean(d[(a, 'navg')]):+.1f}")
check("vit_b16: mean differences below one point",
      abs(np.mean(d[("vit_b16", "clean")])) < 1 and abs(np.mean(d[("vit_b16", "navg")])) < 1,
      f"{np.mean(d[('vit_b16', 'clean')]):+.1f} / {np.mean(d[('vit_b16', 'navg')]):+.1f}")
check("with reference checkpoints ResNet-18 is no better than the CNN under noise",
      mean("resnet18", "ref", "navg") <= mean("cnn", "ref", "navg") + 0.05,
      f"{mean('resnet18', 'ref', 'navg'):.1f} vs {mean('cnn', 'ref', 'navg'):.1f}")
for w in ("clean", "navg", "ravg"):
    check(f"with reference checkpoints ViT-B/16 is the most accurate model ({w})",
          mean("vit_b16", "ref", w) > max(mean("cnn", "ref", w), mean("resnet18", "ref", w)),
          ", ".join(f"{a} {mean(a, 'ref', w):.1f}" for a in ARCHS))
g = {a: float(np.mean([acc(a, "aug", s, "navg") - acc(a, "ref", s, "navg") for s in range(3)])) for a in ARCHS}
check("augmentation gain under noise against reference: smallest for the CNN, largest for ResNet-18",
      g["cnn"] == min(g.values()) and g["resnet18"] == max(g.values()), ", ".join(f"{a} {v:+.1f}" for a, v in g.items()))
c = {a: float(np.mean([acc(a, "aug", s, "clean") - acc(a, "ref", s, "clean") for s in range(3)])) for a in ARCHS}
check("augmentation against reference costs less than one point on original images, for every model",
      all(-1 < v < 0 for v in c.values()), ", ".join(f"{a} {v:+.1f}" for a, v in c.items()))

print("Limitations (v) and Appendix A: learning-rate check")
lrc = json.load(open(os.path.join(OUT, "posthoc", "lr_check_summary.json")))
for a in ("resnet18", "vit_b16"):
    ss = [runs[(a, "std", s)]["best_score"] for s in range(3)]
    sd = float(np.std(ss, ddof=1))
    diff = lrc[a]["score"] - lrc[a]["selected_score"]
    check(f"{a}: change in S is smaller than the seed-to-seed SD of S",
          abs(diff) < sd, f"diff {diff:+.4f}, SD {sd:.4f}")
    check(f"{a}: checked learning rate is above the grid",
          lrc[a]["lr"] > max(float(k) for k in lrc[a]["grid_scores"]) and
          lrc[a]["selected_lr"] == max(float(k) for k in lrc[a]["grid_scores"]),
          f"lr {lrc[a]['lr']:g}, grid {sorted(float(k) for k in lrc[a]['grid_scores'])}")

print("Appendix A: which seed-0 final runs are the selected tuning runs")
reused = sorted(k for k in runs if runs[k].get("reused_tuning_run"))
check("ViT-B/16 seed-0 std and aug final runs are the selected tuning runs; every other run was trained as a final run",
      reused == [("vit_b16", "aug", 0), ("vit_b16", "std", 0)], str(reused))
for a, v in (("vit_b16", "std"), ("vit_b16", "aug")):
    r = runs[(a, v, 0)]
    t = json.load(open(os.path.join(OUT, "tune", r["name"] + ".json")))
    check(f"{a} {v} seed 0: final run and tuning run agree (score, epoch)",
          abs(t["best_score"] - r["best_score"]) < 1e-12 and t["best_epoch"] == r["best_epoch"],
          f"{t['best_score']:.4f} / {r['best_score']:.4f}")

print("Appendix A: compute and software")
tv = lambda d: ".".join(d["torch_version"].split(".")[:2])
colab = [k for k in runs if not (k[0] == "vit_b16" and k[1] == "ref")]
check("protocol runs and CNN / ResNet-18 reference runs: PyTorch 2.14", {tv(runs[k]) for k in colab} == {"2.14"})
check("ViT-B/16 reference runs: PyTorch 2.11", {tv(runs[k]) for k in runs if k[0] == "vit_b16" and k[1] == "ref"} == {"2.11"})
ph = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "posthoc", "tune", "*.json"))]
check("learning-rate check: PyTorch 2.11", len(ph) == 2 and {tv(x) for x in ph} == {"2.11"})
tn = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "tune", "*.json"))]
check("tuning runs on file: PyTorch 2.14", len(tn) > 0 and {tv(x) for x in tn} == {"2.14"}, f"{len(tn)} files")
check("all runs on a GPU", {runs[k]["device"] for k in runs} == {"cuda"})
vt = [runs[k]["train_seconds"] / 60 for k in runs if k[0] == "vit_b16"]
check("ViT-B/16 training time roughly 20 minutes per run", 17 < min(vt) and max(vt) < 23, f"{min(vt):.1f} to {max(vt):.1f} min")
tot = sum(runs[k]["train_seconds"] for k in runs) / 3600
check("total training time matches the macro", f"{tot:.1f}" == N["gpuhours@final"], f"{tot:.2f} h")
check("one data fingerprint", len({runs[k]["data_fingerprint"] for k in runs}) == 1)

print()
if problems:
    print(f"{len(problems)} statement(s) no longer hold:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("all written-out statements hold")
