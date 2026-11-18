"""Consistency checks on a results folder before it is used for the paper or the slides.

For every final run it checks that
  * the JSON name matches its configuration and the file name,
  * the stored confusion matrices, accuracy and macro-F1 can be reproduced from the saved predictions
    and the true labels of the audited data set,
  * all runs share one data fingerprint, parameter count per architecture and the frozen selection.

Usage:  python3 validate_runs.py <out_dir> <data_npz>
Exit code 0 when everything agrees, 1 otherwise.
"""
import glob
import json
import os
import sys

import numpy as np


def macro_f1(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1.mean()


def main(out_dir, data_npz):
    z = np.load(data_npz)
    y = {"val": z["y_val"], "test": z["y_test"]}
    problems, n_checked, fingerprints, params = [], 0, set(), {}
    files = sorted(glob.glob(os.path.join(out_dir, "final", "*.json")))
    counts = {}
    for p in files:
        r = json.load(open(p))
        stem = os.path.basename(p)[:-5]
        c = r["config"]
        if r["name"] != stem:
            problems.append(f"{stem}: name field is {r['name']}")
        want_variant = "aug" if c["aug_p"] > 0 else ("ref" if c["select_on"] == "clean" else "std")
        if r["variant"] != want_variant or f"-{want_variant}-" not in stem or not stem.endswith(f"seed{c['seed']}"):
            problems.append(f"{stem}: variant/seed do not match the configuration")
        sel_path = os.path.join(out_dir, f"selection_{c['arch']}.json")
        if os.path.exists(sel_path):
            sel = json.load(open(sel_path))
            if abs(c["lr"] - sel["lr"]) > 1e-12:
                problems.append(f"{stem}: lr {c['lr']} differs from the frozen selection {sel['lr']}")
            if want_variant == "aug" and (c["aug_p"], c["aug_sigma_max"]) != (sel["aug_p"], sel["aug_sigma_max"]):
                problems.append(f"{stem}: augmentation differs from the frozen selection")
        else:
            problems.append(f"{stem}: selection_{c['arch']}.json is missing")
        fingerprints.add(r.get("data_fingerprint"))
        params.setdefault(c["arch"], set()).add(r["n_params"])
        counts[(c["arch"], want_variant)] = counts.get((c["arch"], want_variant), 0) + 1
        if not (1 <= r["best_epoch"] <= r["epochs_run"] <= c["max_epochs"]) or len(r["history"]) != r["epochs_run"]:
            problems.append(f"{stem}: epoch bookkeeping is inconsistent")
        npz_path = p[:-5] + "_preds.npz"
        if not os.path.exists(npz_path):
            problems.append(f"{stem}: predictions file is missing")
            continue
        preds = np.load(npz_path)
        for split in ("val", "test"):
            for cond, m in r["metrics"][split].items():
                key = f"{split}__{cond}"
                if key not in preds.files:
                    problems.append(f"{stem}: no predictions for {key}")
                    continue
                yhat = preds[key]
                if len(yhat) != len(y[split]):
                    problems.append(f"{stem}: {key} has {len(yhat)} predictions, expected {len(y[split])}")
                    continue
                cm = np.zeros((7, 7), int)
                np.add.at(cm, (y[split], yhat), 1)
                if not np.array_equal(cm, np.array(m["cm"])):
                    problems.append(f"{stem}: {key} confusion matrix does not match the predictions")
                if abs((yhat == y[split]).mean() - m["acc"]) > 1e-9:
                    problems.append(f"{stem}: {key} accuracy does not match the predictions")
                if abs(macro_f1(cm) - m["macro_f1"]) > 1e-6:
                    problems.append(f"{stem}: {key} macro-F1 does not match the predictions")
                n_checked += 1
    if len(fingerprints) != 1:
        problems.append(f"data fingerprints differ: {fingerprints}")
    for a, s in params.items():
        if len(s) != 1:
            problems.append(f"{a}: parameter counts differ: {s}")
    print(f"{len(files)} final runs, {n_checked} (run, split, condition) cells re-computed from predictions")
    print("runs per configuration:", {f"{a}/{v}": n for (a, v), n in sorted(counts.items())})
    print("data fingerprint:", ", ".join(str(f)[:16] for f in fingerprints))
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for q in problems[:40]:
            print("  -", q)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
