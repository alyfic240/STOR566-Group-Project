"""Audit FER-2013, apply the proposal's duplicate filtering, and save fixed splits.
Usage: python3 -I prep_data.py <path to fer2013.csv or icml_face_data.csv> <output .npz>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import ferlib as fl

src, out = sys.argv[1], sys.argv[2]
X, y, usage = fl.read_fer_csv(src)
print("rows:", len(X), "| usage:", {u: int((usage == u).sum()) for u in np.unique(usage)})
data, st = fl.audit_and_dedup(X, y, usage)
print(json.dumps(st, indent=1))
for s in ("train", "val", "test"):
    c = fl.class_counts(data[s][1])
    print(f"{s:5s} n={len(data[s][1]):6d}  " + "  ".join(f"{n}={v}" for n, v in zip(fl.FER_LABELS, c)))
ok = st["final_sizes"] == fl.EXPECTED_SIZES
print("sizes match proposal (27,418 / 3,291 / 3,268):", ok)
x = data["train"][0].astype(np.float64) / 255.0
print(f"train pixel mean {x.mean():.4f} std {x.std():.4f}")
fp = fl.data_fingerprint(data)
print("fingerprint:", fp)
fl.save_data(data, out)
json.dump(dict(audit=st, fingerprint=fp, train_mean=float(x.mean()), train_std=float(x.std()),
               class_counts={s: fl.class_counts(data[s][1]).tolist() for s in data}),
          open(os.path.splitext(out)[0] + "_audit.json", "w"), indent=1)
