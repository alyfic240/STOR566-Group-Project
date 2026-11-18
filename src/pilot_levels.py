"""Level-setting pilot (validation images only; the test split is never loaded into a model here).

Trains one baseline CNN with standard training, selected on clean validation accuracy, then sweeps a
dense grid of noise and resolution severities on the validation split. The five primary levels per
degradation in ferlib.py are fixed from this sweep before any test evaluation.

Usage: python3 -I pilot_levels.py <data.npz> <out_dir> [max_epochs]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ferlib as fl

torch.set_num_threads(2)
data = fl.load_data(sys.argv[1])
out = sys.argv[2]
max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60
os.makedirs(out, exist_ok=True)
assert fl.data_fingerprint(data) == fl.EXPECTED_FINGERPRINT
dev = torch.device("cpu")
logf = open(os.path.join(out, "pilot_log.txt"), "a")


def log(m):
    print(m, flush=True)
    logf.write(m + "\n")
    logf.flush()


cfg = fl.TrainConfig(arch="cnn", lr=1e-3, seed=0, max_epochs=max_epochs, patience=10, select_on="clean")
state, info = fl.train_model(cfg, data, dev, log=log)
torch.save(state, os.path.join(out, "pilot_cnn.pt"))
model = fl.build_model("cnn")
model.load_state_dict(state)

NOISE_GRID = [0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RES_GRID = [44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 6, 4]
conds = [("clean", None)] + [("noise", s) for s in NOISE_GRID] + [("res", s) for s in RES_GRID]
res, _ = fl.evaluate_split(model, data, "val", dev, conds)
yv = data["val"][1]
majority = float(np.bincount(yv).max() / len(yv))
clean = res["clean"]["acc"]
log(f"pilot best epoch {info['best_epoch']} / {info['epochs_run']}; val clean acc {clean:.4f} "
    f"macro-F1 {res['clean']['macro_f1']:.4f}; majority-class rate {majority:.4f}")
for k, v in res.items():
    if k != "clean":
        log(f"  {k:12s} acc {v['acc']:.4f}  rel {v['acc'] / clean:.3f}  macro-F1 {v['macro_f1']:.4f}")
json.dump(dict(info=info, val=res, majority_rate=majority, noise_grid=NOISE_GRID, res_grid=RES_GRID),
          open(os.path.join(out, "pilot_results.json"), "w"))
log("PILOT_DONE")
