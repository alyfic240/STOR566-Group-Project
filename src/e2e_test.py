"""End-to-end smoke test of the full protocol on a small subset (CPU, random-weight ResNet).
Usage: python3 -I e2e_test.py <data.npz> <out_dir>
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ferlib as fl
import analyze

torch.set_num_threads(1)
full = fl.load_data(sys.argv[1])
out = sys.argv[2]
shutil.rmtree(out, ignore_errors=True)
rs = np.random.RandomState(0)
data = {}
for s, n in (("train", 1500), ("val", 300), ("test", 300)):
    idx = np.sort(rs.choice(len(full[s][1]), n, replace=False))
    data[s] = tuple(a[idx] for a in full[s])
fl.PRETRAINED = False
runner = fl.Runner(out, data, torch.device("cpu"), partial_dir=os.path.join(out, "partial"))
grid = [(0.5, 0.1), (0.8, 0.2)]
runner.run_architecture("cnn", [1e-3, 3e-3], grid, seeds=(0, 1), max_epochs=3, patience=2)
# resume check: a second call must reuse every finished run
n_before = len(os.listdir(os.path.join(out, "final")))
runner.run_architecture("cnn", [1e-3, 3e-3], grid, seeds=(0, 1), max_epochs=3, patience=2)
assert len(os.listdir(os.path.join(out, "final"))) == n_before
# ResNet path with random weights, tiny subset
small = {s: tuple(a[:96] for a in data[s]) for s in data}
r2 = fl.Runner(out, small, torch.device("cpu"))
r2.run_architecture("resnet18", [1e-4], [(0.5, 0.1)], seeds=(0,), max_epochs=1, patience=1, batch_size=32)
analyze.main(out)
print("E2E_OK", sorted(os.listdir(os.path.join(out, "analysis"))))
