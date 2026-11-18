"""Check that an interrupted run resumed from its checkpoint gives exactly the uninterrupted result."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import ferlib as fl
torch.set_num_threads(1)
full = fl.load_data(sys.argv[1])
data = {s: tuple(a[:n] for a in full[s]) for s, n in (("train", 600), ("val", 200), ("test", 200))}
dev = torch.device("cpu"); sel = fl.selection_sets(data); quiet = lambda m: None
ck = "/tmp/resume_test.pt"
if os.path.exists(ck): os.remove(ck)
full_cfg = fl.TrainConfig(arch="cnn", lr=3e-3, seed=3, aug_p=0.5, aug_sigma_max=0.2, max_epochs=4, patience=9)
s_ref, i_ref = fl.train_model(full_cfg, data, dev, sel, quiet)
part = fl.TrainConfig(arch="cnn", lr=3e-3, seed=3, aug_p=0.5, aug_sigma_max=0.2, max_epochs=2, patience=9)
fl.train_model(part, data, dev, sel, quiet, ckpt_path=ck)             # "interrupted" after 2 epochs
s_res, i_res = fl.train_model(full_cfg, data, dev, sel, print, ckpt_path=ck)   # continues to epoch 4
same_hist = all(abs(a["train_loss"] - b["train_loss"]) < 1e-9 and a["score"] == b["score"] for a, b in zip(i_ref["history"], i_res["history"]))
same_w = all(torch.equal(s_ref[k], s_res[k]) for k in s_ref)
print("epochs", i_ref["epochs_run"], i_res["epochs_run"], "| identical history:", same_hist, "| identical best weights:", same_w)
assert same_hist and same_w and i_res["epochs_run"] == 4
print("RESUME_OK")
