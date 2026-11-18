"""Smoke test for the ViT path, tuning-state reuse and the three-model analysis (CPU, random weights, tiny subset)."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import ferlib as fl, analyze
torch.set_num_threads(2)
full = fl.load_data(sys.argv[1]); out = sys.argv[2]; shutil.rmtree(out, ignore_errors=True)
fl.PRETRAINED = False
data = {s: tuple(a[:n] for a in full[s]) for s, n in (("train", 400), ("val", 100), ("test", 100))}
dev = torch.device("cpu")
r = fl.Runner(out, data, dev, partial_dir=os.path.join(out, "partial"), reuse_dir=os.path.join(out, "reuse"))
r.run_architecture("cnn", [1e-3], [(0.5, 0.1)], seeds=(0, 1), max_epochs=2, patience=2)
small = {s: tuple(a[:48] for a in data[s]) for s in data}
r2 = fl.Runner(out, small, dev, partial_dir=os.path.join(out, "partial"), reuse_dir=os.path.join(out, "reuse"))
r2.run_architecture("resnet18", [1e-4], [(0.5, 0.1)], seeds=(0, 1), max_epochs=1, patience=1, batch_size=16)
st = dict(fl.ARCH_SETTINGS["vit_b16"]); st.update(lr_grid=[2e-5], aug_grid=[(0.5, 0.2)], max_epochs=2, patience=2, batch_size=16)
sel, res = r2.run_architecture("vit_b16", seeds=(0, 1), **st)
finals = {k: json.load(open(os.path.join(out, "final", k))) for k in os.listdir(os.path.join(out, "final")) if k.startswith("vit") and k.endswith(".json")}
print("vit finals:", {k: (v.get("reused_tuning_run", False), v["config"]["optimizer"], v["config"]["schedule"], v["epochs_run"]) for k, v in finals.items()})
assert sum(v.get("reused_tuning_run", False) for v in finals.values()) == 2, "seed-0 std and aug should be reused"
assert not any(f.startswith("vit") for f in os.listdir(os.path.join(out, "ckpt")))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()) as buf:
    analyze.main(out)
txt = buf.getvalue()
assert "ViT-B/16" in txt and "failed" not in txt, txt[-600:]
print("E2E_VIT_OK", sorted(os.listdir(os.path.join(out, "analysis"))))
