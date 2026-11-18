"""Build the "remaining runs" Colab notebook (v5): no Google Drive mount.

Why: the Drive mount needs a Google sign-in popup on every new runtime. This notebook needs no input at all:
  * ferlib.py is embedded verbatim (the same file as in the v4 notebook),
  * the frozen selection files are embedded, so no tuning is repeated and nothing is re-selected,
  * results are written to the runtime's local disk and also printed as base64 blocks, which Colab saves
    inside the notebook file; collect_from_notebook.py turns those blocks back into the result files.

It runs only what the v4 notebook had not finished:
  * ViT-B/16 reference runs, seeds 0-2 (standard training, checkpoint chosen on original validation accuracy),
  * the post-hoc learning-rate check (one value above the grid for ResNet-18 and ViT-B/16; validation only).

Usage: python3 -I build_notebook_v5_remaining.py <output.ipynb> <results_dir with selection_*.json> [item ...]
Items (default: all, in this order): vit_ref_0 vit_ref_1 vit_ref_2 lr_resnet18 lr_vit_b16
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
ferlib_src = open(os.path.join(here, "ferlib.py")).read()
out_path, results_dir = sys.argv[1], sys.argv[2]
ALL_ITEMS = ["vit_ref_0", "vit_ref_1", "vit_ref_2", "lr_resnet18", "lr_vit_b16"]
items = sys.argv[3:] or ALL_ITEMS
assert all(i in ALL_ITEMS for i in items), items
selections = {a: json.load(open(os.path.join(results_dir, f"selection_{a}.json"))) for a in ("cnn", "resnet18", "vit_b16")}

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(keepends=True)))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=text.strip("\n").splitlines(keepends=True)))


md("""
# STOR 566 Group 9 — remaining runs (v5, no Google Drive prompt)

Companion to `STOR566_Group9_FER_experiments_v4.ipynb`. The 18 runs of the pre-specified protocol and the CNN and
ResNet-18 reference runs are already finished and stored in Drive. This notebook runs only what was left:

- **ViT-B/16 reference runs**, seeds 0–2: standard training at the selected learning rate, checkpoint chosen on
  original validation accuracy only.
- **Post-hoc learning-rate check**: one value above the grid for ResNet-18 and ViT-B/16, seed 0, validation images
  only. It is reported as a limitation check and never changes the frozen selection.

**Kaggle (runs in the background, the page can be closed):** *Settings → Accelerator: GPU T4 x2*, *Internet: On*, then
*Save Version → Save & Run All (Commit)*. About 1 h 45 min. When the version has finished, open its **Output** tab and
download `group9_remaining_results.zip`.

**Colab:** `Runtime → Run all` on a T4 GPU. Google Drive is not mounted; each finished item prints its result files
as base64 blocks (`@@FILE … @@END`), which are saved with the notebook and unpacked with `collect_from_notebook.py`.

Nothing needs input on either platform. `ferlib.py` is the same file as in the v4 notebook, and the selection files
frozen by the v4 run are embedded below, so no tuning is repeated and nothing is re-selected.
""")

code(f"""
# ---- which items to run (finished items are skipped if this runtime already has their files) ----
RUN = {items!r}
""")

code(f"""
import os, sys, time, json, glob, shutil, tarfile, tempfile, base64, hashlib, importlib

ON_KAGGLE = os.path.isdir("/kaggle/working")
# results folder on the runtime's own disk (Kaggle keeps /kaggle/working as the version's output; Drive is not used)
OUT_DIR = "/kaggle/working/stor566_group9_fer" if ON_KAGGLE else "/content/stor566_group9_fer"
PRINT_FILES = not ON_KAGGLE    # Colab: also print result files as base64 blocks, which are saved with the notebook
os.makedirs(OUT_DIR, exist_ok=True)
PARTIAL_DIR = os.path.join(tempfile.gettempdir(), "fer566_partial")

import torch
print("torch", torch.__version__, "| platform:", "Kaggle" if ON_KAGGLE else "Colab", "| results folder:", OUT_DIR)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU found. Runtime -> Change runtime type -> T4 GPU, then Run all again.")

# selection files frozen by the v4 run (written before any final run, copied here unchanged)
SELECTIONS = json.loads({json.dumps(json.dumps(selections))})
for arch, sel in SELECTIONS.items():
    json.dump(sel, open(os.path.join(OUT_DIR, f"selection_{{arch}}.json"), "w"), indent=1)
    print(f"selection_{{arch}}.json: lr={{sel['lr']:g}}, aug p={{sel['aug_p']:g}}, sigma_max={{sel['aug_sigma_max']:g}}, frozen {{sel['frozen_at']}}")
""")

code("%%writefile ferlib.py\n" + ferlib_src)

code("""
# ---- data: download the original labelled CSV, apply the proposal's duplicate filtering, verify ----
import numpy as np
import ferlib as fl
importlib.reload(fl)

NPZ = os.path.join(tempfile.gettempdir(), "fer_dedup.npz")   # not in the results folder: it is not a result
if os.path.exists(NPZ):
    data = fl.load_data(NPZ)
else:
    from huggingface_hub import hf_hub_download
    try:
        tgz = hf_hub_download("DerrickUnleashed/FER-2013", "fer2013.tar.gz", repo_type="dataset")
        csv_path = os.path.join(tempfile.gettempdir(), "fer2013.csv")
        with tarfile.open(tgz) as tf:
            member = next(m for m in tf.getmembers() if m.isfile() and m.name.endswith("fer2013.csv"))
            with tf.extractfile(member) as src, open(csv_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as e:
        print("fer2013.tar.gz route failed:", repr(e), "-> using icml_face_data.csv (same images)")
        csv_path = hf_hub_download("DerrickUnleashed/FER-2013", "icml_face_data.csv", repo_type="dataset")
    X, y, usage = fl.read_fer_csv(csv_path)
    data, audit = fl.audit_and_dedup(X, y, usage)
    fl.save_data(data, NPZ)

sizes = {s: int(len(data[s][1])) for s in data}
assert sizes == fl.EXPECTED_SIZES, f"split sizes {sizes} differ from the proposal {fl.EXPECTED_SIZES}"
assert fl.data_fingerprint(data) == fl.EXPECTED_FINGERPRINT, "images/labels differ from the audited copy"
print("data verified: identical images and labels to the audited copy;",
      "identical row order:", fl.row_order_fingerprint(data) == fl.EXPECTED_ROW_ORDER_FINGERPRINT, "| sizes", sizes)
""")

code("""
# ---- helpers ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
runner = fl.Runner(OUT_DIR, data, device, partial_dir=PARTIAL_DIR)


def emit(*paths):
    \"\"\"Print files as base64 blocks so that they are stored in the notebook itself (Colab only).\"\"\"
    for p in paths if PRINT_FILES else ():
        b = open(p, "rb").read()
        rel = os.path.relpath(p, OUT_DIR)
        s = base64.b64encode(b).decode()
        lines = [f"@@FILE {rel} {len(b)} {hashlib.sha256(b).hexdigest()}"]
        lines += [s[i:i + 200] for i in range(0, len(s), 200)]
        lines.append(f"@@END {rel}")
        print("\\n".join(lines), flush=True)


def reference(arch, seed):
    \"\"\"Reference run: standard training at the selected learning rate, checkpoint chosen on original
    validation accuracy only (ferlib.Runner.run_reference, exactly as in the v4 notebook).\"\"\"
    t0 = time.time()
    res = runner.run_reference(arch, seeds=(seed,), **fl.ARCH_SETTINGS[arch])[0]
    assert res["variant"] == "ref" and res["config"]["seed"] == seed and res["config"]["select_on"] == "clean"
    runner.log(f"[{arch}] reference run seed {seed} done; {(time.time() - t0) / 60:.0f} min")
    emit(os.path.join(OUT_DIR, "final", res["name"] + ".json"), os.path.join(OUT_DIR, "final", res["name"] + "_preds.npz"))
    torch.cuda.empty_cache()


def lr_check(arch, lr):
    \"\"\"Post-hoc check, same code path as v5_run_lr_check.py: one standard-training tuning run (seed 0) at a
    learning rate one step above the grid. Validation images only; never used for selection.\"\"\"
    post = os.path.join(OUT_DIR, "posthoc")
    os.makedirs(post, exist_ok=True)
    prun = fl.Runner(post, data, device, partial_dir=os.path.join(tempfile.gettempdir(), "fer566_partial_posthoc"))
    s = dict(fl.ARCH_SETTINGS[arch])
    sel = json.load(open(os.path.join(OUT_DIR, f"selection_{arch}.json")))
    assert lr > max(float(k) for k in sel["lr_scores"]), "the check is meant for a value above the grid"
    base = dict(arch=arch, batch_size=s["batch_size"], max_epochs=s["max_epochs"], patience=s["patience"],
                **(s.get("extra") or {}))
    t0 = time.time()
    info = prun.tune_run(fl.TrainConfig(lr=lr, seed=0, **base))
    best = info["history"][info["best_epoch"] - 1]
    path = os.path.join(post, "lr_check_summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    summary[arch] = dict(lr=lr, score=info["best_score"], best_epoch=info["best_epoch"],
                         val_acc_clean=best["val_acc_clean"], val_acc_noisy=best["val_acc_noisy"],
                         grid_scores=sel["lr_scores"], selected_lr=sel["lr"],
                         selected_score=sel["lr_scores"][f"{sel['lr']:g}"])
    json.dump(summary, open(path, "w"), indent=1)
    prun.log(f"[posthoc] {arch}: lr {lr:g} -> S = {info['best_score']:.4f} at epoch {info['best_epoch']} "
             f"(clean {best['val_acc_clean']:.4f}, noisy {best['val_acc_noisy']:.4f}); grid: {sel['lr_scores']}, "
             f"selected {sel['lr']:g}; {(time.time() - t0) / 60:.0f} min")
    emit(path, os.path.join(post, "tune", info["name"] + ".json"))
    torch.cuda.empty_cache()


print("ready; items to run:", RUN)
""")

for seed in (0, 1, 2):
    code(f"""
# ---- ViT-B/16 reference run, seed {seed} ----
if "vit_ref_{seed}" in RUN:
    reference("vit_b16", {seed})
else:
    print("skipped")
""")

code("""
# ---- post-hoc learning-rate check: ResNet-18 at 1e-3 (grid: 3e-5, 1e-4, 3e-4) ----
if "lr_resnet18" in RUN:
    lr_check("resnet18", 1e-3)
else:
    print("skipped")
""")

code("""
# ---- post-hoc learning-rate check: ViT-B/16 at 1e-4 (grid: 1e-5, 2e-5, 5e-5) ----
if "lr_vit_b16" in RUN:
    lr_check("vit_b16", 1e-4)
else:
    print("skipped")
""")

code("""
# ---- logs and one zip with every result file of this notebook ----
import zipfile

emit(os.path.join(OUT_DIR, "log.txt"), *sorted(glob.glob(os.path.join(OUT_DIR, "posthoc", "log.txt"))))
bundle = os.path.join(os.path.dirname(OUT_DIR), "group9_remaining_results.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ("final/*.json", "final/*_preds.npz", "posthoc/*.json", "posthoc/tune/*.json", "posthoc/log.txt",
                    "selection_*.json", "log.txt"):
        for p in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
            z.write(p, os.path.relpath(p, OUT_DIR))
            print(hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], os.path.relpath(p, OUT_DIR))
print(f"bundle: {bundle} ({os.path.getsize(bundle) / 1e3:.0f} kB)")
print("ALL DONE", time.strftime("%Y-%m-%d %H:%M:%S"), "UTC")
""")

nb = dict(nbformat=4, nbformat_minor=5,
          metadata=dict(accelerator="GPU", colab=dict(gpuType="T4", provenance=[]),
                        kernelspec=dict(name="python3", display_name="Python 3"),
                        language_info=dict(name="python")),
          cells=cells)
text = json.dumps(nb, indent=1, ensure_ascii=False)
open(out_path, "w", encoding="utf-8").write(text)
print("wrote", out_path, f"{len(text) / 1024:.0f} KB, {len(cells)} cells; items: {items}")
