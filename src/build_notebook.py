"""Build the Colab/Kaggle notebook. ferlib.py and analyze.py are embedded verbatim, so the notebook
runs exactly the code tested in the workspace.
Usage: python3 -I build_notebook.py <output.ipynb>
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
ferlib_src = open(os.path.join(here, "ferlib.py")).read()
analyze_src = open(os.path.join(here, "analyze.py")).read()

cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(keepends=True)))


def code(text):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=text.strip("\n").splitlines(keepends=True)))


md("""
# STOR 566 Group 9 — Facial-Expression Classification under Image Degradation

Runs the full experiment on FER-2013 under one protocol for three models:

- **Models:** baseline CNN trained from scratch (93,575 parameters), ImageNet-pretrained ResNet-18, and
  ImageNet-pretrained ViT-B/16 (`google/vit-base-patch16-224`); the two pretrained models are fine-tuned end to end.
- **Training:** standard, and with Gaussian-noise augmentation → 6 configurations × 3 seeds = **18 final runs**,
  plus 9 reference runs (standard training with the checkpoint chosen on clean validation accuracy only).
- **Evaluation:** original images, 5 Gaussian-noise levels and 5 reduced-resolution levels (plus an extended sweep),
  reporting accuracy, macro-F1, per-class recall and confusion matrices on validation and test.

**How to run (Colab):** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. Approve the Google Drive
prompt once. Nothing else needs input. On a T4 the CNN stage takes about 20 minutes, ResNet-18 about an hour and
ViT-B/16 several hours.

**Safe to interrupt.** Every finished run is saved as one JSON file in Drive. If the session disconnects, open the
notebook again and press *Run all*: finished runs are skipped and the notebook continues where it stopped.

**Kaggle:** set *Accelerator = GPU T4* and *Internet = On*, then *Save Version → Save & Run All*. Results go to
`/kaggle/working/stor566_group9_fer/`.

**Protocol integrity.** Learning rate, stopping epoch and augmentation settings are chosen with validation images only
(original + noisy; never reduced resolution). The choice is written to `selection_<arch>.json` before any final run,
and only final runs are evaluated on the test split. Grids and budgets per model are in `ferlib.ARCH_SETTINGS`.
""")

code("""
# ---- settings (the defaults reproduce the protocol; edit only if you know why) ----
ARCHS      = ["cnn", "resnet18", "vit_b16"]
SEEDS      = (0, 1, 2)
CSV_PATH   = None   # set to a local fer2013.csv / icml_face_data.csv only if the automatic download fails
REQUIRE_DRIVE = True   # Colab: stop if Google Drive is not connected, so results always survive a disconnect
""")

code("""
import os, sys, time, json, glob, shutil, tarfile, tempfile, zipfile, importlib

IN_COLAB = "google.colab" in sys.modules
OUT_DIR = None
if IN_COLAB:
    from google.colab import drive
    for attempt in range(1, 21):
        try:
            drive.mount("/content/drive")
            OUT_DIR = "/content/drive/MyDrive/stor566_group9_fer"
            break
        except Exception as e:
            print(f"[{attempt}/20] Google Drive is not connected yet ({type(e).__name__}). Click 'Connect to Google Drive' "
                  "in the dialog and finish the Google popup.")
            time.sleep(5)
    if OUT_DIR is None and REQUIRE_DRIVE:
        raise RuntimeError("Google Drive was not connected, so results would be lost if the session disconnects. "
                           "Press Run all again and approve the Drive prompt (or set REQUIRE_DRIVE = False).")
if OUT_DIR is None:
    OUT_DIR = "/kaggle/working/stor566_group9_fer" if os.path.isdir("/kaggle/working") else os.path.abspath("stor566_group9_fer")
os.makedirs(OUT_DIR, exist_ok=True)
PARTIAL_DIR = os.path.join(tempfile.gettempdir(), "fer566_partial")   # per-epoch training state, local disk
REUSE_DIR = os.path.join(tempfile.gettempdir(), "fer566_tune_states")  # ViT tuning weights, reused for seed 0

import torch
print("torch", torch.__version__, "| results folder:", OUT_DIR)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU found. Colab: Runtime -> Change runtime type -> T4 GPU, then Run all again.")
""")

code("%%writefile ferlib.py\n" + ferlib_src)
code("%%writefile analyze.py\n" + analyze_src)

code("""
# ---- data: download the original labelled CSV, apply the proposal's duplicate filtering, verify ----
import numpy as np
import ferlib as fl
importlib.reload(fl)

NPZ = os.path.join(OUT_DIR, "fer_dedup.npz")
if os.path.exists(NPZ):
    data = fl.load_data(NPZ)
else:
    csv_path = CSV_PATH
    if csv_path is None:
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
    print(json.dumps(audit, indent=1))
    json.dump(audit, open(os.path.join(OUT_DIR, "data_audit.json"), "w"), indent=1)
    fl.save_data(data, NPZ)

sizes = {s: int(len(data[s][1])) for s in data}
for s in ("train", "val", "test"):
    print(f"{s:5s} n={sizes[s]:6d}  " + "  ".join(f"{n}={v}" for n, v in zip(fl.FER_LABELS, fl.class_counts(data[s][1]))))
assert sizes == fl.EXPECTED_SIZES, f"split sizes {sizes} differ from the proposal {fl.EXPECTED_SIZES}"
assert fl.data_fingerprint(data) == fl.EXPECTED_FINGERPRINT, "images/labels differ from the audited copy"
print("data verified: identical images and labels to the audited copy;",
      "identical row order:", fl.row_order_fingerprint(data) == fl.EXPECTED_ROW_ORDER_FINGERPRINT)
""")

code("""
# ---- experiments (resumable) ----
import analyze
importlib.reload(analyze)

device = torch.device("cuda")

# preflight: one training step per architecture on the GPU, so a configuration problem shows up in the first minute
for arch in ARCHS:
    m = fl.build_model(arch).to(device)
    xb = fl.to_float(data["train"][0][:32]).to(device)
    yb = torch.from_numpy(data["train"][1][:32]).to(device)
    with torch.autocast("cuda"):
        loss = torch.nn.functional.cross_entropy(m(xb), yb)
    loss.backward()
    print(f"preflight {arch}: ok ({fl.count_params(m):,} parameters, loss {loss.item():.3f})")
    del m, xb, yb, loss
torch.cuda.empty_cache()

runner = fl.Runner(OUT_DIR, data, device, partial_dir=PARTIAL_DIR, reuse_dir=REUSE_DIR)
t_start = time.time()
for arch in ARCHS:
    sel, results = runner.run_architecture(arch, seeds=SEEDS, **fl.ARCH_SETTINGS[arch])
    runner.log(f"[{arch}] done: lr={sel['lr']:g}, aug p={sel['aug_p']:g}, sigma_max={sel['aug_sigma_max']:g}; "
               f"elapsed {(time.time() - t_start) / 60:.0f} min")
    analyze.main(OUT_DIR)          # refresh tables and figures after each architecture
""")

code("""
# ---- reference runs: standard training, checkpoint chosen on clean validation accuracy only (resumable) ----
for arch in ARCHS:
    runner.run_reference(arch, seeds=SEEDS, **fl.ARCH_SETTINGS[arch])
    runner.log(f"[{arch}] reference runs done; elapsed {(time.time() - t_start) / 60:.0f} min")
""")

code("""
# ---- results: summary tables and figures ----
from IPython.display import Image, Markdown, display

analyze.main(OUT_DIR)
display(Markdown(open(os.path.join(OUT_DIR, "analysis", "results_summary.md")).read()))
for name in ("fig_degradation_acc", "fig_relative_acc", "fig_by_architecture", "fig_augmentation_effect",
             "fig_per_class_recall", "fig_confusion_clean_std", "fig_learning_curves"):
    p = os.path.join(OUT_DIR, "analysis", name + ".png")
    if os.path.exists(p):
        display(Image(filename=p, width=900))
""")

code("""
# ---- bundle everything needed for the slides and the report into one zip ----
bundle = os.path.join(OUT_DIR, "results_bundle.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ("final/*.json", "final/*_preds.npz", "tune/*.json", "selection_*.json", "analysis/*",
                    "data_audit.json", "log.txt"):
        for p in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
            z.write(p, os.path.relpath(p, OUT_DIR))
print(f"bundle: {bundle} ({os.path.getsize(bundle) / 1e6:.1f} MB)")
n_final = len(glob.glob(os.path.join(OUT_DIR, "final", "*.json")))
print(f"final runs finished: {n_final} / {len(ARCHS) * 3 * len(SEEDS)}")
if IN_COLAB:
    try:
        from google.colab import files
        files.download(bundle)
    except Exception as e:
        print("automatic download skipped:", repr(e), "- the zip is in your Drive folder")
""")

nb = dict(nbformat=4, nbformat_minor=5,
          metadata=dict(accelerator="GPU", colab=dict(gpuType="T4", provenance=[]),
                        kernelspec=dict(name="python3", display_name="Python 3"),
                        language_info=dict(name="python")),
          cells=cells)
text = json.dumps(nb, indent=1, ensure_ascii=False)
open(sys.argv[1], "w", encoding="utf-8").write(text)
print("wrote", sys.argv[1], f"{len(text) / 1024:.0f} KB, {len(cells)} cells")
