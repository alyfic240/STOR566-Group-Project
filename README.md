# Facial-Expression Classification under Image Degradation

STOR 566 (Introduction to Deep Learning, Fall 2026, UNC-Chapel Hill) · final project, Group 9
Dominic Jiang · Alyson Ficca · Emily Nguyen · Isaiah Lu

Facial-expression classifiers are usually compared on clean test images, but faces from cheap cameras or distant
crops are noisy and small. We compare three models on FER-2013 (a small CNN trained from scratch, an
ImageNet-pretrained ResNet-18 and an ImageNet-pretrained ViT-B/16), test them under Gaussian noise and reduced
resolution at five severity levels each, and ask whether adding Gaussian noise during training helps.

**Status (Nov 24, 2026).** All experiments are finished and `paper/main.pdf` is the final report. The presentation is on Dec 1.

## Results

Test accuracy (%), mean ± standard deviation over three seeds, on the duplicate-filtered test set
(3,268 images, majority-class rate 25.9%). "Noise avg." is the mean over
σ ∈ {0.025, 0.05, 0.10, 0.15, 0.20}; "Resolution avg." is the mean over 40, 32, 24, 16 and 8 px.

| Model | Training | Original | Noise avg. | Resolution avg. |
|---|---|---|---|---|
| Baseline CNN (94K parameters, from scratch) | standard | 53.2 ± 1.3 | 37.5 ± 0.4 | 38.0 ± 0.4 |
|  | noise augmentation | 53.5 ± 0.4 | 47.5 ± 0.5 | 39.4 ± 0.3 |
|  | standard, conventional checkpoint | 53.9 ± 0.5 | 34.4 ± 3.6 | 38.7 ± 2.1 |
| ResNet-18 (ImageNet-pretrained) | standard | 61.2 ± 1.4 | 37.8 ± 2.5 | 38.6 ± 0.9 |
|  | noise augmentation | 63.1 ± 1.1 | 56.7 ± 0.4 | 47.7 ± 2.3 |
|  | standard, conventional checkpoint | 63.7 ± 0.3 | 33.7 ± 2.8 | 39.7 ± 1.2 |
| ViT-B/16 (ImageNet-pretrained) | standard | 69.3 ± 1.1 | 44.0 ± 0.3 | 45.8 ± 0.9 |
|  | noise augmentation | 68.5 ± 0.5 | 61.9 ± 0.1 | 49.8 ± 0.7 |
|  | standard, conventional checkpoint | 69.5 ± 0.4 | 43.2 ± 0.3 | 45.7 ± 0.4 |

"Standard" and "noise augmentation" follow the protocol fixed in the proposal: the checkpoint is chosen by the
mean of accuracy on original and on noisy validation images. "Conventional checkpoint" repeats standard training
with the checkpoint chosen on original validation accuracy only; these reference runs measure what the selection
rule itself changes.

1. On original images the larger pretrained models are more accurate (53.2%,
   61.2% and 69.3%).
2. Under standard training that advantage shrinks or vanishes as images degrade. Averaged over the noise levels
   ResNet-18 is only 0.3 points better than the CNN, and at the strongest levels every model falls to
   or below the majority-class rate because its predictions collapse onto one class.
3. Noise augmentation costs little on original images, recovers 10.0 to 18.9 points
   under noise, and transfers to reduced resolution (never seen in training) for the pretrained models
   (+9.2 and +3.9 points) but hardly for the small CNN (+1.4).
4. The rule used to pick checkpoints changes the numbers for standard training and should be reported.

Every table, figure and number in the paper and on the slides is generated from the files in `results/`.

## What is in this repository

| Path | Content |
|---|---|
| `src/` | The pipeline (`ferlib.py`: duplicate filtering, models, degradations, training, selection rule, evaluation, resumable runner) and the analysis, validation and notebook-building scripts |
| `notebooks/` | The two GPU notebooks that produced the results (Colab and Kaggle); `ferlib.py` is embedded in them verbatim |
| `results/` | One JSON file per run (`final/`, `tune/`, `posthoc/`), the saved test and validation predictions of every final run (`final/*_preds.npz`), the frozen selections (`selection_*.json`), the data audit, logs, and the generated tables and figures (`analysis/`, `analysis_extra/`) |
| `paper/` | NeurIPS-format paper: LaTeX source, generated tables and number macros, figures, and the current PDF |
| `slides/` | Source files of the final presentation, slide figures, speaker notes and Q&A preparation |
| `stor_566_proj_setup.ipynb`, `vit_FER2013_setup.ipynb` | The group's earlier notebooks (see the last section) |

## Reproducing

**Data.** FER-2013 is not redistributed here. The notebooks download the original labelled CSV, apply the
duplicate filtering and stop unless the result has exactly 27,418 / 3,291 / 3,268 training / validation / test
images and the expected content fingerprint (recorded in every run file as `data_fingerprint`). To build the
same splits from a local copy of `fer2013.csv` or `icml_face_data.csv`:

```bash
python3 src/prep_data.py path/to/fer2013.csv data/fer_dedup.npz
```

**Experiments.** Open `notebooks/STOR566_Group9_FER_experiments.ipynb` on Google Colab or Kaggle with a T4 GPU and
run all cells. It tunes on validation images, freezes the selection, trains the 18 protocol runs and the
9 reference runs and evaluates them. Finished runs are skipped, so an interrupted session can be continued.
`notebooks/STOR566_Group9_FER_remaining_runs_kaggle.ipynb` is the notebook used for the runs that were done on
Kaggle (the ViT-B/16 reference runs and the learning-rate check). Training the 27 final runs took about
3.7 GPU-hours on a T4, not counting tuning.

**Checks, tables, paper and slides** (CPU only; needs `numpy`, `torch`, `matplotlib` and a LaTeX installation):

```bash
python3 src/validate_runs.py results data/fer_dedup.npz      # recompute every metric from the saved predictions
python3 paper/make_assets.py results data/fer_dedup.npz      # tables, figures and number macros for the paper
python3 paper/check_claims.py results paper                  # re-derive the statements the paper makes in words
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ..
python3 src/slide_figs.py results slides/figures             # figures used on the slides
python3 slides/build_deck.py results                         # slide files (text and numbers) in slides/deck_root/
```

## Protocol in brief

- **Data.** All copies of pixel-identical images with conflicting labels are removed; other duplicates keep one
  copy (priority training > validation > test), so no test image also appears in training.
- **Models.** CNN: three 3×3 convolution blocks (32, 64, 128 channels), global average pooling, linear layer
  (93,575 parameters). ResNet-18 and ViT-B/16 (`google/vit-base-patch16-224`): ImageNet-pretrained,
  seven-class head, fine-tuned end to end on the 48×48 grayscale images resized to 224×224.
- **Degradations**, applied to the 48×48 image before any model-specific preprocessing and identical for every
  model and seed: Gaussian pixel noise (σ on a 0–1 intensity scale) and reduced resolution (downsample to s×s,
  upsample back).
- **Selection** uses validation images only: one score (mean of accuracy on original and on noisy validation
  images) picks the learning rate, the stopping epoch and the augmentation setting. Reduced-resolution images
  never enter selection. The chosen settings are written to `results/selection_*.json` before any test evaluation.
- **Final runs.** Three seeds per configuration; the test set is evaluated once per run.
- Details that matter for reading the numbers (which runs ran on which platform and library version, the
  learning-rate check, which seed-0 runs are the selected tuning runs) are in Appendix A of the paper.

## The earlier notebooks

`stor_566_proj_setup.ipynb` (data preparation, duplicate filtering and the baseline CNN, used for the midterm
review) and `vit_FER2013_setup.ipynb` (first ViT-B/16 experiment after the midterm feedback) are kept unchanged.
The final pipeline builds on them: it reproduces the same split sizes and the same CNN architecture, and the
ViT-B/16 training recipe (AdamW, weight decay 0.01, warm-up and linear decay, batch 32) follows the ViT notebook.
Numbers differ from those notebooks because the final protocol uses one set of severity levels for all models,
selects everything on validation data, and trains three seeds per configuration.
