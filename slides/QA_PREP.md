# Q&A preparation — STOR 566 Group 9 final presentation

Short answers first, detail after. Numbers are from the final results (Nov 17–18): all 18 runs of the
pre-specified protocol (3 models × standard / noise augmentation × 3 seeds), the 9 reference runs (3 per model)
and the learning-rate check. Nothing is pending.
Source of every number: `paper/generated/numbers.json`. Backup slides: 13 (selection scores and learning-rate
check), 14 (checkpoint-rule table), 19 (bootstrap intervals).

## Protocol and fairness

**Did anything on the test set influence your choices?**
No. Learning rate, stopping epoch and augmentation setting were chosen with a validation-only score, the chosen
settings were written to a file with a timestamp before the first test evaluation, and each final model was
evaluated on the test set once. Severity levels were fixed earlier with a pilot CNN on validation images.

**Why is the selection score ½ original + ½ noisy validation accuracy?**
The proposal says we select on original and noisy validation images. Equal weights is the simplest reading.
Reduced resolution is deliberately not in the score, so the resolution results measure transfer. We also ran
reference runs with the checkpoint picked on original validation accuracy only (slide 10, next question).

**What do the reference runs show?** (backup slide 14)
Same training, checkpoint picked the usual way (original validation accuracy only). For ResNet-18 and the CNN they
stop at the same epoch or later (ResNet-18: epochs 6, 2, 4 instead of 6, 1, 1; CNN: 32, 60, 58 instead of
28, 22, 32). On average ResNet-18 gains 2.5 points on original images (63.7% against 61.2%) and loses 4.1 under
noise (33.7% against 37.8%), with the same sign in all three seeds; the CNN gains 0.8 and loses 3.1 on average, but
not every seed agrees. For ViT-B/16, fine-tuned for four epochs, the two rules stop at most one epoch apart
(4, 3, 4 against 3, 2, 4) and differ by +0.2 points on original images and −0.8 under noise. So our rule shifts
standard training a few points toward noise robustness for the two smaller models and hardly matters for ViT-B/16.
The conclusions do not change: ResNet-18 is still no better than the CNN under noise (33.7% against 34.4%),
ViT-B/16 is still the most accurate model (69.5% original, 43.2% noise average, 45.7% resolution average), and
noise augmentation still adds 13.1 (CNN), 23.1 (ResNet-18) and 18.6 (ViT-B/16) points under noise.

**The chosen learning rate is the largest in your grid. Would a larger one have done better?** (backup slide 13)
We checked after everything was frozen: one value beyond the grid, seed 0, validation images only, never used for
selection or for any test number. ResNet-18 at 1×10⁻³ scored S = 0.484 against 0.491 at the chosen 3×10⁻⁴;
ViT-B/16 at 1×10⁻⁴ scored 0.560 against 0.556 at the chosen 5×10⁻⁵. Both differences (−0.007 and +0.004) are
smaller than the seed-to-seed SD of the score at the chosen value (0.008 and 0.005). One seed cannot rule out a
small gain for ViT-B/16, but there is no sign of a large one.

**Is a 94K-parameter CNN vs. an 11M / 86M pretrained network a fair comparison?**
It is a comparison of three methods, not a controlled test of one factor. The models differ in architecture,
size and pre-training at once, and we say so. What is matched: data, degraded test images, selection rule,
tuning budget (3 learning rates, 4 augmentation settings, seed 0), three seeds.

**Why only three seeds? Are the differences significant?**
Compute (free T4 GPUs on Colab and Kaggle; one ViT run is about 20 minutes plus evaluation, and the 27 final runs
took about 3.7 GPU-hours of training). We report the SD over seeds and,
separately, a bootstrap over test images with models fixed. Example: ResNet-18, augmentation minus standard:
original +1.8 [+0.8, +2.8], noise average +18.9 [+17.9, +20.0], resolution average +9.2 [+8.4, +9.9] points.
Large effects are many SDs wide; small ones (CNN on original images: +0.3 [−0.6, +1.3]) are not distinguishable
from zero. Between models under standard training: ResNet-18 minus CNN is +8.1 [+6.9, +9.3] on original images but
+0.3 [−0.7, +1.3] on the noise average; ViT-B/16 minus CNN is +16.1 [+14.7, +17.5] and +6.5 [+5.5, +7.5].

**Why these severity levels?**
A pilot CNN on validation images: we wanted five levels that run from "almost no loss" to "near the majority-class
rate". For the pilot, σ = 0.025 kept 95% of clean accuracy and σ = 0.2 kept 34%; 40 px kept 94% and 8 px 42%.
The midterm's heavier levels (σ ≥ 0.3, 4 px) are in the backup as an extended sweep.

**Why add noise at 48×48 and then resize to 224 for the pretrained models?**
The proposal requires that all models receive the same degraded image, generated before model-specific
preprocessing. A side effect is that pixel noise becomes blob-like at 224×224; that may be one reason the
pretrained models are sensitive to it (hypothesis, not tested).

## Results

**Why is your clean accuracy below the 70%+ numbers reported for FER-2013?**
Four reasons. (1) We removed exact duplicates, including 557 groups shared across train/val/test, so our test
set is cleaner and not comparable to the unfiltered benchmark. (2) By design there is no flip/crop/test-time
augmentation — noise is the only augmentation, so its effect can be isolated. (3) The baseline CNN is tiny on
purpose. (4) The labels are noisy: the dataset authors estimate human accuracy at 65 ± 5%, and the contest
winner reached 71.2% on the original test set. Our ViT-B/16 reaches 69.3 ± 1.1% under the same restrictions.

**Why does the pretrained ResNet-18 lose more under noise than the small CNN?**
We measured it, we did not explain it. Candidate reasons: fine-tuning drives training accuracy to 98% within a few
epochs, so the model leans on fine detail that noise destroys; and ImageNet features respond to texture, and
up-sampled noise looks like texture. It is not an artefact of the early checkpoints our rule picked: with
checkpoints picked on original validation accuracy only, ResNet-18 is less robust still (noise average 33.7%
against 37.8%; at σ = 0.1 it keeps 43% of its original accuracy instead of 54%). The same pattern — better clean
accuracy without better relative robustness — is what Hendrycks & Dietterich reported from AlexNet to ResNet on
ImageNet-C.

**Why does noise augmentation help at low resolution for ResNet-18 but hardly for the CNN?**
Consistent with the frequency argument of Yin et al.: training with Gaussian noise pushes a model away from
high-frequency detail, and down-sampling removes exactly that detail. The small CNN gains +1.4 points versus +9.2
for ResNet-18 and +3.9 for ViT-B/16. We did not test why; one possibility is capacity (three conv layers, receptive
field about 22 of 48 pixels).

**What is "prediction collapse"?**
At σ = 0.2 the standard ResNet-18 assigns on average 71% of all test images to a single class (CNN 61%, ViT-B/16
45%); the largest true class is 25.9%. Which class it is changes with the seed (happy, sad or neutral). So accuracy
near the majority rate is not random guessing; the model has stopped using the image. With augmentation the share
stays between 27% and 35% at σ = 0.2, but at 8 px every model collapses (44%, 72% and 86% with augmentation).

**Why can accuracy fall below the majority-class rate (8 px)?**
Because the collapse is not always onto the majority class. A model that answers "sad" for everything scores 17.5%.

**Why is macro-F1 so much lower than accuracy, and why is its SD large for standard ResNet-18?**
Disgust has 38 test images (1.2%) and recall between 10% and 30%, which pulls the class average down. Two of the
three standard ResNet-18 runs were checkpointed after one epoch, and their minority-class recall differs a lot.

**Does noise augmentation cost clean accuracy?**
Hardly: +0.3 (CNN), +1.8 (ResNet-18) and −0.7 (ViT-B/16) points on original images, paired by seed. The +1.8 for
ResNet-18 mostly reflects that our rule stopped the standard runs early: against checkpoints picked on original
validation accuracy only, the change is −0.5 for the CNN, −0.7 for ResNet-18 and −0.9 for ViT-B/16. For ViT-B/16
under our own rule the bootstrap interval just touches zero (−0.7 [−1.5, 0.0]). So the cost is at most about one
point.

**How does ViT-B/16 compare?**
Best on original images (69.3% against 61.2% and 53.2%; macro-F1 65.3). Under standard training it keeps part of
its lead on average (noise average 44.0 against 37.5 for the CNN; resolution average 45.8 against 38.0) but not at
the strongest levels: 23.0% at σ = 0.2 and 17.6% at 8 px, the lowest of the three at 8 px. It collapses too (45% of
images in one class at σ = 0.2, 58% at 8 px). Noise augmentation gives it +17.9 points under noise and +3.9 at
reduced resolution (68.5% original, 61.9% noise average, 49.8% resolution average), so with augmentation it is the
best model in every condition except 8 px. Its standard-training numbers do not depend on the checkpoint rule:
with checkpoints picked on original validation accuracy only it reaches 69.5% / 43.2% / 45.7%.

## Scope and limits

**Why only Gaussian noise and down-sampling?**
Both are controlled by one number, both are what the proposal fixed, and they stand for two real conditions (sensor
noise, small crops). Blur, compression, occlusion and lighting are the obvious next ones (ImageNet-C has 15 types;
Greco et al. built 18 for RAF-DB).

**What are the main limitations?**
One dataset with noisy labels; synthetic degradations applied one at a time; three seeds; architecture, size and
pre-training confounded; the selected learning rate sits at the upper edge of the grid for both pretrained models
(one step beyond it changed the score by less than its seed-to-seed SD); ViT-B/16 was fine-tuned for four epochs
only.

**What would you do next?**
More corruption types and mixed augmentation; FER+ label distributions to separate label noise from model error;
a wider learning-rate grid and more seeds; compare models at matched clean accuracy.

**How does this relate to prior work?**
Robustness benchmarks exist for object recognition (ImageNet-C) and for a larger face dataset (RAF-DB-C). We add a
pre-specified protocol at FER-2013's native 48×48 resolution, a from-scratch baseline next to two pretrained
models, a test of transfer from noise augmentation to an unseen degradation, and an explicit check of how the
checkpoint rule changes the answer.

## Process

**Were all runs done on the same machine?**
Same code, same data file (checked by a content fingerprint in every run), same frozen settings, a single T4 GPU
with mixed precision throughout. The tuning runs, the 18 protocol runs and the CNN and ResNet-18 reference runs
ran on Google Colab (PyTorch 2.14); the ViT-B/16 reference runs and the learning-rate check ran on Kaggle
(PyTorch 2.11) after the free Colab quota ran out. So the ViT-B/16 comparison of the two checkpoint rules is
across two library versions; its differences are under one point either way, and GPU training is not
bit-reproducible even on one platform. The paper states this in Appendix A.
