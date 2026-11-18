# Extra analyses (test split)

Final runs found: 27. Mean ± sample SD over seeds unless stated otherwise.

## 1. Headline numbers

Noise avg / Resolution avg = accuracy averaged over the five primary severity levels (σ ∈ [0.025, 0.05, 0.1, 0.15, 0.2]; sizes [40, 32, 24, 16, 8]).

| Model | Training | Clean acc | Noise avg acc | Resolution avg acc | Clean macro-F1 | Noise avg macro-F1 | Resolution avg macro-F1 |
|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 53.2 ± 1.3 | 37.5 ± 0.4 | 38.0 ± 0.4 | 45.6 ± 2.4 | 28.8 ± 2.1 | 28.6 ± 1.7 |
| Baseline CNN | noise augmentation | 53.5 ± 0.4 | 47.5 ± 0.5 | 39.4 ± 0.3 | 45.9 ± 1.1 | 39.4 ± 0.5 | 31.2 ± 0.3 |
| Baseline CNN | standard, clean-selected checkpoint | 53.9 ± 0.5 | 34.4 ± 3.6 | 38.7 ± 2.1 | 46.1 ± 1.6 | 26.5 ± 2.8 | 29.6 ± 3.2 |
| ResNet-18 (pretrained) | standard training | 61.2 ± 1.4 | 37.8 ± 2.5 | 38.6 ± 0.9 | 53.4 ± 5.1 | 25.3 ± 2.8 | 27.3 ± 1.5 |
| ResNet-18 (pretrained) | noise augmentation | 63.1 ± 1.1 | 56.7 ± 0.4 | 47.7 ± 2.3 | 57.8 ± 1.1 | 50.5 ± 1.0 | 39.3 ± 2.1 |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | 63.7 ± 0.3 | 33.7 ± 2.8 | 39.7 ± 1.2 | 58.9 ± 0.8 | 23.2 ± 3.2 | 30.5 ± 0.8 |
| ViT-B/16 (pretrained) | standard training | 69.3 ± 1.1 | 44.0 ± 0.3 | 45.8 ± 0.9 | 65.3 ± 2.0 | 37.8 ± 0.6 | 39.9 ± 0.9 |
| ViT-B/16 (pretrained) | noise augmentation | 68.5 ± 0.5 | 61.9 ± 0.1 | 49.8 ± 0.7 | 63.9 ± 0.5 | 56.3 ± 0.7 | 43.2 ± 0.4 |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | 69.5 ± 0.4 | 43.2 ± 0.3 | 45.7 ± 0.4 | 65.8 ± 0.5 | 37.4 ± 1.0 | 39.9 ± 0.7 |

## 2. Prediction collapse

True class shares in the test split: angry 13.8%, disgust 1.2%, fear 14.6%, happy 25.9%, sad 17.5%, surprise 8.8%, neutral 18.2%.

| Model | Training | Condition | Share assigned to most-predicted class | Most-predicted class by seed |
|---|---|---|---|---|
| Baseline CNN | standard training | clean | 30.4 ± 3.1 | happy, happy, happy |
| Baseline CNN | standard training | noise_0.025 | 26.3 ± 1.7 | happy, happy, happy |
| Baseline CNN | standard training | noise_0.05 | 24.9 ± 0.9 | happy, happy, happy |
| Baseline CNN | standard training | noise_0.1 | 38.1 ± 3.6 | neutral, neutral, neutral |
| Baseline CNN | standard training | noise_0.15 | 52.1 ± 13.3 | neutral, happy, neutral |
| Baseline CNN | standard training | noise_0.2 | 61.4 ± 20.6 | neutral, happy, happy |
| Baseline CNN | standard training | res_40 | 33.7 ± 7.0 | happy, happy, sad |
| Baseline CNN | standard training | res_32 | 38.2 ± 6.4 | sad, happy, sad |
| Baseline CNN | standard training | res_24 | 44.2 ± 14.3 | sad, happy, sad |
| Baseline CNN | standard training | res_16 | 45.0 ± 18.6 | sad, happy, happy |
| Baseline CNN | standard training | res_8 | 45.6 ± 7.4 | sad, happy, neutral |
| Baseline CNN | noise augmentation | clean | 27.3 ± 0.4 | happy, happy, happy |
| Baseline CNN | noise augmentation | noise_0.025 | 27.1 ± 0.5 | happy, happy, happy |
| Baseline CNN | noise augmentation | noise_0.05 | 27.6 ± 0.4 | happy, happy, happy |
| Baseline CNN | noise augmentation | noise_0.1 | 29.2 ± 0.5 | happy, happy, happy |
| Baseline CNN | noise augmentation | noise_0.15 | 31.4 ± 0.8 | happy, happy, happy |
| Baseline CNN | noise augmentation | noise_0.2 | 34.6 ± 0.6 | happy, happy, happy |
| Baseline CNN | noise augmentation | res_40 | 31.7 ± 2.9 | sad, sad, happy |
| Baseline CNN | noise augmentation | res_32 | 35.1 ± 5.0 | sad, sad, sad |
| Baseline CNN | noise augmentation | res_24 | 40.6 ± 6.8 | sad, sad, happy |
| Baseline CNN | noise augmentation | res_16 | 42.2 ± 7.4 | sad, sad, happy |
| Baseline CNN | noise augmentation | res_8 | 44.2 ± 15.0 | sad, sad, surprise |
| Baseline CNN | standard, clean-selected checkpoint | clean | 28.4 ± 3.1 | happy, happy, sad |
| Baseline CNN | standard, clean-selected checkpoint | noise_0.025 | 31.1 ± 1.3 | happy, sad, sad |
| Baseline CNN | standard, clean-selected checkpoint | noise_0.05 | 30.0 ± 2.4 | happy, neutral, neutral |
| Baseline CNN | standard, clean-selected checkpoint | noise_0.1 | 52.0 ± 8.2 | neutral, neutral, neutral |
| Baseline CNN | standard, clean-selected checkpoint | noise_0.15 | 60.3 ± 5.1 | neutral, neutral, sad |
| Baseline CNN | standard, clean-selected checkpoint | noise_0.2 | 73.2 ± 13.9 | neutral, neutral, sad |
| Baseline CNN | standard, clean-selected checkpoint | res_40 | 32.9 ± 2.2 | happy, sad, sad |
| Baseline CNN | standard, clean-selected checkpoint | res_32 | 35.7 ± 3.2 | happy, sad, sad |
| Baseline CNN | standard, clean-selected checkpoint | res_24 | 36.2 ± 3.8 | happy, sad, sad |
| Baseline CNN | standard, clean-selected checkpoint | res_16 | 40.3 ± 4.8 | happy, happy, happy |
| Baseline CNN | standard, clean-selected checkpoint | res_8 | 42.9 ± 10.3 | angry, happy, neutral |
| ResNet-18 (pretrained) | standard training | clean | 30.4 ± 2.9 | happy, happy, happy |
| ResNet-18 (pretrained) | standard training | noise_0.025 | 31.2 ± 2.9 | sad, happy, sad |
| ResNet-18 (pretrained) | standard training | noise_0.05 | 37.9 ± 10.0 | sad, happy, happy |
| ResNet-18 (pretrained) | standard training | noise_0.1 | 48.2 ± 13.2 | sad, happy, happy |
| ResNet-18 (pretrained) | standard training | noise_0.15 | 59.4 ± 13.8 | happy, happy, sad |
| ResNet-18 (pretrained) | standard training | noise_0.2 | 71.3 ± 9.9 | happy, happy, sad |
| ResNet-18 (pretrained) | standard training | res_40 | 34.8 ± 7.4 | happy, happy, happy |
| ResNet-18 (pretrained) | standard training | res_32 | 38.6 ± 10.8 | happy, happy, happy |
| ResNet-18 (pretrained) | standard training | res_24 | 53.3 ± 13.5 | happy, happy, happy |
| ResNet-18 (pretrained) | standard training | res_16 | 51.7 ± 12.7 | sad, sad, happy |
| ResNet-18 (pretrained) | standard training | res_8 | 78.7 ± 18.1 | neutral, sad, happy |
| ResNet-18 (pretrained) | noise augmentation | clean | 27.7 ± 2.1 | sad, happy, happy |
| ResNet-18 (pretrained) | noise augmentation | noise_0.025 | 27.4 ± 1.5 | sad, neutral, happy |
| ResNet-18 (pretrained) | noise augmentation | noise_0.05 | 27.5 ± 1.8 | happy, neutral, happy |
| ResNet-18 (pretrained) | noise augmentation | noise_0.1 | 27.8 ± 2.8 | happy, neutral, happy |
| ResNet-18 (pretrained) | noise augmentation | noise_0.15 | 27.5 ± 3.3 | neutral, neutral, happy |
| ResNet-18 (pretrained) | noise augmentation | noise_0.2 | 27.3 ± 2.7 | neutral, neutral, sad |
| ResNet-18 (pretrained) | noise augmentation | res_40 | 29.6 ± 6.6 | sad, happy, happy |
| ResNet-18 (pretrained) | noise augmentation | res_32 | 30.8 ± 9.0 | sad, happy, happy |
| ResNet-18 (pretrained) | noise augmentation | res_24 | 33.5 ± 10.1 | sad, happy, happy |
| ResNet-18 (pretrained) | noise augmentation | res_16 | 39.1 ± 12.7 | sad, neutral, happy |
| ResNet-18 (pretrained) | noise augmentation | res_8 | 71.8 ± 12.6 | sad, happy, happy |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | clean | 27.2 ± 1.2 | happy, happy, happy |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | noise_0.025 | 34.2 ± 5.1 | neutral, neutral, neutral |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | noise_0.05 | 35.0 ± 5.6 | sad, neutral, neutral |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | noise_0.1 | 73.8 ± 14.9 | sad, sad, sad |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | noise_0.15 | 84.2 ± 15.2 | sad, sad, sad |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | noise_0.2 | 87.6 ± 15.3 | sad, sad, sad |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | res_40 | 29.4 ± 2.1 | happy, happy, happy |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | res_32 | 31.4 ± 1.8 | happy, happy, happy |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | res_24 | 39.9 ± 2.9 | happy, happy, happy |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | res_16 | 41.9 ± 14.5 | happy, sad, sad |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | res_8 | 74.9 ± 24.2 | neutral, sad, sad |
| ViT-B/16 (pretrained) | standard training | clean | 26.4 ± 0.4 | happy, happy, happy |
| ViT-B/16 (pretrained) | standard training | noise_0.025 | 25.0 ± 0.3 | happy, happy, happy |
| ViT-B/16 (pretrained) | standard training | noise_0.05 | 29.6 ± 2.1 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard training | noise_0.1 | 38.2 ± 7.2 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard training | noise_0.15 | 38.3 ± 16.6 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard training | noise_0.2 | 45.3 ± 18.4 | sad, angry, angry |
| ViT-B/16 (pretrained) | standard training | res_40 | 25.5 ± 0.6 | happy, happy, happy |
| ViT-B/16 (pretrained) | standard training | res_32 | 26.2 ± 1.2 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard training | res_24 | 34.1 ± 2.6 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard training | res_16 | 38.7 ± 5.6 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard training | res_8 | 58.0 ± 22.3 | sad, sad, angry |
| ViT-B/16 (pretrained) | noise augmentation | clean | 26.3 ± 0.3 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | noise_0.025 | 26.2 ± 0.2 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | noise_0.05 | 26.6 ± 0.3 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | noise_0.1 | 27.1 ± 0.3 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | noise_0.15 | 28.2 ± 0.4 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | noise_0.2 | 29.7 ± 0.6 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | res_40 | 25.9 ± 0.2 | happy, happy, happy |
| ViT-B/16 (pretrained) | noise augmentation | res_32 | 28.7 ± 1.6 | sad, sad, sad |
| ViT-B/16 (pretrained) | noise augmentation | res_24 | 37.4 ± 3.1 | sad, sad, sad |
| ViT-B/16 (pretrained) | noise augmentation | res_16 | 40.6 ± 3.9 | sad, sad, sad |
| ViT-B/16 (pretrained) | noise augmentation | res_8 | 85.8 ± 4.9 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | clean | 25.9 ± 0.4 | happy, happy, happy |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | noise_0.025 | 25.5 ± 1.6 | happy, sad, happy |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | noise_0.05 | 31.7 ± 2.6 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | noise_0.1 | 40.1 ± 4.6 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | noise_0.15 | 39.9 ± 11.9 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | noise_0.2 | 42.8 ± 17.5 | sad, angry, neutral |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | res_40 | 25.6 ± 0.4 | happy, happy, happy |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | res_32 | 26.1 ± 1.4 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | res_24 | 32.6 ± 2.9 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | res_16 | 36.5 ± 3.0 | sad, sad, sad |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | res_8 | 55.4 ± 13.6 | sad, fear, angry |

## 3. Test-set sampling uncertainty

Bootstrap with 5000 resamples of the 3268 test images; trained models are held fixed.

### Seed-averaged accuracy with a 95% bootstrap interval over test images (models held fixed)
| Model | Training | Clean | Noise avg | Resolution avg |
|---|---|---|---|---|
| Baseline CNN | standard training | 53.2 [51.7, 54.6] | 37.5 [36.5, 38.5] | 38.0 [36.9, 39.1] |
| Baseline CNN | noise augmentation | 53.5 [52.0, 54.9] | 47.5 [46.3, 48.8] | 39.4 [38.3, 40.6] |
| Baseline CNN | standard, clean-selected checkpoint | 53.9 [52.5, 55.4] | 34.4 [33.5, 35.2] | 38.7 [37.6, 39.8] |
| ResNet-18 (pretrained) | standard training | 61.2 [59.9, 62.6] | 37.8 [36.8, 38.8] | 38.6 [37.5, 39.5] |
| ResNet-18 (pretrained) | noise augmentation | 63.1 [61.7, 64.5] | 56.7 [55.5, 58.0] | 47.7 [46.6, 48.9] |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | 63.7 [62.4, 65.1] | 33.7 [32.8, 34.6] | 39.7 [38.8, 40.6] |
| ViT-B/16 (pretrained) | standard training | 69.3 [67.8, 70.7] | 44.0 [43.0, 44.9] | 45.8 [44.8, 46.9] |
| ViT-B/16 (pretrained) | noise augmentation | 68.5 [67.0, 70.0] | 61.9 [60.6, 63.2] | 49.8 [48.6, 51.0] |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | 69.5 [68.0, 70.9] | 43.2 [42.3, 44.2] | 45.7 [44.7, 46.7] |

### Noise augmentation minus standard training (same architecture)
Difference in seed-averaged accuracy, percentage points, with a 95% bootstrap interval over test images.
| Comparison | Clean | Noise avg | Resolution avg |
|---|---|---|---|
| Baseline CNN | +0.3 [-0.6, +1.3] | +10.0 [+9.1, +10.8] * | +1.4 [+0.7, +2.1] * |
| ResNet-18 (pretrained) | +1.8 [+0.8, +2.8] * | +18.9 [+17.9, +20.0] * | +9.2 [+8.4, +9.9] * |
| ViT-B/16 (pretrained) | -0.7 [-1.5, +0.0] | +17.9 [+16.9, +18.8] * | +3.9 [+3.3, +4.5] * |

### Clean-selected checkpoint minus proposal-rule checkpoint (standard training)
Difference in seed-averaged accuracy, percentage points, with a 95% bootstrap interval over test images.
| Comparison | Clean | Noise avg | Resolution avg |
|---|---|---|---|
| Baseline CNN | +0.8 [-0.1, +1.6] | -3.1 [-3.8, -2.5] * | +0.7 [+0.1, +1.2] * |
| ResNet-18 (pretrained) | +2.5 [+1.5, +3.5] * | -4.1 [-5.0, -3.3] * | +1.1 [+0.5, +1.8] * |
| ViT-B/16 (pretrained) | +0.2 [-0.4, +0.8] | -0.8 [-1.1, -0.4] * | -0.2 [-0.6, +0.2] |

### Architecture comparisons, standard training
Difference in seed-averaged accuracy, percentage points, with a 95% bootstrap interval over test images.
| Comparison | Clean | Noise avg | Resolution avg |
|---|---|---|---|
| ResNet-18 (pretrained) minus Baseline CNN | +8.1 [+6.9, +9.3] * | +0.3 [-0.7, +1.3] | +0.6 [-0.2, +1.3] |
| ViT-B/16 (pretrained) minus Baseline CNN | +16.1 [+14.7, +17.5] * | +6.5 [+5.5, +7.5] * | +7.8 [+6.8, +8.9] * |
| ViT-B/16 (pretrained) minus ResNet-18 (pretrained) | +8.0 [+7.0, +9.1] * | +6.2 [+5.2, +7.1] * | +7.3 [+6.4, +8.2] * |

### Architecture comparisons, noise augmentation
Difference in seed-averaged accuracy, percentage points, with a 95% bootstrap interval over test images.
| Comparison | Clean | Noise avg | Resolution avg |
|---|---|---|---|
| ResNet-18 (pretrained) minus Baseline CNN | +9.6 [+8.4, +10.8] * | +9.2 [+8.4, +10.1] * | +8.3 [+7.5, +9.2] * |
| ViT-B/16 (pretrained) minus Baseline CNN | +15.1 [+13.7, +16.5] * | +14.4 [+13.3, +15.4] * | +10.4 [+9.3, +11.4] * |
| ViT-B/16 (pretrained) minus ResNet-18 (pretrained) | +5.5 [+4.4, +6.5] * | +5.1 [+4.3, +5.9] * | +2.1 [+1.2, +2.9] * |

### Architecture comparisons, standard, clean-selected checkpoint
Difference in seed-averaged accuracy, percentage points, with a 95% bootstrap interval over test images.
| Comparison | Clean | Noise avg | Resolution avg |
|---|---|---|---|
| ResNet-18 (pretrained) minus Baseline CNN | +9.8 [+8.5, +11.0] * | -0.7 [-1.7, +0.2] | +1.0 [+0.2, +1.8] * |
| ViT-B/16 (pretrained) minus Baseline CNN | +15.5 [+14.1, +16.9] * | +8.8 [+7.9, +9.8] * | +7.0 [+5.9, +8.0] * |
| ViT-B/16 (pretrained) minus ResNet-18 (pretrained) | +5.7 [+4.6, +6.8] * | +9.5 [+8.7, +10.4] * | +6.0 [+5.1, +6.8] * |

`*` marks intervals that exclude zero. These intervals reflect test-set sampling only; seed-to-seed variation is reported separately as the SD over three seeds.

## 4. Per-class recall under a mid-level degradation

Recall on original images -> recall at σ = 0.1 / at 24 px (seed mean, %).
| Model | Training | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 41 → 29 / 15 | 11 → 1 / 1 | 31 → 8 / 22 | 82 → 52 / 76 | 42 → 20 / 48 | 61 → 38 / 21 | 49 → 59 / 20 |
| Baseline CNN | noise augmentation | 34 → 30 / 12 | 10 → 5 / 4 | 30 → 23 / 23 | 78 → 74 / 68 | 50 → 45 / 63 | 59 → 52 / 27 | 56 → 50 / 28 |
| Baseline CNN | standard, clean-selected checkpoint | 43 → 21 / 18 | 10 → 4 / 1 | 29 → 7 / 18 | 80 → 34 / 71 | 49 → 19 / 54 | 58 → 29 / 26 | 51 → 69 / 23 |
| ResNet-18 (pretrained) | standard training | 53 → 17 / 20 | 18 → 0 / 4 | 40 → 3 / 12 | 91 → 68 / 88 | 49 → 43 / 39 | 65 → 13 / 24 | 56 → 23 / 16 |
| ResNet-18 (pretrained) | noise augmentation | 51 → 42 / 22 | 30 → 24 / 17 | 37 → 30 / 26 | 87 → 82 / 79 | 53 → 48 / 61 | 68 → 66 / 48 | 69 → 67 / 53 |
| ResNet-18 (pretrained) | standard, clean-selected checkpoint | 62 → 7 / 28 | 40 → 0 / 2 | 43 → 5 / 31 | 88 → 33 / 83 | 43 → 87 / 37 | 66 → 6 / 22 | 67 → 6 / 20 |
| ViT-B/16 (pretrained) | standard training | 63 → 35 / 51 | 54 → 1 / 9 | 48 → 20 / 28 | 91 → 50 / 71 | 56 → 62 / 66 | 77 → 40 / 53 | 71 → 38 / 22 |
| ViT-B/16 (pretrained) | noise augmentation | 60 → 53 / 50 | 40 → 28 / 14 | 47 → 38 / 29 | 91 → 86 / 78 | 57 → 51 / 74 | 73 → 71 / 55 | 71 → 64 / 32 |
| ViT-B/16 (pretrained) | standard, clean-selected checkpoint | 60 → 32 / 50 | 51 → 1 / 4 | 50 → 26 / 31 | 91 → 46 / 71 | 60 → 64 / 62 | 76 → 38 / 49 | 69 → 31 / 22 |
