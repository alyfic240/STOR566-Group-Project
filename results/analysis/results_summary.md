# Results summary (test split unless stated; mean ± sample SD over seeds, in %)

## RQ1 - original images

| Model | Training | Seeds | Val acc | Test acc | Test macro-F1 | Best epoch |
|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 3 | 53.4 ± 0.7 | 53.2 ± 1.3 | 45.6 ± 2.4 | 28, 22, 32 |
| Baseline CNN | noise augmentation | 3 | 53.7 ± 0.6 | 53.5 ± 0.4 | 45.9 ± 1.1 | 49, 43, 51 |
| ResNet-18 (pretrained) | standard training | 3 | 61.0 ± 2.3 | 61.2 ± 1.4 | 53.4 ± 5.1 | 6, 1, 1 |
| ResNet-18 (pretrained) | noise augmentation | 3 | 62.0 ± 0.7 | 63.1 ± 1.1 | 57.8 ± 1.1 | 4, 3, 5 |
| ViT-B/16 (pretrained) | standard training | 3 | 68.5 ± 0.9 | 69.3 ± 1.1 | 65.3 ± 2.0 | 3, 2, 4 |
| ViT-B/16 (pretrained) | noise augmentation | 3 | 67.9 ± 0.1 | 68.5 ± 0.5 | 63.9 ± 0.5 | 4, 4, 4 |

## RQ2 - test accuracy under Gaussian noise (sigma; primary levels first, then extended)

| Model | Training | clean | 0.025 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 53.2 ± 1.3 | 51.3 ± 0.5 | 47.1 ± 1.5 | 36.1 ± 2.4 | 28.3 ± 0.9 | 24.9 ± 2.5 | 23.5 ± 3.2 | 22.5 ± 3.3 | 22.1 ± 3.8 |
| Baseline CNN | noise augmentation | 53.5 ± 0.4 | 52.9 ± 0.4 | 51.8 ± 0.8 | 48.2 ± 0.7 | 44.4 ± 0.6 | 40.1 ± 0.4 | 33.3 ± 0.2 | 28.6 ± 0.6 | 26.9 ± 0.9 |
| ResNet-18 (pretrained) | standard training | 61.2 ± 1.4 | 55.3 ± 1.2 | 46.2 ± 4.5 | 33.1 ± 5.4 | 28.4 ± 2.2 | 26.1 ± 0.7 | 23.4 ± 3.9 | 22.9 ± 4.5 | 21.8 ± 4.2 |
| ResNet-18 (pretrained) | noise augmentation | 63.1 ± 1.1 | 62.2 ± 0.9 | 60.8 ± 1.0 | 58.1 ± 0.3 | 53.5 ± 0.4 | 49.0 ± 0.5 | 40.4 ± 0.5 | 33.3 ± 0.7 | 27.6 ± 1.2 |
| ViT-B/16 (pretrained) | standard training | 69.3 ± 1.1 | 65.2 ± 0.6 | 58.5 ± 0.2 | 42.1 ± 0.4 | 31.1 ± 1.1 | 23.0 ± 0.3 | 18.2 ± 0.6 | 16.6 ± 1.1 | 15.7 ± 0.9 |
| ViT-B/16 (pretrained) | noise augmentation | 68.5 ± 0.5 | 68.0 ± 0.6 | 66.8 ± 0.3 | 62.3 ± 0.4 | 58.6 ± 0.2 | 53.5 ± 0.3 | 45.2 ± 0.6 | 38.9 ± 0.6 | 34.0 ± 0.6 |

## RQ2 - test macro-F1 under Gaussian noise (sigma; primary levels first, then extended)

| Model | Training | clean | 0.025 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 45.6 ± 2.4 | 43.9 ± 1.5 | 39.8 ± 2.4 | 28.3 ± 3.3 | 18.5 ± 2.6 | 13.5 ± 1.8 | 10.6 ± 2.2 | 9.2 ± 2.2 | 8.8 ± 2.0 |
| Baseline CNN | noise augmentation | 45.9 ± 1.1 | 45.3 ± 0.4 | 44.3 ± 0.7 | 40.1 ± 0.4 | 36.4 ± 1.1 | 31.0 ± 1.1 | 24.8 ± 0.3 | 20.2 ± 0.5 | 17.5 ± 1.2 |
| ResNet-18 (pretrained) | standard training | 53.4 ± 5.1 | 46.3 ± 2.9 | 34.9 ± 4.6 | 20.2 ± 7.4 | 13.9 ± 3.8 | 11.2 ± 1.7 | 8.3 ± 0.8 | 7.6 ± 1.6 | 7.4 ± 1.9 |
| ResNet-18 (pretrained) | noise augmentation | 57.8 ± 1.1 | 56.1 ± 1.5 | 55.0 ± 0.9 | 52.4 ± 1.0 | 46.5 ± 1.3 | 42.7 ± 0.9 | 34.1 ± 1.9 | 27.9 ± 1.0 | 21.8 ± 1.6 |
| ViT-B/16 (pretrained) | standard training | 65.3 ± 2.0 | 59.7 ± 1.1 | 50.7 ± 0.3 | 35.6 ± 0.9 | 25.6 ± 0.7 | 17.5 ± 2.2 | 11.5 ± 2.4 | 9.2 ± 1.3 | 8.0 ± 0.9 |
| ViT-B/16 (pretrained) | noise augmentation | 63.9 ± 0.5 | 63.6 ± 0.3 | 62.8 ± 1.0 | 56.7 ± 0.4 | 52.8 ± 1.4 | 45.6 ± 0.8 | 37.3 ± 0.3 | 30.7 ± 0.5 | 26.0 ± 0.5 |

### Relative accuracy under Gaussian noise (degraded / original, per seed) and 90% threshold

| Model | Training | 0.025 | 0.05 | 0.1 | 0.15 | 0.2 | first primary level below 90% |
|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 96.4 ± 2.0 | 88.6 ± 3.2 | 68.0 ± 4.2 | 53.2 ± 1.7 | 46.8 ± 5.0 | 0.05 |
| Baseline CNN | noise augmentation | 99.0 ± 1.0 | 96.8 ± 2.1 | 90.2 ± 1.6 | 83.1 ± 1.7 | 74.9 ± 0.9 | 0.15 |
| ResNet-18 (pretrained) | standard training | 90.3 ± 3.8 | 75.5 ± 9.0 | 54.2 ± 9.9 | 46.4 ± 4.5 | 42.6 ± 1.6 | 0.05 |
| ResNet-18 (pretrained) | noise augmentation | 98.6 ± 0.4 | 96.5 ± 0.5 | 92.2 ± 1.6 | 84.8 ± 1.0 | 77.8 ± 2.0 | 0.15 |
| ViT-B/16 (pretrained) | standard training | 94.2 ± 0.7 | 84.4 ± 1.1 | 60.9 ± 1.6 | 44.9 ± 0.9 | 33.2 ± 0.7 | 0.05 |
| ViT-B/16 (pretrained) | noise augmentation | 99.2 ± 0.1 | 97.5 ± 1.0 | 90.9 ± 1.0 | 85.6 ± 0.8 | 78.0 ± 0.3 | 0.15 |

## RQ2 - test accuracy under reduced resolution (size; primary levels first, then extended)

| Model | Training | clean | 40 | 32 | 24 | 16 | 8 | 4 |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 53.2 ± 1.3 | 48.4 ± 1.3 | 45.0 ± 0.6 | 38.8 ± 0.7 | 34.4 ± 0.3 | 23.5 ± 3.3 | 14.7 ± 2.8 |
| Baseline CNN | noise augmentation | 53.5 ± 0.4 | 49.7 ± 0.4 | 47.1 ± 0.2 | 41.2 ± 0.5 | 35.9 ± 0.6 | 23.0 ± 1.5 | 16.3 ± 1.2 |
| ResNet-18 (pretrained) | standard training | 61.2 ± 1.4 | 54.3 ± 1.0 | 48.4 ± 1.9 | 39.2 ± 0.9 | 30.1 ± 1.9 | 20.7 ± 4.8 | 19.1 ± 3.1 |
| ResNet-18 (pretrained) | noise augmentation | 63.1 ± 1.1 | 60.3 ± 1.1 | 57.3 ± 1.8 | 52.0 ± 2.2 | 43.5 ± 2.4 | 25.5 ± 4.0 | 23.7 ± 4.0 |
| ViT-B/16 (pretrained) | standard training | 69.3 ± 1.1 | 63.3 ± 0.1 | 59.0 ± 0.7 | 49.5 ± 1.4 | 39.8 ± 2.1 | 17.6 ± 0.4 | 17.2 ± 0.7 |
| ViT-B/16 (pretrained) | noise augmentation | 68.5 ± 0.5 | 64.8 ± 0.3 | 61.9 ± 0.5 | 55.2 ± 1.0 | 47.7 ± 1.1 | 19.2 ± 0.7 | 17.5 ± 0.3 |

## RQ2 - test macro-F1 under reduced resolution (size; primary levels first, then extended)

| Model | Training | clean | 40 | 32 | 24 | 16 | 8 | 4 |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 45.6 ± 2.4 | 39.9 ± 2.8 | 36.0 ± 1.9 | 28.1 ± 2.2 | 23.4 ± 2.6 | 15.5 ± 0.5 | 10.6 ± 1.9 |
| Baseline CNN | noise augmentation | 45.9 ± 1.1 | 41.6 ± 0.6 | 39.3 ± 0.1 | 32.1 ± 0.1 | 26.6 ± 0.7 | 16.5 ± 0.8 | 10.1 ± 0.3 |
| ResNet-18 (pretrained) | standard training | 53.4 ± 5.1 | 45.1 ± 2.7 | 38.1 ± 3.8 | 26.7 ± 2.9 | 18.1 ± 1.4 | 8.4 ± 2.7 | 7.4 ± 2.7 |
| ResNet-18 (pretrained) | noise augmentation | 57.8 ± 1.1 | 54.7 ± 2.0 | 51.3 ± 2.1 | 44.4 ± 2.5 | 34.1 ± 2.5 | 12.3 ± 2.3 | 7.7 ± 0.8 |
| ViT-B/16 (pretrained) | standard training | 65.3 ± 2.0 | 59.6 ± 1.1 | 53.9 ± 0.2 | 42.5 ± 1.8 | 32.8 ± 1.3 | 10.8 ± 2.6 | 7.4 ± 1.1 |
| ViT-B/16 (pretrained) | noise augmentation | 63.9 ± 0.5 | 60.2 ± 0.6 | 56.7 ± 0.4 | 48.9 ± 0.7 | 40.0 ± 1.0 | 10.0 ± 1.6 | 6.0 ± 2.1 |

### Relative accuracy under reduced resolution (degraded / original, per seed) and 90% threshold

| Model | Training | 40 | 32 | 24 | 16 | 8 | first primary level below 90% |
|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 91.0 ± 1.2 | 84.6 ± 2.1 | 73.0 ± 1.2 | 64.8 ± 2.0 | 44.3 ± 7.2 | 32 |
| Baseline CNN | noise augmentation | 92.9 ± 0.9 | 88.1 ± 0.5 | 77.0 ± 0.7 | 67.2 ± 1.6 | 43.0 ± 2.8 | 32 |
| ResNet-18 (pretrained) | standard training | 88.7 ± 2.2 | 79.2 ± 4.6 | 64.1 ± 2.1 | 49.2 ± 3.7 | 33.9 ± 8.2 | 40 |
| ResNet-18 (pretrained) | noise augmentation | 95.6 ± 0.1 | 90.8 ± 1.4 | 82.4 ± 2.1 | 68.9 ± 2.7 | 40.3 ± 5.7 | 24 |
| ViT-B/16 (pretrained) | standard training | 91.4 ± 1.5 | 85.2 ± 2.4 | 71.5 ± 3.0 | 57.5 ± 3.7 | 25.4 ± 0.8 | 32 |
| ViT-B/16 (pretrained) | noise augmentation | 94.5 ± 0.3 | 90.4 ± 0.6 | 80.6 ± 1.0 | 69.6 ± 1.4 | 28.0 ± 0.8 | 24 |

## RQ3 - effect of noise augmentation (augmented minus standard, same seed; percentage points)

| Model | clean | noise_0.025 | noise_0.05 | noise_0.1 | noise_0.15 | noise_0.2 | res_40 | res_32 | res_24 | res_16 | res_8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | +0.3 ± 1.4 | +1.7 ± 0.8 | +4.7 ± 2.3 | +12.1 ± 2.7 | +16.2 ± 0.7 | +15.2 ± 2.5 | +1.3 ± 1.6 | +2.1 ± 0.5 | +2.4 ± 0.2 | +1.5 ± 0.5 | -0.5 ± 1.8 |
| ResNet-18 (pretrained) | +1.8 ± 2.3 | +6.9 ± 1.7 | +14.7 ± 4.0 | +25.0 ± 5.4 | +25.1 ± 2.0 | +22.9 ± 0.8 | +6.0 ± 2.0 | +8.9 ± 2.1 | +12.7 ± 2.9 | +13.4 ± 1.1 | +4.8 ± 3.9 |
| ViT-B/16 (pretrained) | -0.7 ± 0.8 | +2.8 ± 0.2 | +8.4 ± 0.5 | +20.2 ± 0.2 | +27.6 ± 1.0 | +30.5 ± 0.5 | +1.4 ± 0.4 | +3.0 ± 1.2 | +5.7 ± 2.3 | +7.9 ± 2.7 | +1.6 ± 0.9 |

## Reference - standard training with the checkpoint chosen on clean validation accuracy only

Same training and learning rate as 'standard training'; only the stopping rule differs.

| Model | Seeds | Val acc | Test acc | Test macro-F1 | Best epoch | noise_0.025 | noise_0.05 | noise_0.1 | noise_0.15 | noise_0.2 | res_40 | res_32 | res_24 | res_16 | res_8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 3 | 54.5 ± 0.2 | 53.9 ± 0.5 | 46.1 ± 1.6 | 32, 60, 58 | 51.7 ± 1.7 | 45.4 ± 3.1 | 31.2 ± 6.0 | 23.2 ± 4.4 | 20.4 ± 3.2 | 49.7 ± 1.7 | 45.8 ± 3.3 | 39.3 ± 3.9 | 35.0 ± 2.5 | 23.7 ± 1.2 |
| ResNet-18 (pretrained) | 3 | 63.0 ± 0.6 | 63.7 ± 0.3 | 58.9 ± 0.8 | 6, 2, 4 | 56.2 ± 1.6 | 44.8 ± 3.8 | 27.1 ± 4.6 | 21.1 ± 2.9 | 19.2 ± 1.9 | 56.2 ± 0.6 | 51.6 ± 1.3 | 42.1 ± 1.2 | 29.5 ± 3.7 | 19.1 ± 0.6 |
| ViT-B/16 (pretrained) | 3 | 68.7 ± 0.6 | 69.5 ± 0.4 | 65.8 ± 0.5 | 4, 3, 4 | 65.3 ± 0.5 | 57.5 ± 0.5 | 40.3 ± 0.4 | 30.2 ± 0.7 | 22.8 ± 1.0 | 63.7 ± 0.3 | 59.3 ± 0.7 | 49.1 ± 1.8 | 38.6 ± 0.3 | 17.5 ± 1.3 |

### Effect of the selection rule (clean + noisy selection minus clean-only selection, same seed; percentage points of test accuracy)

| Model | clean | noise_0.025 | noise_0.05 | noise_0.1 | noise_0.15 | noise_0.2 | res_40 | res_32 | res_24 | res_16 | res_8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | -0.8 ± 1.0 | -0.5 ± 1.5 | +1.7 ± 2.7 | +4.9 ± 5.3 | +5.1 ± 5.3 | +4.4 ± 4.9 | -1.4 ± 1.3 | -0.8 ± 2.8 | -0.5 ± 3.3 | -0.6 ± 2.7 | -0.2 ± 2.4 |
| ResNet-18 (pretrained) | -2.5 ± 1.6 | -0.9 ± 2.7 | +1.4 ± 7.0 | +6.0 ± 7.5 | +7.3 ± 3.2 | +6.9 ± 2.6 | -2.0 ± 1.3 | -3.1 ± 1.0 | -2.8 ± 1.0 | +0.5 ± 5.4 | +1.6 ± 4.8 |
| ViT-B/16 (pretrained) | -0.2 ± 0.7 | -0.1 ± 0.4 | +1.0 ± 0.5 | +1.9 ± 0.3 | +0.8 ± 1.6 | +0.2 ± 0.8 | -0.4 ± 0.3 | -0.3 ± 0.3 | +0.4 ± 0.5 | +1.2 ± 1.9 | +0.1 ± 1.7 |

## Per-class recall on original test images (%)

| Model | Training | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard training | 41.0 | 10.5 | 31.0 | 81.5 | 42.4 | 61.2 | 49.2 |
| Baseline CNN | noise augmentation | 34.2 | 9.6 | 30.3 | 77.8 | 50.0 | 58.9 | 55.7 |
| ResNet-18 (pretrained) | standard training | 52.7 | 18.4 | 39.9 | 90.9 | 48.9 | 64.7 | 55.6 |
| ResNet-18 (pretrained) | noise augmentation | 51.3 | 29.8 | 37.3 | 86.6 | 52.8 | 68.4 | 68.7 |
| ViT-B/16 (pretrained) | standard training | 62.7 | 53.5 | 47.6 | 91.1 | 56.3 | 76.9 | 70.5 |
| ViT-B/16 (pretrained) | noise augmentation | 60.3 | 40.4 | 46.9 | 91.0 | 57.2 | 72.8 | 70.9 |

## Selected hyperparameters (validation selection score only)

- Baseline CNN: lr 0.001 (scores {'0.0003': 0.4173503494378608, '0.001': 0.45776359769067154, '0.003': 0.44576116681859623}); augmentation p=0.5, sigma_max=0.2 (scores {'0.5|0.1': 0.49787298693406257, '0.5|0.2': 0.5154968094804011, '0.8|0.1': 0.5071406867213613, '0.8|0.2': 0.5092676997872987}); frozen 2026-11-16 18:20:01
- ResNet-18 (pretrained): lr 0.0003 (scores {'3e-05': 0.46292920085080524, '0.0001': 0.48526283804314796, '0.0003': 0.4913400182315406}); augmentation p=0.8, sigma_max=0.2 (scores {'0.5|0.1': 0.5814342145244606, '0.5|0.2': 0.5948040109389243, '0.8|0.1': 0.5803707079914919, '0.8|0.2': 0.5976906715284108}); frozen 2026-11-16 18:59:57
- ViT-B/16 (pretrained): lr 5e-05 (scores {'1e-05': 0.5323609845031906, '2e-05': 0.5439076268611365, '5e-05': 0.5556061987237921}); augmentation p=0.8, sigma_max=0.2 (scores {'0.5|0.1': 0.6403828623518687, '0.5|0.2': 0.6432695229413552, '0.8|0.1': 0.6396232148283196, '0.8|0.2': 0.6464600425402613}); frozen 2026-11-16 21:51:31

SD values describe variation across training seeds; they are not confidence intervals.
