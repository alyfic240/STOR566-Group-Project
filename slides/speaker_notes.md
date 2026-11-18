# Speaker notes — final presentation (12 min talk + 3 min Q&A)

Extracted from the deck's slide files (`slides/deck_root/project/slides/*.html`); the notes are also in the deck itself.
Slides 1–12 are the talk; slides 13–20 are backup slides for questions.
Each note starts with a suggested time for the slide.

### 1. Facial-expression classification under image degradation
*STOR 566 · FINAL PROJECT · GROUP 9*

0:15. Names, one sentence on the topic: how well do facial-expression classifiers hold up when the image gets noisy or small, and does training with noise help. Plan: 12 minutes, 12 slides, backup slides after the takeaways.

### 2. Clean test accuracy is not what a deployed classifier sees
*MOTIVATION*

1:00. Benchmarks report accuracy on clean test images. A model that is 8 points better on clean images is not necessarily 8 points better on what a webcam produces. Read the three questions; they are unchanged from the proposal. One caveat up front: labels are expressions judged by annotators, not emotions.

### 3. Since the midterm: a stronger model, the full stress test, a wider literature base
*PROGRESS*

0:45. At the midterm we had a partly trained baseline CNN and a plan. The feedback was: get a better model than the pretrained ResNet, then stress-test, and widen the literature review. All three are done. Everything you see today was run under one protocol with three seeds per configuration.

### 4. Every model sees the same test images: FER-2013 without duplicates, two degradations, five levels each
*DATA AND DEGRADATIONS*

1:00. Before any modelling we audited the data: 1,516 groups of pixel-identical images, 557 of them crossing the official splits, so a model could be tested on training images. We removed conflicting-label copies and kept one copy of the rest. Two degradations, each controlled by one number, five levels each, chosen with a pilot model on validation images so they run from almost no loss to near the majority-class rate. The same degraded test images are used for every model and seed.

### 5. Every choice is made on validation data; the test set is used once
*MODELS AND PROTOCOL*

1:15. Three models of increasing size: a 94 thousand parameter CNN from scratch, a pretrained ResNet-18, a pretrained ViT-B/16. Learning rate, stopping epoch and the noise-augmentation setting are all picked with one validation score, half original accuracy and half noisy accuracy. The chosen settings were frozen to a file before we looked at any test number. Reduced resolution never enters a decision. We come back on slide 10 to what this selection rule cost us.

### 6. On original images, bigger pretrained models win
*RQ1 · ORIGINAL IMAGES*

0:50. Test accuracy on original images, mean of three seeds with the standard deviation: 53.2, 61.2 and 69.3 percent. The gaps are far larger than the seed-to-seed variation. Macro-F1 is lower for every model because disgust is 1.2 percent of the test set and is rarely recovered. For orientation the dataset authors put human accuracy at about 65 percent; our test set is filtered, so it is not the same benchmark.

### 7. Under standard training, that lead shrinks or disappears as images degrade
*RQ2 · STANDARD TRAINING*

1:30. Standard training, no noise seen in training. All three curves fall fast: every model is below 90 percent of its own clean accuracy by sigma 0.05. At sigma 0.1 the models keep 68, 54 and 61 percent of their clean accuracy. Averaged over the five noise levels ResNet-18 beats the small CNN by 0.3 points instead of 8.1; that difference is inside test-set sampling error. ViT keeps part of its lead, 6.5 points instead of 16.1. At the strongest levels the ranking is gone, and at 8 pixels the best clean model is the worst. Pre-training bought clean accuracy. It did not buy robustness. Same pattern Hendrycks and Dietterich reported on ImageNet-C.

### 8. Noise augmentation costs little on clean images and transfers to low resolution for pretrained models
*RQ3 · NOISE AUGMENTATION*

1:30. Each training image gets Gaussian noise with probability p and a random strength up to sigma-max; p and sigma-max were chosen on validation. On original images the change is small in both directions. On noisy images the gain is large and grows with severity. The interesting part is reduced resolution: it was never used in training or in any choice we made, so this is transfer, not tuning. It transfers for the pretrained models, +9.2 and +3.9 points on average, and hardly at all for the small CNN. That fits the frequency argument of Yin and co-authors: noise training pushes a model away from fine detail, and down-sampling removes exactly that detail. Why the small CNN gains so little we did not test.

### 9. Under heavy degradation the models stop looking at the face: predictions collapse onto one class
*FAILURE ANALYSIS*

1:15. Accuracy near the majority rate could mean guessing. It does not. This is the share of test images put into the single most-predicted class. On clean images it is about 26 to 30 percent, like the true label distribution. Under standard training it climbs: at sigma 0.2, 61, 71 and 45 percent of all images get the same label. Which label changes with the seed, happy, sad or neutral, which is why accuracy can drop below the majority rate. With noise augmentation the models stay near the true distribution under noise, but at 8 pixels they collapse as well. Per class, happy is the most robust; disgust and fear are the first to go (backup slide).

### 10. The checkpoint rule matters: our pre-specified score stops standard models early
*PROTOCOL CHECK*

1:00. We fixed the selection rule in advance, and it has a side effect. For models trained without noise, the noisy half of the score jumps around and is often highest after the first epoch, so the rule stopped two of three ResNet-18 runs at epoch 1. We checked what that costs with reference runs that pick the checkpoint on original validation accuracy only: ResNet-18 gains 2.5 points on original images and loses 4.1 under noise. The conclusions hold either way: ResNet-18 is still no better than the CNN under noise, and noise augmentation still adds 23.1 points. For ViT-B/16 the two rules differ by less than a point; the full table is on a backup slide. The lesson for anyone comparing robustness: say how checkpoints were picked.

### 11. What these results do not show
*LIMITATIONS*

0:45. One dataset with noisy labels. Two synthetic degradations, one at a time. The three methods differ in architecture, size and pre-training at once, so we compare methods, not causes. Three seeds. A small tuning grid whose best learning rate sits at the edge for the pretrained models; a check one step beyond the grid changed the score by less than its seed-to-seed spread. And no flips or crops, so that noise is the only augmentation; that keeps the comparison clean and costs some clean accuracy. Next steps: more corruption types and mixtures, FER+ labels, comparison at matched clean accuracy.

### 12. Three things to remember
*TAKEAWAYS*

0:45. If you remember one thing: report accuracy under the conditions you expect, and say how you picked the checkpoint. Then questions.

---

## Backup slides

### 13. Selection scores on validation images: learning rates and noise-augmentation settings
*BACKUP · SELECTION*

Backup. Tuning runs use seed 0 and validation images only. The chosen learning rate is the largest of the grid for ResNet-18 and ViT-B/16, so after everything was frozen we ran one value beyond the grid. ResNet-18 scored 0.484 against 0.491 at the chosen value, ViT-B/16 0.560 against 0.556. Both differences are smaller than the seed-to-seed standard deviation of the score, so the grid edge did not cost much. The check was never used for selection.

### 14. Standard training under two checkpoint rules: a few points for the smaller models, under one for ViT-B/16
*BACKUP · CHECKPOINT RULE*

Backup for the question 'what if you had picked checkpoints the usual way?'. Same training, but the checkpoint is chosen on original validation accuracy only. ResNet-18 gains 2.5 points on original images and loses 4.1 under noise, with the same sign in all three seeds. For the CNN the averages point the same way, 0.8 and 3.1 points, but not every seed agrees. For ViT-B/16, fine-tuned for four epochs, the two rules differ by less than a point. Our conclusions hold with either rule: ResNet-18 is still no better than the CNN under noise, ViT-B/16 is still the most accurate, and noise augmentation still adds 13 to 23 points under noise.

### 15. Beyond the primary levels: σ up to 0.5 and 4 px
*BACKUP · EXTENDED LEVELS*

Backup. The heavier levels used at the midterm are reported separately as an extended sweep. Beyond sigma 0.2 standard models sit at or below the majority rate; augmented models degrade gradually because they were trained up to sigma-max 0.2 only.

### 16. Relative accuracy under standard training: the small CNN retains the most
*BACKUP · RELATIVE ACCURACY*

Backup. Degraded accuracy divided by original accuracy. First level below 90 percent: sigma 0.05 for all three; 32 pixels for the CNN and ViT-B/16, 40 pixels for ResNet-18. With augmentation: sigma 0.15 for all three, and 24 pixels for the pretrained models.

### 17. Standard training and noise augmentation at every level
*BACKUP · ALL CURVES*

Backup. Solid lines and filled circles are standard training; dashed lines and hollow squares are noise augmentation.

### 18. Which expressions survive: recall on original images, at σ = 0.1 and at 24 px (standard training)
*BACKUP · PER CLASS*

Backup. Recall per class, in percent, mean of three seeds. Happy stays recognisable; disgust and fear drop to near zero under noise for the CNN and ResNet-18. Rising recall for one class under degradation, for example sad for ViT-B/16, is the collapse showing up as apparent recall.

### 19. Differences in test accuracy with 95% bootstrap intervals over test images
*BACKUP · UNCERTAINTY*

Backup for the question 'is that significant?'. Two sources of uncertainty: seeds (SD over three runs) and the finite test set (bootstrap over images, models fixed). Large effects are far outside both; the ResNet-18 versus CNN difference under degradation is not distinguishable from zero.

### 20. Where this project sits in the literature
*BACKUP · RELATED WORK*

Backup. Closest prior work: Greco and co-authors built corrupted versions of the RAF-DB test set. We differ in working at FER-2013's native 48 by 48 resolution, including a from-scratch baseline, fixing the selection protocol in advance, and testing transfer from noise augmentation to an unseen degradation.
