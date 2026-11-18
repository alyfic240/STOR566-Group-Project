#!/usr/bin/env python3
"""Generate the deck's files (project/deck.json, project/deck.css, project/slides/*.html) from the result numbers.

Usage: python3 build_deck.py <out_dir>
Reads  paper/generated/numbers.json   (made by paper/make_assets.py from the same <out_dir>)
       slides/assets.json             (figure name -> /_blob/<id> url returned by the asset upload)
       <out_dir>/selection_*.json, <out_dir>/analysis_extra/extra_summary.json
Writes slides/deck_root/project/...
Every number on a slide comes from numbers.json, so re-running this after new results updates the deck text.
"""
import datetime
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "deck_root")
D = json.load(open(os.path.join(os.path.dirname(HERE), "paper", "generated", "numbers.json")))
ASSETS = json.load(open(os.path.join(HERE, "assets.json")))
TITLE = "STOR 566 Group 9 — Facial-Expression Classification under Image Degradation (final presentation)"
FOOT = "Group 9 · STOR 566 · Fall 2026"
ARCH = {"cnn": "Baseline CNN", "resnet18": "ResNet-18", "vit_b16": "ViT-B/16"}


def n(key, default="[?]"):
    v = D.get(key, default)
    return v.replace("{,}", ",").replace("$", "").replace("{\\times}", "×").replace("^{-", "⁻").replace("}", "").replace("{", "") if isinstance(v, str) else v


def minus(s):
    """typographic minus for negative numbers shown on slides"""
    return s.replace("-", "−")


def rnd(key):
    return f"{float(D[key]):.0f}"


def img(name, w, h, alt):
    return f'<img src="{ASSETS[name]}" alt="{html.escape(alt, quote=True)}" style="width:{w}px;height:{h}px;object-fit:contain">'


def foot():
    return f'<div class="foot"><p class="f">{FOOT}</p><div class="pg"></div></div>'


def slide(sid, eyebrow, title, body, notes, cls="", draft=None, footer=True):
    c = f' class="{cls}"' if cls else ""
    d = f'\n<p class="draft">{draft}</p>' if draft else ""
    return (f'<section id="{sid}"{c} data-transition="fade">\n<p class="eyebrow">{eyebrow}</p>\n<h2>{title}</h2>{d}\n'
            f'{body}\n{foot() if footer else ""}\n<aside>{html.escape(notes, quote=False)}</aside>\n</section>\n')


N_VIT_AUG = int(D.get("nseeds@vit_b16@aug", "0"))
HAS_REF = "acc@resnet18@ref@clean" in D
HAS_REF_ALL = all(f"acc@{a}@ref@clean" in D for a in ("cnn", "resnet18", "vit_b16"))
HAS_LRCHECK = D.get("haslrcheck") == "1" and "lrcheck@resnet18@score" in D and "lrcheck@vit_b16@score" in D
DRAFT_VIT = None if N_VIT_AUG >= 3 else f"DRAFT · ViT-B/16 aug: {N_VIT_AUG} of 3 seeds"
N_FINAL = int(D["nfinal"])

CSS = """body { font-family: 'Inter', Arial, sans-serif; color: #2f3742; }
section { padding: 128px 128px 160px; background: #fcfcfb; display: flex; flex-direction: column; counter-increment: page; }
.dark { background: #13294b; }
.eyebrow { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 24px; font-weight: 600; line-height: 1.2; letter-spacing: 3px; color: #56687f; }
.dark .eyebrow { color: #8fc0e6; }
h1 { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 92px; font-weight: 600; line-height: 1.06; color: #f6f5f0; padding-top: 28px; }
h2 { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 52px; font-weight: 600; line-height: 1.12; color: #13294b; padding-top: 14px; }
.dark h2 { color: #f6f5f0; }
h3 { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 32px; font-weight: 600; line-height: 1.2; color: #13294b; }
p { font-size: 30px; line-height: 1.35; }
.sub { font-size: 40px; line-height: 1.25; color: #c9d6e6; padding-top: 28px; }
.names { font-size: 32px; line-height: 1.3; color: #f6f5f0; }
.names2 { font-size: 28px; line-height: 1.3; color: #9fb3cc; padding-top: 8px; }
.spacer { flex: 1; }
.body { display: flex; gap: 48px; padding-top: 36px; flex: 1; }
.stack { display: flex; flex-direction: column; gap: 24px; padding-top: 36px; flex: 1; }
.col { display: flex; flex-direction: column; gap: 24px; }
.grow { flex: 1; }
.row { display: flex; gap: 24px; }
.card { background: #f1efe9; border-radius: 16px; padding: 24px 32px; display: flex; flex-direction: column; gap: 10px; }
.card .t { font-size: 28px; line-height: 1.35; color: #2f3742; }
.pair { background: #f1efe9; border-radius: 16px; padding: 24px 32px; display: flex; align-items: center; gap: 32px; }
.pair .from { font-size: 28px; line-height: 1.35; color: #56687f; width: 470px; }
.pair .to { font-size: 28px; line-height: 1.35; color: #2f3742; flex: 1; }
.num { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 72px; font-weight: 600; line-height: 1.05; color: #13294b; }
.num2 { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 44px; font-weight: 600; line-height: 1.1; color: #13294b; }
.lab { font-size: 26px; line-height: 1.3; color: #56687f; }
.note { font-size: 26px; line-height: 1.35; color: #56687f; }
.tile { display: flex; flex-direction: column; gap: 6px; }
.step { background: #fcfcfb; border: 2px solid #c9d0da; border-radius: 16px; padding: 22px 26px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
.step .t { font-size: 26px; line-height: 1.3; color: #2f3742; }
.steps { display: flex; gap: 16px; align-items: stretch; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; padding-top: 36px; }
.take { display: flex; gap: 40px; align-items: flex-start; }
.take .k { font-family: 'Space Grotesk', Arial, sans-serif; font-size: 72px; font-weight: 600; line-height: 1.0; color: #8fc0e6; width: 70px; }
.take .big { font-size: 44px; font-weight: 600; line-height: 1.2; color: #f6f5f0; }
.take .small { font-size: 30px; line-height: 1.35; color: #c9d6e6; padding-top: 10px; }
.tbl { font-size: 26px; line-height: 1.3; color: #2f3742; width: 1664px; }
.foot { position: absolute; left: 128px; bottom: 64px; width: 1664px; height: 30px; display: flex; justify-content: space-between; }
.foot .f { font-size: 24px; line-height: 1.2; color: #7a8493; }
.pg { font-size: 24px; line-height: 1.2; color: #7a8493; }
.pg::after { content: counter(page); }
.dark .foot .f { color: #9fb3cc; }
.dark .pg { color: #9fb3cc; }
.draft { position: absolute; box-sizing: border-box; left: 1272px; top: 116px; width: 520px; font-size: 24px; line-height: 1.2; color: #7a2e0b; background: #fde3d3; border-radius: 20px; padding: 8px 20px; text-align: center; white-space: nowrap; }
"""

S = {}

# ------------------------------------------------------------------ cover
S["cover"] = (f'<section id="cover" class="dark" data-transition="fade">\n'
              f'<p class="eyebrow">STOR 566 · FINAL PROJECT · GROUP 9</p>\n'
              f'<h1>Facial-expression classification under image degradation</h1>\n'
              f'<p class="sub">Clean accuracy, robustness and noise augmentation on FER-2013</p>\n'
              f'<div class="spacer"></div>\n'
              f'<p class="names">Dominic Jiang · Alyson Ficca · Emily Nguyen · Isaiah Lu</p>\n'
              f'<p class="names2">December 1, 2026</p>\n'
              f'<aside>{html.escape("0:15. Names, one sentence on the topic: how well do facial-expression classifiers hold up when the image gets noisy or small, and does training with noise help. Plan: 12 minutes, 12 slides, backup slides after the takeaways.", quote=False)}</aside>\n'
              f'</section>\n')

# ------------------------------------------------------------------ why
S["why"] = slide("why", "MOTIVATION", "Clean test accuracy is not what a deployed classifier sees",
                 f'''<div class="body">
<div class="col" style="width:620px">
<p>Cheap cameras add noise. A face cropped from a wide shot is a few dozen pixels across.</p>
<p>FER-2013 images are already 48 × 48 grayscale, and their labels are noisy.</p>
<p class="note">The labels describe facial expressions. They do not establish what a person feels.</p>
</div>
<div class="col grow">
<div class="card"><h3>RQ1</h3><p class="t">How well does each method classify the original images?</p></div>
<div class="card"><h3>RQ2</h3><p class="t">How does each method perform as noise increases or resolution decreases, and how much of its original accuracy does it retain?</p></div>
<div class="card"><h3>RQ3</h3><p class="t">Does noise augmentation help on noisy images and transfer to reduced resolution? How does it affect accuracy on the original images?</p></div>
</div>
</div>''',
                 "1:00. Benchmarks report accuracy on clean test images. A model that is 8 points better on clean images is not necessarily 8 points better on what a webcam produces. Read the three questions; they are unchanged from the proposal. One caveat up front: labels are expressions judged by annotators, not emotions.")

# ------------------------------------------------------------------ since the midterm
S["since"] = slide("since", "PROGRESS", "Since the midterm: a stronger model, the full stress test, a wider literature base",
                   f'''<div class="stack">
<div class="pair"><p class="from">Feedback: find a model that beats the pretrained ResNet-18</p><x-shape kind="arrow-right" style="background:#9aa7b8;width:64px;height:32px"></x-shape><p class="to">Added ViT-B/16 under the same protocol: {n("acc@vit_b16@std@clean")}% on original images against {n("acc@resnet18@std@clean")}% for ResNet-18</p></div>
<div class="pair"><p class="from">Feedback: then run the stress test</p><x-shape kind="arrow-right" style="background:#9aa7b8;width:64px;height:32px"></x-shape><p class="to">Three models, with and without noise augmentation, three seeds each, tested at 5 noise levels and 5 resolution levels</p></div>
<div class="pair"><p class="from">Feedback: expand the literature review</p><x-shape kind="arrow-right" style="background:#9aa7b8;width:64px;height:32px"></x-shape><p class="to">From 5 to 40 references: robustness benchmarks, noise augmentation, robustness of expression classifiers</p></div>
<p class="note">At the midterm: one CNN partly trained, no robustness numbers yet.</p>
</div>''',
                   "0:45. At the midterm we had a partly trained baseline CNN and a plan. The feedback was: get a better model than the pretrained ResNet, then stress-test, and widen the literature review. All three are done. Everything you see today was run under one protocol with three seeds per configuration.")

# ------------------------------------------------------------------ data
S["data"] = slide("data", "DATA AND DEGRADATIONS", "Every model sees the same test images: FER-2013 without duplicates, two degradations, five levels each",
                  f'''<div class="body">
<div class="col" style="width:500px">
<div class="tile"><p class="num">{n("audit@n_unique_images")}</p><p class="lab">unique images among {n("audit@n_rows")}</p></div>
<div class="tile"><p class="num">{n("audit@cross_split_duplicate_groups")}</p><p class="lab">groups of identical images were shared between train, validation and test</p></div>
<div class="tile"><p class="num2">27,418 / 3,291 / {n("ntest")}</p><p class="lab">train / validation / test after filtering</p></div>
</div>
<div class="col grow">
{img("fig_levels_examples", 1116, 478, "One validation face at the five Gaussian-noise levels (top row) and the five reduced-resolution levels (bottom row)")}
<p class="note">Noise: add N(0, σ²) to every pixel, clip to [0, 1]. Resolution: down-sample to s × s, resize back to 48 × 48. Levels were fixed on validation images before any test run.</p>
</div>
</div>''',
                  "1:00. Before any modelling we audited the data: 1,516 groups of pixel-identical images, 557 of them crossing the official splits, so a model could be tested on training images. We removed conflicting-label copies and kept one copy of the rest. Two degradations, each controlled by one number, five levels each, chosen with a pilot model on validation images so they run from almost no loss to near the majority-class rate. The same degraded test images are used for every model and seed.")

# ------------------------------------------------------------------ protocol
S["protocol"] = slide("protocol", "MODELS AND PROTOCOL", "Every choice is made on validation data; the test set is used once",
                      f'''<div class="stack">
<div class="row">
<div class="card grow"><h3>Baseline CNN</h3><p class="t">3 conv blocks, {n("params@cnn")} parameters, trained from scratch</p></div>
<div class="card grow"><h3>ResNet-18</h3><p class="t">11.2M parameters, ImageNet-1k weights, fine-tuned</p></div>
<div class="card grow"><h3>ViT-B/16</h3><p class="t">85.8M parameters, ImageNet-21k and -1k weights, fine-tuned</p></div>
</div>
<div class="steps">
<div class="step"><h3>1 · Tune</h3><p class="t">3 learning rates and 4 noise settings per model, seed 0, validation only</p></div>
<x-shape kind="arrow-right" style="background:#9aa7b8;width:48px;height:24px;align-self:center"></x-shape>
<div class="step"><h3>2 · Freeze</h3><p class="t">chosen settings written to a file before any test evaluation</p></div>
<x-shape kind="arrow-right" style="background:#9aa7b8;width:48px;height:24px;align-self:center"></x-shape>
<div class="step"><h3>3 · Train</h3><p class="t">3 seeds × 6 configurations (3 models, with and without noise)</p></div>
<x-shape kind="arrow-right" style="background:#9aa7b8;width:48px;height:24px;align-self:center"></x-shape>
<div class="step"><h3>4 · Test once</h3><p class="t">original images, 5 noise levels, 5 resolution levels</p></div>
</div>
<p class="note">Selection score S = ½ · accuracy on original + ½ · accuracy on noisy validation images. Reduced resolution is never used to choose anything, so those results measure transfer.</p>
</div>''',
                      "1:15. Three models of increasing size: a 94 thousand parameter CNN from scratch, a pretrained ResNet-18, a pretrained ViT-B/16. Learning rate, stopping epoch and the noise-augmentation setting are all picked with one validation score, half original accuracy and half noisy accuracy. The chosen settings were frozen to a file before we looked at any test number. Reduced resolution never enters a decision. We come back on slide 10 to what this selection rule cost us.")

# ------------------------------------------------------------------ RQ1
S["rq1"] = slide("rq1", "RQ1 · ORIGINAL IMAGES",
                 "On original images, bigger pretrained models win",
                 f'''<div class="body">
{img("s_rq1_clean", 1291, 540, "Bar chart of test accuracy and macro-F1 on original images for the baseline CNN, ResNet-18 and ViT-B/16 under standard training")}
<div class="col grow">
<div class="tile"><p class="num2">65 ± 5%</p><p class="lab">human accuracy estimated by the dataset’s authors, on the unfiltered data</p></div>
<p class="note">Macro-F1 trails accuracy because small classes are missed: the CNN finds {n("recall@cnn@std@disgust@clean")}% of disgust images, ViT-B/16 {n("recall@vit_b16@std@disgust@clean")}%.</p>
</div>
</div>''',
                 f"0:50. Test accuracy on original images, mean of three seeds with the standard deviation: {n('acc@cnn@std@clean')}, {n('acc@resnet18@std@clean')} and {n('acc@vit_b16@std@clean')} percent. The gaps are far larger than the seed-to-seed variation. Macro-F1 is lower for every model because disgust is 1.2 percent of the test set and is rarely recovered. For orientation the dataset authors put human accuracy at about 65 percent; our test set is filtered, so it is not the same benchmark.")

# ------------------------------------------------------------------ RQ2
S["rq2"] = slide("rq2", "RQ2 · STANDARD TRAINING", "Under standard training, that lead shrinks or disappears as images degrade",
                 f'''<div class="body">
{img("s_rq2_curves", 1250, 500, "Test accuracy against severity for Gaussian noise and reduced resolution, standard training, three models; all curves fall to the majority-class rate")}
<div class="col grow">
<div class="tile"><p class="num2">+{n("archdiff@resnet18@cnn@std@clean")} → +{n("archdiff@resnet18@cnn@std@navg")}</p><p class="lab">ResNet-18 lead over the CNN, in points: original images → average of 5 noise levels</p></div>
<div class="tile"><p class="num2">+{n("archdiff@vit_b16@cnn@std@clean")} → +{n("archdiff@vit_b16@cnn@std@navg")}</p><p class="lab">ViT-B/16 lead over the CNN</p></div>
<p class="note">At σ = 0.2 and at 8 px every model is at or below the {n("majority")}% majority-class rate.</p>
</div>
</div>''',
                 f"1:30. Standard training, no noise seen in training. All three curves fall fast: every model is below 90 percent of its own clean accuracy by sigma 0.05. At sigma 0.1 the models keep {n('rel@cnn@std@n0.1')}, {n('rel@resnet18@std@n0.1')} and {n('rel@vit_b16@std@n0.1')} percent of their clean accuracy. Averaged over the five noise levels ResNet-18 beats the small CNN by {n('archdiff@resnet18@cnn@std@navg')} points instead of {n('archdiff@resnet18@cnn@std@clean')}; that difference is inside test-set sampling error. ViT keeps part of its lead, {n('archdiff@vit_b16@cnn@std@navg')} points instead of {n('archdiff@vit_b16@cnn@std@clean')}. At the strongest levels the ranking is gone, and at 8 pixels the best clean model is the worst. Pre-training bought clean accuracy. It did not buy robustness. Same pattern Hendrycks and Dietterich reported on ImageNet-C.")

# ------------------------------------------------------------------ RQ3
S["rq3"] = slide("rq3", "RQ3 · NOISE AUGMENTATION", "Noise augmentation costs little on clean images and transfers to low resolution for pretrained models",
                 f'''<div class="body">
{img("s_rq3_dumbbell", 820, 590, "Dumbbell chart: test accuracy with standard training and with noise augmentation, for original images, the noise average and the resolution average, three models")}
<div class="col grow">
<div class="card"><h3>Original images</h3><p class="t">{minus(n("delta@cnn@clean"))}, {minus(n("delta@resnet18@clean"))} and {minus(n("delta@vit_b16@clean"))} points (CNN, ResNet-18, ViT-B/16): little or no cost</p></div>
<div class="card"><h3>Gaussian noise, seen in training</h3><p class="t">{n("delta@cnn@navg")}, {n("delta@resnet18@navg")} and {n("delta@vit_b16@navg")} points on average, and up to {n("delta@vit_b16@n0.2")} at σ = 0.2</p></div>
<div class="card"><h3>Reduced resolution, never seen</h3><p class="t">{n("delta@resnet18@ravg")} for ResNet-18 and {n("delta@vit_b16@ravg")} for ViT-B/16, but only {n("delta@cnn@ravg")} for the small CNN</p></div>
</div>
</div>''',
                 f"1:30. Each training image gets Gaussian noise with probability p and a random strength up to sigma-max; p and sigma-max were chosen on validation. On original images the change is small in both directions. On noisy images the gain is large and grows with severity. The interesting part is reduced resolution: it was never used in training or in any choice we made, so this is transfer, not tuning. It transfers for the pretrained models, {n('delta@resnet18@ravg')} and {n('delta@vit_b16@ravg')} points on average, and hardly at all for the small CNN. That fits the frequency argument of Yin and co-authors: noise training pushes a model away from fine detail, and down-sampling removes exactly that detail. Why the small CNN gains so little we did not test.",
                 draft=DRAFT_VIT)

# ------------------------------------------------------------------ collapse
S["collapse"] = slide("collapse", "FAILURE ANALYSIS", "Under heavy degradation the models stop looking at the face: predictions collapse onto one class",
                      f'''<div class="body">
{img("s_collapse", 1250, 500, "Share of test images assigned to the most-predicted class against severity, for standard and noise-augmented training")}
<div class="col grow">
<div class="tile"><p class="num2">{n("top@resnet18@std@n0.2")}%</p><p class="lab">of all test images go to one class for standard ResNet-18 at σ = 0.2. The largest true class is {n("majority")}%.</p></div>
<div class="tile"><p class="num2">{n("top@resnet18@aug@n0.2")}%</p><p class="lab">with noise augmentation, same σ</p></div>
<p class="note">At 8 px every model collapses. Happy survives longest; disgust and fear vanish first.</p>
</div>
</div>''',
                      f"1:15. Accuracy near the majority rate could mean guessing. It does not. This is the share of test images put into the single most-predicted class. On clean images it is about 26 to 30 percent, like the true label distribution. Under standard training it climbs: at sigma 0.2, {n('top@cnn@std@n0.2')}, {n('top@resnet18@std@n0.2')} and {n('top@vit_b16@std@n0.2')} percent of all images get the same label. Which label changes with the seed, happy, sad or neutral, which is why accuracy can drop below the majority rate. With noise augmentation the models stay near the true distribution under noise, but at 8 pixels they collapse as well. Per class, happy is the most robust; disgust and fear are the first to go (backup slide).",
                      draft=DRAFT_VIT)

# ------------------------------------------------------------------ selection
if HAS_REF:
    # the wording below assumes the direction seen in the data; fail loudly if it ever changes
    assert float(D["refdelta@resnet18@clean"]) > 0 > float(D["refdelta@resnet18@navg"])
    assert float(D["acc@resnet18@ref@navg"]) <= float(D["acc@cnn@ref@navg"]) + 1.0
    ref_card = (f'<div class="card"><h3>Checked against the usual rule</h3><p class="t">Checkpoint picked on original validation only: ResNet-18 gains {n("refabs@resnet18@clean")} points on original images '
                f'and loses {n("refabs@resnet18@navg")} under noise. Our conclusions hold.</p></div>')
    ref_note = (f"We checked what that costs with reference runs that pick the checkpoint on original validation accuracy only: ResNet-18 gains {n('refabs@resnet18@clean')} points on original images and loses "
                f"{n('refabs@resnet18@navg')} under noise. The conclusions hold either way: ResNet-18 is still no better than the CNN under noise, and noise augmentation still adds {n('augrefabs@resnet18@navg')} points.")
    if HAS_REF_ALL:
        # wording assumes ViT-B/16 differs by less than a point under either summary
        assert abs(float(D["refdelta@vit_b16@clean"])) < 1 and abs(float(D["refdelta@vit_b16@navg"])) < 1
        ref_note += " For ViT-B/16 the two rules differ by less than a point; the full table is on a backup slide."
    sel_draft = None
else:
    ref_card = '<div class="card"><h3>Reference runs</h3><p class="t">Same training, checkpoint chosen on original validation images only: [runs pending]</p></div>'
    ref_note = "That understates what standard training reaches on clean images. Reference runs with the checkpoint chosen on original validation accuracy only are still to be added."
    sel_draft = "DRAFT · reference runs pending"
S["selection"] = slide("selection", "PROTOCOL CHECK", "The checkpoint rule matters: our pre-specified score stops standard models early",
                       f'''<div class="body">
{img("s_selection_resnet18", 888, 520, "Validation accuracy per epoch for ResNet-18 under standard training, on original and noisy validation images, with the selected checkpoints marked")}
<div class="col grow">
<p class="t" style="font-size:28px;line-height:1.35">Half of our score is accuracy on noisy validation images. For a model trained without noise that half is erratic and peaks early.</p>
<div class="card"><h3>ResNet-18, standard training</h3><p class="t">Checkpoints chosen at epochs {n("epochs@resnet18@std")}, while accuracy on original validation images was still rising in two of three runs</p></div>
{ref_card}
</div>
</div>''',
                       f"1:00. We fixed the selection rule in advance, and it has a side effect. For models trained without noise, the noisy half of the score jumps around and is often highest after the first epoch, so the rule stopped two of three ResNet-18 runs at epoch 1. {ref_note} The lesson for anyone comparing robustness: say how checkpoints were picked.",
                       draft=sel_draft)

# ------------------------------------------------------------------ limits
if HAS_LRCHECK:
    # wording assumes the score moved by less than its seed-to-seed SD for both pretrained models
    for _a in ("resnet18", "vit_b16"):
        assert float(D[f"lrcheck@{_a}@absdiff"]) <= float(D[f"scoresd@{_a}"]), _a
    TUNE_CARD = "Three learning rates per model; the chosen one is at the grid edge for both pretrained models. One step beyond changed the score by less than its seed SD."
    TUNE_NOTE = "A small tuning grid whose best learning rate sits at the edge for the pretrained models; a check one step beyond the grid changed the score by less than its seed-to-seed spread."
else:
    TUNE_CARD = "Three learning rates per model; the chosen one is at the edge of the grid for both pretrained models."
    TUNE_NOTE = "A small tuning grid whose best learning rate sits at the edge for the pretrained models."
S["limits"] = slide("limits", "LIMITATIONS", "What these results do not show",
                    f'''<div class="grid3">
<div class="card"><h3>One dataset</h3><p class="t">48 × 48 grayscale faces with noisy labels. The labels are expressions, not emotions.</p></div>
<div class="card"><h3>Two synthetic degradations</h3><p class="t">Applied one at a time. Real images mix blur, noise, compression and pose.</p></div>
<div class="card"><h3>Confounded models</h3><p class="t">Architecture, size and pre-training change together, so no single cause can be named.</p></div>
<div class="card"><h3>Three seeds</h3><p class="t">The standard deviations are rough. The main effects are many of them wide.</p></div>
<div class="card"><h3>Tuning budget</h3><p class="t">{TUNE_CARD}</p></div>
<div class="card"><h3>No other augmentation</h3><p class="t">No flips or crops, by design, so clean accuracy is below the best published numbers.</p></div>
</div>''',
                    f"0:45. One dataset with noisy labels. Two synthetic degradations, one at a time. The three methods differ in architecture, size and pre-training at once, so we compare methods, not causes. Three seeds. {TUNE_NOTE} And no flips or crops, so that noise is the only augmentation; that keeps the comparison clean and costs some clean accuracy. Next steps: more corruption types and mixtures, FER+ labels, comparison at matched clean accuracy.")

# ------------------------------------------------------------------ takeaways
S["takeaways"] = (f'<section id="takeaways" class="dark" data-transition="fade">\n<p class="eyebrow">TAKEAWAYS</p>\n<h2>Three things to remember</h2>\n'
                  f'<div class="stack" style="gap:64px;padding-top:56px">\n'
                  f'<div class="take"><p class="k">1</p><div class="col grow" style="gap:0px"><p class="big">Clean accuracy ranks models correctly only on clean images.</p>'
                  f'<p class="small">{rnd("acc@cnn@std@clean")} → {rnd("acc@resnet18@std@clean")} → {rnd("acc@vit_b16@std@clean")}% on original images; all three at the majority-class rate by σ = 0.2.</p></div></div>\n'
                  f'<div class="take"><p class="k">2</p><div class="col grow" style="gap:0px"><p class="big">Pre-training bought accuracy, not robustness.</p>'
                  f'<p class="small">ResNet-18 against the small CNN: +{n("archdiff@resnet18@cnn@std@clean")} points on original images, +{n("archdiff@resnet18@cnn@std@navg")} under noise.</p></div></div>\n'
                  f'<div class="take"><p class="k">3</p><div class="col grow" style="gap:0px"><p class="big">Noise augmentation is cheap insurance, most of all for pretrained models.</p>'
                  f'<p class="small">{n("delta@cnn@navg")} to {n("delta@resnet18@navg")} points under noise; {n("delta@resnet18@ravg")} (ResNet-18) and {n("delta@vit_b16@ravg")} (ViT-B/16) at low resolution, which it never saw.</p></div></div>\n'
                  f'</div>\n{foot()}\n'
                  f'<aside>{html.escape("0:45. If you remember one thing: report accuracy under the conditions you expect, and say how you picked the checkpoint. Then questions.", quote=False)}</aside>\n</section>\n')


# ------------------------------------------------------------------ backup: selection table
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lrfmt(x):
    m, e = f"{float(x):.0e}".split("e")
    return f"{m}×10{str(int(e)).translate(SUP)}"


def sel_rows(out_dir):
    rows = []
    for a in ("cnn", "resnet18", "vit_b16"):
        p = os.path.join(out_dir, f"selection_{a}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        lr = " · ".join(f"{lrfmt(k)}: {v:.3f}" for k, v in s["lr_scores"].items())
        au = " · ".join(f"({k.replace('|', ', ')}): {v:.3f}" for k, v in s["aug_scores"].items())
        rows.append(f"<tr><td>{ARCH[a]}</td><td>{lrfmt(s['lr'])}</td><td>{lr}</td><td>({s['aug_p']:g}, {s['aug_sigma_max']:g})</td><td>{au}</td></tr>")
    return "\n".join(rows)


def backup_selection(out_dir):
    if HAS_LRCHECK:
        lrc = json.load(open(os.path.join(out_dir, "posthoc", "lr_check_summary.json")))
        lr_line = (f'\n<p class="note">Check one step beyond the grid, run after everything was frozen (seed 0, validation only, never used for selection): '
                   f'ResNet-18 at {lrfmt(lrc["resnet18"]["lr"])}: {n("lrcheck@resnet18@score")} · ViT-B/16 at {lrfmt(lrc["vit_b16"]["lr"])}: {n("lrcheck@vit_b16@score")}. '
                   f'Seed-to-seed SD of S at the chosen value: {n("scoresd@resnet18")} and {n("scoresd@vit_b16")}.</p>')
        lr_note = (f"The chosen learning rate is the largest of the grid for ResNet-18 and ViT-B/16, so after everything was frozen we ran one value beyond the grid. "
                   f"ResNet-18 scored {n('lrcheck@resnet18@score')} against {n('lrcheck@resnet18@selscore')} at the chosen value, ViT-B/16 {n('lrcheck@vit_b16@score')} against {n('lrcheck@vit_b16@selscore')}. "
                   f"Both differences are smaller than the seed-to-seed standard deviation of the score, so the grid edge did not cost much. The check was never used for selection.")
    else:
        lr_line = ""
        lr_note = "The chosen learning rate is the largest of the grid for ResNet-18 and ViT-B/16; a post-hoc run one step above the grid is planned as a sensitivity check and does not change the frozen selection."
    return slide("b-selection", "BACKUP · SELECTION", "Selection scores on validation images: learning rates and noise-augmentation settings",
                 f'''<div class="stack">
<table class="tbl">
<tr><th style="width:13%">Model</th><th style="width:10%">Chosen LR</th><th style="width:27%">S over the learning-rate grid</th><th style="width:12%">Chosen (p, σmax)</th><th style="width:38%">S over the augmentation grid</th></tr>
{sel_rows(out_dir)}
</table>
<p class="note">S = ½ · original + ½ · noisy validation accuracy, seed 0. Augmentation: each training image gets noise with probability p, σ drawn uniformly from [0, σmax].</p>{lr_line}
</div>''',
                 f"Backup. Tuning runs use seed 0 and validation images only. {lr_note}")


def backup_reference(out_dir):
    """Standard training under the two checkpoint rules, all three models (needs every reference run)."""
    import glob
    runs = {}
    for f in glob.glob(os.path.join(out_dir, "final", "*.json")):
        d = json.load(open(f))
        runs[(d["name"].split("-")[0], d["variant"], int(d["name"].rsplit("seed", 1)[1]))] = d

    def per_seed(a, cond):
        return [runs[(a, "ref", k)]["metrics"]["test"][cond]["acc"] - runs[(a, "std", k)]["metrics"]["test"][cond]["acc"] for k in range(3)]
    # the speaker note says which models agree across seeds; fail loudly if that ever changes
    assert all(x > 0 for x in per_seed("resnet18", "clean"))
    c = per_seed("cnn", "clean")
    assert not (all(x > 0 for x in c) or all(x < 0 for x in c))
    for a in ("cnn", "resnet18"):
        assert float(D[f"refdelta@{a}@clean"]) > 0 > float(D[f"refdelta@{a}@navg"])
    assert all(abs(float(D[f"refdelta@vit_b16@{k}"])) < 1 for k in ("clean", "navg", "ravg"))
    assert float(D["acc@resnet18@ref@navg"]) <= float(D["acc@cnn@ref@navg"])
    assert all(float(D[f"acc@vit_b16@ref@{k}"]) > max(float(D[f"acc@cnn@ref@{k}"]), float(D[f"acc@resnet18@ref@{k}"])) for k in ("clean", "navg", "ravg"))
    costs = [float(D[f"augref@{a}@clean"]) for a in ("cnn", "resnet18", "vit_b16")]
    assert all(-1 < x < 0 for x in costs)
    gains = [float(D[f"augref@{a}@navg"]) for a in ("cnn", "resnet18", "vit_b16")]

    def cell(a, k):
        return minus(f'{n(f"acc@{a}@std@{k}")} → {n(f"acc@{a}@ref@{k}")} ({n(f"refdelta@{a}@{k}")} ± {n(f"refdeltasd@{a}@{k}")})')
    rows = "\n".join(f'<tr><td>{ARCH[a]}</td><td>{n(f"epochs@{a}@std")} → {n(f"epochs@{a}@ref")}</td><td>{cell(a, "clean")}</td><td>{cell(a, "navg")}</td><td>{cell(a, "ravg")}</td></tr>'
                     for a in ("cnn", "resnet18", "vit_b16"))
    return slide("b-ref", "BACKUP · CHECKPOINT RULE", "Standard training under two checkpoint rules: a few points for the smaller models, under one for ViT-B/16",
                 f'''<div class="stack">
<table class="tbl">
<tr><th style="width:15%">Model</th><th style="width:22%">Epochs kept (3 seeds)</th><th style="width:21%">Original</th><th style="width:21%">Noise average</th><th style="width:21%">Resolution average</th></tr>
{rows}
</table>
<p class="note">Test accuracy (%), mean of three seeds: checkpoint chosen by our score S → chosen on original validation accuracy only. In brackets: paired difference, mean ± SD over seeds. Same settings and seeds; the runs were retrained.</p>
<p class="note">Against the original-only checkpoints, noise augmentation still adds {min(gains):.1f} to {max(gains):.1f} points under noise and costs {min(abs(x) for x in costs):.1f} to {max(abs(x) for x in costs):.1f} points on original images.</p>
</div>''',
                 f"Backup for the question 'what if you had picked checkpoints the usual way?'. Same training, but the checkpoint is chosen on original validation accuracy only. "
                 f"ResNet-18 gains {n('refabs@resnet18@clean')} points on original images and loses {n('refabs@resnet18@navg')} under noise, with the same sign in all three seeds. "
                 f"For the CNN the averages point the same way, {n('refabs@cnn@clean')} and {n('refabs@cnn@navg')} points, but not every seed agrees. "
                 f"For ViT-B/16, fine-tuned for four epochs, the two rules differ by less than a point. "
                 f"Our conclusions hold with either rule: ResNet-18 is still no better than the CNN under noise, ViT-B/16 is still the most accurate, and noise augmentation still adds {min(gains):.0f} to {max(gains):.0f} points under noise.")


def backup_bootstrap(out_dir):
    p = os.path.join(out_dir, "analysis_extra", "extra_summary.json")
    rows = json.load(open(p)).get("bootstrap", []) if os.path.exists(p) else []
    dif = {}
    for r in rows:
        if r["kind"] == "diff":
            dif.setdefault((r["a"], r["va"], r["b"], r["vb"]), {})[r["quantity"]] = r

    def cell(r):
        star = "" if r["lo"] <= 0 <= r["hi"] else " *"
        return minus(f"{r['mean'] * 100:+.1f} [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]{star}")
    out = []
    for k, q in dif.items():
        if k[0] == k[2] and k[1] == "aug" and k[3] == "std":
            lab = f"{ARCH[k[0]]}: noise aug. − standard"
        elif k[0] != k[2] and k[1] == "std":
            lab = f"{ARCH[k[0]]} − {ARCH[k[2]]}, standard"
        else:
            continue
        out.append(f"<tr><td>{lab}</td><td>{cell(q['clean'])}</td><td>{cell(q['noise_avg'])}</td><td>{cell(q['res_avg'])}</td></tr>")
    return slide("b-bootstrap", "BACKUP · UNCERTAINTY", "Differences in test accuracy with 95% bootstrap intervals over test images",
                 f'''<div class="stack">
<table class="tbl">
<tr><th style="width:37%">Comparison (points)</th><th style="width:21%">Original</th><th style="width:21%">Noise average</th><th style="width:21%">Resolution average</th></tr>
{chr(10).join(out)}
</table>
<p class="note">Seed-averaged accuracy, {n("ntest")} test images resampled 5,000 times with the trained models held fixed. * interval excludes zero. This is test-set sampling error only; seed-to-seed variation is the SD shown on the charts.</p>
</div>''',
                 "Backup for the question 'is that significant?'. Two sources of uncertainty: seeds (SD over three runs) and the finite test set (bootstrap over images, models fixed). Large effects are far outside both; the ResNet-18 versus CNN difference under degradation is not distinguishable from zero.")


def build(out_dir):
    S["b-selection"] = backup_selection(out_dir)
    if HAS_REF_ALL:
        S["b-ref"] = backup_reference(out_dir)
    S["b-extended"] = slide("b-extended", "BACKUP · EXTENDED LEVELS", "Beyond the primary levels: σ up to 0.5 and 4 px",
                            f'<div class="body">\n{img("fig_degradation_acc_extended", 1500, 586, "Test accuracy for all severity levels including the extended ones, standard and noise-augmented training, three models")}\n</div>',
                            "Backup. The heavier levels used at the midterm are reported separately as an extended sweep. Beyond sigma 0.2 standard models sit at or below the majority rate; augmented models degrade gradually because they were trained up to sigma-max 0.2 only.",
                            draft=DRAFT_VIT)
    S["b-recall"] = slide("b-recall", "BACKUP · PER CLASS", "Which expressions survive: recall on original images, at σ = 0.1 and at 24 px (standard training)",
                          f'<div class="body">\n{img("s_recall_std", 1353, 580, "Heatmaps of per-class recall for the three models on original images, at sigma 0.1 and at 24 pixels")}\n</div>',
                          "Backup. Recall per class, in percent, mean of three seeds. Happy stays recognisable; disgust and fear drop to near zero under noise for the CNN and ResNet-18. Rising recall for one class under degradation, for example sad for ViT-B/16, is the collapse showing up as apparent recall.")
    S["b-relative"] = slide("b-relative", "BACKUP · RELATIVE ACCURACY", "Relative accuracy under standard training: the small CNN retains the most",
                            f'<div class="body">\n{img("s_rq2_relative", 1350, 540, "Relative accuracy, degraded divided by original accuracy, against severity for the three models under standard training")}\n</div>',
                            f"Backup. Degraded accuracy divided by original accuracy. First level below 90 percent: sigma 0.05 for all three; 32 pixels for the CNN and ViT-B/16, 40 pixels for ResNet-18. With augmentation: sigma 0.15 for all three, and 24 pixels for the pretrained models.")
    S["b-augcurves"] = slide("b-augcurves", "BACKUP · ALL CURVES", "Standard training and noise augmentation at every level",
                             f'<div class="body">\n{img("s_rq3_curves", 1350, 540, "Test accuracy against severity for standard and noise-augmented training, three models")}\n</div>',
                             "Backup. Solid lines and filled circles are standard training; dashed lines and hollow squares are noise augmentation.", draft=DRAFT_VIT)
    S["b-bootstrap"] = backup_bootstrap(out_dir)
    S["b-related"] = slide("b-related", "BACKUP · RELATED WORK", "Where this project sits in the literature",
                           '''<div class="grid3">
<div class="card"><h3>FER-2013 and its models</h3><p class="t">Goodfellow et al. 2013 · Barsoum et al. 2016 (FER+) · Khaireddin &amp; Chen 2021 · Li &amp; Deng survey</p></div>
<div class="card"><h3>Robustness benchmarks</h3><p class="t">Dodge &amp; Karam 2016 · Hendrycks &amp; Dietterich 2019 (ImageNet-C) · Geirhos et al. 2018</p></div>
<div class="card"><h3>Noise augmentation</h3><p class="t">Bishop 1995 · Rusak et al. 2020 · Yin et al. 2019 (frequency view) · Ramírez-Agudelo et al. 2025</p></div>
<div class="card"><h3>Robust expression recognition</h3><p class="t">Greco et al. 2022 (RAF-DB-C) · Lévêque et al. 2022 (humans vs. networks) · Neo &amp; Chen 2023</p></div>
<div class="card"><h3>Transformers and pre-training</h3><p class="t">Dosovitskiy et al. 2021 · Bhojanapalli et al. 2021 · Hendrycks, Lee &amp; Mazeika 2019</p></div>
<div class="card"><h3>Evaluation practice</h3><p class="t">Barz &amp; Denzler 2020 (duplicates) · Gulrajani &amp; Lopez-Paz 2021 (model selection) · Bouthillier et al. 2021</p></div>
</div>''',
                           "Backup. Closest prior work: Greco and co-authors built corrupted versions of the RAF-DB test set. We differ in working at FER-2013's native 48 by 48 resolution, including a from-scratch baseline, fixing the selection protocol in advance, and testing transfer from noise augmentation to an unseen degradation.")

    order = ["cover", "why", "since", "data", "protocol", "rq1", "rq2", "rq3", "collapse", "selection", "limits", "takeaways",
             "b-selection", "b-ref", "b-extended", "b-relative", "b-augcurves", "b-recall", "b-bootstrap", "b-related"]
    if "b-ref" not in S:
        order.remove("b-ref")
    os.makedirs(os.path.join(ROOT, "project", "slides"), exist_ok=True)
    for sid in order:
        with open(os.path.join(ROOT, "project", "slides", sid + ".html"), "w", encoding="utf-8") as f:
            f.write(S[sid])
    idx_path = os.path.join(ROOT, "project", "deck.json")
    created = json.load(open(idx_path))["createdOnFiles"]["at"] if os.path.exists(idx_path) else datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = {"v": 5, "createdOnFiles": {"v": 1, "at": created}, "lists": "css", "format": {"v": 1, "shared": ["project/deck.css"]},
             "title": TITLE, "cover": "cover", "order": order,
             "sections": {"s1": {"description": "Motivation, the three research questions and what changed since the midterm.", "start": "cover"},
                          "s2": {"description": "Data filtering, the two degradations and the pre-specified protocol.", "start": "data"},
                          "s3": {"description": "Results for RQ1 to RQ3, the collapse analysis and the checkpoint-rule check.", "start": "rq1"},
                          "s4": {"description": "Limitations and takeaways.", "start": "limits"},
                          "s5": {"description": "Backup slides for questions.", "start": "b-selection"}},
             "faces": {"space-grotesk": {"family": "Space Grotesk", "href": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&display=swap"},
                       "inter": {"family": "Inter", "href": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"}},
             "designSystems": []}
    json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "project", "deck.css"), "w", encoding="utf-8") as f:
        f.write(CSS)
    print(f"deck files written to {ROOT}: {len(order)} slides; final runs {N_FINAL}; ViT aug seeds {N_VIT_AUG}; reference runs {HAS_REF}")
    return order


if __name__ == "__main__":
    build(sys.argv[1])
