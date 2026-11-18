"""
ferlib.py - STOR 566 Group 9
Facial-expression classification under image degradation (FER-2013).

One file holds the whole pipeline (data audit, models, degradations, training,
selection rule, evaluation) so that the workspace scripts and the Colab notebook
run exactly the same code.

Protocol source: Group 9 project proposal (Sep 2026). Every choice the proposal
left open is marked "DECISION" below so the group can review it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# FER-2013 label order (index = value of the "emotion" column)
FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
N_CLASSES = 7
IMG = 48

# ----------------------------------------------------------------------------
# Protocol constants. Frozen before any test-set evaluation.
# ----------------------------------------------------------------------------
# DECISION: five primary severity levels per degradation, fixed from the
# validation pilot (see pilot_levels.py). The extended levels reproduce the
# heavier settings shown at the midterm review and are reported separately.
NOISE_PRIMARY = [0.025, 0.05, 0.10, 0.15, 0.20]
NOISE_EXTENDED = [0.30, 0.40, 0.50]
RES_PRIMARY = [40, 32, 24, 16, 8]
RES_EXTENDED = [4]

EVAL_NOISE_SEED = 566009          # evaluation noise never shares a stream with training noise
SPLIT_SEED_OFFSET = {"train": 0, "val": 100_000, "test": 200_000}

# Training-set pixel statistics of the de-duplicated training split (pixels in [0,1]).
# Filled in by prep_data.py and asserted there; used only by the baseline CNN.
TRAIN_MEAN = 0.5060
TRAIN_STD = 0.2547

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Expected sizes and content hash after the proposal's duplicate filtering.
EXPECTED_SIZES = {"train": 27418, "val": 3291, "test": 3268}
EXPECTED_FINGERPRINT = "0c9741cfaef7236e98ba237a0d28bbf5094f2700fce0ae926fc3ec1ccbc2013a"            # order-independent
EXPECTED_ROW_ORDER_FINGERPRINT = "7f7e95f72057c7d0d75da2aaca44ab56b16954bb0b755304e0821f052f7753ae"  # informational

PRETRAINED = True   # set False only for smoke tests without internet

# Tuning grids and training budgets. Every architecture gets three learning rates and the same
# four augmentation settings (probability, sigma_max); optimiser details suit each model.
AUG_GRID = [(0.5, 0.1), (0.5, 0.2), (0.8, 0.1), (0.8, 0.2)]
ARCH_SETTINGS = {
    "cnn": dict(lr_grid=[3e-4, 1e-3, 3e-3], aug_grid=AUG_GRID, max_epochs=150, patience=15, batch_size=128),
    "resnet18": dict(lr_grid=[3e-5, 1e-4, 3e-4], aug_grid=AUG_GRID, max_epochs=12, patience=3, batch_size=128),
    # ViT recipe follows the group's earlier ViT notebook (AdamW, weight decay 0.01, 10% warm-up then
    # linear decay, batch 32); four epochs instead of five because validation accuracy peaked at epoch 3.
    "vit_b16": dict(lr_grid=[1e-5, 2e-5, 5e-5], aug_grid=AUG_GRID, max_epochs=4, patience=4, batch_size=32,
                    extra=dict(optimizer="adamw", weight_decay=0.01, schedule="linear")),
}
VIT_CHECKPOINT = "google/vit-base-patch16-224"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
SPLIT_OF = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
PRIORITY = {"train": 0, "val": 1, "test": 2}


def read_fer_csv(path):
    """Read fer2013.csv (emotion,pixels,Usage) or icml_face_data.csv (emotion, Usage, pixels)."""
    import pandas as pd

    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    missing = {"emotion", "pixels", "Usage"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {missing}; found {list(df.columns)}")
    y = df["emotion"].to_numpy(np.int64)
    usage = np.array([str(u).strip() for u in df["Usage"]])
    X = np.empty((len(df), IMG * IMG), np.uint8)
    for i, p in enumerate(df["pixels"].to_numpy()):
        X[i] = np.array(p.split(), dtype=np.uint8)
    return X, y, usage


def audit_and_dedup(X, y, usage):
    """Proposal rule:
    (1) remove every copy of a pixel-identical image whose copies carry conflicting labels;
    (2) for the remaining duplicate groups keep one copy, priority train > val > test
        (first occurrence inside the chosen split).
    Returns {split: (X uint8 (N,48,48), y int64, original_row_index)} and audit statistics.
    """
    split = np.array([SPLIT_OF[u] for u in usage])
    groups = {}
    for i in range(len(X)):
        groups.setdefault(X[i].tobytes(), []).append(i)
    keep = np.zeros(len(X), bool)
    st = dict(n_rows=int(len(X)), n_unique_images=len(groups), duplicate_groups=0,
              cross_split_duplicate_groups=0, conflicting_label_groups=0,
              rows_removed_conflicting=0, rows_removed_redundant=0)
    for idxs in groups.values():
        if len(idxs) == 1:
            keep[idxs[0]] = True
            continue
        st["duplicate_groups"] += 1
        if len({split[i] for i in idxs}) > 1:
            st["cross_split_duplicate_groups"] += 1
        if len({int(y[i]) for i in idxs}) > 1:
            st["conflicting_label_groups"] += 1
            st["rows_removed_conflicting"] += len(idxs)
            continue
        best = min(idxs, key=lambda i: (PRIORITY[split[i]], i))
        keep[best] = True
        st["rows_removed_redundant"] += len(idxs) - 1
    out = {}
    for s in ("train", "val", "test"):
        m = keep & (split == s)
        out[s] = (X[m].reshape(-1, IMG, IMG).copy(), y[m].astype(np.int64), np.nonzero(m)[0])
    st["raw_sizes"] = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    st["final_sizes"] = {s: int(len(out[s][1])) for s in out}
    return out, st


def data_fingerprint(data):
    """Content hash of the three splits. Independent of row order, so fer2013.csv and
    icml_face_data.csv (same images, different column layout) give the same value."""
    h = hashlib.sha256()
    for s in ("train", "val", "test"):
        X, y = data[s][0], data[s][1]
        rows = sorted(hashlib.md5(np.ascontiguousarray(X[i]).tobytes() + bytes([int(y[i])])).digest()
                      for i in range(len(y)))
        h.update(s.encode())
        h.update(b"".join(rows))
    return h.hexdigest()


def row_order_fingerprint(data):
    """Order-dependent hash: equal values mean identical arrays (and identical noise realisations)."""
    h = hashlib.sha256()
    for s in ("train", "val", "test"):
        h.update(np.ascontiguousarray(data[s][0]).tobytes())
        h.update(np.ascontiguousarray(data[s][1].astype(np.int64)).tobytes())
    return h.hexdigest()


def save_data(data, path):
    np.savez_compressed(path, **{f"{k}_{s}": v for s in data for k, v in zip(("X", "y", "idx"), data[s])})


def load_data(path):
    z = np.load(path)
    return {s: (z[f"X_{s}"], z[f"y_{s}"], z[f"idx_{s}"]) for s in ("train", "val", "test")}


def class_counts(y):
    return np.bincount(y, minlength=N_CLASSES)


# ----------------------------------------------------------------------------
# Degradations (applied to 48x48 images scaled to [0,1], before any model preprocessing)
# ----------------------------------------------------------------------------
def to_float(x_uint8):
    """uint8 (N,48,48) -> float32 tensor (N,1,48,48) in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).float().div_(255.0).unsqueeze(1)


def gaussian_noise(x01, sigma, seed):
    """Independent zero-mean Gaussian draw with standard deviation sigma per pixel, clipped to [0,1].
    The legacy RandomState stream is frozen by NumPy, so the corrupted images are identical on
    every machine and for every model and seed."""
    if sigma == 0:
        return x01.clone()
    rs = np.random.RandomState(seed % (2 ** 32))
    z = torch.from_numpy(rs.standard_normal(size=tuple(x01.shape)).astype(np.float32))
    return (x01 + float(sigma) * z).clamp_(0.0, 1.0)


def reduce_resolution(x01, s):
    """Bilinear downsampling with antialiasing to s x s, then bilinear resizing back to 48 x 48."""
    if s >= IMG:
        return x01.clone()
    lo = F.interpolate(x01, size=(s, s), mode="bilinear", antialias=True, align_corners=False)
    hi = F.interpolate(lo, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return hi.clamp_(0.0, 1.0)


def noise_seed(split, sigma):
    return EVAL_NOISE_SEED + SPLIT_SEED_OFFSET[split] + int(round(sigma * 1000))


def cond_name(kind, level=None):
    if kind == "clean":
        return "clean"
    return f"noise_{level:g}" if kind == "noise" else f"res_{int(level)}"


def all_conditions(extended=True):
    conds = [("clean", None)]
    conds += [("noise", s) for s in NOISE_PRIMARY + (NOISE_EXTENDED if extended else [])]
    conds += [("res", s) for s in RES_PRIMARY + (RES_EXTENDED if extended else [])]
    return conds


def make_condition(x01, split, kind, level=None):
    if kind == "clean":
        return x01
    if kind == "noise":
        return gaussian_noise(x01, level, noise_seed(split, level))
    if kind == "res":
        return reduce_resolution(x01, int(level))
    raise ValueError(kind)


def mixed_noise_set(x01, split):
    """Noisy copy used for model selection: every image is corrupted at one of the five primary
    noise levels (fixed, balanced assignment). One pass per epoch instead of five."""
    n = x01.shape[0]
    rs = np.random.RandomState((EVAL_NOISE_SEED + SPLIT_SEED_OFFSET[split] + 7) % (2 ** 32))
    level_idx = rs.permutation(n) % len(NOISE_PRIMARY)
    sig = torch.tensor(np.array(NOISE_PRIMARY, np.float32)[level_idx]).view(n, 1, 1, 1)
    z = torch.from_numpy(rs.standard_normal(size=tuple(x01.shape)).astype(np.float32))
    return (x01 + sig * z).clamp_(0.0, 1.0)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Proposal baseline: three 3x3 conv blocks (32, 64, 128), each followed by ReLU and 2x2 max
    pooling, then global average pooling and a seven-class linear layer (93,575 parameters)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128, N_CLASSES)

    def forward(self, x01):
        # DECISION: inputs are standardised with the training-set mean/std (a fixed affine map).
        x = (x01 - TRAIN_MEAN) / TRAIN_STD
        return self.classifier(self.features(x).mean(dim=(2, 3)))


class ResNet18FER(nn.Module):
    """torchvision ResNet-18 with ImageNet-1K weights, seven-class head, whole network fine-tuned.
    Input: grayscale copied to three channels, resized to 224x224, ImageNet normalisation."""

    def __init__(self, pretrained=True, weights_path=None):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        if weights_path:
            net = resnet18(weights=None)
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, N_CLASSES)
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x01):
        x = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.net(x)


class ViTB16FER(nn.Module):
    """ViT-B/16 (google/vit-base-patch16-224: ImageNet-21k pre-training, ImageNet-1k fine-tuning) with a
    seven-class head, whole network fine-tuned. Same input handling as the checkpoint's image processor:
    grayscale copied to three channels, bilinear resize to 224x224, mean 0.5 / std 0.5 normalisation."""

    def __init__(self, pretrained=True):
        super().__init__()
        from transformers import ViTConfig, ViTForImageClassification

        if pretrained:
            self.net = ViTForImageClassification.from_pretrained(VIT_CHECKPOINT, num_labels=N_CLASSES,
                                                                 ignore_mismatched_sizes=True)
        else:
            self.net = ViTForImageClassification(ViTConfig(num_labels=N_CLASSES))   # defaults = ViT-B/16, 224

    def forward(self, x01):
        x = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - 0.5) / 0.5
        return self.net(pixel_values=x).logits


def build_model(arch, pretrained=True, weights_path=None):
    if arch == "cnn":
        return BaselineCNN()
    if arch == "resnet18":
        return ResNet18FER(pretrained=pretrained, weights_path=weights_path)
    if arch == "vit_b16":
        return ViTB16FER(pretrained=pretrained)
    raise ValueError(arch)


EVAL_BATCH = {"cnn": 512, "resnet18": 256, "vit_b16": 128}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(y, p):
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    recall = tp / np.maximum(support, 1)
    f1 = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return dict(acc=float(tp.sum() / cm.sum()), macro_f1=float(f1.mean()),
                recall=[float(v) for v in recall], f1=[float(v) for v in f1], cm=cm.tolist())


@torch.no_grad()
def predict(model, x01, device, batch_size=512, amp=False):
    model.eval()
    out = []
    for i in range(0, x01.shape[0], batch_size):
        xb = x01[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy()


def evaluate_split(model, data, split, device, conditions=None, batch_size=512, amp=False):
    """Metrics and predictions for one split under every condition."""
    conditions = conditions or all_conditions()
    x01, y = to_float(data[split][0]), data[split][1]
    res, preds = {}, {}
    for kind, level in conditions:
        name = cond_name(kind, level)
        p = predict(model, make_condition(x01, split, kind, level), device, batch_size, amp)
        res[name] = compute_metrics(y, p)
        preds[name] = p.astype(np.uint8)
    return res, preds


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    arch: str
    lr: float
    seed: int
    aug_p: float = 0.0           # probability that a training image receives Gaussian noise
    aug_sigma_max: float = 0.0   # for a noised image, sigma ~ Uniform(0, aug_sigma_max)
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 10
    select_on: str = "clean+noisy"   # "clean+noisy" (proposal) or "clean" (level-setting pilot only)
    optimizer: str = "adam"          # "adam" or "adamw"
    weight_decay: float = 0.0
    schedule: str = "constant"       # "constant", or "linear": 10% linear warm-up then linear decay to zero

    @property
    def variant(self):
        if self.aug_p > 0:
            return "aug"
        return "ref" if self.select_on == "clean" else "std"

    def name(self):
        aug = f"-p{self.aug_p:g}-s{self.aug_sigma_max:g}" if self.aug_p > 0 else ""
        return f"{self.arch}-{self.variant}-lr{self.lr:g}{aug}-seed{self.seed}"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selection_sets(data):
    """Validation tensors used for LR / stopping / augmentation selection. Reduced-resolution
    images are deliberately absent: they never guide model or augmentation selection."""
    xv = to_float(data["val"][0])
    rs = np.random.RandomState(EVAL_NOISE_SEED % (2 ** 32))
    sub = np.sort(rs.choice(len(data["train"][1]), size=len(data["val"][1]), replace=False))
    return dict(val_clean=xv, val_noisy=mixed_noise_set(xv, "val"), y_val=data["val"][1],
                train_sub=to_float(data["train"][0][sub]), y_train_sub=data["train"][1][sub])


def train_model(cfg: TrainConfig, data, device, sel=None, log=print, weights_path=None, ckpt_path=None):
    """Train one model. Returns (best_state_dict_on_cpu, info) where the checkpoint is the epoch
    with the highest selection score on validation images. If ckpt_path is given, the full training
    state is saved after every epoch and an interrupted run continues from the last finished epoch."""
    sel = sel or selection_sets(data)
    seed_everything(cfg.seed)
    model = build_model(cfg.arch, pretrained=PRETRAINED, weights_path=weights_path).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    amp = device.type == "cuda"
    opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    sched = None
    if cfg.schedule == "linear":
        total = cfg.max_epochs * math.ceil(len(data["train"][1]) / cfg.batch_size)
        warm = int(0.1 * total)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: k / max(1, warm) if k < warm else max(0.0, (total - k) / max(1, total - warm)))

    xtr = torch.from_numpy(np.ascontiguousarray(data["train"][0])).to(device)
    ytr = torch.from_numpy(data["train"][1]).to(device)
    n = xtr.shape[0]
    g_order = torch.Generator().manual_seed(cfg.seed)                     # batch order: same on any device
    g_noise = torch.Generator(device=device).manual_seed(cfg.seed + 7919)  # training noise only
    eval_bs = EVAL_BATCH[cfg.arch]

    best = dict(score=-1.0, epoch=0, state=None)
    history = []
    t0 = time.time()
    start_epoch, elapsed_before = 1, 0.0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["name"] == cfg.name():
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            if sched is not None and ck.get("sched") is not None:
                sched.load_state_dict(ck["sched"])
            g_order.set_state(ck["g_order"])
            g_noise.set_state(ck["g_noise"].to(device) if device.type == "cuda" else ck["g_noise"])
            best, history, elapsed_before = ck["best"], ck["history"], ck["elapsed"]
            start_epoch = len(history) + 1
            log(f"[{cfg.name()}] resumed after epoch {len(history)}")
    stopped = bool(history) and (history[-1]["epoch"] - best["epoch"] >= cfg.patience)
    for epoch in ([] if stopped else range(start_epoch, cfg.max_epochs + 1)):
        model.train()
        perm = torch.randperm(n, generator=g_order).to(device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        te = time.time()
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x = xtr[idx].float().div_(255.0).unsqueeze(1)
            y = ytr[idx]
            if cfg.aug_p > 0:
                b = x.shape[0]
                on = (torch.rand(b, generator=g_noise, device=device) < cfg.aug_p).float()
                sig = torch.rand(b, generator=g_noise, device=device) * cfg.aug_sigma_max
                z = torch.randn(x.shape, generator=g_noise, device=device)
                x = (x + (on * sig).view(b, 1, 1, 1) * z).clamp_(0.0, 1.0)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            loss_sum += loss.detach().float() * len(idx)
            correct += (logits.argmax(1) == y).sum()
        acc_c = float((predict(model, sel["val_clean"], device, eval_bs, amp) == sel["y_val"]).mean())
        acc_n = float((predict(model, sel["val_noisy"], device, eval_bs, amp) == sel["y_val"]).mean())
        acc_t = float((predict(model, sel["train_sub"], device, eval_bs, amp) == sel["y_train_sub"]).mean())
        score = acc_c if cfg.select_on == "clean" else 0.5 * (acc_c + acc_n)
        history.append(dict(epoch=epoch, train_loss=float(loss_sum) / n, train_acc_running=float(correct) / n,
                            train_acc_eval=acc_t, val_acc_clean=acc_c, val_acc_noisy=acc_n, score=score,
                            epoch_seconds=round(time.time() - te, 1)))
        flag = ""
        if score > best["score"] + 1e-12:
            best = dict(score=score, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            flag = " *"
        log(f"[{cfg.name()}] ep {epoch:3d} loss {history[-1]['train_loss']:.4f} "
            f"train {acc_t:.4f} val {acc_c:.4f} val-noisy {acc_n:.4f} score {score:.4f} "
            f"({history[-1]['epoch_seconds']:.0f}s){flag}")
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            torch.save(dict(name=cfg.name(), model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                            sched=sched.state_dict() if sched is not None else None,
                            g_order=g_order.get_state(), g_noise=g_noise.get_state().cpu(), best=best, history=history,
                            elapsed=elapsed_before + time.time() - t0), tmp)
            os.replace(tmp, ckpt_path)
        if epoch - best["epoch"] >= cfg.patience:
            break
    info = dict(config=asdict(cfg), name=cfg.name(), variant=cfg.variant, best_epoch=best["epoch"],
                best_score=best["score"], epochs_run=len(history), history=history,
                train_seconds=round(elapsed_before + time.time() - t0, 1), n_params=count_params(model),
                device=str(device), torch_version=torch.__version__)
    return best["state"], info


# ----------------------------------------------------------------------------
# Experiment runner (resumable: every finished run is a JSON file)
# ----------------------------------------------------------------------------
class Runner:
    HEAVY = ("vit_b16",)   # large models: no per-epoch state files, no saved checkpoints, tuning run reused

    def __init__(self, out_dir, data, device, log_file=None, weights_path=None, save_checkpoints=True,
                 partial_dir=None, reuse_dir=None):
        """partial_dir: if set, every run saves its training state there after each epoch and an
        interrupted run continues from its last finished epoch (use a fast local disk).
        reuse_dir: if set, the best weights of each tuning run of a HEAVY model are kept there, and the
        seed-0 final run with the selected setting (same configuration and seed) reuses them
        instead of training the identical run a second time."""
        self.out, self.data, self.device = out_dir, data, device
        self.weights_path, self.save_checkpoints = weights_path, save_checkpoints
        self.partial_dir, self.reuse_dir = partial_dir, reuse_dir
        for d in ("tune", "final", "ckpt") + tuple(x for x in (partial_dir, reuse_dir) if x):
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        self.log_file = log_file or os.path.join(out_dir, "log.txt")
        self.sel = selection_sets(data)
        self.fingerprint = data_fingerprint(data)

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def _path(self, stage, cfg):
        return os.path.join(self.out, stage, cfg.name() + ".json")

    def _train(self, stage, cfg):
        use_partial = self.partial_dir and cfg.arch not in self.HEAVY
        ck = os.path.join(self.partial_dir, f"{stage}-{cfg.name()}.pt") if use_partial else None
        state, info = train_model(cfg, self.data, self.device, self.sel, self.log, self.weights_path, ck)
        return state, info, ck

    @staticmethod
    def _done(ck):
        if ck and os.path.exists(ck):
            try:
                os.remove(ck)
            except OSError:
                pass

    def tune_run(self, cfg):
        """Selection run: touches training and validation images only."""
        p = self._path("tune", cfg)
        if os.path.exists(p):
            return json.load(open(p))
        state, info, ck = self._train("tune", cfg)
        info["data_fingerprint"] = self.fingerprint
        if self.reuse_dir and cfg.arch in self.HEAVY:
            torch.save(state, self._reuse_path(cfg))
        json.dump(info, open(p, "w"))
        self._done(ck)
        return info

    def _reuse_path(self, cfg):
        return os.path.join(self.reuse_dir, cfg.name() + ".pt")

    def final_run(self, cfg):
        """Final run: trained, then evaluated on validation and test under every condition."""
        p = self._path("final", cfg)
        if os.path.exists(p):
            return json.load(open(p))
        tune_json = self._path("tune", cfg)
        if (self.reuse_dir and cfg.arch in self.HEAVY and os.path.exists(self._reuse_path(cfg))
                and os.path.exists(tune_json)):
            state, info, ck = torch.load(self._reuse_path(cfg), map_location="cpu"), json.load(open(tune_json)), None
            info["reused_tuning_run"] = True
            self.log(f"[{cfg.name()}] final run = selected tuning run (same configuration and seed), not retrained")
        else:
            state, info, ck = self._train("final", cfg)
        model = build_model(cfg.arch, pretrained=False).to(self.device)
        model.load_state_dict(state)
        amp = self.device.type == "cuda"
        bs = EVAL_BATCH[cfg.arch]
        info["metrics"], preds = {}, {}
        for split in ("val", "test"):
            info["metrics"][split], pr = evaluate_split(model, self.data, split, self.device, None, bs, amp)
            preds.update({f"{split}__{k}": v for k, v in pr.items()})
        info["data_fingerprint"] = self.fingerprint
        np.savez_compressed(os.path.join(self.out, "final", cfg.name() + "_preds.npz"), **preds)
        if self.save_checkpoints and cfg.arch not in self.HEAVY:
            try:
                torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in state.items()},
                           os.path.join(self.out, "ckpt", cfg.name() + ".pt"))
            except Exception as e:  # a full disk must not lose the metrics
                self.log(f"checkpoint not saved: {e}")
        json.dump(info, open(p, "w"))
        self._done(ck)
        m = info["metrics"]["test"]["clean"]
        self.log(f"FINAL {cfg.name()} best_epoch {info['best_epoch']} test acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f}")
        return info

    def run_architecture(self, arch, lr_grid, aug_grid, seeds=(0, 1, 2), max_epochs=60, patience=10,
                         batch_size=128, extra=None):
        """Stage 1: learning rate (standard training, seed 0). Stage 2: augmentation probability
        and strength (seed 0, chosen LR). Both use the validation selection score only. The choice
        is written to selection_<arch>.json before any final run (and so before any test image)."""
        sel_path = os.path.join(self.out, f"selection_{arch}.json")
        base = dict(arch=arch, batch_size=batch_size, max_epochs=max_epochs, patience=patience, **(extra or {}))
        if os.path.exists(sel_path):
            sel = json.load(open(sel_path))
        else:
            lr_scores = {}
            for lr in lr_grid:
                lr_scores[lr] = self.tune_run(TrainConfig(lr=lr, seed=seeds[0], **base))["best_score"]
            best_lr = max(lr_grid, key=lambda v: (lr_scores[v], -lr_grid.index(v)))
            self.log(f"[{arch}] LR selection {lr_scores} -> {best_lr:g}")
            aug_scores = {}
            for p, s in aug_grid:
                r = self.tune_run(TrainConfig(lr=best_lr, seed=seeds[0], aug_p=p, aug_sigma_max=s, **base))
                aug_scores[f"{p:g}|{s:g}"] = r["best_score"]
            best_aug = max(aug_grid, key=lambda ps: (aug_scores[f"{ps[0]:g}|{ps[1]:g}"], -aug_grid.index(ps)))
            sel = dict(arch=arch, lr=best_lr, aug_p=best_aug[0], aug_sigma_max=best_aug[1],
                       lr_scores={f"{k:g}": v for k, v in lr_scores.items()}, aug_scores=aug_scores,
                       std_reference_score=lr_scores[best_lr], frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       noise_primary=NOISE_PRIMARY, res_primary=RES_PRIMARY)
            json.dump(sel, open(sel_path, "w"), indent=1)
            self.log(f"[{arch}] augmentation selection {aug_scores} -> p={best_aug[0]:g}, sigma_max={best_aug[1]:g}")
        results = []
        for seed in seeds:
            results.append(self.final_run(TrainConfig(lr=sel["lr"], seed=seed, **base)))
            results.append(self.final_run(TrainConfig(lr=sel["lr"], seed=seed, aug_p=sel["aug_p"],
                                                      aug_sigma_max=sel["aug_sigma_max"], **base)))
        return sel, results

    def run_reference(self, arch, lr_grid=None, aug_grid=None, seeds=(0, 1, 2), max_epochs=60, patience=10,
                      batch_size=128, extra=None):
        """Reference runs ("ref"): standard training at the selected learning rate, with the checkpoint
        chosen on clean validation accuracy alone - the conventional rule. Comparing them with the "std"
        runs (same training, checkpoint chosen on clean + noisy validation) isolates what the selection
        rule itself changes. Requires selection_<arch>.json, so it never runs before selection is frozen."""
        sel = json.load(open(os.path.join(self.out, f"selection_{arch}.json")))
        base = dict(arch=arch, batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                    select_on="clean", **(extra or {}))
        return [self.final_run(TrainConfig(lr=sel["lr"], seed=seed, **base)) for seed in seeds]
