import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class CFG:
    dataset: str = "mnist"  # mnist | cifar10
    data_root: str = "/content/data"
    out_dir: str = "/content/runs/linsep_heads_mnist"
    seed: int = 43

    # Backbone training
    epochs: int = 20
    batch_size: int = 256
    eval_batch_size: int = 512
    lr: float = 1e-3
    min_lr_ratio: float = 0.05
    warmup_epochs: int = 0
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    num_workers: int = 2

    # Tiny ViT architecture
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1

    #  Use -1 for full dataset.
    train_subset: int = -1
    test_subset: int = -1

    # Checkpoints and posthoc analysis
    save_epochs: str = "0,1,2,5,10,20"
    probe_train_samples: int = 5000
    probe_test_samples: int = 2500
    bootstrap_samples: int = 1000

    # Nonlinear heads
    include_rbf_svm: bool = True
    include_mlp: bool = True
    include_rf: bool = True
    include_knn: bool = True
    include_gp: bool = False  # GP is slow; enable only for small subsamples.
    gp_train_samples: int = 800
    rf_trees: int = 300

    # Split-conformal diagnostics. These use only the training-feature pool for
    # fitting/selection/calibration, then report coverage and set size on the test split.
    enable_conformal: bool = True
    conformal_alpha: float = 0.10
    conformal_fit_fraction: float = 0.60
    conformal_selection_fraction: float = 0.20
    conformal_calib_fraction: float = 0.20
    save_conformal_examples: bool = True
    # Conformal score: "rank" gives nonempty top-label sets; "lac" uses 1 - class score and may be empty.
    conformal_score: str = "rank"
    # "holdout" splits the held-out/test feature pool into calibration/evaluation,
    # which gives a cleaner exchangeability diagnostic than calibrating on the training pool.
    # "train" keeps the original behavior: fit/selection/calibration from training features, evaluate on test.
    conformal_calibration_source: str = "holdout"
    conformal_holdout_calib_fraction: float = 0.50
    coverage_tolerance: float = 0.01

    # Recovery mode: analyze existing checkpoints without retraining.
    posthoc_only: bool = False


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    out = []
    for p in str(text).split(','):
        p = p.strip()
        if p:
            out.append(int(p))
    return sorted(set(out))


def maybe_subset(ds, n: int, seed: int):
    if n is None or n <= 0 or n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n].tolist()
    return Subset(ds, idx)


def set_epoch_lr(optimizer: torch.optim.Optimizer, epoch: int, cfg: CFG) -> float:
    # epoch is 1-based
    min_lr = cfg.lr * cfg.min_lr_ratio
    if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
        lr = cfg.lr * epoch / max(1, cfg.warmup_epochs)
    else:
        denom = max(1, cfg.epochs - cfg.warmup_epochs)
        t = min(1.0, max(0.0, (epoch - cfg.warmup_epochs) / denom))
        lr = min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * t))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


# -------------------------
# Data
# -------------------------

def dataset_meta(name: str) -> Tuple[int, int]:
    name = name.lower()
    if name == "mnist":
        return 1, 10
    if name == "cifar10":
        return 3, 10
    raise ValueError(f"Unknown dataset: {name}")


def get_dataloaders(cfg: CFG):
    ds = cfg.dataset.lower()
    if ds == "mnist":
        train_tf = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_tf = train_tf
        train_ds = datasets.MNIST(cfg.data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(cfg.data_root, train=False, download=True, transform=test_tf)
    elif ds == "cifar10":
        # Mild augmentation. Keep it simple so feature geometry remains easy to interpret.
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(ds)

    train_ds = maybe_subset(train_ds, cfg.train_subset, cfg.seed)
    test_ds = maybe_subset(test_ds, cfg.test_subset, cfg.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, train_eval_loader, test_loader


# -------------------------
# Model
# -------------------------

class TinyViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, num_classes: int,
                 embed_dim: int, num_layers: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.shape[1]]
        x = self.encoder(x)
        return self.norm(x[:, 0])

    def forward(self, x: torch.Tensor, return_features: bool = False):
        z = self.forward_features(x)
        logits = self.head(z)
        if return_features:
            return logits, z
        return logits


# -------------------------
# Backbone training
# -------------------------

@torch.no_grad()
def evaluate_backbone(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * len(y)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += len(y)
    return total_loss / total_n, total_correct / total_n


def train_one_epoch(model, loader, optimizer, scaler, device, cfg: CFG) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    use_amp = cfg.use_amp and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
        total_loss += float(loss.item()) * len(y)
        total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        total_n += len(y)
    return total_loss / total_n, total_correct / total_n


def save_checkpoint(model, epoch: int, ckpt_dir: Path, cfg: CFG):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
    }, ckpt_dir / f"epoch_{epoch:03d}.pt")


def list_checkpoints(ckpt_dir: Path) -> List[Path]:
    return sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))


# -------------------------
# Feature extraction and geometry
# -------------------------

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device,
                     max_samples: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs, ys = [], []
    seen = 0
    for x, y in loader:
        if max_samples > 0 and seen >= max_samples:
            break
        x = x.to(device, non_blocking=True)
        _, z = model(x, return_features=True)
        z = z.detach().cpu().float().numpy()
        y = y.detach().cpu().numpy()
        if max_samples > 0 and seen + len(y) > max_samples:
            keep = max_samples - seen
            z, y = z[:keep], y[:keep]
        xs.append(z)
        ys.append(y)
        seen += len(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def within_between_ratio(X: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    mu = X.mean(axis=0, keepdims=True)
    within, between = 0.0, 0.0
    for c in classes:
        Xc = X[y == c]
        muc = Xc.mean(axis=0, keepdims=True)
        within += float(((Xc - muc) ** 2).sum())
        between += float(len(Xc) * ((muc - mu) ** 2).sum())
    return within / max(between, 1e-12)


def effective_rank(X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    # covariance eigenvalues via singular values
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eig = (s ** 2) / max(1, X.shape[0] - 1)
    eig = eig[eig > 1e-12]
    if len(eig) == 0:
        return 0.0
    p = eig / eig.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def participation_ratio(X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eig = (s ** 2) / max(1, X.shape[0] - 1)
    denom = np.sum(eig ** 2)
    if denom <= 0:
        return 0.0
    return float((np.sum(eig) ** 2) / denom)


def signed_true_class_margin(decision_values: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Positive margin means the true class score is above every other class score.
    scores = np.asarray(decision_values)
    if scores.ndim == 1:
        # binary convention: score > 0 corresponds to class 1
        yy = (y == np.max(y)).astype(int)
        return np.where(yy == 1, scores, -scores)
    true = scores[np.arange(len(y)), y]
    masked = scores.copy()
    masked[np.arange(len(y)), y] = -np.inf
    other = masked.max(axis=1)
    return true - other


# -------------------------
# Heads and statistical tests
# -------------------------

def make_heads(cfg: CFG, n_train: int) -> Dict[str, object]:
    heads: Dict[str, object] = {}
    heads["linear_logreg"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10.0, max_iter=3000, solver="lbfgs", n_jobs=-1, random_state=cfg.seed),
    )
    heads["linear_svm"] = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=8000, dual=False, random_state=cfg.seed),
    )
    if cfg.include_knn:
        heads["knn_distance"] = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
        )
    if cfg.include_rf:
        heads["random_forest"] = RandomForestClassifier(
            n_estimators=cfg.rf_trees,
            max_features="sqrt",
            n_jobs=-1,
            random_state=cfg.seed,
        )
    if cfg.include_rbf_svm:
        heads["rbf_svm"] = make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", kernel="rbf", decision_function_shape="ovr", random_state=cfg.seed),
        )
    if cfg.include_mlp:
        heads["small_mlp"] = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), activation="relu", alpha=1e-4,
                          max_iter=300, early_stopping=True, n_iter_no_change=20,
                          random_state=cfg.seed),
        )
    if cfg.include_gp and n_train <= cfg.gp_train_samples:
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        heads["gaussian_process"] = make_pipeline(
            StandardScaler(),
            GaussianProcessClassifier(kernel=kernel, random_state=cfg.seed, max_iter_predict=50, n_restarts_optimizer=0),
        )
    return heads


def bootstrap_delta_ci(y_true: np.ndarray, pred_base: np.ndarray, pred_head: np.ndarray,
                       n_boot: int, seed: int) -> Tuple[float, float, float]:
    d = (pred_head == y_true).astype(float) - (pred_base == y_true).astype(float)
    delta = float(d.mean())
    if n_boot <= 0:
        return delta, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    vals = d[idx].mean(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return delta, float(lo), float(hi)


def mcnemar_exact(y_true: np.ndarray, pred_base: np.ndarray, pred_head: np.ndarray) -> Tuple[int, int, float]:
    base_ok = pred_base == y_true
    head_ok = pred_head == y_true
    n01 = int((~base_ok & head_ok).sum())  # base wrong, head right
    n10 = int((base_ok & ~head_ok).sum())  # base right, head wrong
    discordant = n01 + n10
    p = 1.0 if discordant == 0 else float(binomtest(min(n01, n10), discordant, 0.5).pvalue)
    return n01, n10, p



def _safe_stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/test indices. Falls back to unstratified if a class is too small."""
    idx = np.arange(len(y))
    strat = y
    try:
        a, b = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    except Exception:
        a, b = train_test_split(idx, test_size=test_size, random_state=seed, stratify=None)
    return np.asarray(a), np.asarray(b)


def split_for_conformal(X: np.ndarray, y: np.ndarray, cfg: CFG, epoch: int) -> Tuple[np.ndarray, ...]:
    """Original train-pool split: fit / selection / calibration.

    Kept for comparison via --conformal-calibration-source train.
    For the note, the default is now --conformal-calibration-source holdout,
    which calibrates and evaluates on two disjoint halves of the held-out pool.
    """
    n = len(y)
    fit_frac = float(cfg.conformal_fit_fraction)
    sel_frac = float(cfg.conformal_selection_fraction)
    cal_frac = float(cfg.conformal_calib_fraction)
    total = fit_frac + sel_frac + cal_frac
    if total <= 0:
        raise ValueError("conformal split fractions must sum to a positive value")
    fit_frac, sel_frac, cal_frac = fit_frac / total, sel_frac / total, cal_frac / total
    rng = np.random.default_rng(cfg.seed + 17 * epoch)
    idx = rng.permutation(n)
    n_fit = max(1, int(round(n * fit_frac)))
    n_sel = max(1, int(round(n * sel_frac)))
    if n_fit + n_sel >= n:
        raise ValueError("Not enough samples for conformal selection/calibration split")
    fit_idx = idx[:n_fit]
    sel_idx = idx[n_fit:n_fit + n_sel]
    cal_idx = idx[n_fit + n_sel:]
    return X[fit_idx], y[fit_idx], X[sel_idx], y[sel_idx], X[cal_idx], y[cal_idx]


def split_fit_selection(X: np.ndarray, y: np.ndarray, cfg: CFG, epoch: int) -> Tuple[np.ndarray, ...]:
    """Split training features into fit and model-selection pools."""
    n = len(y)
    fit_frac = float(cfg.conformal_fit_fraction)
    sel_frac = float(cfg.conformal_selection_fraction)
    total = fit_frac + sel_frac
    if total <= 0:
        raise ValueError("fit/selection fractions must sum to a positive value")
    fit_frac = fit_frac / total
    rng = np.random.default_rng(cfg.seed + 17 * epoch)
    idx = rng.permutation(n)
    n_fit = max(1, int(round(n * fit_frac)))
    if n_fit >= n:
        n_fit = n - 1
    fit_idx = idx[:n_fit]
    sel_idx = idx[n_fit:]
    return X[fit_idx], y[fit_idx], X[sel_idx], y[sel_idx]


def split_holdout_calib_eval(X: np.ndarray, y: np.ndarray, cfg: CFG, epoch: int) -> Tuple[np.ndarray, ...]:
    """Split held-out features into calibration and final evaluation pools.

    Calibration and evaluation then come from the same original split, which makes
    conformal coverage easier to interpret. This is the default path.
    """
    n = len(y)
    frac = float(cfg.conformal_holdout_calib_fraction)
    if not (0.05 <= frac <= 0.95):
        raise ValueError("conformal_holdout_calib_fraction should be between 0.05 and 0.95")
    rng = np.random.default_rng(cfg.seed + 1009 + 17 * epoch)
    idx = rng.permutation(n)
    n_cal = max(1, int(round(n * frac)))
    if n_cal >= n:
        n_cal = n - 1
    cal_idx = idx[:n_cal]
    eval_idx = idx[n_cal:]
    return X[cal_idx], y[cal_idx], X[eval_idx], y[eval_idx]


def _softmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _aligned_scores(clf, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return nonnegative class scores aligned to `classes`.

    Prefer predict_proba when available. Otherwise use a softmax of decision_function
    values. For conformal validity, these only need to be fixed scores; they do not
    need to be calibrated probabilities.
    """
    est_classes = getattr(clf, "classes_", None)
    if est_classes is None and hasattr(clf, "named_steps"):
        for step in reversed(list(clf.named_steps.values())):
            if hasattr(step, "classes_"):
                est_classes = step.classes_
                break
    if est_classes is None:
        est_classes = classes
    est_classes = np.asarray(est_classes)

    if hasattr(clf, "predict_proba"):
        raw = np.asarray(clf.predict_proba(X), dtype=float)
        raw = np.clip(raw, 0.0, None)
        raw = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)
    elif hasattr(clf, "decision_function"):
        dec = np.asarray(clf.decision_function(X), dtype=float)
        if dec.ndim == 1:
            raw = np.column_stack([-dec, dec])
        else:
            raw = dec
        raw = _softmax(raw)
    else:
        pred = np.asarray(clf.predict(X))
        raw = np.zeros((len(pred), len(est_classes)), dtype=float)
        class_to_col = {c: j for j, c in enumerate(est_classes)}
        for i, c in enumerate(pred):
            if c in class_to_col:
                raw[i, class_to_col[c]] = 1.0

    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    class_to_col = {c: j for j, c in enumerate(classes)}
    for j, c in enumerate(est_classes):
        if c in class_to_col and j < raw.shape[1]:
            out[:, class_to_col[c]] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum.squeeze() <= 1e-12
    if np.any(bad):
        out[bad, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / np.clip(row_sum, 1e-12, None)


def _conformal_quantile(nonconformity_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile for marginal 1-alpha coverage."""
    scores = np.sort(np.asarray(nonconformity_scores, dtype=float))
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def _rank_scores_before(scores: np.ndarray) -> np.ndarray:
    """For each class, sum scores of strictly higher-ranked classes.

    This produces nonconformity scores where the top class has score 0, so prediction
    sets are nonempty. It is useful when top-1 accuracy already exceeds the target
    coverage: uncertainty should collapse to singleton sets rather than empty sets.
    """
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    order = np.argsort(-scores, axis=1)
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    csum_before_sorted = np.concatenate([
        np.zeros((n, 1), dtype=float),
        np.cumsum(sorted_scores, axis=1)[:, :-1]
    ], axis=1)
    out = np.zeros_like(scores, dtype=float)
    np.put_along_axis(out, order, csum_before_sorted, axis=1)
    return out


def _nonconformity_matrix(scores: np.ndarray, method: str) -> np.ndarray:
    method = str(method).lower()
    if method == "lac":
        # Least ambiguous classifier score: small when class score/probability is high.
        # This can produce empty sets when target coverage is below top-1 accuracy.
        return 1.0 - scores
    if method in {"rank", "aps", "nonempty"}:
        # Nonempty rank/adaptive-style score: top class is always included.
        return _rank_scores_before(scores)
    raise ValueError(f"Unknown conformal_score={method!r}; use 'rank' or 'lac'.")


def evaluate_conformal_for_checkpoint(epoch: int, heads: Dict[str, object],
                                      Xpool: np.ndarray, ypool: np.ndarray,
                                      Xtest: np.ndarray, ytest: np.ndarray,
                                      cfg: CFG) -> Tuple[List[dict], Optional[pd.DataFrame]]:
    """Fit heads, select without test leakage, calibrate conformal sets, evaluate.

    Default path:
      - fit and select heads using the training-feature pool;
      - split the held-out/test-feature pool into calibration and final evaluation;
      - report coverage and set size only on the final evaluation half.

    This is cleaner than calibrating on the training pool and evaluating on the official
    test split, because calibration/evaluation are exchangeable draws from the same pool.
    """
    classes = np.sort(np.unique(np.concatenate([ypool, ytest])))
    class_to_idx = {c: j for j, c in enumerate(classes)}

    source = str(cfg.conformal_calibration_source).lower()
    if source == "train":
        Xfit, yfit, Xsel, ysel, Xcal, ycal = split_for_conformal(Xpool, ypool, cfg, epoch)
        Xeval, yeval = Xtest, ytest
    elif source in {"holdout", "test"}:
        Xfit, yfit, Xsel, ysel = split_fit_selection(Xpool, ypool, cfg, epoch)
        Xcal, ycal, Xeval, yeval = split_holdout_calib_eval(Xtest, ytest, cfg, epoch)
    else:
        raise ValueError("conformal_calibration_source must be 'holdout' or 'train'")

    rows: List[dict] = []
    example_rows: List[pd.DataFrame] = []
    fitted = {}
    selection_accs = {}

    for name in heads:
        start = time.time()
        try:
            clf = clone(make_heads(cfg, len(yfit))[name])
            X_fit_head, y_fit_head = Xfit, yfit
            if name == "gaussian_process" and len(yfit) > cfg.gp_train_samples:
                rng = np.random.default_rng(cfg.seed + 101 * epoch)
                idx = rng.choice(len(yfit), size=cfg.gp_train_samples, replace=False)
                X_fit_head, y_fit_head = Xfit[idx], yfit[idx]
            clf.fit(X_fit_head, y_fit_head)
            fitted[name] = clf
            selection_accs[name] = float(accuracy_score(ysel, clf.predict(Xsel)))
        except Exception as e:
            rows.append({
                "epoch": epoch, "head": name, "head_family": "linear" if name.startswith("linear") else "nonlinear",
                "alpha": cfg.conformal_alpha, "coverage_target": 1.0 - cfg.conformal_alpha,
                "conformal_score": cfg.conformal_score, "calibration_source": source,
                "n_fit": len(yfit), "n_selection": len(ysel), "n_calib": len(ycal), "n_test": len(yeval),
                "selection_acc": np.nan, "test_acc": np.nan, "qhat": np.nan,
                "coverage": np.nan, "coverage_gap": np.nan, "avg_set_size": np.nan,
                "median_set_size": np.nan, "singleton_rate": np.nan, "empty_rate": np.nan,
                "valid_at_target": False,
                "selected_overall": False, "selected_nonlinear": False,
                "fit_seconds": float(time.time() - start), "error": repr(e),
            })

    if selection_accs:
        selected_overall = max(selection_accs, key=selection_accs.get)
        nonlin_names = [k for k in selection_accs if not k.startswith("linear")]
        selected_nonlinear = max(nonlin_names, key=lambda k: selection_accs[k]) if nonlin_names else None
    else:
        selected_overall = None
        selected_nonlinear = None

    for name, clf in fitted.items():
        start = time.time()
        try:
            cal_scores = _aligned_scores(clf, Xcal, classes)
            cal_nonconf_mat = _nonconformity_matrix(cal_scores, cfg.conformal_score)
            cal_true_cols = np.array([class_to_idx[y] for y in ycal])
            cal_nonconf = cal_nonconf_mat[np.arange(len(ycal)), cal_true_cols]
            qhat = _conformal_quantile(cal_nonconf, cfg.conformal_alpha)

            eval_scores = _aligned_scores(clf, Xeval, classes)
            eval_nonconf_mat = _nonconformity_matrix(eval_scores, cfg.conformal_score)
            pred = classes[np.argmax(eval_scores, axis=1)]
            pred_sets = eval_nonconf_mat <= qhat
            set_sizes = pred_sets.sum(axis=1)
            true_cols = np.array([class_to_idx[y] for y in yeval])
            covered = pred_sets[np.arange(len(yeval)), true_cols]
            coverage = float(np.mean(covered))
            target = 1.0 - cfg.conformal_alpha
            valid_at_target = bool(coverage >= target - float(cfg.coverage_tolerance))
            row = {
                "epoch": epoch,
                "head": name,
                "head_family": "linear" if name.startswith("linear") else "nonlinear",
                "alpha": float(cfg.conformal_alpha),
                "coverage_target": target,
                "conformal_score": cfg.conformal_score,
                "calibration_source": source,
                "n_fit": len(yfit),
                "n_selection": len(ysel),
                "n_calib": len(ycal),
                "n_test": len(yeval),
                "selection_acc": float(selection_accs.get(name, np.nan)),
                "test_acc": float(accuracy_score(yeval, pred)),
                "qhat": qhat,
                "coverage": coverage,
                "coverage_gap": coverage - target,
                "avg_set_size": float(np.mean(set_sizes)),
                "median_set_size": float(np.median(set_sizes)),
                "singleton_rate": float(np.mean(set_sizes == 1)),
                "empty_rate": float(np.mean(set_sizes == 0)),
                "valid_at_target": valid_at_target,
                "selected_overall": bool(name == selected_overall),
                "selected_nonlinear": bool(name == selected_nonlinear),
                "fit_seconds": float(time.time() - start),
                "error": "",
            }
            rows.append(row)

            if cfg.save_conformal_examples:
                max_score = eval_scores.max(axis=1)
                ex = pd.DataFrame({
                    "epoch": epoch,
                    "head": name,
                    "true": yeval,
                    "pred": pred,
                    "correct": pred == yeval,
                    "covered": covered,
                    "set_size": set_sizes,
                    "top_score": max_score,
                    "true_score": eval_scores[np.arange(len(yeval)), true_cols],
                    "qhat": qhat,
                    "conformal_score": cfg.conformal_score,
                    "calibration_source": source,
                    "selected_overall": bool(name == selected_overall),
                    "selected_nonlinear": bool(name == selected_nonlinear),
                })
                example_rows.append(ex)
        except Exception as e:
            rows.append({
                "epoch": epoch, "head": name, "head_family": "linear" if name.startswith("linear") else "nonlinear",
                "alpha": cfg.conformal_alpha, "coverage_target": 1.0 - cfg.conformal_alpha,
                "conformal_score": cfg.conformal_score, "calibration_source": source,
                "n_fit": len(yfit), "n_selection": len(ysel), "n_calib": len(ycal), "n_test": len(yeval),
                "selection_acc": float(selection_accs.get(name, np.nan)), "test_acc": np.nan, "qhat": np.nan,
                "coverage": np.nan, "coverage_gap": np.nan, "avg_set_size": np.nan,
                "median_set_size": np.nan, "singleton_rate": np.nan, "empty_rate": np.nan,
                "valid_at_target": False,
                "selected_overall": bool(name == selected_overall), "selected_nonlinear": bool(name == selected_nonlinear),
                "fit_seconds": float(time.time() - start), "error": repr(e),
            })

    examples = pd.concat(example_rows, ignore_index=True) if example_rows else None
    return rows, examples

def evaluate_heads_for_checkpoint(epoch: int, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                                  device: torch.device, cfg: CFG) -> Tuple[List[dict], dict, Optional[pd.DataFrame], List[dict], Optional[pd.DataFrame]]:
    Xtr, ytr = extract_features(model, train_loader, device, cfg.probe_train_samples)
    Xte, yte = extract_features(model, test_loader, device, cfg.probe_test_samples)

    # Optionally subsample GP even more; other heads use the same common sample for fair comparison.
    heads = make_heads(cfg, len(ytr))

    geom = {
        "epoch": epoch,
        "n_train_features": len(ytr),
        "n_test_features": len(yte),
        "feature_dim": Xtr.shape[1],
        "within_between_ratio_train": within_between_ratio(Xtr, ytr),
        "within_between_ratio_test": within_between_ratio(Xte, yte),
        "effective_rank_train": effective_rank(Xtr),
        "participation_ratio_train": participation_ratio(Xtr),
    }

    rows = []
    predictions = {}
    for name, clf in heads.items():
        start = time.time()
        try:
            Xfit, yfit = Xtr, ytr
            if name == "gaussian_process" and len(ytr) > cfg.gp_train_samples:
                rng = np.random.default_rng(cfg.seed + epoch)
                idx = rng.choice(len(ytr), size=cfg.gp_train_samples, replace=False)
                Xfit, yfit = Xtr[idx], ytr[idx]
            clf.fit(Xfit, yfit)
            pred_train = clf.predict(Xtr)
            pred_test = clf.predict(Xte)
            train_acc = accuracy_score(ytr, pred_train)
            test_acc = accuracy_score(yte, pred_test)
            row = {
                "epoch": epoch,
                "head": name,
                "head_family": "linear" if name.startswith("linear") else "nonlinear",
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "fit_seconds": float(time.time() - start),
                "error": "",
            }
            if name == "linear_logreg":
                try:
                    dec = clf.decision_function(Xte)
                    m = signed_true_class_margin(dec, yte)
                    geom.update({
                        "linear_margin_mean": float(np.mean(m)),
                        "linear_margin_median": float(np.median(m)),
                        "linear_margin_p10": float(np.percentile(m, 10)),
                        "linear_margin_frac_negative": float(np.mean(m < 0)),
                    })
                except Exception as e:
                    geom["linear_margin_error"] = repr(e)
            predictions[name] = pred_test
            rows.append(row)
            print(f"    {name:16s} test_acc={test_acc:.4f} train_acc={train_acc:.4f} fit={row['fit_seconds']:.1f}s")
        except Exception as e:
            rows.append({
                "epoch": epoch,
                "head": name,
                "head_family": "linear" if name.startswith("linear") else "nonlinear",
                "train_acc": np.nan,
                "test_acc": np.nan,
                "fit_seconds": float(time.time() - start),
                "error": repr(e),
            })
            print(f"    [warn] {name} failed: {e}")

    # Deltas and paired tests against the regularized linear probe.
    base_name = "linear_logreg"
    pred_base = predictions.get(base_name)
    if pred_base is not None:
        base_acc = next(r["test_acc"] for r in rows if r["head"] == base_name)
        for row in rows:
            pred = predictions.get(row["head"])
            if pred is None:
                row.update({
                    "delta_vs_linear": np.nan,
                    "delta_ci_low": np.nan,
                    "delta_ci_high": np.nan,
                    "room_to_improve": 1.0 - base_acc,
                    "relative_gain_of_remaining_error": np.nan,
                    "mcnemar_n01": np.nan,
                    "mcnemar_n10": np.nan,
                    "mcnemar_p": np.nan,
                })
                continue
            delta, lo, hi = bootstrap_delta_ci(yte, pred_base, pred, cfg.bootstrap_samples, cfg.seed + epoch)
            n01, n10, p = mcnemar_exact(yte, pred_base, pred)
            room = max(1e-12, 1.0 - base_acc)
            row.update({
                "delta_vs_linear": delta,
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "room_to_improve": 1.0 - base_acc,
                "relative_gain_of_remaining_error": delta / room,
                "mcnemar_n01": n01,
                "mcnemar_n10": n10,
                "mcnemar_p": p,
            })

    # Additional deltas and paired tests against the strongest linear baseline at this checkpoint.
    # This is the fairer comparison for the note: if a nonlinear head only beats
    # logistic regression but not a linear SVM, the gain is not really a nonlinear gain.
    valid_linear_rows = [
        r for r in rows
        if r.get("head_family") == "linear"
        and not pd.isna(r.get("test_acc", np.nan))
        and str(r.get("error", "")) in {"", "nan"}
    ]
    if valid_linear_rows:
        best_linear_row = max(valid_linear_rows, key=lambda r: float(r["test_acc"]))
        best_linear_name = best_linear_row["head"]
        pred_best_linear = predictions.get(best_linear_name)
        best_linear_acc = float(best_linear_row["test_acc"])
        if pred_best_linear is not None:
            for row in rows:
                pred = predictions.get(row["head"])
                row["best_linear_head"] = best_linear_name
                row["best_linear_acc"] = best_linear_acc
                if pred is None:
                    row.update({
                        "delta_vs_best_linear": np.nan,
                        "delta_best_linear_ci_low": np.nan,
                        "delta_best_linear_ci_high": np.nan,
                        "relative_gain_vs_best_linear_remaining_error": np.nan,
                        "mcnemar_vs_best_linear_n01": np.nan,
                        "mcnemar_vs_best_linear_n10": np.nan,
                        "mcnemar_vs_best_linear_p": np.nan,
                    })
                    continue
                delta_bl, lo_bl, hi_bl = bootstrap_delta_ci(yte, pred_best_linear, pred, cfg.bootstrap_samples, cfg.seed + 17 * epoch + 7)
                n01_bl, n10_bl, p_bl = mcnemar_exact(yte, pred_best_linear, pred)
                room_bl = max(1e-12, 1.0 - best_linear_acc)
                row.update({
                    "delta_vs_best_linear": delta_bl,
                    "delta_best_linear_ci_low": lo_bl,
                    "delta_best_linear_ci_high": hi_bl,
                    "relative_gain_vs_best_linear_remaining_error": delta_bl / room_bl,
                    "mcnemar_vs_best_linear_n01": n01_bl,
                    "mcnemar_vs_best_linear_n10": n10_bl,
                    "mcnemar_vs_best_linear_p": p_bl,
                })

    conformal_rows: List[dict] = []
    conformal_examples: Optional[pd.DataFrame] = None
    if cfg.enable_conformal:
        try:
            print("    split-conformal diagnostics...")
            conformal_rows, conformal_examples = evaluate_conformal_for_checkpoint(epoch, heads, Xtr, ytr, Xte, yte, cfg)
            selected = [r for r in conformal_rows if r.get("selected_overall") and not r.get("error")]
            if selected:
                r = selected[0]
                print(f"    selected_by_selection={r['head']} conf_coverage={r['coverage']:.3f} avg_set_size={r['avg_set_size']:.3f}")
        except Exception as e:
            conformal_rows = [{
                "epoch": epoch, "head": "__conformal_failed__", "head_family": "diagnostic",
                "alpha": cfg.conformal_alpha, "coverage_target": 1.0 - cfg.conformal_alpha,
                "error": repr(e),
            }]
            conformal_examples = None
            print(f"    [warn] conformal diagnostics failed: {e}")

    # PCA projection for an optional visual check.
    pca_df = None
    try:
        pca = PCA(n_components=2, random_state=cfg.seed)
        Z = pca.fit_transform(Xte)
        pca_df = pd.DataFrame({"epoch": epoch, "pc1": Z[:, 0], "pc2": Z[:, 1], "y": yte})
    except Exception:
        pass

    return rows, geom, pca_df, conformal_rows, conformal_examples


# -------------------------
# Plots
# -------------------------

def save_plots(results: pd.DataFrame, geom: pd.DataFrame, out_dir: Path, conformal: Optional[pd.DataFrame] = None, conformal_examples: Optional[pd.DataFrame] = None) -> None:
    fig_dir = ensure_dir(out_dir / "figures")

    # 1. Head test accuracy over epochs.
    plt.figure(figsize=(9, 5))
    for head, df in results.groupby("head"):
        df = df.sort_values("epoch")
        plt.plot(df["epoch"], df["test_acc"], marker="o", label=head)
    plt.xlabel("backbone epoch")
    plt.ylabel("test accuracy on frozen features")
    plt.title("Linear and nonlinear heads on the same frozen features")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "head_accuracy_by_epoch.png", dpi=180)
    plt.close()

    # 2. Gain over linear probe.
    plt.figure(figsize=(9, 5))
    df_gain = results[results["head"] != "linear_logreg"].copy()
    for head, df in df_gain.groupby("head"):
        df = df.sort_values("epoch")
        plt.plot(df["epoch"], df["delta_vs_linear"], marker="o", label=head)
        if df["delta_ci_low"].notna().all():
            plt.fill_between(df["epoch"], df["delta_ci_low"], df["delta_ci_high"], alpha=0.15)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("backbone epoch")
    plt.ylabel("test accuracy gain over logistic linear probe")
    plt.title("Does the nonlinear head still have room to help?")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "nonlinear_gain_by_epoch.png", dpi=180)
    plt.close()

    # 3. Gain as a function of linear accuracy.
    base = results[results["head"] == "linear_logreg"][["epoch", "test_acc"]].rename(columns={"test_acc": "linear_acc"})
    joined = results.merge(base, on="epoch", how="left")
    joined = joined[(joined["head_family"] == "nonlinear") & joined["delta_vs_linear"].notna()]
    plt.figure(figsize=(7, 5))
    for head, df in joined.groupby("head"):
        plt.scatter(df["linear_acc"], df["delta_vs_linear"], label=head)
        df = df.sort_values("linear_acc")
        plt.plot(df["linear_acc"], df["delta_vs_linear"], alpha=0.6)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("linear probe test accuracy")
    plt.ylabel("nonlinear gain over linear probe")
    plt.title("Nonlinear gain usually shrinks as linear readability improves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "gain_vs_linear_accuracy.png", dpi=180)
    plt.close()

    # 4. Geometry diagnostics.
    if not geom.empty:
        plt.figure(figsize=(9, 5))
        if "within_between_ratio_test" in geom:
            plt.plot(geom["epoch"], geom["within_between_ratio_test"], marker="o", label="within/between ratio, test")
        if "linear_margin_frac_negative" in geom:
            plt.plot(geom["epoch"], geom["linear_margin_frac_negative"], marker="o", label="fraction negative linear margins")
        plt.xlabel("backbone epoch")
        plt.ylabel("diagnostic value")
        plt.title("Feature geometry becomes easier for a linear head")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "geometry_diagnostics.png", dpi=180)
        plt.close()

    # 5. Compact summary: linear error vs best nonlinear gain.
    if not joined.empty:
        best = joined.groupby("epoch", as_index=False)["delta_vs_linear"].max().rename(columns={"delta_vs_linear": "best_nonlinear_gain"})
        best = best.merge(base, on="epoch", how="left")
        best["linear_error"] = 1 - best["linear_acc"]
        plt.figure(figsize=(6, 5))
        plt.scatter(best["linear_error"], best["best_nonlinear_gain"])
        for _, r in best.iterrows():
            plt.annotate(str(int(r["epoch"])), (r["linear_error"], r["best_nonlinear_gain"]), fontsize=8)
        plt.xlabel("linear probe error")
        plt.ylabel("best nonlinear gain")
        plt.title("Room left by the linear probe vs best nonlinear gain")
        plt.tight_layout()
        plt.savefig(fig_dir / "best_gain_vs_linear_error.png", dpi=180)
        plt.close()

    # 6. Split-conformal uncertainty diagnostics.
    if conformal is not None and not conformal.empty and "avg_set_size" in conformal:
        conf = conformal[conformal["error"].fillna("") == ""].copy() if "error" in conformal else conformal.copy()
        if not conf.empty:
            plt.figure(figsize=(9, 5))
            for head, df in conf.groupby("head"):
                df = df.sort_values("epoch")
                plt.plot(df["epoch"], df["avg_set_size"], marker="o", label=head)
            plt.xlabel("backbone epoch")
            plt.ylabel("average conformal set size")
            plt.title("At fixed coverage, does the nonlinear head reduce uncertainty?")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / "conformal_avg_set_size_by_epoch.png", dpi=180)
            plt.close()

            plt.figure(figsize=(9, 5))
            for head, df in conf.groupby("head"):
                df = df.sort_values("epoch")
                plt.plot(df["epoch"], df["coverage"], marker="o", label=head)
            target = float(conf["coverage_target"].dropna().iloc[0]) if conf["coverage_target"].notna().any() else 0.90
            plt.axhline(target, linestyle="--", linewidth=1, label=f"target={target:.2f}")
            plt.xlabel("backbone epoch")
            plt.ylabel("empirical coverage")
            plt.title("Split-conformal coverage on the held-out test split")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / "conformal_coverage_by_epoch.png", dpi=180)
            plt.close()

            plt.figure(figsize=(9, 5))
            for head, df in conf.groupby("head"):
                df = df.sort_values("epoch")
                plt.plot(df["epoch"], df["singleton_rate"], marker="o", label=head)
            plt.xlabel("backbone epoch")
            plt.ylabel("singleton prediction-set rate")
            plt.title("As features linearize, conformal sets collapse to singletons")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / "conformal_singleton_rate_by_epoch.png", dpi=180)
            plt.close()

            selected = conf[conf.get("selected_overall", False) == True].copy()
            if not selected.empty:
                plt.figure(figsize=(7, 5))
                plt.plot(selected["epoch"], selected["avg_set_size"], marker="o", label="selected head avg set size")
                plt.plot(selected["epoch"], selected["test_acc"], marker="o", label="selected head accuracy")
                plt.xlabel("backbone epoch")
                plt.title("Head selected on the selection split: accuracy vs uncertainty")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(fig_dir / "conformal_selected_head_by_epoch.png", dpi=180)
                plt.close()

    if conformal_examples is not None and not conformal_examples.empty:
        try:
            final_epoch = int(conformal_examples["epoch"].max())
            ex = conformal_examples[conformal_examples["epoch"] == final_epoch].copy()
            # Use max score as a simple confidence proxy; plot binned set sizes.
            bins = pd.qcut(ex["max_score"], q=10, duplicates="drop")
            b = ex.assign(score_bin=bins).groupby(["head", "score_bin"], observed=True).agg(
                max_score_mean=("max_score", "mean"), avg_set_size=("set_size", "mean")
            ).reset_index()
            plt.figure(figsize=(7, 5))
            for head, df in b.groupby("head"):
                df = df.sort_values("max_score_mean")
                plt.plot(df["max_score_mean"], df["avg_set_size"], marker="o", label=head)
            plt.xlabel("mean top class score in bin")
            plt.ylabel("average conformal set size")
            plt.title(f"Final epoch {final_epoch}: uncertainty concentrates on low-score examples")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "conformal_set_size_vs_top_score_final.png", dpi=180)
            plt.close()
        except Exception:
            pass


def write_summary(results: pd.DataFrame, geom: pd.DataFrame, out_dir: Path, conformal: Optional[pd.DataFrame] = None) -> None:
    base = results[results["head"] == "linear_logreg"][["epoch", "test_acc"]].rename(columns={"test_acc": "linear_acc"})
    joined = results.merge(base, on="epoch", how="left")
    nonlin = joined[joined["head_family"] == "nonlinear"].copy()
    lines = []
    lines.append("# Linear separability vs nonlinear heads: run summary\n")
    if not base.empty:
        first = base.sort_values("epoch").iloc[0]
        last = base.sort_values("epoch").iloc[-1]
        lines.append(f"- Linear probe accuracy moved from **{first.linear_acc:.4f}** at epoch {int(first.epoch)} to **{last.linear_acc:.4f}** at epoch {int(last.epoch)}.\n")
    if not nonlin.empty:
        best_by_epoch = nonlin.groupby("epoch", as_index=False)["delta_vs_linear"].max()
        first = best_by_epoch.sort_values("epoch").iloc[0]
        last = best_by_epoch.sort_values("epoch").iloc[-1]
        lines.append(f"- Best nonlinear gain moved from **{first.delta_vs_linear:+.4f}** at epoch {int(first.epoch)} to **{last.delta_vs_linear:+.4f}** at epoch {int(last.epoch)}.\n")
        tmp = best_by_epoch.merge(base, on="epoch", how="left")
        if len(tmp) >= 3:
            corr = tmp["linear_acc"].corr(tmp["delta_vs_linear"], method="spearman")
            lines.append(f"- Spearman correlation between linear accuracy and best nonlinear gain: **{corr:.3f}**. Negative values support the bottleneck story.\n")
    if not geom.empty and "within_between_ratio_test" in geom:
        g = geom.sort_values("epoch")
        lines.append(f"- Test within/between scatter ratio moved from **{g.iloc[0].within_between_ratio_test:.4f}** to **{g.iloc[-1].within_between_ratio_test:.4f}**. Lower means tighter class clouds relative to class separation.\n")
    if conformal is not None and not conformal.empty and "avg_set_size" in conformal:
        conf = conformal[conformal["error"].fillna("") == ""].copy() if "error" in conformal else conformal.copy()
        if not conf.empty:
            final_epoch = int(conf["epoch"].max())
            final = conf[conf["epoch"] == final_epoch].sort_values("avg_set_size")
            best_unc = final.iloc[0]
            lines.append(f"- At the final checkpoint, the smallest split-conformal average set size was **{best_unc.avg_set_size:.3f}** from **{best_unc.head}**, with coverage **{best_unc.coverage:.3f}** at target **{best_unc.coverage_target:.2f}**.\n")
            selected = final[final.get("selected_overall", False) == True]
            if len(selected):
                r = selected.iloc[0]
                lines.append(f"- The head selected on the independent selection split was **{r.head}**; on test it had accuracy **{r.test_acc:.4f}**, coverage **{r.coverage:.3f}**, and average set size **{r.avg_set_size:.3f}**.\n")

    lines.append("\nInterpretation: nonlinear heads are most informative as a diagnostic. Large positive deltas mean the frozen representation still contains label structure that is not linearly readable. Small deltas at high linear-probe accuracy mean the backbone has already done the geometric work. Conformal prediction adds a second diagnostic: at fixed coverage, did the nonlinear head actually reduce uncertainty, or did it only flip a few borderline point predictions?\n")
    (out_dir / "summary.md").write_text("".join(lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------

def run(cfg: CFG) -> None:
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.out_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(json.dumps(asdict(cfg), indent=2))

    train_loader, train_eval_loader, test_loader = get_dataloaders(cfg)
    in_chans, num_classes = dataset_meta(cfg.dataset)
    model = TinyViT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    save_epochs = set(parse_int_list(cfg.save_epochs))
    if cfg.posthoc_only:
        existing = list_checkpoints(ckpt_dir)
        if not existing:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}. Run training first, or disable --posthoc-only.")
        print(f"Posthoc-only mode: reusing {len(existing)} existing checkpoints from {ckpt_dir}")
    else:
        save_checkpoint(model, 0, ckpt_dir, cfg)
        history = []
        for epoch in range(1, cfg.epochs + 1):
            lr = set_epoch_lr(optimizer, epoch, cfg)
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
            test_loss, test_acc = evaluate_backbone(model, test_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": lr,
                "epoch_seconds": time.time() - t0,
            }
            history.append(row)
            print(f"epoch={epoch:03d} train_acc={train_acc:.4f} test_acc={test_acc:.4f} lr={lr:.2e} time={row['epoch_seconds']:.1f}s")
            if epoch in save_epochs:
                save_checkpoint(model, epoch, ckpt_dir, cfg)
        pd.DataFrame(history).to_csv(out_dir / "backbone_history.csv", index=False)

    print("\nPosthoc analysis on frozen checkpoint features...")
    result_rows, geom_rows, pca_rows, conformal_rows, conformal_example_rows = [], [], [], [], []
    for ckpt_path in list_checkpoints(ckpt_dir):
        payload = torch.load(ckpt_path, map_location=device)
        epoch = int(payload["epoch"])
        model.load_state_dict(payload["model_state_dict"])
        print(f"\nCheckpoint epoch {epoch}")
        rows, geom, pca_df, conf_rows, conf_ex = evaluate_heads_for_checkpoint(epoch, model, train_eval_loader, test_loader, device, cfg)
        result_rows.extend(rows)
        geom_rows.append(geom)
        conformal_rows.extend(conf_rows)
        if conf_ex is not None:
            conformal_example_rows.append(conf_ex.sample(n=min(len(conf_ex), 5000), random_state=cfg.seed))
        if pca_df is not None:
            # Keep only a manageable number of PCA points in the CSV.
            pca_rows.append(pca_df.sample(n=min(len(pca_df), 1200), random_state=cfg.seed))

    results = pd.DataFrame(result_rows)
    geom = pd.DataFrame(geom_rows).sort_values("epoch")
    results.to_csv(out_dir / "head_results.csv", index=False)
    geom.to_csv(out_dir / "separability_metrics.csv", index=False)
    conformal = pd.DataFrame(conformal_rows) if conformal_rows else pd.DataFrame()
    conformal_examples = pd.concat(conformal_example_rows, ignore_index=True) if conformal_example_rows else pd.DataFrame()
    if not conformal.empty:
        conformal.to_csv(out_dir / "conformal_results.csv", index=False)
    if not conformal_examples.empty:
        conformal_examples.to_csv(out_dir / "conformal_examples.csv", index=False)
    if pca_rows:
        pd.concat(pca_rows, ignore_index=True).to_csv(out_dir / "pca_points.csv", index=False)

    save_plots(results, geom, out_dir, conformal=conformal, conformal_examples=conformal_examples)
    write_summary(results, geom, out_dir, conformal=conformal)

    print("\nSaved:")
    print(f"  {out_dir / 'backbone_history.csv'}")
    print(f"  {out_dir / 'head_results.csv'}")
    print(f"  {out_dir / 'separability_metrics.csv'}")
    if not conformal.empty:
        print(f"  {out_dir / 'conformal_results.csv'}")
    if not conformal_examples.empty:
        print(f"  {out_dir / 'conformal_examples.csv'}")
    print(f"  {out_dir / 'summary.md'}")
    print(f"  {out_dir / 'figures'}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--data-root", default="/content/data")
    p.add_argument("--out-dir", default="/content/runs/linsep_heads_mnist")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--train-subset", type=int, default=-1)
    p.add_argument("--test-subset", type=int, default=-1)
    p.add_argument("--save-epochs", default="0,1,2,5,10,20")
    p.add_argument("--probe-train-samples", type=int, default=5000)
    p.add_argument("--probe-test-samples", type=int, default=2500)
    p.add_argument("--bootstrap-samples", type=int, default=1000)

    p.add_argument("--no-rbf-svm", action="store_true")
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--no-rf", action="store_true")
    p.add_argument("--no-knn", action="store_true")
    p.add_argument("--include-gp", action="store_true")
    p.add_argument("--gp-train-samples", type=int, default=800)
    p.add_argument("--rf-trees", type=int, default=300)

    p.add_argument("--no-conformal", action="store_true", help="Disable split-conformal uncertainty diagnostics.")
    p.add_argument("--conformal-alpha", type=float, default=0.10, help="Miscoverage level; 0.10 targets 90% coverage.")
    p.add_argument("--conformal-fit-fraction", type=float, default=0.60)
    p.add_argument("--conformal-selection-fraction", type=float, default=0.20)
    p.add_argument("--conformal-calib-fraction", type=float, default=0.20)
    p.add_argument("--no-conformal-examples", action="store_true", help="Do not save per-example conformal diagnostics.")
    p.add_argument("--conformal-score", type=str, default="rank", choices=["rank", "lac"], help="rank gives nonempty top-label sets; lac uses 1-class-score and may be empty.")
    p.add_argument("--conformal-calibration-source", type=str, default="holdout", choices=["holdout", "train"], help="holdout splits the held-out/test pool into calibration/evaluation; train uses calibration from training features.")
    p.add_argument("--conformal-holdout-calib-fraction", type=float, default=0.50, help="Fraction of held-out pool used for calibration when source=holdout.")
    p.add_argument("--coverage-tolerance", type=float, default=0.01, help="Tolerance for marking conformal rows valid_at_target.")

    p.add_argument("--posthoc-only", action="store_true", help="Reuse existing checkpoints in --out-dir/checkpoints and skip training.")
    return p


def main():
    args = build_argparser().parse_args()
    # Dataset-aware defaults for CIFAR if the user does not override.
    warmup = args.warmup_epochs
    if args.dataset == "cifar10" and warmup == 0:
        warmup = 5
    cfg = CFG(
        dataset=args.dataset,
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_epochs=warmup,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        save_epochs=args.save_epochs,
        probe_train_samples=args.probe_train_samples,
        probe_test_samples=args.probe_test_samples,
        bootstrap_samples=args.bootstrap_samples,
        include_rbf_svm=not args.no_rbf_svm,
        include_mlp=not args.no_mlp,
        include_rf=not args.no_rf,
        include_knn=not args.no_knn,
        include_gp=args.include_gp,
        gp_train_samples=args.gp_train_samples,
        rf_trees=args.rf_trees,
        enable_conformal=not args.no_conformal,
        conformal_alpha=args.conformal_alpha,
        conformal_fit_fraction=args.conformal_fit_fraction,
        conformal_selection_fraction=args.conformal_selection_fraction,
        conformal_calib_fraction=args.conformal_calib_fraction,
        save_conformal_examples=not args.no_conformal_examples,
        conformal_score=args.conformal_score,
        conformal_calibration_source=args.conformal_calibration_source,
        conformal_holdout_calib_fraction=args.conformal_holdout_calib_fraction,
        coverage_tolerance=args.coverage_tolerance,
        posthoc_only=args.posthoc_only,
    )
    run(cfg)


if __name__ == "__main__":
    main()