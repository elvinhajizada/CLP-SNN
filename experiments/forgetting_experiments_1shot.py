#!/usr/bin/env python
"""
Continual Learning Forgetting Experiments (1-shot scenario)

Evaluates multiple continual learning classifiers on the OpenLoris dataset,
measuring catastrophic forgetting using BWT and FM metrics. Includes comprehensive
visualization of accuracy evolution and per-class forgetting patterns.

Date: 2026
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

# Paths
REPO_ROOT = Path(__file__).parent.parent
MASTER_DATA_FOLDER = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FORGETTING_RESULTS_DIR = RESULTS_DIR / "forgetting"
IMAGES_DIR = REPO_ROOT / "images"

# Create directories if they don't exist
FORGETTING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Experiment parameters
DATASET_NAME = "openloris"
NUM_CLASSES = 40
FEATURE_SIZE = 1280
DEVICE = "cpu"
SEEDS = [10, 20, 30]
N_BASE_CLASSES = 10

# Shot configuration
K_SHOTS = 1  # 1-shot learning
N_EPOCHS = 1

# Random seed for reproducibility
RANDOM_SEED = 42

# Set False to skip .pdf output and save only .png (faster on headless servers)
SAVE_PDF = True

# Classifier configuration
CLASSIFIER_TYPES = [
    "perceptron",
    "fine_tune",
    "ncm",
    "replay_20",
    "slda",
    "slda_const_sigma",
    "clp",
    "clp_snn",
]

# Matplotlib configuration
PLOT_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1,
    "axes.linewidth": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "savefig.dpi": 600,
    "figure.figsize": (3.5, 2.5),
    "svg.fonttype": "none",
}

SEABORN_PALETTE = sns.color_palette("tab10")
CLASSIFIER_COLORS = [
    '#606060',
    '#B0B0B0',
    SEABORN_PALETTE[2],
    SEABORN_PALETTE[3],
    SEABORN_PALETTE[1],
    SEABORN_PALETTE[1],
    SEABORN_PALETTE[9],
    SEABORN_PALETTE[9],
]
CLASSIFIER_LINESTYLES = ["-", "-", "-", "-", "--", ":", "-", "-."]
CLASSIFIER_DISPLAY_NAMES = [
    "Perceptron",
    "Fine Tuning",
    "NCM",
    "Replay",
    "SLDA",
    "SLDA (Frozen Σ)",
    "CLP",
    "CLP-SNN",
]

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────

# Add repo root to path for imports
sys.path.insert(0, str(REPO_ROOT))

# Import models
from models.CLP import ContinuallyLearningPrototypes
from models.CLP_SNN import CLPSNN
from models.SLDA import StreamingLDA
from models.NCM import NearestClassMean
from models.Replay import StreamingSoftmax
from models.Perceptron import Perceptron

# Configure matplotlib
plt.rcParams.update(PLOT_CONFIG)

# Verify Arial font availability
available_fonts = {f.name for f in fm.fontManager.ttflist}
if "Arial" in available_fonts:
    print("✓ Arial font is available")
else:
    print("⚠ Arial not found, using system default sans-serif")

# Seed random number generators
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")


# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


def load_and_prepare_test_data(
    data_folder: Path, n_test_samples: int = 2400
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and preprocess test data."""
    print("\n[*] Loading test data...")

    X_test = np.load(data_folder / "X_test.npy")
    y_test = np.load(data_folder / "y_test.npy")

    unique_classes, class_counts = np.unique(y_test, return_counts=True)
    samples_per_class = n_test_samples // len(unique_classes)

    # Balance test set
    balanced_indices = []
    for cls in unique_classes:
        class_indices = np.where(y_test == cls)[0]
        chosen_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )
        balanced_indices.extend(chosen_indices)

    y_test = y_test[balanced_indices]
    X_test = X_test[balanced_indices]

    X_test_raw_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Normalize test set
    X_test_norm = X_test / np.expand_dims(np.linalg.norm(X_test, axis=1), axis=1)
    X_test_norm_t = torch.tensor(X_test_norm, dtype=torch.float32)

    print(f"  Test set: {X_test.shape[0]} samples, {len(unique_classes)} classes")
    return X_test_raw_t, X_test_norm_t, y_test_t, y_test


def instantiate_classifier(
    classifier_type: str,
    feature_size: int,
    num_classes: int,
    device: str,
) -> object:
    """
    Instantiate a classifier based on type.

    Parameters
    ----------
    classifier_type : str
        Type of classifier to instantiate
    feature_size : int
        Input feature dimension
    num_classes : int
        Number of output classes
    device : str
        Device to run on ('cpu' or 'cuda')

    Returns
    -------
    object
        Instantiated classifier

    Raises
    ------
    ValueError
        If classifier_type is unknown
    """
    if classifier_type == "slda":
        return StreamingLDA(
            feature_size,
            num_classes,
            backbone=None,
            shrinkage_param=1e-4,
            streaming_update_sigma=True,
            device=device,
        )

    elif classifier_type == "slda_const_sigma":
        return StreamingLDA(
            feature_size,
            num_classes,
            backbone=None,
            shrinkage_param=1e-4,
            streaming_update_sigma=False,
            device=device,
        )

    elif classifier_type == "ncm":
        return NearestClassMean(feature_size, num_classes, backbone=None, device=device)

    elif classifier_type == "fine_tune":
        return StreamingSoftmax(
            feature_size,
            num_classes,
            use_replay=False,
            backbone=None,
            lr=0.001,
            weight_decay=1e-5,
            device=device,
        )

    elif classifier_type == "replay_20":
        return StreamingSoftmax(
            feature_size,
            num_classes,
            use_replay=True,
            backbone=None,
            lr=0.001,
            weight_decay=1e-5,
            replay_samples=50,
            max_buffer_size=800,
            device=device,
        )

    elif classifier_type == "perceptron":
        return Perceptron(feature_size, num_classes, backbone=None, device=device)

    elif classifier_type == "clp":
        return ContinuallyLearningPrototypes(
            feature_size,
            n_protos=400,
            num_classes=num_classes,
            backbone=None,
            alpha_init=1,
            sim_th_init=0.75,
            n_wta=1,
            k_hit=1,
            k_miss=1,
            adaptive_th=False,
            learn_outliers=True,
            device=device,
        )

    elif classifier_type == "clp_snn":
        return CLPSNN(
            feature_size,
            n_protos=600,
            num_classes=num_classes,
            threshold=0.9,
            g_inc=0.5,
            use_quantization=True,
            device=device,
        )

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


# ────────────────────────────────────────────────────────────────────────────
# 1-Shot Learning Experiment
# ────────────────────────────────────────────────────────────────────────────


def run_1shot_experiment() -> Tuple[np.ndarray, np.ndarray]:
    """
    Run 1-shot learning experiment.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (accuracies, R_matrix)
    """
    print("\n" + "=" * 80)
    print("STARTING 1-SHOT LEARNING EXPERIMENT")
    print("=" * 80)

    data_folder = MASTER_DATA_FOLDER / "1shot"

    # Load test data
    X_test_raw_t, X_test_norm_t, y_test_t, _ = load_and_prepare_test_data(
        MASTER_DATA_FOLDER, n_test_samples=NUM_CLASSES * 60
    )

    # Initialize result arrays
    accuracies = np.zeros((NUM_CLASSES, len(CLASSIFIER_TYPES), len(SEEDS)))
    base_accs = np.zeros((NUM_CLASSES - N_BASE_CLASSES + 1, len(CLASSIFIER_TYPES), len(SEEDS)))
    per_cls_accs = np.zeros((NUM_CLASSES, NUM_CLASSES))
    R = np.full((NUM_CLASSES, NUM_CLASSES, len(CLASSIFIER_TYPES), len(SEEDS)), np.nan)

    # Main loop over seeds
    for s, seed in enumerate(SEEDS):
        print(f"\n[*] Running seed {s+1}/{len(SEEDS)} (seed={seed})...")

        X_train = torch.load(
            data_folder / f"X_train_{K_SHOTS}_shot_{seed}.pt", map_location=DEVICE
        )
        y_train = torch.load(
            data_folder / f"y_train_{K_SHOTS}_shot_{seed}.pt", map_location=DEVICE
        )

        X_train_norm = X_train / X_train.norm(dim=1, keepdim=True)

        train_set_raw = TensorDataset(X_train, y_train)
        train_set_norm = TensorDataset(X_train_norm, y_train)

        train_loader_raw = DataLoader(
            train_set_raw, batch_size=1, num_workers=0, pin_memory=False, shuffle=False
        )
        train_loader_norm = DataLoader(
            train_set_norm, batch_size=1, num_workers=0, pin_memory=False, shuffle=False
        )

        class_order = torch.unique_consecutive(y_train).tolist()
        base_cls_labels = class_order[:N_BASE_CLASSES]

        counts = torch.stack([(y_train == label).sum() for label in class_order]).cpu()
        shots = np.cumsum(counts)

        # Loop over classifiers
        for c, classifier_type in enumerate(CLASSIFIER_TYPES):
            print(f"  [{c+1}/{len(CLASSIFIER_TYPES)}] {classifier_type}...", end=" ", flush=True)

            classifier = instantiate_classifier(
                classifier_type, FEATURE_SIZE, NUM_CLASSES, DEVICE
            )

            # Choose data loaders based on classifier
            if classifier_type == "replay_20":
                train_loader = train_loader_raw
                X_test_t = X_test_raw_t
            else:
                train_loader = train_loader_norm
                X_test_t = X_test_norm_t

            i = 0
            check_point = -1

            # Stream training samples
            for x, y in train_loader:
                classifier.fit(x.view(FEATURE_SIZE), y.view(1), i)
                i += 1

                # Checkpoint: evaluate after each class
                if i in shots:
                    check_point += 1

                    # Evaluate on learned classes
                    labels_learned = class_order[: check_point + 1]
                    mask = torch.isin(y_test_t, torch.tensor(labels_learned))

                    X_test_filtered = X_test_t[mask]
                    y_test_filtered = y_test_t[mask]

                    probas = classifier.predict(X_test_filtered)
                    _, pred = probas.topk(1, 1, True, True)
                    y_pred = pred.t().cpu().numpy().squeeze()

                    acc = metrics.accuracy_score(y_test_filtered.numpy(), y_pred)
                    accuracies[check_point, c, s] = acc

                    # Confusion matrix and per-class accuracy
                    cm = metrics.confusion_matrix(
                        y_test_filtered.numpy(), y_pred, labels=class_order[: check_point + 1]
                    )
                    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
                    per_cls_accs[: check_point + 1, check_point] = per_class_accuracy

                    # Fill R matrix
                    R[check_point, : check_point + 1, c, s] = per_class_accuracy

                    # Base class accuracy (for fairness metric)
                    if check_point >= N_BASE_CLASSES - 1:
                        mask_base = torch.isin(y_test_t, torch.tensor(base_cls_labels))
                        X_test_filtered_base = X_test_t[mask_base]
                        y_test_filtered_base = y_test_t[mask_base]

                        probas_base = classifier.predict(X_test_filtered_base)
                        _, pred_base = probas_base.topk(1, 1, True, True)
                        y_pred_base = pred_base.t().cpu().numpy().squeeze()

                        acc_base = metrics.accuracy_score(
                            y_test_filtered_base.numpy(), y_pred_base
                        )
                        base_accs[check_point - N_BASE_CLASSES + 1, c, s] = acc_base

            print("✓")

    print(f"\n[✓] 1-shot experiment completed!")
    return accuracies, R


# ────────────────────────────────────────────────────────────────────────────
# Forgetting Metrics Computation
# ────────────────────────────────────────────────────────────────────────────


def save_results(accuracies: np.ndarray, R: np.ndarray, label: str = "1shot") -> None:
    """
    Save core results to disk for later reuse.

    Parameters
    ----------
    accuracies : np.ndarray
        Accuracy matrix shape (n_steps, n_classifiers, n_seeds)
    R : np.ndarray
        R matrix shape (n_steps, n_classes, n_classifiers, n_seeds)
    label : str
        Label for the results (e.g., '1shot', '25shot')
    """
    filepath_acc = FORGETTING_RESULTS_DIR / f"accuracies_{label}.npy"
    filepath_r = FORGETTING_RESULTS_DIR / f"R_matrix_{label}.npy"

    np.save(filepath_acc, accuracies)
    np.save(filepath_r, R)

    print(f"[✓] Results saved: {filepath_acc.name}, {filepath_r.name}")


def load_results(label: str = "1shot") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load core results from disk.

    Parameters
    ----------
    label : str
        Label for the results (e.g., '1shot', '25shot')

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (accuracies, R_matrix)
    """
    filepath_acc = FORGETTING_RESULTS_DIR / f"accuracies_{label}.npy"
    filepath_r = FORGETTING_RESULTS_DIR / f"R_matrix_{label}.npy"

    if not filepath_acc.exists() or not filepath_r.exists():
        raise FileNotFoundError(
            f"Results not found. Run without --use-cache flag first, or check {FORGETTING_RESULTS_DIR}"
        )

    accuracies = np.load(filepath_acc)
    R = np.load(filepath_r)

    print(f"[✓] Results loaded: {filepath_acc.name}, {filepath_r.name}")
    return accuracies, R


def compute_forgetting_metrics(
    R: np.ndarray, num_classes: int, num_classifiers: int, num_seeds: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute True-Peak FM, temporal FM curves, and per-class drop matrix from R matrix.

    FM_j = max_t R[t,j] − R[N-1,j]: drop from ever-best accuracy to final accuracy.
    Always ≥ 0; avoids the diagonal-anchor artefact where ramp-up methods (e.g. Replay)
    appear to have negative forgetting.

    Parameters
    ----------
    R : np.ndarray
        Accuracy matrix R[t, j, c, s]
    num_classes : int
        Number of classes
    num_classifiers : int
        Number of classifiers
    num_seeds : int
        Number of seeds

    Returns
    -------
    Tuple
        (fm_mean, fm_std, fm_curve_mean, fm_curve_std, per_class_drop_mat)
    """
    N = num_classes
    fm_all = np.full((num_classifiers, num_seeds), np.nan)
    fm_curve = np.full((N - 1, num_classifiers, num_seeds), np.nan)

    for c in range(num_classifiers):
        for s in range(num_seeds):
            # True-Peak FM: FM_j = max_t R[t,j] - R[N-1,j]
            fm_terms = []
            for j in range(N - 1):
                pk = np.nanmax(R[:, j, c, s])
                final = R[N - 1, j, c, s]
                if np.isnan(pk) or np.isnan(final):
                    continue
                fm_terms.append(pk - final)
            fm_all[c, s] = np.mean(fm_terms) if fm_terms else np.nan

            # Temporal FM curve: running True-Peak FM at each learning step t
            for t_idx, t in enumerate(range(1, N)):
                fm_t = []
                for j in range(t):
                    run_peak = np.nanmax(R[j : t + 1, j, c, s])
                    rtj = R[t, j, c, s]
                    if np.isnan(run_peak) or np.isnan(rtj):
                        continue
                    fm_t.append(run_peak - rtj)
                fm_curve[t_idx, c, s] = np.mean(fm_t) if fm_t else np.nan

    fm_mean = np.nanmean(fm_all, axis=1)
    fm_std = np.nanstd(fm_all, axis=1)
    fm_curve_mean = np.nanmean(fm_curve, axis=2)
    fm_curve_std = np.nanstd(fm_curve, axis=2)

    # Per-class drop: peak - final for each class
    peak_R = np.nanmax(R, axis=0)  # (N, n_cls, n_seeds)
    per_class_drop = np.nanmean(peak_R - R[N - 1, :, :, :], axis=-1)  # (N, n_cls)
    per_class_drop_mat = per_class_drop.T  # (n_cls, N)

    return fm_mean, fm_std, fm_curve_mean, fm_curve_std, per_class_drop_mat


# ────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ────────────────────────────────────────────────────────────────────────────


def plot_forgetting_metrics_combined(
    fm_mean: np.ndarray,
    fm_std: np.ndarray,
    per_class_drop_mat: np.ndarray,
    fm_curve_mean: np.ndarray,
    fm_curve_std: np.ndarray,
    num_classes: int,
) -> None:
    """
    Plot combined forgetting metrics: FM bar chart (left), per-class drop heatmap (middle),
    and temporal FM curve (right) in a 1x3 figure.
    Excludes SLDA (Frozen Σ) from display.

    Parameters
    ----------
    fm_mean : np.ndarray
        Mean FM per classifier (n_classifiers,)
    fm_std : np.ndarray
        Std FM per classifier (n_classifiers,)
    per_class_drop_mat : np.ndarray
        Per-class drop matrix shape (n_classifiers, n_classes)
    fm_curve_mean : np.ndarray
        Mean FM curve over seeds shape (n_steps, n_classifiers)
    fm_curve_std : np.ndarray
        Std FM curve over seeds shape (n_steps, n_classifiers)
    num_classes : int
        Number of classes
    """
    # Exclude SLDA (Frozen Σ) - index 5
    exclude_idx = 5
    display_indices = [i for i in range(len(CLASSIFIER_TYPES)) if i != exclude_idx]

    ct_labels = [
        "Perceptron",
        "Fine Tuning",
        "NCM",
        "Replay",
        "SLDA",
        "CLP",
        "CLP-SNN",
    ]

    x = np.arange(len(ct_labels))
    width = 0.35
    N = num_classes

    # Filter data to exclude frozen sigma
    fm_mean_filtered = fm_mean[display_indices]
    fm_std_filtered = fm_std[display_indices]
    per_class_drop_filtered = per_class_drop_mat[display_indices, :]
    fm_curve_mean_filtered = fm_curve_mean[:, display_indices]
    fm_curve_std_filtered = fm_curve_std[:, display_indices]

    fig, axes = plt.subplots(1, 3, figsize=(7.087, 2.8))
    fig.patch.set_alpha(0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1: Forgetting Metric Bar Chart (FM only, no BWT, with error bars)
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    colors_filtered = [CLASSIFIER_COLORS[i] for i in display_indices]
    ax.bar(
        x,
        fm_mean_filtered,
        width,
        yerr=fm_std_filtered,
        capsize=3,
        color=colors_filtered,
        error_kw=dict(elinewidth=0.75, ecolor="black"),
    )
    # Annotate each FM bar with mean only (no std labels)
    for i, val in enumerate(fm_mean_filtered):
        y_pos = val + 0.025
        ax.text(
            x[i],
            y_pos,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="black",
        )
    y_hi_fm = float(fm_mean_filtered.max())
    ax.set_ylim(-0.02, y_hi_fm * 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ct_labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Forgetting Metric", fontsize=7)
    ax.set_title("Forgetting Metric\n(↓ lower = better)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2: Per-Class Drop Heatmap (peak - final for each class)
    # ─────────────────────────────────────────────────────────────────────────
    # per_class_drop_filtered shape: (7_classifiers, N)
    # Rows = classifiers, Columns = classes
    ax = axes[1]
    im = ax.imshow(
        per_class_drop_filtered,
        aspect="auto",
        origin="upper",
        vmin=0,
        vmax=0.6,
        cmap="Reds",
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(ct_labels)))
    ax.set_yticklabels(ct_labels, fontsize=6)
    ax.set_xticks(np.arange(0, N, 4))
    ax.set_xticklabels(np.arange(1, N + 1, 4), fontsize=6)
    ax.set_xlabel("Class (introduction order)", fontsize=7)
    ax.set_title("Per-Class Drop\n(peak − final)", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.set_label("Δ acc", fontsize=6)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3: Temporal Forgetting Metric Curve
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[2]
    t_axis = np.arange(2, N + 1)  # 2..40 classes seen

    linestyles_filtered = [CLASSIFIER_LINESTYLES[i] for i in display_indices]
    for idx, i in enumerate(display_indices):
        ax.plot(
            t_axis,
            fm_curve_mean_filtered[:, idx],
            color=CLASSIFIER_COLORS[i],
            linestyle=linestyles_filtered[idx],
            linewidth=1.2 if i == 7 else 1.0,
            label=ct_labels[idx],
            alpha=0.9,
        )
        ax.fill_between(
            t_axis,
            fm_curve_mean_filtered[:, idx] - fm_curve_std_filtered[:, idx],
            fm_curve_mean_filtered[:, idx] + fm_curve_std_filtered[:, idx],
            color=CLASSIFIER_COLORS[i],
            alpha=0.08,
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=5.5)
    ax.set_xlabel("# Classes Seen", fontsize=7)
    ax.set_ylabel("Forgetting Metric", fontsize=7)
    ax.set_title("Temporal Forgetting Metric\n(evolution over learning)", fontsize=7)
    ax.set_xlim([2, 40])
    ax.set_xticks(np.arange(2, 41, 8))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)

    plt.tight_layout(pad=0.4, w_pad=0.3)
    if SAVE_PDF:
        plt.savefig(
            IMAGES_DIR / "forgetting_metrics_combined_1shot.pdf", format="pdf", bbox_inches="tight"
        )
    plt.savefig(
        IMAGES_DIR / "forgetting_metrics_combined_1shot.png", format="png", dpi=600, bbox_inches="tight"
    )
    print(f"[✓] Saved: forgetting_metrics_combined_1shot.* (fm_chart + per_class_drop + temporal_curve)")
    plt.close()


def plot_forgetting_metrics_bar_chart(fm_mean: np.ndarray, fm_std: np.ndarray) -> None:
    """Plot True-Peak FM as a bar chart for all 8 classifiers."""
    ct_labels = [
        "Perceptron",
        "Fine Tuning",
        "NCM",
        "Replay",
        "SLDA",
        "SLDA\n(Frozen Σ)",
        "CLP",
        "CLP-SNN",
    ]

    x = np.arange(len(ct_labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.5))
    fig.patch.set_alpha(0.0)

    ax.bar(
        x,
        fm_mean,
        width,
        yerr=fm_std,
        capsize=3,
        color=CLASSIFIER_COLORS,
        error_kw=dict(elinewidth=0.75, ecolor="black"),
    )
    for i, (val, std) in enumerate(zip(fm_mean, fm_std)):
        y_pos = val + abs(std) + 0.025
        ax.text(x[i], y_pos, f"{val:.2f}\n±{std:.2f}",
                ha="center", va="bottom", fontsize=4.8, color="black", linespacing=1.1)
    y_hi_fm = float((fm_mean + fm_std).max())
    ax.set_ylim(-0.02, y_hi_fm * 1.65)
    ax.set_xticks(x)
    ax.set_xticklabels(ct_labels, rotation=15, ha="right")
    ax.set_ylabel("True-Peak FM")
    ax.set_title("True-Peak Forgetting Measure\n(↓ lower = less forgetting)")

    plt.tight_layout()
    if SAVE_PDF:
        plt.savefig(IMAGES_DIR / "forgetting_metrics_1shot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(IMAGES_DIR / "forgetting_metrics_1shot.png", format="png", dpi=600, bbox_inches="tight")
    print(f"[✓] Saved: forgetting_metrics_1shot.*")
    plt.close()


def plot_temporal_evolution(R: np.ndarray, num_classes: int) -> None:
    """Plot temporal evolution of True-Peak FM curve."""
    N = num_classes
    n_cls = R.shape[2]
    n_s = R.shape[3]

    tp_fm_curve = np.full((N - 1, n_cls, n_s), np.nan)

    for t_idx, t in enumerate(range(1, N)):
        for c in range(n_cls):
            for s in range(n_s):
                tp_fm_t = []
                for j in range(t):
                    rtj = R[t, j, c, s]
                    run_peak = np.nanmax(R[j : t + 1, j, c, s])
                    if not np.isnan(run_peak) and not np.isnan(rtj):
                        tp_fm_t.append(run_peak - rtj)
                tp_fm_curve[t_idx, c, s] = np.mean(tp_fm_t) if tp_fm_t else np.nan

    tp_fm_mean = np.nanmean(tp_fm_curve, axis=2)
    tp_fm_std = np.nanstd(tp_fm_curve, axis=2)

    t_axis = np.arange(1, N) + 1

    fig, ax = plt.subplots(1, 1, figsize=(7.087, 2.5))
    fig.patch.set_alpha(0.0)

    for i in range(n_cls):
        ax.plot(
            t_axis,
            tp_fm_mean[:, i],
            color=CLASSIFIER_COLORS[i],
            linestyle=CLASSIFIER_LINESTYLES[i],
            linewidth=1.2 if i == n_cls - 1 else 1.0,
            label=CLASSIFIER_DISPLAY_NAMES[i],
            alpha=0.9,
        )
        ax.fill_between(
            t_axis,
            tp_fm_mean[:, i] - tp_fm_std[:, i],
            tp_fm_mean[:, i] + tp_fm_std[:, i],
            color=CLASSIFIER_COLORS[i],
            alpha=0.08,
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=5.5)
    ax.set_xlabel("# Classes Seen", fontsize=7)
    ax.set_ylabel("True-Peak FM", fontsize=7)
    ax.set_title("Temporal True-Peak Forgetting Measure", fontsize=7)
    ax.set_xlim([2, 40])
    ax.set_xticks(np.arange(2, 41, 8))

    plt.tight_layout()
    if SAVE_PDF:
        plt.savefig(IMAGES_DIR / "forgetting_curves_1shot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(IMAGES_DIR / "forgetting_curves_1shot.png", format="png", dpi=600, bbox_inches="tight")
    print(f"[✓] Saved: forgetting_curves_1shot.*")
    plt.close()


def plot_per_class_forgetting(R: np.ndarray, num_classes: int) -> None:
    """Plot per-class forgetting at end of stream."""
    N = num_classes
    n_cls = R.shape[2]

    # True-Peak FM per class: max_t R[t,j] - R[N-1,j]
    peak_R_per_class = np.nanmax(R, axis=0)  # (N, n_cls, n_seeds)
    per_class_frgt = np.nanmean(
        peak_R_per_class - R[N - 1, :, :, :], axis=-1
    )  # (N, n_cls)

    fig, ax = plt.subplots(figsize=(7.087, 2.5))
    fig.patch.set_alpha(0.0)

    class_x = np.arange(N)
    for i in range(n_cls):
        ax.plot(
            class_x + 1,
            per_class_frgt[:, i],
            color=CLASSIFIER_COLORS[i],
            linestyle=CLASSIFIER_LINESTYLES[i],
            linewidth=1.0,
            label=CLASSIFIER_DISPLAY_NAMES[i],
            alpha=0.85,
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Class (in order of introduction)")
    ax.set_ylabel("True-Peak FM per class\n(max_t R[t,j] − R[N-1,j])")
    ax.set_title("Per-class True-Peak Forgetting at End of Stream (1-shot, seed-averaged)")
    ax.set_xlim([1, N])
    ax.set_xticks(np.arange(1, N + 1, 4))
    ax.legend(loc="upper right", frameon=False, ncol=2)

    plt.tight_layout()
    if SAVE_PDF:
        plt.savefig(
            IMAGES_DIR / "per_class_forgetting_1shot.pdf", format="pdf", bbox_inches="tight"
        )
    plt.savefig(
        IMAGES_DIR / "per_class_forgetting_1shot.png", format="png", dpi=600, bbox_inches="tight"
    )
    print(f"[✓] Saved: per_class_forgetting_1shot.*")
    plt.close()

    # R matrix heatmap
    R_avg = np.nanmean(R, axis=-1)
    hm_idx = [0, 1, 2, 3, 4, 6, 7]
    hm_labels = [CLASSIFIER_DISPLAY_NAMES[i] for i in hm_idx]
    n_hm = len(hm_idx)

    fig, axes = plt.subplots(1, n_hm, figsize=(7.087, 2.8), sharey=True)
    fig.patch.set_alpha(0.0)

    for col, ci in enumerate(hm_idx):
        ax = axes[col]
        mat = R_avg[:, :, ci]
        im = ax.imshow(mat, aspect="auto", origin="upper", vmin=0, vmax=1, cmap="RdYlGn")

        for j in range(N):
            ax.plot(j, j, "k.", markersize=1.5, alpha=0.6)

        ax.set_title(hm_labels[col], fontsize=6, pad=2)
        ax.set_xlabel("Class j", fontsize=6)
        ax.tick_params(labelsize=6)

        if col == 0:
            ax.set_ylabel("Learning step t")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.18, shrink=0.6, aspect=30)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Accuracy", fontsize=6)

    fig.suptitle(
        "R[t, j]: accuracy of class j at learning step t (seed-averaged)\n"
        "Dots mark the diagonal (class j tested right after first introduction)",
        fontsize=7,
        y=1.02,
    )

    if SAVE_PDF:
        plt.savefig(IMAGES_DIR / "R_matrix_heatmap_1shot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(IMAGES_DIR / "R_matrix_heatmap_1shot.png", format="png", dpi=600, bbox_inches="tight")
    print(f"[✓] Saved: R_matrix_heatmap_1shot.*")
    plt.close()


def plot_per_class_accuracy_heatmap(
    R: np.ndarray,
    num_classes: int,
    fm_mean: np.ndarray = None,
    fm_std: np.ndarray = None,
) -> None:
    """Plot per-class accuracy evolution heatmap.

    Parameters
    ----------
    fm_mean, fm_std : optional
        True-Peak FM per classifier (shape: n_classifiers).  When provided,
        each subplot title shows the FM value so the figure is self-contained.
    """
    N = num_classes
    n_cls = R.shape[2]

    R_seed = np.nanmean(R, axis=-1)

    # Mask upper triangle (classes not yet introduced)
    raw_acc = R_seed.copy()
    for t in range(N):
        for j in range(t + 1, N):
            raw_acc[t, j, :] = np.nan

    hm_idx = [0, 1, 2, 3, 4, 6, 7]
    hm_labels = [CLASSIFIER_DISPLAY_NAMES[i] for i in hm_idx]
    n_hm = len(hm_idx)

    fig, axes = plt.subplots(1, n_hm, figsize=(7.087, 2.8), sharey=True)
    fig.patch.set_alpha(0.0)

    for col, ci in enumerate(hm_idx):
        ax = axes[col]
        mat = raw_acc[:, :, ci].T
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            extent=[1, N, N, 0],
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
            interpolation="nearest",
        )

        ax.plot(np.arange(1, N + 1), np.arange(N), "k-", linewidth=0.6, alpha=0.5)

        # Title: classifier name + FM value if available
        if fm_mean is not None and fm_std is not None:
            title = f"{hm_labels[col]}\nFM={fm_mean[ci]:.2f}±{fm_std[ci]:.2f}"
        else:
            title = hm_labels[col]
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel("Step t", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xticks(np.arange(0, N + 1, 10))
        ax.set_yticks(np.arange(0, N, 10))

        if col == 0:
            ax.set_ylabel("Class j")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.18, shrink=0.6, aspect=30)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Accuracy R[t, j]", fontsize=6)

    fig.suptitle(
        "Per-class accuracy evolution R[t,j] (seed-averaged)\n"
        "Black line = introduction step  |  green = high acc,  yellow = moderate,  red = low/forgotten",
        fontsize=7,
        y=1.02,
    )

    if SAVE_PDF:
        plt.savefig(IMAGES_DIR / "per_class_acc_heatmap_1shot.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(IMAGES_DIR / "per_class_acc_heatmap_1shot.png", format="png", dpi=600, bbox_inches="tight")
    print(f"[✓] Saved: per_class_acc_heatmap_1shot.*")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────


def main(use_cache: bool = False) -> None:
    """Main entry point.

    Parameters
    ----------
    use_cache : bool
        If True, load cached results instead of rerunning experiment.
    """
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING FORGETTING EXPERIMENTS (1-SHOT)")
    print("=" * 80)

    # Load or run 1-shot experiment
    if use_cache:
        print("\n[*] Loading cached results...")
        accuracies, R = load_results("1shot")
    else:
        accuracies, R = run_1shot_experiment()
        print("\n[*] Saving results for future use...")
        save_results(accuracies, R, "1shot")

    # Compute forgetting metrics
    print("\n[*] Computing forgetting metrics...")
    fm_mean, fm_std, fm_curve_mean, fm_curve_std, per_class_drop_mat = compute_forgetting_metrics(
        R, NUM_CLASSES, len(CLASSIFIER_TYPES), len(SEEDS)
    )

    print(f"\n{'Classifier':<22} {'True-Peak FM (lower = better)':>30}")
    print("-" * 54)
    for c, ct in enumerate(CLASSIFIER_TYPES):
        print(f"{ct:<22} {fm_mean[c]:.4f} ± {fm_std[c]:.4f}")

    # Generate visualizations
    print("\n[*] Generating visualizations...")
    plot_forgetting_metrics_combined(fm_mean, fm_std, per_class_drop_mat, fm_curve_mean, fm_curve_std, NUM_CLASSES)
    plot_forgetting_metrics_bar_chart(fm_mean, fm_std)
    plot_temporal_evolution(R, NUM_CLASSES)
    plot_per_class_forgetting(R, NUM_CLASSES)
    plot_per_class_accuracy_heatmap(R, NUM_CLASSES, fm_mean=fm_mean, fm_std=fm_std)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR.resolve()}")
    print(f"Plots saved to: {IMAGES_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continual Learning Forgetting Experiments (1-shot scenario)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached results instead of rerunning experiment",
    )
    args = parser.parse_args()
    main(use_cache=args.use_cache)
