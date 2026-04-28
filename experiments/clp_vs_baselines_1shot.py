"""
clp_vs_baselines_1shot.py
-------------------------
1-shot OpenLoris continual-learning accuracy comparison of CLP and CLP-SNN
against six baseline classifiers (Fig. 3a, Table 1 in
"Real-time Continual Learning on Intel Loihi 2").

Protocol:
  * 40 classes x 60 frames per class (pre-balanced), one frame fit at a time.
  * Class-incremental order: file is pre-ordered by class, so each class
    boundary corresponds to seeing all 60 frames of one new class.
  * After every class boundary, evaluate on a 60-per-class balanced test set
    filtered to the classes seen so far -> 40 checkpoints.
  * Mean +/- std across seeds [10, 20, 30].

Hyperparameters match those used in `forgetting_experiments_1shot.py`
(the paper-chosen 1-shot configs).

Outputs saved to experiments/results/clp_vs_baselines_1shot/:
  accuracies.npz                   - raw (40, 8, 3) array + variant names
  acc_trend_1shot_baselines.png/pdf - mean line + std shading per variant
  final_accuracy_table.csv
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Make repo root importable when running from experiments/
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.CLP import ContinuallyLearningPrototypes  # noqa: E402
from models.CLP_SNN import CLPSNN  # noqa: E402
from models.SLDA import StreamingLDA  # noqa: E402
from models.NCM import NearestClassMean  # noqa: E402
from models.Replay import StreamingSoftmax  # noqa: E402
from models.Perceptron import Perceptron  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = _REPO / "data"
TRAIN_DIR = DATA_DIR / "1shot"
TEST_X_PATH = DATA_DIR / "X_test.npy"
TEST_Y_PATH = DATA_DIR / "y_test.npy"

RESULTS_DIR = _REPO / "experiments" / "results" / "clp_vs_baselines_1shot"

SEEDS = [10, 20, 30]
NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60         # frames per class
N_STEPS = NUM_CLASSES  # 40 checkpoints, one per class boundary
DEVICE = "cpu"
GLOBAL_SEED = 42
SAVE_PDF = True   # set False to skip .pdf output and save only .png

PLOT_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "lines.linewidth": 1,
    "axes.linewidth": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.dpi": 600,
    "figure.figsize": (3.5, 2.5),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_set():
    """Load OpenLoris test features, balance to 60/class, L2-normalise."""
    np.random.seed(GLOBAL_SEED)
    X = np.load(TEST_X_PATH)
    y = np.load(TEST_Y_PATH)
    unique_classes = np.unique(y)
    balanced_idx = []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        balanced_idx.extend(np.random.choice(cls_idx, N_FRAMES, replace=False))
    X = X[balanced_idx]
    y = y[balanced_idx]
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return (torch.from_numpy(X).float(),
            torch.from_numpy(X_norm).float(),
            torch.from_numpy(y).long())


def load_train_split(seed: int):
    """Load one seed's 1-shot train set; return raw X, normalised X, y, class boundaries."""
    X_raw = torch.load(TRAIN_DIR / f"X_train_1_shot_{seed}.pt",
                       map_location="cpu", weights_only=True).float()
    y = torch.load(TRAIN_DIR / f"y_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).long()
    X_norm = X_raw / X_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)

    class_order = torch.unique_consecutive(y).tolist()
    counts = torch.stack([(y == c).sum() for c in class_order]).cpu().numpy()
    boundaries = np.cumsum(counts)  # [60, 120, ..., 2400]
    return X_raw, X_norm, y, boundaries, class_order


# ── Classifier factory ────────────────────────────────────────────────────────

VARIANTS = [
    "Perceptron",
    "Fine Tuning",
    "NCM",
    "Replay",
    "SLDA",
    "SLDA (Frozen Σ)",
    "CLP",
    "CLP-SNN",
]


def build_classifier(name: str):
    """Instantiate one classifier with the paper-chosen 1-shot hyperparameters
    (matching forgetting_experiments_1shot.py)."""
    d = DEVICE
    if name == "Perceptron":
        return Perceptron(FEATURE_SIZE, NUM_CLASSES, backbone=None, device=d)

    elif name == "Fine Tuning":
        return StreamingSoftmax(
            FEATURE_SIZE, NUM_CLASSES, use_replay=False,
            backbone=None, lr=0.001, weight_decay=1e-5, device=d,
        )

    elif name == "NCM":
        return NearestClassMean(FEATURE_SIZE, NUM_CLASSES, backbone=None, device=d)

    elif name == "Replay":
        return StreamingSoftmax(
            FEATURE_SIZE, NUM_CLASSES, use_replay=True, backbone=None,
            lr=0.001, weight_decay=1e-5, replay_samples=50,
            max_buffer_size=800, device=d,
        )

    elif name == "SLDA":
        return StreamingLDA(
            FEATURE_SIZE, NUM_CLASSES, backbone=None,
            shrinkage_param=1e-4, streaming_update_sigma=True, device=d,
        )

    elif name == "SLDA (Frozen Σ)":
        return StreamingLDA(
            FEATURE_SIZE, NUM_CLASSES, backbone=None,
            shrinkage_param=1e-4, streaming_update_sigma=False, device=d,
        )

    elif name == "CLP":
        return ContinuallyLearningPrototypes(
            FEATURE_SIZE, n_protos=400, num_classes=NUM_CLASSES,
            backbone=None, alpha_init=1, sim_th_init=0.75, n_wta=1,
            k_hit=1, k_miss=1, adaptive_th=False,
            learn_outliers=True, device=d,
        )

    elif name == "CLP-SNN":
        return CLPSNN(
            FEATURE_SIZE, n_protos=600, num_classes=NUM_CLASSES,
            threshold=0.9, g_inc=0.5, use_quantization=True, device=d,
        )

    raise ValueError(f"Unknown variant: {name!r}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_seen_only(clf, X_test_raw, X_test_norm, y_test, seen_classes, use_raw):
    """Evaluate accuracy on the test subset whose labels are in `seen_classes`."""
    X = X_test_raw if use_raw else X_test_norm
    mask = torch.zeros_like(y_test, dtype=torch.bool)
    for c in seen_classes:
        mask |= (y_test == c)
    if mask.sum() == 0:
        return 0.0
    Xs, ys = X[mask], y_test[mask]
    probas = clf.predict(Xs)
    _, pred = probas.topk(1, 1, True, True)
    y_pred = pred.t().cpu().numpy().squeeze()
    return metrics.accuracy_score(ys.numpy(), y_pred)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(PLOT_CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = _REPO / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    print("Loading test set...")
    X_test_raw, X_test_norm, y_test = load_test_set()
    print(f"  test set: X={tuple(X_test_raw.shape)}, y={tuple(y_test.shape)}")

    accuracies = np.zeros((N_STEPS, len(VARIANTS), len(SEEDS)))

    for s, seed in enumerate(SEEDS):
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        X_tr_raw, X_tr, y_tr, boundaries, class_order = load_train_split(seed)
        print(f"  train: {len(y_tr)} samples, {len(boundaries)} class boundaries")

        for c, name in enumerate(VARIANTS):
            print(f"  [{c+1}/{len(VARIANTS)}] {name}...", end=" ", flush=True)
            clf = build_classifier(name)
            use_raw = (name in ("Replay"))
            X_train = X_tr_raw if (name == "Replay") else X_tr

            chk = 0
            for i, (xi, yi) in enumerate(zip(X_train, y_tr), start=1):
                clf.fit(xi.view(FEATURE_SIZE), yi.view(1), i - 1)
                if i in boundaries and chk < N_STEPS:
                    seen = class_order[: chk + 1]
                    acc = evaluate_seen_only(
                        clf, X_test_raw, X_test_norm, y_test, seen, use_raw
                    )
                    accuracies[chk, c, s] = acc
                    chk += 1

            print(f"final={accuracies[-1, c, s]*100:.1f}%")

    # ── Save raw results ──────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "accuracies.npz",
        accuracies=accuracies,
        variant_names=np.array(VARIANTS),
    )

    mean_acc = accuracies.mean(axis=2)
    std_acc = accuracies.std(axis=2)

    import csv
    with open(RESULTS_DIR / "final_accuracy_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Mean (%)", "Std (%)"])
        for c, name in enumerate(VARIANTS):
            w.writerow([name, f"{mean_acc[-1,c]*100:.2f}", f"{std_acc[-1,c]*100:.2f}"])

    # ── Plot ──────────────────────────────────────────────────────────────────
    p = sns.color_palette("tab10")
    colors = ['#606060', '#B0B0B0', p[2], p[3], p[1], p[1], p[9], p[9]]
    dashes = ["-", "-", "-", "-", "--", ":", "-", "-."]

    t = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for c, name in enumerate(VARIANTS):
        label = f"{name}: {mean_acc[-1,c]*100:.1f}±{std_acc[-1,c]*100:.1f}%"
        ax.plot(t, mean_acc[:, c], color=colors[c], linestyle=dashes[c],
                linewidth=1.0, label=label, alpha=0.9)
        ax.fill_between(t, mean_acc[:, c] - std_acc[:, c],
                        mean_acc[:, c] + std_acc[:, c],
                        color=colors[c], alpha=0.1)
    ax.set_xlabel("Incremental Learning Step (# of classes seen)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.01])
    ax.set_xlim([0.8, N_STEPS + 0.2])
    ax.set_xticks(t[::5])
    ax.legend(title="Mean final accuracy", loc="lower left",
              frameon=False, ncol=2, fontsize=5)
    plt.tight_layout()
    for ext in (("pdf", "png") if SAVE_PDF else ("png",)):
        plt.savefig(images_dir / f"acc_trend_1shot_baselines.{ext}",
                    format=ext, bbox_inches="tight")
    plt.close()

    print("\nFinal accuracies (mean ± std):")
    for c, name in enumerate(VARIANTS):
        print(f"  {name:<22}: {mean_acc[-1,c]*100:.1f} ± {std_acc[-1,c]*100:.1f}%")
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Figures saved to {images_dir}/")


if __name__ == "__main__":
    main()
