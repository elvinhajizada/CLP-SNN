"""
clp_vs_baselines_25shot.py
--------------------------
25-shot OpenLoris continual-learning comparison of CLP, CLP-SNN, and six
baseline classifiers (Fig. 3b, Table 1 in
"Real-time Continual Learning on Intel Loihi 2").

Protocol:
  * 40 classes x 25 shots x ~60 frames/shot ≈ 60,000 training samples.
  * Data is reordered into instance-incremental rounds: each round presents one
    video clip from every class before advancing. Evaluation after every 40 videos
    (one full round across all classes) → 25 checkpoints total.
  * Mean ± std across seeds [10, 20, 30].

Outputs saved to experiments/results/clp_vs_baselines_25shot/:
  accuracies.npz          - raw (n_steps, n_variants, n_seeds) array + variant names
  accuracy_trend.png/pdf  - mean line + std shading per variant
  final_accuracy_table.csv
"""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
TRAIN_DIR = DATA_DIR / "25shot"
TEST_X_PATH = DATA_DIR / "X_test.npy"
TEST_Y_PATH = DATA_DIR / "y_test.npy"

RESULTS_DIR = _REPO / "experiments" / "results" / "clp_vs_baselines_25shot"

SEEDS = [10, 20, 30]
NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60         # frames per video clip
K_SHOT = 25           # shots (video clips) per class
N_VIDEOS_PER_STEP = 40  # evaluation every 40 video clips (one full round)
N_STEPS = K_SHOT * NUM_CLASSES // N_VIDEOS_PER_STEP  # 25 checkpoints
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def reorder_by_instance_rounds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fixed_class_order: bool = False,
    seed: int = None,
    max_size: int = 60,
):
    """Reorder 25-shot data into instance-incremental rounds.

    Each round presents one video clip from every class before advancing.
    Returns reordered X, y arrays and per-video shot counts.
    """
    rng = random.Random(seed)

    changes = np.diff(y_train) != 0
    boundaries = np.insert(changes, 0, True)
    group_indices = np.where(boundaries)[0]

    adjusted = [group_indices[0]]
    for i in range(1, len(group_indices)):
        while group_indices[i] - adjusted[-1] > max_size:
            adjusted.append(adjusted[-1] + max_size)
        adjusted.append(group_indices[i])
    group_indices = np.array(adjusted)
    group_sizes = np.diff(np.append(group_indices, len(y_train)))

    chunks_by_class = defaultdict(list)
    for i, size in enumerate(group_sizes):
        label = y_train[group_indices[i]]
        x_chunk = X_train[group_indices[i]: group_indices[i] + size]
        y_chunk = y_train[group_indices[i]: group_indices[i] + size]
        chunks_by_class[label].append((x_chunk, y_chunk))

    all_classes = sorted(chunks_by_class.keys())
    num_rounds = max(len(v) for v in chunks_by_class.values())

    class_orders = []
    if fixed_class_order:
        order = all_classes.copy()
        rng.shuffle(order)
        class_orders = [order] * num_rounds
    else:
        for _ in range(num_rounds):
            round_classes = [c for c in all_classes if chunks_by_class[c]]
            rng.shuffle(round_classes)
            class_orders.append(round_classes.copy())

    X_new, y_new, shots = [], [], []
    for round_order in class_orders:
        for cls in round_order:
            if chunks_by_class[cls]:
                x_chunk, y_chunk = chunks_by_class[cls].pop(0)
                X_new.extend(x_chunk)
                y_new.extend(y_chunk)
                shots.append(len(y_chunk))

    return np.array(X_new), np.array(y_new), shots


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
    """Load one seed's 25-shot train set, reorder into instance rounds.

    Returns raw X (for Replay), normalised X (for all other classifiers), y, checkpoints.
    Both X arrays use the same instance-round ordering (same RNG seed).
    """
    X_raw = np.load(TRAIN_DIR / f"X_train_{K_SHOT}_shot_{seed}.npy")
    y = np.load(TRAIN_DIR / f"y_train_{K_SHOT}_shot_{seed}.npy")
    X_norm = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)
    X_r_norm, y_r, shots_per_video = reorder_by_instance_rounds(
        X_norm, y, fixed_class_order=False, seed=seed, max_size=N_FRAMES
    )
    X_r_raw, _, _ = reorder_by_instance_rounds(
        X_raw, y, fixed_class_order=False, seed=seed, max_size=N_FRAMES
    )
    shots_cumsum = np.cumsum(shots_per_video)
    checkpoints = shots_cumsum[N_VIDEOS_PER_STEP - 1:: N_VIDEOS_PER_STEP]
    return (torch.from_numpy(X_r_norm).float(),
            torch.from_numpy(X_r_raw).float(),
            torch.from_numpy(y_r).long(),
            checkpoints)


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
    d = DEVICE
    if name == "Perceptron":
        return Perceptron(FEATURE_SIZE, NUM_CLASSES, backbone=None, device=d)

    elif name == "Fine Tuning":
        return StreamingSoftmax(
            FEATURE_SIZE, NUM_CLASSES, use_replay=False,
            backbone=None, lr=0.003, weight_decay=1e-5, device=d,
        )

    elif name == "NCM":
        return NearestClassMean(FEATURE_SIZE, NUM_CLASSES, backbone=None, device=d)

    elif name == "Replay":
        return StreamingSoftmax(
            FEATURE_SIZE, NUM_CLASSES, use_replay=True, backbone=None,
            lr=0.09, weight_decay=1e-5, replay_samples=50,
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
            FEATURE_SIZE, n_protos=2600, num_classes=NUM_CLASSES,
            backbone=None, alpha_init=1, sim_th_init=0.75, n_wta=1,
            k_hit=1, k_miss=1, adaptive_th=False,
            learn_outliers=False, device=d,
        )

    elif name == "CLP-SNN":
        return CLPSNN(
            FEATURE_SIZE, n_protos=300, num_classes=NUM_CLASSES,
            threshold=0.0, g_inc=0.5, use_quantization=False, device=d,
        )

    raise ValueError(f"Unknown variant: {name!r}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(clf, X_test_raw, X_test_norm, y_test, use_raw: bool):
    X = X_test_raw if use_raw else X_test_norm
    probas = clf.predict(X)
    _, pred = probas.topk(1, 1, True, True)
    y_pred = pred.t().cpu().numpy().squeeze()
    return metrics.accuracy_score(y_test.numpy(), y_pred)


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
        X_tr, X_tr_raw, y_tr, checkpoints = load_train_split(seed)
        print(f"  train: {len(y_tr)} samples, {N_STEPS} checkpoints")

        for c, name in enumerate(VARIANTS):
            print(f"  [{c+1}/{len(VARIANTS)}] {name}...", end=" ", flush=True)
            clf = build_classifier(name)
            use_raw = (name == "Replay")
            X_train = X_tr_raw if use_raw else X_tr

            i = 0
            chk = 0
            for xi, yi in zip(X_train, y_tr):
                clf.fit(xi.view(FEATURE_SIZE), yi.view(1), i)
                i += 1
                if i in checkpoints and chk < N_STEPS:
                    acc = evaluate(clf, X_test_raw, X_test_norm, y_test, use_raw)
                    accuracies[chk, c, s] = acc
                    chk += 1

            print(f"final={accuracies[-1, c, s]*100:.1f}%")

    # ── Save results ──────────────────────────────────────────────────────────
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
    import seaborn as sns
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
    ax.set_xlabel("Incremental Learning Step (# of shots)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.01])
    ax.set_xlim([0.8, N_STEPS + 0.2])
    ax.set_xticks(t[::5])
    ax.legend(title="Mean final accuracy", loc="lower right",
              frameon=False, ncol=2, fontsize=5)
    plt.tight_layout()
    for ext in (("pdf", "png") if SAVE_PDF else ("png",)):
        plt.savefig(images_dir / f"acc_trend_25shot.{ext}",
                    format=ext, bbox_inches="tight")
    plt.close()

    print("\nFinal accuracies (mean ± std):")
    for c, name in enumerate(VARIANTS):
        print(f"  {name:<22}: {mean_acc[-1,c]*100:.1f} ± {std_acc[-1,c]*100:.1f}%")
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Figures saved to {images_dir}/")


if __name__ == "__main__":
    main()
