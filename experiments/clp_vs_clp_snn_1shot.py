"""
clp_vs_clp_snn_1shot.py
-----------------------
1-shot OpenLoris continual-learning comparison of the original CLP
(`models.CLP.ContinuallyLearningPrototypes`) against CLP-SNN
(`models.CLP_SNN.CLPSNN`) across its float and integer learning-rule variants.

Protocol mirrors cell 5 of `experiments/acc_trend_clp_loihi_paper_final.ipynb`:
  * 40 classes x 60 frames per class (pre-balanced), one frame fit at a time.
  * After every class boundary, evaluate on a 60-per-class balanced test set
    filtered to the classes seen so far -> 40 checkpoints.
  * Mean +/- std across seeds [10, 20, 30].

Variants (all share n_protos=600, threshold=0.75, feature_size=1280):
  CLP              - paper config, adaptive_protos=True, learn_outliers=True
  CLP-NA           - same, adaptive_protos=False
  SNN-float        - float Taylor update, adaptive, no voting
  SNN-float-NA     - float, adaptive_protos=False
  SNN-float-vote   - float, voting on
  SNN-int          - Lava-faithful int update, adaptive, no voting
  SNN-int-NA       - int, adaptive_protos=False
  SNN-int-vote     - int, voting on

Outputs land in `experiments/results/clp_vs_clp_snn_1shot/`:
  accuracies.npz          - raw (40, n_variants, n_seeds) array + variant names
  accuracy_trend.png      - mean line + std shading per variant
  final_accuracy_table.csv
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics

# Make repo root importable when running from experiments/
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.CLP import ContinuallyLearningPrototypes  # noqa: E402
from models.CLP_SNN import CLPSNN  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = _REPO / "data"
TRAIN_DIR = DATA_DIR / "1shot"
TEST_X_PATH = DATA_DIR / "X_test.npy"
TEST_Y_PATH = DATA_DIR / "y_test.npy"

RESULTS_DIR = _REPO / "experiments" / "results" / "clp_vs_clp_snn_1shot"

SEEDS = [10, 20, 30]
NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60
N_PROTOS = 600
THRESHOLD = 0.85
DEVICE = "cpu"
GLOBAL_SEED = 42
SAVE_PDF = True   # set False to skip .pdf output and save only .png


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_set() -> tuple[torch.Tensor, torch.Tensor]:
    """Load OpenLoris test features, balance to 60/class, L2-normalise."""
    X = np.load(TEST_X_PATH)
    y = np.load(TEST_Y_PATH)

    # Use global numpy random state (matches notebook)
    unique_classes = np.unique(y)
    samples_per_class = N_FRAMES

    balanced_idx = []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        chosen = np.random.choice(cls_idx, size=samples_per_class, replace=False)
        balanced_idx.extend(chosen)
    balanced_idx = np.asarray(balanced_idx)

    X = X[balanced_idx]
    y = y[balanced_idx]

    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def load_train_split(seed: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, list]:
    """Load one seed's 1-shot train set; return normalised X, y, class-boundary
    shots (cumulative sample counts for checkpoints), and class_order list."""
    X = torch.load(TRAIN_DIR / f"X_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True)
    y = torch.load(TRAIN_DIR / f"y_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True)
    X = X.float()
    y = y.long()

    # L2-normalise
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Class order of first appearance (file is pre-ordered by class)
    class_order = torch.unique_consecutive(y).tolist()

    # Per-class counts to locate checkpoints (one per class boundary)
    counts = torch.stack([(y == c).sum() for c in class_order]).cpu().numpy()
    shots = np.cumsum(counts)  # e.g. [60, 120, ..., 2400]

    return X, y, shots, class_order


# ── Variant builders ──────────────────────────────────────────────────────────

VARIANTS = [
    "CLP",
    "CLP-NA",
    "SNN-float",
    "SNN-float-NA",
    "SNN-float-vote",
    "SNN-int",
    "SNN-int-NA",
    "SNN-int-vote",
]


def build_variant(name: str, seed: int) -> torch.nn.Module:
    """Instantiate a fresh classifier for the given variant name and seed.

    `seed` is threaded through CLP-SNN's RNG so the integer stochastic round
    differs per seed (reproducibly).
    """
    if name == "CLP":
        return ContinuallyLearningPrototypes(
            FEATURE_SIZE,
            n_protos=N_PROTOS,
            num_classes=NUM_CLASSES,
            backbone=None,
            alpha_init=1,
            sim_th_init=THRESHOLD,
            n_wta=1,
            k_hit=0.5,
            k_miss=0.5,
            adaptive_th=False,
            learn_outliers=True,
            device=DEVICE,
        )
    if name == "CLP-NA":
        return ContinuallyLearningPrototypes(
            FEATURE_SIZE,
            n_protos=N_PROTOS,
            num_classes=NUM_CLASSES,
            backbone=None,
            alpha_init=1,
            sim_th_init=THRESHOLD,
            n_wta=1,
            k_hit=1,
            k_miss=1,
            adaptive_th=False,
            learn_outliers=True,
            adaptive_protos=False,
            device=DEVICE,
        )

    # CLP-SNN variants
    snn_defaults = dict(
        feature_size=FEATURE_SIZE,
        n_protos=N_PROTOS,
        num_classes=NUM_CLASSES,
        threshold=THRESHOLD,
        g_inc=0.5,
        device=DEVICE,
        use_pseudo_labels=True,
        seed=seed,
    )
    if name == "SNN-float":
        return CLPSNN(**snn_defaults, use_quantization=False,
                      adaptive_protos=True, enable_voting=False)
    if name == "SNN-float-NA":
        return CLPSNN(**snn_defaults, use_quantization=False,
                      adaptive_protos=False, enable_voting=False)
    if name == "SNN-float-vote":
        return CLPSNN(**snn_defaults, use_quantization=False,
                      adaptive_protos=True, enable_voting=True)
    if name == "SNN-int":
        return CLPSNN(**snn_defaults, use_quantization=True,
                      adaptive_protos=True, enable_voting=False)
    if name == "SNN-int-NA":
        return CLPSNN(**snn_defaults, use_quantization=True,
                      adaptive_protos=False, enable_voting=False)
    if name == "SNN-int-vote":
        return CLPSNN(**snn_defaults, use_quantization=True,
                      adaptive_protos=True, enable_voting=True)

    raise ValueError(f"unknown variant: {name}")


# ── Train + eval for one (variant, seed) ─────────────────────────────────────

def train_and_eval(
    classifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    shots: np.ndarray,
    class_order: list,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> np.ndarray:
    """Stream the 1-shot train set one sample at a time, evaluate on the test
    set at each class boundary. Returns a (num_classes,) accuracy trajectory."""
    acc_trace = np.zeros(NUM_CLASSES, dtype=np.float64)
    check_point = -1

    shots_set = set(int(s) for s in shots)

    for i, (x, y) in enumerate(zip(X_train, y_train), start=1):
        classifier.fit(x.view(FEATURE_SIZE), y.view(1), i - 1)

        if i in shots_set:
            check_point += 1
            labels_learned = class_order[: check_point + 1]
            mask = torch.isin(y_test, torch.tensor(labels_learned))
            X_test_f = X_test[mask]
            y_test_f = y_test[mask]

            probas = classifier.predict(X_test_f)
            _, pred = probas.topk(1, 1, True, True)
            y_pred = pred.t().cpu().numpy().squeeze()
            acc_trace[check_point] = metrics.accuracy_score(
                y_test_f.cpu().numpy(), y_pred
            )

    return acc_trace


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    print(f"Loading test set from {TEST_X_PATH}")
    X_test, y_test = load_test_set()
    print(f"  test set: X={tuple(X_test.shape)}, y={tuple(y_test.shape)}")

    n_variants = len(VARIANTS)
    n_seeds = len(SEEDS)
    accuracies = np.zeros((NUM_CLASSES, n_variants, n_seeds), dtype=np.float64)

    total = n_seeds * n_variants
    done = 0
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ({s_idx + 1}/{n_seeds}) ===")
        X_tr, y_tr, shots, class_order = load_train_split(seed)
        print(f"  train: X={tuple(X_tr.shape)}, {len(class_order)} classes, "
              f"{int(shots[-1])} total samples")

        for v_idx, name in enumerate(VARIANTS):
            done += 1
            print(f"  [{done}/{total}] {name} ...", flush=True)
            clf = build_variant(name, seed=seed)
            traj = train_and_eval(
                clf, X_tr, y_tr, shots, class_order, X_test, y_test
            )
            accuracies[:, v_idx, s_idx] = traj
            print(f"      final acc = {traj[-1] * 100:.2f}%")

    # ── Save raw results ─────────────────────────────────────────────────────
    npz_path = RESULTS_DIR / "accuracies.npz"
    np.savez(npz_path, accuracies=accuracies,
             variants=np.array(VARIANTS), seeds=np.array(SEEDS))
    print(f"\nSaved raw results: {npz_path}")

    # ── Final accuracy table ─────────────────────────────────────────────────
    finals = accuracies[-1]  # (n_variants, n_seeds)
    means = finals.mean(axis=1) * 100
    stds = finals.std(axis=1) * 100

    csv_path = RESULTS_DIR / "final_accuracy_table.csv"
    with open(csv_path, "w") as f:
        header = ["variant", "mean", "std"] + [f"seed_{s}" for s in SEEDS]
        f.write(",".join(header) + "\n")
        for i, name in enumerate(VARIANTS):
            per_seed = [f"{finals[i, j] * 100:.2f}" for j in range(n_seeds)]
            f.write(f"{name},{means[i]:.2f},{stds[i]:.2f}," + ",".join(per_seed) + "\n")
    print(f"Saved final accuracy table: {csv_path}")

    print("\nFinal accuracy (mean +/- std across seeds):")
    print("-" * 50)
    width = max(len(n) for n in VARIANTS)
    for i, name in enumerate(VARIANTS):
        print(f"  {name:<{width}}  {means[i]:6.2f} +/- {stds[i]:5.2f} %")
    print("-" * 50)

    # ── Trend plot ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return

    mean_trace = accuracies.mean(axis=2) * 100  # (40, n_variants)
    std_trace = accuracies.std(axis=2) * 100

    t = np.arange(1, NUM_CLASSES + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, n_variants))

    for i, name in enumerate(VARIANTS):
        ax.plot(t, mean_trace[:, i], label=f"{name} ({mean_trace[-1, i]:.1f}%)",
                color=palette[i], linewidth=1.4)
        ax.fill_between(t, mean_trace[:, i] - std_trace[:, i],
                        mean_trace[:, i] + std_trace[:, i],
                        color=palette[i], alpha=0.15, linewidth=0)

    ax.set_xlabel("Classes seen")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("OpenLoris 1-shot: CLP vs CLP-SNN variants")
    ax.set_xlim(1, NUM_CLASSES)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(RESULTS_DIR / "accuracy_trend.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "accuracy_trend.png", dpi=150)
    print(f"Saved trend plot: {RESULTS_DIR / 'accuracy_trend.png'}")


if __name__ == "__main__":
    main()
