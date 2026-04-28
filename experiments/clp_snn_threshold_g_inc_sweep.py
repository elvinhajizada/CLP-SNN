"""
clp_snn_threshold_g_inc_sweep.py
---------------------------------
Parameter sweep of CLP-SNN (float and int variants) across threshold and g_inc.

Investigates whether the observed instability in SNN-int can be mitigated by
tuning the similarity threshold or goodness-increment decay rate.

Protocol mirrors clp_vs_clp_snn_1shot.py:
  * 40 classes x 60 frames per class (pre-balanced), one frame fit at a time.
  * After every class boundary, evaluate on a 60-per-class balanced test set
    filtered to the classes seen so far -> 40 checkpoints.
  * Mean +/- std across seeds [10, 20, 30].

Parameter grid:
  thresholds: {0.7, 0.75, 0.8}
  g_inc values: {0.1, 0.2, 0.5, 1}
  variants: SNN-float, SNN-int (both with adaptive_protos=True, enable_voting=True)

Outputs land in `experiments/results/clp_snn_threshold_g_inc_sweep/`:
  results_flat.csv           - flat table [variant, threshold, g_inc, seed, final_acc]
  heatmap_snn_float.png      - threshold x g_inc heatmap for float variant
  heatmap_snn_int.png        - threshold x g_inc heatmap for int variant
  detailed_accuracies.npz    - full accuracy trajectories
"""

import argparse
import os
import random
import sys
from pathlib import Path
from itertools import product

import numpy as np
import torch
from sklearn import metrics

# Make repo root importable when running from experiments/
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.CLP_SNN import CLPSNN  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = _REPO / "data"
TRAIN_DIR = DATA_DIR / "1shot"
TEST_X_PATH = DATA_DIR / "X_test.npy"
TEST_Y_PATH = DATA_DIR / "y_test.npy"

RESULTS_DIR = _REPO / "experiments" / "results" / "clp_snn_threshold_g_inc_sweep"

SEEDS = [10, 20, 30]
NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60
N_PROTOS = 600
DEVICE = "cpu"
GLOBAL_SEED = 42
SAVE_PDF = True   # set False to skip .pdf output and save only .png

# Parameter sweep axes
THRESHOLDS = [0.7, 0.75, 0.8, 0.85, 0.9]
G_INCS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
VARIANTS = ["SNN-float", "SNN-int"]


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


# ── Variant builder ───────────────────────────────────────────────────────────

def build_variant(variant_name: str, threshold: float, g_inc: float,
                  seed: int) -> torch.nn.Module:
    """Instantiate a fresh CLP-SNN classifier with specified parameters.

    Args:
        variant_name: "SNN-float" or "SNN-int"
        threshold: similarity threshold
        g_inc: goodness increment per correct prediction
        seed: RNG seed for int quantization
    """
    use_quantization = (variant_name == "SNN-int")

    return CLPSNN(
        feature_size=FEATURE_SIZE,
        n_protos=N_PROTOS,
        num_classes=NUM_CLASSES,
        threshold=threshold,
        g_inc=g_inc,
        device=DEVICE,
        use_pseudo_labels=True,
        enable_voting=True,
        adaptive_protos=True,
        use_quantization=use_quantization,
        seed=seed,
    )


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

def save_results_archive(results: dict) -> None:
    """Save full accuracy trajectories to .npz for later reconstruction."""
    # Convert dict with tuple keys to structured format for NPZ
    trajectories = {}
    for (variant, threshold, g_inc, seed), traj in results.items():
        key = f"{variant}_{threshold:.2f}_{g_inc:.2f}_seed{seed}"
        trajectories[key] = traj
    
    archive_path = RESULTS_DIR / "detailed_accuracies.npz"
    np.savez(archive_path, **trajectories)
    print(f"Saved detailed accuracies: {archive_path}")


def load_results_from_csv(csv_path: Path) -> tuple[dict, list]:
    """Load results from flat CSV, reconstruct (variant, threshold, g_inc, seed) keys,
    and return accumulated final accuracies. Also returns list of unique (variant, threshold, g_inc)."""
    results = {}
    seen_params = set()
    
    with open(csv_path, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(",")
            variant = parts[0]
            threshold = float(parts[1])
            g_inc = float(parts[2])
            seed = int(parts[3])
            final_acc = float(parts[4]) / 100.0  # Convert back to [0, 1]
            
            key = (variant, threshold, g_inc, seed)
            # For plotting purposes, we only need the final accuracy
            results[key] = np.array([final_acc])  # Wrap in array for consistency
            seen_params.add((variant, threshold, g_inc))
    
    return results, sorted(seen_params)


def plot_from_saved(results_dir: Path) -> None:
    """Regenerate heatmap plots from previously saved results CSV."""
    csv_path = results_dir / "results_flat.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run full experiment first.")
        return
    
    print(f"Loading results from {csv_path}")
    results, seen_params = load_results_from_csv(csv_path)
    
    # Extract unique values from seen parameters
    variants_seen = sorted(set(v[0] for v in seen_params))
    thresholds_seen = sorted(set(v[1] for v in seen_params))
    g_incs_seen = sorted(set(v[2] for v in seen_params))
    
    print(f"  Variants: {variants_seen}")
    print(f"  Thresholds: {thresholds_seen}")
    print(f"  g_inc values: {g_incs_seen}")
    
    # Try to import matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping heatmap plots")
        return
    
    # Regenerate heatmaps
    for variant in variants_seen:
        n_g_incs = len(g_incs_seen)
        n_thresholds = len(thresholds_seen)
        
        # Build heatmap data: (n_g_incs, n_thresholds) with mean across seeds
        heatmap = np.zeros((n_g_incs, n_thresholds))
        for g_idx, g_inc in enumerate(g_incs_seen):
            for th_idx, threshold in enumerate(thresholds_seen):
                accs = []
                for seed in SEEDS:
                    key = (variant, threshold, g_inc, seed)
                    if key in results:
                        # Extract final accuracy (results[key] is now wrapped array)
                        accs.append(float(results[key][0]) * 100)
                if accs:
                    heatmap[g_idx, th_idx] = np.mean(accs)
                else:
                    heatmap[g_idx, th_idx] = np.nan
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=30, vmax=70)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_thresholds))
        ax.set_yticks(np.arange(n_g_incs))
        ax.set_xticklabels([f"{th:.2f}" for th in thresholds_seen])
        ax.set_yticklabels([f"{g:.2f}" for g in g_incs_seen])
        
        ax.set_xlabel("Threshold")
        ax.set_ylabel("g_inc")
        ax.set_title(f"{variant}: Final Accuracy (%) by Threshold & g_inc\n(mean across {len(SEEDS)} seeds)")
        
        # Annotate cells with accuracy values
        for g_idx in range(n_g_incs):
            for th_idx in range(n_thresholds):
                val = heatmap[g_idx, th_idx]
                if not np.isnan(val):
                    ax.text(th_idx, g_idx, f"{val:.1f}",
                           ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax, label="Accuracy (%)")
        fig.tight_layout()
        stem = f"heatmap_{variant.lower().replace('-', '_')}"
        if SAVE_PDF:
            fig.savefig(results_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(results_dir / f"{stem}.png", dpi=150)
        print(f"Saved heatmap: {results_dir / (stem + '.png')}")
        plt.close(fig)
    
    print("\nPlots regenerated successfully!")


def main() -> None:
    # ── Parse command-line arguments ──────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Parameter sweep of CLP-SNN across threshold and g_inc. "
                    "Use --plot-only to regenerate plots from previously saved results."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiments and regenerate plots from saved CSV results",
    )
    args = parser.parse_args()
    
    # If --plot-only flag is set, just regenerate plots and exit
    if args.plot_only:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_from_saved(RESULTS_DIR)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    print(f"Loading test set from {TEST_X_PATH}")
    X_test, y_test = load_test_set()
    print(f"  test set: X={tuple(X_test.shape)}, y={tuple(y_test.shape)}")

    # Results storage: dict keyed by (variant, threshold, g_inc)
    results = {}
    csv_rows = []  # For flat CSV export

    n_thresholds = len(THRESHOLDS)
    n_g_incs = len(G_INCS)
    n_variants = len(VARIANTS)
    n_seeds = len(SEEDS)
    total = n_variants * n_thresholds * n_g_incs * n_seeds
    done = 0

    for variant in VARIANTS:
        print(f"\n=== {variant} ===")

        for threshold in THRESHOLDS:
            for g_inc in G_INCS:
                print(f"\n  threshold={threshold}, g_inc={g_inc}")

                for s_idx, seed in enumerate(SEEDS):
                    done += 1
                    print(f"    [{done}/{total}] seed {seed} ({s_idx + 1}/{n_seeds}) ...",
                          flush=True)

                    X_tr, y_tr, shots, class_order = load_train_split(seed)
                    clf = build_variant(variant, threshold, g_inc, seed=seed)
                    traj = train_and_eval(
                        clf, X_tr, y_tr, shots, class_order, X_test, y_test
                    )

                    # Store result
                    key = (variant, threshold, g_inc, seed)
                    results[key] = traj

                    final_acc = traj[-1] * 100
                    print(f"      final acc = {final_acc:.2f}%")

                    # Accumulate for CSV
                    csv_rows.append({
                        "variant": variant,
                        "threshold": threshold,
                        "g_inc": g_inc,
                        "seed": seed,
                        "final_accuracy": final_acc,
                    })

    # ── Save flat CSV ─────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "results_flat.csv"
    with open(csv_path, "w") as f:
        header = ["variant", "threshold", "g_inc", "seed", "final_accuracy"]
        f.write(",".join(header) + "\n")
        for row in csv_rows:
            f.write(f"{row['variant']},{row['threshold']},{row['g_inc']},"
                   f"{row['seed']},{row['final_accuracy']:.2f}\n")
    print(f"\nSaved flat results: {csv_path}")

    # ── Summary table per (threshold, g_inc) pair ─────────────────────────────
    print("\nFinal accuracy by (threshold, g_inc):")
    print("=" * 80)

    for variant in VARIANTS:
        print(f"\n{variant}:")
        print("-" * 80)
        print(f"{'g_inc':>6}", end="")
        for th in THRESHOLDS:
            print(f"  {th:5.2f}", end="")
        print()

        for g_inc in G_INCS:
            print(f"{g_inc:6.1f}", end="")
            for threshold in THRESHOLDS:
                # Compute mean across seeds for this (threshold, g_inc)
                accs = []
                for seed in SEEDS:
                    key = (variant, threshold, g_inc, seed)
                    if key in results:
                        accs.append(results[key][-1] * 100)
                if accs:
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    print(f"  {mean_acc:5.1f}±{std_acc:4.1f}", end="")
                else:
                    print(f"      N/A  ", end="")
            print()
        print()

    # ── Save detailed accuracies ──────────────────────────────────────────────
    save_results_archive(results)

    # ── Heatmap plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping heatmap plots")
        return

    for variant in VARIANTS:
        # Build heatmap data: (n_g_incs, n_thresholds) with mean across seeds
        heatmap = np.zeros((n_g_incs, n_thresholds))
        for g_idx, g_inc in enumerate(G_INCS):
            for th_idx, threshold in enumerate(THRESHOLDS):
                accs = []
                for seed in SEEDS:
                    key = (variant, threshold, g_inc, seed)
                    if key in results:
                        accs.append(results[key][-1] * 100)
                if accs:
                    heatmap[g_idx, th_idx] = np.mean(accs)
                else:
                    heatmap[g_idx, th_idx] = np.nan

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=20, vmax=80)

        # Set ticks and labels
        ax.set_xticks(np.arange(n_thresholds))
        ax.set_yticks(np.arange(n_g_incs))
        ax.set_xticklabels([f"{th:.2f}" for th in THRESHOLDS])
        ax.set_yticklabels([f"{g:.2f}" for g in G_INCS])

        ax.set_xlabel("Threshold")
        ax.set_ylabel("g_inc")
        ax.set_title(f"{variant}: Final Accuracy (%) by Threshold & g_inc\n(mean across 3 seeds)")

        # Annotate cells with accuracy values
        for g_idx in range(n_g_incs):
            for th_idx in range(n_thresholds):
                val = heatmap[g_idx, th_idx]
                if not np.isnan(val):
                    ax.text(th_idx, g_idx, f"{val:.1f}",
                           ha="center", va="center", color="black", fontsize=10)

        plt.colorbar(im, ax=ax, label="Accuracy (%)")
        fig.tight_layout()

        fig_path = RESULTS_DIR / f"heatmap_{variant.lower().replace('-', '_')}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved heatmap: {fig_path}")
        plt.close(fig)

    print("\n" + "=" * 80)
    print(f"All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
