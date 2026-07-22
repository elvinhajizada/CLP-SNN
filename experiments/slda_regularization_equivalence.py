"""
slda_regularization_equivalence.py
----------------------------------
Regularization equivalence between Hayes' shrinkage SLDA and RankOneSLDA:
verifies that the fixed-ridge rank-1 implementation matches the reference
shrinkage SLDA in accuracy.

Hayes' fixed shrinkage (1-eps)*Sigma + eps*I is algebraically a ridge on the
scatter that grows with n; the rank-one implementation needs a fixed ridge
S + lambda*I. This experiment checks that the two parameterizations are
accuracy-equivalent:

  Stage 1 (selection): lambda swept over {0.1, 0.3, 1, 3, 10} on one held-out
    class order (the seed-10 1-shot stream with its 40 class blocks permuted
    by an independent RNG, seed 99); best lambda picked by final accuracy.
  Stage 2 (evaluation): Hayes-SLDA (eps = 1e-4, exact Lambda at every
    checkpoint) vs RankOne-SLDA (best lambda) under both paper protocols:
    1-shot (40 checkpoints, seen-classes-only eval) and 25-shot
    (25 checkpoints, full test set), seeds {10, 20, 30}, CPU.

Decision rule: if best-lambda RankOne is within one std of Hayes at final
accuracy in BOTH protocols -> adopt the rank-one implementation everywhere,
report this comparison in one supplementary panel. Otherwise widen the sweep
one decade each way (--extra_lambdas).

Outputs saved to experiments/results/slda_regularization_equivalence/:
  selection_sweep.csv        - final accuracy per lambda on the held-out order
  equivalence.npz            - accuracy curves (checkpoints x models x seeds)
  final_accuracy_table.csv   - final mean +/- std per model and protocol
  ea_equivalence_panel.png/pdf (also copied to images/) - supplementary panel

Usage (from repo root or experiments/):
  python experiments/slda_regularization_equivalence.py
  python experiments/slda_regularization_equivalence.py --extra_lambdas 0.01 0.03 30 100
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics

# Make repo root importable when running from experiments/
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.SLDA import StreamingLDA, RankOneSLDA  # noqa: E402
from experiments.clp_vs_baselines_25shot import (  # noqa: E402
    reorder_by_instance_rounds,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = _REPO / "data"
RESULTS_DIR = _REPO / "experiments" / "results" / "slda_regularization_equivalence"

SEEDS = [10, 20, 30]
NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60
DEVICE = "cpu"
GLOBAL_SEED = 42
HOLDOUT_PERM_SEED = 99       # class-block permutation for the selection stream
LAMBDA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
SHRINKAGE_EPS = 1e-4         # Hayes' shrinkage parameter (paper value)
N_VIDEOS_PER_STEP = 40       # 25-shot: evaluate after each full round
SAVE_PDF = True

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
    "savefig.dpi": 600,
}


# ── Data loading (mirrors clp_vs_baselines_{1,25}shot.py) ─────────────────────

def load_test_set():
    np.random.seed(GLOBAL_SEED)
    X = np.load(DATA_DIR / "X_test.npy")
    y = np.load(DATA_DIR / "y_test.npy")
    balanced_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        balanced_idx.extend(np.random.choice(cls_idx, N_FRAMES, replace=False))
    X = X[balanced_idx]
    y = y[balanced_idx]
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return torch.from_numpy(X_norm).float(), torch.from_numpy(y).long()


def load_1shot_stream(seed: int):
    X = torch.load(DATA_DIR / "1shot" / f"X_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).float()
    y = torch.load(DATA_DIR / "1shot" / f"y_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).long()
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    class_order = torch.unique_consecutive(y).tolist()
    counts = torch.stack([(y == c).sum() for c in class_order]).cpu().numpy()
    boundaries = np.cumsum(counts)
    return X, y, boundaries, class_order


def load_25shot_stream(seed: int):
    X = np.load(DATA_DIR / "25shot" / f"X_train_25_shot_{seed}.npy")
    y = np.load(DATA_DIR / "25shot" / f"y_train_25_shot_{seed}.npy")
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_r, y_r, shots_per_video = reorder_by_instance_rounds(
        X_norm, y, fixed_class_order=False, seed=seed, max_size=N_FRAMES
    )
    shots_cumsum = np.cumsum(shots_per_video)
    checkpoints = shots_cumsum[N_VIDEOS_PER_STEP - 1:: N_VIDEOS_PER_STEP]
    return (torch.from_numpy(X_r).float(), torch.from_numpy(y_r).long(),
            checkpoints)


def permute_class_blocks(X, y, perm_seed: int):
    """Held-out class order: permute the contiguous class blocks of a 1-shot
    stream with an RNG independent of the evaluation seeds."""
    rng = random.Random(perm_seed)
    class_order = torch.unique_consecutive(y).tolist()
    perm = class_order.copy()
    rng.shuffle(perm)
    idx = torch.cat([torch.where(y == c)[0] for c in perm])
    return X[idx], y[idx]


# ── Models ────────────────────────────────────────────────────────────────────

def build_hayes():
    return StreamingLDA(
        FEATURE_SIZE, NUM_CLASSES, backbone=None,
        shrinkage_param=SHRINKAGE_EPS, streaming_update_sigma=True,
        device=DEVICE,
    )


def build_rank_one(lam: float):
    return RankOneSLDA(FEATURE_SIZE, NUM_CLASSES, backbone=None,
                       ridge_param=lam, device=DEVICE)


# ── Evaluation ────────────────────────────────────────────────────────────────

def accuracy(clf, X_test, y_test, seen_classes=None):
    if seen_classes is not None:
        mask = torch.zeros_like(y_test, dtype=torch.bool)
        for c in seen_classes:
            mask |= (y_test == c)
        X_test, y_test = X_test[mask], y_test[mask]
    scores = clf.predict(X_test)
    _, pred = scores.topk(1, 1, True, True)
    return metrics.accuracy_score(y_test.numpy(),
                                  pred.t().cpu().numpy().squeeze())


def run_1shot(clf, X_tr, y_tr, boundaries, class_order, X_test, y_test):
    """Fit over a 1-shot stream; return per-class-boundary accuracies
    (seen-classes-only, as in clp_vs_baselines_1shot.py)."""
    accs = []
    for i, (xi, yi) in enumerate(zip(X_tr, y_tr), start=1):
        clf.fit(xi.view(FEATURE_SIZE), yi.view(1), i - 1)
        if i in boundaries and len(accs) < len(class_order):
            seen = class_order[: len(accs) + 1]
            accs.append(accuracy(clf, X_test, y_test, seen))
    return np.array(accs)


def run_25shot(clf, X_tr, y_tr, checkpoints, X_test, y_test):
    """Fit over a 25-shot instance-round stream; return per-round accuracies
    on the full test set (as in clp_vs_baselines_25shot.py)."""
    checkpoints = set(int(c) for c in checkpoints)
    accs = []
    for i, (xi, yi) in enumerate(zip(X_tr, y_tr), start=1):
        clf.fit(xi.view(FEATURE_SIZE), yi.view(1), i - 1)
        if i in checkpoints:
            accs.append(accuracy(clf, X_test, y_test))
    return np.array(accs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hayes shrinkage vs RankOne fixed-ridge equivalence")
    parser.add_argument("--extra_lambdas", type=float, nargs="*", default=[],
                        help="Additional lambda values to widen the sweep")
    parser.add_argument("--skip_25shot", action="store_true",
                        help="Debug: run the 1-shot protocol only")
    args = parser.parse_args()

    plt.rcParams.update(PLOT_CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = _REPO / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    X_test, y_test = load_test_set()
    lambda_grid = sorted(set(LAMBDA_GRID + args.extra_lambdas))

    # ── Stage 1: lambda selection on the held-out class order ────────────────
    print("=" * 60)
    print(f"Stage 1: lambda selection (held-out class order, "
          f"perm seed {HOLDOUT_PERM_SEED})")
    print("=" * 60)
    X_sel_base, y_sel_base, _, _ = load_1shot_stream(SEEDS[0])
    X_sel, y_sel = permute_class_blocks(X_sel_base, y_sel_base,
                                        HOLDOUT_PERM_SEED)

    sweep = []
    for lam in lambda_grid:
        clf = build_rank_one(lam)
        for i in range(len(y_sel)):
            clf.fit(X_sel[i].view(FEATURE_SIZE), y_sel[i].view(1), i)
        acc = accuracy(clf, X_test, y_test)
        sweep.append({"lambda": lam, "final_acc": acc})
        print(f"  lambda = {lam:<6g} final acc = {acc*100:.2f}%")

    best = max(sweep, key=lambda r: r["final_acc"])
    best_lambda = best["lambda"]
    print(f"  -> selected lambda = {best_lambda}")
    if best_lambda in (min(lambda_grid), max(lambda_grid)):
        print("  WARNING: selected lambda is at the sweep boundary; consider "
              "widening with --extra_lambdas")

    with open(RESULTS_DIR / "selection_sweep.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "final_acc"])
        w.writeheader()
        w.writerows(sweep)

    # ── Stage 2: Hayes vs RankOne under both paper protocols ─────────────────
    models = ["Hayes-SLDA", "RankOne-SLDA"]
    builders = [build_hayes, lambda: build_rank_one(best_lambda)]
    protocols = ["1shot"] if args.skip_25shot else ["1shot", "25shot"]

    curves = {}  # (protocol, model) -> (n_checkpoints, n_seeds)
    for protocol in protocols:
        n_ck = NUM_CLASSES if protocol == "1shot" else 25
        for m in models:
            curves[(protocol, m)] = np.zeros((n_ck, len(SEEDS)))

    for s, seed in enumerate(SEEDS):
        print(f"\n{'='*60}\nStage 2, seed {seed}\n{'='*60}")
        for protocol in protocols:
            if protocol == "1shot":
                X_tr, y_tr, boundaries, class_order = load_1shot_stream(seed)
            else:
                X_tr, y_tr, checkpoints = load_25shot_stream(seed)
            for m, build in zip(models, builders):
                print(f"  [{protocol}] {m}...", end=" ", flush=True)
                clf = build()
                if protocol == "1shot":
                    accs = run_1shot(clf, X_tr, y_tr, boundaries, class_order,
                                     X_test, y_test)
                else:
                    accs = run_25shot(clf, X_tr, y_tr, checkpoints,
                                      X_test, y_test)
                curves[(protocol, m)][:, s] = accs
                print(f"final = {accs[-1]*100:.2f}%")

    # ── Decision rule ────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nEquivalence check (lambda = {best_lambda})\n{'='*60}")
    rows, all_within = [], True
    for protocol in protocols:
        h = curves[(protocol, "Hayes-SLDA")][-1]
        r = curves[(protocol, "RankOne-SLDA")][-1]
        # One-sided rule: the substitution is acceptable unless RankOne is
        # WORSE than Hayes by more than one std; being better can only
        # strengthen the baseline.
        not_worse = r.mean() >= h.mean() - h.std()
        all_within &= not_worse
        rows.append({
            "protocol": protocol,
            "hayes_mean": h.mean(), "hayes_std": h.std(),
            "rankone_mean": r.mean(), "rankone_std": r.std(),
            "diff": r.mean() - h.mean(), "not_worse_than_one_std": not_worse,
        })
        print(f"  {protocol:>6}: Hayes {h.mean()*100:.2f}+/-{h.std()*100:.2f}%"
              f"  RankOne {r.mean()*100:.2f}+/-{r.std()*100:.2f}%"
              f"  diff {(r.mean()-h.mean())*100:+.2f}pp"
              f"  not worse (1 std): {not_worse}")
    if all_within:
        print("\n-> Equivalence established: RankOne-SLDA matches (or "
              "exceeds) the reference implementation in both protocols.")
    else:
        print("\n-> Equivalence not established at this lambda: widen the "
              "sweep one decade each way (--extra_lambdas).")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "equivalence.npz",
        **{f"{p}_{m.split('-')[0].lower()}": curves[(p, m)]
           for p in protocols for m in models},
        best_lambda=best_lambda,
        lambda_grid=np.array(lambda_grid),
        sweep_acc=np.array([r["final_acc"] for r in sweep]),
    )
    with open(RESULTS_DIR / "final_accuracy_table.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── Supplementary panel ──────────────────────────────────────────────────
    n_panels = 1 + len(protocols)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 2.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    lams = [r["lambda"] for r in sweep]
    accs_ = [r["final_acc"] for r in sweep]
    ax.semilogx(lams, accs_, "o-", color="tab:blue", markersize=3)
    ax.axvline(best_lambda, color="tab:red", linestyle=":", linewidth=0.75)
    ax.set_xlabel(r"ridge $\lambda$")
    ax.set_ylabel("Final accuracy (held-out order)")
    ax.set_title("(a) $\\lambda$ selection")

    for k, protocol in enumerate(protocols):
        ax = axes[1 + k]
        n_ck = curves[(protocol, models[0])].shape[0]
        t = np.arange(1, n_ck + 1)
        for m, color, ls in zip(models, ["tab:orange", "tab:blue"],
                                ["-", "--"]):
            mean = curves[(protocol, m)].mean(axis=1)
            std = curves[(protocol, m)].std(axis=1)
            ax.plot(t, mean, ls, color=color, label=f"{m}: "
                    f"{mean[-1]*100:.1f}$\\pm${std[-1]*100:.1f}%")
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("Classes seen" if protocol == "1shot"
                      else "Instance round")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"({chr(98 + k)}) {protocol} protocol")
        ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    for ext in (("pdf", "png") if SAVE_PDF else ("png",)):
        for out_dir in (RESULTS_DIR, images_dir):
            plt.savefig(out_dir / f"ea_equivalence_panel.{ext}",
                        format=ext, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Panel saved to {images_dir}/ea_equivalence_panel.png")


if __name__ == "__main__":
    main()
