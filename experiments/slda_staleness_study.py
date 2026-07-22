"""
slda_staleness_study.py
-----------------------
Staleness robustness study: prediction accuracy when predictions are served
from periodically refreshed weights instead of per-sample-fresh ones.

All arms share ONE fitted RankOneSLDA trajectory per (protocol, seed); they
differ only in which weights serve predictions. Three arm families:

  fresh      : per-sample-fresh maintained W (what RankOneSLDA provides).
  stale-full : snapshot of (W, b, cK) refreshed every rho fits — nothing
               about the model is visible between refreshes, including new
               classes. The fully amortized deployment.
  stale-Lam  : the periodic-precision variant — class means/weights update
               every sample (new classes immediately visible), only the
               precision Lambda is refreshed every rho fits. Emulated
               accuracy-identically at O(d^2)/sample by maintaining each
               arm's W incrementally: when mean y moves by delta_mu,
               W[:, y] += Lambda_snap @ delta_mu; at each refresh the arm's
               W is a copy of the fresh model's W (which equals
               Lambda_current @ muK^T at that instant). The naive emulation
               (W = Lambda_stale @ muK^T per predict) would be O(d^2 K).

Comparing the two stale families decomposes the freshness value: stale-full
vs fresh prices ALL staleness; stale-Lam vs fresh prices precision staleness
alone (the O(d^2) part); their gap prices mean staleness (the O(d) part).

Refresh cadence (all stale arms): snapshot after fitting samples
i = 0, rho, 2*rho, ... (Hayes' periodic cadence). rho = 1 is algebraically
the fresh arm and serves as a built-in sanity anchor (exact for stale-full,
asserted; up to fp-tie noise for stale-Lam, reported).

Two accuracy measures per arm:

  1. Prequential (test-then-train): predict each stream sample before fitting
     it. Measures adaptation speed; inflated by the temporal correlation of
     consecutive video frames, so it does NOT measure generalization.
  2. Time-averaged held-out accuracy: at periodic stream positions, evaluate
     the arm's current predictor on a balanced held-out probe restricted to
     the classes seen so far. Measures what a deployed system serving queries
     between updates delivers on unseen data (the generalization analog).
     The eval cadence (53, prime) is deliberately coprime to every refresh
     period so evaluations cycle through all staleness phases; a cadence that
     divided rho would always sample the same phase (e.g. eval-every-60 with
     rho = 60 would measure only maximal staleness) and bias the average.

Protocols: 1-shot and 25-shot, seeds {10, 20, 30}, CPU.

Note on cadence alignment (1-shot): refreshes at i = 0, 60, 120, ... land
right AFTER the first frame of each 60-frame class block, so every stale-full
snapshot contains one frame of the incoming class. This is the charitable
alignment; one step earlier and each new class would be invisible to a
stale-full arm for a full block. (stale-Lam arms see new classes through the
fresh means regardless.)

Outputs saved to experiments/results/slda_staleness_study/:
  staleness.npz             - per-sample hit arrays, held-out trajectories
  staleness_summary.csv     - both metrics per arm, protocol (mean +/- std)
  eb_staleness_panel.png/pdf (also copied to images/):
    row 1 - rolling-window prequential accuracy vs stream position
    row 2 - both accuracy measures vs refresh period rho (log x),
            stale-full vs stale-Lam

Usage (from repo root or experiments/):
  python experiments/slda_staleness_study.py [--ridge 1.0]
      [--periods 1 2 5 10 30 60 120 300 600] [--eval_every 53] [--threads 4]
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Make repo root importable when running from experiments/
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.SLDA import RankOneSLDA  # noqa: E402
from experiments.clp_vs_baselines_25shot import (  # noqa: E402
    reorder_by_instance_rounds,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = _REPO / "data"
RESULTS_DIR = _REPO / "experiments" / "results" / "slda_staleness_study"

SEEDS = [10, 20, 30]
NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60
DEVICE = "cpu"
GLOBAL_SEED = 42
N_PROBE_PER_CLASS = 10   # balanced held-out probe: 10 x 40 = 400 samples
ROLLING_PLOT_PERIODS = [60, 300]  # stale arms shown in the rolling panel
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

def load_probe_set():
    """Balanced held-out probe (N_PROBE_PER_CLASS per class, L2-normalised),
    drawn with the experiments' balancing seed."""
    np.random.seed(GLOBAL_SEED)
    X = np.load(DATA_DIR / "X_test.npy")
    y = np.load(DATA_DIR / "y_test.npy")
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        idx.extend(np.random.choice(cls_idx, N_PROBE_PER_CLASS, replace=False))
    X = X[idx]
    y = y[idx]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def load_1shot_stream(seed: int):
    X = torch.load(DATA_DIR / "1shot" / f"X_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).float()
    y = torch.load(DATA_DIR / "1shot" / f"y_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).long()
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return X, y


def load_25shot_stream(seed: int):
    X = np.load(DATA_DIR / "25shot" / f"X_train_25_shot_{seed}.npy")
    y = np.load(DATA_DIR / "25shot" / f"y_train_25_shot_{seed}.npy")
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_r, y_r, _ = reorder_by_instance_rounds(
        X_norm, y, fixed_class_order=False, seed=seed, max_size=N_FRAMES
    )
    return torch.from_numpy(X_r).float(), torch.from_numpy(y_r).long()


# ── Prediction from an arbitrary (W, b, cK) state ─────────────────────────────

def masked_argmax(x, W, b, cK):
    """Single-sample argmax with the unseen-class masking of predict()."""
    s = x @ W - b
    s[cK == 0] = s.min() - 1
    return int(s.argmax())


def probe_accuracy(W, b, cK, X_probe, y_probe, seen_mask):
    """Accuracy on the probe subset whose labels are in the seen set.
    Unseen-by-arm classes are masked as in predict()."""
    X, y = X_probe[seen_mask], y_probe[seen_mask]
    s = X @ W - b
    not_seen = torch.where(cK == 0)[0]
    if len(not_seen) > 0:
        s[:, not_seen] = s.min(dim=1, keepdim=True)[0] - 1
    return (s.argmax(dim=1) == y).float().mean().item()


def snapshot(model):
    b = 0.5 * torch.sum(model.muK.t() * model.W, dim=0)
    return {"W": model.W.clone(), "b": b, "cK": model.cK.clone()}


# ── Test-then-train run with all arm families ─────────────────────────────────

def run_stream(X_tr, y_tr, ridge, periods, X_probe, y_probe, eval_every):
    """One pass: predict-before-fit with the fresh arm, one stale-full arm and
    one stale-Lam arm per refresh period, all sharing the same fitted model.

    Arm keys: 'fresh', ('full', rho), ('lam', rho). rho = 1 arms alias the
    fresh behavior; the 'lam' rho = 1 arm is skipped (identical by
    construction and its per-step Lambda copies are pure overhead) and filled
    with the fresh results.

    Returns:
      hits      - {arm: (n,) bool}
      heldout   - {arm: (n_evals,) float}
      positions - (n_evals,) int, stream positions of the held-out evals
    """
    model = RankOneSLDA(FEATURE_SIZE, NUM_CLASSES, backbone=None,
                        ridge_param=ridge, device=DEVICE)
    n = len(y_tr)
    lam_periods = [r for r in periods if r > 1]
    arms = (["fresh"] + [("full", r) for r in periods]
            + [("lam", r) for r in lam_periods])
    hits = {arm: np.zeros(n, dtype=bool) for arm in arms}
    heldout = {arm: [] for arm in arms}
    positions = []

    full_snaps = {r: snapshot(model) for r in periods}
    lam_snaps = {r: model.Lambda.clone() for r in lam_periods}
    lam_W = {r: model.W.clone() for r in lam_periods}

    for i in range(n):
        x, y = X_tr[i], int(y_tr[i])

        # ── predict before fit (test-then-train) ─────────────────────────
        b_fresh = 0.5 * torch.sum(model.muK.t() * model.W, dim=0)
        hits["fresh"][i] = masked_argmax(x, model.W, b_fresh, model.cK) == y
        for r in periods:
            sn = full_snaps[r]
            hits[("full", r)][i] = masked_argmax(
                x, sn["W"], sn["b"], sn["cK"]) == y
        for r in lam_periods:
            # fresh means, stale Lambda: b from fresh muK and the arm's W
            b_arm = 0.5 * torch.sum(model.muK.t() * lam_W[r], dim=0)
            hits[("lam", r)][i] = masked_argmax(
                x, lam_W[r], b_arm, model.cK) == y

        # ── train ────────────────────────────────────────────────────────
        mu_y_old = model.muK[y].clone()
        model.fit(x, y_tr[i].view(1), i)

        # stale-Lam arms track the mean update under their frozen Lambda
        delta_mu = model.muK[y] - mu_y_old
        for r in lam_periods:
            lam_W[r][:, y] += lam_snaps[r] @ delta_mu

        # ── refresh (Hayes' cadence: after fitting i = 0, rho, 2*rho, ...) ─
        for r in periods:
            if i % r == 0:
                full_snaps[r] = snapshot(model)
        for r in lam_periods:
            if i % r == 0:
                lam_snaps[r] = model.Lambda.clone()
                # at refresh, fresh W IS Lambda_current @ muK^T
                lam_W[r] = model.W.clone()

        # ── held-out probe eval (cadence coprime to periods, see docstring) ─
        if (i + 1) % eval_every == 0 or (i + 1) == n:
            seen_classes = torch.where(model.cK > 0)[0]
            seen_mask = torch.isin(y_probe, seen_classes)
            positions.append(i + 1)
            # recompute b: b_fresh above predates this step's fit
            b_now = 0.5 * torch.sum(model.muK.t() * model.W, dim=0)
            heldout["fresh"].append(probe_accuracy(
                model.W, b_now, model.cK, X_probe, y_probe, seen_mask))
            for r in periods:
                sn = full_snaps[r]
                heldout[("full", r)].append(probe_accuracy(
                    sn["W"], sn["b"], sn["cK"], X_probe, y_probe, seen_mask))
            for r in lam_periods:
                b_arm = 0.5 * torch.sum(model.muK.t() * lam_W[r], dim=0)
                heldout[("lam", r)].append(probe_accuracy(
                    lam_W[r], b_arm, model.cK, X_probe, y_probe, seen_mask))

    heldout = {arm: np.array(v) for arm, v in heldout.items()}
    # rho = 1 stale-Lam arm is identical to fresh by construction
    if 1 in periods:
        hits[("lam", 1)] = hits["fresh"].copy()
        heldout[("lam", 1)] = heldout["fresh"].copy()
    return hits, heldout, np.array(positions)


# ── Main ──────────────────────────────────────────────────────────────────────

def arm_key(arm):
    if arm == "fresh":
        return "fresh"
    fam, r = arm
    return f"{fam}_rho{r}"


def main():
    parser = argparse.ArgumentParser(
        description="Staleness study with refresh-period sweep, "
                    "stale-full vs stale-Lambda arms, and held-out metric")
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="Fixed ridge lambda (value selected in "
                             "slda_regularization_equivalence.py)")
    parser.add_argument("--periods", type=int, nargs="*",
                        default=[1, 2, 5, 10, 30, 60, 120, 300, 600])
    parser.add_argument("--eval_every", type=int, default=53,
                        help="Held-out eval cadence; keep coprime to periods")
    parser.add_argument("--threads", type=int, default=4,
                        help="torch CPU threads; small values are often "
                             "faster for these per-sample ops and run cooler")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    for rho in args.periods:
        if rho > 1 and args.eval_every % rho == 0:
            print(f"WARNING: eval_every={args.eval_every} is a multiple of "
                  f"rho={rho}; held-out evals will sample a single staleness "
                  f"phase for that arm")

    plt.rcParams.update(PLOT_CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = _REPO / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    X_probe, y_probe = load_probe_set()
    protocols = ["1shot", "25shot"]
    all_arms = (["fresh"] + [("full", r) for r in args.periods]
                + [("lam", r) for r in args.periods if r > 1]
                + ([("lam", 1)] if 1 in args.periods else []))

    all_hits = {}
    all_heldout = {}
    all_positions = {}
    rows = []

    for protocol in protocols:
        for seed in SEEDS:
            print(f"[{protocol}] seed {seed}...", flush=True)
            if protocol == "1shot":
                X_tr, y_tr = load_1shot_stream(seed)
            else:
                X_tr, y_tr = load_25shot_stream(seed)
            hits, heldout, positions = run_stream(
                X_tr, y_tr, args.ridge, args.periods,
                X_probe, y_probe, args.eval_every)
            all_hits[(protocol, seed)] = hits
            all_heldout[(protocol, seed)] = heldout
            all_positions[(protocol, seed)] = positions

            # rho = 1 stale-full arm must equal fresh exactly
            if 1 in args.periods:
                assert (hits[("full", 1)] == hits["fresh"]).all(), \
                    "rho=1 stale-full arm must be identical to the fresh arm"

            for arm in hits:
                rows.append({
                    "protocol": protocol, "seed": seed,
                    "arm": arm_key(arm),
                    "prequential_acc": hits[arm].mean(),
                    "heldout_timeavg_acc": heldout[arm].mean(),
                })
            if 60 in args.periods:
                print(f"    fresh: preq {hits['fresh'].mean()*100:.2f}%  "
                      f"ho {heldout['fresh'].mean()*100:.2f}%  |  "
                      f"full-60: preq {hits[('full', 60)].mean()*100:.2f}%  "
                      f"ho {heldout[('full', 60)].mean()*100:.2f}%  |  "
                      f"Lam-60: preq {hits[('lam', 60)].mean()*100:.2f}%  "
                      f"ho {heldout[('lam', 60)].mean()*100:.2f}%",
                      flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*76}\nStaleness summary (mean +/- std over seeds "
          f"{SEEDS})\n{'='*76}")
    for protocol in protocols:
        print(f"  {protocol}:")
        print(f"    {'arm':>12}  {'prequential':>18}  "
              f"{'held-out (time-avg)':>20}")
        for arm in all_arms:
            preq = np.array([all_hits[(protocol, s)][arm].mean()
                             for s in SEEDS])
            ho = np.array([all_heldout[(protocol, s)][arm].mean()
                           for s in SEEDS])
            print(f"    {arm_key(arm):>12}  {preq.mean()*100:7.2f} +/- "
                  f"{preq.std()*100:4.2f}%  {ho.mean()*100:8.2f} +/- "
                  f"{ho.std()*100:4.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "staleness.npz",
        **{f"{p}_{s}_hits_{arm_key(a)}": all_hits[(p, s)][a]
           for p in protocols for s in SEEDS for a in all_hits[(p, s)]},
        **{f"{p}_{s}_heldout_{arm_key(a)}": all_heldout[(p, s)][a]
           for p in protocols for s in SEEDS for a in all_heldout[(p, s)]},
        **{f"{p}_{s}_positions": all_positions[(p, s)]
           for p in protocols for s in SEEDS},
        ridge=args.ridge, periods=np.array(args.periods),
        eval_every=args.eval_every,
    )
    with open(RESULTS_DIR / "staleness_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── Panel ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.4))

    # row 1: rolling-window prequential accuracy vs stream position
    rolling_arms = (["fresh"]
                    + [("full", r) for r in ROLLING_PLOT_PERIODS
                       if r in args.periods]
                    + [("lam", r) for r in ROLLING_PLOT_PERIODS
                       if r in args.periods])
    colors = {"fresh": "tab:blue"}
    fam_colors = {"full": ["tab:orange", "tab:red"],
                  "lam": ["tab:green", "tab:olive"]}
    for fam in ("full", "lam"):
        for k, r in enumerate([r for r in ROLLING_PLOT_PERIODS
                               if r in args.periods]):
            colors[(fam, r)] = fam_colors[fam][k % 2]

    for ax, protocol in zip(axes[0], protocols):
        n_min = min(len(all_hits[(protocol, s)]["fresh"]) for s in SEEDS)
        window = max(100, n_min // 12)
        kernel = np.ones(window) / window
        for arm in rolling_arms:
            curves = np.stack([
                np.convolve(all_hits[(protocol, s)][arm][:n_min].astype(float),
                            kernel, mode="valid")
                for s in SEEDS])
            t = np.arange(window, n_min + 1)
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            if arm == "fresh":
                label, ls = "fresh", "-"
            else:
                fam, r = arm
                label = ("full" if fam == "full" else "$\\Lambda$") \
                    + f"-stale $\\rho$={r}"
                ls = "--" if fam == "full" else ":"
            ax.plot(t, mean, ls, color=colors[arm], label=label)
            ax.fill_between(t, mean - std, mean + std, color=colors[arm],
                            alpha=0.12, linewidth=0)
        ax.set_xlabel("Stream position")
        ax.set_ylabel(f"Prequential acc. ({window}-sample window)")
        ax.set_title(f"{protocol}")
        ax.set_ylim([0, 1.02])
        ax.legend(frameon=False, loc="lower left", ncol=2)

    # row 2: both metrics vs refresh period rho, per stale family
    for ax, protocol in zip(axes[1], protocols):
        rhos = np.array(args.periods)
        series = [
            ("full-stale, prequential", "full",
             lambda s, a: all_hits[(protocol, s)][a].mean(),
             "tab:orange", "o", "-"),
            ("full-stale, held-out", "full",
             lambda s, a: all_heldout[(protocol, s)][a].mean(),
             "tab:red", "s", "-"),
            ("$\\Lambda$-stale, prequential", "lam",
             lambda s, a: all_hits[(protocol, s)][a].mean(),
             "tab:green", "o", "--"),
            ("$\\Lambda$-stale, held-out", "lam",
             lambda s, a: all_heldout[(protocol, s)][a].mean(),
             "tab:olive", "s", "--"),
        ]
        for label, fam, getter, color, marker, ls in series:
            mean = np.array([np.mean([getter(s, (fam, r)) for s in SEEDS])
                             for r in rhos])
            std = np.array([np.std([getter(s, (fam, r)) for s in SEEDS])
                            for r in rhos])
            ax.errorbar(rhos, mean, yerr=std, color=color, marker=marker,
                        markersize=3, capsize=2, linewidth=1, linestyle=ls,
                        label=label)
        ax.set_xscale("log")
        ax.set_xlabel(r"Refresh period $\rho$ (samples)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{protocol} ($\\rho$=1 $\\equiv$ per-sample fresh)")
        ax.set_ylim([0, 1.02])
        ax.legend(frameon=False, loc="lower left")

    plt.tight_layout()
    for ext in (("pdf", "png") if SAVE_PDF else ("png",)):
        for out_dir in (RESULTS_DIR, images_dir):
            plt.savefig(out_dir / f"eb_staleness_panel.{ext}",
                        format=ext, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Panel saved to {images_dir}/eb_staleness_panel.png")


if __name__ == "__main__":
    main()
