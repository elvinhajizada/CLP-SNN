"""
plot_slda_heldout_trajectory.py
-------------------------------
Render held-out accuracy vs stream position from the staleness study's
saved trajectories (no experiment re-run needed).

Reads experiments/results/slda_staleness_study/staleness.npz, which stores
the balanced 400-sample held-out probe accuracy (restricted to classes seen
so far) every eval_every = 53 samples, for every arm and seed.

The fresh line is taken from the full_rho1 arm (identical to fresh by
construction).

Output: eb_heldout_trajectory.png/pdf in the results dir and images/.

Usage (from repo root or experiments/):
  python experiments/plot_slda_heldout_trajectory.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RESULTS_DIR = _REPO / "experiments" / "results" / "slda_staleness_study"
SEEDS = [10, 20, 30]
SAVE_PDF = True

# arm key in npz -> (label, color, linestyle)
ARMS = [
    ("full_rho1", "fresh", "tab:blue", "-"),
    ("full_rho60", "full-stale $\\rho$=60", "tab:orange", "--"),
    ("full_rho300", "full-stale $\\rho$=300", "tab:red", "--"),
    ("lam_rho60", "$\\Lambda$-stale $\\rho$=60", "tab:green", ":"),
    ("lam_rho300", "$\\Lambda$-stale $\\rho$=300", "tab:olive", ":"),
]

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


def main():
    plt.rcParams.update(PLOT_CONFIG)
    d = np.load(RESULTS_DIR / "staleness.npz")
    images_dir = _REPO / "images"

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for ax, protocol in zip(axes, ["1shot", "25shot"]):
        # positions are identical across seeds up to the last (stream-length)
        # entry; truncate to the shortest run
        pos_all = [d[f"{protocol}_{s}_positions"] for s in SEEDS]
        n_min = min(len(p) for p in pos_all)
        pos = pos_all[0][:n_min]
        for key, label, color, ls in ARMS:
            curves = np.stack([d[f"{protocol}_{s}_heldout_{key}"][:n_min]
                               for s in SEEDS])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            ax.plot(pos, mean, ls, color=color, label=label)
            ax.fill_between(pos, mean - std, mean + std, color=color,
                            alpha=0.12, linewidth=0)
        ax.set_xlabel("Stream position")
        ax.set_ylabel("Held-out accuracy (seen classes)")
        ax.set_title(protocol)
        ax.set_ylim([0, 1.02])
        ax.legend(frameon=False, loc="lower left", ncol=2)

    plt.tight_layout()
    for ext in (("pdf", "png") if SAVE_PDF else ("png",)):
        for out_dir in (RESULTS_DIR, images_dir):
            plt.savefig(out_dir / f"eb_heldout_trajectory.{ext}",
                        format=ext, bbox_inches="tight")
    plt.close()
    print(f"Saved eb_heldout_trajectory.png/pdf to {RESULTS_DIR} and "
          f"{images_dir}")


if __name__ == "__main__":
    main()
