"""
pareto_plots.py
----------------
Three-panel Pareto frontier figure (Fig. 3 c/d/e of "Real-time Continual
Learning on Intel Loihi 2"):

  c) Accuracy vs. learning energy efficiency (Hz/J), log-y
  d) Accuracy vs. max learning frequency (Hz),       log-y
  e) Frequency vs. energy efficiency (log-log), with regression line
     through non-Loihi points

Latency and energy values are taken from the revised post-rebuttal
benchmark spreadsheet `table_1_revision.xlsx`, sheet "final table".
The spreadsheet itself lives outside the repository (Proton Drive),
so the relevant rows are inlined below.

Accuracy values are *unchanged* from the original submission and were
measured on Intel Loihi 2 using the proprietary Lava-INL toolchain
(Intel INRC, NDA-restricted). They cannot be regenerated from this
repository alone — see the manuscript for the experimental protocol.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent
IMAGES_DIR = _REPO / "images"

SAVE_PDF = True


# ── Data (inlined from table_1_revision.xlsx, sheet "final table") ────────────
# Subset of 9 rows matching the original Fig. 3 layout:
#   CLP-SNN/Loihi (INT8) + CLP, NCM, Replay, SLDA-periodic(k=1) on CPU/GPU (fp32).

METHODS = ["CLP-SNN", "CLP", "CLP", "ncm", "ncm",
           "replay", "replay", "SLDA", "SLDA"]
DEVICES = ["Loihi 2", "CPU", "GPU", "CPU", "GPU",
           "CPU", "GPU", "CPU", "GPU"]

LATENCY_MS = np.array([
    0.33,        # CLP-SNN  / Loihi / INT8
    1.07555,     # CLP      / CPU   / fp32
    2.58396,     # CLP      / GPU   / fp32
    0.33803,     # ncm      / CPU   / fp32
    0.68864,     # ncm      / GPU   / fp32
    6.58365,     # replay   / CPU   / fp32
    2.47102,     # replay   / GPU   / fp32
    73.19625,    # SLDA(k=1)/ CPU   / fp32
    37.28893,    # SLDA(k=1)/ GPU   / fp32
])

ENERGY_MJ = np.array([
    0.05,        # CLP-SNN  / Loihi
    8.66,        # CLP      / CPU
    14.74,       # CLP      / GPU
    1.85,        # ncm      / CPU
    3.89,        # ncm      / GPU
    50.92,       # replay   / CPU
    14.39,       # replay   / GPU
    677.84,      # SLDA(k=1)/ CPU
    333.37,      # SLDA(k=1)/ GPU
])

# 25-shot accuracy values from Table 1 of the manuscript (Loihi 2 evaluations
# for CLP-SNN; PyTorch reference evaluations for the baselines). The Loihi
# accuracy comes from the proprietary Lava-INL toolchain — see module docstring.
ACCURACY = np.array([90.0, 93.0, 93.0, 84.5, 84.5,
                     91.6, 91.6, 95.7, 95.7])

# Derived metrics
LATENCY_S = LATENCY_MS / 1e3                 # ms -> s
ENERGY_J = ENERGY_MJ / 1e3                   # mJ -> J
THROUGHPUT = 1.0 / LATENCY_S                 # Hz
ENERGY_EFFICIENCY = 1.0 / ENERGY_J           # Hz/J


# ── Styling ───────────────────────────────────────────────────────────────────

PLOT_RC = {
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
    "svg.fonttype": "none",  # keep text as text in SVG (editable in Illustrator)
}


def _styles():
    """Per-point colors and markers (one entry per row in the data arrays).

    Color encodes algorithm; marker encodes architecture.
    """
    p = sns.color_palette("tab10")
    # Color scheme matches experiments/clp_vs_baselines_1shot.py:
    #   CLP / CLP-SNN = p[9] (cyan)
    #   NCM           = p[2] (green)
    #   Replay        = p[3] (red)
    #   SLDA          = p[1] (orange)
    colors = [p[9], p[9], p[9],   # CLP-SNN/Loihi, CLP/CPU, CLP/GPU
              p[2], p[2],          # NCM/CPU, NCM/GPU
              p[3], p[3],          # Replay/CPU, Replay/GPU
              p[1], p[1]]          # SLDA/CPU, SLDA/GPU
    markers = ["*", "o", "D", "o", "D", "o", "D", "o", "D"]
    return colors, markers


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _scatter(ax, x, y, colors, markers):
    """Scatter all 9 points; first point (Loihi) is rendered larger."""
    for i in range(len(x)):
        s = 50 if i == 0 else 9
        ax.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=s)


def _fit_log_y(x, y_log_target):
    """Fit log10(y) = a*x + b on the given (x, y) points; return (xs, ys)."""
    log_y = np.log10(y_log_target)
    coefs = np.polyfit(x, log_y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = 10 ** (coefs[0] * xs + coefs[1])
    return xs, ys


def _fit_log_log(x, y):
    """Fit log10(y) = a*log10(x) + b; return (xs, ys, coefs, r2)."""
    log_x = np.log10(x)
    log_y = np.log10(y)
    coefs = np.polyfit(log_x, log_y, 1)
    y_pred = coefs[0] * log_x + coefs[1]
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    ys = 10 ** (coefs[0] * np.log10(xs) + coefs[1])
    return xs, ys, coefs, r2


def plot_pareto(fig, axes):
    colors, markers = _styles()

    # Panel C: Accuracy vs Energy Efficiency (log-y) ---------------------------
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_ylabel("Learning Energy Efficiency (Hz/J)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Energy Efficiency vs Accuracy")
    ax.set_xlim([80, 97])
    _scatter(ax, ACCURACY, ENERGY_EFFICIENCY, colors, markers)
    xs, ys = _fit_log_y(ACCURACY[3:], ENERGY_EFFICIENCY[3:])  # non-CLP rows
    ax.plot(xs, ys, "b--", linewidth=0.4, alpha=0.8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.2)

    # Panel D: Accuracy vs Throughput (log-y) ----------------------------------
    ax = axes[1]
    ax.set_yscale("log")
    ax.set_ylabel("Max Learning Frequency (Hz)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Throughput vs Accuracy")
    ax.set_xlim([80, 97])
    _scatter(ax, ACCURACY, THROUGHPUT, colors, markers)
    xs, ys = _fit_log_y(ACCURACY[3:], THROUGHPUT[3:])
    ax.plot(xs, ys, "b--", linewidth=0.4, alpha=0.8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.2)

    # Panel E: Throughput vs Energy Efficiency (log-log) -----------------------
    ax = axes[2]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([1, 30000])
    ax.set_xlim([10, 4000])
    ax.set_xlabel("Max Learning Frequency (Hz)")
    ax.set_ylabel("Learning Energy Efficiency (Hz/J)")
    ax.set_title("Energy Efficiency & Throughput")
    _scatter(ax, THROUGHPUT, ENERGY_EFFICIENCY, colors, markers)
    xs, ys, coefs_e, r2_e = _fit_log_log(THROUGHPUT[1:], ENERGY_EFFICIENCY[1:])
    ax.plot(xs, ys, "b--", linewidth=0.4, alpha=0.8)
    ax.text(0.04, 0.96,
            f"slope = {coefs_e[0]:.2f}\n$R^2$ = {r2_e:.3f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75))
    ax.grid(True, which="both", linestyle="--", linewidth=0.2)
    # Stash on the function for main() to print
    plot_pareto._last_panel_e_fit = (coefs_e[0], coefs_e[1], r2_e)

    # ── Legends ───────────────────────────────────────────────────────────────
    p = sns.color_palette("tab10")
    algo_names = ["CLP", "NCM", "Replay", "SLDA"]
    algo_colors = [p[9], p[2], p[3], p[1]]
    algo_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=5, label=name)
        for c, name in zip(algo_colors, algo_names)
    ]
    arch_handles = [
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="black", markersize=10, label="Loihi 2"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="black", markersize=5,
               label="GPU (Nvidia\nJetson Orin Nano)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="black", markersize=5,
               label="CPU (6-core Arm\nCortex-A78AE)"),
    ]
    main_legend = axes[0].legend(handles=algo_handles, title="OCL Algorithms",
                                 fontsize=7, loc="lower left",
                                 bbox_to_anchor=(-0.02, 0))
    axes[0].add_artist(main_legend)
    axes[1].legend(handles=arch_handles, title="Reference Architectures",
                   fontsize=7, loc="lower left", bbox_to_anchor=(-0.02, 0))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(PLOT_RC)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(figsize=(7.087, 3), ncols=3, nrows=1)
    plot_pareto(fig, axes)
    plt.tight_layout()

    out_png = IMAGES_DIR / "pareto_plots.png"
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight")
    fig.savefig(IMAGES_DIR / "pareto_plots.svg", format="svg",
                bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(IMAGES_DIR / "pareto_plots.pdf", format="pdf",
                    bbox_inches="tight")
    plt.close(fig)

    exts = "png+svg" + ("+pdf" if SAVE_PDF else "")
    print(f"Saved {out_png.with_suffix('').relative_to(_REPO)}.{{{exts}}}")
    slope, intercept, r2 = plot_pareto._last_panel_e_fit
    print(f"Panel E log-log fit: slope = {slope:.3f}, "
          f"intercept = {intercept:.3f}, R^2 = {r2:.4f}")


if __name__ == "__main__":
    main()
