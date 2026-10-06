"""
pareto_plots.py
----------------
Three-panel Pareto frontier figure (Fig. 3 c/d/e):

  c) Accuracy vs. learning energy efficiency (samples/J), log-y
  d) Accuracy vs. learning throughput      (samples/s), log-y
  e) Throughput vs. energy efficiency (log-log), with a regression line
     through the conventional (non-Loihi) points

Latency and energy are read from `benchmarks/reports_table1.csv`, the unified
Orin Nano harness (1-shot seed-10 stream), and composed into a per-sample
streaming step by the same rule the manuscript's Table 1 uses:

    step = update                if the update pass already produces the
                                 prediction for the incoming sample
    step = update + query        otherwise

The per-model flags (`_PRED_IN_FIT` below) follow from the implementations:
CLP and Replay form the prediction inside `fit` (models/CLP.py, the
best-matching prototype; models/Replay.py, row 0 of the training forward
pass), NCM and SLDA do not.

Accuracy values are 25-shot class-incremental accuracies from Table 1 of the
manuscript. The Loihi 2 row (both its cost and its accuracy) was measured on
Intel Loihi 2 with the proprietary Lava-INL toolchain (Intel INRC,
NDA-restricted) and cannot be regenerated from this repository alone; it is
declared below with its source.
"""

from pathlib import Path

import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent
IMAGES_DIR = _REPO / "images"
TABLE1_CSV = _REPO / "benchmarks" / "reports_table1.csv"

SAVE_PDF = True


# ── Data ──────────────────────────────────────────────────────────────────────

# The nine Table 1 rows: CLP-SNN on Loihi 2, plus four conventional learners on
# the Orin CPU and GPU at FP32. FP16 rows and the naive per-sample-inversion
# SLDA anchor are Extended Data and are not plotted.
#   csv Method -> (display name, _PRED_IN_FIT key)
_ROWS = [
    ("CLP", "orin-gpu", "fp32", "CLP", "CLP"),
    ("CLP", "orin-cpu", "fp32", "CLP", "CLP"),
    ("ncm", "orin-gpu", "fp32", "NCM", "NCM"),
    ("ncm", "orin-cpu", "fp32", "NCM", "NCM"),
    ("replay", "orin-gpu", "fp32", "Replay", "Replay"),
    ("replay", "orin-cpu", "fp32", "Replay", "Replay"),
    ("SLDA-rank1", "orin-gpu", "fp32", "SLDA", "SLDA (rank-one)"),
    ("SLDA-rank1", "orin-cpu", "fp32", "SLDA", "SLDA (rank-one)"),
]

# Does the update pass already produce the prediction for the incoming sample?
_PRED_IN_FIT = {
    "CLP": True,
    "NCM": False,
    "Replay": True,
    "SLDA (rank-one)": False,
}

# 25-shot class-incremental accuracy (%), Table 1 of the manuscript.
ACCURACY_25SHOT = {
    "CLP-SNN": 90.0,
    "CLP": 93.0,
    "NCM": 84.5,
    "Replay": 91.6,
    "SLDA": 96.2,
}

# CLP-SNN on Loihi 2: one on-chip pass covers scoring and plasticity, so there
# is no separate query cost to add. Round-2 Table 1, Lava-INL measurement.
LOIHI = dict(method="CLP-SNN", device="Loihi 2", latency_ms=0.33, energy_mj=0.05)


def load_points():
    """Return the nine plotted points, step-composed from the Table 1 harness.

    Each point is (method, device, latency_ms, energy_mj, accuracy, rule).
    """
    with open(TABLE1_CSV, newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))

    points = [(LOIHI["method"], LOIHI["device"], LOIHI["latency_ms"],
               LOIHI["energy_mj"], ACCURACY_25SHOT[LOIHI["method"]],
               "single on-chip pass")]

    for csv_name, device, precision, short, op_key in _ROWS:
        row = next(r for r in csv_rows
                   if r["Method"] == csv_name and r["Device"] == device
                   and r["Precision"] == precision)
        pred_in_fit = _PRED_IN_FIT[op_key]
        if pred_in_fit:
            lat = float(row["C_upd Latency (ms)"])
            energy = float(row["C_upd Total E (mJ)"])
            rule = "update only (prediction inside update pass)"
        else:
            lat = float(row["C_upd+C_qry Latency (ms)"])
            energy = float(row["C_upd+C_qry Total E (mJ)"])
            rule = "update + query"
        dev = "GPU" if device.endswith("gpu") else "CPU"
        points.append((short, dev, lat, energy, ACCURACY_25SHOT[short], rule))

    return points


POINTS = load_points()

METHODS = [p[0] for p in POINTS]
DEVICES = [p[1] for p in POINTS]
LATENCY_MS = np.array([p[2] for p in POINTS])
ENERGY_MJ = np.array([p[3] for p in POINTS])
ACCURACY = np.array([p[4] for p in POINTS])

# Derived metrics. Both are per-sample rates over the streaming step, so the
# units are samples per second and samples per joule, not hertz and hertz per
# joule: one "event" here is one learn-and-predict step on one sample.
THROUGHPUT = 1e3 / LATENCY_MS            # samples/s
ENERGY_EFFICIENCY = 1e3 / ENERGY_MJ      # samples/J


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

FRONT_COLOR = "0.45"


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
    algo_color = {"CLP-SNN": p[9], "CLP": p[9], "NCM": p[2],
                  "Replay": p[3], "SLDA": p[1]}
    arch_marker = {"Loihi 2": "*", "CPU": "o", "GPU": "D"}
    colors = [algo_color[m] for m in METHODS]
    markers = [arch_marker[d] for d in DEVICES]
    return colors, markers


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _scatter(ax, x, y, colors, markers):
    """Scatter all points; the Loihi point (index 0) is rendered larger."""
    for i in range(len(x)):
        s = 50 if i == 0 else 9
        ax.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=s, zorder=3)


def pareto_front(x, y):
    """Indices of the non-dominated points when maximising both x and y.

    Returned in increasing x, which is the order they are drawn in.
    """
    idx = []
    for i in range(len(x)):
        dominated = any(
            (x[j] >= x[i] and y[j] >= y[i]) and (x[j] > x[i] or y[j] > y[i])
            for j in range(len(x)) if j != i
        )
        if not dominated:
            idx.append(i)
    return sorted(idx, key=lambda i: x[i])


def _draw_front(ax, x, y, idx):
    """Connect the non-dominated points with a staircase, as R4 asked.

    The frontier is the best y attainable at accuracy *at least* x, so it drops
    at each frontier point and runs flat until the next one: vertical first,
    then horizontal. Drawing it the other way round would imply the higher
    efficiency is still available at the higher accuracy, which it is not.
    """
    xs, ys = [x[i] for i in idx], [y[i] for i in idx]
    step_x, step_y = [xs[0]], [ys[0]]
    for k in range(1, len(xs)):
        step_x += [xs[k - 1], xs[k]]
        step_y += [ys[k], ys[k]]
    ax.plot(step_x, step_y, color=FRONT_COLOR, linewidth=0.7,
            linestyle="-", alpha=0.9, zorder=2)


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
    ax.set_ylabel("Learning energy efficiency (samples/J)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Energy Efficiency vs Accuracy")
    ax.set_xlim([82, 98])
    ax.set_ylim([10, 1.2e5])
    front_c = pareto_front(ACCURACY, ENERGY_EFFICIENCY)
    _draw_front(ax, ACCURACY, ENERGY_EFFICIENCY, front_c)
    _scatter(ax, ACCURACY, ENERGY_EFFICIENCY, colors, markers)
    ax.grid(True, which="both", linestyle="--", linewidth=0.2)

    # Panel D: Accuracy vs Throughput (log-y) ----------------------------------
    ax = axes[1]
    ax.set_yscale("log")
    ax.set_ylabel("Learning throughput (samples/s)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Throughput vs Accuracy")
    ax.set_xlim([82, 98])
    ax.set_ylim([100, 6.0e4])
    front_d = pareto_front(ACCURACY, THROUGHPUT)
    _draw_front(ax, ACCURACY, THROUGHPUT, front_d)
    _scatter(ax, ACCURACY, THROUGHPUT, colors, markers)
    ax.grid(True, which="both", linestyle="--", linewidth=0.2)

    # Panel E: Throughput vs Energy Efficiency (log-log) -----------------------
    # The conventional points trace an energy-latency line: on a fixed power
    # envelope, going faster buys proportionally better energy per sample. The
    # fit is over the eight Orin rows only; Loihi's offset from it is the
    # quantity of interest.
    ax = axes[2]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning throughput (samples/s)")
    ax.set_ylabel("Learning energy efficiency (samples/J)")
    ax.set_title("Energy Efficiency & Throughput")
    _scatter(ax, THROUGHPUT, ENERGY_EFFICIENCY, colors, markers)
    xs, ys, coefs_e, r2_e = _fit_log_log(THROUGHPUT[1:], ENERGY_EFFICIENCY[1:])
    ax.plot(xs, ys, "b--", linewidth=0.4, alpha=0.8)
    ax.set_xlim([100, 6000])
    ax.set_ylim([10, 40000])
    ax.text(0.04, 0.96,
            f"slope = {coefs_e[0]:.2f}\n$R^2$ = {r2_e:.3f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75))
    ax.grid(True, which="both", linestyle="--", linewidth=0.2)
    # Stash on the function for main() to print
    plot_pareto._last_panel_e_fit = (coefs_e[0], coefs_e[1], r2_e)
    plot_pareto._fronts = (front_c, front_d)

    # ── Legends ───────────────────────────────────────────────────────────────
    p = sns.color_palette("tab10")
    algo_names = ["CLP", "NCM", "Replay", "SLDA"]
    algo_colors = [p[9], p[2], p[3], p[1]]
    algo_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=5, label=name)
        for c, name in zip(algo_colors, algo_names)
    ]
    algo_handles.append(
        Line2D([0], [0], color=FRONT_COLOR, linewidth=0.7,
               label="Pareto frontier"))
    arch_handles = [
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="black", markersize=10, label="Loihi 2"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="black", markersize=5,
               label="GPU (Jetson Orin Nano)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="black", markersize=5,
               label="CPU (Arm Cortex-A78AE)"),
    ]
    # Both legends sit in regions the data leaves empty: the top-left corner of
    # each accuracy panel, above the highest low-accuracy point and left of the
    # frontier's first step.
    main_legend = axes[0].legend(handles=algo_handles, title="OCL Algorithms",
                                 fontsize=6, title_fontsize=6, loc="upper left",
                                 handletextpad=0.4, borderpad=0.35,
                                 labelspacing=0.3, framealpha=0.85)
    axes[0].add_artist(main_legend)
    axes[1].legend(handles=arch_handles, title="Reference Architectures",
                   fontsize=6, title_fontsize=6, loc="upper left",
                   handletextpad=0.4, borderpad=0.35, labelspacing=0.3,
                   framealpha=0.85)


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

    print("\nStep-composed points (from benchmarks/reports_table1.csv):")
    print(f"  {'method':9s} {'dev':7s} {'lat ms':>7s} {'mJ':>7s} {'acc':>6s} "
          f"{'samples/s':>10s} {'samples/J':>10s}  rule")
    for i, (m, d, lat, e, acc, rule) in enumerate(POINTS):
        print(f"  {m:9s} {d:7s} {lat:7.2f} {e:7.2f} {acc:6.1f} "
              f"{THROUGHPUT[i]:10.0f} {ENERGY_EFFICIENCY[i]:10.0f}  {rule}")

    front_c, front_d = plot_pareto._fronts
    for label, idx in (("panel c (accuracy vs energy efficiency)", front_c),
                       ("panel d (accuracy vs throughput)", front_d)):
        names = " -> ".join(f"{POINTS[i][0]}/{POINTS[i][1]}" for i in idx)
        print(f"\nPareto frontier, {label}: {names}")

    slope, intercept, r2 = plot_pareto._last_panel_e_fit
    print(f"\nPanel E log-log fit over the 8 conventional points: "
          f"slope = {slope:.3f}, intercept = {intercept:.3f}, R^2 = {r2:.4f}")
    pred = 10 ** (slope * np.log10(THROUGHPUT[0]) + intercept)
    print(f"  Loihi: {ENERGY_EFFICIENCY[0]:.0f} samples/J measured vs "
          f"{pred:.1f} predicted by the conventional line "
          f"-> {ENERGY_EFFICIENCY[0] / pred:.0f}x above it")


if __name__ == "__main__":
    main()
