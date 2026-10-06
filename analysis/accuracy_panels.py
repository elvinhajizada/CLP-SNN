"""
accuracy_panels.py
------------------
Accuracy trajectories of Fig. 3 a/b (1-shot and 25-shot class-incremental
OpenLORIS), drawn in the style of the original figure so the output can be
spliced into the composite Fig. 3 SVG.

Every curve ends at the final accuracy reported in Table 1 of the manuscript:

  CLP-SNN       measured on Loihi 2: data/loihi2/
  others        experiments/results/clp_vs_baselines_{1,25}shot/accuracies.npz
                (each run seeded on its own, so the stochastic methods repeat;
                SLDA is the exact rank-one implementation with lambda = 1, and
                is checked against experiments/slda_regularization_equivalence.py)

Output: images/accuracy_panels.svg / .pdf (7.087 x 3.54 in, text as text).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_REPO = Path(__file__).resolve().parent.parent
RES = _REPO / "experiments" / "results"
IMAGES_DIR = _REPO / "images"

plt.rcParams.update({
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
    "svg.fonttype": "none",
})

# (label, colour, linestyle, linewidth), in legend order
_p = sns.color_palette("tab10")
STYLE = [
    ("Perceptron",      "#606060", "-",  1.0),
    ("Fine-tuning",     "#B0B0B0", "-",  1.0),
    ("NCM",             _p[2],     "-",  1.0),
    ("Replay",          _p[3],     "-",  1.0),
    ("SLDA",            _p[1],     "-",  1.0),
    ("CLP",             _p[9],     "-",  1.5),
    ("CLP-SNN",         _p[9],     "-.", 1.5),
]

# Final accuracies of Table 1 (%), mean over three class orders.
TABLE1 = {
    "1shot":  {"NCM": 53.1, "Replay": 55.1, "SLDA": 54.8, "CLP": 57.0, "CLP-SNN": 55.4},
    "25shot": {"NCM": 84.5, "Replay": 90.8, "SLDA": 96.2, "CLP": 93.0, "CLP-SNN": 90.0},
}


def load(protocol):
    """Return accuracies (steps, 7 methods, 3 class orders) in STYLE order."""
    eq = np.load(RES / "slda_regularization_equivalence" / "equivalence.npz")
    if protocol == "1shot":
        base = np.load(RES / "clp_vs_baselines_1shot" / "accuracies.npz")["accuracies"]
    else:
        base = np.load(RES / "clp_vs_baselines_25shot" / "accuracies.npz")["accuracies"]
    chip = np.load(_REPO / "data" / "loihi2" / f"accuracies_clp_loihi_{protocol}.npy")[:, 0]
    acc = np.stack([base[:, 0], base[:, 1], base[:, 2], base[:, 3],
                    base[:, 4], base[:, 6], chip], axis=1)
    assert np.allclose(acc[-1, 4], eq[f"{protocol}_rankone"][-1], atol=1e-6), \
        "baseline-run SLDA differs from the rank-one equivalence run"
    finals = 100 * acc[-1].mean(axis=1)
    for name, want in TABLE1[protocol].items():
        got = finals[[s[0] for s in STYLE].index(name)]
        assert abs(got - want) < 0.05 + 1e-9, f"{protocol} {name}: {got:.2f} vs Table 1 {want}"
    return acc


def plot():
    acc1, acc25 = load("1shot"), load("25shot")
    fig, (ax1, ax2) = plt.subplots(figsize=(7.087, 3.54), ncols=2, nrows=1)
    fig.patch.set_alpha(0.0)

    m1 = acc1.mean(axis=2)[1:]                      # steps 2..40
    t1 = np.arange(2, 41)
    m25 = np.vstack([np.zeros((1, acc25.shape[1])), acc25.mean(axis=2)])  # shot 0..25
    t25 = np.arange(0, acc25.shape[0] + 1)

    for i, (label, colour, ls, lw) in enumerate(STYLE):
        ax1.plot(t1, m1[:, i], color=colour, label=label, linewidth=lw, linestyle=ls, alpha=0.9)
        ax2.plot(t25, m25[:, i], color=colour, label=label, linewidth=lw, linestyle=ls, alpha=0.9)

    ax1.set_xlabel("Incremental Learning Step \n (# of Classes Seen)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.2, 1])
    ax1.set_xlim([1.8, 40.2])
    ax1.set_xticks(t1[::4])

    ax2.set_xlabel("Incremental Learning Step \n (# of shots)")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.set_xlim([0, t25[-1] + 0.1])
    ax2.set_xticks(t25[0::5])
    ax2.tick_params(axis="x", which="major", length=5, width=1, direction="in", labelsize=7)

    ax1.legend(loc="upper center", frameon=False, ncol=4, fontsize=9,
               handlelength=3.0, bbox_to_anchor=(1, 1.23))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.85, wspace=0.2)

    IMAGES_DIR.mkdir(exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(IMAGES_DIR / f"accuracy_panels.{ext}", format=ext)
    plt.close(fig)
    for name, acc in (("1-shot", acc1), ("25-shot", acc25)):
        print(name, {s[0]: round(100 * acc[-1, i].mean(), 2) for i, s in enumerate(STYLE)})


if __name__ == "__main__":
    plot()
