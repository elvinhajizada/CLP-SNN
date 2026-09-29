"""
plot_confirmation.py
--------------------
Step 5 figure and tables: final accuracy vs activation threshold (mean ± std
over seeds 10/20/30) for CLP, allocation-only CLP-SNN (cached hw sweep), the
released adaptive CLP-SNN and the fixed adaptive CLP-SNN, per learning path
(rows) and setting (columns). Means below the axis floor (collapsed runs) are
drawn as a down-triangle on the floor and labelled with their value.

Writes results/adaptive_protos/comparison_{1shot,25shot}.csv and
images/adaptive_protos_confirmation.{png,pdf}.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent.parent), str(_HERE.parent), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_ablation as ra  # noqa: E402

THS = [0.70, 0.75, 0.80, 0.85]
SEEDS = [10, 20, 30]
FIXED = {"FP32": "fix", "INT8": "fix+H4b2"}
YLIM = {"1shot": (46, 61), "25shot": (86, 94.5)}
# validated categorical slots 1-4 (dataviz reference palette, light mode)
STYLE = {
    "CLP":                dict(color="#2a78d6", marker="o", linestyle="-"),
    "allocation-only":    dict(color="#eb6834", marker="s", linestyle="--"),
    "adaptive, released": dict(color="#eda100", marker="v", linestyle=":"),
    "adaptive, fixed":    dict(color="#1baf7a", marker="D", linestyle="-"),
}


def series(path, setting, runs, base):
    out = {}
    for name, key in (("CLP", "CLP"), ("allocation-only", f"{path}-NA")):
        out[name] = [[base[(setting, key, th)][s][0] for s in SEEDS] for th in THS]
        out[name + " protos"] = [[base[(setting, key, th)][s][1] for s in SEEDS]
                                 for th in THS]
    for name, cfg in (("adaptive, released", "base"),
                      ("adaptive, fixed", FIXED[path])):
        acc, pro = [], []
        for th in THS:
            rs = {r["seed"]: r for r in runs if r["path"] == path
                  and r["config"] == cfg and r["setting"] == setting
                  and r["threshold"] == th}
            acc.append([rs[s]["final_acc"] for s in SEEDS])
            pro.append([rs[s]["n_protos"] for s in SEEDS])
        out[name], out[name + " protos"] = acc, pro
    return out


def main():
    runs = ra.load_runs()
    base = ra.load_cached_baselines()
    plt.rcParams.update(ra.hw.b1.PLOT_CONFIG)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), squeeze=False)
    for setting in ("1shot", "25shot"):
        rows = []
        for path in ("FP32", "INT8"):
            s = series(path, setting, runs, base)
            for name in STYLE:
                for th, a, p in zip(THS, s[name], s[name + " protos"]):
                    a = np.array(a) * 100
                    rows.append([path, name, f"{th:.2f}", f"{a.mean():.2f}",
                                 f"{a.std():.2f}", f"{a.min():.2f}",
                                 f"{np.mean(p):.0f}"])
        with open(ra.OUT / f"comparison_{setting}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Path", "Variant", "Threshold", "Final acc (%)",
                        "Std", "Min seed", "Protos"])
            w.writerows(rows)

    for i, path in enumerate(("FP32", "INT8")):
        for j, setting in enumerate(("1shot", "25shot")):
            ax = axes[i][j]
            s = series(path, setting, runs, base)
            lo, hi = YLIM[setting]
            for name, st in STYLE.items():
                a = np.array(s[name]) * 100
                m, sd = a.mean(axis=1), a.std(axis=1)
                inside = m >= lo
                ax.errorbar(np.array(THS)[inside], m[inside], yerr=sd[inside],
                            markersize=4, linewidth=1.2, capsize=2,
                            label=name, **st)
                for th, v in zip(np.array(THS)[~inside], m[~inside]):
                    ax.plot(th, lo + 0.15, marker="v", markersize=6,
                            color=st["color"], linestyle="none")
                    ax.annotate(f"{v:.0f}%", (th, lo + 0.15),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=6, color="#555555")
            ax.set_ylim(lo, hi)
            ax.set_xticks(THS)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(f"{path}, {setting.replace('shot', '-shot')}")
            if j == 0:
                ax.set_ylabel("Final accuracy (%)")
            if i == 1:
                ax.set_xlabel("Activation threshold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    img = ra.hw.IMAGES_DIR / "adaptive_protos_confirmation"
    for ext in ("png", "pdf"):
        plt.savefig(f"{img}.{ext}", bbox_inches="tight", dpi=200)
    print(f"figure -> {img}.png; tables -> {ra.OUT}/comparison_*.csv")


if __name__ == "__main__":
    main()
