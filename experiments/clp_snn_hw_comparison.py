"""
clp_snn_hw_comparison.py
------------------------
Threshold sweep comparing CLP and CLP-SNN simulation variants against the
CLP-SNN Loihi 2 hardware results, on the 1-shot and 25-shot OpenLoris
benchmarks.

Motivation: the Loihi 2 deployment ran CLP-SNN in allocation-only mode
(prototypes frozen after initial allocation — the limitation stated in the
paper), at a high activation threshold with a large prototype pool. The
committed simulation configs run the opposite regime (adaptive updates,
low threshold, few prototypes). This experiment maps the full operating
space: adaptive vs non-adaptive x FP32 vs INT8 x threshold in [0.70, 0.85],
reporting final accuracy, AAA (mean accuracy over the published checkpoint
cadence), and the number of allocated prototypes (neurons).

Variants:
  CLP      - ContinuallyLearningPrototypes, sim_th_init = threshold
  FP32-A   - CLPSNN float,  adaptive_protos=True  (update on hit/miss)
  FP32-NA  - CLPSNN float,  adaptive_protos=False (allocation-only, HW mode)
  INT8-A   - CLPSNN integer pipeline, adaptive
  INT8-NA  - CLPSNN integer pipeline, allocation-only (closest to Loihi 2)

Protocol identical to clp_vs_baselines_{1shot,25shot}.py (imported):
same seeds, data ordering, test set, and checkpoint cadence
(40 class boundaries / 25 rounds).

Results accumulate in experiments/results/clp_snn_hw_comparison/:
  summary.csv                        - one row per (setting, variant, th, seed),
                                       upserted, so partial runs compose
  curves_<setting>_<variant>_th<th>_s<seed>.npy - checkpoint accuracy curve
  comparison table + figure via --plot (also auto-run after sweeps)

Usage:
  python experiments/clp_snn_hw_comparison.py --setting 1shot
  python experiments/clp_snn_hw_comparison.py --setting 25shot --seeds 30
  python experiments/clp_snn_hw_comparison.py --plot        # figure/table only
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
for p in (str(_REPO), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import clp_vs_baselines_1shot as b1  # noqa: E402
import clp_vs_baselines_25shot as b25  # noqa: E402
from models.CLP import ContinuallyLearningPrototypes  # noqa: E402
from models.CLP_SNN import CLPSNN  # noqa: E402

RESULTS_DIR = _REPO / "experiments" / "results" / "clp_snn_hw_comparison"
IMAGES_DIR = _REPO / "images"
LOIHI_DIR = _REPO / "data" / "loihi2"  # CLP-SNN accuracies measured on Loihi 2

SEEDS = [10, 20, 30]
THRESHOLDS = [0.70, 0.75, 0.80, 0.85]
VARIANTS = ["CLP", "FP32-A", "FP32-NA", "INT8-A", "INT8-NA"]
N_PROTOS = 16000         # large pool so allocation never saturates
GLOBAL_SEED = 42
SAVE_PDF = True

SUMMARY = RESULTS_DIR / "summary.csv"
SUMMARY_FIELDS = ["setting", "variant", "threshold", "seed",
                  "final_acc", "aaa", "n_protos_alloc", "n_protos_pool"]


# ── Classifier factory ────────────────────────────────────────────────────────

def build(variant: str, threshold: float, setting: str, pool: int):
    if variant == "CLP":
        return ContinuallyLearningPrototypes(
            b1.FEATURE_SIZE, n_protos=pool, num_classes=b1.NUM_CLASSES,
            backbone=None, alpha_init=1, sim_th_init=threshold, n_wta=1,
            k_hit=1, k_miss=1, adaptive_th=False,
            learn_outliers=(setting == "1shot"), device="cpu",
        )
    quant = variant.startswith("INT8")
    adaptive = variant.endswith("-A")
    return CLPSNN(
        b1.FEATURE_SIZE, n_protos=pool, num_classes=b1.NUM_CLASSES,
        threshold=threshold, g_inc=0.5, use_quantization=quant,
        adaptive_protos=adaptive, device="cpu",
    )


def allocated_count(clf) -> int:
    if isinstance(clf, CLPSNN):
        return int(clf.allocator.get_allocated_count())
    return int(clf._n_allocated())


# ── Streams (published checkpoint cadence) ────────────────────────────────────

def run_1shot(clf, seed, test_data):
    X_test_raw, X_test_norm, y_test = test_data
    _, X_tr, y_tr, boundaries, class_order = b1.load_train_split(seed)
    accs = []
    chk = 0
    for i, (xi, yi) in enumerate(zip(X_tr, y_tr), start=1):
        clf.fit(xi.view(b1.FEATURE_SIZE), yi.view(1), i - 1)
        if i in boundaries and chk < b1.N_STEPS:
            seen = class_order[: chk + 1]
            accs.append(b1.evaluate_seen_only(
                clf, X_test_raw, X_test_norm, y_test, seen, False))
            chk += 1
    return np.array(accs)


def run_25shot(clf, seed, test_data):
    X_test_raw, X_test_norm, y_test = test_data
    X_tr, _, y_tr, checkpoints = b25.load_train_split(seed)
    pos_set = set(int(c) for c in checkpoints)
    accs = []
    for i, (xi, yi) in enumerate(zip(X_tr, y_tr), start=1):
        clf.fit(xi.view(b25.FEATURE_SIZE), yi.view(1), i - 1)
        if i in pos_set:
            accs.append(b25.evaluate(
                clf, X_test_raw, X_test_norm, y_test, False))
    return np.array(accs)


# ── Incremental summary ───────────────────────────────────────────────────────

def load_summary():
    rows = {}
    if SUMMARY.exists():
        with open(SUMMARY, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["setting"], r["variant"],
                       float(r["threshold"]), int(r["seed"]))
                rows[key] = r
    return rows


def save_summary(rows):
    with open(SUMMARY, "w", newline="", encoding="utf-8") as f:
        # restval covers rows from before the pool column existed (pool=4000)
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="4000")
        w.writeheader()
        for key in sorted(rows):
            w.writerow({k: rows[key].get(k, "4000") for k in SUMMARY_FIELDS})


def upsert(rows, setting, variant, th, seed, curve, n_alloc, pool):
    rows[(setting, variant, th, seed)] = dict(
        setting=setting, variant=variant, threshold=th, seed=seed,
        final_acc=f"{curve[-1]:.6f}", aaa=f"{curve.mean():.6f}",
        n_protos_alloc=n_alloc, n_protos_pool=pool,
    )
    np.save(RESULTS_DIR /
            f"curves_{setting}_{variant}_th{th:.2f}_s{seed}.npy", curve)
    save_summary(rows)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_setting(setting, seeds, variants, thresholds, pool, rerun=False):
    rows = load_summary()
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    test_data = b1.load_test_set() if setting == "1shot" else b25.load_test_set()
    runner = run_1shot if setting == "1shot" else run_25shot

    n_total = len(seeds) * len(variants) * len(thresholds)
    n_done = 0
    for seed in seeds:
        for th in thresholds:
            for variant in variants:
                n_done += 1
                key = (setting, variant, th, seed)
                if key in rows and not rerun:
                    print(f"[{n_done}/{n_total}] {setting} {variant} "
                          f"th={th:.2f} seed={seed}: cached, skipping")
                    continue
                print(f"[{n_done}/{n_total}] {setting} {variant} "
                      f"th={th:.2f} seed={seed}...", end=" ", flush=True)
                np.random.seed(GLOBAL_SEED)
                random.seed(GLOBAL_SEED)
                torch.manual_seed(GLOBAL_SEED)
                clf = build(variant, th, setting, pool)
                curve = runner(clf, seed, test_data)
                n_alloc = allocated_count(clf)
                upsert(rows, setting, variant, th, seed, curve, n_alloc, pool)
                sat = "  ** POOL SATURATED **" if n_alloc >= pool else ""
                print(f"final={curve[-1]*100:.1f}%  AAA={curve.mean()*100:.1f}%"
                      f"  protos={n_alloc}/{pool}{sat}")


# ── Hardware reference ────────────────────────────────────────────────────────

def load_hardware(setting):
    """Return (final mean, final std, aaa mean, aaa std) or None."""
    path = LOIHI_DIR / f"accuracies_clp_loihi_{setting}.npy"
    if not path.exists():
        return None
    a = np.load(path)[:, 0, :]  # (checkpoints, seeds)
    final, aaa = a[-1], a.mean(axis=0)
    return final.mean(), final.std(), aaa.mean(), aaa.std()


# ── Table + figure ────────────────────────────────────────────────────────────

def aggregate(rows, setting):
    """{(variant, th): dict of mean/std arrays} over available seeds."""
    out = {}
    for (s, variant, th, seed), r in rows.items():
        if s != setting:
            continue
        out.setdefault((variant, th), []).append(
            (float(r["final_acc"]), float(r["aaa"]),
             int(r["n_protos_alloc"])))
    agg = {}
    for key, vals in out.items():
        v = np.array(vals)  # (n_seeds, 3)
        agg[key] = dict(final=v[:, 0].mean(), final_std=v[:, 0].std(),
                        aaa=v[:, 1].mean(), aaa_std=v[:, 1].std(),
                        protos=v[:, 2].mean(), protos_std=v[:, 2].std(),
                        n_seeds=len(vals))
    return agg


def write_table(rows, setting):
    agg = aggregate(rows, setting)
    if not agg:
        return
    path = RESULTS_DIR / f"comparison_{setting}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Threshold", "Final acc (%)", "Final std",
                    "AAA (%)", "AAA std", "Protos", "Protos std", "Seeds"])
        for variant in VARIANTS:
            for th in THRESHOLDS:
                a = agg.get((variant, th))
                if a is None:
                    continue
                w.writerow([variant, f"{th:.2f}",
                            f"{a['final']*100:.2f}", f"{a['final_std']*100:.2f}",
                            f"{a['aaa']*100:.2f}", f"{a['aaa_std']*100:.2f}",
                            f"{a['protos']:.0f}", f"{a['protos_std']:.0f}",
                            a["n_seeds"]])
        hw = load_hardware(setting)
        if hw is not None:
            w.writerow(["CLP-SNN (Loihi 2)", "-",
                        f"{hw[0]*100:.2f}", f"{hw[1]*100:.2f}",
                        f"{hw[2]*100:.2f}", f"{hw[3]*100:.2f}",
                        "n/a", "n/a", 3])
    print(f"table -> {path}")


VARIANT_STYLE = {
    "CLP":     dict(color="#1f77b4", marker="o"),
    "FP32-A":  dict(color="#2ca02c", marker="s"),
    "FP32-NA": dict(color="#2ca02c", marker="s", linestyle="--"),
    "INT8-A":  dict(color="#d62728", marker="^"),
    "INT8-NA": dict(color="#d62728", marker="^", linestyle="--"),
}


def plot(rows, settings=("1shot", "25shot")):
    plt.rcParams.update(b1.PLOT_CONFIG)
    settings = [s for s in settings if aggregate(rows, s)]
    if not settings:
        print("nothing to plot")
        return
    fig, axes = plt.subplots(2, len(settings),
                             figsize=(3.5 * len(settings), 4.6), squeeze=False)
    for j, setting in enumerate(settings):
        agg = aggregate(rows, setting)
        ax_acc, ax_pro = axes[0][j], axes[1][j]
        for variant in VARIANTS:
            ths = [th for th in THRESHOLDS if (variant, th) in agg]
            if not ths:
                continue
            style = VARIANT_STYLE[variant]
            fin = np.array([agg[(variant, th)]["final"] for th in ths])
            fstd = np.array([agg[(variant, th)]["final_std"] for th in ths])
            pro = np.array([agg[(variant, th)]["protos"] for th in ths])
            label = variant + (" (allocation-only)"
                               if variant.endswith("-NA") else "")
            ax_acc.errorbar(ths, fin * 100, yerr=fstd * 100, markersize=3,
                            linewidth=1, capsize=2, label=label, **style)
            ax_pro.plot(ths, pro, markersize=3, linewidth=1, **style)
        hw = load_hardware(setting)
        if hw is not None:
            ax_acc.axhline(hw[0] * 100, color="k", linewidth=0.8,
                           linestyle=":")
            ax_acc.axhspan((hw[0] - hw[1]) * 100, (hw[0] + hw[1]) * 100,
                           color="k", alpha=0.08,
                           label="CLP-SNN Loihi 2 (final)")
        ax_acc.set_title(f"{setting}")
        ax_acc.set_ylabel("Final accuracy (%)")
        ax_pro.set_ylabel("Allocated prototypes")
        ax_pro.set_xlabel("Activation threshold")
        for ax in (ax_acc, ax_pro):
            ax.set_xticks(THRESHOLDS)
    axes[0][0].legend(frameon=False, fontsize=5, loc="lower left")
    plt.tight_layout()
    for ext in (("pdf", "png") if SAVE_PDF else ("png",)):
        plt.savefig(IMAGES_DIR / f"clp_snn_hw_comparison.{ext}",
                    format=ext, bbox_inches="tight")
    plt.close()
    print(f"figure -> {IMAGES_DIR / 'clp_snn_hw_comparison.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["1shot", "25shot", "both"],
                    default=None, help="omit with --plot to only re-plot")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=VARIANTS)
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    ap.add_argument("--n-protos", type=int, default=N_PROTOS,
                    help=f"prototype pool size (default {N_PROTOS})")
    ap.add_argument("--rerun", action="store_true",
                    help="recompute configs already in summary.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if args.setting:
        settings = ["1shot", "25shot"] if args.setting == "both" \
            else [args.setting]
        for s in settings:
            run_setting(s, args.seeds, args.variants, args.thresholds,
                        pool=args.n_protos, rerun=args.rerun)

    rows = load_summary()
    for s in ("1shot", "25shot"):
        write_table(rows, s)
    plot(rows)


if __name__ == "__main__":
    main()
