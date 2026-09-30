"""
clp_snn_gap_decomposition.py
----------------------------
Where the accuracy gap between CLP and CLP-SNN on Loihi 2 comes from.

Uses the Loihi 2 deployment emulator (models/CLP_SNN_Loihi.py), whose
constants all follow from the deployed code, and switches the chip's features
on one at a time, starting from CLP:

  CLP                     CLP as in Table 1 (theta 0.75)
  alloc-only INT8         prototypes imprinted once; allocation on novelty or
                          error; 7-bit inputs; CLP's threshold 0.75; argmax
  + chip preprocessing    negative features clipped before normalising
  + chip threshold        cosine 0.708 / 0.507
  + spike-timing winner   winner resolved in time steps, label vote on ties
  + chip test subset      25-shot only: the chip script drew the test frames
                          with the run seed instead of seed 42
  Loihi 2                 measured (data/loihi2/)

Spike-timing levels average several tie-breaking seeds. The last emulator
level is also compared with the chip's accuracy curves (validation).

--sweep adds the accuracy / prototype-count trade-off of the full emulator
over the similarity threshold.

Outputs in experiments/results/clp_snn_gap_decomposition/:
  ladder.csv, validation.csv, sweep.csv

Usage:
  python experiments/clp_snn_gap_decomposition.py
  python experiments/clp_snn_gap_decomposition.py --sweep
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
for p in (str(_REPO), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import clp_vs_baselines_1shot as b1  # noqa: E402
import clp_vs_baselines_25shot as b25  # noqa: E402
from models.CLP import ContinuallyLearningPrototypes  # noqa: E402
from models.CLP_SNN_Loihi import CLPSNNLoihi  # noqa: E402
from models.clp_snn_configs import LOIHI_CONFIGS  # noqa: E402

RESULTS_DIR = _REPO / "experiments" / "results" / "clp_snn_gap_decomposition"
LOIHI_DIR = _REPO / "data" / "loihi2"
SEEDS = [10, 20, 30]
TIE_REPS = 3                      # tie-breaking seeds for spike-timing levels
BIG_POOL = 16000                  # pool for the levels before the chip's
SWEEP_TH = {"1shot": [0.60, 0.65, 0.708, 0.75, 0.80, 0.85],
            "25shot": [0.50, 0.55, 0.60, 0.65, 0.708, 0.75, 0.80, 0.85]}


def levels(setting):
    """(name, emulator options, test subset, repetitions) per ladder level."""
    L = [("alloc-only INT8, theta 0.75",
          dict(preprocess="sim", winner="argmax", threshold=0.75,
               n_protos=BIG_POOL), "sim", 1),
         ("+ chip preprocessing",
          dict(preprocess="chip", winner="argmax", threshold=0.75,
               n_protos=BIG_POOL), "sim", 1),
         ("+ chip threshold",
          dict(preprocess="chip", winner="argmax", n_protos=BIG_POOL), "sim", 1),
         ("+ spike-timing winner",
          dict(preprocess="chip", winner="spike"), "sim", TIE_REPS)]
    if setting == "25shot":
        L.append(("+ chip test subset",
                  dict(preprocess="chip", winner="spike"), "chip", TIE_REPS))
    return L


# ── Data ──────────────────────────────────────────────────────────────────────

def test_set(setting, subset, seed):
    """Test features/labels: seed-42 subset (all methods) or the chip's."""
    if subset == "sim":
        X_raw, _, y = (b1 if setting == "1shot" else b25).load_test_set()
        return X_raw, y
    np.random.seed(seed)              # as clp_25_shot_openloris.py on chip
    X, y = np.load(b25.TEST_X_PATH), np.load(b25.TEST_Y_PATH)
    idx = []
    for c in np.unique(y):
        idx.extend(np.random.choice(np.where(y == c)[0], b25.N_FRAMES, replace=False))
    return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx]).long()


def accuracy(clf, X, y):
    return (clf.predict(X).argmax(1) == y).float().mean().item() * 100


def run_stream(clf, setting, seed, subset):
    """Checkpoint accuracy curve on the published cadence."""
    X_te, y_te = test_set(setting, subset, seed)
    X_te = X_te / X_te.norm(dim=1, keepdim=True)
    curve = []
    if setting == "1shot":
        _, X_tr, y_tr, bounds, order = b1.load_train_split(seed)
        for i, (x, y) in enumerate(zip(X_tr, y_tr), start=1):
            clf.fit(x, y, i - 1)
            if i in bounds and len(curve) < b1.N_STEPS:
                m = torch.isin(y_te, torch.tensor(order[: len(curve) + 1]))
                curve.append(accuracy(clf, X_te[m], y_te[m]))
    else:
        X_tr, _, y_tr, cps = b25.load_train_split(seed)
        cps = set(int(c) for c in cps)
        for i, (x, y) in enumerate(zip(X_tr, y_tr), start=1):
            clf.fit(x, y, i - 1)
            if i in cps:
                curve.append(accuracy(clf, X_te, y_te))
    return np.array(curve)


def build_clp(setting):
    """CLP as configured for Table 1."""
    return ContinuallyLearningPrototypes(
        b1.FEATURE_SIZE, n_protos=400 if setting == "1shot" else 2600,
        num_classes=b1.NUM_CLASSES, backbone=None, alpha_init=1,
        sim_th_init=0.75, n_wta=1, k_hit=1, k_miss=1, adaptive_th=False,
        learn_outliers=(setting == "1shot"), device="cpu")


# ── Jobs ──────────────────────────────────────────────────────────────────────

def job(args):
    kind, setting, name, opts, subset, seed, rep = args
    torch.set_num_threads(1)
    if kind == "clp":
        clf = build_clp(setting)
    else:
        cfg = dict(LOIHI_CONFIGS[f"paper_{setting}"])
        cfg.update(opts)
        clf = CLPSNNLoihi(b1.FEATURE_SIZE, b1.NUM_CLASSES, seed=rep, **cfg)
    curve = run_stream(clf, setting, seed, subset)
    n = None if kind == "clp" else clf.get_num_prototypes_used()
    return args, curve, n


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep the threshold of the full emulator")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for st in ("1shot", "25shot"):
        jobs += [("clp", st, "CLP", {}, "sim", s, 0) for s in SEEDS]
        for name, opts, subset, reps in levels(st):
            jobs += [("emu", st, name, opts, subset, s, r)
                     for s in SEEDS for r in range(reps)]
        if args.sweep:
            jobs += [("emu", st, f"sweep {th}", dict(threshold=th, n_protos=BIG_POOL),
                      "sim", s, 0) for th in SWEEP_TH[st] for s in SEEDS]

    res = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for a, curve, n in ex.map(job, jobs):
            res.setdefault((a[1], a[2]), {}).setdefault(a[5], []).append((curve, n))
            print(f"  {a[1]:6s} {a[2]:30s} seed {a[5]} rep {a[6]}: {curve[-1]:6.2f}", flush=True)

    def summary(st, name):
        """Per-seed final, per-seed AAA and mean curve (averaged over reps)."""
        per = res[(st, name)]
        curves = np.array([np.mean([c for c, _ in per[s]], 0) for s in SEEDS])
        n = [np.mean([k for _, k in per[s]]) if per[s][0][1] is not None else np.nan
             for s in SEEDS]
        return curves[:, -1], curves.mean(1), curves, np.array(n)

    ladder, valid = [], []
    for st in ("1shot", "25shot"):
        chip = np.load(LOIHI_DIR / f"accuracies_clp_loihi_{st}.npy")[:, 0, :].T * 100
        names = ["CLP"] + [n for n, *_ in levels(st)]
        prev = None
        print(f"\n== {st}: final accuracy (%), mean over seeds; step = change from row above")
        for name in names:
            fin, aaa, _, n = summary(st, name)
            step = fin - prev if prev is not None else np.full(3, np.nan)
            print(f"  {name:30s} {fin.mean():6.2f}  step {np.nanmean(step):+6.2f}"
                  f"  per-seed {np.round(step, 2)}  protos {np.nanmean(n):7.0f}")
            ladder.append(dict(setting=st, level=name, final=round(fin.mean(), 2),
                               step=round(float(np.nanmean(step)), 2) if prev is not None else "",
                               aaa=round(aaa.mean(), 2), protos=round(float(np.nanmean(n)))
                               if not np.isnan(np.nanmean(n)) else ""))
            prev = fin
        step = chip[:, -1] - prev
        print(f"  {'Loihi 2 (measured)':30s} {chip[:, -1].mean():6.2f}  step {step.mean():+6.2f}"
              f"  per-seed {np.round(step, 2)}")
        ladder.append(dict(setting=st, level="Loihi 2 (measured)", final=round(chip[:, -1].mean(), 2),
                           step=round(step.mean(), 2), aaa=round(chip.mean(), 2), protos=""))

        # Validation: full emulator vs chip, on the chip's test protocol
        fin, aaa, curves, n = summary(st, names[-1])
        rmse = np.sqrt(((curves - chip) ** 2).mean())
        print(f"  validation: emulator final {fin.mean():.2f} vs chip {chip[:, -1].mean():.2f}; "
              f"AAA {aaa.mean():.2f} vs {chip.mean():.2f}; curve RMSE {rmse:.2f} pp")
        valid.append(dict(setting=st, emulator_final=round(fin.mean(), 2),
                          chip_final=round(chip[:, -1].mean(), 2),
                          emulator_aaa=round(aaa.mean(), 2), chip_aaa=round(chip.mean(), 2),
                          curve_rmse=round(rmse, 2)))

    write_csv(RESULTS_DIR / "ladder.csv", ladder, ["setting", "level", "final", "step", "aaa", "protos"])
    write_csv(RESULTS_DIR / "validation.csv", valid, list(valid[0]))

    if args.sweep:
        rows = []
        print("\n== threshold sweep (full emulator)")
        for st in ("1shot", "25shot"):
            for th in SWEEP_TH[st]:
                fin, aaa, _, n = summary(st, f"sweep {th}")
                print(f"  {st:6s} cosine {th:.3f}: final {fin.mean():6.2f}  AAA {aaa.mean():6.2f}"
                      f"  protos {n.mean():7.0f}")
                rows.append(dict(setting=st, threshold=th, final=round(fin.mean(), 2),
                                 aaa=round(aaa.mean(), 2), protos=round(n.mean())))
        write_csv(RESULTS_DIR / "sweep.csv", rows, list(rows[0]))


if __name__ == "__main__":
    main()
