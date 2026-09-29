"""
run_ablation.py
---------------
Runs named CLPSNNAdaptive configurations on the 1-shot / 25-shot OpenLORIS
streams with the protocol of clp_snn_hw_comparison.py (same loaders, seeds,
test set and checkpoint cadence; its stream runners are imported).

One JSON per run in experiments/results/adaptive_protos/runs/ (resumable,
safe for parallel workers); `--table` aggregates them next to the cached
CLP / NA rows of clp_snn_hw_comparison/summary.csv.

Usage:
  python experiments/adaptive_protos/run_ablation.py --setting both \
      --paths FP32 --configs base H1a H1b H1ab --seeds 10 --thresholds 0.7 0.85
  python experiments/adaptive_protos/run_ablation.py --table
"""

import argparse
import csv
import itertools
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_REPO = _EXP.parent
for p in (str(_REPO), str(_EXP), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import clp_snn_hw_comparison as hw  # noqa: E402
from variants import CLPSNNAdaptive  # noqa: E402

OUT = _REPO / "experiments" / "results" / "adaptive_protos"
RUNS = OUT / "runs"
POOL = 16000
GLOBAL_SEED = 42

# H1ab = intended schedule: alloc alpha 1, then 1/2, 1/3, ... (steps of +-1)
_H1AB = dict(alpha_after_inc=True, g_inc_pos=1.0, g_inc_neg=1.0)
CONFIGS = {
    "base":      {},
    "H1a":       dict(alpha_after_inc=True),
    "H1b":       dict(g_inc_pos=1.0, g_inc_neg=1.0),
    "H1ab":      _H1AB,
    "H1ab+H2a":  {**_H1AB, "g_floor": 1.0},
    "H1ab+H2b":  {**_H1AB, "g_inc_neg": 0.0},
    "H1ab+H2c1": {**_H1AB, "neg_alpha_shift": 1},
    "H1ab+H2c2": {**_H1AB, "neg_alpha_shift": 2},
    "H1ab+H2d":  {**_H1AB, "oracle_renorm": True},
    "H1ab+H3":   {**_H1AB, "alloc_on_miss": True},
    # added after step 1 diagnosis (unit sphere repelling under r < 0)
    "H1ab+H2e":  {**_H1AB, "neg_update": "none"},
    "H1ab+H2f":  {**_H1AB, "neg_update": "decay"},
    "H1ab+H2e+H3": {**_H1AB, "neg_update": "none", "alloc_on_miss": True},
    "H1ab+H2f+H3": {**_H1AB, "neg_update": "decay", "alloc_on_miss": True},
    # combined fix chosen in step 3, and the H4 knobs on top of it
    "fix":       {**_H1AB, "neg_update": "decay", "alloc_on_miss": True},
    "fix+H4a":   {**_H1AB, "neg_update": "decay", "alloc_on_miss": True,
                  "alpha_int_max": 7},
    "fix+H4b1":  {**_H1AB, "neg_update": "decay", "alloc_on_miss": True,
                  "w_scale_shift": 1},
    "fix+H4b2":  {**_H1AB, "neg_update": "decay", "alloc_on_miss": True,
                  "w_scale_shift": 2},
    # post-hoc (user question): final configs with the released g_inc = 0.5
    "fix-g05":   dict(alpha_after_inc=True, neg_update="decay",
                      alloc_on_miss=True),
    "fix+H4b2-g05": dict(alpha_after_inc=True, neg_update="decay",
                         alloc_on_miss=True, w_scale_shift=2),
    # single-factor H2 on the released schedule, for the collapse diagnosis
    "H2a":       dict(g_floor=1.0),
    "H2b":       dict(g_inc_neg=0.0),
    "H2c1":      dict(neg_alpha_shift=1),
    "H2d":       dict(oracle_renorm=True),
}


def register(name, kw):
    CONFIGS[name] = kw


def run_path(setting, path, config, th, seed):
    return RUNS / f"{setting}_{path}_{config}_th{th:.2f}_s{seed}.json"


def build(path, config, th):
    return CLPSNNAdaptive(
        hw.b1.FEATURE_SIZE, n_protos=POOL, num_classes=hw.b1.NUM_CLASSES,
        threshold=th, g_inc=0.5, use_quantization=(path == "INT8"),
        adaptive_protos=True, device="cpu", **CONFIGS[config])


def top_share(clf, X_test):
    """Fraction of test samples whose best-matching prototype is the single
    most frequent one (1/n_protos-ish when healthy, -> 1 when collapsed)."""
    n = clf.allocator.get_allocated_count()
    W = clf.prototypes[:n]
    Xq = clf._quantize_input(X_test) if clf.use_quantization else X_test
    best = (Xq @ W.T).argmax(dim=1)
    return torch.bincount(best).max().item() / len(best)


def one_run(job):
    setting, path, config, th, seed = job
    torch.set_num_threads(1)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    test = (hw.b1.load_test_set() if setting == "1shot"
            else hw.b25.load_test_set())
    clf = build(path, config, th)
    t0 = time.perf_counter()
    runner = hw.run_1shot if setting == "1shot" else hw.run_25shot
    curve = runner(clf, seed, test)
    n = clf.allocator.get_allocated_count()
    norms = clf.prototypes[:n].norm(dim=1)
    res = dict(setting=setting, path=path, config=config, threshold=th,
               seed=seed, final_acc=float(curve[-1]), aaa=float(curve.mean()),
               n_protos=int(n), n_miss_alloc=int(clf.n_miss_alloc),
               max_norm=float(norms.max()), mean_norm=float(norms.mean()),
               top_share=float(top_share(clf, test[1])),
               seconds=time.perf_counter() - t0, curve=curve.tolist())
    run_path(setting, path, config, th, seed).write_text(json.dumps(res))
    return res


def launch(jobs, workers, rerun=False):
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if rerun or not run_path(*j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} cached, "
          f"running {len(todo)} on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one_run, todo):
            print(f"{r['setting']} {r['path']} {r['config']:10s} "
                  f"th={r['threshold']:.2f} s{r['seed']}: "
                  f"final={r['final_acc']*100:5.1f}  protos={r['n_protos']:5d}  "
                  f"max|w|={r['max_norm']:.2f}  top={r['top_share']:.2f}  "
                  f"({r['seconds']:.0f}s)", flush=True)


# ── Aggregation ──────────────────────────────────────────────────────────────

def load_runs():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def load_cached_baselines():
    """{(setting, variant, th): [(final, protos), ...]} from the hw sweep."""
    out = {}
    with open(hw.SUMMARY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] in ("CLP", "FP32-NA", "INT8-NA"):
                out.setdefault((r["setting"], r["variant"], float(r["threshold"])),
                               {})[int(r["seed"])] = (float(r["final_acc"]),
                                                      int(r["n_protos_alloc"]))
    return out


def table(seeds=None, out_csv=None):
    runs = load_runs()
    base = load_cached_baselines()
    groups = {}
    for r in runs:
        if seeds and r["seed"] not in seeds:
            continue
        groups.setdefault((r["setting"], r["path"], r["config"],
                           r["threshold"]), []).append(r)
    rows = []
    for (setting, path, config, th), rs in sorted(groups.items()):
        s = sorted(x["seed"] for x in rs)
        acc = np.array([x["final_acc"] for x in rs]) * 100
        clp = base.get((setting, "CLP", th), {})
        na = base.get((setting, f"{path}-NA", th), {})
        clp_m = np.mean([clp[k][0] for k in s if k in clp]) * 100 if clp else np.nan
        na_m = np.mean([na[k][0] for k in s if k in na]) * 100 if na else np.nan
        rows.append(dict(
            setting=setting, path=path, config=config, th=f"{th:.2f}",
            seeds=",".join(map(str, s)),
            final=f"{acc.mean():.2f}", std=f"{acc.std():.2f}",
            min=f"{acc.min():.2f}",
            d_clp=f"{acc.mean() - clp_m:+.2f}", d_na=f"{acc.mean() - na_m:+.2f}",
            protos=f"{np.mean([x['n_protos'] for x in rs]):.0f}",
            clp_protos=(f"{np.mean([clp[k][1] for k in s if k in clp]):.0f}"
                        if clp else ""),
            max_norm=f"{max(x['max_norm'] for x in rs):.2f}",
            top=f"{max(x['top_share'] for x in rs):.2f}"))
    if not rows:
        print("no runs")
        return
    keys = list(rows[0])
    w = [max(len(k), *(len(r[k]) for r in rows)) for k in keys]
    print("  ".join(k.ljust(n) for k, n in zip(keys, w)))
    for r in rows:
        print("  ".join(r[k].ljust(n) for k, n in zip(keys, w)))
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=keys)
            dw.writeheader()
            dw.writerows(rows)
        print(f"table -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["1shot", "25shot", "both"])
    ap.add_argument("--paths", nargs="+", default=["FP32"],
                    choices=["FP32", "INT8"])
    ap.add_argument("--configs", nargs="+", default=["base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[10])
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.70, 0.85])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--table-seeds", type=int, nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()

    if args.setting:
        unknown = [c for c in args.configs if c not in CONFIGS]
        if unknown:
            sys.exit(f"unknown configs: {unknown}")
        settings = ["1shot", "25shot"] if args.setting == "both" \
            else [args.setting]
        jobs = list(itertools.product(settings, args.paths, args.configs,
                                      args.thresholds, args.seeds))
        launch(jobs, args.workers, args.rerun)
    if args.table or not args.setting:
        table(args.table_seeds, args.csv)


if __name__ == "__main__":
    main()
