"""
slda_drift_study.py
-------------------
Numerical drift study for the FP32 maintained precision of RankOneSLDA:
drift of the Sherman-Morrison-maintained Lambda over the ~60k-sample 25-shot
stream, measured against an FP64 from-scratch inv(S + lambda*I) reference at
periodic checkpoints.

Decision rule: if the max score deviation is material, hold Lambda in FP64
(13 MB, arithmetic still trivial) or maintain a Cholesky factor with periodic
rebuilds; otherwise FP32 needs no insurance. This script produces the numbers
that justify the choice.

Metrics at each checkpoint (default every 5000 samples + final):
  * relative Lambda deviation: max|Lambda_fp32 - Lambda_ref| / max|Lambda_ref|
  * relative score deviation on the balanced test set
  * argmax agreement vs the reference
  * test accuracy of the FP32 model and the reference

Outputs saved to experiments/results/slda_drift_study/:
  drift_seed{seed}.csv  - per-checkpoint metrics
  drift_seed{seed}.npz  - same, as arrays

Usage (from repo root or experiments/):
  python experiments/slda_drift_study.py [--seed 10] [--ridge 1.0]
                                         [--checkpoint_every 5000]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make repo root importable when running from experiments/
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.SLDA import RankOneSLDA  # noqa: E402
from experiments.clp_vs_baselines_25shot import (  # noqa: E402
    reorder_by_instance_rounds,
)

DATA_DIR = _REPO / "data"
RESULTS_DIR = _REPO / "experiments" / "results" / "slda_drift_study"

NUM_CLASSES = 40
FEATURE_SIZE = 1280
N_FRAMES = 60
GLOBAL_SEED = 42


# ── Data (mirrors clp_vs_baselines_25shot.py) ─────────────────────────────────

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


def load_train_stream(seed: int):
    X = np.load(DATA_DIR / "25shot" / f"X_train_25_shot_{seed}.npy")
    y = np.load(DATA_DIR / "25shot" / f"y_train_25_shot_{seed}.npy")
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_r, y_r, _ = reorder_by_instance_rounds(
        X_norm, y, fixed_class_order=False, seed=seed, max_size=N_FRAMES
    )
    return torch.from_numpy(X_r).float(), torch.from_numpy(y_r).long()


# ── FP64 reference (Hayes' scatter recursion, shadow-accumulated) ─────────────

def shadow_update(S, muK, cK, n, x, y):
    v = x - muK[y]
    c = n / (n + 1)
    if c > 0:
        S += c * torch.outer(v, v)
    muK[y] += v / (cK[y] + 1)
    cK[y] += 1
    return n + 1


def ref_scores(S, muK, cK, X, lam):
    Lam = torch.linalg.inv(
        S + lam * torch.eye(S.shape[0], dtype=S.dtype))
    W = Lam @ muK.t()
    b = 0.5 * torch.sum(muK.t() * W, dim=0)
    scores = X.to(S.dtype) @ W - b
    not_visited_ix = torch.where(cK == 0)[0]
    min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
    scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
        len(not_visited_ix), len(X)).transpose(1, 0)
    return Lam, scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FP32 drift study of RankOneSLDA over the 25-shot stream")
    parser.add_argument("--seed", type=int, default=10, choices=[10, 20, 30])
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=4,
                        help="torch CPU threads; small values are often "
                             "faster for these per-sample ops and run cooler")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional cap on stream length (partial run)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading 25-shot stream (seed {args.seed})...")
    X_tr, y_tr = load_train_stream(args.seed)
    X_probe, y_probe = load_test_set()
    n_total = len(y_tr)
    if args.max_steps is not None:
        n_total = min(n_total, args.max_steps)
    print(f"  train: {n_total} samples; probe: {len(y_probe)} samples; "
          f"{args.threads} torch threads")

    model = RankOneSLDA(FEATURE_SIZE, NUM_CLASSES, ridge_param=args.ridge,
                        device="cpu")

    S = torch.zeros((FEATURE_SIZE, FEATURE_SIZE), dtype=torch.float64)
    muK = torch.zeros((NUM_CLASSES, FEATURE_SIZE), dtype=torch.float64)
    cK = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    n = 0

    # Shadow scatter is accumulated in blocks: buffer sqrt(c)*v rows and flush
    # S += B^T @ B every SHADOW_BLOCK steps (one BLAS-3 call instead of
    # per-step 13 MB outer-product writes; ~10x cheaper, same FP64 result).
    SHADOW_BLOCK = 512
    buf = []

    def flush_shadow():
        if buf:
            B = torch.stack(buf)
            S.add_(B.t() @ B)
            buf.clear()

    t0 = time.perf_counter()
    rows = []
    for i in range(n_total):
        model.fit(X_tr[i], y_tr[i].view(1), i)

        # shadow recursion, means per-step, scatter deferred to blocks
        x64 = X_tr[i].double()
        y = int(y_tr[i])
        v = x64 - muK[y]
        c = n / (n + 1)
        if c > 0:
            buf.append(v * (c ** 0.5))
        muK[y] += v / (cK[y] + 1)
        cK[y] += 1
        n += 1
        if len(buf) >= SHADOW_BLOCK:
            flush_shadow()

        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"    ... step {i + 1}/{n_total} "
                  f"({rate:.0f} samples/s)", flush=True)

        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == n_total:
            flush_shadow()
            Lam_ref, scores_ref = ref_scores(S, muK, cK, X_probe, args.ridge)
            scores_m = model.predict(X_probe).double()

            lam_scale = Lam_ref.abs().max().item()
            score_scale = scores_ref.abs().max().item()
            pred_m = scores_m.argmax(dim=1)
            pred_ref = scores_ref.argmax(dim=1)
            row = {
                "step": i + 1,
                "rel_lambda_dev":
                    (model.Lambda.double() - Lam_ref).abs().max().item()
                    / lam_scale,
                "rel_score_dev":
                    (scores_m - scores_ref).abs().max().item() / score_scale,
                "argmax_agreement":
                    (pred_m == pred_ref).float().mean().item(),
                "acc_fp32": (pred_m == y_probe).float().mean().item(),
                "acc_ref": (pred_ref == y_probe).float().mean().item(),
                "min_denom": model.min_denom,
            }
            rows.append(row)
            print(f"  step {row['step']:6d}  "
                  f"rel_Lambda_dev {row['rel_lambda_dev']:.3e}  "
                  f"rel_score_dev {row['rel_score_dev']:.3e}  "
                  f"agree {row['argmax_agreement']:.5f}  "
                  f"acc_fp32 {row['acc_fp32']:.4f}  "
                  f"acc_ref {row['acc_ref']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"drift_seed{args.seed}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(
        RESULTS_DIR / f"drift_seed{args.seed}.npz",
        **{k: np.array([r[k] for r in rows]) for k in rows[0].keys()},
        ridge=args.ridge,
    )

    # ── Summary and decision input ────────────────────────────────────────────
    max_score_dev = max(r["rel_score_dev"] for r in rows)
    min_agree = min(r["argmax_agreement"] for r in rows)
    max_acc_gap = max(abs(r["acc_fp32"] - r["acc_ref"]) for r in rows)
    print("\nSummary over the full stream:")
    print(f"  max relative score deviation : {max_score_dev:.3e}")
    print(f"  min argmax agreement         : {min_agree:.5f}")
    print(f"  max |acc_fp32 - acc_ref|     : {max_acc_gap:.5f}")
    print(f"  min SM denominator           : {model.min_denom:.6f}")
    if max_score_dev < 1e-3 and min_agree >= 0.999:
        print("\nDecision input: FP32 drift is immaterial over the 25-shot "
              "stream; no drift insurance (FP64 Lambda or Cholesky refresh) "
              "needed.")
    else:
        print("\nDecision input: FP32 drift is material; hold Lambda in FP64 "
              "or maintain a Cholesky factor with periodic rebuilds.")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
