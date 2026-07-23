"""
FP16 precision gate for the naive StreamingLDA benchmark rows.

Concern: with use_fp16=True the covariance recursion
    Sigma <- (n*Sigma + delta) / (n+1)
accumulates in FP16. The per-step relative increment scales as 1/n, and FP16
resolves relative changes only down to ~2^-11, so Sigma updates begin to be
rounded away once n approaches ~2048. On the 2400-sample 1-shot stream this is
borderline; at 25-shot lengths it is structural.

Gate: run FP32 and FP16 StreamingLDA (periodic Lambda, rho=60) side by side
over the real seed-10 1-shot stream. At every rho fits, compare
  - argmax agreement on the full test set (FP16 vs FP32 reference)
  - relative Sigma deviation ||Sigma16 - Sigma32|| / ||Sigma32||
  - fraction of Sigma entries whose FP16 update was rounded to zero while the
    FP32 update was nonzero (the stall diagnostic)

PASS criterion (matches the E-A substitution bar): argmax agreement >= 99.9%
at every checkpoint. If the gate fails, either keep Sigma/Lambda state in FP32
with FP16 used only for the score GEMM, or drop the FP16 SLDA row.

Usage:  python experiments/slda_fp16_gate.py [--rho 60] [--device cpu]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from models.SLDA import StreamingLDA

DATA = _REPO / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()

    X = torch.load(DATA / "1shot" / "X_train_1_shot_10.pt",
                   map_location="cpu", weights_only=False).float()
    y = torch.load(DATA / "1shot" / "y_train_1_shot_10.pt",
                   map_location="cpu", weights_only=False).long()
    X = X / X.norm(dim=1, keepdim=True)
    X_test = torch.from_numpy(np.load(DATA / "X_test.npy")).float()
    X_test = X_test / X_test.norm(dim=1, keepdim=True)

    d = X.shape[1]
    kwargs = dict(shrinkage_param=1e-4, streaming_update_sigma=True,
                  streaming_update_lambda=True, lambda_update_period=args.rho,
                  device=args.device)
    m32 = StreamingLDA(d, 40, **kwargs, use_fp16=False)
    m16 = StreamingLDA(d, 40, **kwargs, use_fp16=True)

    print(f"FP16 gate: rho={args.rho}, {X.shape[0]} stream samples, "
          f"{X_test.shape[0]} test samples, device={args.device}")
    print(f"{'n':>6} {'agree':>8} {'relSigma':>10} {'stalled%':>9}")

    min_agree = 1.0
    for i in range(X.shape[0]):
        is_ckpt = (i + 1) % args.rho == 0 or i == X.shape[0] - 1
        if is_ckpt:
            s32_prev = m32.Sigma.clone()
            s16_prev = m16.Sigma.clone().float()

        m32.fit(X[i], y[i].view(1), i)
        m16.fit(X[i], y[i].view(1), i)

        if is_ckpt:
            # stall diagnostic on this step's update
            d32 = (m32.Sigma - s32_prev).abs()
            d16 = (m16.Sigma.float() - s16_prev).abs()
            stalled = ((d16 == 0) & (d32 > 0)).float().mean().item()

            rel_sigma = ((m16.Sigma.float() - m32.Sigma).norm()
                         / m32.Sigma.norm()).item()

            p32 = m32.predict(X_test).argmax(dim=1)
            p16 = m16.predict(X_test).argmax(dim=1)
            agree = (p32 == p16).float().mean().item()
            min_agree = min(min_agree, agree)
            print(f"{i+1:>6} {agree:>8.4f} {rel_sigma:>10.2e} {stalled*100:>8.2f}%")

    verdict = "PASS" if min_agree >= args.threshold else "FAIL"
    print(f"\nMinimum argmax agreement: {min_agree:.4f} "
          f"(threshold {args.threshold}) -> {verdict}")
    if verdict == "FAIL":
        print("Recommendation: keep Sigma/Lambda state FP32 and restrict FP16 "
              "to the score GEMM, or drop the FP16 SLDA rows from Table 1/SI.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
