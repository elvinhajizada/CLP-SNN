"""
check_default.py
----------------
Step 0 of the study, kept as a record: CLPSNNAdaptive with default knobs
reproduces the PRE-FIX CLPSNN (models/CLP_SNN.py at commit 3a5e9c7) bit for
bit. Against the current (fixed) model it reports MISMATCH by design; the
current check is check_fixed.py. Also reports wall time per sample.

Usage: python experiments/adaptive_protos/check_default.py [--n 2400]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
for p in (str(_EXP.parent), str(_EXP), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import clp_vs_baselines_1shot as b1  # noqa: E402
import clp_vs_baselines_25shot as b25  # noqa: E402
from models.CLP_SNN import CLPSNN  # noqa: E402
from variants import CLPSNNAdaptive  # noqa: E402


def run(cls, quant, X, y, th):
    clf = cls(b1.FEATURE_SIZE, n_protos=4000, num_classes=b1.NUM_CLASSES,
              threshold=th, g_inc=0.5, use_quantization=quant,
              adaptive_protos=True, device="cpu")
    t0 = time.perf_counter()
    for i, (xi, yi) in enumerate(zip(X, y)):
        clf.fit(xi.view(b1.FEATURE_SIZE), yi.view(1), i)
    return clf, (time.perf_counter() - t0) / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000,
                    help="25-shot samples to use (1-shot always full stream)")
    args = ap.parse_args()

    _, X1, y1, _, _ = b1.load_train_split(10)
    X25, _, y25, _ = b25.load_train_split(20)
    streams = [("1shot s10", X1, y1), (f"25shot s20[:{args.n}]",
                                       X25[:args.n], y25[:args.n])]
    ok = True
    for name, X, y in streams:
        for quant in (False, True):
            for th in (0.70, 0.85):
                a, ta = run(CLPSNN, quant, X, y, th)
                b, tb = run(CLPSNNAdaptive, quant, X, y, th)
                same = all(torch.equal(getattr(a, k), getattr(b, k)) for k in
                           ("prototypes", "proto_labels", "goodness", "alphas"))
                n = a.allocator.get_allocated_count()
                ok &= same
                print(f"{name:18s} {'INT8' if quant else 'FP32'} th={th:.2f} "
                      f"protos={n:5d} bit-exact={same}  "
                      f"{ta*1e3:.2f} / {tb*1e3:.2f} ms per sample")
    print("ALL BIT-EXACT" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
