"""
check_fixed.py
--------------
Verifies the corrected models.CLP_SNN.CLPSNN (reward on the Hebbian term
only, goodness updated before alpha, goodness floored at 1):

1. bit-exact with the tested research variant
   CLPSNNAdaptive(alpha_after_inc=True, g_floor=1.0, neg_update="decay")
   on full 1-shot and 25-shot streams, FP32 and INT8, including misses;
2. a run of incorrect predictions from ||w|| = 1 shrinks the norm
   (the old rule grew it).

Usage: python experiments/adaptive_protos/check_fixed.py
"""

import sys
from pathlib import Path

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

KW = dict(alpha_after_inc=True, g_floor=1.0, neg_update="decay")


def run(clf, X, y):
    for i, (xi, yi) in enumerate(zip(X, y)):
        clf.fit(xi.view(b1.FEATURE_SIZE), yi.view(1), i)
    return clf


def make(cls, quant, th, **kw):
    return cls(b1.FEATURE_SIZE, n_protos=16000, num_classes=b1.NUM_CLASSES,
               threshold=th, g_inc=0.5, use_quantization=quant,
               adaptive_protos=True, device="cpu", **kw)


def main():
    torch.set_num_threads(4)
    _, X1, y1, _, _ = b1.load_train_split(10)
    X25, _, y25, _ = b25.load_train_split(20)
    ok = True
    for name, X, y, ths in (("1shot s10", X1, y1, (0.70, 0.90)),
                            ("25shot s20", X25, y25, (0.70, 0.85))):
        for quant in (False, True):
            for th in ths:
                a = run(make(CLPSNN, quant, th), X, y)
                trace = []
                b = run(make(CLPSNNAdaptive, quant, th, trace=trace, **KW), X, y)
                same = all(torch.equal(getattr(a, k), getattr(b, k)) for k in
                           ("prototypes", "proto_labels", "goodness", "alphas"))
                misses = sum(1 for t in trace if t[2] < 0)
                ok &= same
                print(f"{name:10s} {'INT8' if quant else 'FP32'} th={th:.2f} "
                      f"misses={misses:5d} bit-exact={same}", flush=True)

    # Miss streak on one prototype that keeps winning: each input has
    # cosine 0.8 with the current prototype (a winner always has y > 0).
    # Start slightly long (||w|| = 1.05, as after an early large-alpha
    # step) with goodness 3; compare with the released rule r*(x - y*w).
    for quant in (False, True):
        clf = make(CLPSNN, quant, 0.0)
        g = torch.Generator().manual_seed(0)
        w0 = torch.randn(b1.FEATURE_SIZE, generator=g).abs()
        w0 = 1.05 * w0 / w0.norm()
        clf._allocate_prototype(w0, 0)
        clf.goodness[0] = 3.0
        old_w, old_g = w0.clone(), 3.0
        norms, old_norms = [], []
        for _ in range(8):
            w = clf.prototypes[0]
            u = w / w.norm()
            v = torch.randn(b1.FEATURE_SIZE, generator=g)
            v -= (v @ u) * u
            v /= v.norm()
            xv = 0.8 * u + 0.6 * v
            xv = clf._quantize_input(xv) if quant else xv
            clf._update_winner(0, xv, r=-1.0)
            norms.append(clf.prototypes[0].norm().item())
            # released rule, float, same inputs, old goodness bookkeeping
            a = 1.0 / max(old_g, 1.0)
            y = (old_w @ xv).item()
            old_w = old_w - a * (xv - y * old_w)
            old_g -= 0.5
            old_norms.append(old_w.norm().item())
        shrinks = norms[-1] < 1.0
        ok &= shrinks
        print(f"miss streak {'INT8' if quant else 'FP32'}: new ||w|| "
              + " ".join(f"{n:.3f}" for n in norms) + f"  shrinks={shrinks}")
        print(f"                 old ||w|| "
              + " ".join(f"{n:.3f}" for n in old_norms))
    print("ALL CHECKS PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
