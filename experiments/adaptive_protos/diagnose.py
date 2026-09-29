"""
diagnose.py
-----------
Step 1: instrumented single runs for H1/H2 (and H4 with --shadow).

For each case it records every learning update (winner, r, alpha used,
goodness before, norm before/after) and reports:
  H1  alpha used on each prototype's first hit; final cos(w, mean of the
      samples it absorbed) and ||w|| over allocated prototypes
  H2  the prototype whose norm grew first past 1.5: its event history up to
      and just past that point, and when accuracy collapsed
  H4  (--shadow, INT8) cos(w_int, w_float_shadow) and share of updates that
      left the integer weights unchanged

Saves the trace as .npz in results/adaptive_protos/diag/ plus a norm plot.

Usage:
  python experiments/adaptive_protos/diagnose.py 25shot FP32 base 0.80 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent.parent), str(_HERE.parent), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_ablation as ra  # noqa: E402
from variants import CLPSNNAdaptive  # noqa: E402

DIAG = ra.OUT / "diag"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting", choices=["1shot", "25shot"])
    ap.add_argument("path", choices=["FP32", "INT8"])
    ap.add_argument("config")
    ap.add_argument("threshold", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--history", type=int, default=25)
    args = ap.parse_args()
    DIAG.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    hw = ra.hw
    trace = []
    clf = CLPSNNAdaptive(
        hw.b1.FEATURE_SIZE, n_protos=ra.POOL, num_classes=hw.b1.NUM_CLASSES,
        threshold=args.threshold, g_inc=0.5,
        use_quantization=(args.path == "INT8"), adaptive_protos=True,
        device="cpu", trace=trace, track_means=True, shadow=args.shadow,
        **ra.CONFIGS[args.config])

    if args.setting == "1shot":
        test = hw.b1.load_test_set()
        _, X, y, marks, _ = hw.b1.load_train_split(args.seed)
    else:
        test = hw.b25.load_test_set()
        X, _, y, marks = hw.b25.load_train_split(args.seed)
    marks = set(int(m) for m in marks)

    # accuracy on all test samples of classes seen so far at each checkpoint
    seen, acc_t = set(), []
    for i, (xi, yi) in enumerate(zip(X, y), start=1):
        seen.add(int(yi))
        clf.fit(xi.view(hw.b1.FEATURE_SIZE), yi.view(1), i - 1)
        if i in marks:
            acc = hw.b1.evaluate_seen_only(clf, test[0], test[1], test[2],
                                           sorted(seen), False)
            acc_t.append((clf.num_updates, acc))

    tr = np.array(trace, dtype=np.float64)  # step, winner, r, alpha, g, n0, n1
    tag = f"{args.setting}_{args.path}_{args.config}_th{args.threshold:.2f}_s{args.seed}"
    n = clf.allocator.get_allocated_count()
    W = clf.prototypes[:n]
    norms = W.norm(dim=1)
    means = clf.absorb_sum[:n]
    cos_mean = torch.nn.functional.cosine_similarity(W, means, dim=1)
    ok = torch.isfinite(norms)
    print(f"== {tag}: final acc {acc_t[-1][1]*100:.1f}%, {n} protos, "
          f"{len(tr)} updates ({(tr[:, 2] > 0).sum():.0f} hits, "
          f"{(tr[:, 2] < 0).sum():.0f} misses)")

    # ── H1 ──
    first = {}
    for row in tr:
        if row[2] > 0 and int(row[1]) not in first:
            first[int(row[1])] = row[3]
    fa = np.array(list(first.values()))
    print(f"H1  alpha on first hit: "
          + ", ".join(f"{v:.3f}: {np.mean(np.isclose(fa, v))*100:.0f}%"
                      for v in np.unique(fa.round(3))[:4]))
    print(f"H1  ||w|| mean {norms[ok].mean():.3f} (p95 {norms[ok].quantile(0.95):.3f}), "
          f"cos(w, absorbed mean) mean {cos_mean[ok].mean():.4f} "
          f"(p5 {cos_mean[ok].quantile(0.05):.4f}); non-finite protos {(~ok).sum().item()}")
    hit_rows = tr[tr[:, 2] > 0]
    miss_rows = tr[tr[:, 2] < 0]
    for name, rows in (("hit", hit_rows), ("miss", miss_rows)):
        if len(rows):
            ratio = rows[:, 6] / rows[:, 5]
            ratio = ratio[np.isfinite(ratio)]
            print(f"H2  {name:4s} updates: alpha mean {rows[:, 3].mean():.3f}, "
                  f"alpha==1 {np.mean(rows[:, 3] >= 0.999)*100:.0f}%, "
                  f"norm ratio after/before median {np.median(ratio):.4f}, "
                  f"p99 {np.quantile(ratio, 0.99):.4f}")

    # ── H2: first prototype to pass ||w|| > 1.5 ──
    big = np.where(tr[:, 6] > 1.5)[0]
    if len(big):
        k = big[0]
        pid = int(tr[k, 1])
        hist = tr[tr[:, 1] == pid]
        pos = np.where(hist[:, 0] == tr[k, 0])[0][0]
        lo = max(0, pos - args.history)
        print(f"H2  first ||w||>1.5: proto {pid} at update {int(tr[k, 0])} "
              f"(its event {pos + 1}); history (step r alpha g_before |w| -> |w|):")
        for row in hist[lo: pos + 6]:
            print(f"      {int(row[0]):6d} {int(row[2]):+d} {row[3]:.3f} "
                  f"{row[4]:6.2f}  {row[5]:9.3f} -> {row[6]:9.3f}")
        wins_after = np.mean(tr[k:, 1] == pid)
        print(f"H2  share of later updates won by proto {pid}: {wins_after*100:.0f}%")
    else:
        print("H2  no prototype passed ||w|| > 1.5")
    drop = [s for s, a in acc_t if a < 0.2]
    if drop:
        print(f"H2  accuracy first < 20% at update {drop[0]}")

    # ── H4 ──
    if args.shadow:
        S = clf.shadow_protos[:n]
        cs = torch.nn.functional.cosine_similarity(W, S, dim=1)
        print(f"H4  cos(w_int, w_float_shadow) mean {cs[ok].mean():.4f} "
              f"(p5 {cs[ok].quantile(0.05):.4f}); "
              f"cos(shadow, absorbed mean) {torch.nn.functional.cosine_similarity(S, means, dim=1).mean():.4f}")
        still = np.isclose(tr[:, 5], tr[:, 6])
        print(f"H4  updates leaving ||w|| unchanged: {still.mean()*100:.1f}% "
              f"(hits {still[tr[:, 2] > 0].mean()*100:.1f}%)")

    np.savez(DIAG / f"{tag}.npz", trace=tr, acc=np.array(acc_t),
             norms=norms.numpy(), cos_mean=cos_mean.numpy())
    fig, ax = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    ax[0].semilogy(tr[:, 0], np.where(np.isfinite(tr[:, 6]), tr[:, 6], np.nan),
                   ",", alpha=0.4)
    ax[0].set_ylabel("||w|| after update")
    a = np.array(acc_t)
    ax[1].plot(a[:, 0], a[:, 1] * 100, "-o", ms=2)
    ax[1].set_ylabel("accuracy (%)")
    ax[1].set_xlabel("sample")
    ax[0].set_title(tag, fontsize=8)
    plt.tight_layout()
    plt.savefig(DIAG / f"{tag}.png", dpi=120)
    print(f"-> {DIAG / tag}.npz/.png")


if __name__ == "__main__":
    main()
