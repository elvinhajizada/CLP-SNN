"""
Self-normalization analysis: similarity to cluster center across dimensions.

Generates Supplemental Fig. S1: a 3-row × 4-column grid showing
  Row 0 — similarity of learned prototype to cluster center
  Row 1 — weight-vector norm ‖w‖
  Row 2 — raw dot product w·center
across three synthetic dimensions (d = 64, 256, 1280) and one OpenLoris column.

Three learning-rule variants are compared:
  A — CLP:              Hebbian update + explicit L2 renorm
  B — CLP-SNN (Float):  Taylor self-normalizing rule (no explicit renorm)
  F — CLP-SNN (INT8):   Lava-faithful INT32 MAC, FLOOR shift, shared-scalar SR

Run from the repo root or from analysis/:
    python analysis/self_norm_analysis.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO / "data" / "1shot"
IMAGES_DIR = _REPO / "images"

# ── Config ─────────────────────────────────────────────────────────────────────
SAVE_PDF = True
DIMENSIONS = [64, 256, 1280]
N_SYNTH = 230
SEED = 0
ALPHA_MIN = 0
G_INC = 0.5
TARGET_MIN_SIM = 0.75
OL_CLASS = 0
OL_SEEDS = [10, 20, 30]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1,
    "axes.linewidth": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "savefig.dpi": 600,
})

# ── INT constants (15-bit accumulator) ────────────────────────────────────────
_ACC_MAX = (1 << 15) - 1
_ACC_MIN = -(1 << 15) - 1


# ── Helpers ────────────────────────────────────────────────────────────────────

def quantize_input_int(x: np.ndarray) -> np.ndarray:
    """L2-normalize → round(x * 128) → INT8 signed [-128, 127]."""
    n = np.linalg.norm(x)
    if n > 0:
        x = x / n
    return np.clip(np.round(x * 128).astype(np.int64), -128, 127)


def variant_F_step(
    w_int: np.ndarray,
    x_int: np.ndarray,
    alpha_int: np.int64,
    r_int: int,
    rng: np.random.Generator,
):
    """One CLP update step mirroring Lava's LearningRuleApplierBitApprox."""
    d = w_int.shape[0]

    # Forward pass: y = w·x quantized to 7-bit
    y_acc = np.dot(w_int.astype(np.int64), x_int.astype(np.int64))
    y_int = int(np.clip((y_acc + np.int64(64)) >> 7, -128, 127))

    # Init 15-bit accumulator (w lifted by 7 bits)
    result = (w_int.astype(np.int32) << 7).astype(np.int32)

    # Product 1: +α · x
    mac1 = np.ones(d, dtype=np.int32)
    mac1 *= np.int32(1)
    mac1 *= np.clip(x_int, -128, 127).astype(np.int32)
    mac1 *= np.int32(alpha_int)
    mac1 = np.clip(mac1 * np.int32(r_int), _ACC_MIN, _ACC_MAX)
    result = np.clip(result + mac1, _ACC_MIN, _ACC_MAX)

    # Product 2: −α · y · w
    mac2 = np.ones(d, dtype=np.int32)
    mac2 *= np.int32(1)
    mac2 *= np.int32(np.clip(y_int, -128, 127))
    mac2 *= np.clip(w_int.astype(np.int32), -512, 511)
    mac2 *= np.int32(-alpha_int)
    mac2 = np.right_shift(mac2, 7)
    mac2 = np.clip(mac2 * np.int32(r_int), _ACC_MIN, _ACC_MAX)
    result = np.clip(result + mac2, _ACC_MIN, _ACC_MAX)

    # Single stochastic round (15 → 8 bits), shared scalar rng
    rnd = rng.random()
    integer = np.floor_divide(result, 128).astype(np.int32)
    frac = (result - integer * 128).astype(np.float64) / 128.0
    w_new = integer + (frac > rnd).astype(np.int32)
    return np.clip(w_new, -128, 127).astype(np.int64)


def find_sigma_for_min_sim(
    dim: int, n_samples: int, target_min_sim: float = 0.75,
    rng_seed: int = 0, tol: float = 0.02,
) -> float:
    """Binary-search for sigma such that cluster min cosine similarity ≈ target."""
    rng = np.random.default_rng(rng_seed)
    target_angle = np.arccos(np.clip(target_min_sim, -0.99, 0.99))
    sigma_est = np.tan(target_angle) / np.sqrt(max(1, dim))
    sigma_lo = 0.001
    sigma_hi = max(0.5, sigma_est * 2.0)

    for _ in range(20):
        sigma = (sigma_lo + sigma_hi) / 2.0
        centre = rng.standard_normal(dim)
        centre /= np.linalg.norm(centre)
        samples = centre + sigma * rng.standard_normal((n_samples, dim))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        cluster_centre = samples.mean(axis=0)
        cluster_centre /= np.linalg.norm(cluster_centre)
        min_sim = float(np.min(samples @ cluster_centre))
        if min_sim > target_min_sim:
            sigma_lo = sigma
        else:
            sigma_hi = sigma
        if abs(min_sim - target_min_sim) < tol:
            return sigma

    return (sigma_lo + sigma_hi) / 2.0


def run_with_center_similarity(
    samples: np.ndarray, alpha_min: int = 0, rng_seed: int = 0, g_inc: float = 0.5,
) -> dict:
    """Run Variants A, B, F; track similarity to cluster center, norm, raw DP."""
    centre = samples.mean(axis=0)
    centre /= np.linalg.norm(centre)

    w_A = samples[0].copy()
    w_B = samples[0].copy()
    w_int_F = quantize_input_int(samples[0])

    rng_F = np.random.default_rng(rng_seed + 7919)

    g = 1.0
    sim_A, sim_B, sim_F = [], [], []
    norm_A, norm_B, norm_F = [], [], []
    raw_sim_A, raw_sim_B, raw_sim_F = [], [], []

    for t in range(1, len(samples)):
        x = samples[t]
        g += g_inc
        alpha = 1.0 / g

        # Variant A: Hebbian + explicit L2 renorm
        w_A = w_A + alpha * x
        w_A /= np.linalg.norm(w_A)

        # Variant B: float Taylor self-normalizing
        y_B = float(w_B @ x)
        w_B = w_B + alpha * (x - w_B * y_B)

        # Variant F: Lava-faithful INT
        x_int = quantize_input_int(x)
        alpha_int = np.int64(max(alpha_min, round(128 / g)))
        w_int_F = variant_F_step(w_int_F, x_int, alpha_int, 1, rng_F)

        # Units for similarity / norm
        w_A_u = w_A / (np.linalg.norm(w_A) + 1e-12)
        w_B_u = w_B / (np.linalg.norm(w_B) + 1e-12)
        w_F_f = w_int_F.astype(np.float64)
        w_F_u = w_F_f / (np.linalg.norm(w_F_f) + 1e-12)

        sa = float(np.dot(w_A_u, centre))
        sb = float(np.dot(w_B_u, centre))
        sf = float(np.dot(w_F_u, centre))
        na = float(np.linalg.norm(w_A))
        nb = float(np.linalg.norm(w_B))
        nf = float(np.linalg.norm(w_F_f) / 128.0)

        sim_A.append(sa); sim_B.append(sb); sim_F.append(sf)
        norm_A.append(na); norm_B.append(nb); norm_F.append(nf)
        raw_sim_A.append(sa * na)
        raw_sim_B.append(sb * nb)
        raw_sim_F.append(sf * nf)

    return {
        "sim_A": np.array(sim_A), "sim_B": np.array(sim_B), "sim_F": np.array(sim_F),
        "norm_A": np.array(norm_A), "norm_B": np.array(norm_B), "norm_F": np.array(norm_F),
        "raw_sim_A": np.array(raw_sim_A), "raw_sim_B": np.array(raw_sim_B),
        "raw_sim_F": np.array(raw_sim_F),
        "centre": centre,
    }


# ── Load OpenLoris class OL_CLASS (pool across OL_SEEDS) ──────────────────────

def load_openloris_class(cls: int = OL_CLASS, seeds: list = OL_SEEDS) -> np.ndarray | None:
    """Pool 1-shot training frames for `cls` across seeds; return L2-normalized array."""
    chunks = []
    for seed in seeds:
        pt = DATA_DIR / f"X_train_1_shot_{seed}.pt"
        yt = DATA_DIR / f"y_train_1_shot_{seed}.pt"
        if not pt.exists():
            continue
        X = torch.load(pt, map_location="cpu", weights_only=True).float().numpy()
        y = torch.load(yt, map_location="cpu", weights_only=True).long().numpy()
        Xc = X[y == cls]
        if len(Xc):
            norms = np.linalg.norm(Xc, axis=1, keepdims=True)
            chunks.append(Xc / np.where(norms > 0, norms, 1.0))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 1: sigma search per dimension...")
    sigmas = {}
    for dim in DIMENSIONS:
        sigma = find_sigma_for_min_sim(dim, N_SYNTH + 1, TARGET_MIN_SIM,
                                       rng_seed=SEED + dim, tol=0.02)
        sigmas[dim] = sigma
        print(f"  d={dim}: sigma={sigma:.6f}")

    print("\nPhase 2: synthetic experiments...")
    synth_results = {}
    for dim in DIMENSIONS:
        rng = np.random.default_rng(SEED + dim)
        centre = rng.standard_normal(dim)
        centre /= np.linalg.norm(centre)
        samples = centre + sigmas[dim] * rng.standard_normal((N_SYNTH + 1, dim))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        synth_results[dim] = run_with_center_similarity(
            samples, alpha_min=ALPHA_MIN, rng_seed=SEED, g_inc=G_INC)
        print(f"  d={dim}: done")

    print("\nPhase 3: OpenLoris class {OL_CLASS}...")
    ol_samples = load_openloris_class(OL_CLASS, OL_SEEDS)
    ol_result = None
    if ol_samples is not None and len(ol_samples) > 1:
        ol_result = run_with_center_similarity(
            ol_samples, alpha_min=ALPHA_MIN, rng_seed=SEED, g_inc=G_INC)
        print(f"  loaded {len(ol_samples)} frames; done")
    else:
        print("  OpenLoris data not found — column will be blank")

    # ── Figure ──────────────────────────────────────────────────────────────────
    colors = {"A": "black", "B": "royalblue", "F": "crimson"}
    labels = {"A": "CLP", "B": "CLP-SNN (Float)", "F": "CLP-SNN (INT8)"}
    linestyles = {"A": "-", "B": "--", "F": "-."}

    fig, axes = plt.subplots(3, 4, figsize=(7.087, 4.5))
    fig.patch.set_facecolor("white")

    row_labels = ["Sim to center", "‖w‖", "w · center"]
    row_keys = [("sim_A", "sim_B", "sim_F"), ("norm_A", "norm_B", "norm_F"),
                ("raw_sim_A", "raw_sim_B", "raw_sim_F")]
    row_ylims = [(0.5, 1.0), (0.8, 1.2), None]
    col_titles_synth = [f"d = {d}" for d in DIMENSIONS]
    col_title_ol = f"OpenLoris (d=1280)"

    for row, (keys, ylim_hint, ylabel) in enumerate(zip(row_keys, row_ylims, row_labels)):
        kA, kB, kF = keys

        # Synthetic columns
        for col, dim in enumerate(DIMENSIONS):
            ax = axes[row, col]
            res = synth_results[dim]
            steps = np.arange(1, len(res[kA]) + 1)
            for var, k in zip(["A", "B", "F"], [kA, kB, kF]):
                ax.plot(steps, res[k], color=colors[var], lw=0.8,
                        ls=linestyles[var],
                        label=labels[var] if (row == 0 and col == 0) else None)
            if ylim_hint is not None:
                lo, hi = ylim_hint
                m = (hi - lo) * 0.1
                ax.set_ylim([lo - m, hi + m])
            else:
                vals = np.concatenate([res[kA], res[kB], res[kF]])
                v_lo, v_hi = float(vals.min()), float(vals.max())
                m = (v_hi - v_lo) * 0.1 if v_hi > v_lo else 0.1
                ax.set_ylim([v_lo - m, v_hi + m])
            if row == 0:
                ax.set_title(col_titles_synth[col], fontsize=7, fontweight="bold")
            ax.set_xlabel("Update step", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=6.5, loc="lower right", ncol=1)

        # OpenLoris column
        ax = axes[row, 3]
        if ol_result is not None:
            steps_ol = np.arange(1, len(ol_result[kA]) + 1)
            for var, k in zip(["A", "B", "F"], [kA, kB, kF]):
                ax.plot(steps_ol, ol_result[k], color=colors[var], lw=0.8,
                        ls=linestyles[var])
            if ylim_hint is not None:
                lo, hi = ylim_hint
                m = (hi - lo) * 0.1
                ax.set_ylim([lo - m, hi + m])
            else:
                vals = np.concatenate([ol_result[kA], ol_result[kB], ol_result[kF]])
                v_lo, v_hi = float(vals.min()), float(vals.max())
                m = (v_hi - v_lo) * 0.1 if v_hi > v_lo else 0.1
                ax.set_ylim([v_lo - m, v_hi + m])
        else:
            ax.text(0.5, 0.5, "data not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
        if row == 0:
            ax.set_title(col_title_ol, fontsize=7, fontweight="bold")
        ax.set_xlabel("Update step", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "Similarity to Cluster Center, Norm & Raw Dot Product\n"
        f"(target min_sim={TARGET_MIN_SIM}, g_inc={G_INC}, α_min={ALPHA_MIN})",
        fontsize=7, fontweight="bold",
    )
    plt.tight_layout(pad=0.6, h_pad=1.0, w_pad=0.5)

    stem = IMAGES_DIR / "self_norm_analysis"
    if SAVE_PDF:
        plt.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(stem.with_suffix(".png"), format="png", bbox_inches="tight")
    print(f"\nSaved: {stem}.png" + (f" / {stem}.pdf" if SAVE_PDF else ""))
    plt.close()

    # Summary table
    print("\nFinal similarity to cluster center:")
    print(f"  {'dim':>5}  {'A (CLP)':>10}  {'B (Float)':>10}  {'F (INT8)':>10}")
    for dim in DIMENSIONS:
        r = synth_results[dim]
        print(f"  {dim:>5}  {r['sim_A'][-1]:>10.4f}  {r['sim_B'][-1]:>10.4f}  {r['sim_F'][-1]:>10.4f}")


if __name__ == "__main__":
    main()
