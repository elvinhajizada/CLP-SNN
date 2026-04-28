"""
Sigma sensitivity analysis: synthetic data only.

Compare Variants A, B, F across:
  - Two dimensions: 128, 1280
  - Four sigma values: very tight, tight, medium, large
  
For each (dimension, sigma) pair, generate a synthetic cluster and run all variants,
tracking: similarity to center, norm evolution, and raw dot products.

This demonstrates how quantization variants respond to different cluster tightness.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ── Nature Communications figure configuration ─────────────────────────────

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
    "svg.fonttype": "none",
})


# ── Stochastic-rounding helpers (Loihi BitApproximate mechanism) ──────────────

def _stoch_shift(raw: np.ndarray, shift: int, rng: np.random.Generator) -> np.ndarray:
    """Per-element stochastic rounding of an integer right-shift."""
    scale = 1 << shift
    floor_val = raw >> shift
    frac = raw - (floor_val << shift)
    bump = rng.random(size=raw.shape) < frac.astype(np.float64) / scale
    return floor_val + bump.astype(np.int64)


def _stoch_shift_scalar(raw: np.int64, shift: int, rng: np.random.Generator) -> np.int64:
    """Scalar stochastic rounding (used for the y_int dot-product result)."""
    scale = 1 << shift
    floor_val = raw >> shift
    frac = int(raw - (floor_val << shift))
    bump = np.int64(1 if rng.random() < frac / scale else 0)
    return floor_val + bump


# ── Variant F: faithful Lava BitApproximate mirror ──────────────────────────

_ACC_MAX = (1 << 15) - 1
_ACC_MIN = -(1 << 15) - 1


def variant_F_step(
    w_int: np.ndarray,
    x_int: np.ndarray,
    alpha_int: np.int64,
    r_int: int,
    rng: np.random.Generator,
):
    """One CLP update step, mirroring Lava's LearningRuleApplierBitApprox."""
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

    # Single stochastic round (15 → 8 bits), one shared scalar rnd
    rnd = rng.random()
    integer = np.floor_divide(result, 128).astype(np.int32)
    frac = (result - integer * 128).astype(np.float64) / 128.0
    w_new = integer + (frac > rnd).astype(np.int32)
    return np.clip(w_new, -128, 127).astype(np.int64), mac1, mac2


def quantize_input_int(x: np.ndarray) -> np.ndarray:
    """L2-normalize → round(x * 128) → INT8 signed [-128, 127]."""
    n = np.linalg.norm(x)
    if n > 0:
        x = x / n
    return np.clip(np.round(x * 128).astype(np.int64), -128, 127)


def run_with_center_similarity(
    samples: np.ndarray, alpha_min: int = 0, rng_seed: int = 0,
    g_inc: float = 0.2
):
    """Run Variants A, B, C, D, F and track similarity to cluster center."""
    # Cluster center: geometric mean, L2-normalized
    centre = samples.mean(axis=0)
    centre /= np.linalg.norm(centre)
    
    # Initialize all variants
    w_A = samples[0].copy()
    w_B = samples[0].copy()
    w_int_C = quantize_input_int(samples[0])
    w_int_D = quantize_input_int(samples[0])
    w_int_F = quantize_input_int(samples[0])
    
    rng = np.random.default_rng(rng_seed)
    rng_F = np.random.default_rng(rng_seed + 7919)
    
    g = 1.0
    
    # Output arrays
    sim_to_center_A = []
    sim_to_center_B = []
    sim_to_center_C = []
    sim_to_center_D = []
    sim_to_center_F = []
    norm_A = []
    norm_B = []
    norm_C = []
    norm_D = []
    norm_F = []
    raw_sim_A = []
    raw_sim_B = []
    raw_sim_F = []
    
    for t in range(1, len(samples)):
        x = samples[t]
        g += g_inc
        alpha = 1.0 / g
        
        # ── Variant A: Hebbian + explicit L2 renorm ──────────────────────────
        w_A = w_A + alpha * x
        w_A /= np.linalg.norm(w_A)
        
        # ── Variant B: float self-normalizing rule ───────────────────────────
        y_B = float(w_B @ x)
        w_B = w_B + alpha * (x - w_B * y_B)
        
        # ── Variant C: INT8 signed, S=128, deterministic round-half-up ───────
        x_int = quantize_input_int(x)
        alpha_int = np.int64(max(alpha_min, round(128 / g)))
        y_acc = np.dot(w_int_C, x_int)
        y_int = np.clip((y_acc + np.int64(64)) >> 7, -128, 127)
        p1 = (alpha_int * x_int + np.int64(64)) >> 7
        p2 = (alpha_int * y_int * w_int_C + np.int64(8192)) >> 14
        w_int_C = np.clip(w_int_C + p1 - p2, -128, 127)
        
        # ── Variant D: INT8 signed, S=128, per-dim stochastic rounding ───────
        x_int = quantize_input_int(x)
        alpha_int = np.int64(max(alpha_min, round(128 / g)))
        y_acc = np.dot(w_int_D, x_int)
        y_int = np.clip(_stoch_shift_scalar(y_acc, 7, rng), -128, 127)
        p1 = _stoch_shift(alpha_int * x_int, 7, rng)
        p2 = _stoch_shift(alpha_int * y_int * w_int_D, 14, rng)
        w_int_D = np.clip(w_int_D + p1 - p2, -128, 127)
        
        # ── Variant F: faithful Lava BitApprox mirror ────────────────────────
        x_int = quantize_input_int(x)
        alpha_int = np.int64(max(alpha_min, round(128 / g)))
        w_int_F, _, _ = variant_F_step(w_int_F, x_int, alpha_int, 1, rng_F)
        
        # ── Compute similarities to cluster center ────────────────────────────
        w_A_unit = w_A / (np.linalg.norm(w_A) + 1e-12)
        w_B_unit = w_B / (np.linalg.norm(w_B) + 1e-12)
        w_C_unit = w_int_C.astype(np.float64) / (np.linalg.norm(w_int_C.astype(np.float64)) + 1e-12)
        w_D_unit = w_int_D.astype(np.float64) / (np.linalg.norm(w_int_D.astype(np.float64)) + 1e-12)
        w_F_unit = w_int_F.astype(np.float64) / (np.linalg.norm(w_int_F.astype(np.float64)) + 1e-12)
        
        sim_a = float(np.dot(w_A_unit, centre))
        sim_b = float(np.dot(w_B_unit, centre))
        sim_c = float(np.dot(w_C_unit, centre))
        sim_d = float(np.dot(w_D_unit, centre))
        sim_f = float(np.dot(w_F_unit, centre))
        
        norm_a = np.linalg.norm(w_A)
        norm_b = np.linalg.norm(w_B)
        norm_c = np.linalg.norm(w_int_C) / 128.0
        norm_d = np.linalg.norm(w_int_D) / 128.0
        norm_f = np.linalg.norm(w_int_F) / 128.0
        
        sim_to_center_A.append(sim_a)
        sim_to_center_B.append(sim_b)
        sim_to_center_C.append(sim_c)
        sim_to_center_D.append(sim_d)
        sim_to_center_F.append(sim_f)
        
        norm_A.append(norm_a)
        norm_B.append(norm_b)
        norm_C.append(norm_c)
        norm_D.append(norm_d)
        norm_F.append(norm_f)
        
        # Raw dot products (unnormalized) = sim × norm
        raw_sim_A.append(sim_a * norm_a)
        raw_sim_B.append(sim_b * norm_b)
        raw_sim_F.append(sim_f * norm_f)
    
    return {
        "sim_A": np.array(sim_to_center_A),
        "sim_B": np.array(sim_to_center_B),
        "sim_C": np.array(sim_to_center_C),
        "sim_D": np.array(sim_to_center_D),
        "sim_F": np.array(sim_to_center_F),
        "norm_A": np.array(norm_A),
        "norm_B": np.array(norm_B),
        "norm_C": np.array(norm_C),
        "norm_D": np.array(norm_D),
        "norm_F": np.array(norm_F),
        "raw_sim_A": np.array(raw_sim_A),
        "raw_sim_B": np.array(raw_sim_B),
        "raw_sim_F": np.array(raw_sim_F),
        "centre": centre,
    }


# ── Parameters ───────────────────────────────────────────────────────────────

DIMENSIONS = [128, 1280]
N_SYNTH = 230
SEED = 0
ALPHA_MIN = 0
G_INC = 0.2

# Four sigma levels to explore cluster tightness
SIGMA_LABELS = ["Very Tight", "Tight", "Medium", "Large"]
SIGMAS = {
    "Very Tight": 0.010,   # Very tight cluster
    "Tight":      0.025,   # Tight cluster
    "Medium":     0.050,   # Medium spread
    "Large":      0.100,   # Large spread
}

# ── Main execution ────────────────────────────────────────────────────────────

print("=" * 80)
print("SIGMA SENSITIVITY ANALYSIS (Synthetic Data Only)")
print("=" * 80)

# Run experiments: all combinations of (dimension, sigma)
results = {}  # (dim, sigma_label) -> result dict

for dim in DIMENSIONS:
    for sigma_label in SIGMA_LABELS:
        sigma = SIGMAS[sigma_label]
        print(f"\n[Dimension {dim}, Sigma: {sigma_label} ({sigma:.3f})]", end=" ", flush=True)
        
        # Generate cluster
        rng = np.random.default_rng(SEED + dim + hash(sigma_label) % 100)
        init_centre = rng.standard_normal(dim)
        init_centre /= np.linalg.norm(init_centre)
        samples = init_centre + sigma * rng.standard_normal((N_SYNTH + 1, dim))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        # Verify cluster stats
        cluster_centre = samples.mean(axis=0)
        cluster_centre /= np.linalg.norm(cluster_centre)
        sims = samples @ cluster_centre
        min_sim_actual = float(np.min(sims))
        mean_sim_actual = float(np.mean(sims))
        
        # Run variants
        result = run_with_center_similarity(samples, alpha_min=ALPHA_MIN, rng_seed=SEED, g_inc=G_INC)
        results[(dim, sigma_label)] = result
        
        print(f"min_sim={min_sim_actual:.3f}, mean_sim={mean_sim_actual:.3f}")


# ── Helper: lighten/darken colors for sigma shading ────────────────────────

def shade_color(color, sigma_idx, num_sigmas=4):
    """Create shaded version of color based on sigma index (0=darkest, num_sigmas-1=lightest)."""
    import matplotlib.colors as mcolors
    c = mcolors.to_rgb(color)
    # Blend towards white: higher sigma_idx -> lighter
    blend_amount = sigma_idx / (num_sigmas - 1) * 0.6 if num_sigmas > 1 else 0.3
    shaded = tuple(c[i] + (1 - c[i]) * blend_amount for i in range(3))
    return shaded


# ── Plot: Single figure with 3 rows (metrics) × 2 columns (dimensions) ──────
# Each subplot shows all 4 sigmas with all 3 variants

base_colors = {
    "CLP": "black",
    "CLP-SNN (Float)": "royalblue",
    "CLP-SNN (INT8)": "crimson",
}

variant_names = {
    "sim_A": "CLP",
    "sim_B": "CLP-SNN (Float)",
    "sim_F": "CLP-SNN (INT8)",
    "norm_A": "CLP",
    "norm_B": "CLP-SNN (Float)",
    "norm_F": "CLP-SNN (INT8)",
    "raw_sim_A": "CLP",
    "raw_sim_B": "CLP-SNN (Float)",
    "raw_sim_F": "CLP-SNN (INT8)",
}

fig, axes = plt.subplots(3, 2, figsize=(7.087, 6.0))
fig.patch.set_facecolor("white")

# Row 0: Similarity to cluster center
for col_idx, dim in enumerate(DIMENSIONS):
    ax = axes[0, col_idx]
    
    for sigma_idx, sigma_label in enumerate(SIGMA_LABELS):
        result = results[(dim, sigma_label)]
        steps = np.arange(1, len(result["sim_A"]) + 1)
        sigma_val = SIGMAS[sigma_label]
        
        # Plot CLP, CLP-SNN (Float), CLP-SNN (INT8) with shaded colors
        color_a = shade_color(base_colors["CLP"], sigma_idx, len(SIGMA_LABELS))
        color_b = shade_color(base_colors["CLP-SNN (Float)"], sigma_idx, len(SIGMA_LABELS))
        color_f = shade_color(base_colors["CLP-SNN (INT8)"], sigma_idx, len(SIGMA_LABELS))
        
        ax.plot(steps, result["sim_A"], color=color_a, lw=0.8, ls="-", label=f"σ={sigma_val:.3f} CLP")
        ax.plot(steps, result["sim_B"], color=color_b, lw=0.8, ls="--", label=f"σ={sigma_val:.3f} CLP-SNN (Float)")
        ax.plot(steps, result["sim_F"], color=color_f, lw=0.8, ls="-.", label=f"σ={sigma_val:.3f} CLP-SNN (INT8)")
    
    # Gather all data for y-axis scaling
    all_sims = []
    for sigma_label in SIGMA_LABELS:
        result = results[(dim, sigma_label)]
        all_sims.extend([result["sim_A"], result["sim_B"], result["sim_F"]])
    all_sims = np.concatenate(all_sims)
    sim_min = np.min(all_sims)
    sim_max = np.max(all_sims)
    margin = (sim_max - sim_min) * 0.1
    ax.set_ylim([max(0.5, sim_min - margin), min(1.0, sim_max + margin)])
    
    ax.set_title(f"Dimension {dim}: Similarity to Center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Update step", fontsize=7)
    ax.set_ylabel("Cosine similarity", fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    if col_idx == 0:  # Legend only on leftmost subplot of this row
        ax.legend(fontsize=6, loc="best", ncol=2)

# Row 1: Norm evolution
for col_idx, dim in enumerate(DIMENSIONS):
    ax = axes[1, col_idx]
    
    for sigma_idx, sigma_label in enumerate(SIGMA_LABELS):
        result = results[(dim, sigma_label)]
        steps = np.arange(1, len(result["norm_A"]) + 1)
        sigma_val = SIGMAS[sigma_label]
        
        color_a = shade_color(base_colors["CLP"], sigma_idx, len(SIGMA_LABELS))
        color_b = shade_color(base_colors["CLP-SNN (Float)"], sigma_idx, len(SIGMA_LABELS))
        color_f = shade_color(base_colors["CLP-SNN (INT8)"], sigma_idx, len(SIGMA_LABELS))
        
        ax.plot(steps, result["norm_A"], color=color_a, lw=0.8, ls="-", label=f"σ={sigma_val:.3f} CLP")
        ax.plot(steps, result["norm_B"], color=color_b, lw=0.8, ls="--", label=f"σ={sigma_val:.3f} CLP-SNN (Float)")
        ax.plot(steps, result["norm_F"], color=color_f, lw=0.8, ls="-.", label=f"σ={sigma_val:.3f} CLP-SNN (INT8)")
    
    all_norms = []
    for sigma_label in SIGMA_LABELS:
        result = results[(dim, sigma_label)]
        all_norms.extend([result["norm_A"], result["norm_B"], result["norm_F"]])
    all_norms = np.concatenate(all_norms)
    norm_min = np.min(all_norms)
    norm_max = np.max(all_norms)
    margin = (norm_max - norm_min) * 0.1
    ax.set_ylim([max(0.8, norm_min - margin), min(1.2, norm_max + margin)])
    if dim == 1280:
        ax.set_ylim([0.8, 1.4])
    ax.set_title(f"Dimension {dim}: Norm Evolution", fontsize=7, fontweight="bold")
    ax.set_xlabel("Update step", fontsize=7)
    ax.set_ylabel("‖w‖", fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)

# Row 2: Raw dot products
for col_idx, dim in enumerate(DIMENSIONS):
    ax = axes[2, col_idx]
    
    for sigma_idx, sigma_label in enumerate(SIGMA_LABELS):
        result = results[(dim, sigma_label)]
        steps = np.arange(1, len(result["raw_sim_A"]) + 1)
        sigma_val = SIGMAS[sigma_label]
        
        color_a = shade_color(base_colors["CLP"], sigma_idx, len(SIGMA_LABELS))
        color_b = shade_color(base_colors["CLP-SNN (Float)"], sigma_idx, len(SIGMA_LABELS))
        color_f = shade_color(base_colors["CLP-SNN (INT8)"], sigma_idx, len(SIGMA_LABELS))
        
        ax.plot(steps, result["raw_sim_A"], color=color_a, lw=0.8, ls="-", label=f"σ={sigma_val:.3f} CLP")
        ax.plot(steps, result["raw_sim_B"], color=color_b, lw=0.8, ls="--", label=f"σ={sigma_val:.3f} CLP-SNN (Float)")
        ax.plot(steps, result["raw_sim_F"], color=color_f, lw=0.8, ls="-.", label=f"σ={sigma_val:.3f} CLP-SNN (INT8)")
    
    all_raw_sims = []
    for sigma_label in SIGMA_LABELS:
        result = results[(dim, sigma_label)]
        all_raw_sims.extend([result["raw_sim_A"], result["raw_sim_B"], result["raw_sim_F"]])
    all_raw_sims = np.concatenate(all_raw_sims)
    raw_sim_min = np.min(all_raw_sims)
    raw_sim_max = np.max(all_raw_sims)
    margin = (raw_sim_max - raw_sim_min) * 0.1 if raw_sim_max > raw_sim_min else 0.1
    # ax.set_ylim([raw_sim_min - margin, raw_sim_max + margin])
    ax.set_ylim([0.7, 1.1])
    
    ax.set_title(f"Dimension {dim}: Raw Dot Product", fontsize=7, fontweight="bold")
    ax.set_xlabel("Update step", fontsize=7)
    ax.set_ylabel("w · center", fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)

fig.suptitle("Sigma Sensitivity Analysis: Synthetic Data", fontsize=13, fontweight="bold")
plt.tight_layout(pad=1.0)
plt.savefig("sigma_sensitivity_analysis.pdf", format="pdf", bbox_inches="tight", dpi=600)
plt.savefig("sigma_sensitivity_analysis.png", format="png", bbox_inches="tight", dpi=600)
print("\n[OK] Saved: sigma_sensitivity_analysis.pdf (7.087 inch width)")

plt.show()

# ── Print summary table ───────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY: Final Similarity to Cluster Center (last step)")
print("=" * 80)

for dim in DIMENSIONS:
    print(f"\nDimension {dim}:")
    print(f"{'Sigma Label':>15}  {'Value':>8}  {'A (ref)':>10}  {'B (float)':>10}  {'F (Lava)':>10}")
    print("-" * 75)
    for sigma_label in SIGMA_LABELS:
        result = results[(dim, sigma_label)]
        sigma_val = SIGMAS[sigma_label]
        print(
            f"{sigma_label:>15}  "
            f"{sigma_val:>8.4f}  "
            f"{result['sim_A'][-1]:>10.4f}  "
            f"{result['sim_B'][-1]:>10.4f}  "
            f"{result['sim_F'][-1]:>10.4f}"
        )

print("\n" + "=" * 80)
print("SUMMARY: Error vs Variant B (float reference)")
print("=" * 80)

for dim in DIMENSIONS:
    print(f"\nDimension {dim}:")
    print(f"{'Sigma Label':>15}  {'Value':>8}  {'ΔF':>10}  {'% Error':>10}")
    print("-" * 75)
    for sigma_label in SIGMA_LABELS:
        result = results[(dim, sigma_label)]
        sigma_val = SIGMAS[sigma_label]
        err_F = abs(result['sim_F'][-1] - result['sim_B'][-1])
        pct_err = 100.0 * err_F / (abs(result['sim_B'][-1]) + 1e-12)
        print(
            f"{sigma_label:>15}  "
            f"{sigma_val:>8.4f}  "
            f"{err_F:>10.4f}  "
            f"{pct_err:>10.2f}%"
        )

print("\n✓ Done")
