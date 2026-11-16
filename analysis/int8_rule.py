"""
Numpy reference of the CLP-SNN integer learning rule ("Variant F").

One update of the winning prototype, mirroring Lava's
LearningRuleApplierBitApprox: int32 products, floor right-shift, 15-bit
saturating accumulator and one shared-scalar stochastic round per step.

    dw = alpha * (r * x - |r| * y * w),    y = w.x,    r in {+1, -1}

The sign of the reward reaches only the Hebbian product. The decay product is
gated by |r|, which is 1 for the prototype that learns, so it carries no
reward factor here.

`CLPSNN._update_winner_int8` in models/CLP_SNN.py is the torch port of this
function; tests/test_clp_snn.py checks the two against each other for both
signs of the reward. analysis/self_norm_analysis.py and
analysis/sigma_sensitivity_analysis.py import it for Supplementary Figs. S2
and S3.
"""

import numpy as np

# 15-bit accumulator saturation (Lava W_ACCUMULATOR_U = 15)
_ACC_MAX = (1 << 15) - 1
_ACC_MIN = -(1 << 15) - 1


def quantize_input_int(x: np.ndarray) -> np.ndarray:
    """L2-normalize, then round(x * 128) to signed 8-bit [-128, 127]."""
    n = np.linalg.norm(x)
    if n > 0:
        x = x / n
    return np.clip(np.round(x * 128).astype(np.int64), -128, 127)


def variant_F_step(
    w_int: np.ndarray,
    x_int: np.ndarray,
    alpha_int: int,
    r_int: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One integer update of the winner; returns the new 8-bit weights.

    w_int, x_int : weights and input at scale 128
    alpha_int    : learning rate at scale 128, round(128 / g)
    r_int        : reward, +1 (correct prediction) or -1 (incorrect)
    rng          : generator for the shared-scalar stochastic round
    """
    d = w_int.shape[0]

    # Forward pass: y = w.x quantized to 7 bits
    y_acc = np.dot(w_int.astype(np.int64), x_int.astype(np.int64))
    y_int = int(np.clip((y_acc + np.int64(64)) >> 7, -128, 127))

    # 15-bit accumulator, initialized with w lifted by 7 bits
    result = (w_int.astype(np.int32) << 7).astype(np.int32)

    # Product 1: r * alpha * x  (the reward signs the Hebbian term)
    mac1 = np.ones(d, dtype=np.int32)
    mac1 *= np.clip(x_int, -128, 127).astype(np.int32)
    mac1 *= np.int32(alpha_int)
    mac1 = np.clip(mac1 * np.int32(r_int), _ACC_MIN, _ACC_MAX)
    result = np.clip(result + mac1, _ACC_MIN, _ACC_MAX)

    # Product 2: -alpha * y * w  (the decay; no reward factor)
    mac2 = np.ones(d, dtype=np.int32)
    mac2 *= np.int32(np.clip(y_int, -128, 127))
    mac2 *= np.clip(w_int.astype(np.int32), -512, 511)
    mac2 *= np.int32(-alpha_int)
    mac2 = np.right_shift(mac2, 7)
    mac2 = np.clip(mac2, _ACC_MIN, _ACC_MAX)
    result = np.clip(result + mac2, _ACC_MIN, _ACC_MAX)

    # Single stochastic round (15 -> 8 bits), one scalar shared by all weights
    rnd = rng.random()
    integer = np.floor_divide(result, 128).astype(np.int32)
    frac = (result - integer * 128).astype(np.float64) / 128.0
    w_new = integer + (frac > rnd).astype(np.int32)
    return np.clip(w_new, -128, 127).astype(np.int64)
