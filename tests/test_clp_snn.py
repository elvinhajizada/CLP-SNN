"""
Tests for the two CLP-SNN models.

1. models/CLP_SNN_Loihi.py (CLPSNNLoihi), with the deployment parameters in
   LOIHI_CONFIGS, reproduces the accuracy curves measured on Loihi 2
   (data/loihi2/).
2. The integer update of models/CLP_SNN.py (CLPSNN._update_winner_int8) equals
   the numpy reference in analysis/int8_rule.py for both signs of the reward.
3. The weight norm of an adaptive CLPSNN prototype stays bounded over a run of
   consecutive incorrect predictions (reward -1).
4. Majority voting counts prototypes of class 0.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "experiments", REPO / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clp_snn_gap_decomposition as gap  # noqa: E402
from int8_rule import quantize_input_int, variant_F_step  # noqa: E402
from models.CLP_SNN import CLPSNN  # noqa: E402
from models.CLP_SNN_Loihi import CLPSNNLoihi  # noqa: E402
from models.clp_snn_configs import LOIHI_CONFIGS  # noqa: E402

D = 1280

# Tolerances of the emulator against the chip, in percentage points, for one
# tie-breaking seed. The emulator breaks label ties at random, as the chip
# does, so single runs differ by a few tenths of a point.
FINAL_TOL = 0.5     # |mean final accuracy, emulator - chip|
CURVE_RMSE_TOL = 1.0


# ---------------------------------------------------------------------------
# 1. Emulator vs the accuracy curves measured on Loihi 2
# ---------------------------------------------------------------------------

def _emulator_vs_chip(setting):
    chip = np.load(REPO / "data" / "loihi2"
                   / f"accuracies_clp_loihi_{setting}.npy")[:, 0, :].T * 100
    curves = np.array([
        gap.run_stream(
            CLPSNNLoihi(D, 40, seed=0, **LOIHI_CONFIGS[f"paper_{setting}"]),
            setting, seed, protocol="chip")
        for seed in gap.SEEDS])
    final_diff = curves[:, -1].mean() - chip[:, -1].mean()
    rmse = float(np.sqrt(((curves - chip) ** 2).mean()))
    return final_diff, rmse


def test_emulator_reproduces_chip_1shot():
    final_diff, rmse = _emulator_vs_chip("1shot")
    assert abs(final_diff) <= FINAL_TOL, f"final accuracy off by {final_diff:+.2f} pp"
    assert rmse <= CURVE_RMSE_TOL, f"curve RMSE {rmse:.2f} pp"


def test_emulator_reproduces_chip_25shot():      # about one minute
    final_diff, rmse = _emulator_vs_chip("25shot")
    assert abs(final_diff) <= FINAL_TOL, f"final accuracy off by {final_diff:+.2f} pp"
    assert rmse <= CURVE_RMSE_TOL, f"curve RMSE {rmse:.2f} pp"


# ---------------------------------------------------------------------------
# 2. Torch integer update vs the numpy reference
# ---------------------------------------------------------------------------

def _unit(rng, centre=None, sigma=0.01):
    """Random unit vector, or one from a cluster around `centre` (cosine
    similarity about 0.94 to the centre at sigma = 0.01)."""
    v = rng.standard_normal(D) if centre is None else centre + sigma * rng.standard_normal(D)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("r", [1, -1])
def test_int8_update_matches_numpy_reference(r):
    rng = np.random.default_rng(0)
    for trial in range(100):
        centre = _unit(rng)
        w_int = quantize_input_int(_unit(rng, centre))
        x_int = quantize_input_int(_unit(rng, centre))
        g = float(rng.choice([1.5, 2.0, 3.5, 8.0, 40.0]))
        alpha_int = max(0, min(127, round(128 / g)))

        expected = variant_F_step(w_int, x_int, alpha_int, r,
                                  np.random.default_rng(trial))

        model = CLPSNN(D, n_protos=1, device="cpu", seed=trial)
        model.prototypes[0] = torch.tensor(w_int / 128.0, dtype=torch.float32)
        model.goodness[0] = g
        model._update_winner_int8(
            0, torch.tensor(x_int / 128.0, dtype=torch.float32), float(r))
        got = (model.prototypes[0] * 128).round().to(torch.int64).numpy()

        assert np.array_equal(got, expected), f"trial {trial}, r = {r}, g = {g}"


# ---------------------------------------------------------------------------
# 3. Weight norm under consecutive incorrect predictions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_quantization", [False, True])
@pytest.mark.parametrize("g0, min_misses", [(1.5, 1), (50.0, 10)])
def test_norm_bounded_under_consecutive_misses(use_quantization, g0, min_misses):
    """Reward -1 applied to one prototype again and again, for as long as the
    prototype could still win (positive similarity to the input). Covers a
    young prototype (g0 = 1.5, so alpha = 1 on the first miss) and a mature
    one (g0 = 50). The norm must not grow."""
    rng = np.random.default_rng(1)
    centre = _unit(rng)
    model = CLPSNN(D, n_protos=1, device="cpu", g_inc=0.5,
                   use_quantization=use_quantization)

    def sample():
        x = torch.tensor(_unit(rng, centre), dtype=torch.float32)
        return model._quantize_input(x) if use_quantization else x

    model.prototypes[0] = sample()
    model.goodness[0] = g0
    norm0 = model.prototypes[0].norm().item()

    misses, max_norm = 0, norm0
    for _ in range(30):
        x = sample()
        if (model.prototypes[0] @ x).item() <= 0:
            break
        model._update_winner(0, x, r=-1.0)
        misses += 1
        max_norm = max(max_norm, model.prototypes[0].norm().item())

    assert misses >= min_misses, f"only {misses} misses were exercised"
    assert torch.isfinite(model.prototypes).all()
    assert max_norm <= norm0 + 0.02, \
        f"norm grew from {norm0:.3f} to {max_norm:.3f} over {misses} misses"


# ---------------------------------------------------------------------------
# 4. Voting includes class 0
# ---------------------------------------------------------------------------

def test_voting_counts_class_zero():
    model = CLPSNN(8, n_protos=4, device="cpu", enable_voting=True)
    model.proto_labels[:3] = torch.tensor([0, 0, 5])
    # Candidates sorted by similarity; class 0 holds two of the three votes
    candidates = [(2, 0.90), (0, 0.85), (1, 0.80)]
    winner = model._perform_voting(candidates)
    assert winner == 0, "majority label 0 should win, via its most similar prototype"
