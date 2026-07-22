"""
Invariant tests for RankOneSLDA (Sherman-Morrison SLDA):

  Exactness: at checkpoints along the 1-shot stream, the maintained Lambda
     matches inv(S + lambda*I) with S recomputed from an FP64 shadow
     accumulation of Hayes' scatter recursion; likewise for prediction scores.
  Safety: the Sherman-Morrison denominator is >= 1 at every step
     (guaranteed analytically; the assert catches implementation bugs).
  Prediction agreement: argmax agreement vs a from-scratch inv(S+lambda*I)
     SLDA over a full 1-shot stream >= 99.9%.

Plus an FP64 synthetic exactness test that validates the algebra itself
(deviation at machine precision, decoupled from FP32 drift), and a check that
the first-ever sample leaves the precision untouched (c = 0).

The FP32 drift study over the 60k-sample 25-shot stream lives in
experiments/slda_drift_study.py.

Run from the repo root:  pytest tests/ -v
Runtime: ~1-2 min on CPU (dominated by the FP64 shadow scatter accumulation).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.SLDA import RankOneSLDA  # noqa: E402

RIDGE = 1.0          # any lambda works; tests are self-consistent under it
FEATURE_SIZE = 1280
NUM_CLASSES = 40
SEED = 10            # one seed suffices for invariant checks
N_FRAMES = 60
GLOBAL_SEED = 42     # matches the experiments' test-set balancing seed
CHECKPOINT_EVERY = 400

DATA_DIR = REPO / "data"


# ── FP64 shadow reference ─────────────────────────────────────────────────────

def shadow_update(S, muK, cK, n, x, y):
    """One step of Hayes' scatter recursion in FP64: S <- S + c*v*v^T with
    c = n/(n+1) (global counter), v the deviation from the pre-update mean."""
    v = x - muK[y]
    c = n / (n + 1)
    if c > 0:
        S += c * torch.outer(v, v)
    muK[y] += v / (cK[y] + 1)
    cK[y] += 1
    return n + 1


def ref_lambda(S, lam):
    d = S.shape[0]
    return torch.linalg.inv(S + lam * torch.eye(d, dtype=S.dtype))


def ref_scores(Lam, muK, cK, X):
    """From-scratch SLDA scores with the same unseen-class masking as predict()."""
    W = Lam @ muK.t()
    b = 0.5 * torch.sum(muK.t() * W, dim=0)
    scores = X.to(Lam.dtype) @ W - b
    not_visited_ix = torch.where(cK == 0)[0]
    min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
    scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
        len(not_visited_ix), len(X)).transpose(1, 0)
    return scores


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_set():
    """OpenLoris test features, balanced to 60/class, L2-normalised
    (mirrors experiments/clp_vs_baselines_1shot.py)."""
    np.random.seed(GLOBAL_SEED)
    X = np.load(DATA_DIR / "X_test.npy")
    y = np.load(DATA_DIR / "y_test.npy")
    balanced_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        balanced_idx.extend(np.random.choice(cls_idx, N_FRAMES, replace=False))
    X = X[balanced_idx]
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return torch.from_numpy(X_norm).float()


def load_train_stream(seed):
    X = torch.load(DATA_DIR / "1shot" / f"X_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).float()
    y = torch.load(DATA_DIR / "1shot" / f"y_train_1_shot_{seed}.pt",
                   map_location="cpu", weights_only=True).long()
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return X, y


# ── Stream run (shared by the stream tests) ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def stream_metrics():
    """Run the full 1-shot stream once; collect per-checkpoint deviations of
    the maintained FP32 state vs the FP64 from-scratch reference."""
    X_tr, y_tr = load_train_stream(SEED)
    X_probe = load_test_set()

    model = RankOneSLDA(FEATURE_SIZE, NUM_CLASSES, ridge_param=RIDGE,
                        device="cpu")

    S = torch.zeros((FEATURE_SIZE, FEATURE_SIZE), dtype=torch.float64)
    muK = torch.zeros((NUM_CLASSES, FEATURE_SIZE), dtype=torch.float64)
    cK = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    n = 0

    checkpoints = []
    n_total = len(y_tr)
    for i in range(n_total):
        x64 = X_tr[i].double()
        model.fit(X_tr[i], y_tr[i].view(1), i)
        n = shadow_update(S, muK, cK, n, x64, int(y_tr[i]))

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == n_total:
            Lam_ref = ref_lambda(S, RIDGE)
            scores_ref = ref_scores(Lam_ref, muK, cK, X_probe)
            scores_model = model.predict(X_probe).double()

            lambda_dev = (model.Lambda.double() - Lam_ref).abs().max().item()
            score_dev = (scores_model - scores_ref).abs().max().item()
            score_scale = scores_ref.abs().max().item()
            agree = (scores_model.argmax(dim=1)
                     == scores_ref.argmax(dim=1)).float().mean().item()
            checkpoints.append({
                "step": i + 1,
                "lambda_dev": lambda_dev,
                "lambda_scale": Lam_ref.abs().max().item(),
                "score_dev": score_dev,
                "score_scale": score_scale,
                "argmax_agreement": agree,
            })

    return {"checkpoints": checkpoints, "model": model, "n_probe": len(X_probe)}


# ── Exactness ─────────────────────────────────────────────────────────────

def test_lambda_exactness(stream_metrics):
    """Maintained FP32 Lambda tracks inv(S + lambda*I) at every checkpoint."""
    for ck in stream_metrics["checkpoints"]:
        rel = ck["lambda_dev"] / ck["lambda_scale"]
        assert rel < 1e-4, (
            f"step {ck['step']}: relative Lambda deviation {rel:.2e} "
            f"(abs {ck['lambda_dev']:.2e}, scale {ck['lambda_scale']:.2e})")


def test_score_exactness(stream_metrics):
    """Scores from the maintained W match the from-scratch reference."""
    for ck in stream_metrics["checkpoints"]:
        rel = ck["score_dev"] / ck["score_scale"]
        assert rel < 1e-3, (
            f"step {ck['step']}: relative score deviation {rel:.2e} "
            f"(abs {ck['score_dev']:.2e}, scale {ck['score_scale']:.2e})")


# ── Safety ────────────────────────────────────────────────────────────────

def test_denominator_safety(stream_metrics):
    """denom = 1 + c*v^T*Lambda*v >= 1 at every step of the stream (fit()
    raises on violation; min_denom double-checks the tracked minimum)."""
    assert stream_metrics["model"].min_denom >= 1.0 - 1e-6


# ── Prediction agreement ──────────────────────────────────────────────────

def test_prediction_agreement(stream_metrics):
    """Argmax agreement vs the from-scratch inv(S+lambda*I) SLDA >= 99.9%,
    aggregated over all checkpoints of the full 1-shot stream."""
    cks = stream_metrics["checkpoints"]
    agreement = sum(ck["argmax_agreement"] for ck in cks) / len(cks)
    assert agreement >= 0.999, (
        f"aggregate argmax agreement {agreement:.5f} < 0.999; per-checkpoint: "
        f"{[round(ck['argmax_agreement'], 5) for ck in cks]}")


# ── Algebra validation (FP64, synthetic) ──────────────────────────────────────

def test_algebra_exact_fp64():
    """With FP64 state the recursion is exact to machine precision: this
    isolates the algebra (steps 1-7) from floating-point drift."""
    torch.manual_seed(0)
    d, K, n_samples = 48, 5, 300
    X = torch.randn(n_samples, d, dtype=torch.float64)
    X = X / X.norm(dim=1, keepdim=True)
    y = torch.randint(0, K, (n_samples,))

    model = RankOneSLDA(d, K, ridge_param=RIDGE, device="cpu",
                        dtype=torch.float64)
    S = torch.zeros((d, d), dtype=torch.float64)
    muK = torch.zeros((K, d), dtype=torch.float64)
    cK = torch.zeros(K, dtype=torch.float64)
    n = 0
    for i in range(n_samples):
        model.fit(X[i], y[i].view(1), i)
        n = shadow_update(S, muK, cK, n, X[i], int(y[i]))

    Lam_ref = ref_lambda(S, RIDGE)
    W_ref = Lam_ref @ muK.t()
    assert (model.Lambda - Lam_ref).abs().max().item() < 1e-11
    assert (model.W - W_ref).abs().max().item() < 1e-11


def test_first_sample_leaves_precision_untouched():
    """First-ever sample: c = 0, so Lambda stays I/lambda; the class mean and
    its weight column are still updated."""
    d, K = 16, 3
    model = RankOneSLDA(d, K, ridge_param=RIDGE, device="cpu",
                        dtype=torch.float64)
    x = torch.randn(d, dtype=torch.float64)
    model.fit(x, torch.tensor([1]), 0)
    assert torch.equal(model.Lambda,
                       torch.eye(d, dtype=torch.float64) / RIDGE)
    assert torch.allclose(model.muK[1], x)
    assert torch.allclose(model.W[:, 1], x / RIDGE)
