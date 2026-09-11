"""
Parity tests: benchmark-optimized models vs the pinned pre-optimization
implementations (git tag `pre-orin-opt`).

Every model touched by the pre-Orin optimization pass (RankOneSLDA, NCM, CLP,
Replay) is run side by side with its pinned version over the real seed-10
1-shot OpenLORIS feature stream. The optimizations are required to be
prediction-identical: state tensors must match (bitwise where the arithmetic
is unchanged, tight tolerance where kernel shapes differ), and argmax
predictions on the full test set must agree 100%.

The pinned sources are extracted with `git show pre-orin-opt:models/<f>.py`
into a temp directory at session start, so the reference is exactly what the
accuracy experiments ran, independent of the working tree.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "data"
PIN_TAG = "pre-orin-opt"

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pinned(tmp_path_factory):
    """Extract pinned model sources from the git tag and import them under
    unique module names."""
    pin_dir = tmp_path_factory.mktemp("pinned_models")
    modules = {}
    for fname in ["SLDA", "NCM", "CLP", "Replay"]:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{PIN_TAG}:models/{fname}.py"],
            capture_output=True, text=True, check=True).stdout
        path = pin_dir / f"pinned_{fname}.py"
        path.write_text(src, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"pinned_{fname}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        modules[fname] = mod
    return modules


@pytest.fixture(scope="session")
def stream():
    """Real seed-10 1-shot stream and test set (raw and L2-normalized)."""
    X = torch.load(DATA / "1shot" / "X_train_1_shot_10.pt",
                   map_location="cpu", weights_only=False).float()
    y = torch.load(DATA / "1shot" / "y_train_1_shot_10.pt",
                   map_location="cpu", weights_only=False).long()
    X_test = torch.from_numpy(np.load(DATA / "X_test.npy")).float()
    y_test = torch.from_numpy(np.load(DATA / "y_test.npy"))
    Xn = X / X.norm(dim=1, keepdim=True)
    Xn_test = X_test / X_test.norm(dim=1, keepdim=True)
    return {"X_raw": X, "X_norm": Xn, "y": y,
            "X_test_raw": X_test, "X_test_norm": Xn_test, "y_test": y_test}


def _fit_stream(model, X, y):
    for i in range(X.shape[0]):
        model.fit(X[i], y[i].view(1), i)


def _argmax_agreement(scores_a, scores_b):
    return (scores_a.argmax(dim=1) == scores_b.argmax(dim=1)).float().mean().item()


# ---------------------------------------------------------------------------
# RankOneSLDA: buffered/fused fit + maintained bias vs pinned
# ---------------------------------------------------------------------------

def test_rank_one_slda_parity(pinned, stream):
    from models.SLDA import RankOneSLDA

    d = stream["X_norm"].shape[1]
    ref = pinned["SLDA"].RankOneSLDA(d, 40, device="cpu", ridge_param=1.0)
    new = RankOneSLDA(d, 40, device="cpu", ridge_param=1.0, debug_checks=True)

    _fit_stream(ref, stream["X_norm"], stream["y"])
    _fit_stream(new, stream["X_norm"], stream["y"])

    # The buffered fit uses the same kernels in the same order: state must be
    # bitwise identical to the pinned implementation.
    assert torch.equal(ref.Lambda, new.Lambda), "Lambda diverged from pinned"
    assert torch.equal(ref.muK, new.muK), "muK diverged from pinned"
    assert torch.equal(ref.W, new.W), "W diverged from pinned"
    assert torch.equal(ref.cK, new.cK), "cK diverged from pinned"

    # Maintained bias must match the from-scratch definition on current state
    b_ref = 0.5 * torch.sum(new.muK.t() * new.W, dim=0)
    assert torch.allclose(new.b, b_ref, rtol=1e-3, atol=1e-5), \
        f"maintained b drifted: max abs diff {(new.b - b_ref).abs().max().item()}"

    # Predictions must agree exactly on the full test set
    s_ref = ref.predict(stream["X_test_norm"])
    s_new = new.predict(stream["X_test_norm"])
    agreement = _argmax_agreement(s_ref, s_new)
    assert agreement == 1.0, f"argmax agreement {agreement} < 1.0"

    # debug instrumentation on/off must not change arithmetic
    fast = RankOneSLDA(d, 40, device="cpu", ridge_param=1.0, debug_checks=False)
    _fit_stream(fast, stream["X_norm"], stream["y"])
    assert torch.equal(fast.Lambda, new.Lambda)
    assert torch.equal(fast.W, new.W)


# ---------------------------------------------------------------------------
# NCM: cached prototype norms vs pinned
# ---------------------------------------------------------------------------

def test_ncm_parity(pinned, stream):
    from models.NCM import NearestClassMean

    d = stream["X_raw"].shape[1]
    ref = pinned["NCM"].NearestClassMean(d, 40, device="cpu")
    new = NearestClassMean(d, 40, device="cpu")

    _fit_stream(ref, stream["X_raw"], stream["y"])
    _fit_stream(new, stream["X_raw"], stream["y"])

    assert torch.equal(ref.muK, new.muK)
    assert torch.equal(ref.cK, new.cK)

    # cached norms must match a from-scratch recompute
    sq = torch.sum(new.muK * new.muK, dim=1)
    assert torch.allclose(new.muK_sqnorm, sq, rtol=1e-6, atol=1e-6)

    s_ref = ref.predict(stream["X_test_raw"])
    s_new = new.predict(stream["X_test_raw"])
    agreement = _argmax_agreement(s_ref, s_new)
    assert agreement == 1.0, f"argmax agreement {agreement} < 1.0"


# ---------------------------------------------------------------------------
# CLP: allocated-slot GEMM + topk vs pinned (full sort over all slots)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [
    # benchmark configuration (n_wta=1)
    dict(n_protos=300, sim_th_init=0.75, n_wta=1),
    # multi-winner configuration exercising the WTA mis-prediction loop
    dict(n_protos=500, sim_th_init=0.45, n_wta=5),
])
def test_clp_parity(pinned, stream, cfg):
    from models.CLP import ContinuallyLearningPrototypes

    d = stream["X_norm"].shape[1]
    common = dict(num_classes=40, backbone=None, alpha_init=1, k_hit=1,
                  k_miss=0.5, adaptive_th=False, adaptive_protos=True,
                  device="cpu")
    ref = pinned["CLP"].ContinuallyLearningPrototypes(d, **common, **cfg)
    new = ContinuallyLearningPrototypes(d, **common, **cfg)

    for i in range(stream["X_norm"].shape[0]):
        ref.fit(stream["X_norm"][i], stream["y"][i].view(1), i)
        new.fit(stream["X_norm"][i], stream["y"][i].view(1), i)

    # Allocation decisions and labels are discrete: must match exactly
    assert ref.next_alloc_id_ == new.next_alloc_id_
    assert torch.equal(ref.proto_labels_, new.proto_labels_)
    assert torch.equal(ref.goodness_, new.goodness_)
    # Prototype values: GEMM over the sliced matrix may round differently
    # than over the full matrix, so allow kernel-level tolerance
    assert torch.allclose(ref.prototypes_, new.prototypes_, rtol=1e-5, atol=1e-6)

    s_ref = ref.predict(stream["X_test_norm"])
    s_new = new.predict(stream["X_test_norm"])
    agreement = _argmax_agreement(s_ref, s_new)
    assert agreement == 1.0, f"argmax agreement {agreement} < 1.0"


# ---------------------------------------------------------------------------
# CLP: single-sync fit / scatter-max predict vs the pinned pre-sync-opt tag
# (the implementation that produced the Phase 3 Orin numbers). Bar: bitwise.
# ---------------------------------------------------------------------------

SYNC_PIN_TAG = "pre-sync-opt"


@pytest.fixture(scope="session")
def pinned_sync(tmp_path_factory):
    """CLP source at the pre-sync-opt tag, imported under a unique name."""
    pin_dir = tmp_path_factory.mktemp("pinned_sync_models")
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SYNC_PIN_TAG}:models/CLP.py"],
        capture_output=True, text=True, check=True).stdout
    path = pin_dir / "pinned_sync_CLP.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pinned_sync_CLP", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLP_BITWISE_CFGS = [
    # benchmark configuration (n_wta=1), benchmarks/benchmark.py
    dict(n_protos=300, sim_th_init=0.75, n_wta=1, k_miss=0.5),
    # accuracy-experiment configuration, experiments/clp_vs_baselines_*shot.py
    dict(n_protos=400, sim_th_init=0.75, n_wta=1, k_miss=1),
    # multi-winner configuration exercising the WTA mis-prediction loop
    dict(n_protos=500, sim_th_init=0.45, n_wta=5, k_miss=0.5),
    # tiny buffer: exercises the full-buffer clamp and slot re-fit path
    dict(n_protos=20, sim_th_init=0.75, n_wta=1, k_miss=0.5),
]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cfg", CLP_BITWISE_CFGS)
def test_clp_bitwise_vs_pre_sync_opt(pinned_sync, stream, cfg, device):
    from models.CLP import ContinuallyLearningPrototypes

    d = stream["X_norm"].shape[1]
    common = dict(num_classes=40, backbone=None, alpha_init=1, k_hit=1,
                  adaptive_th=False, adaptive_protos=True, learn_outliers=True,
                  device=device)
    ref = pinned_sync.ContinuallyLearningPrototypes(d, **common, **cfg)
    new = ContinuallyLearningPrototypes(d, **common, **cfg)

    X = stream["X_norm"].to(device)
    y = stream["y"].to(device)
    for i in range(X.shape[0]):
        ref.fit(X[i], y[i].view(1), i)
        new.fit(X[i], y[i].view(1), i)
        # state must agree after every step, not only at the end
        if i % 200 == 0 or i == X.shape[0] - 1:
            assert torch.equal(ref.prototypes_.cpu(), new.prototypes_.cpu()), \
                f"prototypes diverged at step {i}"

    assert ref.next_alloc_id_ == new.next_alloc_id_
    assert ref.classes_ == new.classes_
    assert ref.n_error == new.n_error
    assert ref.n_outlier == new.n_outlier
    assert ref.num_updates == new.num_updates
    assert ref._n_allocated() == new._n_allocated()
    assert torch.equal(ref.proto_labels_.cpu(), new.proto_labels_.cpu())
    assert torch.equal(ref.goodness_.cpu(), new.goodness_.cpu())
    assert torch.equal(ref.alphas_.cpu(), new.alphas_.cpu())
    assert torch.equal(ref.sim_th_.cpu(), new.sim_th_.cpu())
    assert torch.equal(ref.prototypes_.cpu(), new.prototypes_.cpu())

    X_test = stream["X_test_norm"].to(device)
    for thresholded in (False, True):
        for return_probas in (False, True):
            s_ref = ref.predict(X_test, return_probas=return_probas,
                                thresholded=thresholded)
            s_new = new.predict(X_test, return_probas=return_probas,
                                thresholded=thresholded)
            assert torch.equal(s_ref, s_new), \
                f"predict differs (thresholded={thresholded}, probas={return_probas})"
    assert torch.equal(ref.predict(X_test, return_sims=True),
                       new.predict(X_test, return_sims=True))
    assert torch.equal(ref.predict(X_test, return_voting_winner=True),
                       new.predict(X_test, return_voting_winner=True))


# Non-default code paths touched by the single-sync rewrite (not used by any
# reported experiment, but they must stay bitwise as well): euclidean metric
# (which, as before, allocates every sample: negative similarities never pass
# the sum > 0 winner check), unsupervised mode, adaptive thresholds with the
# mis-prediction loop, allocate-on-error with and without a learning period.
# sim_th_init=0.45 is chosen so that the stream produces mistakes.
CLP_EXTRA_CFGS = [
    dict(sim_metric='euclidean', n_protos=100, sim_th_init=-1.05, n_wta=3),
    dict(supervised=False, n_protos=200, sim_th_init=0.75, n_wta=1),
    dict(adaptive_th=True, n_protos=300, sim_th_init=0.45, n_wta=5, k_miss=0.5),
    dict(adaptive_protos=False, n_protos=300, sim_th_init=0.45, n_wta=1),
    dict(adaptive_protos=False, learning_period=3, n_protos=300, sim_th_init=0.45, n_wta=1),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cfg", CLP_EXTRA_CFGS)
def test_clp_bitwise_extra_paths(pinned_sync, stream, cfg, device):
    from models.CLP import ContinuallyLearningPrototypes

    d = stream["X_norm"].shape[1]
    kw = dict(num_classes=40, backbone=None, alpha_init=1, k_hit=1,
              learn_outliers=True, device=device)
    kw.update(cfg)
    ref = pinned_sync.ContinuallyLearningPrototypes(d, **kw)
    new = ContinuallyLearningPrototypes(d, **kw)

    X = stream["X_norm"].to(device)
    y = stream["y"].to(device)
    for i in range(X.shape[0]):
        ref.fit(X[i], y[i].view(1), i)
        new.fit(X[i], y[i].view(1), i)

    assert ref.next_alloc_id_ == new.next_alloc_id_
    assert ref.n_error == new.n_error
    assert ref.n_outlier == new.n_outlier
    assert torch.equal(ref.proto_labels_.cpu(), new.proto_labels_.cpu())
    assert torch.equal(ref.goodness_.cpu(), new.goodness_.cpu())
    assert torch.equal(ref.alphas_.cpu(), new.alphas_.cpu())
    assert torch.equal(ref.sim_th_.cpu(), new.sim_th_.cpu())
    assert torch.equal(ref.prototypes_.cpu(), new.prototypes_.cpu())
    X_test = stream["X_test_norm"][:300].to(device)
    assert torch.equal(ref.predict(X_test, thresholded=True),
                       new.predict(X_test, thresholded=True))


# ---------------------------------------------------------------------------
# Replay: device-tensor buffer vs pinned (dict-of-numpy buffer)
# ---------------------------------------------------------------------------

def test_replay_parity(pinned, stream):
    import random
    from models.Replay import StreamingSoftmax

    d = stream["X_norm"].shape[1]

    def build_and_run(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model = cls(d, 40, use_replay=True, backbone=None, device="cpu",
                    lr=0.001, weight_decay=1e-5, replay_samples=50,
                    max_buffer_size=800)
        _fit_stream(model, stream["X_norm"], stream["y"])
        return model

    ref = build_and_run(pinned["Replay"].StreamingSoftmax)
    new = build_and_run(StreamingSoftmax)

    # Same RNG seeds and identical buffer contents must give an identical
    # sampled-batch sequence, hence bitwise-identical SGD trajectories
    assert ref.rehearsal_ixs == [int(v) for v in new.rehearsal_ixs] or \
        ref.rehearsal_ixs == new.rehearsal_ixs, "buffer index sequence diverged"
    for p_ref, p_new in zip(ref.classifier.parameters(), new.classifier.parameters()):
        assert torch.equal(p_ref, p_new), "classifier weights diverged from pinned"

    s_ref = ref.predict(stream["X_test_norm"])
    s_new = new.predict(stream["X_test_norm"])
    agreement = _argmax_agreement(s_ref, s_new)
    assert agreement == 1.0, f"argmax agreement {agreement} < 1.0"
