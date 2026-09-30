"""
CLP-SNN Loihi 2 deployment emulator (allocation-only).

Reproduces, in integer arithmetic, what the CLP-SNN network deployed on Loihi 2
computes for each sample:

Input       negative features are clipped, the vector is L2-normalised and
            rounded to 7 bits, x_int = round(128 x)
Imprint     a new prototype stores W = x_int and is not updated afterwards
Similarity  s_k = <W_k, x_int> / 128^2, the cosine similarity at 7-bit precision
Winner      prototypes spike earlier the higher their similarity; one time step
            spans `resolution` of similarity, so prototypes whose similarities
            fall in the same step spike together and vote on the label (random
            choice among tied labels). A winner must spike within `t_wait`
            steps, i.e. have s >= `threshold`.
Training    no winner (novelty) or an incorrect vote allocates a new prototype
Testing     no allocation; the prediction is the voted label of the first step

Options switch individual chip features off (``preprocess="sim"``,
``winner="argmax"``) to measure each one's effect on accuracy
(experiments/clp_snn_gap_decomposition.py).
"""

import random

import numpy as np
import torch


class CLPSNNLoihi:
    """Allocation-only CLP-SNN as deployed on Loihi 2.

    Parameters
    ----------
    feature_size : int
        Input dimension.
    num_classes : int
        Number of classes (labels 0 .. num_classes-1).
    threshold : float
        Similarity a prototype needs to win during training (cosine units).
    resolution : float
        Similarity spanned by one time step of spike-timing winner selection.
    t_wait : int
        Time steps a winner is given; prototypes above
        threshold + (t_wait - 1) * resolution all spike in the first step.
    n_protos : int
        Prototype pool size.
    preprocess : {"chip", "sim"}
        "chip" clips negative features before normalising; "sim" normalises
        and quantises without clipping, like CLPSNN.
    winner : {"spike", "argmax"}
        "spike" resolves the winner in time steps with label voting;
        "argmax" picks the most similar prototype.
    seed : int
        Seed for the readout's random tie-breaking.

    threshold and resolution are rounded to the 1/128^2 grid of the 7-bit
    similarity.
    """

    SCALE = 128

    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        threshold: float,
        resolution: float,
        t_wait: int,
        n_protos: int,
        preprocess: str = "chip",
        winner: str = "spike",
        seed: int = 0,
    ) -> None:
        assert preprocess in ("chip", "sim") and winner in ("spike", "argmax")
        self.feature_size = feature_size
        self.num_classes = num_classes
        q = self.SCALE ** 2
        self.th = round(threshold * q)       # integer similarity <W, x_int>
        self.step = round(resolution * q)
        self.t_wait = t_wait
        self.n_protos = n_protos
        self.preprocess = preprocess
        self.winner = winner
        self.rng = random.Random(seed)

        self.W = np.zeros((n_protos, feature_size))
        self.labels = np.full(n_protos, -1, dtype=np.int64)
        self.n_alloc = 0

    # ── Chip arithmetic ────────────────────────────────────────────────────

    def _quantize(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.preprocess == "chip":
            X = np.maximum(X, 0)
        X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
        return np.round(X * self.SCALE)  # half-to-even, as numpy on the host

    def _decide(self, sims: np.ndarray, training: bool) -> int:
        """Label chosen for one sample from the integer similarities.

        Returns -1 during training when no prototype reaches the threshold.
        """
        labels = self.labels[: self.n_alloc]
        if self.winner == "argmax":
            i = int(np.argmax(sims))
            return int(labels[i]) if (not training or sims[i] >= self.th) else -1
        # Spike time step: 0 for the most similar, one step per `resolution`
        k = np.maximum(0, self.t_wait - 1 - np.floor((sims - self.th) / self.step))
        kmin = k.min()
        if training and kmin > self.t_wait - 1:
            return -1
        idx = np.flatnonzero(k == kmin)
        counts = np.bincount(labels[idx])
        cands = np.flatnonzero(counts == counts.max())
        return int(cands[0]) if cands.size == 1 else int(self.rng.choice(list(cands)))

    # ── Public interface (matches CLPSNN) ──────────────────────────────────

    def fit(self, x: torch.Tensor, y: torch.Tensor, i: int = 0) -> None:
        """Present one labelled training sample."""
        x_q = self._quantize(torch.as_tensor(x).cpu().numpy().reshape(1, -1))[0]
        label = int(torch.as_tensor(y).item())
        pred = -1
        if self.n_alloc > 0:
            sims = self.W[: self.n_alloc] @ x_q
            pred = self._decide(sims, training=True)
        # Novelty (no winner) or an incorrect vote allocates a new prototype
        if pred != label and self.n_alloc < self.n_protos:
            self.W[self.n_alloc] = x_q
            self.labels[self.n_alloc] = label
            self.n_alloc += 1

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """One-hot scores (batch, num_classes) of the chip's decision."""
        X_q = self._quantize(torch.as_tensor(X).cpu().numpy())
        out = torch.zeros(X_q.shape[0], self.num_classes)
        if self.n_alloc == 0:
            return out
        S = X_q @ self.W[: self.n_alloc].T
        for r, sims in enumerate(S):
            out[r, self._decide(sims, training=False)] = 1
        return out

    def get_num_prototypes_used(self) -> int:
        return self.n_alloc
