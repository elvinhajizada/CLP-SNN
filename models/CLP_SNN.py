"""
CLP-SNN: Continual Learning with Prototypes for Spiking Neural Networks
Code accompanying "Real-time Continual Learning on Intel Loihi 2" (under review).

Key properties:
- Self-normalizing local learning rule: Δw = α·r·(x − w·y)  where y = wᵀx
  The term −α·r·y·w provides implicit weight-norm regularization (no explicit renorm).
- Temporal Winner-Take-All: single winner k* = argmax(wₖᵀx) subject to y_{k*} > θ
- Goodness-based adaptive learning rate: α_i = 1/g_i (g only increments on r > 0)
- Three-factor rule: r ∈ {+1, −1}  (correct → +1, incorrect → −1)
- Only the winner prototype is updated per sample
- Supervised allocation: new prototype for unseen class OR when no winner exceeds θ

Learning rule has exactly two implementations, selected by `use_quantization`:
  * False → Float self-normalizing Taylor update (Variant B).
  * True  → Lava-faithful bit-approximate INT pipeline (Variant F).
            Mirrors `LearningRuleApplierBitApprox.apply()` 1:1: int32 MAC,
            per-product FLOOR right-shift, 15-bit saturation, single
            shared-scalar stochastic round per step. See
            `LearningConnectionModelBitApproximate` in
            `src/lava/magma/core/model/py/connection.py` and
            `LearningRuleApplierBitApprox.apply()` in
            `src/lava/magma/core/learning/learning_rule_applier.py`
            (lava repo) for the hardware spec; see
            `analysis/norm_demo.py::variant_F_step` for the numpy reference.

Optional extensions (via flags):
- Pseudo-labeling (use_pseudo_labels): Unlabeled prototypes get negative labels for novelty detection
- Voting (enable_voting): Multi-winner voting for spike outputs (hardware NSM behavior)
- Adaptive prototypes (adaptive_protos): Update winner on error vs. allocate new prototype
"""

import os
import numpy as np
import torch
import torch.nn as nn
import random
from typing import Optional, Tuple, List


# ──────────────────────────────────────────────────────────────────────────────
# SNN_Allocator: Manages prototype slot allocation logic
# ──────────────────────────────────────────────────────────────────────────────

class SNN_Allocator:
    """Manages prototype slot allocation and tracking.

    Handles:
    - Next free prototype slot tracking
    - Class discovery for supervised mode
    - Buffer fullness checks
    - Allocation decision logic based on multiple triggers
    """

    def __init__(self, n_protos: int):
        """Initialize allocator.

        Parameters
        ----------
        n_protos : int
            Maximum number of prototype slots available.
        """
        self.n_protos = n_protos
        self.next_alloc_id = 0
        self.classes_seen = set()

    def should_allocate_new_class(self, label: int) -> bool:
        """Check if label is a new (unseen) class.

        Parameters
        ----------
        label : int
            Input class label.

        Returns
        -------
        bool
            True if this is the first time seeing this label.
        """
        return label not in self.classes_seen

    def mark_class_seen(self, label: int) -> None:
        """Mark a class label as observed.

        Parameters
        ----------
        label : int
            Class label to mark as seen.
        """
        self.classes_seen.add(label)

    def can_allocate(self) -> bool:
        """Check if buffer has free slots.

        Returns
        -------
        bool
            True if next_alloc_id < n_protos.
        """
        return self.next_alloc_id < self.n_protos

    def allocate_slot(self, label: int) -> int:
        """Allocate and claim a prototype slot.

        Parameters
        ----------
        label : int
            Label to assign to allocated prototype.

        Returns
        -------
        int
            Slot ID if allocation succeeds, or -1 if buffer full.
        """
        if not self.can_allocate():
            return -1
        slot_id = self.next_alloc_id
        self.next_alloc_id += 1
        return slot_id

    def get_next_alloc_id(self) -> int:
        """Get current next allocation ID (read-only).

        Returns
        -------
        int
            Current value of next_alloc_id.
        """
        return self.next_alloc_id

    def get_allocated_count(self) -> int:
        """Get number of allocated slots.

        Returns
        -------
        int
            Same as next_alloc_id (number of prototypes in use).
        """
        return self.next_alloc_id

    def reset(self) -> None:
        """Reset allocator to initial state.

        Clears all tracking and returns to empty buffer.
        """
        self.next_alloc_id = 0
        self.classes_seen.clear()

    def get_state_dict(self) -> dict:
        """Get allocator state for serialization.

        Returns
        -------
        dict
            State dictionary with allocator parameters.
        """
        return {
            "next_alloc_id": self.next_alloc_id,
            "classes_seen": self.classes_seen.copy(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load allocator state from dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary (from get_state_dict).
        """
        self.next_alloc_id = state_dict["next_alloc_id"]
        self.classes_seen = state_dict["classes_seen"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# CLPSNN: Main class
# ──────────────────────────────────────────────────────────────────────────────


class CLPSNN(nn.Module):
    """
    CLP-SNN: Continual Learning with Prototypes for Spiking Neural Networks.

    Implements the algorithm from arXiv:2511.01553 in software, simulating the
    Loihi 2 neuromorphic hardware behaviour at the algorithmic level.

    Parameters
    ----------
    feature_size : int
        Dimensionality of (L2-normalised) input feature vectors.
    n_protos : int
        Number of pre-allocated prototype slots (paper default: 300).
    num_classes : int
        Number of output classes.
    threshold : float
        WTA activation threshold θ.  A prototype only wins if its dot-product
        similarity exceeds this value.  With zero-initialised weights the first
        sample of any class always falls below (dot = 0 ≤ 0), triggering
        allocation.  Default 0.0 matches the paper's threshold check y > θ.
    device : str
        Torch device string ('cuda' or 'cpu').
    """

    INT8_SCALE: float = 128.0   # S = 2^7, matches hardware trace scale
    W_WEIGHT_MAX: int = 127     # 8-bit signed weight clamp
    W_WEIGHT_MIN: int = -128
    # 15-bit accumulator saturation (Lava W_ACCUMULATOR_U = 15).
    # Matches numpy variant_F_step: _ACC_MAX = (1<<15)-1, _ACC_MIN = -(1<<15)-1.
    W_ACC_MAX: int = (1 << 15) - 1    # +32767
    W_ACC_MIN: int = -(1 << 15) - 1   # -32769

    def __init__(
        self,
        feature_size: int,
        n_protos: int = 300,
        num_classes: int = 40,
        threshold: float = 0.75,
        g_inc: float = 0.5,
        device: str = "cuda",
        use_pseudo_labels: bool = True,
        enable_voting: bool = False,
        voting_strategy: str = "majority",
        adaptive_protos: bool = True,
        use_quantization: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        self.feature_size = feature_size
        self.n_protos = n_protos
        self.num_classes = num_classes
        self.threshold = threshold
        self.g_inc = g_inc
        self.device = device

        # ── Feature flags (optional hardware-aligned behaviors) ───────────
        self.use_pseudo_labels = use_pseudo_labels    # Negative labels for unlabeled protos
        self.enable_voting = enable_voting            # Multi-winner voting support
        self.voting_strategy = voting_strategy        # 'majority' voting strategy
        self.adaptive_protos = adaptive_protos        # Update (True) vs allocate on error (False)
        self.use_quantization = use_quantization      # False = FP32 mode, skips INT8 everywhere

        # Shared-scalar stochastic round RNG (Lava-faithful).  One scalar draw
        # per integer update, shared across all weight dims.  Numpy Generator
        # gives us reproducible state that we can persist in save/load.
        self._lava_seed = int(seed)
        self._lava_rng = np.random.default_rng(self._lava_seed)

        # ── Prototype store ────────────────────────────────────────────────
        # Weights stored as float but kept at INT8 resolution after every
        # update.  Initialised to zero so the first sample always allocates.
        self.register_buffer(
            "prototypes", torch.zeros(n_protos, feature_size)
        )
        self.register_buffer(
            "proto_labels", torch.full((n_protos,), -1, dtype=torch.long)
        )
        # Goodness counter g_i — tracks cumulative correct predictions
        self.register_buffer("goodness", torch.ones(n_protos))
        # Per-prototype learning rate α_i = 1/max(g_i, 1)
        self.register_buffer("alphas", torch.ones(n_protos))

        # ── Allocator (manages prototype slot allocation) ────────────────
        self.allocator = SNN_Allocator(n_protos)

        # ── Voting state (for multi-winner scenarios) ──────────────────────
        self.last_winner_id: Optional[int] = None     # Track winner for pseudo-labels
        self.last_inferred_label: int = 0             # Track prediction for error feedback

        # ── Mutable Python state ───────────────────────────────────────────
        self.num_updates: int = 0

        # Backbone kept as None for interface compatibility with other models
        self.backbone = None

        self.to(device)

    # ──────────────────────────────────────────────────────────────────────
    # Quantisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Loihi input preprocessing: clip negatives → re-L2-normalize → round(x·128)/128.

        Mirrors the hardware pipeline exactly:
          1. Clip negatives to 0 (hardware ingests non-negative traces only)
          2. Re-L2-normalize (after clipping, norm changes → renormalize)
          3. Quantize to 7-bit resolution: round(x · 128) / 128

        Works for both 1-D (single sample) and 2-D (batch) tensors.
        """
        # x = x.clamp(min=0) # Removed claming to allow full range of input values, including negatives.
        ## Also commented out re-normalization as there is not clamping step that would change the norm, 
        ## and we want to preserve the original input distribution as much as possible for better performance.
        # if x.dim() == 1:
        #     norm = x.norm()
        #     if norm > 0:
        #         x = x / norm
        # else:
        #     norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        #     x = x / norms
        return (x * self.INT8_SCALE).round() / self.INT8_SCALE

    # ──────────────────────────────────────────────────────────────────────
    # Voting and pseudo-labeling helpers
    # ──────────────────────────────────────────────────────────────────────

    def _find_winners(self, sims: torch.Tensor) -> List[Tuple[int, float]]:
        """Find multiple winners (for voting scenarios).

        Returns all prototypes above threshold, sorted by similarity.

        Args:
            sims: (n_alloc,) similarity tensor.

        Returns:
            List of (idx, sim) tuples sorted in descending order, or empty list.
        """
        if sims.numel() == 0:
            return []
        
        # Get indices of prototypes above threshold
        above_threshold = sims > self.threshold
        if not above_threshold.any():
            return []
        
        # Get indices and similarities of above-threshold prototypes
        winner_indices = torch.where(above_threshold)[0]
        winner_sims = sims[winner_indices]
        
        # Sort by similarity (descending)
        sorted_sims, sorted_idx_in_winners = torch.sort(winner_sims, descending=True)
        sorted_indices = winner_indices[sorted_idx_in_winners]
        
        # Return as list of (idx, sim) tuples
        return [(idx.item(), sim.item()) for idx, sim in zip(sorted_indices, sorted_sims)]

    def _perform_voting(self, candidates: List[Tuple[int, float]]) -> int:
        """Perform majority voting on candidate prototypes.

        Args:
            candidates: List of (winner_idx, sim) tuples (sorted by sim).

        Returns:
            Index of winning prototype after voting, or -1 if no valid winner.
        """
        if not candidates:
            return -1
        
        if len(candidates) == 1:
            return candidates[0][0]
        
        # Get labels of all candidates
        winner_indices = [c[0] for c in candidates]
        voted_labels = self.proto_labels[winner_indices]
        
        # Filter out unlabeled prototypes (label == -1 or < 0 if pseudo-labeled)
        valid_mask = voted_labels > 0
        valid_labels = voted_labels[valid_mask]
        valid_indices = [winner_indices[i] for i, m in enumerate(valid_mask) if m]
        
        if len(valid_labels) == 0:
            # All unlabeled - pick highest similarity
            return candidates[0][0]
        
        # Count votes
        label_counts = torch.bincount(valid_labels)
        max_count = label_counts.max()
        
        # Find all labels with max count
        tied_labels = torch.where(label_counts == max_count)[0]
        
        if len(tied_labels) == 1:
            # Clear winner
            winning_label = tied_labels[0].item()
        else:
            # Tie-breaking: random selection among tied labels (matches hardware)
            winning_label = tied_labels[random.randint(0, len(tied_labels) - 1)].item()
        
        # Return one prototype with winning label (highest similarity among those)
        winning_proto_indices = [i for i, idx in enumerate(valid_indices) 
                                  if self.proto_labels[idx].item() == winning_label]
        if winning_proto_indices:
            # Return the highest similarity prototype with winning label
            best_idx_in_valid = min(winning_proto_indices, 
                                     key=lambda i: -candidates[sum(1 for m in valid_mask[:winner_indices.index(valid_indices[i])] if m)][1])
            return valid_indices[winning_proto_indices[0]]
        
        return candidates[0][0]

    def _get_pseudo_label(self, proto_idx: int) -> int:
        """Generate a pseudo-label for an unlabeled prototype.

        Matches Loihi NSM behavior: pseudo_label = -(proto_idx + 1)

        Args:
            proto_idx: Prototype index.

        Returns:
            Pseudo-label (negative integer).
        """
        return -1 * (proto_idx + 1)

    def _assign_pseudo_label(self, proto_idx: int) -> None:
        """Assign a pseudo-label to an unlabeled prototype.

        Args:
            proto_idx: Index of prototype to label.
        """
        if self.proto_labels[proto_idx] < 0:  # Currently unlabeled
            self.proto_labels[proto_idx] = self._get_pseudo_label(proto_idx)

    # ──────────────────────────────────────────────────────────────────────
    # Core algorithmic primitives
    # ──────────────────────────────────────────────────────────────────────

    def _compute_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Dot-product similarities between x and all allocated prototypes.

        Args:
            x: (input_dim,) L2-normalised, INT8-quantised input vector.

        Returns:
            sims: (next_alloc,) float tensor, or empty if nothing allocated.
        """
        n_alloc = self.allocator.get_allocated_count()
        if n_alloc == 0:
            return torch.empty(0, device=self.device)
        W = self.prototypes[:n_alloc]  # (n_alloc, d)
        return W @ x  # (n_alloc,)

    def _find_winner(self, sims: torch.Tensor) -> Tuple[int, float]:
        """Temporal WTA: return index of highest-similarity prototype if > θ.

        Simulates Loihi 2 lateral inhibition — the prototype that would fire
        first (highest membrane potential) suppresses all others.

        Returns:
            (winner_idx, winner_sim) or (-1, -inf) when no prototype wins.
        """
        if sims.numel() == 0:
            return -1, float("-inf")
        winner_sim, winner_idx = sims.max(dim=0)
        if winner_sim.item() <= self.threshold:
            return -1, winner_sim.item()
        return winner_idx.item(), winner_sim.item()

    def _allocate_prototype(self, x: torch.Tensor, label: int) -> int:
        """Initialise a new prototype slot with input x and given label.

        The weight vector is set to x (already INT8-quantised) so the
        prototype immediately represents the new class/instance.
        Goodness and α are reset to 1 for the fresh slot.

        Args:
            x: (input_dim,) quantised input vector.
            label: Integer class label (positive).

        Returns:
            Allocated slot ID, or -1 if buffer is full.
        """
        slot_id = self.allocator.allocate_slot(label)
        if slot_id < 0:
            return -1  # buffer full — silently drop (matches Loihi fixed memory)
        
        self.prototypes[slot_id] = x  # x is already quantised
        self.proto_labels[slot_id] = label
        self.goodness[slot_id] = 1.0
        self.alphas[slot_id] = 1.0
        
        return slot_id

    def _update_winner(self, winner_idx: int, x: torch.Tensor, r: float) -> None:
        """Apply the self-normalising local learning rule to the winner.

        Rule (Eq. 1–3 / 6 in arXiv:2511.01553):
            Δw  = α · r · (x − w · y)   where y = wᵀx

        Dispatches to either the float path (Variant B) or the Lava-faithful
        integer path (Variant F) depending on `use_quantization`.  Goodness
        g_i and learning rate α_i are updated only on r > 0 (Eq. 9–11).
        """
        if self.use_quantization:
            self._update_winner_int_F(winner_idx, x, r)
        else:
            self._update_winner_float_B(winner_idx, x, r)

        # Update goodness and decay learning rate only on positive reward
        if r > 0:
            self.goodness[winner_idx] += self.g_inc
            self.alphas[winner_idx] = 1.0 / max(self.goodness[winner_idx].item(), 1.0)  # Prevent division by zero

        if r < 0:
            self.goodness[winner_idx] -= self.g_inc
            self.alphas[winner_idx] = 1.0 / max(self.goodness[winner_idx].item(), 1.0)  # Prevent division by zero

    def _update_winner_float_B(self, winner_idx: int, x: torch.Tensor, r: float) -> None:
        """Variant B — FP32 self-normalising Taylor update.

        Δw = α · r · (x − w · y),  y = wᵀx
        The −α·r·y·w term is a first-order Taylor expansion of explicit L2
        renorm, keeping ‖w‖ ≈ 1 implicitly without division.
        """
        w = self.prototypes[winner_idx]
        alpha = self.alphas[winner_idx].item()
        y = (w * x).sum().item()
        self.prototypes[winner_idx] = w + alpha * r * (x - w * y)

    def _update_winner_int_F(self, winner_idx: int, x: torch.Tensor, r: float) -> None:
        """Variant F — Lava-faithful bit-approximate integer update.

        Ported from analysis/norm_demo.py::variant_F_step.
        Mirrors Lava's LearningRuleApplierBitApprox.apply() 1:1; see
        LearningConnectionModelBitApproximate in connection.py and
        LearningRuleApplierBitApprox.apply() in learning_rule_applier.py
        (both in src/lava/magma/core/ in the lava repo) for the spec.

        Pipeline:
            result ← w_int << 7                         # lift into 15-bit accumulator
            P1:  mac = α · x_int                        # int32 MAC (11-bit), shift_r = 0
                 result += r · mac                      # saturating add
            P2:  mac = −α · y_int · w_int               # int32 MAC (20-bit)
                 mac >>= 7                              # Lava shift_r=5 + user +2 compensation
                                                         #  (matches 2^-9 paper scale)
                 result += r · mac                      # saturating add
            Single shared-scalar stochastic round (one rng.random() per call),
            then >> 7 back to 8-bit stored weight.

        Alpha packing note: α_int is packed into Lava's s_mantissa slot, which
        hardware restricts to [-8, 7].  Our sim uses a wider range (up to 127)
        as a sweepable knob — a known, intentional divergence from strict HW.
        """
        S = int(self.INT8_SCALE)  # 128
        g = (self.goodness[winner_idx].item())

        # 7-bit fixed-point learning rate (α = 1/g at scale S)
        alpha_int = max(0, min(127, round(S / max(g, 1.0))))
        if alpha_int == 0:
            # α underflows to 0 (g ≥ 256) → prototype frozen
            return

        r_int = int(r)
        w = self.prototypes[winner_idx]
        device = w.device

        # Snapshot to int32 (8-bit signed / 7-bit traces)
        w_int = (w * S).round().to(torch.int32)
        x_int = (x * S).round().to(torch.int32)

        # Forward: y = w·x quantised to 7-bit trace (shared-bias +64 round-half-up
        # on the lift to match variant_F_step's (y_acc + 64) >> 7).
        y_acc = int((w_int * x_int).sum().item())
        y_int = int(max(-128, min(127, (y_acc + 64) >> 7)))

        # Init 15-bit accumulator: w lifted by 7 bits
        result = torch.bitwise_left_shift(w_int, 7)

        acc_max = int(self.W_ACC_MAX)
        acc_min = int(self.W_ACC_MIN)

        # ---- Product 1:  +α · x1  ---------------------------------------
        # factor_width_sum = 7 + W_S_MANT(4) = 11;  shift_r = max(0, 11−15) = 0
        mac1 = torch.clamp(x_int, -128, 127)
        mac1 = mac1 * torch.tensor(alpha_int, dtype=torch.int32, device=device)
        mac1 = mac1 * torch.tensor(r_int, dtype=torch.int32, device=device)
        mac1 = torch.clamp(mac1, acc_min, acc_max)
        result = torch.clamp(result + mac1, acc_min, acc_max)

        # ---- Product 2:  −α · y1 · w  -----------------------------------
        # factor_width_sum = 7 + 9 + 4 = 20;  Lava shift_r = 5.
        # +2 compensating shift (user-level 2^-9) → total 7, so effective scale
        # matches P1's 2^-7.  See LearningConnectionModelBitApproximate (lava repo).
        y_scalar = torch.tensor(
            max(-128, min(127, y_int)), dtype=torch.int32, device=device
        )
        mac2 = torch.clamp(w_int, -512, 511) * y_scalar
        mac2 = mac2 * torch.tensor(-alpha_int, dtype=torch.int32, device=device)
        # FLOOR arithmetic right-shift (matches numpy np.right_shift on int32,
        # i.e. torch.bitwise_right_shift which is arithmetic for signed dtypes).
        mac2 = torch.bitwise_right_shift(mac2, 7)
        mac2 = mac2 * torch.tensor(r_int, dtype=torch.int32, device=device)
        mac2 = torch.clamp(mac2, acc_min, acc_max)
        result = torch.clamp(result + mac2, acc_min, acc_max)

        # ---- Single shared-scalar stochastic round (15 → 8 bits) --------
        rnd = float(self._lava_rng.random())
        # Floor toward −∞ (torch.div with rounding_mode='floor')
        integer = torch.div(result, 128, rounding_mode="floor").to(torch.int32)
        frac = (result - integer * 128).to(torch.float64) / 128.0
        w_new = integer + (frac > rnd).to(torch.int32)
        w_new = torch.clamp(w_new, self.W_WEIGHT_MIN, self.W_WEIGHT_MAX)

        self.prototypes[winner_idx] = w_new.to(w.dtype) / S

    # ──────────────────────────────────────────────────────────────────────
    # Public interface — matches other models in the repo
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, i: int = 0) -> None:
        """Update model with a single labelled sample.

        Behavior is controlled by flags:
        - use_pseudo_labels (True): Unlabeled prototypes get pseudo-labels for novelty detection
        - enable_voting (True): Multi-winner voting on spike outputs
        - adaptive_protos (True): Update winner on error; (False): Allocate new proto on error

        Args:
            x: (input_dim,) or (1, input_dim) feature vector, L2-normalised.
            y: scalar or (1,) integer class label.
            i: sample index (unused internally, kept for interface parity).
        """
        x = x.to(self.device).float().view(self.feature_size)
        y_int = int(y.item())

        # Quantise input: clip negatives → re-L2-normalize → round(x·128)/128
        x_q = self._quantize_input(x) if self.use_quantization else x

        # ── Supervised allocation for unseen classes ───────────────────────
        # First occurrence of a class label always allocates a fresh prototype
        # (supervised mode extension of novelty-detection rule)
        if self.allocator.should_allocate_new_class(y_int):
            self.allocator.mark_class_seen(y_int)
            self._allocate_prototype(x_q, y_int)
            self.num_updates += 1
            return

        # ── Compute similarities and find winner(s) ────────────────────────
        sims = self._compute_similarities(x_q)

        if self.enable_voting:
            # Multi-winner voting scenario
            candidates = self._find_winners(sims)
            if not candidates:
                # No prototype above threshold → allocate new one
                self._allocate_prototype(x_q, y_int)
                self.num_updates += 1
                return
            
            # Perform voting to select representative
            winner_idx = self._perform_voting(candidates)
        else:
            # Single Winner-Take-All
            winner_idx, winner_sim = self._find_winner(sims)

        # ── Handle no-winner case ────────────────────────────────────────
        if winner_idx < 0:
            # No prototype above threshold → novelty → allocate new prototype
            self._allocate_prototype(x_q, y_int)
            self.num_updates += 1
            return

        # ── Winner found: handle pseudo-labeling and updates ──────────────
        self.last_winner_id = winner_idx
        predicted_label = self.proto_labels[winner_idx].item()

        # Handle pseudo-labeling: if winner has pseudo-label, assign real label
        if self.use_pseudo_labels and predicted_label < 0:
            # Pseudo-label upgrade: assign the user-provided label
            self.proto_labels[winner_idx] = y_int
            self.last_inferred_label = y_int
            self.num_updates += 1
            return

        # ── Determine correctness and update strategy ────────────────────
        is_correct = (predicted_label == y_int)
        self.last_inferred_label = predicted_label

        if is_correct:
            # Correct prediction
            if self.adaptive_protos:
                # Update with positive reward
                self._update_winner(winner_idx, x_q, r=1.0)
        else:
            # Incorrect prediction / mismatch
            if self.adaptive_protos:
                # Update with negative reward (self-normalizing learning)
                self._update_winner(winner_idx, x_q, r=-1.0)
            else:
                # Allocation-heavy strategy: allocate new prototype instead
                self._allocate_prototype(x_q, y_int)

        # ── Pseudo-label assignment on first encounter ───────────────────
        if self.use_pseudo_labels and predicted_label < 0:
            self._assign_pseudo_label(winner_idx)

        self.num_updates += 1

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Compute class similarity scores for a batch of inputs.

        For each class, the score is the maximum dot-product similarity over
        all of that class's allocated prototypes.

        Args:
            X: (batch_size, input_dim) float tensor, L2-normalised.

        Returns:
            scores: (batch_size, num_classes) float tensor on CPU.
        """
        X = X.to(self.device).float()
        batch_size = X.shape[0]

        n_alloc = self.allocator.get_allocated_count()
        if n_alloc == 0:
            return torch.zeros(batch_size, self.num_classes)

        W = self.prototypes[:n_alloc]        # (n_alloc, d)
        labels = self.proto_labels[:n_alloc] # (n_alloc,)

        # Quantise inputs and compute all similarities in one matmul
        X_q = self._quantize_input(X) if self.use_quantization else X  # (batch, d)
        sims = X_q @ W.T                              # (batch, n_alloc)

        # Aggregate: max similarity per class
        scores = torch.full(
            (batch_size, self.num_classes), float("-inf"), device=self.device
        )
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                scores[:, c] = sims[:, mask].max(dim=1).values

        # Replace −inf (unseen classes) with 0
        scores = torch.where(
            scores == float("-inf"), torch.zeros_like(scores), scores
        )
        return scores.cpu()

    @torch.no_grad()
    def evaluate_(self, test_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate accuracy on a DataLoader.

        Returns:
            probas: (N, num_classes) scores
            labels: (N,) ground-truth labels
        """
        all_probas = []
        all_labels = []
        for batch in test_loader:
            batch_x, batch_y = batch[0], batch[1]
            probas = self.predict(batch_x)
            all_probas.append(probas)
            all_labels.append(batch_y)
        return torch.cat(all_probas), torch.cat(all_labels)

    def get_num_prototypes_used(self) -> int:
        """Return the number of allocated prototype slots."""
        return self.allocator.get_allocated_count()

    def save_model(self, save_path: str, save_name: str) -> None:
        """Save prototype store and auxiliary state."""
        d = {
            "prototypes": self.prototypes.cpu(),
            "proto_labels": self.proto_labels.cpu(),
            "goodness": self.goodness.cpu(),
            "alphas": self.alphas.cpu(),
            "allocator_state": self.allocator.get_state_dict(),
            "use_pseudo_labels": self.use_pseudo_labels,
            "enable_voting": self.enable_voting,
            "voting_strategy": self.voting_strategy,
            "adaptive_protos": self.adaptive_protos,
            "use_quantization": self.use_quantization,
            "lava_rng_state": self._lava_rng.bit_generator.state,
            "lava_seed": self._lava_seed,
        }
        torch.save(d, os.path.join(save_path, save_name + ".pth"))

    def load_model(self, save_file: str) -> None:
        """Load prototype store and auxiliary state."""
        d = torch.load(save_file, map_location=self.device, weights_only=False)
        self.prototypes = d["prototypes"].to(self.device)
        self.proto_labels = d["proto_labels"].to(self.device)
        self.goodness = d["goodness"].to(self.device)
        self.alphas = d["alphas"].to(self.device)
        self.allocator.load_state_dict(d["allocator_state"])
        # Load flags if they exist in checkpoint (backward compatibility)
        self.use_pseudo_labels = d.get("use_pseudo_labels", self.use_pseudo_labels)
        self.enable_voting = d.get("enable_voting", self.enable_voting)
        self.voting_strategy = d.get("voting_strategy", self.voting_strategy)
        self.adaptive_protos = d.get("adaptive_protos", self.adaptive_protos)
        self.use_quantization = d.get("use_quantization", self.use_quantization)
        if "lava_seed" in d:
            self._lava_seed = int(d["lava_seed"])
        if "lava_rng_state" in d:
            self._lava_rng.bit_generator.state = d["lava_rng_state"]
