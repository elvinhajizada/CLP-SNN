"""
variants.py
-----------
CLPSNNAdaptive: CLPSNN with opt-in knobs for the adaptive-prototype
experiments (see FINDINGS.md). It implements the learning update itself, so
with every knob at its default it reproduces the PRE-FIX released model
(models/CLP_SNN.py at commit 3a5e9c7) bit for bit; that is the baseline of
the study (check_default.py, step 0). The fixed released model equals
alpha_after_inc=True, g_floor=1.0, neg_update="decay" (check_fixed.py).

Knobs
  alpha_after_inc  H1a  update goodness first, then use alpha = 1/max(g, 1)
                        of the NEW goodness for this update (intended design:
                        allocation = alpha 1, first learning step = alpha 1/2
                        with g_inc_pos = 1). Default False = released code,
                        which uses the alpha stored by the previous event.
  g_inc_pos        H1b  goodness increment on a hit (default: g_inc)
  g_inc_neg        H2b  goodness decrement on a miss (default: g_inc; 0 = keep)
  g_floor          H2a  lower bound on goodness after each event (None = none)
  neg_alpha_shift  H2c  negative-reward step uses alpha / 2^k (0 = off)
  oracle_renorm    H2d  explicit L2 renorm after each float update (sim-only
                        upper bound, not Loihi-realizable)
  neg_update       H2e/f  weight change on a miss: 'oja' (released,
                        dw = -a(x - y w), unit sphere repelling), 'none'
                        (adapt on hits only), 'decay' (dw = a(r x - y w):
                        reward gates only the Hebbian product, so the
                        normalising term contracts for both reward signs)
  alloc_on_miss    H3   on a miss, also allocate the sample (CLP learn_outliers)
  alpha_int_max    H4a  clamp for the integer learning-rate mantissa
                        (released sim: 127; Loihi s_mantissa: 7)
  w_scale_shift    H4b  store weights at scale 128 * 2^k (finer 8-bit grid,
                        max |w_i| = 127 / (128 * 2^k)); the Hebbian product
                        carries a 2^k factor so the rule is unchanged in real
                        units (Lava rule strings allow 2^k factors, e.g.
                        dw = "2^2*x1*y0" in the Loihi CLP tutorial)

Diagnostics (off by default, do not change the learning dynamics)
  trace            list collecting one tuple per learning update:
                   (step, winner, r, alpha_used, g_before, norm_before, norm_after)
  track_means      keep per-prototype sums of absorbed samples (allocation
                   sample + hit samples) to measure drift from the sample mean
  shadow           float shadow copy of every prototype that receives the same
                   events with the float rule (H4: integer vs float drift)
"""

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.CLP_SNN import CLPSNN  # noqa: E402


class CLPSNNAdaptive(CLPSNN):

    def __init__(self, *args,
                 alpha_after_inc: bool = False,
                 g_inc_pos=None,
                 g_inc_neg=None,
                 g_floor=None,
                 neg_alpha_shift: int = 0,
                 oracle_renorm: bool = False,
                 neg_update: str = "oja",
                 alloc_on_miss: bool = False,
                 alpha_int_max: int = 127,
                 w_scale_shift: int = 0,
                 trace=None,
                 track_means: bool = False,
                 shadow: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_after_inc = alpha_after_inc
        self.g_inc_pos = self.g_inc if g_inc_pos is None else g_inc_pos
        self.g_inc_neg = self.g_inc if g_inc_neg is None else g_inc_neg
        self.g_floor = g_floor
        self.neg_alpha_shift = int(neg_alpha_shift)
        self.oracle_renorm = oracle_renorm
        assert neg_update in ("oja", "none", "decay")
        self.neg_update = neg_update
        self.alloc_on_miss = alloc_on_miss
        self.alpha_int_max = int(alpha_int_max)
        self.w_scale_shift = int(w_scale_shift)

        self.trace = trace
        self.track_means = track_means
        self.shadow = shadow
        if track_means:
            self.absorb_sum = torch.zeros_like(self.prototypes)
        if shadow:
            self.shadow_protos = torch.zeros_like(self.prototypes)

        self._cur_y = None
        self._pending_alloc = False
        self.n_miss_alloc = 0

    # ── Allocation ──────────────────────────────────────────────────────────

    def _allocate_prototype(self, x, label):
        slot = super()._allocate_prototype(x, label)
        if slot >= 0:
            if self.track_means:
                self.absorb_sum[slot] = x
            if self.shadow:
                self.shadow_protos[slot] = x
        return slot

    # ── Learning update ─────────────────────────────────────────────────────

    def _update_winner(self, winner_idx, x, r):
        g_before = self.goodness[winner_idx].item()
        step = self.g_inc_pos if r > 0 else -self.g_inc_neg
        g_after = g_before + step
        if self.g_floor is not None:
            g_after = max(g_after, self.g_floor)
        g_eff = g_after if self.alpha_after_inc else g_before

        if self.alpha_after_inc:
            # float32 rounding as the stored alpha buffer would give
            alpha_f = float(np.float32(1.0 / max(g_eff, 1.0)))
        else:
            alpha_f = self.alphas[winner_idx].item()
        if r < 0 and self.neg_alpha_shift:
            alpha_f = alpha_f / (1 << self.neg_alpha_shift)

        w = self.prototypes[winner_idx]
        norm_before = w.norm().item() if self.trace is not None else 0.0

        skip = r < 0 and self.neg_update == "none"
        r_decay = 1.0 if (r < 0 and self.neg_update == "decay") else r
        if skip:
            alpha_used = 0.0
        elif self.use_quantization:
            S = int(self.INT8_SCALE)
            alpha_int = max(0, min(self.alpha_int_max, round(S / max(g_eff, 1.0))))
            if r < 0 and self.neg_alpha_shift:
                alpha_int = alpha_int >> self.neg_alpha_shift
            self._int_step(winner_idx, x, r, alpha_int, r_decay)
            alpha_used = alpha_int / S
        else:
            y = (w * x).sum().item()
            if r_decay == r:
                self.prototypes[winner_idx] = w + alpha_f * r * (x - w * y)
            else:
                self.prototypes[winner_idx] = w + alpha_f * (r * x - w * y)
            if self.oracle_renorm:
                wn = self.prototypes[winner_idx]
                self.prototypes[winner_idx] = wn / wn.norm().clamp(min=1e-12)
            alpha_used = alpha_f

        if self.shadow and not skip:
            ws = self.shadow_protos[winner_idx]
            ys = (ws * x).sum().item()
            self.shadow_protos[winner_idx] = ws + alpha_used * (r * x - r_decay * ws * ys)
        if self.track_means and r > 0:
            self.absorb_sum[winner_idx] += x

        # goodness / stored alpha bookkeeping (same float32 buffers as base)
        if self.g_floor is None and not self.alpha_after_inc \
                and self.g_inc_pos == self.g_inc and self.g_inc_neg == self.g_inc:
            # released arithmetic, kept literally for bit-exactness
            if r > 0:
                self.goodness[winner_idx] += self.g_inc
            else:
                self.goodness[winner_idx] -= self.g_inc
            self.alphas[winner_idx] = 1.0 / max(self.goodness[winner_idx].item(), 1.0)
        else:
            self.goodness[winner_idx] = g_after
            self.alphas[winner_idx] = 1.0 / max(g_after, 1.0)

        if self.trace is not None:
            self.trace.append((self.num_updates, winner_idx, r, alpha_used,
                               g_before, norm_before,
                               self.prototypes[winner_idx].norm().item()))

        if r < 0 and self.alloc_on_miss:
            self._pending_alloc = True

    def _int_step(self, winner_idx, x, r, alpha_int, r_decay=None):
        """Body of the pre-fix CLPSNN._update_winner_int_F (now _update_winner_int8)
        with alpha_int passed in;
        r_decay is the reward factor on the -a*y*w product (default r)."""
        if alpha_int == 0:
            return
        S = int(self.INT8_SCALE)
        r_int = int(r)
        w = self.prototypes[winner_idx]
        device = w.device

        k = self.w_scale_shift
        Sw = S << k
        w_int = (w * Sw).round().to(torch.int32)
        x_int = (x * S).round().to(torch.int32)
        y_acc = int((w_int * x_int).sum().item())
        y_int = int(max(-128, min(127, (y_acc + (64 << k)) >> (7 + k))))

        result = torch.bitwise_left_shift(w_int, 7)
        acc_max = int(self.W_ACC_MAX)
        acc_min = int(self.W_ACC_MIN)

        mac1 = torch.clamp(x_int, -128, 127)
        mac1 = mac1 * torch.tensor(alpha_int, dtype=torch.int32, device=device)
        mac1 = mac1 * torch.tensor(r_int, dtype=torch.int32, device=device)
        if k:
            mac1 = torch.bitwise_left_shift(mac1, k)
        mac1 = torch.clamp(mac1, acc_min, acc_max)
        result = torch.clamp(result + mac1, acc_min, acc_max)

        y_scalar = torch.tensor(
            max(-128, min(127, y_int)), dtype=torch.int32, device=device
        )
        mac2 = torch.clamp(w_int, -512, 511) * y_scalar
        mac2 = mac2 * torch.tensor(-alpha_int, dtype=torch.int32, device=device)
        mac2 = torch.bitwise_right_shift(mac2, 7)
        r2 = r_int if r_decay is None else int(r_decay)
        mac2 = mac2 * torch.tensor(r2, dtype=torch.int32, device=device)
        mac2 = torch.clamp(mac2, acc_min, acc_max)
        result = torch.clamp(result + mac2, acc_min, acc_max)

        rnd = float(self._lava_rng.random())
        integer = torch.div(result, 128, rounding_mode="floor").to(torch.int32)
        frac = (result - integer * 128).to(torch.float64) / 128.0
        w_new = integer + (frac > rnd).to(torch.int32)
        w_new = torch.clamp(w_new, self.W_WEIGHT_MIN, self.W_WEIGHT_MAX)
        self.prototypes[winner_idx] = w_new.to(w.dtype) / Sw

    # ── Fit wrapper (H3: allocate the missed sample) ────────────────────────

    @torch.no_grad()
    def fit(self, x, y, i=0):
        self._pending_alloc = False
        super().fit(x, y, i)
        if self._pending_alloc:
            xv = x.to(self.device).float().view(self.feature_size)
            x_q = self._quantize_input(xv) if self.use_quantization else xv
            if self._allocate_prototype(x_q, int(y.item())) >= 0:
                self.n_miss_alloc += 1
            self._pending_alloc = False
