# Adaptive prototypes in CLP-SNN simulation: findings

Side research, not for the manuscript unless decided otherwise. Plan: `~/.claude/plans/i-want-you-to-tingly-salamander.md`.
Constraint: fixes must be Loihi-realizable; `oracle_renorm` is a sim-only upper bound.

Protocol: decisions on seed 10 plus the known collapse cases; seeds 20/30 over the full grid are confirmation only.
Success (fixed in advance):
- FP32 adaptive within 1 pp of CLP at every threshold, both settings; no seed more than 3 pp below CLP; std at most 2x CLP's.
- INT8 adaptive within 2 pp of INT8-NA at th ≥ 0.75; no collapse anywhere.

## Predictions (written before the runs)

| id | hypothesis | prediction |
|---|---|---|
| H1 | Stored-α off-by-one: the first hit after allocation runs at α = 1 (not 1/2), and `g_inc = 0.5` slows the decay further | Trace shows α = 1 on each prototype's first hit. H1a+H1b (intended 1, 1/2, 1/3, …) closes most of the 1-shot FP32 gap (−1 to −2 pp) and lowers mean ‖w‖ |
| H2 | Negative branch at α = 1 is norm-expanding; a miss lowers g, which keeps α at 1 → runaway → one prototype wins all | In collapse cases max ‖w‖ > 1.5 and the top-prototype share → 1. H2d (oracle renorm), H2b (no decrement) and H2c (halved negative step) remove the collapse; H2a (floor g ≥ 1) alone does not |
| H3 | Adaptive mode discards the missed sample | Allocate-on-miss raises the 1-shot prototype count toward CLP's and reaches CLP parity |
| H4 | INT8 averaging lives near the rounding floor (x_int ≈ 3-4 LSB) | After the FP32 fix, INT8 still trails FP32; the shadow-float cosine drops as α_int shrinks |

## Step 0: reproduction

- `check_default.py`: `CLPSNNAdaptive` at default knobs is bit-exact with `models.CLP_SNN.CLPSNN` (prototypes, labels, goodness, alphas; FP32 and INT8; 1-shot seed 10 full stream, 25-shot seed 20 first 6,000 samples).
- `run_ablation.py --configs base` reproduces the cached `clp_snn_hw_comparison/summary.csv` rows to every digit: 1-shot seed 10 FP32 52.8 / 54.1, INT8 46.7 / 51.3 (th 0.70 / 0.85); 25-shot seed 10 FP32 91.8 / 93.2, INT8 5.1 / 90.5; collapse cases 25-shot FP32 th 0.80 s10 = 2.5, th 0.70 s20 = 2.5, 1-shot INT8 th 0.70 s30 = 6.8.
- Cost: 0.1 ms (FP32) / 0.2 ms (INT8) per sample; a 25-shot run is 10-80 s.

## Step 1: diagnosis

Files: `results/adaptive_protos/diag/*.npz|png` (`diagnose.py`).

**H1 supported.** On every prototype the first hit after allocation runs at α = 1 (FP32: 100% of prototypes; INT8: α_int = 127/128). The first hit takes ‖w‖ from 1.00 to about 1.12. Healthy runs end with mean ‖w‖ ≈ 1.01 (p95 1.02-1.03) and cos(w, mean of absorbed samples) ≈ 0.99.

**H2 supported, with a sharper mechanism than predicted.**
- FP32 collapse (25-shot th 0.80 s10): prototype 29 gets its first hit at α = 1 (‖w‖ 1.12), then a run of misses. Each miss lowers g (1.5 → 1 → 0.5 → 0 → −0.5 → …), so α stays at 1, and ‖w‖ goes 1.31 → 1.95 → 3.35 → 5.92 → 11.7 → 26.3 → 75.2 → … → inf/NaN. That prototype then wins 99% of all later updates, and the test accuracy is chance (2.5%). The th 0.70 s20 case is identical (prototype 485).
- INT8 collapse (1-shot th 0.70 s30) happens at **small α (≈ 0.05)**. Quantisation noise on hits lifts ‖w‖ to about 1.05. A run of about 15 consecutive misses then drifts it to 1.9, after which it wins 79% of later updates.
- Mechanism: to first order the norm change of the Oja step is `Δn ≈ −r·α·cosθ·(n² − 1)`. For r = +1 the unit sphere is attracting; **for r = −1 it is repelling at any α**. Large α (H1 plus unbounded goodness decrement) makes the runaway explosive, but a small α only slows it. Prediction revised before step 2: H2b/H2c reduce but may not remove collapse, especially in INT8.
- Post-diagnosis interventions added (both Loihi-realizable): **H2e** no weight change on a miss (adapt on hits only); **H2f** reward gates only the Hebbian product, `Δw = α(r·x − y·w)`, so the normalising term contracts for both signs.

**1-shot FP32 has zero misses** (th 0.70 s10: 2,314 updates, all hits), because every class arrives as one contiguous block. The 1-shot FP32 gap to CLP therefore cannot come from H2 or H3; only H1 and the allocation count (86 vs CLP's 104) remain.

## Step 2: single-factor ablations (FP32, dev)

Dev cases: seed 10 at th 0.70 / 0.85 (both settings) plus the two FP32 collapse cases (25-shot th 0.80 s10, th 0.70 s20). Δ vs CLP is on the same seeds (cached rows). Final accuracy %:

| config | 1-shot .70 | 1-shot .85 | 25-shot .70 (s10,s20) | 25-shot .80 | 25-shot .85 | max ‖w‖ (25-shot .70) |
|---|---|---|---|---|---|---|
| CLP (cached) | 54.00 | 54.33 | 92.21 | 93.33 | 93.33 | 1 (renorm) |
| base (released) | 52.83 | 54.08 | 47.17 (min 2.5) | **2.50** | 93.17 | 1.41 / NaN |
| H1a timing fix | 53.67 | 53.87 | 91.58 | 92.92 | 93.04 | 2.04 |
| H1b g_inc = 1 | 53.42 | 53.92 | 47.52 (min 2.5) | 92.96 | 92.92 | 1.33 |
| **H1ab** (intended) | 53.33 | 53.71 | 92.00 | 93.25 | 93.33 | 1.59 |
| H1ab+H2a floor | = H1ab | = | 92.02 | 93.25 | 93.33 | 1.59 |
| H1ab+H2b no decrement | = | = | 91.96 | 93.29 | 93.33 | 1.22 |
| H1ab+H2c1 / c2 neg α/2, /4 | = | = | 91.79 / 91.23 | 93.21 / 93.17 | 93.33 / 93.29 | 1.17 / 1.10 |
| H1ab+H2d oracle renorm | 53.92 | 54.08 | 92.15 | 93.38 | 93.33 | 1.00 |
| H1ab+H2e no neg update | = | = | 90.40 | 93.04 | 93.21 | 1.06 |
| H1ab+H2f decay-fixed | = | = | 91.48 | 93.17 | 93.25 | 1.06 |
| H1ab+H3 alloc on miss | = | = | **92.92** | 93.38 | 93.33 | 1.22 |
| H1ab+H2e+H3 | = | = | **92.81** | 93.29 | 93.33 | 1.06 |
| H1ab+H2f+H3 | = | = | 92.29 | 93.25 | 93.25 | 1.06 |

Prototypes at 25-shot .70: CLP 1,114; base 900; H1ab 1,068; H1ab+H3 1,184; H1ab+H2e+H3 1,192; NA 2,650.

Verdicts:
- **H1 supported, and it is the main cause.** The timing fix alone (H1a) removes both FP32 collapses: with α = 1/2 on the first hit, no prototype starts the miss sequence with inflated norm. `g_inc = 1` alone (H1b) does not. H1ab is within 0.25 pp of CLP in 25-shot and 0.6-0.7 pp in 1-shot.
- **H2 partly refuted as predicted in the revision.** The floor (H2a) is a no-op, as predicted. H2b/H2c lower max ‖w‖ but do not change accuracy. Removing or fixing the negative branch (H2e/H2f) bounds the norm (1.06) but costs accuracy at th 0.70 when the missed sample is discarded: the push-away step carries information.
- **H3 supported in 25-shot** (+0.7 pp over CLP at th 0.70 with 6% more prototypes, still under half of NA's). There are no misses in 1-shot, so H3 is a no-op there, as are all H2 variants.
- 1-shot residual (H1ab −0.6 pp vs CLP, oracle −0.1 to −0.25 pp): the Taylor norm drift (‖w‖ ≤ 1.04) and allocation count (96 vs 104). This is within the 1 pp criterion on seed 10.

## Step 3: combined FP32 config

INT8 dev results (step 4 below) informed the choice, because H1ab alone meets the FP32 criteria on dev but leaves the repelling negative branch in place (max ‖w‖ 1.59).

**`fix` = H1ab + H2f + H3**: α after the goodness increment, steps of ±1 (intended 1, 1/2, 1/3, …); on a miss the reward gates only the Hebbian product (`Δw = α(−x − y·w)`), and the missed sample is also allocated. All three are Loihi-realizable in principle: the order of the goodness update, a rule with the reward factor on one product only, and allocation-on-error, which the chip already does. This is the same config as `H1ab+H2f+H3`.

## Step 4: INT8

Dev: seed 10 at th 0.70 / 0.75 / 0.85 plus the INT8 collapse cases (25-shot th 0.85 s20, 1-shot th 0.70 s30). Δ vs CLP on the same seeds:

| config | 1-shot .70 | 1-shot .75 | 1-shot .85 | 25-shot .70 | 25-shot .75 | 25-shot .85 |
|---|---|---|---|---|---|---|
| base | −29.8 (collapse) | | −3.0 | −87.2 | −83.4 | −15.7 |
| H1ab | −5.6 | | −2.4 | −5.6 | −6.7 | −3.0 |
| H1ab+H2b | = H1ab | | = | **−81.9 (collapse)** | −90.0 | −84.1 |
| H1ab+H3 | −5.2 | | −2.4 | −2.3 (9.5k protos) | −1.3 (8.4k) | −2.0 (12k) |
| fix | −5.2 | −4.6 | −2.4 | −2.2 | −2.2 | −2.0 |
| fix+H4a (α_int ≤ 7) | −4.0 | −4.4 | −1.5 | −1.8 | −2.1 | −1.9 (+37% protos) |
| fix+H4b1 (weights ×2) | −3.4 | −2.5 | −1.5 | −0.6 | −1.3 | −0.5 |
| **fix+H4b2 (weights ×4)** | **−1.4** | **−1.3** | **−1.4** | **−0.2** | **−0.8** | **−0.3** |

- **H1 in INT8:** H1ab removes every collapse, but INT8 stays 2-7 pp behind.
- **H2b is catastrophic in INT8** (every dev case collapses, max ‖w‖ ≈ 28-36), while it is harmless in FP32. The cause is not diagnosed: it is consistent with the repelling r < 0 branch acting at small α, but it was not traced, because `fix` drops that branch anyway.
- **H3 alone floods the pool in INT8** (8-12k prototypes); it needs H2f next to it.
- **H4 supported** (shadow diagnostic, `diag/*INT8_H1ab+H2f+H3*`). A float shadow driven by the same decisions stays at cos 0.9999 (1-shot) / 0.996 (25-shot) to the absorbed-sample mean, while the integer prototype is at 0.991 / 0.979, and 33% of 25-shot hit updates change no weight. Inputs average 2.7 LSB at scale 128, so averaged steps fall under the rounding floor. A 4× finer weight grid (max |w_i| = 0.248; the largest feature element is 0.433 but only 0.005% of elements exceed 0.248) removes most of the gap. The hardware-faithful α range barely matters for accuracy.

**INT8 config: `fix+H4b2`.**

## Step 5: confirmation (full grid)

Configs frozen before this step. The grid is seeds 10/20/30 × th {0.70, 0.75, 0.80, 0.85} × both settings; seeds 20/30 at the new thresholds were never used for a decision. Tables are in `results/adaptive_protos/comparison_{1shot,25shot}.csv`; the figure is `images/adaptive_protos_confirmation.png`. Δ is the mean over 3 seeds vs the cached CLP / NA rows:

| | th | FP32 released | FP32 **fix** | INT8 released | INT8 **fix+H4b2** | Δ fix vs CLP (FP32 / INT8) | Δ vs NA (INT8) |
|---|---|---|---|---|---|---|---|
| 1-shot | .70 | 54.88 | 55.92 | 33.65 ± 19.0 | 54.88 | −0.65 / −1.69 | −1.54 |
| | .75 | 55.61 | 56.72 | 47.81 | 54.94 | −0.25 / −2.03 | −1.92 |
| | .80 | 56.51 | 56.22 | 52.50 | 55.62 | −0.56 / −1.15 | −0.94 |
| | .85 | 56.28 | 56.79 | 53.35 | 56.50 | −0.36 / −0.65 | −0.67 |
| 25-shot | .70 | 62.14 ± 42.2 | 92.35 | 57.11 ± 36.9 | 92.04 | +0.06 / −0.25 | −0.36 |
| | .75 | 92.29 | 92.68 | 61.22 ± 36.3 | 92.03 | −0.28 / −0.93 | −0.72 |
| | .80 | 62.62 ± 42.5 | 92.74 | 90.10 | 92.35 | −0.39 / −0.78 | −0.60 |
| | .85 | 93.29 | 93.15 | 81.97 ± 12.2 | 92.97 | −0.07 / −0.25 | −0.24 |

Prototypes (fix, FP32 / INT8, vs CLP / NA): 1-shot .85 254 / 261 vs 281 / 453; 25-shot .70 1,212 / 1,264 vs 1,127 / 2,673 (FP32-NA); 25-shot .85 4,445 / 4,521 vs 4,831 / 7,907.

Against the criteria fixed in advance:
- FP32 within 1 pp of CLP at every threshold: **met** (worst −0.65).
- FP32 no seed more than 3 pp below CLP: **met** (worst −1.25).
- FP32 std at most 2× CLP's: **not met** at 25-shot .75 (0.65 vs 0.22, 3.0×) and .80 (0.37 vs 0.17, 2.2×). CLP's 25-shot std is under 0.25 pp, so a single seed 1 pp low trips a relative bound; I note that but do not re-tune.
- INT8 within 2 pp of INT8-NA at th ≥ 0.75 (3-seed mean): **met** (worst −1.92, 1-shot .75). On the held-out seeds alone that cell is −2.02, a miss by 0.02 pp; seed 30 is −2.29.
- INT8 no collapse at any threshold: **met** (min over 24 runs 52.1% 1-shot, 91.7% 25-shot; max ‖w‖ ≤ 1.12).

## Conclusion

1. **The main cause is an implementation bug (H1).** The first update after allocation used α = 1 instead of the intended 1/2, because α was read before the goodness increment. Together with `g_inc = 0.5`, that inflates each new prototype's norm on its first hit. A subsequent run of misses (each lowering goodness, keeping α = 1) then drives ‖w‖ to overflow, and one prototype wins everything. Fixing the timing alone removes every FP32 collapse.
2. **The Oja-style negative update is structurally unstable (H2).** For r = −1 the self-normalising term becomes anti-normalising: the unit sphere is repelling at any α. In INT8 this collapses runs even at α ≈ 0.05. Gating the reward onto the Hebbian product only (H2f) makes both branches contracting. Together with allocating the missed sample (H3), this reaches CLP parity in FP32 with fewer prototypes than CLP at th ≥ 0.75 and about half of allocation-only.
3. **The remaining INT8 gap is update precision (H4).** Averaged steps fall below the 8-bit rounding floor, and a 4× finer weight grid closes most of it. INT8 adaptive then sits 0.25-2 pp below CLP, within 2 pp of allocation-only, with 40-55% fewer prototypes than allocation-only.

## Flags
- **H1 bug in the released model.** `models/CLP_SNN.py:482-490` (float) and `:533-536` (INT8) apply the stored α before the goodness update, so the first post-allocation step is α = 1, not 1/2. The released model file is untouched; the fix lives in `variants.py` (`alpha_after_inc`). The paper's CLP-SNN simulation numbers (1-shot, INT8 adaptive, g_inc 0.5, th 0.9) come from this code. **This could affect a reviewer response.**
- **Code vs Methods.** main.tex:509 has `g ← max(1, g + r)` with steps of ±1; the code uses `g_inc = 0.5` with no floor. The docstrings at `CLP_SNN.py:480, :487` say goodness changes only on positive reward, which is also wrong. **This could affect a reviewer response.**
- **Hardware α range (corrected 2026-09-29).** The user confirms that α on chip can go up to 127, so the in-code note at `CLP_SNN.py:528-530` ("s_mantissa ... restricts to [-8, 7]") is wrong, and the H4a knob (α_int ≤ 7) tested a constraint that does not exist. `fix+H4b2` therefore needs no α divergence from hardware. The 4× weight grid still assumes a weight exponent plus a 2^k factor on the Hebbian product; that has not been checked against Lava-Loihi or run on chip.
- **Discussion's 2-3× prototype claim.** `00-STATUS.md:464` supports it with INT8-A counts at th 0.75 (105 vs 242, 1,473 vs 3,673), taken from runs that scored 47.8% and 61.2%. With the fixed rule the ratio vs allocation-only is 1.7-2.1× (INT8 fix+H4b2: 1-shot 124 vs 242 at .75, 261 vs 453 at .85; 25-shot 1,816 vs 3,673 at .75, 4,521 vs 7,907 at .85).
- **Not diagnosed:** why H2b (no goodness decrement) collapses INT8 but not FP32.

## Post-hoc: goodness step 1 vs 0.5 in the final configs

Run after step 5 to answer a user question, not used for any decision (`fix-g05`, `fix+H4b2-g05`, full grid). The steps are ±1 in `fix`; the released code uses ±0.5. Mean final accuracy, g = 1 minus g = 0.5 (pp):

| | .70 | .75 | .80 | .85 |
|---|---|---|---|---|
| FP32 1-shot | +0.90 | +0.64 | −0.36 | +0.44 |
| FP32 25-shot | +0.93 | +0.89 | +0.21 | +0.04 |
| INT8 1-shot | +1.17 | −0.06 | +0.57 | −0.18 |
| INT8 25-shot | +1.33 | +0.79 | +0.51 | −0.28 |

With the timing and negative-branch fixes in place, the step size no longer decides stability (no collapse either way, max ‖w‖ ≤ 1.11). Steps of 1 buy up to about 1 pp at low thresholds and cost 10-15% more prototypes (faster α decay consolidates prototypes sooner, so more inputs fall outside them and allocate). At th 0.85 the difference is within noise.

## Theory: norm dynamics of the Taylor rule (H2)

Let n = ‖w‖, ‖x‖ = 1, c = cos∠(w, x), y = wᵀx = n·c, r ∈ {−1, +1}. For Δw = α·r·(x − y·w) the new norm is **exactly** (numerically checked to 1e-14)

  ‖w'‖² = n² + 2·α·r·y·(1 − n²) + α²·(1 − 2y² + y²n²).

- At n = 1 this is 1 + α²·sin²θ for **both** signs: the Methods derivation is right that the rule matches renormalisation to first order at the unit sphere, for r = −1 too.
- Off the sphere, write δ = n² − 1. To first order, δ' = (1 − 2·α·r·n·c)·δ + α²·(…). The winner has c > 0, since it passed the threshold.
  - **r = +1**: the multiplier is 1 − 2αnc < 1, so deviations shrink. The unit sphere is attracting, up to a small offset δ* ≈ α(1 − c²)/(2c) that vanishes as α decays. This is the self-normalising property.
  - **r = −1**: the multiplier is 1 + 2αnc > 1, and the α² term is also positive, so there is no fixed point with n ≥ 1. The unit sphere is repelling. Any excess norm, including the unavoidable α²sin²θ, INT8 rounding, or the α = 1 first-hit bug, grows geometrically with consecutive misses, and faster as n grows, because y = nc enters the multiplier.
- **Feedback through the competition.** Winner selection and the threshold use the raw y = n·c, so a prototype with n > 1 wins any input with c > θ/n: its acceptance cone widens, it captures other classes, and it collects more misses (r = −1). The goodness rule raises α on each miss (α = 1/max(1, g − 1) reaches 1 after a few misses), which enlarges the multiplier. Three loops compound until one prototype wins everything (observed: 2.5% = chance).
- In CLP none of this can happen, because the explicit renorm pins n = 1 after every update; the "a miss makes the prototype more plastic" design becomes dangerous only once renormalisation is replaced by the Taylor rule.
- **H2f** (Δw = α(r·x − y·w)): ‖w'‖² − n² ≈ 2αy(r − n²). For r = +1 it is identical to the rule above; for r = −1 it is −2αy(1 + n²) < 0, so a miss always shrinks the norm. Its tangential (direction) change equals the renormalised push-away to first order; only the radial part differs, so direction learning is unchanged. A prototype with more misses than hits shrinks, loses competitions, and retires itself.
