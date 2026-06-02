# Analysis Scripts

Research-level analysis of the CLP-SNN learning rule — separate from the production `models/` implementations.

| File | Purpose | Paper reference |
|---|---|---|
| `self_norm_analysis.py` | Similarity-to-cluster-center analysis across dimensions (d = 64, 256, 1280) and on OpenLoris features. Generates the 3-row × 4-column supplemental figure (similarity / norm / raw dot product for Variants A, B, F). | Supplemental Fig. S1 |
| `sigma_sensitivity_analysis.py` | Sensitivity of weight-norm stability to input feature variance σ. | Supplemental Fig. S2 |
| `pareto_plots.py` | Pareto frontier of accuracy vs. latency / energy across CLP-SNN (Loihi 2), CLP, NCM, Replay, and SLDA-periodic on Jetson Orin Nano CPU/GPU. Latency/energy from `table_1_revision.xlsx`; accuracy from Loihi 2 evaluations (proprietary Lava-INL toolchain). | Fig. 3 c, d, e |

These scripts do not need to be run to reproduce the main paper results. They document the design and validation of the self-normalizing integer learning rule.
