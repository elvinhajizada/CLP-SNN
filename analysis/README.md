# Analysis Scripts

Research-level analysis of the CLP-SNN learning rule — separate from the production `models/` implementations.

| File | Purpose | Paper reference |
|---|---|---|
| `self_norm_analysis.py` | Similarity-to-cluster-center analysis across dimensions (d = 64, 256, 1280) and on OpenLoris features (class 13, frames with cosine >= 0.75 to the class centre). Generates the 3-row × 4-column supplemental figure (similarity / norm / raw dot product for Variants A, B, F). | Supplementary Fig. S2 |
| `sigma_sensitivity_analysis.py` | Sensitivity of weight-norm stability to input feature variance σ. | Supplementary Fig. S3 |
| `pareto_plots.py` | Pareto frontier of accuracy vs. latency / energy across CLP-SNN (Loihi 2), CLP, NCM, Replay, and rank-one SLDA on Jetson Orin Nano CPU/GPU. Latency/energy from `benchmarks/reports_table1.csv`, composed into a per-sample step as in Table 1; the Loihi 2 point was measured on chip (proprietary Lava-Loihi toolchain). | Fig. 3 c, d, e |

These scripts do not need to be run to reproduce the main paper results. They document the design and validation of the self-normalizing integer learning rule.
