# Analysis Scripts

Research-level analysis of the CLP-SNN learning rule — separate from the production `models/` implementations.

| File | Purpose | Paper reference |
|---|---|---|
| `self_norm_analysis.py` | Similarity-to-cluster-center analysis across dimensions (d = 64, 256, 1280) and on OpenLoris features. Generates the 3-row × 4-column supplemental figure (similarity / norm / raw dot product for Variants A, B, F). | Supplemental Fig. S1 |
| `sigma_sensitivity_analysis.py` | Sensitivity of weight-norm stability to input feature variance σ. | Supplemental Fig. S2 |

These scripts do not need to be run to reproduce the main paper results. They document the design and validation of the self-normalizing integer learning rule.
