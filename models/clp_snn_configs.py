"""
Named CLPSNN simulator configurations.

PAPER configs reproduce CLP-SNN as deployed on Loihi 2 (the numbers reported
in the paper are measured on the chip; these simulator configs mirror it):
allocation-only INT8 (prototypes imprinted once, misclassified inputs
allocated as new prototypes) and the chip's prototype pool. The thresholds
are the effective similarity thresholds chosen for the on-chip experiments.
The chip's measured accuracies are in data/loihi2/.

OTHER configs are not used for any paper number. They run the full adaptive
rule (winner updated on correct and incorrect predictions) or give the
allocation-only network more prototypes; see README "CLP-SNN simulator
configurations" for their accuracy and prototype counts.

Usage:
    from models.CLP_SNN import CLPSNN
    from models.clp_snn_configs import CONFIGS
    clf = CLPSNN(1280, num_classes=40, device="cpu", **CONFIGS["paper_1shot"])
"""

CONFIGS = {
    # ── Paper: chip-mirroring (reproduce CLP-SNN on Loihi 2) ──────────────
    "paper_1shot": dict(adaptive_protos=False, use_quantization=True,
                        threshold=0.66, n_protos=230, g_inc=0.5),
    "paper_25shot": dict(adaptive_protos=False, use_quantization=True,
                         threshold=0.47, n_protos=1400, g_inc=0.5),

    # ── Other: full adaptive rule (simulation only, not on chip) ─────────
    "adaptive_int8_1shot": dict(adaptive_protos=True, use_quantization=True,
                                threshold=0.90, n_protos=600, g_inc=0.5),
    "adaptive_int8_25shot": dict(adaptive_protos=True, use_quantization=True,
                                 threshold=0.85, n_protos=6000, g_inc=0.5),
    "adaptive_fp32_1shot": dict(adaptive_protos=True, use_quantization=False,
                                threshold=0.80, n_protos=600, g_inc=0.5),
    "adaptive_fp32_25shot": dict(adaptive_protos=True, use_quantization=False,
                                 threshold=0.85, n_protos=6000, g_inc=0.5),

    # ── Other: allocation-only with a large pool and a higher threshold ──
    "alloc_int8_1shot_large": dict(adaptive_protos=False, use_quantization=True,
                                   threshold=0.85, n_protos=1000, g_inc=0.5),
    "alloc_int8_25shot_large": dict(adaptive_protos=False, use_quantization=True,
                                    threshold=0.85, n_protos=10000, g_inc=0.5),
}
