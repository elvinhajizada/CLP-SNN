"""
Named CLP-SNN configurations.

LOIHI_CONFIGS reproduce CLP-SNN as deployed on Loihi 2, whose accuracies the
paper reports, with the emulator in models/CLP_SNN_Loihi.py (allocation-only,
7-bit arithmetic, spike-timing winner selection). threshold and resolution are
in cosine-similarity units. The chip's measured accuracies are in data/loihi2/.

CONFIGS are simulator (models/CLP_SNN.py) settings that are not used for any
paper number. They run the full adaptive rule (winner updated on correct and
incorrect predictions) or give the allocation-only network more prototypes;
see README "CLP-SNN configurations" for their accuracy and prototype counts.

Usage:
    from models.CLP_SNN_Loihi import CLPSNNLoihi
    from models.clp_snn_configs import LOIHI_CONFIGS
    clf = CLPSNNLoihi(1280, num_classes=40, **LOIHI_CONFIGS["paper_1shot"])

    from models.CLP_SNN import CLPSNN
    from models.clp_snn_configs import CONFIGS
    clf = CLPSNN(1280, num_classes=40, device="cpu", **CONFIGS["adaptive_fp32_1shot"])
"""

LOIHI_CONFIGS = {
    # ── Paper: CLP-SNN on Loihi 2 ────────────────────────────────────────
    "paper_1shot": dict(threshold=0.708, resolution=0.0244, t_wait=11,
                        n_protos=230),
    "paper_25shot": dict(threshold=0.5066, resolution=0.0427, t_wait=11,
                         n_protos=1400),
}

CONFIGS = {
    # ── Simulator: full adaptive rule (simulation only, not on chip) ─────
    "adaptive_int8_1shot": dict(adaptive_protos=True, use_quantization=True,
                                threshold=0.90, n_protos=600, g_inc=0.5),
    "adaptive_int8_25shot": dict(adaptive_protos=True, use_quantization=True,
                                 threshold=0.85, n_protos=6000, g_inc=0.5),
    "adaptive_fp32_1shot": dict(adaptive_protos=True, use_quantization=False,
                                threshold=0.80, n_protos=600, g_inc=0.5),
    "adaptive_fp32_25shot": dict(adaptive_protos=True, use_quantization=False,
                                 threshold=0.85, n_protos=6000, g_inc=0.5),

    # ── Simulator: allocation-only with a large pool and a higher threshold
    "alloc_int8_1shot_large": dict(adaptive_protos=False, use_quantization=True,
                                   threshold=0.85, n_protos=1000, g_inc=0.5),
    "alloc_int8_25shot_large": dict(adaptive_protos=False, use_quantization=True,
                                    threshold=0.85, n_protos=10000, g_inc=0.5),
}
