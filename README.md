# CLP-SNN: Real-time Continual Learning on Intel Loihi 2

Code release for the paper:

> **Real-time Continual Learning on Intel Loihi 2**  
> *Nature Communications* (under review)

CLP-SNN is a spiking neural network for online continual learning, featuring a self-normalizing three-factor local learning rule, neurogenesis, and metaplasticity — implemented on Intel's Loihi 2 neuromorphic chip and benchmarked against an NVIDIA Jetson Orin Nano GPU.

---

## Repository Structure

| Folder / File | Description | Paper section |
|---|---|---|
| `models/CLP_SNN.py` | Main contribution: spiking CLP with float (Taylor) and INT (Lava-faithful) learning paths | Methods §2.1–2.2 |
| `models/CLP.py` | Original CLP baseline | Methods §2.1 |
| `models/SLDA.py` | Streaming LDA (standard and Frozen Σ variants) | Baselines |
| `models/NCM.py` | Nearest Class Mean baseline | Baselines |
| `models/Replay.py` | Streaming softmax + experience replay | Baselines |
| `models/Perceptron.py` | Perceptron / fine-tuning baseline | Baselines |
| `experiments/clp_vs_baselines_1shot.py` | 1-shot accuracy comparison: CLP, CLP-SNN, and 6 baselines | Results Fig. 3a, Table 1 |
| `experiments/clp_vs_baselines_25shot.py` | 25-shot accuracy comparison: same 8 main classifiers | Results Fig. 3b, Table 1 |
| `experiments/clp_vs_clp_snn_1shot.py` | 1-shot ablation of CLP-SNN float vs INT learning variants | Supplemental |
| `experiments/forgetting_experiments_1shot.py` | True-Peak FM forgetting analysis | Results Fig. 4 |
| `experiments/clp_snn_threshold_g_inc_sweep.py` | Threshold / g_inc hyperparameter sweep | Supplemental |
| `notebooks/` | Companion notebooks reproducing all paper figures | — |
| `analysis/` | Self-normalization and quantization analysis scripts | Supplemental Figs. S1–S2 |
| `benchmarks/benchmark.py` | Latency/energy benchmarking on Jetson Orin Nano | Results Table 1 |
| `data/` | Pre-extracted OpenLORIS features (EfficientNet-B0, 1280-dim, L2-normalized) | — |

---

## Setup

```bash
# 1. Install PyTorch (choose your platform)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124  # GPU (CUDA 12.4)
# or: pip install torch==2.5.1  # CPU-only

# 2. Install remaining dependencies
pip install -r requirements.txt
```

---

## Data

`data/` contains pre-extracted EfficientNet-B0 features for the OpenLORIS-Scene dataset (40 identity classes):

| Path | Description |
|---|---|
| `data/1shot/X_train_1_shot_{10,20,30}.pt` | 1-shot training sets (60 frames/class, 3 seeds) |
| `data/1shot/y_train_1_shot_{10,20,30}.pt` | Corresponding labels |
| `data/25shot/X_train_25_shot_{10,20,30}.npy` | 25-shot training sets |
| `data/25shot/y_train_25_shot_{10,20,30}.npy` | Corresponding labels |
| `data/X_test.npy` | Test features (60 samples/class, balanced, seed 42) |
| `data/y_test.npy` | Test labels |

Features are 1280-dimensional EfficientNet-B0 outputs, L2-normalized before all prototype-based classifiers.
The raw OpenLORIS dataset can be downloaded from [the OpenLORIS project](https://lifelong-robotic-vision.github.io/dataset/scene.html).

---

## Running Experiments

All scripts are run from the repo root or from `experiments/`:

```bash
# 1-shot accuracy comparison: 8 main classifiers (Fig. 3a, Table 1 1-shot column)
python experiments/clp_vs_baselines_1shot.py

# 25-shot accuracy comparison: same 8 main classifiers (Fig. 3b, Table 1 25-shot column)
python experiments/clp_vs_baselines_25shot.py

# CLP-SNN float vs INT learning-rule ablation (Supplemental)
python experiments/clp_vs_clp_snn_1shot.py

# Forgetting analysis — BWT and True-Peak FM (Fig. 4 / Supplemental)
python experiments/forgetting_experiments_1shot.py

# Hyperparameter sweep — threshold vs g_inc (Supplemental)
python experiments/clp_snn_threshold_g_inc_sweep.py
```

Results and figures are saved to `experiments/results/` and `images/` respectively.
Each script has a `SAVE_PDF = True` flag near the top — set it to `False` to skip
`.pdf` output and save only `.png` (useful on headless servers without a PDF backend).

Companion notebooks in `notebooks/` walk through each experiment interactively.

---

## Hardware Benchmarks (Jetson Orin Nano)

`benchmarks/benchmark.py` reproduces the latency and energy measurements in Table 1.
Requires a Jetson Orin Nano with JetPack SDK 6.2.1, PyTorch 2.4.0, and the `jtop` package.

```bash
python benchmarks/benchmark.py \
    --algorithm clp \
    --data_path data/1shot/X_train_1_shot_10.npy \
    --device_type orin \
    --compute_device cuda
```

See `benchmarks/README.md` for full instructions.

---

## Note on Loihi 2 Hardware Implementation

The Loihi 2 on-chip implementation of CLP-SNN requires access to Intel's proprietary Lava-Loihi framework and Loihi 2 hardware. The simulation implementation in `models/CLP_SNN.py` mirrors the hardware design faithfully (see `LearningConnectionModelBitApproximate` in the lava repo for the integer pipeline specification). Researchers with Loihi 2 access may contact the authors for the hardware implementation code.

---

## Citation

```bibtex
@article{anonymous2026clpsnn,
  title   = {Real-time Continual Learning on Intel Loihi 2},
  author  = {Anonymous},
  journal = {Nature Communications},
  year    = {2026}
}
```
