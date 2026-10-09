# CLP-SNN: Real-time Continual Learning on Intel Loihi 2

Code release for the paper:

> **Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network**  
> https://arxiv.org/abs/2511.01553

CLP-SNN is a spiking neural network for online continual learning, featuring a self-normalizing three-factor local learning rule, neurogenesis, and metaplasticity — implemented on Intel's Loihi 2 neuromorphic chip and benchmarked against an NVIDIA Jetson Orin Nano GPU.

---

## Repository Structure

| Folder / File | Description | Paper section |
|---|---|---|
| `models/CLP_SNN.py` | Main contribution: spiking CLP with float (Taylor) and INT (Lava-faithful) learning paths | Methods §2.1–2.2 |
| `models/CLP_SNN_Loihi.py` | Emulator of the CLP-SNN deployment on Loihi 2 (allocation-only, integer arithmetic, spike-timing winner) | Results, Supplemental |
| `models/clp_snn_configs.py` | Named CLP-SNN configurations: Loihi 2 deployment parameters and simulator presets | — |
| `models/CLP.py` | Original CLP baseline | Methods §2.1 |
| `models/SLDA.py` | Streaming LDA (standard and Frozen Σ variants) | Baselines |
| `models/NCM.py` | Nearest Class Mean baseline | Baselines |
| `models/Replay.py` | Streaming softmax + experience replay | Baselines |
| `models/Perceptron.py` | Perceptron / fine-tuning baseline | Baselines |
| `experiments/clp_vs_baselines_1shot.py` | 1-shot accuracy comparison: CLP, CLP-SNN, and 6 baselines | Results Fig. 3a, Table 1 |
| `experiments/clp_vs_baselines_25shot.py` | 25-shot accuracy comparison: same 8 main classifiers | Results Fig. 3b, Table 1 |
| `experiments/clp_vs_clp_snn_1shot.py` | 1-shot ablation of CLP-SNN float vs INT learning variants | Supplemental |
| `experiments/forgetting_experiments_1shot.py` | True-Peak FM forgetting analysis | Results Fig. 4 |
| `experiments/clp_snn_gap_decomposition.py` | Sources of the accuracy gap between CLP and CLP-SNN on Loihi 2 | Supplemental |
| `experiments/clp_snn_threshold_g_inc_sweep.py` | Threshold / g_inc hyperparameter sweep | Supplemental |
| `notebooks/` | Companion notebooks that walk through the experiments; the scripts in this table produce the paper figures | — |
| `analysis/` | Accuracy and cost panels, self-normalization analysis scripts | Results Fig. 3, Supplemental Figs. S2–S3 |
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

> **The dataset is NOT included in GitHub's "Download ZIP".**
> Files under `data/` are tracked with [git-lfs](https://git-lfs.com);
> Use one of the options below to obtain the real ~1.2 GB feature dataset.

**For peer review: download from the anonymized OSF link (no git-lfs required).**
Download `clp_snn_openloris_features.zip` (1.14 GB, MD5 `066e5df76f1080197584454df7f2d549`) from
[this view-only link](https://osf.io/bty67/overview?view_only=5092dcdc02304641afbc0b52a20b65b1)
and unzip it into `data/` as in Option 1.

**Option 1 — Download from Zenodo (no git-lfs required).**
Download `clp_snn_openloris_features.zip` from the Zenodo record
[DOI: 10.5281/zenodo.20492557](https://doi.org/10.5281/zenodo.20492557)
and unzip it into `data/` (`unzip clp_snn_openloris_features.zip -d data`), so the files land in
`data/1shot/`, `data/25shot/`, `data/X_test.npy`, etc.

**Option 2 — Clone with git-lfs.**
Install [git-lfs](https://git-lfs.com) once per machine, then clone the repo:

```bash
git lfs install
git clone https://github.com/elvinhajizada/CLP-SNN.git
```

If you already cloned the repo *without* LFS, run `git lfs pull` from inside
the clone to fetch the missing data.

### Contents of `data/`

`data/` contains pre-extracted EfficientNet-B0 features for the OpenLORIS-Scene dataset (40 identity classes):

| Path | Description |
|---|---|
| `data/1shot/X_train_1_shot_{10,20,30}.pt` | 1-shot training sets (60 frames/class, 3 seeds) |
| `data/1shot/y_train_1_shot_{10,20,30}.pt` | Corresponding labels |
| `data/25shot/X_train_25_shot_{10,20,30}.npy` | 25-shot training sets |
| `data/25shot/y_train_25_shot_{10,20,30}.npy` | Corresponding labels |
| `data/X_test.npy` | Test feature pool (53,295 features); each experiment script draws its class-balanced test subset at run time |
| `data/y_test.npy` | Test labels |
| `data/loihi2/accuracies_clp_loihi_{1shot,25shot}.npy` | CLP-SNN accuracy curves measured on Loihi 2 (checkpoints × 1 × seeds 10/20/30) |
| `data/loihi2/loihi2_cost_benchmark.csv` | CLP-SNN per-sample OCL step on Loihi 2, aggregated from the on-chip measurement logs (latency, power and energy of the Table 1 benchmark network; power is the Loihi 2 chip power, not full-system power) |

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

# Accuracy gap between CLP and CLP-SNN on Loihi 2, factor by factor (Supplemental)
python experiments/clp_snn_gap_decomposition.py --sweep
```

Results and figures are saved to `experiments/results/` and `images/` respectively.
Each script has a `SAVE_PDF = True` flag near the top — set it to `False` to skip
`.pdf` output and save only `.png` (useful on headless servers without a PDF backend).

Companion notebooks in `notebooks/` walk through each experiment interactively.

---

## CLP-SNN configurations

### Paper: CLP-SNN on Loihi 2

The CLP-SNN accuracies reported in the paper are measured on Loihi 2.
`models/CLP_SNN_Loihi.py` emulates that deployment in integer arithmetic, and the
experiment scripts use it with the deployment parameters in `LOIHI_CONFIGS`:

```python
from models.CLP_SNN_Loihi import CLPSNNLoihi
from models.clp_snn_configs import LOIHI_CONFIGS
clf = CLPSNNLoihi(1280, num_classes=40, **LOIHI_CONFIGS["paper_1shot"])
```

The emulator follows the deployed network, and no parameter is fitted to the chip's results:

- **input:** negative features clipped, L2-normalised and rounded to 7 bits, x_int = round(128·x);
- **allocation-only:** on novelty or an incorrect prediction, a new prototype is imprinted with W = x_int and then frozen;
- **threshold:** during training a prototype wins only if its cosine similarity to the input is at least 0.708 (1-shot) or 0.507 (25-shot);
- **winner selection:** by spike time; one time step spans 0.024 (1-shot) or 0.043 (25-shot) of cosine similarity, and prototypes that spike in the same step vote on the label.

| Preset | Threshold | Resolution | Pool | Final acc. (%) | AAA (%) | Loihi 2 final (%) | Loihi 2 AAA (%) |
|---|---|---|---|---|---|---|---|
| `paper_1shot` | 0.708 | 0.024 | 230 | 55.5 ± 2.6 | 66.7 | 55.4 ± 2.5 | 66.6 |
| `paper_25shot` | 0.507 | 0.043 | 1,400 | 90.4 ± 0.4 | 81.1 | 90.0 ± 0.4 | 81.0 |

Evaluated as on the chip, the emulator reproduces the measured accuracy: 55.4% vs
55.4% (1-shot) and 90.1% vs 90.0% (25-shot), with curve RMSE 0.4 / 0.5 points. Ties
are broken at random, as on the chip, so single runs vary by a few tenths of a point.

`experiments/clp_snn_gap_decomposition.py` adds the features of the deployment to CLP
one at a time (final accuracy in %, prototypes in parentheses, mean over seeds 10/20/30):

| Configuration | 1-shot | 25-shot |
|---|---|---|
| CLP | 57.0 (127) | 93.0 (1,769) |
| Imprint-only, INT8, CLP's threshold (0.75) | 56.9 (242) | 92.7 (3,673) |
| + chip input preprocessing | 57.2 (210) | 92.5 (3,083) |
| + chip threshold (0.708 / 0.507) | 55.9 (167) | 90.4 (1,217) |
| + spike-timing winner selection | 55.4 (168) | 90.5 (1,302) |

Imprint-only operation matches CLP's accuracy at CLP's threshold, but with about twice
as many prototypes; the full adaptive rule (simulator presets below) avoids that overhead.

With `--sweep` it also reports accuracy and prototype count over the threshold.

### Simulator configurations (not used for paper numbers)

`CONFIGS` holds settings for the simulator in `models/CLP_SNN.py`:

```python
from models.CLP_SNN import CLPSNN
from models.clp_snn_configs import CONFIGS
clf = CLPSNN(1280, num_classes=40, device="cpu", **CONFIGS["adaptive_fp32_1shot"])
```

**Adaptive presets** run the full learning rule: the winner is also updated on correct and incorrect predictions. This path runs in simulation only; the chip deployment is allocation-only.

**`alloc_*_large` presets** give the allocation-only network a higher threshold and a larger pool.

Mean over seeds 10/20/30, OpenLORIS test set.

| Preset | Setting | Final acc. (%) | AAA (%) | Prototypes |
|---|---|---|---|---|
| `adaptive_int8_1shot` | adaptive INT8, θ 0.90, 600 slots | 55.4 ± 1.1 | 67.0 | 461 |
| `adaptive_fp32_1shot` | adaptive FP32, θ 0.80, 600 slots | 56.6 ± 1.7 | 68.5 | 145 |
| `alloc_int8_1shot_large` | allocation-only INT8, θ 0.85, 1,000 slots | 57.2 ± 2.2 | 68.7 | 455 |
| `adaptive_int8_25shot` | adaptive INT8, θ 0.85, 6,000 slots | 91.7 ± 0.4 | 82.9 | 4,614 |
| `adaptive_fp32_25shot` | adaptive FP32, θ 0.85, 6,000 slots | 93.2 ± 0.3 | 84.7 | 3,983 |
| `alloc_int8_25shot_large` | allocation-only INT8, θ 0.85, 10,000 slots | 93.2 ± 0.2 | 85.0 | 7,902 |

For comparison, CLP scores 57.0 (1-shot) and 93.0 (25-shot).

Adaptive INT8 loses some accuracy at low thresholds, because small averaging steps fall below the 8-bit rounding floor; FP32 does not. `experiments/clp_snn_hw_comparison.py` sweeps the threshold for all simulator variants.

---

## Hardware Benchmarks (Jetson Orin Nano)

`benchmarks/benchmark.py` reproduces the latency and energy measurements in Table 1.
Requires a Jetson Orin Nano with JetPack SDK 6.2.1, PyTorch 2.4.0, and the `jtop` package.

```bash
python benchmarks/benchmark.py \
    --algorithm clp \
    --data_path data/1shot/X_train_1_shot_10.pt \
    --device_type orin \
    --compute_device cuda
```

See `benchmarks/README.md` for full instructions.

---

## Note on Loihi 2 Hardware Implementation

The Loihi 2 on-chip implementation of CLP-SNN requires access to Intel's proprietary Lava-Loihi framework and Loihi 2 hardware. `models/CLP_SNN_Loihi.py` emulates the deployed network, and `models/CLP_SNN.py` implements the learning rule's integer pipeline (see `LearningConnectionModelBitApproximate` in the lava repo). Researchers with Loihi 2 access may contact the authors for the hardware implementation code.

---

## Citation

```bibtex
@article{hajizada2026continual,
  title={Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network},
  author={Hajizada, Elvin and Rager, Danielle and Shea, Timothy and Campos-Macias, Leobardo and Wild, Andreas and H{\"u}llermeier, Eyke and Sandamirskaya, Yulia and Davies, Mike},
  journal={arXiv preprint arXiv:2511.01553},
  year={2026}
}
```
