# Hardware Benchmarks (Jetson Orin Nano)

`benchmark.py` reproduces the latency and energy measurements in Table 1 of the paper.

## Hardware Requirements

- NVIDIA Jetson Orin Nano 8GB
- JetPack SDK 6.2.1 (15W TDP mode)
- PyTorch 2.4.0
- `jetson-stats` package: `pip install jetson-stats` (provides `jtop`)
- `bm_utils` package (timing/power monitors and report writer)

## Quick Start (full Table 1 suite)

```bash
sudo bash benchmarks/prepare_system_for_benchmarking.sh   # clear caches, lock clocks, thermal check
bash benchmarks/run_openloris_benchmarks.sh               # all 13 Table 1 runs
```

## Usage (single run)

```bash
python benchmark.py \
    --algorithm <clp|ncm|slda|replay> \
    --data_path <path/to/X_train.npy> \
    --device_type orin \
    --compute_device <cpu|cuda> \
    [--dtype <fp32|fp16>] \
    [--k_shot 1] \
    [--seed 10] \
    [--num_classes 40] \
    [--num_samples N] \
    [--skip_first_n 10]
```

`--dtype fp16` enables half-precision (NCM, SLDA, Replay; requires `--compute_device cuda`).
CLP runs in FP32 only. For SLDA, `--k_shot` also sets the Λ recompute period
(`k=1`: inversion every sample; `k=60`: amortized).

### Example — reproduce CLP GPU results

```bash
python benchmark.py \
    --algorithm clp \
    --data_path ../data/1shot/X_train_1_shot_10.npy \
    --device_type orin \
    --compute_device cuda
```

### Example — SLDA in FP16

```bash
python benchmark.py \
    --algorithm slda \
    --dtype fp16 \
    --data_path ../data/1shot/X_train_1_shot_10.npy \
    --device_type orin \
    --compute_device cuda
```

## Measurement Methodology

- **GPU timing** uses CUDA events (`torch.cuda.Event`) with an end-event sync,
  so asynchronous kernel launches are measured correctly; CPU timing uses
  `time.time()`.
- A **warmup stage** (~20 samples) runs before the timed loop on CUDA (kernel
  compilation, allocator, autotuning); counters and the replay buffer are
  reset afterwards.
- The first `--skip_first_n` post-warmup samples are excluded from statistics.
- The reported mean latency is **outlier-filtered at the 95th percentile**
  (standard practice for real-time systems); the raw mean, median, p95, and
  p99 are also printed.
- Feature tensors are made contiguous before the run for efficient GEMM.

## Expected Output

Results are saved to a `reports/` directory with per-sample latency (ms) and energy (mJ) measurements, matching Table 1.

## Notes

- Run with the device in **15W TDP mode** (matches paper measurements).
- Energy values include `CPU_GPU_CV` and SOC components.
- The benchmark excludes I/O between host and device (data is pre-loaded).
