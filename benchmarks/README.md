# Hardware Benchmarks (Jetson Orin Nano)

`benchmark.py` reproduces the latency and energy measurements in Table 1 of the paper.

Running this on the device? Start with `ORIN_RUNBOOK.md` in this folder: it is
the step-by-step operator guide (setup, sanity run, full suite, what to send back).

## Hardware Requirements

- NVIDIA Jetson Orin Nano 8GB
- JetPack SDK 6.2.1 (15W TDP mode)
- PyTorch 2.4.0
- `jetson-stats` package: `pip install jetson-stats` (provides `jtop`)
- `bm_utils` package (timing/power monitors and report writer)

## Quick Start (full Table 1 suite)

```bash
sudo bash benchmarks/prepare_system_for_benchmarking.sh   # clear caches, lock clocks, thermal check
bash benchmarks/run_openloris_benchmarks.sh               # all 14 Table 1 / SI runs
```

## Usage (single run)

```bash
python benchmark.py \
    --algorithm <clp|ncm|slda|slda_rank1|replay> \
    --op <fit|predict|both> \
    --data_path <path/to/X_train.npy> \
    --device_type orin \
    --compute_device <cpu|cuda> \
    [--dtype <fp32|fp16>] \
    [--lambda_period 1] \
    [--seed 10] \
    [--num_classes 40] \
    [--num_samples N] \
    [--skip_first_n 10] \
    [--no-power]
```

- `--algorithm slda_rank1` is the paper's SLDA baseline (exact rank-1 precision
  maintenance, no inversion ever); `--algorithm slda` is the naive reference
  implementation with an explicit `torch.linalg.inv` on every fit
  (`--lambda_period 1`, the SI cost-anchor setting and the default).
- `--op` selects the timed primitive: `fit` measures C_upd, `predict` measures
  C_qry (after an untimed training pass), `both` runs a timed fit pass followed
  by a timed predict pass on the trained model. Predict reports are written
  under `<model_name>-qry`.
- `--dtype fp16` enables half-precision (NCM, naive SLDA, Replay; requires
  `--compute_device cuda`). CLP runs in FP32 only. `slda_rank1` requires FP32
  state (Sherman-Morrison drift; see the SI drift study). For naive SLDA,
  FP16 applies to the GEMMs only; the Sigma/Lambda/mean statistics stay FP32
  because FP16 accumulation stalls and collapses accuracy (gate:
  `experiments/slda_fp16_gate.py`, FP16-state agreement drops below 2% by the
  end of the 1-shot stream; mixed-precision passes at 99.94%+).
- `--compile` is the compiled-overhead control run (Table 1 run-matrix item 4):
  it wraps `fit`/`predict` in `torch.compile(mode="reduce-overhead")`, which
  lowers to CUDA Graphs and strips per-kernel Python dispatch and launch
  overhead. Run it for NCM on cuda (`ncm-compiled` report). Warmup is raised
  to 50 samples automatically so compilation and graph capture happen before
  timing. `NCM.predict` contains two expected graph-break points (the
  unseen-class mask and the final `.cpu()` transfer); if the capture degrades
  to eager (latency matches the uncompiled run), fall back to a manual
  `torch.cuda.CUDAGraph` capture of the distance GEMM.
- `--no-power` skips the `bm_utils` monitors and report writer and prints
  latency statistics only; use it for smoke runs on machines without the
  Jetson stack.

### Example — SLDA rank-1, update and query cost

```bash
python benchmark.py \
    --algorithm slda_rank1 \
    --op both \
    --data_path ../data/1shot/X_train_1_shot_10.pt \
    --device_type orin \
    --compute_device cuda
```

## Measurement Methodology

- **GPU timing** uses CUDA events (`torch.cuda.Event`) with an end-event sync,
  so asynchronous kernel launches are measured correctly; CPU timing uses
  `time.perf_counter()`.
- A **warmup stage** (default 20 samples, `--warmup_samples`) runs before every
  timed pass on CPU and CUDA alike (allocator, caches, kernel compilation);
  the replay buffer is reset after fit-warmup.
- **Power/energy monitors start after warmup**, so warmup work never
  contaminates energy statistics or wall time.
- The first `--skip_first_n` post-warmup samples are excluded from statistics.
- The **reported latency is the raw mean** over post-skip samples. For
  streaming learners with intrinsic periodic spikes (naive-SLDA Λ refresh,
  replay eviction, CLP allocation) the raw mean is the amortized per-sample
  cost by definition; percentile filtering would delete exactly those samples.
  Median, p95, p99, and a p95-filtered mean (OS-jitter diagnostic only) are
  printed and saved alongside.
- Feature tensors are made contiguous before the run for efficient GEMM.
- Model-side debug instrumentation (e.g. the rank-1 Sherman-Morrison
  denominator assertion) is disabled in benchmark runs via `debug_checks=False`
  and excluded from measured cost; it stays enabled in accuracy runs and tests.

## Expected Output

Results are saved to a `reports/` directory with per-sample latency (ms) and
energy (mJ) measurements, matching Table 1. One `exp_N/` folder is written per
timed op: `<model_name>` for fit (C_upd), `<model_name>-qry` for predict (C_qry).

## Notes

- Run with the device in **15W TDP mode** (matches paper measurements).
- Energy values include `CPU_GPU_CV` and SOC components.
- The benchmark excludes I/O between host and device (data is pre-loaded).
