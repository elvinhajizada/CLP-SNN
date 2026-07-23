# Orin Nano Benchmark Runbook (CLP-SNN Table 1)

One pass through this document produces every hardware number for the paper:
per-sample latency (C_upd = fit, C_qry = predict) and energy for all
baselines, under a single unified harness.

## Setup

1. Check out the two repos **side by side** (the harness finds `bm_utils` in
   any sibling folder named `ai.ncl.jetson-benchmarks*`):

   ```
   <workdir>/CLP-SNN/
   <workdir>/ai.ncl.jetson-benchmarks/     # only bm_utils/ is used
   ```

2. Environment: JetPack 6.2.1, 15W power mode, PyTorch 2.4.0, plus:

   ```bash
   pip install jetson-stats pandas
   ```

3. All input data ships inside `CLP-SNN/data/` (git-tracked). Nothing else to
   download.

Note: the harness starts `tegrastats` **without sudo** (the monitor's
`docker=True` path). If VDD_IN comes back empty in the sanity run, run the
python commands with sudo instead.

## Step 1: prepare the system

```bash
cd CLP-SNN
sudo bash benchmarks/prepare_system_for_benchmarking.sh   # caches, clocks, thermal check
```

## Step 2: sanity run (~2 min)

One short powered run to validate the jtop/tegrastats/Reporter path before
committing to the full suite:

```bash
python3 benchmarks/benchmark.py \
    --algorithm ncm --op both \
    --data_path data/1shot/X_train_1_shot_10.pt \
    --compute_device cuda \
    --logs_path benchmarks/reports/ \
    --num_samples 200
```

Check that `benchmarks/reports/exp_0/` and `exp_1/` exist and contain
nonzero power/energy statistics. If power monitoring misbehaves, add
`--no-power` to isolate: it prints latency only and skips all monitors.
Delete the sanity `exp_*` folders before Step 3.

## Step 3: full suite (14 runs, roughly an hour)

```bash
bash benchmarks/run_openloris_benchmarks.sh 2>&1 | tee benchmarks/reports/console.log
```

The suite (all on the 1-shot seed-10 stream, `--op both`):

| # | Runs | Configs |
|---|------|---------|
| 1 | CLP | GPU FP32, CPU FP32 |
| 2 | NCM | GPU FP32, GPU FP16, CPU FP32 |
| 3 | NCM compiled control | GPU FP32 + `--compile` |
| 4 | Replay | GPU FP32, GPU FP16, CPU FP32 |
| 5 | SLDA naive (per-sample inversion) | GPU FP32, GPU FP16, CPU FP32 |
| 6 | SLDA rank-1 (paper baseline) | GPU FP32, CPU FP32 |

Failed runs don't abort the suite; the summary at the end lists them.

### The compiled NCM run (one thing to watch)

Run 3 wraps fit/predict in `torch.compile(mode="reduce-overhead")` (CUDA
Graphs) to bound Python dispatch overhead. Warmup is raised to 50 samples
automatically. Please check its console output for recompile or graph-break
warnings, and note whether its latency actually drops below the plain NCM GPU
run. If it doesn't (capture likely degraded to eager at the unseen-class mask
or the final `.cpu()`), just report that; a manual CUDA Graph fallback exists
as a plan B on our side.

## What to send back

- The whole `benchmarks/reports/` folder (one `exp_N/` per timed op:
  `<model>` = fit/C_upd, `<model>-qry` = predict/C_qry)
- `console.log` from Step 3
- The `nvpmodel -q` output and jtop-reported temperatures if anything looked
  thermally suspicious

## Reference

Full flag documentation and measurement methodology: `benchmarks/README.md`.
Single-run template for re-running anything in isolation:

```bash
python3 benchmarks/benchmark.py --algorithm <clp|ncm|slda|slda_rank1|replay> \
    --op both --data_path data/1shot/X_train_1_shot_10.pt \
    --compute_device <cpu|cuda> [--dtype fp16] [--compile] \
    --logs_path benchmarks/reports/
```
