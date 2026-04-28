# Hardware Benchmarks (Jetson Orin Nano)

`benchmark.py` reproduces the latency and energy measurements in Table 1 of the paper.

## Hardware Requirements

- NVIDIA Jetson Orin Nano 8GB
- JetPack SDK 6.2.1 (15W TDP mode)
- PyTorch 2.4.0
- `jetson-stats` package: `pip install jetson-stats` (provides `jtop`)

## Usage

```bash
python benchmark.py \
    --algorithm <clp|ncm|slda|replay> \
    --data_path <path/to/X_train.npy> \
    --device_type orin \
    --compute_device <cpu|cuda> \
    [--k_shot 1] \
    [--seed 10] \
    [--num_classes 40]
```

### Example — reproduce CLP GPU results

```bash
python benchmark.py \
    --algorithm clp \
    --data_path ../data/1shot/X_train_1_shot_10.npy \
    --device_type orin \
    --compute_device cuda
```

## Expected Output

Results are saved to a `reports/` directory with per-sample latency (ms) and energy (mJ) measurements, matching Table 1.

## Notes

- Run with the device in **15W TDP mode** (matches paper measurements).
- Energy values include `CPU_GPU_CV` and SOC components.
- FP16 variants: add `--dtype fp16` if supported by the script version.
- The benchmark excludes I/O between host and device (data is pre-loaded).
