"""
Latency and energy benchmarking for OCL algorithms on Jetson Orin Nano.

Reproduces Table 1 hardware measurements from "Real-time Continual Learning on Intel Loihi 2".

Measurement methodology:
  - GPU latencies via CUDA events (torch.cuda.Event) with end-event sync;
    CPU latencies via time.time()
  - CUDA warmup stage (~20 samples) before the timed run; replay buffer and
    counters are reset after warmup
  - First N post-warmup samples excluded from statistics (--skip_first_n)
  - Reported mean latency is outlier-filtered at the 95th percentile
    (standard practice for real-time systems); p95/p99 also printed

Requirements (Jetson Orin Nano only):
  - JetPack SDK 6.2.1, PyTorch 2.4.0
  - bm_utils package (jtop_stats, tegrastats_monitor, reporter, time_stats)
  - jtop installed: pip install jetson-stats

Usage:
  python benchmark.py --algorithm clp --data_path ../data/1shot/X_train_1_shot_10.npy \\
                      --device_type orin --compute_device cuda
  python benchmark.py --algorithm slda --dtype fp16 --compute_device cuda ...
"""
import numpy as np
import sys
import torch
import time
import argparse
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

torch.set_printoptions(precision=2, sci_mode=False)

import os

# Add repo root to path so models/ is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Unified Benchmarking Script for Continual Learning Algorithms')
parser.add_argument('--algorithm', type=str, required=True,
                    choices=['clp', 'ncm', 'slda', 'replay'],
                    help='Learning algorithm to benchmark: clp, ncm, slda, or replay')
parser.add_argument('--logs_path', type=str, default=None,
                    help='Path to save logs and reports. If not provided, uses default path based on DEVICE.')
parser.add_argument('--device_type', type=str, default='orin',
                    choices=['orin', 'atom'],
                    help='Device type: orin or atom (default: orin)')
parser.add_argument('--compute_device', type=str, default='cpu',
                    choices=['cpu', 'cuda'],
                    help='Compute device: cpu or cuda (default: cpu)')
parser.add_argument('--dtype', type=str, default='fp32',
                    choices=['fp32', 'fp16'],
                    help='Precision: fp32 or fp16 (fp16 requires cuda; default: fp32)')
parser.add_argument('--k_shot', type=int, default=1,
                    help='K-shot learning (default: 1)')
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed (default: 10)')
parser.add_argument('--num_classes', type=int, default=40,
                    help='Number of classes (default: 40)')
parser.add_argument('--feature_size', type=int, default=1280,
                    help='Feature size (default: 1280)')
parser.add_argument('--num_samples', type=int, default=None,
                    help='Cap on number of samples to benchmark (default: all)')
parser.add_argument('--skip_first_n', type=int, default=10,
                    help='Skip first N post-warmup samples in statistics (default: 10)')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to data file containing X_train and y_train')
args = parser.parse_args()

use_fp16 = args.dtype == 'fp16'
if use_fp16 and args.algorithm == 'clp':
    parser.error('--dtype fp16 is not supported for CLP (fp16 applies to ncm, slda, replay)')
if use_fp16 and args.compute_device != 'cuda':
    parser.error('--dtype fp16 requires --compute_device cuda (Tensor Cores)')

# Import models based on algorithm choice
if args.algorithm == 'clp':
    from models.CLP import ContinuallyLearningPrototypes
elif args.algorithm == 'ncm':
    from models.NCM import NearestClassMean
elif args.algorithm == 'slda':
    from models.SLDA import StreamingLDA
elif args.algorithm == 'replay':
    from models.Replay import StreamingSoftmax

DEVICE = args.device_type
if DEVICE == 'atom':
    from bm_utils.cpu_stats import CPUStats
else:
    from bm_utils.jtop_stats import JTOPStats
    from bm_utils.tegrastats_monitor import TegraStatsMonitor

from bm_utils.reporter import Reporter
from bm_utils.time_stats import TimeStats

# General parameters for the experiment
batch_size = 1
seed = args.seed
num_classes = args.num_classes
test_order = None

device = args.compute_device
cl_times = []
i = 0
k = args.k_shot

feature_size = args.feature_size

# Load data
with open(args.data_path, 'rb') as f:
    X_train = np.load(f)
    y_train = np.load(f)

# Convert to torch tensors and slice feature size; the slice can produce a
# non-contiguous view, so enforce contiguity for efficient GEMM
X_train = torch.from_numpy(X_train)[:, :feature_size].contiguous().to(device)
y_train = torch.from_numpy(y_train).to(device)

# Normalize data for algorithms that need it
if args.algorithm in ['clp', 'slda', 'replay']:
    X_train = X_train / X_train.norm(dim=1, keepdim=True)

# Convert to FP16 upfront if requested - avoids repeated conversions in the loop
if use_fp16:
    print(f"Converting data to FP16 for {args.algorithm.upper()}")
    X_train = X_train.half().contiguous()

train_set = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=1,
                        num_workers=0, pin_memory=False, shuffle=False)

# Initialize classifier based on algorithm
if args.algorithm == 'clp':
    classifier = ContinuallyLearningPrototypes(
        feature_size,
        n_protos=300,
        num_classes=num_classes,
        backbone=None,
        alpha_init=1,
        sim_th_init=0.75,
        n_wta=1,
        k_hit=1,
        k_miss=0.5,
        adaptive_th=False,
        adaptive_protos=True,
        device=device
    )
    model_name = "CLP"

elif args.algorithm == 'ncm':
    classifier = NearestClassMean(
        feature_size,
        num_classes,
        backbone=None,
        device=device,
        use_fp16=use_fp16
    )
    model_name = "ncm"

elif args.algorithm == 'slda':
    streaming_update_sigma = True
    streaming_update_lambda = True
    classifier = StreamingLDA(
        feature_size,
        num_classes,
        backbone=None,
        shrinkage_param=1e-4,
        streaming_update_sigma=streaming_update_sigma,
        streaming_update_lambda=streaming_update_lambda,
        lambda_update_period=k,  # k=1: Lambda per sample; k=60: amortized
        device=device,
        use_fp16=use_fp16
    )
    model_name = "SLDA-periodic"

elif args.algorithm == 'replay':
    classifier = StreamingSoftmax(
        feature_size,
        num_classes,
        use_replay=True,
        backbone=None,
        lr=0.001,
        weight_decay=1e-5,
        replay_samples=50,
        max_buffer_size=800,
        device=device,
        use_fp16=use_fp16
    )
    model_name = "replay"

reporter_config = {
    'batch_size': batch_size,
    'model_name': model_name,
    'dataset': f"X_train_{k}_shot_{seed}",
    'device': 'orin-gpu' if device == 'cuda' else 'orin-cpu',
}

# Use command line argument if provided, otherwise use default based on DEVICE
if args.logs_path is not None:
    logs_path = args.logs_path
elif DEVICE == 'atom':
    logs_path = '/home/atom-01-gdc/ai.ncl.jetson-benchmarks/clp/reports/'
else:
    logs_path = '/clp/reports/'

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

reporter = Reporter(logs_path, reporter_config)

time_stats = TimeStats()

if DEVICE == 'atom':
    cpu_stats = CPUStats(os.getpid(), time_interval=0.2)
else:
    jstats = TegraStatsMonitor(frequency=0.05, docker=True)
    jstats.start()

time_start = time.time()

# Warmup for CUDA operations (kernel compilation, allocator, autotuning);
# especially important for FP16
starter = ender = None
if device == 'cuda':
    print("Warming up CUDA kernels...")
    warmup_samples = min(20, len(train_loader))
    for idx, (x, y) in enumerate(train_loader):
        if idx >= warmup_samples:
            break
        classifier.fit(x.view(feature_size,), y.view(1,), idx)
    torch.cuda.synchronize()
    print(f"Warmup complete ({warmup_samples} samples)")

    # Reset counters and clear buffer after warmup
    time_stats = TimeStats()
    if hasattr(classifier, 'latent_dict'):
        classifier.latent_dict.clear()
        classifier.rehearsal_ixs.clear()
        classifier.class_id_to_item_ix_dict.clear()
        print("Replay buffer cleared after warmup")

    # CUDA events for precise timing (no per-sample synchronize overhead)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

# Determine number of epochs based on algorithm
num_epochs = 3 if args.algorithm == 'ncm' else 1

# Store latencies for statistical analysis
all_latencies = []          # every timed sample (ms)
latencies_for_stats = []    # samples after skip_first_n (ms)
sample_count = 0

for ep in range(num_epochs):
    for x, y in train_loader:
        if args.num_samples is not None and sample_count >= args.num_samples:
            break

        if device == 'cuda':
            # CUDA events time the actual GPU work; only the end event syncs
            starter.record()
            classifier.fit(x.view(feature_size,), y.view(1,), i)
            ender.record()
            ender.synchronize()
            inference_time = starter.elapsed_time(ender) / 1000  # ms -> s
        else:
            start_time = time.time()
            classifier.fit(x.view(feature_size,), y.view(1,), i)
            inference_time = time.time() - start_time

        all_latencies.append(inference_time * 1000)
        if sample_count >= args.skip_first_n:
            time_stats.add(0.0, 0.0, inference_time)
            latencies_for_stats.append(inference_time * 1000)

        i += 1
        sample_count += 1
    if args.num_samples is not None and sample_count >= args.num_samples:
        break

# Latency distribution statistics with 95th-percentile outlier filtering
filtered_mean_latency_ms = None
if all_latencies:
    stats_np = np.array(latencies_for_stats) if latencies_for_stats else np.array(all_latencies)
    p95 = np.percentile(stats_np, 95)
    filtered = stats_np[stats_np <= p95]
    filtered_mean_latency_ms = np.mean(filtered)

    print(f"\n{'='*60}")
    print("Latency Distribution Statistics:")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(all_latencies)}")
    print(f"Samples used for stats (after skipping first {args.skip_first_n}): {len(stats_np)}")
    print(f"Outliers removed (beyond p95 {p95:.2f}ms): {len(stats_np) - len(filtered)}")
    print(f"  Mean latency:        {np.mean(stats_np):.3f} ms")
    print(f"  Std deviation:       {np.std(stats_np):.3f} ms")
    print(f"  Median latency:      {np.median(stats_np):.3f} ms")
    print(f"  Min / Max:           {np.min(stats_np):.3f} / {np.max(stats_np):.3f} ms")
    print(f"  95th / 99th pct:     {np.percentile(stats_np, 95):.3f} / {np.percentile(stats_np, 99):.3f} ms")
    print(f"  Filtered mean (p95): {filtered_mean_latency_ms:.3f} ms  <- reported")
    print(f"{'='*60}\n")

if DEVICE == 'atom':
    cpu_stats.stop_monitoring()
    power_results = cpu_stats.get_only_stats()
    static_power = cpu_stats.compute_static_power()
else:
    jstats.stop()
    jresults = jstats.get_only_stats()
    static_power = jstats.compute_static_power()

time_results_stats = time_stats.get_stats()

# Report the p95-filtered mean; fall back to the raw mean if unavailable
if filtered_mean_latency_ms is not None:
    mean_inference_time_us = filtered_mean_latency_ms * 1000  # ms -> us
else:
    mean_inference_time_us = time_results_stats[2] * 1e6

experiments_results = {
    'mean_inference_time_us': mean_inference_time_us,
    'mean_input_time_us': time_results_stats[0] * 1e6,
    'mean_output_time_us': time_results_stats[1] * 1e6,
    'accuracy': 'na',
    'validation_time_s': time.time() - time_start
}

precision_tag = 'fp16' if use_fp16 else 'fp32'
if DEVICE == 'atom':
    experiments_results = reporter.compute_statistics_cpu(
        experiments_results, static_power=static_power, power=power_results)
    reporter.report(experiments_results, precision_tag, time_stats, cpu_stats)
else:
    experiments_results = reporter.compute_statistics_gpu(
        experiments_results, static_power=static_power, power=jresults)
    reporter.report(experiments_results, precision_tag, time_stats, jstats)

reporter.save_pickle_report()

print(f"\n{'='*60}")
print(f"Benchmarking completed for {args.algorithm.upper()} algorithm ({precision_tag})")
print(f"{'='*60}")
print(f"Model: {model_name}")
print(f"Dataset: X_train_{k}_shot_{seed}")
print(f"Total samples processed: {i}")
print(f"Mean inference time (p95-filtered): {experiments_results['mean_inference_time_us']:.2f} μs")
print(f"Total validation time: {experiments_results['validation_time_s']:.2f} s")
print(f"Results saved to: {logs_path}")
print(f"{'='*60}\n")
