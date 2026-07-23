"""
Latency and energy benchmarking for OCL algorithms on Jetson Orin Nano.

Reproduces Table 1 hardware measurements from "Real-time Continual Learning on Intel Loihi 2".

Measurement methodology:
  - GPU latencies via CUDA events (torch.cuda.Event) with end-event sync;
    CPU latencies via time.perf_counter()
  - Warmup stage (~20 samples) before every timed pass, on CPU and CUDA alike
    (allocator, caches, kernel compilation/autotuning); the replay buffer is
    reset after fit-warmup
  - Power/energy monitors start AFTER warmup, so warmup work never
    contaminates energy statistics
  - First N post-warmup samples excluded from statistics (--skip_first_n)
  - The headline latency is the RAW MEAN over post-skip samples. For streaming
    learners with intrinsic periodic spikes (replay eviction, CLP prototype
    allocation) the raw mean IS the amortized per-sample cost; percentile
    filtering would delete exactly those samples.
    Median/p95/p99 and a p95-outlier-filtered mean (OS-jitter diagnostic only,
    never reported) are printed and saved alongside.
  - --op selects the timed primitive: fit (C_upd), predict (C_qry), or both
    (timed fit pass first, then timed predict on the trained model)

Requirements (Jetson Orin Nano only):
  - JetPack SDK 6.2.1, PyTorch 2.4.0
  - bm_utils package (jtop_stats, tegrastats_monitor, reporter, time_stats)
  - jtop installed: pip install jetson-stats

Usage:
  python benchmark.py --algorithm slda_rank1 --op both \\
                      --data_path ../data/1shot/X_train_1_shot_10.pt \\
                      --device_type orin --compute_device cuda
  python benchmark.py --algorithm slda --dtype fp16 ...
  Local smoke test (no Jetson / no bm_utils): add --no-power
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

# bm_utils lives in the sibling ai.ncl.jetson-benchmarks repo; make it
# importable when that repo is checked out next to this one
for _cand in sorted(_REPO.parent.glob('ai.ncl.jetson-benchmarks*')):
    if (_cand / 'bm_utils').is_dir():
        sys.path.insert(0, str(_cand))
        break

# Parse command line arguments
parser = argparse.ArgumentParser(description='Unified Benchmarking Script for Continual Learning Algorithms')
parser.add_argument('--algorithm', type=str, required=True,
                    choices=['clp', 'ncm', 'slda', 'slda_rank1', 'replay'],
                    help='Learning algorithm to benchmark. "slda" is the naive '
                         'reference implementation (explicit inverse every '
                         '--lambda_period fits); "slda_rank1" is the paper\'s '
                         'SLDA baseline (exact rank-1 precision maintenance).')
parser.add_argument('--op', type=str, default='fit',
                    choices=['fit', 'predict', 'both'],
                    help='Timed primitive: fit (C_upd), predict (C_qry), or both '
                         '(fit pass first, then predict on the trained model)')
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
parser.add_argument('--lambda_period', type=int, default=1,
                    help='Naive SLDA only: Lambda recompute period in fits '
                         '(1 = per-sample inversion, the SI cost-anchor setting). '
                         'Default: 1')
parser.add_argument('--k_shot', type=int, default=1,
                    help='K-shot protocol of the dataset, used for report labeling only (default: 1)')
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed (default: 10)')
parser.add_argument('--num_classes', type=int, default=40,
                    help='Number of classes (default: 40)')
parser.add_argument('--feature_size', type=int, default=1280,
                    help='Feature size (default: 1280)')
parser.add_argument('--num_samples', type=int, default=None,
                    help='Cap on number of samples per timed pass (default: all)')
parser.add_argument('--skip_first_n', type=int, default=10,
                    help='Skip first N post-warmup samples in statistics (default: 10)')
parser.add_argument('--warmup_samples', type=int, default=20,
                    help='Untimed warmup samples before each timed pass (default: 20)')
parser.add_argument('--compile', action='store_true',
                    help='Compiled-overhead control: wrap fit/predict in '
                         'torch.compile(mode="reduce-overhead") (CUDA Graphs) '
                         'to bound Python dispatch overhead. Intended for the '
                         'NCM control run on cuda; reports are tagged '
                         '"-compiled". Warmup is raised to >=50 samples so '
                         'compilation and graph capture finish before timing.')
parser.add_argument('--no-power', dest='no_power', action='store_true',
                    help='Skip bm_utils power monitors and report writer; print '
                         'latency statistics only (for local smoke runs off-device)')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to data file containing X_train and y_train')
args = parser.parse_args()

use_fp16 = args.dtype == 'fp16'
if use_fp16 and args.algorithm == 'clp':
    parser.error('--dtype fp16 is not supported for CLP (fp16 applies to ncm, slda, replay)')
if use_fp16 and args.algorithm == 'slda_rank1':
    parser.error('--dtype fp16 is not supported for slda_rank1: the maintained '
                 'Sherman-Morrison state requires FP32 (see the t4 drift study)')
if use_fp16 and args.compute_device != 'cuda':
    parser.error('--dtype fp16 requires --compute_device cuda (Tensor Cores)')

# Import models based on algorithm choice
if args.algorithm == 'clp':
    from models.CLP import ContinuallyLearningPrototypes
elif args.algorithm == 'ncm':
    from models.NCM import NearestClassMean
elif args.algorithm == 'slda':
    from models.SLDA import StreamingLDA
elif args.algorithm == 'slda_rank1':
    from models.SLDA import RankOneSLDA
elif args.algorithm == 'replay':
    from models.Replay import StreamingSoftmax

DEVICE = args.device_type
if not args.no_power:
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

device = args.compute_device
k = args.k_shot

feature_size = args.feature_size

# Load data. Accepted layouts:
#   - X_*.pt with a sibling y_*.pt (torch tensors, original Jetson layout)
#   - X_*.npy with a sibling y_*.npy
#   - single .npy holding X then y (two arrays saved into one file)
_data_path = Path(args.data_path)
if _data_path.suffix == '.pt':
    X_train = torch.load(_data_path, map_location='cpu', weights_only=True)
    y_train = torch.load(
        _data_path.with_name(_data_path.name.replace('X_train', 'y_train')),
        map_location='cpu', weights_only=True)
else:
    with open(_data_path, 'rb') as f:
        X_train = torch.from_numpy(np.load(f))
        try:
            y_train = torch.from_numpy(np.load(f))
        except (ValueError, EOFError, OSError):
            y_train = None
    if y_train is None:
        y_train = torch.from_numpy(np.load(
            _data_path.with_name(_data_path.name.replace('X_train', 'y_train'))))

# Slice feature size; the slice can produce a non-contiguous view, so enforce
# contiguity for efficient GEMM
X_train = X_train[:, :feature_size].contiguous().to(device)
y_train = y_train.to(device)

# Normalize data for algorithms that need it
if args.algorithm in ['clp', 'slda', 'slda_rank1', 'replay']:
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
    classifier = StreamingLDA(
        feature_size,
        num_classes,
        backbone=None,
        shrinkage_param=1e-4,
        streaming_update_sigma=True,
        streaming_update_lambda=True,
        lambda_update_period=args.lambda_period,
        device=device,
        use_fp16=use_fp16
    )
    model_name = f"SLDA-naive-rho{args.lambda_period}"

elif args.algorithm == 'slda_rank1':
    classifier = RankOneSLDA(
        feature_size,
        num_classes,
        backbone=None,
        ridge_param=1.0,
        device=device,
        debug_checks=False  # denom instrumentation excluded from measured cost
    )
    model_name = "SLDA-rank1"

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

# Compiled-overhead control (run-matrix item 4): reduce-overhead mode uses
# CUDA Graphs to strip per-kernel Python dispatch and launch overhead. Known
# graph-break points fall back to eager per segment (NCM.predict's
# unseen-class mask and trailing .cpu() sync); check the torch.compile logs
# on device to confirm what was captured.
if args.compile:
    classifier.fit = torch.compile(classifier.fit, mode='reduce-overhead')
    classifier.predict = torch.compile(classifier.predict, mode='reduce-overhead')
    model_name += '-compiled'
    if args.warmup_samples < 50:
        print(f"--compile: raising warmup from {args.warmup_samples} to 50 "
              f"samples (compilation + CUDA Graph capture)")
        args.warmup_samples = 50

# Use command line argument if provided, otherwise use default based on DEVICE
if args.logs_path is not None:
    logs_path = args.logs_path
elif DEVICE == 'atom':
    logs_path = '/home/atom-01-gdc/ai.ncl.jetson-benchmarks/clp/reports/'
else:
    logs_path = '/clp/reports/'

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

precision_tag = 'fp16' if use_fp16 else 'fp32'


# ---------------------------------------------------------------------------
# Timed primitives and passes
# ---------------------------------------------------------------------------

def fit_op(x, y, idx):
    classifier.fit(x.view(feature_size,), y.view(1,), idx)


def predict_op(x, y, idx):
    classifier.predict(x)


def run_untimed_pass(op_fn, epochs):
    idx = 0
    for ep in range(epochs):
        for x, y in train_loader:
            op_fn(x, y, idx)
            idx += 1


def warmup(op_fn, phase_label):
    """Untimed warmup on CPU and CUDA alike (allocator, caches, autotuning)."""
    n = min(args.warmup_samples, len(train_loader))
    print(f"Warming up {phase_label} ({n} samples)...")
    for idx, (x, y) in enumerate(train_loader):
        if idx >= n:
            break
        op_fn(x, y, idx)
    if device == 'cuda':
        torch.cuda.synchronize()


def reset_replay_buffer():
    if hasattr(classifier, 'reset_buffer'):
        classifier.reset_buffer()
        print("Replay buffer cleared after warmup")


def timed_pass(op_fn, epochs, time_stats=None):
    """Run op_fn over the stream, returning per-sample latencies in ms."""
    latencies_ms = []
    sample_count = 0
    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    for ep in range(epochs):
        for x, y in train_loader:
            if args.num_samples is not None and sample_count >= args.num_samples:
                break
            if device == 'cuda':
                # CUDA events time the actual GPU work; only the end event syncs
                starter.record()
                op_fn(x, y, sample_count)
                ender.record()
                ender.synchronize()
                elapsed_s = starter.elapsed_time(ender) / 1000  # ms -> s
            else:
                t0 = time.perf_counter()
                op_fn(x, y, sample_count)
                elapsed_s = time.perf_counter() - t0

            latencies_ms.append(elapsed_s * 1000)
            if time_stats is not None and sample_count >= args.skip_first_n:
                time_stats.add(0.0, 0.0, elapsed_s)
            sample_count += 1
        if args.num_samples is not None and sample_count >= args.num_samples:
            break
    return latencies_ms


def latency_statistics(latencies_ms, phase_label):
    """Raw-mean headline plus distribution stats.

    The raw mean is the reported number: it is the amortized per-sample cost,
    which by definition includes intrinsic periodic spikes (replay eviction,
    prototype allocation). The p95-filtered mean is printed
    only as an OS-jitter diagnostic and is never the reported value.
    """
    stats_np = np.array(latencies_ms[args.skip_first_n:]) \
        if len(latencies_ms) > args.skip_first_n else np.array(latencies_ms)
    p95 = np.percentile(stats_np, 95)
    filtered = stats_np[stats_np <= p95]

    stats = {
        'n_total': len(latencies_ms),
        'n_stats': len(stats_np),
        'mean_ms': float(np.mean(stats_np)),
        'std_ms': float(np.std(stats_np)),
        'median_ms': float(np.median(stats_np)),
        'min_ms': float(np.min(stats_np)),
        'max_ms': float(np.max(stats_np)),
        'p95_ms': float(p95),
        'p99_ms': float(np.percentile(stats_np, 99)),
        'p95_filtered_mean_ms': float(np.mean(filtered)),
    }

    print(f"\n{'='*60}")
    print(f"Latency Distribution Statistics [{phase_label}]:")
    print(f"{'='*60}")
    print(f"Total samples processed: {stats['n_total']}")
    print(f"Samples used for stats (after skipping first {args.skip_first_n}): {stats['n_stats']}")
    print(f"  Mean latency:        {stats['mean_ms']:.3f} ms  <- reported (amortized cost)")
    print(f"  Std deviation:       {stats['std_ms']:.3f} ms")
    print(f"  Median latency:      {stats['median_ms']:.3f} ms")
    print(f"  Min / Max:           {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms")
    print(f"  95th / 99th pct:     {stats['p95_ms']:.3f} / {stats['p99_ms']:.3f} ms")
    print(f"  p95-filtered mean:   {stats['p95_filtered_mean_ms']:.3f} ms  (jitter diagnostic only)")
    print(f"{'='*60}\n")
    return stats


def start_monitors():
    if args.no_power:
        return None
    if DEVICE == 'atom':
        mon = CPUStats(os.getpid(), time_interval=0.2)
        return mon
    mon = TegraStatsMonitor(frequency=0.05, docker=True)
    mon.start()
    return mon


def stop_monitors(mon):
    if mon is None:
        return None, None
    if DEVICE == 'atom':
        mon.stop_monitoring()
        return mon.get_only_stats(), mon.compute_static_power()
    mon.stop()
    return mon.get_only_stats(), mon.compute_static_power()


def report_phase(phase, stats, time_stats, mon, power_results, static_power, wall_time_s):
    """Write one bm_utils report entry for a timed pass (skipped with --no-power)."""
    if args.no_power:
        return
    phase_model_name = model_name if phase == 'fit' else model_name + '-qry'
    reporter = Reporter(logs_path, {
        'batch_size': batch_size,
        'model_name': phase_model_name,
        'dataset': f"X_train_{k}_shot_{seed}",
        'device': 'orin-gpu' if device == 'cuda' else 'orin-cpu',
    })
    experiments_results = {
        'mean_inference_time_us': stats['mean_ms'] * 1000,   # raw mean (amortized)
        'median_inference_time_us': stats['median_ms'] * 1000,
        'p95_inference_time_us': stats['p95_ms'] * 1000,
        'p99_inference_time_us': stats['p99_ms'] * 1000,
        'p95_filtered_mean_us': stats['p95_filtered_mean_ms'] * 1000,  # diagnostic
        'mean_input_time_us': 0.0,
        'mean_output_time_us': 0.0,
        'accuracy': 'na',
        'validation_time_s': wall_time_s,
    }
    if DEVICE == 'atom':
        experiments_results = reporter.compute_statistics_cpu(
            experiments_results, static_power=static_power, power=power_results)
        reporter.report(experiments_results, precision_tag, time_stats, mon)
    else:
        experiments_results = reporter.compute_statistics_gpu(
            experiments_results, static_power=static_power, power=power_results)
        reporter.report(experiments_results, precision_tag, time_stats, mon)
    reporter.save_pickle_report()


# ---------------------------------------------------------------------------
# Phase schedule
# ---------------------------------------------------------------------------

phases = ['fit', 'predict'] if args.op == 'both' else [args.op]
fit_epochs = 3 if args.algorithm == 'ncm' else 1

# C_qry must be measured on a trained model; if fit is not itself a timed
# phase, train untimed first.
if args.op == 'predict':
    print("Training model (untimed) before C_qry measurement...")
    run_untimed_pass(fit_op, fit_epochs)
    if device == 'cuda':
        torch.cuda.synchronize()

phase_stats = {}
for phase in phases:
    op_fn = fit_op if phase == 'fit' else predict_op
    epochs = fit_epochs if phase == 'fit' else 1

    warmup(op_fn, phase)
    if phase == 'fit':
        # Timed fit pass starts from fresh-stream buffer state
        reset_replay_buffer()

    # Monitors start only now: warmup is excluded from energy and wall time
    time_stats = TimeStats() if not args.no_power else None
    mon = start_monitors()
    time_start = time.time()

    latencies_ms = timed_pass(op_fn, epochs, time_stats)

    wall_time_s = time.time() - time_start
    power_results, static_power = stop_monitors(mon)

    stats = latency_statistics(latencies_ms, phase)
    phase_stats[phase] = stats
    report_phase(phase, stats, time_stats, mon, power_results, static_power, wall_time_s)

print(f"\n{'='*60}")
print(f"Benchmarking completed for {args.algorithm.upper()} ({precision_tag}, op={args.op})")
print(f"{'='*60}")
print(f"Model: {model_name}")
print(f"Dataset: X_train_{k}_shot_{seed}")
for phase, stats in phase_stats.items():
    label = 'C_upd (fit)' if phase == 'fit' else 'C_qry (predict)'
    print(f"{label}: mean {stats['mean_ms']*1000:.2f} us | median {stats['median_ms']*1000:.2f} us "
          f"| p99 {stats['p99_ms']*1000:.2f} us over {stats['n_stats']} samples")
if not args.no_power:
    print(f"Results saved to: {logs_path}")
print(f"{'='*60}\n")
