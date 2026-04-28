"""
Latency and energy benchmarking for OCL algorithms on Jetson Orin Nano.

Reproduces Table 1 hardware measurements from "Real-time Continual Learning on Intel Loihi 2".

Requirements (Jetson Orin Nano only):
  - JetPack SDK 6.2.1, PyTorch 2.4.0
  - bm_utils package (jtop_stats, tegrastats_monitor, reporter, time_stats)
  - jtop installed: pip install jetson-stats

Usage:
  python benchmark.py --algorithm clp --data_path ../data/1shot/X_train_1_shot_10.npy \\
                      --device_type orin --compute_device cuda
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
parser.add_argument('--k_shot', type=int, default=1,
                    help='K-shot learning (default: 1)')
parser.add_argument('--seed', type=int, default=10,
                    help='Random seed (default: 10)')
parser.add_argument('--num_classes', type=int, default=40,
                    help='Number of classes (default: 40)')
parser.add_argument('--feature_size', type=int, default=1280,
                    help='Feature size (default: 1280)')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to data file containing X_train and y_train')
args = parser.parse_args()

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

# Convert to torch tensors and slice feature size
X_train = torch.from_numpy(X_train).to(device)[:, :feature_size]
y_train = torch.from_numpy(y_train).to(device)

# Normalize data for algorithms that need it
if args.algorithm in ['clp', 'slda', 'replay']:
    X_train = X_train / X_train.norm(dim=1, keepdim=True)

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
        device=device
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
        lambda_update_period=60,
        device=device
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
        device=device
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

# Determine number of epochs based on algorithm
num_epochs = 3 if args.algorithm == 'ncm' else 1

for ep in range(num_epochs):
    for x, y in train_loader:
        start_time = time.time()
        classifier.fit(x.view(feature_size,), y.view(1,), i)
        inference_time = time.time() - start_time
        time_stats.add(0.0, 0.0, inference_time)
        i += 1

if DEVICE == 'atom':
    cpu_stats.stop_monitoring()
    power_results = cpu_stats.get_only_stats()
    static_power = cpu_stats.compute_static_power()
else:
    jstats.stop()
    jresults = jstats.get_only_stats()
    static_power = jstats.compute_static_power()

time_results_stats = time_stats.get_stats()

experiments_results = {
    'mean_inference_time_us': time_results_stats[2] * 1e6,
    'mean_input_time_us': time_results_stats[0] * 1e6,
    'mean_output_time_us': time_results_stats[1] * 1e6,
    'accuracy': 'na',
    'validation_time_s': time.time() - time_start
}

if DEVICE == 'atom':
    experiments_results = reporter.compute_statistics_cpu(
        experiments_results, static_power=static_power, power=power_results)
    reporter.report(experiments_results, 'fp32', time_stats, cpu_stats)
else:
    experiments_results = reporter.compute_statistics_gpu(
        experiments_results, static_power=static_power, power=jresults)
    reporter.report(experiments_results, 'fp32', time_stats, jstats)

reporter.save_pickle_report()

print(f"\n{'='*60}")
print(f"Benchmarking completed for {args.algorithm.upper()} algorithm")
print(f"{'='*60}")
print(f"Model: {model_name}")
print(f"Dataset: X_train_{k}_shot_{seed}")
print(f"Total samples processed: {i}")
print(f"Mean inference time: {experiments_results['mean_inference_time_us']:.2f} μs")
print(f"Total validation time: {experiments_results['validation_time_s']:.2f} s")
print(f"Results saved to: {logs_path}")
print(f"{'='*60}\n")
