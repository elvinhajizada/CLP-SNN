#!/bin/bash

# Run the full OpenLORIS Table 1 benchmark suite (latency + energy) on Jetson Orin.
# Every run measures both C_upd (fit) and C_qry (predict) via --op both.
# Usage (from repo root, after: sudo bash benchmarks/prepare_system_for_benchmarking.sh):
#   bash benchmarks/run_openloris_benchmarks.sh

set -e

trap 'echo ""; echo "Interrupted! Cleaning up..."; exit 130' INT TERM

echo "=============================================="
echo "OpenLORIS Benchmarking Suite (CLP-SNN Table 1)"
echo "=============================================="
echo ""

# Run from the repo root
cd "$(dirname "$0")/.."

# Configuration
DATA_PATH="${DATA_PATH:-data/1shot/X_train_1_shot_10.pt}"
LOGS_PATH="${LOGS_PATH:-benchmarks/reports/}"
FEATURE_SIZE=1280
NUM_CLASSES=40
SEED=10

TOTAL_RUNS=14  # CLP(2) + NCM(3) + NCM-compiled(1) + Replay(3) + SLDA-naive per-sample(3) + SLDA-rank1(2)
CURRENT_RUN=0
FAILED_RUNS=()

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Logs Path: $LOGS_PATH"
echo "  Feature Size: $FEATURE_SIZE"
echo "  Num Classes: $NUM_CLASSES"
echo "  Seed: $SEED"
echo "  Total Runs: $TOTAL_RUNS"
echo ""
echo "=============================================="
echo ""

# Function to run a single benchmark
run_benchmark() {
    local algorithm=$1
    local device=$2
    local dtype=$3            # fp32 | fp16
    local lambda_period=${4:-1}    # naive SLDA only (1 = per-sample inversion)
    local extra_flags=${5:-}       # e.g. --compile
    local run_label="$algorithm (${dtype^^}${extra_flags:+, $extra_flags}) on $device"

    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $run_label"
    echo "----------------------------------------------"

    if python3 benchmarks/benchmark.py \
        --algorithm "$algorithm" \
        --op both \
        --compute_device "$device" \
        --dtype "$dtype" \
        --data_path "$DATA_PATH" \
        --logs_path "$LOGS_PATH" \
        --feature_size $FEATURE_SIZE \
        --num_classes $NUM_CLASSES \
        --lambda_period "$lambda_period" \
        $extra_flags \
        --seed $SEED; then
        echo "SUCCESS: $run_label"
    else
        echo "FAILED: $run_label"
        FAILED_RUNS+=("$run_label")
        return 1
    fi
    echo ""
}

# 1. CLP - FP32 only (GPU and CPU)
run_benchmark "clp" "cuda" "fp32" || true
sleep 2
run_benchmark "clp" "cpu" "fp32" || true
sleep 2

# 2. NCM - FP32 + FP16 on GPU, FP32 on CPU
run_benchmark "ncm" "cuda" "fp32" || true
sleep 2
run_benchmark "ncm" "cuda" "fp16" || true
sleep 2
run_benchmark "ncm" "cpu" "fp32" || true
sleep 2

# 3. NCM compiled-overhead control (torch.compile reduce-overhead / CUDA Graphs)
run_benchmark "ncm" "cuda" "fp32" 1 "--compile" || true
sleep 2

# 4. Replay - FP32 + FP16 on GPU, FP32 on CPU
run_benchmark "replay" "cuda" "fp32" || true
sleep 2
run_benchmark "replay" "cuda" "fp16" || true
sleep 2
run_benchmark "replay" "cpu" "fp32" || true
sleep 2

# 5. SLDA naive reference (per-sample inversion) - SI cost anchor
run_benchmark "slda" "cuda" "fp32" 1 || true
sleep 2
run_benchmark "slda" "cuda" "fp16" 1 || true
sleep 2
run_benchmark "slda" "cpu" "fp32" 1 || true
sleep 2

# 6. SLDA rank-1 (paper baseline; FP32 state required) - Table 1
run_benchmark "slda_rank1" "cuda" "fp32" || true
sleep 2
run_benchmark "slda_rank1" "cpu" "fp32" || true
sleep 2

# Summary
echo "=============================================="
echo "OpenLORIS Benchmarking Complete!"
echo "=============================================="
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $((TOTAL_RUNS - ${#FAILED_RUNS[@]}))"
echo "Failed: ${#FAILED_RUNS[@]}"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    exit 1
fi

echo ""
echo "All benchmarks completed. Results in: $LOGS_PATH"
echo "(each run writes one exp_N/ folder per timed op: '<model>' = fit/C_upd, '<model>-qry' = predict/C_qry)"
echo "=============================================="
