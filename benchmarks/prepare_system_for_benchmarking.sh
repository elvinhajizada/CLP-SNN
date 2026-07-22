#!/bin/bash

# System preparation script for Jetson Orin benchmarking
# Run with: sudo bash scripts/prepare_system_for_benchmarking.sh

echo "=========================================="
echo "Preparing Jetson Orin for Benchmarking"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root: sudo bash $0"
    exit 1
fi

# 1. Show current power mode
echo "1. Checking power mode..."
nvpmodel -q
echo ""

# 2. Clear system caches
echo "2. Clearing system caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
echo "✓ Caches cleared"
echo ""

# 3. Check and report thermal state
echo "3. Checking thermal state..."
temps=$(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | awk '{print $1/1000}')
max_temp=$(echo "$temps" | sort -rn | head -1)
echo "   Max temperature: ${max_temp}°C"
if (( $(echo "$max_temp > 70" | bc -l) )); then
    echo "   ⚠️  TEMPERATURE TOO HIGH - wait for cooling before benchmarking"
    echo "   Recommended: < 60°C for clean results"
elif (( $(echo "$max_temp > 60" | bc -l) )); then
    echo "   ⚠️  Temperature elevated - consider waiting"
else
    echo "   ✓ Temperature good"
fi
echo ""

# 4. Lock clocks to maximum
echo "4. Locking clocks to maximum..."
jetson_clocks
sleep 2
echo "✓ Clocks locked"
echo ""

# 5. Verify CPU frequencies
echo "5. Verifying CPU frequencies..."
freqs=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{print $1/1000}')
echo "   CPU frequencies (MHz):"
cpu_num=0
for freq in $freqs; do
    echo "     CPU${cpu_num}: ${freq} MHz"
    cpu_num=$((cpu_num + 1))
done
echo ""

# 6. Check memory availability
echo "6. Checking memory..."
free -h | grep "Mem:"
mem_avail=$(free | grep Mem | awk '{print ($7/$2) * 100}')
echo "   Available: ${mem_avail}%"
if (( $(echo "$mem_avail < 30" | bc -l) )); then
    echo "   ⚠️  Low memory - may cause issues"
else
    echo "   ✓ Memory sufficient"
fi
echo ""

# 7. Check for heavy background processes
echo "7. Checking background processes..."
proc_count=$(ps aux | wc -l)
echo "   Total processes: $proc_count"
echo "   Top CPU consumers:"
ps aux --sort=-%cpu | head -6 | tail -5 | awk '{printf "     %-20s %5s%%\n", $11, $3}'
echo ""

# 8. Check GPU usage
echo "8. Checking GPU state..."
if command -v tegrastats &> /dev/null; then
    timeout 1s tegrastats 2>/dev/null | grep -o "GR3D_FREQ [0-9]*%" | head -1
    echo "   (Should be 0% or minimal for clean benchmark)"
else
    echo "   tegrastats not available"
fi
echo ""

# 9. Final settling time
echo "9. Allowing system to settle (10 seconds)..."
sleep 10
echo "✓ System settled"
echo ""

echo "=========================================="
echo "System Ready for Benchmarking!"
echo "=========================================="
echo ""
echo "Recommendations:"
echo "  - Run benchmarks immediately after this script"
echo "  - Close jtop if open (adds overhead)"
echo "  - Minimize other active terminals/processes"
echo "  - For best results, run multiple iterations"
echo ""
echo "To disable jetson_clocks later:"
echo "  sudo jetson_clocks --restore"
echo ""
