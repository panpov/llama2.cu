#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Error: Missing test file."
    echo "Usage: $0 <source.cu> <number of runs>"
    exit 1
fi

if [[ ! "$1" == *.cu ]]; then
    echo "Error: File must be a CUDA source file ending in .cu"
    exit 1
fi

if [ "$#" -lt 2 ]; then
    echo "Number of runs not specified. Defaulting to 3."
fi

SRC="$1"
BASE_NAME=$(basename "$SRC")
BIN="./${BASE_NAME//./}"
NUM_RUNS="${2:-3}"

# Configuration
MODELS=("stories15M.bin" "stories42M.bin" "stories110M.bin")
PROMPT="One day, Lily met a Shoggoth"
STEPS=256
TEMP=0.8
TOP_P=0.9
SEED=42

# 1. Compile run_new.cu targeting the DGX Spark GB10 architecture (sm_121 via native)
echo "Compiling $SRC for NVIDIA DGX Spark..."
# Once optmized, run: nvcc -O3 --use_fast_math -arch=native "$SRC" -o "$BIN" -lcublas
nvcc -O3 "$SRC" -o "$BIN" -lm -lcublas
if [ $? -ne 0 ]; then
    echo "Compilation failed! Please check arguments and/or CUDA toolkit installation."
    exit 1
fi

echo "===================================================================================================="
echo "Starting Inference Benchmarks on NVIDIA DGX Spark"
echo "===================================================================================================="

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "Model file $model not found! Skipping..."
        continue
    fi

    model_size_bytes=$(stat -c%s "$model" 2>/dev/null || stat -f%z "$model")
    model_size_mb=$(awk "BEGIN {printf \"%.2f\", $model_size_bytes / (1024 * 1024)}")

    echo ""
    echo "----------------------------------------------------------------------------------------------------"
    echo "Model: $model | Size: ${model_size_mb} MiB | Steps: $STEPS | Runs: $NUM_RUNS"
    echo "----------------------------------------------------------------------------------------------------"

    # Warm-up run (discarded to warm up OS page cache and GPU clocks)
    echo "Running warm-up pass..."
    $BIN "$model" -n "$STEPS" -t "$TEMP" -p "$TOP_P" -s "$SEED" -i "$PROMPT" > /dev/null 2>&1

    sum_speed=0
    sum_bw=0
    sum_itl=0
    sum_ttft=0
    sum_power=0
    sum_energy=0
    sum_mem=0
    peak_mem=0

    for ((i=1; i<=NUM_RUNS; i++)); do
        rm -f /tmp/gpu_power.log /tmp/gpu_mem.log /tmp/rss_run.log

        # Start separate background pollers for GPU power and per-process CUDA memory
        nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 20 > /tmp/gpu_power.log 2>/dev/null &
        power_pid=$!
        nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits -lms 20 > /tmp/gpu_mem.log 2>/dev/null &
        mem_pid=$!

        # Brief pause so NVML pollers finish initializing before the fast binary starts
        sleep 0.2

        # Run inference while capturing Peak Host RSS via kernel getrusage (%M in KiB)
        start_ns=$(date +%s%N)
        output=$(/usr/bin/time -f "%M" -o /tmp/rss_run.log $BIN "$model" -n "$STEPS" -t "$TEMP" -p "$TOP_P" -s "$SEED" -i "$PROMPT" 2>&1 >/dev/null)
        end_ns=$(date +%s%N)

        # Stop background pollers
        kill $power_pid $mem_pid 2>/dev/null
        wait $power_pid $mem_pid 2>/dev/null

        # 1. Speed (tok/s) from run_new.cu stderr
        speed=$(echo "$output" | awk '/achieved tok\/s:/ {printf "%.2f", $3}')
        if [ -z "$speed" ]; then speed=0; fi

        # 2. BW (GB/s) = (Model Size in Bytes * Speed) / 1e9
        bw=$(awk "BEGIN {printf \"%.2f\", ($model_size_bytes * $speed) / 1e9}")

        # 3. ITL (ms) = 1000 / Speed
        itl=$(awk "BEGIN {printf \"%.2f\", ($speed > 0) ? (1000 / $speed) : 0}")

        # 4. TTFT (ms) = Total Wall Time - Timed Generation Time for (STEPS - 1) tokens
        ttft=$(awk "BEGIN {
            wall_ms = ($end_ns - $start_ns) / 1e6;
            gen_ms = ($speed > 0) ? ((($STEPS - 1) * 1000) / $speed) : 0;
            diff = wall_ms - gen_ms;
            printf \"%.2f\", (diff > 0) ? diff : 0
        }")

        # 5. Power (W) averaged across active samples
        power=$(awk '$1 ~ /^[0-9.]+$/ {sum+=$1; c++} END {printf "%.2f", (c>0) ? sum/c : 0}' /tmp/gpu_power.log)

        # 6. Energy (J/tok) = Power / Speed
        energy=$(awk "BEGIN {printf \"%.4f\", ($speed > 0) ? ($power / $speed) : 0}")

        # 7. Mem (MiB) = Host Peak RSS (mmap/buffers) + CUDA Compute App Allocation (cudaMalloc)
        rss_mib=$(awk '$1 ~ /^[0-9]+$/ {printf "%.1f", $1 / 1024}' /tmp/rss_run.log)
        if [ -z "$rss_mib" ]; then rss_mib=0; fi
        cuda_mib=$(awk '$1 ~ /^[0-9.]+$/ {if ($1>max) max=$1} END {printf "%.1f", max+0}' /tmp/gpu_mem.log)
        mem=$(awk "BEGIN {printf \"%.1f\", $rss_mib + $cuda_mib}")

        # Track Peak Memory across runs
        peak_mem=$(awk "BEGIN {printf \"%.1f\", ($mem > $peak_mem) ? $mem : $peak_mem}")

        # Print all 7 metrics for the current run
        echo "  Run $i | Speed: ${speed} tok/s | BW: ${bw} GB/s | ITL: ${itl} ms | TTFT: ${ttft} ms | Power: ${power} W | Energy: ${energy} J/tok | Mem: ${mem} MiB (RSS: ${rss_mib}, CUDA: ${cuda_mib})"

        # Accumulate totals
        sum_speed=$(awk "BEGIN {print $sum_speed + $speed}")
        sum_bw=$(awk "BEGIN {print $sum_bw + $bw}")
        sum_itl=$(awk "BEGIN {print $sum_itl + $itl}")
        sum_ttft=$(awk "BEGIN {print $sum_ttft + $ttft}")
        sum_power=$(awk "BEGIN {print $sum_power + $power}")
        sum_energy=$(awk "BEGIN {print $sum_energy + $energy}")
        sum_mem=$(awk "BEGIN {print $sum_mem + $mem}")

        # Allow for cleanup/reset
        sleep 1
    done

    # Compute averages across runs
    avg_speed=$(awk "BEGIN {printf \"%.2f\", $sum_speed / $NUM_RUNS}")
    avg_bw=$(awk "BEGIN {printf \"%.2f\", $sum_bw / $NUM_RUNS}")
    avg_itl=$(awk "BEGIN {printf \"%.2f\", $sum_itl / $NUM_RUNS}")
    avg_ttft=$(awk "BEGIN {printf \"%.2f\", $sum_ttft / $NUM_RUNS}")
    avg_power=$(awk "BEGIN {printf \"%.2f\", $sum_power / $NUM_RUNS}")
    avg_energy=$(awk "BEGIN {printf \"%.4f\", $sum_energy / $NUM_RUNS}")
    avg_mem=$(awk "BEGIN {printf \"%.1f\", $sum_mem / $NUM_RUNS}")

    echo "===================================================================================================="
    echo "Summary ($model):"
    echo "  Speed:  $avg_speed tok/s"
    echo "  BW:     $avg_bw GB/s"
    echo "  ITL:    $avg_itl ms"
    echo "  TTFT:   $avg_ttft ms"
    echo "  Power:  $avg_power W"
    echo "  Energy: $avg_energy J/tok"
    echo "  Mem:    $avg_mem MiB (Peak: $peak_mem MiB)"
    echo "===================================================================================================="
done

rm -f /tmp/gpu_power.log /tmp/gpu_mem.log /tmp/rss_run.log