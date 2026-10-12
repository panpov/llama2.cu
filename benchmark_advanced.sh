#!/bin/bash

# Configuration
MODELS=("stories15M.bin" "stories42M.bin" "stories110M.bin")
PROMPT="One day, Lily met a Shoggoth"
STEPS=256
TEMP=0.8
TOP_P=0.9
NUM_RUNS=3

# 1. Compile run.cu with CUDA support
echo "Compiling run.cu for NVIDIA DGX Spark..."
nvcc -DUSE_CUDA -O3 -o runcuda run.cu -lm -lcublas
if [ $? -ne 0 ]; then
    echo "Compilation failed! Please check CUDA toolkit installation."
    exit 1
fi

echo "=================================================="
echo "Starting Comprehensive Inference Benchmarks"
echo "=================================================="

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "Model file $model not found! Skipping..."
        continue
    fi

    model_size_bytes=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model")
    model_size_mb=$(echo "scale=2; $model_size_bytes / 1024 / 1024" | bc)
    
    echo ""
    echo "--------------------------------------------------"
    echo "Testing Model: $model (Size: ${model_size_mb} MB)"
    echo "--------------------------------------------------"

    # Warm-up run
    ./runcuda "$model" -n "$STEPS" -t "$TEMP" -p "$TOP_P" -i "$PROMPT" > /dev/null 2>&1

    sum_tok_s=0
    sum_bandwidth=0
    sum_itl=0
    sum_power=0
    sum_energy=0
    peak_vram=0

    for ((i=1; i<=NUM_RUNS; i++)); do
        # Start background nvidia-smi poller for power (Watts) and memory (MiB)
        nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits -lms 100 > /tmp/gpu_metrics_run$i.log &
        poller_pid=$!

        # Capture run output (stderr holds tok/s)
        start_time=$(date +%s%N)
        output=$(./runcuda "$model" -n "$STEPS" -t "$TEMP" -p "$TOP_P" -i "$PROMPT" 2>&1 >/dev/null)
        end_time=$(date +%s%N)

        # Kill poller
        kill $poller_pid 2>/dev/null
        wait $poller_pid 2>/dev/null

        # Parse tok/s
        tok_s=$(echo "$output" | grep "achieved tok/s" | awk '{print $3}')
        if [ -z "$tok_s" ]; then tok_s=0; fi

        # Calculate metrics
        bandwidth_gb=$(echo "scale=4; ($model_size_bytes * $tok_s) / 1000000000" | bc)
        itl_ms=$(echo "scale=2; 1000 / $tok_s" | bc)
        
        # Approximate TTFT (time taken for the first token or total duration minus generation time)
        total_duration_ms=$(echo "scale=2; ($end_time - $start_time) / 1000000" | bc)
        # In this loop structure, total time minus decoding time gives an estimate of prefill/TTFT
        gen_time_ms=$(echo "scale=2; ($STEPS * 1000) / $tok_s" | bc)
        ttft_ms=$(echo "scale=2; $total_duration_ms - $gen_time_ms" | bc)
        if (( $(echo "$ttft_ms < 0" | bc -l) )); then ttft_ms=10; fi # fallback floor

        # Parse power and memory from poller log
        if [ -f /tmp/gpu_metrics_run$i.log ]; then
            avg_run_power=$(awk -F',' '{sum+=$1; count++} END {if (count>0) print sum/count; else print 0}' /tmp/gpu_metrics_run$i.log)
            max_run_mem=$(awk -F',' '{if ($2>max) max=$2} END {print max+0}' /tmp/gpu_metrics_run$i.log)
            if (( $(echo "$max_run_mem > $peak_vram" | bc -l) )); then peak_vram=$max_run_mem; fi
        else
            avg_run_power=0
            max_run_mem=0
        fi

        # Energy Efficiency: Joules per token = Watts / (tokens/sec)
        if (( $(echo "$tok_s > 0" | bc -l) )); then
            joules_per_token=$(echo "scale=4; $avg_run_power / $tok_s" | bc)
        else
            joules_per_token=0
        fi

        echo "  Run $i: $tok_s tok/s | ITL: ${itl_ms}ms | TTFT: ~${ttft_ms}ms | Power: ${avg_run_power}W | Energy: ${joules_per_token} J/tok"

        sum_tok_s=$(echo "$sum_tok_s + $tok_s" | bc)
        sum_bandwidth=$(echo "$sum_bandwidth + $bandwidth_gb" | bc)
        sum_itl=$(echo "$sum_itl + $itl_ms" | bc)
        sum_power=$(echo "$sum_power + $avg_run_power" | bc)
        sum_energy=$(echo "$sum_energy + $joules_per_token" | bc)
    done

    # Averages
    avg_tok_s=$(echo "scale=2; $sum_tok_s / $NUM_RUNS" | bc)
    avg_bandwidth=$(echo "scale=2; $sum_bandwidth / $NUM_RUNS" | bc)
    avg_itl=$(echo "scale=2; $sum_itl / $NUM_RUNS" | bc)
    avg_power=$(echo "scale=2; $sum_power / $NUM_RUNS" | bc)
    avg_energy=$(echo "scale=4; $sum_energy / $NUM_RUNS" | bc)

    echo "=================================================="
    echo "Final Summary for $model:"
    echo "  Average Speed:      $avg_tok_s tok/s"
    echo "  Achieved Bandwidth: $avg_bandwidth GB/s"
    echo "  Inter-Token Latency:$avg_itl ms"
    echo "  Avg Power Draw:     $avg_power W"
    echo "  Energy Efficiency:  $avg_energy Joules/token"
    echo "  Peak Memory Usage:  $peak_vram MiB"
    echo "=================================================="
done