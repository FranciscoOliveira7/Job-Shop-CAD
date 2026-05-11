#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
SEQ_SRC="$ROOT_DIR/sequencial/jobshop_seq.c"
PAR_SRC="$ROOT_DIR/parallel/jobshop_par.c"
DATA_DIR="$ROOT_DIR/data"
BUILD_DIR="$ROOT_DIR/.benchmark-bin"
OUT_FILE="$ROOT_DIR/benchmark_results.txt"
REPS=${1:-5}
THREADS=(1 2 4 8 16 32)
INPUT_FILE="${2:-$DATA_DIR/ft06.jss}"

mkdir -p "$BUILD_DIR"

SEQ_BIN="$BUILD_DIR/jobshop_seq"
PAR_BIN="$BUILD_DIR/jobshop_par"

gcc -O2 -Wall -Wextra -o "$SEQ_BIN" "$SEQ_SRC"
gcc -O2 -Wall -Wextra -fopenmp -o "$PAR_BIN" "$PAR_SRC"

get_time_seconds() {
    printf '%s' "$1" | grep -oE 'Tempo medio[^:]*: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1
}

get_time_parallel() {
    printf '%s' "$1" | grep -oE 'Tempo medio \([0-9]+ repeticoes\): [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1
}

get_makespan() {
    printf '%s' "$1" | grep -oE 'Makespan: [0-9]+' | grep -oE '[0-9]+' | tail -1
}

average() {
    awk -v sum="$1" -v n="$2" 'BEGIN { if (n > 0) printf "%.6f", sum / n; else printf "0.000000" }'
}

ns_to_ms() {
    awk -v t="$1" 'BEGIN { printf "%.6f", t / 1000000 }'
}

run_seq_case() {
    local input_file=$1
    local label=$2
    local total=0
    local makespan=0
    local target_time_ns=$((5 * 1000000000))  # 60 seconds in nanoseconds

    local start_ns=$(date +%s%N)
    local elapsed_ns=0
    local runs=0

    # Keep running until we hit 1 minute
    while [ $elapsed_ns -lt $target_time_ns ]; do
        "$SEQ_BIN" "$input_file" "$BUILD_DIR/seq.out" >/dev/null
        makespan=$(head -1 "$BUILD_DIR/seq.out")
        runs=$((runs + 1))
        
        local end_ns=$(date +%s%N)
        elapsed_ns=$((end_ns - start_ns))
    done

    local avg
    avg=$(ns_to_ms "$(awk -v t="$elapsed_ns" -v r="$runs" 'BEGIN { if (r > 0) printf "%.0f", t / r; else printf "0" }')")
    printf "SEQ  | %-20s | makespan=%-5s | avg_run=%10.6f ms | runs=%d | total_time=%d s\n" "$label" "$makespan" "$avg" "$runs" $((elapsed_ns / 1000000000)) | tee -a "$OUT_FILE" >/dev/stderr
    echo "$avg"
}

run_par_case() {
    local input_file=$1
    local label=$2
    local seq_time=$3
    local target_time_ns=$((5 * 1000000000))  # 60 seconds in nanoseconds

    for T in "${THREADS[@]}"; do
        local total=0
        local makespan=0
        local start_ns=$(date +%s%N)
        local elapsed_ns=0
        local runs=0

        # Keep running until we hit ~1 minute
        while [ $elapsed_ns -lt $target_time_ns ]; do
            "$PAR_BIN" "$input_file" "$BUILD_DIR/par.out" "$T" >/dev/null
            makespan=$(head -1 "$BUILD_DIR/par.out")
            runs=$((runs + 1))
            
            local end_ns=$(date +%s%N)
            elapsed_ns=$((end_ns - start_ns))
        done

        local avg speedup
        avg=$(ns_to_ms "$(awk -v t="$elapsed_ns" -v r="$runs" 'BEGIN { if (r > 0) printf "%.0f", t / r; else printf "0" }')")
        speedup=$(awk -v s="$seq_time" -v p="$avg" 'BEGIN { if (p > 0) printf "%.3f", s / p; else printf "N/A" }')
        printf "PAR  | %-20s | T=%-2s | makespan=%-5s | avg_run=%10.6f ms | speedup=%s | runs=%d\n" "$label" "$T" "$makespan" "$avg" "$speedup" "$runs" | tee -a "$OUT_FILE" >/dev/stderr
    done
}

rm -f "$OUT_FILE"
{
    echo "Greedy Benchmark - Job-Shop Scheduling"
    echo "Running on all inputs in: $DATA_DIR"
    echo "Repetitions: $REPS"
    echo "Threads tested: ${THREADS[*]}"
    echo "Machine: $(uname -n) with $(nproc) logical CPUs"
    echo "============================================================"
} | tee "$OUT_FILE"

# Test all .jss files in the data directory
for input_file in "$DATA_DIR"/*.jss; do
    if [ ! -f "$input_file" ]; then
        continue
    fi
    
    case_label=$(basename "$input_file" .jss)
    file_size=$(wc -l < "$input_file")
    echo "" | tee -a "$OUT_FILE"
    echo "=== Testing: $case_label ($file_size lines) ===" | tee -a "$OUT_FILE"
    
    seq_time=$(run_seq_case "$input_file" "$case_label")
    run_par_case "$input_file" "$case_label" "$seq_time"
done

echo "============================================================" | tee -a "$OUT_FILE"
echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
echo "" | tee -a "$OUT_FILE"
echo "=== Benchmark Usage ===" | tee -a "$OUT_FILE"
echo "Run with custom repetitions: $0 <reps>" | tee -a "$OUT_FILE"
echo "Or specify one file: $0 <reps> <input_file>" | tee -a "$OUT_FILE"
echo "Example: $0 10" | tee -a "$OUT_FILE"