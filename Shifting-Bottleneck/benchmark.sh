#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
SEQ_SRC="$ROOT_DIR/sequential/jobshop_seq.c"
PAR_SRC="$ROOT_DIR/parallel/jobshop_par.c"
DATA_DIR="$ROOT_DIR/data"
BUILD_DIR="$ROOT_DIR/.benchmark-bin"
OUT_FILE="$ROOT_DIR/benchmark_results.txt"
TARGET_SEC=${1:-60}
THREADS=(1 2 4 8 16 32)

mkdir -p "$BUILD_DIR"

SEQ_BIN="$BUILD_DIR/jobshop_seq"
PAR_BIN="$BUILD_DIR/jobshop_par"

gcc -O2 -Wall -Wextra -o "$SEQ_BIN" "$SEQ_SRC"
gcc -O2 -Wall -Wextra -fopenmp -o "$PAR_BIN" "$PAR_SRC"

ns_to_ms() {
    awk -v t="$1" 'BEGIN { printf "%.6f", t / 1000000 }'
}

run_seq_case() {
    local input_file=$1
    local label=$2
    local start_ns=$(date +%s%N)
    local target_ns=$((TARGET_SEC * 1000000000))
    local runs=0
    local makespan=0
    local elapsed=0

    while [ $elapsed -lt $target_ns ]; do
        "$SEQ_BIN" "$input_file" "$BUILD_DIR/seq.out" >/dev/null
        makespan=$(head -1 "$BUILD_DIR/seq.out")
        runs=$((runs + 1))
        elapsed=$(( $(date +%s%N) - start_ns ))
    done

    avg_ms=$(ns_to_ms $(awk -v t="$elapsed" -v r="$runs" 'BEGIN { if (r>0) printf "%d", t / r; else print 0 }'))
    printf "SEQ  | %-20s | makespan=%-5s | avg_run=%10.6f ms | runs=%d | total_time=%d s\n" \
        "$label" "$makespan" "$avg_ms" "$runs" $((elapsed/1000000000)) | tee -a "$OUT_FILE" >/dev/stderr
    echo "$avg_ms"
}

run_par_case() {
    local input_file=$1
    local label=$2
    local seq_time_ms=$3

    for T in "${THREADS[@]}"; do
        local start_ns=$(date +%s%N)
        local target_ns=$((TARGET_SEC * 1000000000))
        local runs=0
        local makespan=0
        local elapsed=0

        while [ $elapsed -lt $target_ns ]; do
            "$PAR_BIN" "$input_file" "$BUILD_DIR/par.out" "$T" >/dev/null
            makespan=$(head -1 "$BUILD_DIR/par.out")
            runs=$((runs + 1))
            elapsed=$(( $(date +%s%N) - start_ns ))
        done

        avg_ms=$(ns_to_ms $(awk -v t="$elapsed" -v r="$runs" 'BEGIN { if (r>0) printf "%d", t / r; else print 0 }'))
        speedup=$(awk -v s="$seq_time_ms" -v p="$avg_ms" 'BEGIN { if (p>0) printf "%.3f", s / p; else print "N/A" }')
        printf "PAR  | %-20s | T=%-2s | makespan=%-5s | avg_run=%10.6f ms | speedup=%s | runs=%d\n" \
            "$label" "$T" "$makespan" "$avg_ms" "$speedup" "$runs" | tee -a "$OUT_FILE" >/dev/stderr
    done
}

rm -f "$OUT_FILE"
{
    echo "Shifting-Bottleneck Benchmark - Job-Shop Scheduling"
    echo "Running on all inputs in: $DATA_DIR"
    echo "Target seconds per config: $TARGET_SEC"
    echo "Threads tested: ${THREADS[*]}"
    echo "Machine: $(uname -n) with $(nproc) logical CPUs"
    echo "============================================================"
} | tee "$OUT_FILE"

# Test the same representative inputs as Greedy, plus the larger 50x50 and 100x100 cases
INPUTS=("ft06" "gg03" "benchmark_test" "benchmark_large_50x50" "benchmark_100x100")

for input_name in "${INPUTS[@]}"; do
    input_file="$DATA_DIR/${input_name}.jss"
    [ -f "$input_file" ] || continue
    case_label=$(basename "$input_file" .jss)
    file_size=$(wc -l < "$input_file")
    echo "" | tee -a "$OUT_FILE"
    echo "=== Testing: $case_label ($file_size lines) ===" | tee -a "$OUT_FILE"

    seq_time_ms=$(run_seq_case "$input_file" "$case_label")
    run_par_case "$input_file" "$case_label" "$seq_time_ms"
done

echo "============================================================" | tee -a "$OUT_FILE"
echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
