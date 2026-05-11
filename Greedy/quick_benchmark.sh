#!/bin/bash
# Quick test: 10 seconds per config instead of 60

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
SEQ_SRC="$ROOT_DIR/sequencial/jobshop_seq.c"
PAR_SRC="$ROOT_DIR/parallel/jobshop_par.c"
BUILD_DIR="$ROOT_DIR/.benchmark-bin"
OUT_FILE="$ROOT_DIR/quick_benchmark_results.txt"

mkdir -p "$BUILD_DIR"
SEQ_BIN="$BUILD_DIR/jobshop_seq"
PAR_BIN="$BUILD_DIR/jobshop_par"

gcc -O2 -Wall -Wextra -o "$SEQ_BIN" "$SEQ_SRC" 2>/dev/null
gcc -O2 -Wall -Wextra -fopenmp -o "$PAR_BIN" "$PAR_SRC" 2>/dev/null

TARGET_SEC=${1:-10}

ns_to_ms() {
    awk -v t="$1" 'BEGIN { printf "%.3f", t / 1000000 }'
}

{
    echo "Quick Benchmark Test (target: ${TARGET_SEC}s per config)"
    echo "File: data/benchmark_200x200.jss"
    echo "============================================================"
} | tee "$OUT_FILE"

target_ns=$((TARGET_SEC * 1000000000))
input_file="$ROOT_DIR/data/benchmark_200x200.jss"

# Sequential
start_ns=$(date +%s%N)
runs=0
while true; do
    "$SEQ_BIN" "$input_file" /tmp/q.out >/dev/null
    makespan=$(head -1 /tmp/q.out)
    runs=$((runs + 1))
    elapsed=$(($(date +%s%N) - start_ns))
    [ $elapsed -ge $target_ns ] && break
done

avg=$(ns_to_ms $((elapsed / runs)))
echo "" | tee -a "$OUT_FILE"
echo "SEQ | makespan=$makespan | avg=$(echo $avg)ms | runs=$runs | time=$((elapsed/1000000000))s" | tee -a "$OUT_FILE"

# Parallel (just 1 thread to save time)
for threads in 1 8 32; do
    start_ns=$(date +%s%N)
    runs=0
    while true; do
        "$PAR_BIN" "$input_file" /tmp/q.out "$threads" >/dev/null
        makespan=$(head -1 /tmp/q.out)
        runs=$((runs + 1))
        elapsed=$(($(date +%s%N) - start_ns))
        [ $elapsed -ge $((TARGET_SEC * 1000000000)) ] && break
    done
    avg=$(ns_to_ms $((elapsed / runs)))
    echo "PC$threads | makespan=$makespan | avg=$(echo $avg)ms | runs=$runs | time=$((elapsed/1000000000))s" | tee -a "$OUT_FILE"
done

echo "============================================================" | tee -a "$OUT_FILE"
echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
