#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

SEQ_SRC="bb_seqteste.c"
PAR_SRC="bb_parteste.c"
DATA_DIR="data"
BUILD_DIR=".benchmark-bin-bb"
OUT_FILE="benchmark_results_bb.txt"

# Rule from assignment: benchmark each configuration long enough to be stable.
TARGET_SEC=${1:-60}
INPUT_FILE=${2:-$DATA_DIR/ft06.jss}
THREADS=(1 2 4 8 16 32)

if [ ! -f "$SEQ_SRC" ]; then
    echo "Missing source file: $SEQ_SRC" >&2
    exit 1
fi

if [ ! -f "$PAR_SRC" ]; then
    echo "Missing source file: $PAR_SRC" >&2
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Missing input file: $INPUT_FILE" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

SEQ_BIN="$BUILD_DIR/bb_seqteste"
PAR_BIN="$BUILD_DIR/bb_parteste"

gcc -O2 -Wall -Wextra -o "$SEQ_BIN" "$SEQ_SRC"
gcc -O2 -Wall -Wextra -fopenmp -o "$PAR_BIN" "$PAR_SRC"

ns_to_ms() {
    awk -v t="$1" 'BEGIN { printf "%.6f", t / 1000000 }'
}

run_seq_case() {
    local input_file=$1
    local start_ns elapsed_ns target_ns runs makespan avg_ms

    start_ns=$(date +%s%N)
    target_ns=$((TARGET_SEC * 1000000000))
    elapsed_ns=0
    runs=0
    makespan=0

    while [ "$elapsed_ns" -lt "$target_ns" ]; do
        "$SEQ_BIN" "$input_file" "$BUILD_DIR/seq.out" >/dev/null
        makespan=$(head -1 "$BUILD_DIR/seq.out")
        runs=$((runs + 1))
        elapsed_ns=$(( $(date +%s%N) - start_ns ))
    done

    avg_ms=$(ns_to_ms "$(awk -v t="$elapsed_ns" -v r="$runs" 'BEGIN { if (r > 0) printf "%.0f", t / r; else print 0 }')")

    printf "SEQ  | T=1  | makespan=%-6s | avg_run=%10s ms | runs=%d | total=%ds\n" \
        "$makespan" "$avg_ms" "$runs" $((elapsed_ns / 1000000000)) | tee -a "$OUT_FILE" >/dev/stderr

    echo "$avg_ms;$makespan"
}

run_par_case() {
    local input_file=$1
    local seq_time_ms=$2
    local seq_makespan=$3

    for T in "${THREADS[@]}"; do
        local start_ns elapsed_ns target_ns runs makespan avg_ms speedup
        start_ns=$(date +%s%N)
        target_ns=$((TARGET_SEC * 1000000000))
        elapsed_ns=0
        runs=0
        makespan=0

        while [ "$elapsed_ns" -lt "$target_ns" ]; do
            "$PAR_BIN" "$input_file" "$BUILD_DIR/par.out" "$T" >/dev/null
            makespan=$(head -1 "$BUILD_DIR/par.out")
            runs=$((runs + 1))
            elapsed_ns=$(( $(date +%s%N) - start_ns ))
        done

        avg_ms=$(ns_to_ms "$(awk -v t="$elapsed_ns" -v r="$runs" 'BEGIN { if (r > 0) printf "%.0f", t / r; else print 0 }')")
        speedup=$(awk -v s="$seq_time_ms" -v p="$avg_ms" 'BEGIN { if (p > 0) printf "%.3f", s / p; else print "N/A" }')

        status="OK"
        if [ "$makespan" != "$seq_makespan" ]; then
            status="DIFF"
        fi

        printf "PAR  | T=%-2s | makespan=%-6s | avg_run=%10s ms | speedup=%s | runs=%d | %s\n" \
            "$T" "$makespan" "$avg_ms" "$speedup" "$runs" "$status" | tee -a "$OUT_FILE" >/dev/stderr

        printf "PC%-2s | %10s | %8s | %s\n" "$T" "$avg_ms" "$makespan" "$speedup" >> "$OUT_FILE"
    done
}

rm -f "$OUT_FILE"
{
    echo "Branch-and-Bound Benchmark (bb_seqteste.c vs bb_parteste.c)"
    echo "Input: $INPUT_FILE"
    echo "Target seconds per configuration: $TARGET_SEC"
    echo "Threads tested: ${THREADS[*]}"
    echo "============================================================"
} | tee "$OUT_FILE"

seq_result=$(run_seq_case "$INPUT_FILE")
seq_time_ms=${seq_result%%;*}
seq_makespan=${seq_result##*;}

echo "------------------------------------------------------------" | tee -a "$OUT_FILE"
echo "Table for report (Secao C format):" | tee -a "$OUT_FILE"
printf "%-4s | %-10s | %-8s | %-7s\n" "Type" "Time(ms)" "Makespan" "Speedup" | tee -a "$OUT_FILE"
printf "%-4s | %-10s | %-8s | %-7s\n" "SEQ" "$seq_time_ms" "$seq_makespan" "1.000" | tee -a "$OUT_FILE"

run_par_case "$INPUT_FILE" "$seq_time_ms" "$seq_makespan"

echo "============================================================" | tee -a "$OUT_FILE"
echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
