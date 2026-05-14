#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

SEQ_SRC="bb_seqteste.c"
PAR_SRC="bb_parteste.c"
DATA_DIR="data"
BUILD_DIR="/dev/shm/.benchmark-bin-bb"
OUT_FILE="benchmark_results_bb.txt"

INPUT_FILE=$DATA_DIR/ft06.jss
REPETITIONS=3

usage() {
    cat <<EOF
Usage:
    $0 [input_file]

Examples:
    $0
    $0 data/ft06.jss
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--input)
            shift
            INPUT_FILE=${1-}
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            if [ -z "${FIRST_POS_ARG-}" ]; then
                FIRST_POS_ARG=$1
            else
                echo "Too many positional arguments." >&2
                usage >&2
                exit 1
            fi
            ;;
    esac
    shift
done

if [ -n "${FIRST_POS_ARG-}" ]; then
    if [ -f "$FIRST_POS_ARG" ]; then
        INPUT_FILE=$FIRST_POS_ARG
    fi
fi

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

sec_to_ms() {
    awk -v t="$1" 'BEGIN { printf "%.6f", t * 1000 }'
}

run_seq_case() {
    local input_file=$1
    local start_ns elapsed_ns target_ns runs makespan avg_ms avg_sec

    start_ns=$(date +%s%N)
    out=$(taskset -c 0 env "$SEQ_BIN" "$input_file" "$BUILD_DIR/seq.out" 2>/dev/null)
    makespan=$(printf "%s\n" "$out" | awk -F": " '/Melhor makespan/ {print $2; exit}')
    avg_sec=$(printf "%s\n" "$out" | awk -F": " '/Tempo medio por repeticao/ {print $2; exit}' | awk '{print $1}')
    if [ -z "$makespan" ]; then
        makespan=$(head -1 "$BUILD_DIR/seq.out" 2>/dev/null || echo 0)
    fi
    runs=1
    elapsed_ns=$(( $(date +%s%N) - start_ns ))

    if [ -n "$avg_sec" ]; then
        avg_ms=$(sec_to_ms "$avg_sec")
    else
        avg_ms=$(ns_to_ms "$(awk -v t="$elapsed_ns" -v r="$runs" 'BEGIN { if (r > 0) printf "%.0f", t / r; else print 0 }')")
    fi

    printf "SEQ  | T=1  | makespan=%-6s | avg_run=%10s ms | runs=%d | total=%ds\n" \
        "$makespan" "$avg_ms" "$runs" $((elapsed_ns / 1000000000)) | tee -a "$OUT_FILE" >/dev/stderr

    echo "$avg_ms;$makespan"
}

run_par_case() {
    local input_file=$1
    local seq_time_ms=$2
    local seq_makespan=$3

    for T in "${THREADS[@]}"; do
        local start_ns elapsed_ns runs makespan avg_ms avg_sec speedup
        start_ns=$(date +%s%N)
        # pin threads to cores 0..T-1 for more consistent performance
        if command -v taskset >/dev/null 2>&1; then
            out=$(taskset -c 0-$((T-1)) env OMP_NUM_THREADS=$T OMP_PLACES=cores OMP_PROC_BIND=spread OMP_DYNAMIC=false "$PAR_BIN" "$input_file" "$BUILD_DIR/par.out" "$T" 2>/dev/null)
        else
            out=$(env OMP_NUM_THREADS=$T OMP_PLACES=cores OMP_PROC_BIND=spread OMP_DYNAMIC=false "$PAR_BIN" "$input_file" "$BUILD_DIR/par.out" "$T" 2>/dev/null)
        fi
        makespan=$(printf "%s\n" "$out" | awk -F": " '/Melhor makespan/ {print $2; exit}')
        avg_sec=$(printf "%s\n" "$out" | awk -F": " '/Tempo medio por repeticao/ {print $2; exit}' | awk '{print $1}')
        if [ -z "$makespan" ]; then
            makespan=$(head -1 "$BUILD_DIR/par.out" 2>/dev/null || echo 0)
        fi
        runs=1
        elapsed_ns=$(( $(date +%s%N) - start_ns ))

        if [ -n "$avg_sec" ]; then
            avg_ms=$(sec_to_ms "$avg_sec")
        else
            avg_ms=$(ns_to_ms "$(awk -v t="$elapsed_ns" -v r="$runs" 'BEGIN { if (r > 0) printf "%.0f", t / r; else print 0 }')")
        fi
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
    echo "Program repetitions: $REPETITIONS"
    echo "Threads tested: ${THREADS[*]}"
    echo "============================================================"
} | tee "$OUT_FILE"

seq_result=$(run_seq_case "$INPUT_FILE")
seq_time_ms=${seq_result%%;*}
seq_makespan=${seq_result##*;}

echo "------------------------------------------------------------" | tee -a "$OUT_FILE"
echo "Table for report (Secao C format):" | tee -a "$OUT_FILE"
printf "%-4s | %-10s | %-8s | %-7s\n" "Type" "Avg(ms)" "Makespan" "Speedup" | tee -a "$OUT_FILE"
printf "%-4s | %-10s | %-8s | %-7s\n" "SEQ" "$seq_time_ms" "$seq_makespan" "1.000" | tee -a "$OUT_FILE"

run_par_case "$INPUT_FILE" "$seq_time_ms" "$seq_makespan"

echo "============================================================" | tee -a "$OUT_FILE"
echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
