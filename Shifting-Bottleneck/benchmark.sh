#!/bin/bash
# Benchmark script – Computação de Alto Desempenho
# Usage: ./benchmark.sh <input_file> [reps] [time_limit_sec]

INPUT=${1:-data/ft06.jss}
REPS=${2:-5}
LIMIT=${3:-30}
OUT_FILE="benchmark_results.txt"
SEQ=./sequential/jobshop_seq
PAR=./parallel/jobshop_par
THREADS=(1 2 4 8 16 32)

get_time()    { echo "$1" | grep "Time"     | grep -oP '[0-9]+\.[0-9]+'; }
get_makespan(){ echo "$1" | grep "Makespan" | grep -oP '[0-9]+' | tail -1; }

sum_times() {
    local total=0
    for t in "$@"; do total=$(echo "$total + $t" | bc); done
    echo $total
}

echo "Benchmark: $INPUT | ${REPS} reps | ${LIMIT}s limit" | tee "$OUT_FILE"
echo "Config: $(nproc) logical CPUs on $(uname -n)" | tee -a "$OUT_FILE"
echo "============================================================" | tee -a "$OUT_FILE"

# ── Sequential ─────────────────────────────────────────────────
times=()
makespan=0
for i in $(seq 1 $REPS); do
    result=$($SEQ "$INPUT" /tmp/bench_seq.out $LIMIT)
    t=$(get_time "$result")
    makespan=$(get_makespan "$result")
    times+=("$t")
done
total=$(sum_times "${times[@]}")
avg=$(echo "scale=3; $total / $REPS" | bc)
printf "%-6s | makespan=%-5s | avg_time=%10.3f ms | speedup= 1.000\n" \
    "SEQ" "$makespan" "$avg" | tee -a "$OUT_FILE"
seq_time=$avg

# ── Parallel ───────────────────────────────────────────────────
for T in "${THREADS[@]}"; do
    times=()
    makespan=0
    for i in $(seq 1 $REPS); do
        result=$($PAR "$INPUT" /tmp/bench_par.out $T $LIMIT)
        t=$(get_time "$result")
        makespan=$(get_makespan "$result")
        times+=("$t")
    done
    total=$(sum_times "${times[@]}")
    avg=$(echo "scale=3; $total / $REPS" | bc)
    speedup=$(echo "scale=3; $seq_time / $avg" | bc 2>/dev/null || echo "N/A")
    printf "PC%-4s | makespan=%-5s | avg_time=%10.3f ms | speedup=%s\n" \
        "$T" "$makespan" "$avg" "$speedup" | tee -a "$OUT_FILE"
done

echo "============================================================" | tee -a "$OUT_FILE"
echo "Results saved to: $OUT_FILE"
