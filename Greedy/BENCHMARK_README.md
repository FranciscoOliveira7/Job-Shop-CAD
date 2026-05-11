# Greedy Benchmark Guide

This benchmark measures the performance of the sequential and parallel Greedy job-shop scheduling implementations.

## Assignment Requirements (Secção C)

The assignment requires:

- Input file that generates **at least 1 minute** on sequential version
- Benchmark table with thread counts: SEQ, PC1, PC2, PC4, PC8, PC16, PC32
- Comparison of sequential vs parallel results (same makespan expected)
- Speedup calculation: S = T_sequential / T_parallel
- Analysis of speedup variation

## Usage

### Quick Start (uses ft06.jss)

```bash
./benchmark.sh 5
```

This runs 5 repetitions with the default ft06.jss input.

### With Custom Repetitions

```bash
./benchmark.sh 10
```

This runs 10 repetitions instead of 5.

### With Custom Input File

```bash
./benchmark.sh 5 data/your_input.jss
```

## Generating a Larger Input (Required for Assignment)

The default inputs (ft06, gg03) are too small for proper benchmarking. You need an input that takes **at least 1 minute** on the sequential version.

### Generate automatically:

```bash
chmod +x generate_benchmark_input.sh
./generate_benchmark_input.sh 20 20 data/benchmark_large.jss
```

This generates a 20×20 job-shop instance. Adjust parameters if needed:

```bash
./generate_benchmark_input.sh 30 30 data/benchmark_xlarge.jss
```

### Run benchmark on large input:

```bash
./benchmark.sh 5 data/benchmark_large.jss
```

## Understanding the Output

The benchmark produces:

1. **Execution time table**: Shows average time per thread configuration
2. **Makespan**: Final completion time (should be same for seq and all parallel configs)
3. **Speedup**: Ratio of sequential to parallel time
   - Speedup > 1: parallel is faster
   - Speedup < 1: sequential is faster (overhead problem)
   - Speedup ≈ 1: no parallelization benefit

## For Your Report

Include in section **C. Análise do Desempenho**:

1. **Table format**:
   | Threads | Time (ms) | Makespan | Speedup |
   |---------|-----------|----------|---------|
   | SEQ | XX.XX | YYYY | 1.000 |
   | PC1 | XX.XX | YYYY | X.XXX |
   | ... | ... | ... | ... |

2. **Graph**: X-axis = threads, Y-axis = execution time (ms)

3. **Analysis**: Comment on why speedup decreases at higher thread counts
   - For Greedy (simpler algorithm), thread overhead dominates on small/medium inputs
   - Machine contention on shared resources
   - OpenMP overhead (thread creation, synchronization)

## Files Generated

- `benchmark.sh` - Main benchmark script
- `generate_benchmark_input.sh` - Input generator for larger instances
- `benchmark_results.txt` - Results of last benchmark run (automatically created)
