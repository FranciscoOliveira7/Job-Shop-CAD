# Greedy Benchmark - Assignment Instructions

## Current Status

✅ Benchmark script created and updated  
✅ Input generator script created  
✅ Documentation created

## What You Have

1. **benchmark.sh** - Main benchmark script
   - Compiles both sequential and parallel Greedy implementations
   - Measures execution time with configurable repetitions
   - Calculates speedup
   - Supports custom input files and thread counts (1, 2, 4, 8, 16, 32)

2. **generate_benchmark_input.sh** - Creates larger test inputs
   - Usage: `./generate_benchmark_input.sh <jobs> <machines> <output_file>`
   - Example: `./generate_benchmark_input.sh 50 50 data/benchmark_50x50.jss`

3. **Sample inputs pre-generated in data/**
   - `ft06.jss` (6x6) - too small for proper benchmarking
   - `gg03.jss` (3x3) - too small
   - `benchmark_large_50x50.jss` (50x50) - should be closer to 1 minute requirement

## How to Run for Your Assignment

### Step 1: Create a proper large input

The assignment requires **at least 1 minute** on sequential version. Test different sizes:

```bash
# Try this first (will be faster to test)
./benchmark.sh 2 data/benchmark_large_50x50.jss

# If too fast, generate even larger
./generate_benchmark_input.sh 100 100 data/benchmark_100x100.jss
./benchmark.sh 2 data/benchmark_100x100.jss

# For final results use more repetitions
./benchmark.sh 10 data/benchmark_100x100.jss
```

### Step 2: Get your benchmark data

Once you find the right input size:

```bash
# Run final benchmark with 5-10 repetitions
./benchmark.sh 10 data/benchmark_YOUR_SIZE.jss
```

Results are saved in `benchmark_results.txt`

### Step 3: Add to your report (Secção C)

Include these in your PDF report:

**C.1 - Execution Time Table:**
Copy the results from benchmark_results.txt into a formatted table:

| Threads | Makespan | Time (ms) | Speedup |
| ------- | -------- | --------- | ------- |
| SEQ     | YYYY     | XX.XX     | 1.000   |
| PC1     | YYYY     | XX.XX     | X.XXX   |
| PC2     | YYYY     | XX.XX     | X.XXX   |
| PC4     | YYYY     | XX.XX     | X.XXX   |
| PC8     | YYYY     | XX.XX     | X.XXX   |
| PC16    | YYYY     | XX.XX     | X.XXX   |
| PC32    | YYYY     | XX.XX     | X.XXX   |

**C.2 - Graph:**
Plot thread count (X-axis) vs execution time in ms (Y-axis)

**C.3 - Speedup Analysis:**
Comment on why speedup is < 1:

- Greedy algorithm is simple → thread overhead dominates
- Machine contention on locks
- OpenMP thread creation/synchronization cost
- Input too small relative to parallelization overhead

## Current Benchmark Results (15x15 test)

```
SEQ  | benchmark_test | makespan=261   | avg_time=  1.636086 ms | speedup=1.000
PAR  | benchmark_test | T=1  | makespan=261   | avg_time=  1.850006 ms | speedup=0.884
PAR  | benchmark_test | T=2  | makespan=283   | avg_time=  2.005105 ms | speedup=0.816
PAR  | benchmark_test | T=4  | makespan=288   | avg_time=  2.116014 ms | speedup=0.773
PAR  | benchmark_test | T=8  | makespan=276   | avg_time= 19.366682 ms | speedup=0.084
PAR  | benchmark_test | T=16 | makespan=311   | avg_time=  9.169472 ms | speedup=0.178
PAR  | benchmark_test | T=32 | makespan=292   | avg_time= 19.146617 ms | speedup=0.085
```

Note: This input is still too small. Use the larger ones.

## Quick Reference Commands

```bash
# Generate a large input (takes a few seconds)
./generate_benchmark_input.sh 100 100 data/benchmark_final.jss

# Test with 2 repetitions (quick check)
./benchmark.sh 2 data/benchmark_final.jss

# Final benchmark with 10 repetitions (for report)
./benchmark.sh 10 data/benchmark_final.jss

# View results
cat benchmark_results.txt
```

## Notes

- The Greedy algorithm is simpler than Shifting Bottleneck, so parallelization overhead is more visible
- Larger inputs (100x100 or more) may be needed to reach 1-minute sequential runtime
- Results file automatically overwrites with each run - copy results before running again if needed
