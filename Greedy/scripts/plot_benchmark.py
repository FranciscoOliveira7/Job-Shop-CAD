#!/usr/bin/env python3
"""
Parse `benchmark_results.txt` and generate plots (avg runtime and speedup) per input.

Usage: python3 scripts/plot_benchmark.py [path/to/benchmark_results.txt]

Generates PNG files under `graphs/`.
"""
import os
import re
import sys
from collections import defaultdict, OrderedDict


def parse_results(path):
    tests = OrderedDict()
    cur = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^=== Testing: (.+?) \(', line)
            if m:
                cur = m.group(1).strip()
                tests[cur] = {'baseline': None, 'par': {}}
                continue

            if not cur:
                continue

            if line.startswith('SEQ'):
                m_avg = re.search(r'avg_run=\s*([0-9.]+)\s*ms', line)
                if m_avg:
                    tests[cur]['baseline'] = float(m_avg.group(1))
                m_mk = re.search(r'makespan=\s*(\d+)', line)
                if m_mk:
                    tests[cur]['makespan'] = int(m_mk.group(1))
            elif line.startswith('PAR'):
                # find thread T and avg
                m_t = re.search(r'T=(\d+)', line)
                m_avg = re.search(r'avg_run=\s*([0-9.]+)\s*ms', line)
                m_mk = re.search(r'makespan=\s*(\d+)', line)
                if m_t and m_avg:
                    t = int(m_t.group(1))
                    avg = float(m_avg.group(1))
                    mk = int(m_mk.group(1)) if m_mk else None
                    tests[cur]['par'][t] = {'avg': avg, 'makespan': mk}

    return tests


def make_plots(tests, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib is required. Install with: python3 -m pip install matplotlib')
        raise

    os.makedirs(out_dir, exist_ok=True)

    # collect threads set for combined plot
    all_threads = set()
    for name, data in tests.items():
        all_threads.update(data['par'].keys())
    threads_sorted = sorted(all_threads)

    # Combined speedup plot
    plt.figure(figsize=(8,6))
    for name, data in tests.items():
        baseline = data['baseline']
        if not baseline:
            continue
        xs = []
        ys = []
        for t in threads_sorted:
            if t in data['par']:
                xs.append(t)
                ys.append(baseline / data['par'][t]['avg'])
        if xs:
            plt.plot(xs, ys, marker='o', label=name)
    plt.xlabel('Threads')
    plt.ylabel('Speedup (SEQ_avg / PAR_avg)')
    plt.xscale('log', base=2)
    plt.xticks(threads_sorted, threads_sorted)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.title('Benchmark: Speedup curves (combined)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combined_speedup.png'))
    plt.close()

    # Per-test plots
    for name, data in tests.items():
        baseline = data['baseline']
        if not baseline:
            continue
        threads = sorted(data['par'].keys())
        avgs = [data['par'][t]['avg'] for t in threads]
        speedups = [baseline / a if a > 0 else 0 for a in avgs]

        # avg runtime plot
        plt.figure(figsize=(6,4))
        plt.plot(threads, avgs, marker='o')
        plt.xlabel('Threads')
        plt.ylabel('Avg run (ms)')
        plt.xscale('log', base=2)
        plt.xticks(threads, threads)
        plt.yscale('log')
        plt.title(f'{name} — Avg runtime')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.tight_layout()
        outpath = os.path.join(out_dir, f'{name}_avg_runtime.png')
        plt.savefig(outpath)
        plt.close()

        # speedup plot
        plt.figure(figsize=(6,4))
        plt.plot(threads, speedups, marker='o')
        plt.xlabel('Threads')
        plt.ylabel('Speedup')
        plt.xscale('log', base=2)
        plt.xticks(threads, threads)
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.title(f'{name} — Speedup (baseline seq={baseline} ms)')
        plt.tight_layout()
        outpath = os.path.join(out_dir, f'{name}_speedup.png')
        plt.savefig(outpath)
        plt.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results.txt'
    if not os.path.exists(path):
        print('benchmark_results.txt not found at', path)
        sys.exit(1)

    tests = parse_results(path)
    out_dir = os.path.join(os.path.dirname(path), 'graphs')
    make_plots(tests, out_dir)
    print('Graphs written to', out_dir)


if __name__ == '__main__':
    main()
