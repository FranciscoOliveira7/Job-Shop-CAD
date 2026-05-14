#!/usr/bin/env python3
"""
Parse `benchmark_results_bb.txt` and generate plots (avg runtime and speedup).

Usage: python3 scripts/plot_benchmark_bb.py [path/to/benchmark_results_bb.txt]

Generates PNG files under `graphs/`.
"""
import os
import re
import sys
from pathlib import Path


def parse_results(path):
    """Parse benchmark_results_bb.txt and extract results."""
    results = {'baseline': None, 'threads': {}, 'makespan': {}}
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse SEQ line
            if line.startswith('SEQ'):
                m_avg = re.search(r'avg_run=\s*([0-9.]+)\s*ms', line)
                m_mk = re.search(r'makespan=\s*(\d+)', line)
                if m_avg:
                    results['baseline'] = float(m_avg.group(1))
                if m_mk:
                    results['makespan']['SEQ'] = int(m_mk.group(1))
            
            # Parse PAR lines
            elif line.startswith('PAR'):
                m_t = re.search(r'T=(\d+)', line)
                m_avg = re.search(r'avg_run=\s*([0-9.]+)\s*ms', line)
                m_mk = re.search(r'makespan=\s*(\d+)', line)
                if m_t and m_avg:
                    t = int(m_t.group(1))
                    avg = float(m_avg.group(1))
                    mk = int(m_mk.group(1)) if m_mk else None
                    results['threads'][t] = {'avg': avg}
                    if mk:
                        results['makespan'][f'PC{t}'] = mk
    
    return results


def make_plots(results, out_dir):
    """Generate runtime and speedup plots."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib is required. Install with: python3 -m pip install matplotlib')
        raise
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not results['baseline']:
        print('No baseline (SEQ) result found.')
        return
    
    threads = sorted(results['threads'].keys())
    avgs = [results['threads'][t]['avg'] for t in threads]
    speedups = [results['baseline'] / a if a > 0 else 0 for a in avgs]
    baseline = results['baseline']
    
    # Plot 1: Average runtime
    plt.figure(figsize=(6, 4))
    plt.plot(threads, avgs, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Threads')
    plt.ylabel('Avg run (ms)')
    plt.xscale('log', base=2)
    plt.xticks(threads, threads)
    plt.yscale('log')
    plt.title('Branch-and-Bound — Avg runtime')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bb_avg_runtime.png'), dpi=100)
    plt.close()
    
    # Plot 2: Speedup
    plt.figure(figsize=(6, 4))
    plt.plot(threads, speedups, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Threads')
    plt.ylabel('Speedup')
    plt.xscale('log', base=2)
    plt.xticks(threads, threads)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.title(f'Branch-and-Bound — Speedup (baseline seq={baseline:.2f} ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bb_speedup.png'), dpi=100)
    plt.close()
    
    print(f'Graphs written to {out_dir}')
    print(f'  - bb_avg_runtime.png')
    print(f'  - bb_speedup.png')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results_bb.txt'
    
    if not os.path.exists(path):
        print(f'Error: {path} not found')
        sys.exit(1)
    
    results = parse_results(path)
    out_dir = os.path.join(os.path.dirname(path), 'graphs')
    make_plots(results, out_dir)


if __name__ == '__main__':
    main()
