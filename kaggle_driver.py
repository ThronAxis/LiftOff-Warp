#!/usr/bin/env python3
"""
LIFTOFF Kaggle Notebook Driver — Full Benchmark & Visualization
Run this script on Kaggle with GPU accelerator enabled.
"""

import subprocess
import os
import csv
import sys

# ════════════════════════════════════════════════════════════════════════
# 1. GPU Verification
# ════════════════════════════════════════════════════════════════════════

def verify_gpu():
    """Check GPU availability and print device info."""
    print("=" * 60)
    print("  LIFTOFF — GPU Environment Verification")
    print("=" * 60)
    
    gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if gpu_info.returncode != 0:
        print("ERROR: nvidia-smi not found. No GPU available.")
        sys.exit(1)
    print(gpu_info.stdout)
    
    nvcc_ver = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if nvcc_ver.returncode != 0:
        print("ERROR: nvcc not found. CUDA toolkit not installed.")
        sys.exit(1)
    print(nvcc_ver.stdout)

# ════════════════════════════════════════════════════════════════════════
# 2. Build Helper
# ════════════════════════════════════════════════════════════════════════

def detect_sm():
    """Detect SM version from nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        cap = result.stdout.strip().split('\n')[0].replace('.', '')
        return cap
    return '75'  # default T4

def nvcc_build(src, out, sm='75', extra_flags=None):
    """Build a CUDA source file with nvcc."""
    flags = [
        'nvcc', '-O3', f'-arch=sm_{sm}',
        '--use_fast_math',
        '-std=c++17',
        '-I./liftoff',
        '-I/usr/local/cuda/include',
        src, '-o', out
    ]
    if extra_flags:
        flags.extend(extra_flags)
    
    print(f"\n[BUILD] {' '.join(flags)}")
    result = subprocess.run(flags, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"NVCC ERROR:\n{result.stderr}")
        return False
    print(f"Build OK: {out}")
    return True

# ════════════════════════════════════════════════════════════════════════
# 3. Build & Run
# ════════════════════════════════════════════════════════════════════════

def run_benchmarks():
    """Build and run the benchmark suite."""
    sm = detect_sm()
    print(f"\nDetected SM version: sm_{sm}")
    
    # Build correctness tests
    print("\n── Building Correctness Tests ──────────────────────────────")
    test_ok = nvcc_build(
        'liftoff/tests/correctness.cu', 'liftoff_test',
        sm=sm, extra_flags=['-lineinfo']
    )
    
    if test_ok:
        print("\n── Running Correctness Tests ───────────────────────────────")
        result = subprocess.run(['./liftoff_test'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("WARNING: Some tests failed!")
            print(result.stderr)
    
    # Build benchmarks
    print("\n── Building Benchmarks ─────────────────────────────────────")
    bench_ok = nvcc_build(
        'liftoff/bench/benchmark_main.cu', 'liftoff_bench',
        sm=sm, extra_flags=['-lineinfo']
    )
    
    if bench_ok:
        print("\n── Running Benchmarks ──────────────────────────────────────")
        result = subprocess.run(['./liftoff_bench'], capture_output=True, text=True)
        print(result.stdout)
        return True
    return False

# ════════════════════════════════════════════════════════════════════════
# 4. Plotting
# ════════════════════════════════════════════════════════════════════════

def plot_bench_results(csv_path='liftoff_bench_results.csv'):
    """Generate benchmark visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available for plotting.")
        return
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    names, medians, gops = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row['name'].split('(')[0].strip())
            medians.append(float(row['median_us']))
            gops.append(float(row['gops']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('LIFTOFF Warp Primitives — Benchmark Results',
                 fontsize=16, fontweight='bold', color='#1a1a2e')
    
    colors = ['#2196F3', '#f44336', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    colors = colors[:len(names)]
    
    # Latency plot
    bars1 = ax1.barh(names, medians, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Median Latency (μs)', fontsize=12)
    ax1.set_title('Kernel Latency (lower is better)', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars1, medians):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}μs', va='center', fontsize=10, fontweight='bold')
    
    # Throughput plot
    bars2 = ax2.barh(names, gops, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Throughput (GOPS)', fontsize=12)
    ax2.set_title('Compute Throughput (higher is better)', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars2, gops):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('liftoff_benchmark.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Plot saved: liftoff_benchmark.png")

# ════════════════════════════════════════════════════════════════════════
# 5. Register Pressure & PTX Analysis
# ════════════════════════════════════════════════════════════════════════

def analyze_register_pressure(sm='75'):
    """Check register usage of compiled kernels."""
    print("\n── Register Pressure Analysis ──────────────────────────────")
    result = subprocess.run(
        ['nvcc', '-O3', f'-arch=sm_{sm}', '--use_fast_math', '-std=c++17',
         '-I./liftoff', '--ptxas-options=-v',
         'liftoff/bench/benchmark_main.cu', '-o', '/dev/null'],
        capture_output=True, text=True
    )
    for line in result.stderr.split('\n'):
        if 'registers' in line.lower() or 'smem' in line.lower():
            print(f"  {line.strip()}")

def analyze_ptx(sm='75'):
    """Generate and inspect PTX for shuffle instructions."""
    print("\n── PTX Instruction Analysis ────────────────────────────────")
    result = subprocess.run(
        ['nvcc', '-O3', f'-arch=sm_{sm}', '--use_fast_math', '-std=c++17',
         '-I./liftoff', '-ptx',
         'liftoff/bench/benchmark_main.cu', '-o', 'liftoff_bench.ptx'],
        capture_output=True, text=True
    )
    if result.returncode == 0 and os.path.exists('liftoff_bench.ptx'):
        with open('liftoff_bench.ptx') as f:
            ptx = f.read()
        
        shfl_count = ptx.count('shfl')
        ld_shared = ptx.count('ld.shared')
        st_shared = ptx.count('st.shared')
        
        print(f"  Shuffle instructions: {shfl_count}")
        print(f"  Shared memory loads:  {ld_shared}")
        print(f"  Shared memory stores: {st_shared}")
        print(f"  Shuffle-to-shared ratio: {shfl_count}:{ld_shared + st_shared}")

# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    verify_gpu()
    success = run_benchmarks()
    if success:
        plot_bench_results()
        sm = detect_sm()
        analyze_register_pressure(sm)
        analyze_ptx(sm)
    print("\n✓ LIFTOFF Kaggle notebook complete.")
