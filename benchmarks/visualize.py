#!/usr/bin/env python3
"""
Visualize benchmark results.

Creates plots for:
- SSIM vs Compression Ratio
- PSNR vs Compression Ratio
- Processing Time vs Image Size
- Throughput comparison
- Memory usage
- Feature preservation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_results(results_dir: Path) -> List[Dict]:
    """Load benchmark results from JSON."""
    results_file = results_dir / "benchmark_results.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file) as f:
        data = json.load(f)
    
    return data.get("results", [])


def plot_ssim_vs_ratio(results: List[Dict], output_dir: Path):
    """Plot SSIM vs compression ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = set(r["method_name"] for r in results)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(sorted(methods)):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        ssims = [r["ssim"] for r in method_results]
        
        ax.scatter(ratios, ssims, label=method, alpha=0.7, s=100, color=colors[i])
        
        # Add trend line
        if len(ratios) > 1:
            z = np.polyfit(ratios, ssims, 2)
            p = np.poly1d(z)
            x_line = np.linspace(min(ratios), max(ratios), 100)
            ax.plot(x_line, p(x_line), '--', alpha=0.5, color=colors[i])
    
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("SSIM (higher is better)", fontsize=12)
    ax.set_title("Structural Similarity vs Compression Ratio", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ssim_vs_ratio.png", dpi=150)
    plt.close()
    print(f"Saved: ssim_vs_ratio.png")


def plot_psnr_vs_ratio(results: List[Dict], output_dir: Path):
    """Plot PSNR vs compression ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = set(r["method_name"] for r in results)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(sorted(methods)):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        psnrs = [r["psnr"] for r in method_results]
        
        ax.scatter(ratios, psnrs, label=method, alpha=0.7, s=100, color=colors[i])
        
        if len(ratios) > 1:
            z = np.polyfit(ratios, psnrs, 2)
            p = np.poly1d(z)
            x_line = np.linspace(min(ratios), max(ratios), 100)
            ax.plot(x_line, p(x_line), '--', alpha=0.5, color=colors[i])
    
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("PSNR (dB, higher is better)", fontsize=12)
    ax.set_title("Peak Signal-to-Noise Ratio vs Compression Ratio", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "psnr_vs_ratio.png", dpi=150)
    plt.close()
    print(f"Saved: psnr_vs_ratio.png")


def plot_time_vs_size(results: List[Dict], output_dir: Path):
    """Plot processing time vs image size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = set(r["method_name"] for r in results)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(sorted(methods)):
        method_results = [r for r in results if r["method_name"] == method]
        # Use total pixels as x-axis
        sizes = [r["image_size"][0] * r["image_size"][1] / 1e6 for r in method_results]
        times = [r["total_time_ms"] for r in method_results]
        
        ax.scatter(sizes, times, label=method, alpha=0.7, s=100, color=colors[i])
    
    ax.set_xlabel("Image Size (Megapixels)", fontsize=12)
    ax.set_ylabel("Processing Time (ms)", fontsize=12)
    ax.set_title("Processing Time vs Image Size", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_size.png", dpi=150)
    plt.close()
    print(f"Saved: time_vs_size.png")


def plot_throughput(results: List[Dict], output_dir: Path):
    """Plot throughput comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = sorted(set(r["method_name"] for r in results))
    throughputs = []
    
    for method in methods:
        method_results = [r for r in results if r["method_name"] == method]
        avg_throughput = np.mean([r["throughput_mbps"] for r in method_results])
        throughputs.append(avg_throughput)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, throughputs, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel("Throughput (MP/s)", fontsize=12)
    ax.set_title("Average Processing Throughput", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "throughput.png", dpi=150)
    plt.close()
    print(f"Saved: throughput.png")


def plot_accuracy_radar(results: List[Dict], output_dir: Path):
    """Plot radar chart of accuracy metrics."""
    methods = sorted(set(r["method_name"] for r in results))
    
    metrics = ["ssim", "psnr", "feature_similarity", "edge_preservation", 
               "color_fidelity", "structural_similarity"]
    metric_labels = ["SSIM", "PSNR", "Feature Sim", "Edge Pres", 
                     "Color Fidelity", "Struct Sim"]
    
    # Normalize PSNR to 0-1 range (assuming max 50 dB)
    def normalize_metric(r, metric):
        val = r[metric]
        if metric == "psnr":
            return min(val / 50.0, 1.0)
        return val
    
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5),
                            subplot_kw=dict(polar=True))
    
    if len(methods) == 1:
        axes = [axes]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for ax, method in zip(axes, methods):
        method_results = [r for r in results if r["method_name"] == method]
        
        values = [np.mean([normalize_metric(r, m) for r in method_results]) 
                 for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(method, size=12, y=1.08)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_radar.png", dpi=150)
    plt.close()
    print(f"Saved: accuracy_radar.png")


def plot_cost_reduction(results: List[Dict], output_dir: Path):
    """Plot inference cost reduction metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = sorted(set(r["method_name"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Pixels removed
    ax = axes[0]
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        pixels_removed = [r["pixels_removed_percent"] for r in method_results]
        ax.scatter(ratios, pixels_removed, label=method, alpha=0.7, 
                  s=100, color=colors[i])
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Pixels Removed (%)")
    ax.set_title("Pixel Removal Efficiency")
    ax.grid(True, alpha=0.3)
    
    # Memory usage
    ax = axes[1]
    memory_avgs = []
    for method in methods:
        method_results = [r for r in results if r["method_name"] == method]
        avg_mem = np.mean([r["peak_memory_mb"] for r in method_results])
        memory_avgs.append(avg_mem)
    ax.bar(methods, memory_avgs, color=colors, alpha=0.8)
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Usage")
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Iterations (energy function calls)
    ax = axes[2]
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        if method_results and "energy_function_calls" in method_results[0]:
            ratios = [r["actual_compression_ratio"] for r in method_results]
            iterations = [r["energy_function_calls"] for r in method_results]
            ax.scatter(ratios, iterations, label=method, alpha=0.7, 
                      s=100, color=colors[i])
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Energy Function Calls")
    ax.set_title("Computational Cost")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "cost_reduction.png", dpi=150)
    plt.close()
    print(f"Saved: cost_reduction.png")


def generate_summary_image(results: List[Dict], output_dir: Path):
    """Generate a summary comparison image."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    methods = sorted(set(r["method_name"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # SSIM vs Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        ssims = [r["ssim"] for r in method_results]
        ax1.scatter(ratios, ssims, label=method, alpha=0.7, color=colors[i])
    ax1.set_xlabel("Compression Ratio")
    ax1.set_ylabel("SSIM")
    ax1.set_title("SSIM vs Compression")
    ax1.grid(True, alpha=0.3)
    
    # PSNR vs Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        psnrs = [r["psnr"] for r in method_results]
        ax2.scatter(ratios, psnrs, label=method, alpha=0.7, color=colors[i])
    ax2.set_xlabel("Compression Ratio")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR vs Compression")
    ax2.grid(True, alpha=0.3)
    
    # Throughput
    ax3 = fig.add_subplot(gs[0, 2])
    throughputs = []
    for method in methods:
        method_results = [r for r in results if r["method_name"] == method]
        avg = np.mean([r["throughput_mbps"] for r in method_results])
        throughputs.append(avg)
    ax3.bar(methods, throughputs, color=colors, alpha=0.8)
    ax3.set_ylabel("Throughput (MP/s)")
    ax3.set_title("Processing Speed")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Feature Similarity
    ax4 = fig.add_subplot(gs[1, 0])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        feat_sims = [r["feature_similarity"] for r in method_results]
        ax4.scatter(ratios, feat_sims, label=method, alpha=0.7, color=colors[i])
    ax4.set_xlabel("Compression Ratio")
    ax4.set_ylabel("Feature Similarity")
    ax4.set_title("Semantic Preservation")
    ax4.grid(True, alpha=0.3)
    
    # Edge Preservation
    ax5 = fig.add_subplot(gs[1, 1])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        ratios = [r["actual_compression_ratio"] for r in method_results]
        edges = [r["edge_preservation"] for r in method_results]
        ax5.scatter(ratios, edges, label=method, alpha=0.7, color=colors[i])
    ax5.set_xlabel("Compression Ratio")
    ax5.set_ylabel("Edge Preservation")
    ax5.set_title("Edge Quality")
    ax5.grid(True, alpha=0.3)
    
    # Time vs Size
    ax6 = fig.add_subplot(gs[1, 2])
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method_name"] == method]
        sizes = [r["image_size"][0] * r["image_size"][1] / 1e6 for r in method_results]
        times = [r["total_time_ms"] for r in method_results]
        ax6.scatter(sizes, times, label=method, alpha=0.7, color=colors[i])
    ax6.set_xlabel("Image Size (MP)")
    ax6.set_ylabel("Time (ms)")
    ax6.set_title("Scalability")
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    # Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create summary statistics
    summary_data = []
    for method in methods:
        method_results = [r for r in results if r["method_name"] == method]
        summary_data.append([
            method,
            f"{np.mean([r['ssim'] for r in method_results]):.3f}",
            f"{np.mean([r['psnr'] for r in method_results]):.1f}",
            f"{np.mean([r['feature_similarity'] for r in method_results]):.3f}",
            f"{np.mean([r['throughput_mbps'] for r in method_results]):.2f}",
            f"{np.mean([r['total_time_ms'] for r in method_results]):.1f}",
        ])
    
    table = ax7.table(
        cellText=summary_data,
        colLabels=["Method", "Avg SSIM", "Avg PSNR", "Avg Feature Sim", "Throughput", "Avg Time"],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax7.set_title("Summary Statistics", fontsize=14, pad=20)
    
    plt.suptitle("Semantic-Compress Benchmark Summary", fontsize=16, y=0.98)
    plt.savefig(output_dir / "benchmark_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: benchmark_summary.png")


def main():
    results_dir = Path("benchmark_results")
    output_dir = results_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading benchmark results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} results")
    
    print("\nGenerating visualizations...")
    plot_ssim_vs_ratio(results, output_dir)
    plot_psnr_vs_ratio(results, output_dir)
    plot_time_vs_size(results, output_dir)
    plot_throughput(results, output_dir)
    plot_accuracy_radar(results, output_dir)
    plot_cost_reduction(results, output_dir)
    generate_summary_image(results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
