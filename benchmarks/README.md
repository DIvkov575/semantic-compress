# Semantic-Compress Benchmarks

Rigorous benchmarking suite for evaluating semantic-compress performance, cost reduction, and accuracy.

## Overview

This benchmark suite evaluates:

### Performance Metrics
- **Processing Time**: Breakdown of preprocessing, energy computation, and seam carving
- **Throughput**: Megapixels processed per second
- **Memory Usage**: Peak memory consumption during compression
- **Scalability**: Performance across different image sizes

### Inference Cost Reduction
- **Compression Ratio**: Actual vs target compression ratios achieved
- **Pixels Removed**: Percentage of pixels removed while preserving semantics
- **Energy Function Calls**: Computational cost in terms of model forward passes
- **Bits Per Pixel (BPP)**: Storage efficiency

### Accuracy Metrics
- **SSIM**: Structural Similarity Index (structural preservation)
- **MS-SSIM**: Multi-Scale SSIM (multi-scale structural quality)
- **PSNR**: Peak Signal-to-Noise Ratio (pixel-level fidelity)
- **LPIPS**: Learned Perceptual Image Patch Similarity (perceptual distance)
- **Feature Similarity**: Cosine similarity of deep features (semantic preservation)
- **Edge Preservation**: Correlation of gradient magnitudes
- **Color Fidelity**: Histogram correlation in LAB color space
- **Structural Similarity**: Local structure tensor analysis

## Installation

```bash
cd benchmarks
pip install -r requirements.txt
```

## Usage

### Run All Benchmarks

```bash
python run_benchmarks.py
```

Options:
- `--sizes`: Image sizes to test (default: 256 512 1024)
- `--ratios`: Target compression ratios (default: 0.9 0.8 0.7 0.6 0.5)
- `--output`: Output directory (default: benchmark_results)

Example:
```bash
python run_benchmarks.py --sizes 512 1024 --ratios 0.8 0.6 0.4 --output my_results
```

### Run Baseline Comparison

```bash
python compare_baselines.py
```

Compares semantic-compress against:
- Bilinear resizing
- Bicubic resizing
- Lanczos resizing
- JPEG compression
- WebP compression

### Visualize Results

```bash
python visualize.py
```

Generates plots:
- `ssim_vs_ratio.png`: SSIM vs compression ratio
- `psnr_vs_ratio.png`: PSNR vs compression ratio
- `time_vs_size.png`: Processing time scalability
- `throughput.png`: Throughput comparison
- `accuracy_radar.png`: Radar chart of accuracy metrics
- `cost_reduction.png`: Cost reduction metrics
- `benchmark_summary.png`: Comprehensive summary

## Results Structure

Results are saved in `benchmark_results/`:

```
benchmark_results/
├── benchmark_results.json      # Raw benchmark data
├── benchmark_report.md         # Markdown report
├── comparison_results.json     # Baseline comparison data
├── comparison_report.md        # Comparison report
└── plots/
    ├── ssim_vs_ratio.png
    ├── psnr_vs_ratio.png
    ├── benchmark_summary.png
    └── ...
```

## Interpreting Results

### Good Performance Indicators
- **High SSIM** (>0.9): Good structural preservation
- **High PSNR** (>30 dB): Good pixel-level fidelity
- **Low LPIPS** (<0.1): Low perceptual distance
- **High Feature Similarity** (>0.9): Good semantic preservation
- **High Throughput** (>1 MP/s): Fast processing

### Cost Reduction
- **High Pixel Removal %**: Effective compression
- **Low Energy Function Calls**: Efficient computation
- **Low BPP**: Good storage efficiency

## Example Output

```
Running benchmarks on 3 image sizes
Compression ratios: [0.9, 0.8, 0.7, 0.6, 0.5]

Benchmarking image size: (256, 256)
  Testing Gradient...
  Testing Saliency...
  Testing Deep...
  Testing Hybrid...

Results saved to: benchmark_results/benchmark_results.json
Report saved to: benchmark_results/benchmark_report.md
```

## Extending Benchmarks

To add a new energy function:

```python
from semantic_compress.energy import EnergyFunction

class MyEnergyFunction(EnergyFunction):
    def compute(self, image):
        # Your implementation
        pass

# In run_benchmarks.py, add to energy_functions list
energy_functions = [
    ...
    MyEnergyFunction(),
]
```

## Citation

When using these benchmarks, please cite:

```bibtex
@software{semantic_compress_benchmarks,
  title = {Semantic-Compress Benchmarks},
  author = {divkov575},
  year = {2026},
  url = {https://github.com/divkov/semantic-compress}
}
```
