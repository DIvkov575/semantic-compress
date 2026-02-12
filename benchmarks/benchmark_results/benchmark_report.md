# Semantic Compress Benchmark Report

Generated: 2026-02-12 06:22:41

## Summary

### Saliency

- Average SSIM: 0.8506
- Average PSNR: 19.23 dB
- Average Time: 361.54 ms
- Average Throughput: 0.05 MP/s
- Average Memory: 0.00 MB

### Deep

- Average SSIM: 0.8399
- Average PSNR: 19.25 dB
- Average Time: 2008.17 ms
- Average Throughput: 0.01 MP/s
- Average Memory: 0.00 MB

### Gradient

- Average SSIM: 0.7681
- Average PSNR: 12.50 dB
- Average Time: 340.83 ms
- Average Throughput: 0.05 MP/s
- Average Memory: 0.73 MB

### Hybrid

- Average SSIM: 0.8501
- Average PSNR: 19.23 dB
- Average Time: 2141.62 ms
- Average Throughput: 0.01 MP/s
- Average Memory: 0.00 MB

## Detailed Results

| Method | Size | Target Ratio | Actual Ratio | Time (ms) | SSIM | PSNR | LPIPS |
|--------|------|--------------|--------------|-----------|------|------|-------|
| Gradient | 128x128 | 0.80 | 0.793 | 340.8 | 0.7681 | 12.50 | 0.2312 |
| Saliency | 128x128 | 0.80 | 0.793 | 361.5 | 0.8506 | 19.23 | 0.0728 |
| Deep | 128x128 | 0.80 | 0.793 | 2008.2 | 0.8399 | 19.25 | 0.0753 |
| Hybrid | 128x128 | 0.80 | 0.793 | 2141.6 | 0.8501 | 19.23 | 0.0726 |
