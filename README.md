# Semantic Compress

**Deep Energy-Based Image Compression for Semantic Quality Preservation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for semantic-aware image compression that preserves perceptual and semantic quality by intelligently removing content with the lowest "deep energy" - a measure of semantic importance derived from deep neural network feature maps.

## Overview

Traditional image compression treats all pixels equally, leading to uniform quality loss. Semantic Compress uses deep learning to identify and preserve important image content (faces, objects, textures) while aggressively compressing less important regions (smooth backgrounds, repetitive patterns).

### Key Features

- **Deep Energy Functions**: Use pre-trained CNNs (VGG, ResNet, EfficientNet) to compute pixel importance
- **Content-Aware Resizing**: Seam carving with semantic energy preserves important content
- **Smart Cropping**: Find and preserve the most semantically important regions
- **Hybrid Compression**: Combine multiple techniques for optimal results
- **Multiple Energy Sources**: Gradient, saliency, deep features, or hybrid combinations

## Installation

```bash
# Clone the repository
git clone https://github.com/divkov/semantic-compress.git
cd semantic-compress

# Install dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision
- NumPy, Pillow, scikit-image, scipy

## Quick Start

```python
from semantic_compress import SemanticCompressor, DeepEnergyFunction

# Use deep energy for semantic-aware compression
compressor = SemanticCompressor(
    energy_function=DeepEnergyFunction(model_name="vgg16")
)

# Compress image to 50% size while preserving semantic quality
result = compressor.compress("photo.jpg", scale=0.5, method="hybrid")
result.save("compressed.jpg")

print(f"Compression ratio: {result.compression_ratio:.1%}")
```

## Energy Functions

The core of semantic compression is the energy function, which assigns importance scores to pixels.

### Gradient Energy (Traditional)

```python
from semantic_compress import GradientEnergyFunction

# Classic seam carving energy based on image gradients
energy_fn = GradientEnergyFunction()
```

### Deep Energy (Semantic)

```python
from semantic_compress import DeepEnergyFunction

# Use VGG16 features for semantic importance
energy_fn = DeepEnergyFunction(
    model_name="vgg16",  # or 'vgg19', 'resnet18', 'resnet50', 'efficientnet_b0'
    aggregate_method="weighted"  # 'sum', 'max', 'weighted'
)
```

### Saliency Energy (Visual Attention)

```python
from semantic_compress import SaliencyEnergyFunction

# Spectral residual saliency (fast, no ML)
energy_fn = SaliencyEnergyFunction(method="spectral")

# Fine-grained saliency (uses segmentation model)
energy_fn = SaliencyEnergyFunction(method="fine_grained")
```

### Hybrid Energy (Combined)

```python
from semantic_compress import HybridEnergyFunction

# Combine multiple energy sources
energy_fn = HybridEnergyFunction(
    energies=[
        GradientEnergyFunction(),
        DeepEnergyFunction(),
        SaliencyEnergyFunction(),
    ],
    weights=[0.3, 0.5, 0.2],
    adaptive_weights=False,  # Set True to auto-adjust per image
)
```

## Compression Methods

### 1. Seam Carving

Remove/insert seams (connected pixel paths) with lowest energy.

```python
result = compressor.compress(
    "image.jpg",
    scale=0.7,
    method="seam_carving",
)
```

### 2. Smart Crop

Crop to the most semantically important region.

```python
result = compressor.compress(
    "image.jpg",
    target_pixels=1000000,  # ~1MP
    method="smart_crop",
)
```

### 3. Hybrid (Recommended)

Combines smart crop + seam carving for best quality/size ratio.

```python
result = compressor.compress(
    "image.jpg",
    scale=0.5,
    method="hybrid",
)
```

## CLI Usage

```bash
# Basic compression with deep energy
python -m semantic_compress photo.jpg --scale 0.5 --energy deep

# Visualize what the model considers important
python -m semantic_compress photo.jpg --visualize-energy -o energy_map.png

# Show which seams would be removed
python -m semantic_compress photo.jpg --visualize-seams 20 -o seams.png

# Analyze content importance
python -m semantic_compress photo.jpg --analyze

# Use specific model and method
python -m semantic_compress photo.jpg -s 0.6 -e deep --model resnet50 -m hybrid
```

## How It Works

### Deep Energy Concept

Deep energy measures pixel importance using activations from pre-trained CNNs:

1. **Feature Extraction**: Pass image through layers of a pre-trained network (VGG, ResNet, etc.)
2. **Activation Mapping**: Sum absolute activations across feature channels at each spatial location
3. **Multi-Scale Aggregation**: Combine features from multiple layers (low-level edges → high-level semantics)
4. **Upsampling**: Resize feature importance maps back to original resolution

Pixels with higher deep energy correspond to:
- Object boundaries and textures
- Semantically meaningful patterns
- Regions that contribute to classification

Pixels with low deep energy:
- Smooth backgrounds
- Repetitive textures
- Redundant information

### Seam Carving with Deep Energy

Instead of removing low-gradient seams (traditional), we remove low-**semantic-energy** seams:

1. Compute deep energy map for entire image
2. Find optimal seam (8-connected path) with minimum cumulative energy
3. Remove seam, preserving high-energy content
4. Repeat until target size reached

This preserves faces, objects, and important textures while removing background pixels.

## Examples

### Example 1: Photo Compression

```python
from semantic_compress import compress_image

# Simple one-liner
result = compress_image(
    "family_photo.jpg",
    scale=0.6,
    method="hybrid",
)
result.save("family_compressed.jpg")
```

### Example 2: Batch Processing

```python
from pathlib import Path
from semantic_compress import SemanticCompressor, DeepEnergyFunction

compressor = SemanticCompressor(energy_function=DeepEnergyFunction())

images = list(Path("photos/").glob("*.jpg"))
results = compressor.compress_batch(
    images,
    scale=0.5,
    method="hybrid",
)

for img_path, result in zip(images, results):
    result.save(f"compressed/{img_path.name}")
```

### Example 3: Content Analysis

```python
compressor = SemanticCompressor(energy_function=DeepEnergyFunction())
stats = compressor.analyze_content("image.jpg")

print(f"High importance region: {stats['high_importance_ratio']:.1%}")
print(f"Mean energy: {stats['mean_energy']:.3f}")
```

## Research Background

This library implements techniques from seminal papers in content-aware image processing:

1. **Seam Carving**: Avidan & Shamir, "Seam Carving for Content-Aware Image Resizing" (SIGGRAPH 2007)
2. **Deep Feature Importance**: Mahendran & Vedaldi, "Understanding Deep Image Representations by Inverting Them" (CVPR 2015)
3. **Saliency Detection**: Hou & Zhang, "Saliency Detection: A Spectral Residual Approach" (CVPR 2007)
4. **Semantic Compression**: Works on using CNNs for perceptual quality metrics and learned image compression

## Performance

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| Gradient Energy | ⚡ Fast | ⭐⭐⭐ General | Batch processing |
| Deep Energy (VGG16) | 🐢 Medium | ⭐⭐⭐⭐⭐ Semantic | Quality-critical |
| Deep Energy (ResNet) | 🐢 Medium | ⭐⭐⭐⭐⭐ Semantic | Quality-critical |
| Saliency (Spectral) | ⚡ Fast | ⭐⭐⭐⭐ Visual attention | Real-time |
| Hybrid | 🐢 Medium | ⭐⭐⭐⭐⭐ Best | Recommended |

## API Reference

### SemanticCompressor

Main compression interface.

```python
compressor = SemanticCompressor(
    energy_function: EnergyFunction = None,
    quality_metric: str = "perceptual"
)

compressor.compress(
    image: str | Path | np.ndarray | Image.Image,
    target_bytes: int = None,
    target_pixels: int = None,
    max_dimension: int = None,
    scale: float = None,
    method: str = "hybrid",  # 'seam_carving', 'smart_crop', 'hybrid', 'jpeg'
    preserve_aspect: bool = True,
    show_progress: bool = True,
) -> CompressionResult
```

### SeamCarver

Low-level seam carving interface.

```python
carver = SeamCarver(
    energy_function: EnergyFunction = None,
    keep_mask: np.ndarray = None,
    remove_mask: np.ndarray = None,
)

resized = carver.resize(
    image,
    target_height: int = None,
    target_width: int = None,
    scale_height: float = None,
    scale_width: float = None,
)
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional pre-trained models for energy computation
- GPU optimization
- Real-time video compression
- Learned energy functions fine-tuned on perceptual metrics
- Additional compression codecs

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{semantic_compress,
  title = {Semantic Compress: Deep Energy-Based Image Compression},
  author = {divkov},
  year = {2026},
  url = {https://github.com/divkov/semantic-compress}
}
```
