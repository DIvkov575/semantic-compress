#!/usr/bin/env python3
"""Example: Compress a photo while preserving semantic quality."""

import sys
from pathlib import Path

# Add library to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_compress import (
    DeepEnergyFunction,
    HybridEnergyFunction,
    SaliencyEnergyFunction,
    SemanticCompressor,
)


def main():
    # Example 1: Basic compression with deep energy
    print("Example 1: Basic compression with deep energy")
    print("=" * 50)

    compressor = SemanticCompressor(
        energy_function=DeepEnergyFunction(model_name="vgg16")
    )

    # This would work with a real image:
    # result = compressor.compress("photo.jpg", scale=0.5, method="hybrid")
    # result.save("photo_compressed.jpg")

    print("Compressor configured with VGG16 deep energy")
    print("Usage: result = compressor.compress('image.jpg', scale=0.5)")
    print()

    # Example 2: Compare different energy functions
    print("Example 2: Available energy functions")
    print("=" * 50)

    energies = {
        "Gradient (Traditional)": "GradientEnergyFunction()",
        "Deep (VGG16)": "DeepEnergyFunction('vgg16')",
        "Deep (ResNet50)": "DeepEnergyFunction('resnet50')",
        "Saliency": "SaliencyEnergyFunction()",
        "Hybrid": "HybridEnergyFunction()",
    }

    for name, code in energies.items():
        print(f"  {name}: {code}")
    print()

    # Example 3: Content analysis
    print("Example 3: Analyze image content")
    print("=" * 50)
    print("stats = compressor.analyze_content('image.jpg')")
    print("print(f'High importance: {stats[\"high_importance_ratio\"]:.1%}')")
    print()

    # Example 4: Batch processing
    print("Example 4: Batch compression")
    print("=" * 50)
    print("""
from pathlib import Path

compressor = SemanticCompressor(energy_function=DeepEnergyFunction())

images = list(Path("photos/").glob("*.jpg"))
results = compressor.compress_batch(images, scale=0.5, method="hybrid")

for img_path, result in zip(images, results):
    result.save(f"compressed/{img_path.name}")
    print(f"Saved: {img_path.name} ({result.compression_ratio:.1%})")
""")


if __name__ == "__main__":
    main()
