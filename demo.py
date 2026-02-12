#!/usr/bin/env python3
"""
Quick demo of semantic-compress library.
Run this to see the library in action (requires a sample image).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_compress import (
    DeepEnergyFunction,
    GradientEnergyFunction,
    HybridEnergyFunction,
    SaliencyEnergyFunction,
    SeamCarver,
    SemanticCompressor,
)


def create_sample_image(path: str = "sample.png") -> str:
    """Create a sample image with distinct regions for testing."""
    print("Creating sample image...")

    # Create 400x400 image
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Sky background (blue)
    img[:200, :] = [135, 206, 235]

    # Grass (green)
    img[200:, :] = [34, 139, 34]

    # Sun (yellow circle with high importance)
    center_x, center_y = 320, 80
    radius = 40
    y, x = np.ogrid[:400, :400]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img[mask] = [255, 215, 0]

    # Tree (brown trunk + green leaves)
    # Trunk
    img[250:350, 100:120] = [101, 67, 33]
    # Leaves
    img[150:260, 60:160] = [0, 128, 0]

    # House (red roof, white walls)
    # Walls
    img[220:320, 220:320] = [255, 255, 255]
    # Roof
    for i in range(60):
        width = 60 - i
        start = 240 - i
        img[160+i, 240-width:240+width] = [178, 34, 34]

    # Convert and save
    pil_img = Image.fromarray(img)
    pil_img.save(path)
    print(f"Sample image saved to: {path}")
    return path


def demo_energy_functions(image_path: str) -> None:
    """Demo different energy functions."""
    print("\n" + "="*60)
    print("DEMO: Energy Functions")
    print("="*60)

    img = np.array(Image.open(image_path).convert("RGB"))

    energy_functions = [
        ("Gradient (Traditional)", GradientEnergyFunction()),
        ("Saliency (Spectral)", SaliencyEnergyFunction(method="spectral")),
    ]

    # Add deep energy if torch is available
    try:
        import torch
        energy_functions.append(
            ("Deep Energy (VGG16)", DeepEnergyFunction(model_name="vgg16"))
        )
        energy_functions.append(
            ("Hybrid", HybridEnergyFunction())
        )
    except ImportError:
        print("Note: PyTorch not available, skipping deep energy demos")

    for name, energy_fn in energy_functions:
        print(f"\nComputing {name}...")
        energy = energy_fn.compute(img)
        print(f"  Shape: {energy.shape}")
        print(f"  Min: {energy.min():.3f}, Max: {energy.max():.3f}")
        print(f"  Mean: {energy.mean():.3f}, Std: {energy.std():.3f}")

        # Save visualization
        from matplotlib import colormaps
        cmap = colormaps.get_cmap("hot")
        vis = (cmap(energy)[:, :, :3] * 255).astype(np.uint8)
        vis_path = f"energy_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        Image.fromarray(vis).save(vis_path)
        print(f"  Saved visualization: {vis_path}")


def demo_seam_carving(image_path: str) -> None:
    """Demo seam carving compression."""
    print("\n" + "="*60)
    print("DEMO: Seam Carving")
    print("="*60)

    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    print(f"Original size: {w}x{h}")

    # Create carver with gradient energy
    carver = SeamCarver(energy_function=GradientEnergyFunction())

    # Compress to 75% width
    target_w = int(w * 0.75)
    print(f"\nRemoving {w - target_w} vertical seams...")
    result = carver.resize(img, target_width=target_w, show_progress=True)
    print(f"New size: {result.shape[1]}x{result.shape[0]}")

    output_path = "compressed_gradient.png"
    Image.fromarray(result.astype(np.uint8)).save(output_path)
    print(f"Saved: {output_path}")

    # Visualize seams
    print("\nVisualizing seams that would be removed...")
    vis_seams = carver.visualize_seams(img, n_seams=50, direction="vertical")
    seams_path = "seams_visualization.png"
    Image.fromarray(vis_seams).save(seams_path)
    print(f"Saved: {seams_path}")


def demo_semantic_compressor(image_path: str) -> None:
    """Demo the high-level compressor."""
    print("\n" + "="*60)
    print("DEMO: Semantic Compressor")
    print("="*60)

    # Use hybrid energy
    try:
        compressor = SemanticCompressor(
            energy_function=HybridEnergyFunction()
        )
    except:
        compressor = SemanticCompressor()

    print("\nAnalyzing content...")
    stats = compressor.analyze_content(image_path)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\nCompressing with hybrid method (50%)...")
    result = compressor.compress(image_path, scale=0.5, method="hybrid")
    print(f"Original: {result.original_size}")
    print(f"Compressed: {result.compressed_size}")
    print(f"Ratio: {result.compression_ratio:.1%}")

    output_path = "compressed_hybrid.png"
    result.save(output_path)
    print(f"Saved: {output_path}")


def main():
    """Run all demos."""
    print("="*60)
    print("SEMANTIC COMPRESS - Demo")
    print("="*60)

    # Create sample image
    sample_path = create_sample_image()

    # Run demos
    demo_energy_functions(sample_path)
    demo_seam_carving(sample_path)
    demo_semantic_compressor(sample_path)

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  sample.png - Original test image")
    print("  energy_*.png - Energy map visualizations")
    print("  compressed_gradient.png - Seam carving result")
    print("  seams_visualization.png - Seams overlay")
    print("  compressed_hybrid.png - Hybrid compression result")


if __name__ == "__main__":
    main()
