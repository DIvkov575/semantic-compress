#!/usr/bin/env python3
"""CLI for semantic image compression."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from semantic_compress import (
    DeepEnergyFunction,
    GradientEnergyFunction,
    HybridEnergyFunction,
    SaliencyEnergyFunction,
    SemanticCompressor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-aware image compression preserving content quality"
    )
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("-o", "--output", type=str, help="Output image path")
    parser.add_argument(
        "-s", "--scale", type=float, help="Scale factor (e.g., 0.5 for half size)"
    )
    parser.add_argument(
        "-m", "--method", type=str, default="hybrid",
        choices=["seam_carving", "smart_crop", "hybrid", "jpeg"],
        help="Compression method"
    )
    parser.add_argument(
        "-e", "--energy", type=str, default="hybrid",
        choices=["gradient", "deep", "saliency", "hybrid"],
        help="Energy function for content importance"
    )
    parser.add_argument(
        "--model", type=str, default="vgg16",
        choices=["vgg16", "vgg19", "resnet18", "resnet50", "efficientnet_b0"],
        help="Deep model for energy computation"
    )
    parser.add_argument(
        "--visualize-energy", action="store_true",
        help="Visualize energy map"
    )
    parser.add_argument(
        "--visualize-seams", type=int, metavar="N",
        help="Visualize N seams that would be removed"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="Output JPEG quality"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze content and print statistics"
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Setup energy function
    if args.energy == "gradient":
        energy_fn = GradientEnergyFunction()
    elif args.energy == "deep":
        energy_fn = DeepEnergyFunction(model_name=args.model)
    elif args.energy == "saliency":
        energy_fn = SaliencyEnergyFunction()
    else:  # hybrid
        energy_fn = HybridEnergyFunction()

    # Handle visualization modes
    if args.visualize_energy or args.visualize_seams:
        from semantic_compress.seam_carving import SeamCarver

        img = Image.open(args.input).convert("RGB")
        img_array = img.numpy() if hasattr(img, 'numpy') else img

        carver = SeamCarver(energy_function=energy_fn)

        if args.visualize_energy:
            vis = carver.visualize_energy(img)
            output = args.output or f"{input_path.stem}_energy.png"
            Image.fromarray(vis).save(output)
            print(f"Energy map saved to: {output}")

        if args.visualize_seams:
            vis = carver.visualize_seams(img, n_seams=args.visualize_seams)
            output = args.output or f"{input_path.stem}_seams.png"
            Image.fromarray(vis).save(output)
            print(f"Seam visualization saved to: {output}")

        return

    # Handle analysis mode
    if args.analyze:
        compressor = SemanticCompressor(energy_function=energy_fn)
        stats = compressor.analyze_content(args.input)
        print(f"Content Analysis for: {args.input}")
        print(f"  Image size: {stats['size']}")
        print(f"  Mean energy: {stats['mean_energy']:.3f}")
        print(f"  Max energy: {stats['max_energy']:.3f}")
        print(f"  Energy std: {stats['energy_std']:.3f}")
        print(f"  High importance pixels: {stats['high_importance_ratio']:.1%}")
        print(f"  Low importance pixels: {stats['low_importance_ratio']:.1%}")
        return

    # Compression mode
    if args.scale is None:
        print("Error: Please specify --scale or use --visualize-energy/--analyze", file=sys.stderr)
        sys.exit(1)

    compressor = SemanticCompressor(energy_function=energy_fn)
    result = compressor.compress(
        args.input,
        scale=args.scale,
        method=args.method,
    )

    # Save result
    output_path = args.output or f"{input_path.stem}_compressed.jpg"
    result.save(output_path, quality=args.quality)

    print(f"Compressed: {args.input}")
    print(f"  Original: {result.original_size}")
    print(f"  Compressed: {result.compressed_size}")
    print(f"  Ratio: {result.compression_ratio:.2%}")
    print(f"  Method: {result.method}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
