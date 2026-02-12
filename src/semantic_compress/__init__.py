"""Semantic Compress: Deep energy-based image compression library."""

from semantic_compress.compressor import SemanticCompressor
from semantic_compress.energy import (
    DeepEnergyFunction,
    GradientEnergyFunction,
    HybridEnergyFunction,
    SaliencyEnergyFunction,
)
from semantic_compress.seam_carving import SeamCarver

__version__ = "0.1.0"
__all__ = [
    "SemanticCompressor",
    "DeepEnergyFunction",
    "GradientEnergyFunction",
    "HybridEnergyFunction",
    "SaliencyEnergyFunction",
    "SeamCarver",
]
