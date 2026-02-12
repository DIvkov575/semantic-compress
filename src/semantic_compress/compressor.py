"""Main semantic image compression interface."""

import io
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
from PIL import Image

from semantic_compress.energy import (
    DeepEnergyFunction,
    EnergyFunction,
    GradientEnergyFunction,
    HybridEnergyFunction,
    SaliencyEnergyFunction,
)
from semantic_compress.seam_carving import SeamCarver


class CompressionResult:
    """Result of compression operation."""

    def __init__(
        self,
        image: np.ndarray,
        original_size: tuple[int, int],
        compressed_size: tuple[int, int],
        method: str,
        metadata: Optional[dict] = None,
    ):
        self.image = image
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.method = method
        self.metadata = metadata or {}

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (pixels out / pixels in)."""
        orig_pixels = self.original_size[0] * self.original_size[1]
        new_pixels = self.compressed_size[0] * self.compressed_size[1]
        return new_pixels / orig_pixels

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image."""
        if self.image.max() <= 1.0:
            arr = (self.image * 255).astype(np.uint8)
        else:
            arr = self.image.astype(np.uint8)
        return Image.fromarray(arr)

    def save(self, path: Union[str, Path], quality: int = 95) -> None:
        """Save compressed image."""
        img = self.to_pil()
        img.save(path, quality=quality, optimize=True)


class SemanticCompressor:
    """High-level interface for semantic-aware image compression.

    Combines multiple compression techniques:
    1. Content-aware resizing (seam carving with deep energy)
    2. Smart cropping (center + face/object detection)
    3. Quality adjustment based on content importance
    """

    def __init__(
        self,
        energy_function: Optional[EnergyFunction] = None,
        quality_metric: str = "perceptual",
    ):
        """Initialize semantic compressor.

        Args:
            energy_function: Energy function for content importance
            quality_metric: Quality metric to optimize for ('perceptual', 'semantic', 'psnr')
        """
        self.energy_function = energy_function
        self.quality_metric = quality_metric
        self.seam_carver = SeamCarver(energy_function=energy_function)

    def compress(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        target_bytes: Optional[int] = None,
        target_pixels: Optional[int] = None,
        max_dimension: Optional[int] = None,
        scale: Optional[float] = None,
        method: Literal[
            "seam_carving", "smart_crop", "hybrid", "jpeg"
        ] = "hybrid",
        preserve_aspect: bool = True,
        show_progress: bool = True,
    ) -> CompressionResult:
        """Compress image using semantic-aware methods.

        Args:
            image: Input image (path, array, or PIL Image)
            target_bytes: Target file size in bytes
            target_pixels: Target total pixel count
            max_dimension: Maximum width or height
            scale: Scale factor (0.5 = half size)
            method: Compression method to use
            preserve_aspect: Whether to preserve aspect ratio
            show_progress: Show progress indicators

        Returns:
            CompressionResult with compressed image and metadata
        """
        # Load image
        img = self._load_image(image)
        original_size = (img.height, img.width)
        img_array = np.array(img).astype(np.float32)

        # Determine target dimensions
        target_h, target_w = self._calculate_target_size(
            original_size[0],
            original_size[1],
            target_bytes=target_bytes,
            target_pixels=target_pixels,
            max_dimension=max_dimension,
            scale=scale,
        )

        # Apply compression method
        if method == "seam_carving":
            result = self._compress_seam_carving(
                img_array, target_h, target_w, show_progress
            )
        elif method == "smart_crop":
            result = self._compress_smart_crop(img_array, target_h, target_w)
        elif method == "hybrid":
            result = self._compress_hybrid(
                img_array, target_h, target_w, show_progress
            )
        elif method == "jpeg":
            result = self._compress_jpeg_quality(img_array, target_bytes)
        else:
            raise ValueError(f"Unknown compression method: {method}")

        return CompressionResult(
            image=result,
            original_size=original_size,
            compressed_size=(result.shape[0], result.shape[1]),
            method=method,
            metadata={
                "energy_function": type(self.energy_function).__name__
                if self.energy_function
                else "default",
                "preserve_aspect": preserve_aspect,
            },
        )

    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _calculate_target_size(
        self,
        height: int,
        width: int,
        target_bytes: Optional[int] = None,
        target_pixels: Optional[int] = None,
        max_dimension: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> tuple[int, int]:
        """Calculate target dimensions based on constraints."""
        if scale is not None:
            return (int(height * scale), int(width * scale))

        if target_pixels is not None:
            ratio = np.sqrt(target_pixels / (height * width))
            return (int(height * ratio), int(width * ratio))

        if target_bytes is not None:
            # Rough estimate: 3 bytes per pixel for uncompressed
            # JPEG compression ratio varies, estimate ~10:1
            estimated_pixels = target_bytes / 0.3
            ratio = np.sqrt(estimated_pixels / (height * width))
            return (int(height * ratio), int(width * ratio))

        if max_dimension is not None:
            max_dim = max(height, width)
            if max_dim > max_dimension:
                scale = max_dimension / max_dim
                return (int(height * scale), int(width * scale))
            return (height, width)

        # Default: no change
        return (height, width)

    def _compress_seam_carving(
        self,
        image: np.ndarray,
        target_h: int,
        target_w: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Compress using seam carving."""
        carver = SeamCarver(energy_function=self.energy_function)
        return carver.resize(
            image,
            target_height=target_h,
            target_width=target_w,
            show_progress=show_progress,
        )

    def _compress_smart_crop(
        self,
        image: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Compress using smart cropping (saliency-aware)."""
        h, w = image.shape[:2]

        # If target is smaller, crop to important region
        if target_h < h or target_w < w:
            # Compute energy map
            if self.energy_function is None:
                energy_func = HybridEnergyFunction()
            else:
                energy_func = self.energy_function

            energy = energy_func.compute(image)

            # Find crop window with maximum energy
            best_score = -np.inf
            best_crop = (0, 0, w, h)

            # Try different crop positions
            for y in range(0, h - target_h + 1, max(1, (h - target_h) // 10)):
                for x in range(0, w - target_w + 1, max(1, (w - target_w) // 10)):
                    crop_energy = energy[y : y + target_h, x : x + target_w]
                    score = crop_energy.sum()
                    if score > best_score:
                        best_score = score
                        best_crop = (x, y, target_w, target_h)

            x, y, cw, ch = best_crop
            result = image[y : y + ch, x : x + cw]
        else:
            # Target is larger, just resize normally
            from PIL import Image

            pil_img = Image.fromarray(image.astype(np.uint8))
            result = np.array(pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS))

        return result

    def _compress_hybrid(
        self,
        image: np.ndarray,
        target_h: int,
        target_w: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Hybrid compression: smart crop + seam carving."""
        h, w = image.shape[:2]

        # First, do a rough crop if reduction is significant
        if target_h < h * 0.7 or target_w < w * 0.7:
            # Intermediate size: 20% larger than target
            inter_h = int(target_h * 1.2)
            inter_w = int(target_w * 1.2)
            image = self._compress_smart_crop(image, inter_h, inter_w)

        # Then apply seam carving for fine adjustment
        return self._compress_seam_carving(image, target_h, target_w, show_progress)

    def _compress_jpeg_quality(
        self,
        image: np.ndarray,
        target_bytes: Optional[int],
    ) -> np.ndarray:
        """Compress by adjusting JPEG quality."""
        if target_bytes is None:
            return image

        pil_img = Image.fromarray(image.astype(np.uint8))

        # Binary search for optimal quality
        low, high = 10, 95
        best_result = None
        best_diff = float("inf")

        while low <= high:
            mid = (low + high) // 2
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=mid, optimize=True)
            size = buffer.tell()

            diff = abs(size - target_bytes)
            if diff < best_diff:
                best_diff = diff
                best_result = np.array(Image.open(buffer))

            if size > target_bytes:
                high = mid - 1
            else:
                low = mid + 1

        return best_result if best_result is not None else image

    def compress_batch(
        self,
        images: list[Union[str, Path, np.ndarray, Image.Image]],
        **kwargs,
    ) -> list[CompressionResult]:
        """Compress multiple images."""
        return [self.compress(img, **kwargs) for img in images]

    def analyze_content(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> dict:
        """Analyze image content and return importance metrics."""
        img = self._load_image(image)
        img_array = np.array(img).astype(np.float32)

        # Use hybrid energy for comprehensive analysis
        if self.energy_function is None:
            energy_func = HybridEnergyFunction()
        else:
            energy_func = self.energy_function

        energy = energy_func.compute(img_array)

        return {
            "mean_energy": float(energy.mean()),
            "max_energy": float(energy.max()),
            "energy_std": float(energy.std()),
            "high_importance_ratio": float((energy > 0.7).sum() / energy.size),
            "low_importance_ratio": float((energy < 0.3).sum() / energy.size),
            "size": img_array.shape[:2],
        }


def compress_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    **kwargs,
) -> CompressionResult:
    """Convenience function for one-off compression.

    Example:
        >>> result = compress_image("photo.jpg", scale=0.5, method="hybrid")
        >>> result.save("compressed.jpg")
    """
    compressor = SemanticCompressor()
    return compressor.compress(image, **kwargs)
