"""Seam carving implementation for content-aware image resizing.

Based on "Seam Carving for Content-Aware Image Resizing" by Avidan & Shamir (2007).
Extended with deep energy functions for semantic-aware resizing.
"""

from typing import Callable, Literal, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from semantic_compress.energy import EnergyFunction, GradientEnergyFunction


class SeamCarver:
    """Content-aware image resizing using seam carving.

    Seam carving removes or inserts seams (connected paths of pixels)
    with the lowest energy, preserving important content while resizing.
    """

    def __init__(
        self,
        energy_function: Optional[EnergyFunction] = None,
        keep_mask: Optional[np.ndarray] = None,
        remove_mask: Optional[np.ndarray] = None,
    ):
        """Initialize seam carver.

        Args:
            energy_function: Energy function for pixel importance (default: gradient)
            keep_mask: Binary mask of pixels to preserve (1 = must keep)
            remove_mask: Binary mask of pixels to remove (1 = should remove first)
        """
        self.energy_function = energy_function or GradientEnergyFunction()
        self.keep_mask = keep_mask
        self.remove_mask = remove_mask

    def resize(
        self,
        image: Union[np.ndarray, Image.Image],
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        scale_height: Optional[float] = None,
        scale_width: Optional[float] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Resize image using seam carving.

        Args:
            image: Input image
            target_height: Target height in pixels
            target_width: Target width in pixels
            scale_height: Scale factor for height (alternative to target_height)
            scale_width: Scale factor for width (alternative to target_width)
            show_progress: Show progress bar

        Returns:
            Resized image as numpy array
        """
        img = self._to_numpy(image)
        h, w = img.shape[:2]

        # Calculate target dimensions
        if target_height is None and scale_height is not None:
            target_height = int(h * scale_height)
        if target_width is None and scale_width is not None:
            target_width = int(w * scale_width)

        target_height = target_height or h
        target_width = target_width or w

        # Determine number of seams to remove/add
        delta_h = h - target_height
        delta_w = w - target_width

        result = img.copy()

        # Remove horizontal seams (reduce height)
        if delta_h > 0:
            result = self._remove_horizontal_seams(result, delta_h, show_progress)
        elif delta_h < 0:
            result = self._add_horizontal_seams(result, -delta_h, show_progress)

        # Remove vertical seams (reduce width)
        if delta_w > 0:
            result = self._remove_vertical_seams(result, delta_w, show_progress)
        elif delta_w < 0:
            result = self._add_vertical_seams(result, -delta_w, show_progress)

        return result

    def _to_numpy(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to numpy array."""
        if isinstance(image, Image.Image):
            return np.array(image)
        return image.astype(np.float32)

    def _compute_energy(self, image: np.ndarray) -> np.ndarray:
        """Compute energy map with mask constraints."""
        energy = self.energy_function.compute(image)

        # Apply keep mask (infinite energy for pixels to preserve)
        if self.keep_mask is not None:
            energy[self.keep_mask > 0] = 1e10

        # Apply remove mask (negative energy for pixels to remove)
        if self.remove_mask is not None:
            energy[self.remove_mask > 0] = -1e10

        return energy

    def _find_vertical_seam(self, energy: np.ndarray) -> np.ndarray:
        """Find optimal vertical seam using dynamic programming.

        Returns array of column indices for each row.
        """
        h, w = energy.shape

        # DP table: cumulative minimum energy
        dp = energy.copy()
        backtrack = np.zeros_like(dp, dtype=np.int32)

        # Fill DP table from top to bottom
        for i in range(1, h):
            for j in range(w):
                # Check neighbors in previous row
                min_idx = j
                min_val = dp[i - 1, j]

                if j > 0 and dp[i - 1, j - 1] < min_val:
                    min_idx = j - 1
                    min_val = dp[i - 1, j - 1]
                if j < w - 1 and dp[i - 1, j + 1] < min_val:
                    min_idx = j + 1

                dp[i, j] += min_val
                backtrack[i, j] = min_idx

        # Find minimum in last row
        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1])

        # Backtrack to find full seam
        for i in range(h - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]

        return seam

    def _find_horizontal_seam(self, energy: np.ndarray) -> np.ndarray:
        """Find optimal horizontal seam (transpose and use vertical seam)."""
        return self._find_vertical_seam(energy.T)

    def _remove_vertical_seam(
        self, image: np.ndarray, seam: np.ndarray
    ) -> np.ndarray:
        """Remove vertical seam from image."""
        h, w = image.shape[:2]
        new_w = w - 1

        if len(image.shape) == 3:
            result = np.zeros((h, new_w, image.shape[2]), dtype=image.dtype)
        else:
            result = np.zeros((h, new_w), dtype=image.dtype)

        for i in range(h):
            col = seam[i]
            if len(image.shape) == 3:
                result[i, :col] = image[i, :col]
                result[i, col:] = image[i, col + 1 :]
            else:
                result[i, :col] = image[i, :col]
                result[i, col:] = image[i, col + 1 :]

        # Update masks if present
        if self.keep_mask is not None:
            self.keep_mask = self._remove_vertical_seam(self.keep_mask, seam)
        if self.remove_mask is not None:
            self.remove_mask = self._remove_vertical_seam(self.remove_mask, seam)

        return result

    def _remove_horizontal_seam(
        self, image: np.ndarray, seam: np.ndarray
    ) -> np.ndarray:
        """Remove horizontal seam from image."""
        # Transpose, remove vertical seam, transpose back
        image_t = image.transpose(1, 0, 2) if len(image.shape) == 3 else image.T
        result_t = self._remove_vertical_seam(image_t, seam)
        return result_t.transpose(1, 0, 2) if len(image.shape) == 3 else result_t.T

    def _add_vertical_seam(
        self, image: np.ndarray, seam: np.ndarray
    ) -> np.ndarray:
        """Duplicate vertical seam to expand image."""
        h, w = image.shape[:2]
        new_w = w + 1

        if len(image.shape) == 3:
            result = np.zeros((h, new_w, image.shape[2]), dtype=image.dtype)
        else:
            result = np.zeros((h, new_w), dtype=image.dtype)

        for i in range(h):
            col = seam[i]
            if len(image.shape) == 3:
                result[i, :col] = image[i, :col]
                # Average of neighbors for new pixel
                if col < w:
                    result[i, col] = (image[i, col] + image[i, col - 1]) / 2
                else:
                    result[i, col] = image[i, col - 1]
                result[i, col + 1 :] = image[i, col:]
            else:
                result[i, :col] = image[i, :col]
                if col < w:
                    result[i, col] = (image[i, col] + image[i, col - 1]) / 2
                else:
                    result[i, col] = image[i, col - 1]
                result[i, col + 1 :] = image[i, col:]

        return result

    def _add_horizontal_seam(
        self, image: np.ndarray, seam: np.ndarray
    ) -> np.ndarray:
        """Duplicate horizontal seam to expand image."""
        image_t = image.transpose(1, 0, 2) if len(image.shape) == 3 else image.T
        result_t = self._add_vertical_seam(image_t, seam)
        return result_t.transpose(1, 0, 2) if len(image.shape) == 3 else result_t.T

    def _remove_vertical_seams(
        self, image: np.ndarray, n: int, show_progress: bool = True
    ) -> np.ndarray:
        """Remove n vertical seams."""
        result = image.copy()
        iterator = tqdm(range(n), desc="Removing vertical seams") if show_progress else range(n)

        for _ in iterator:
            energy = self._compute_energy(result)
            seam = self._find_vertical_seam(energy)
            result = self._remove_vertical_seam(result, seam)

        return result

    def _remove_horizontal_seams(
        self, image: np.ndarray, n: int, show_progress: bool = True
    ) -> np.ndarray:
        """Remove n horizontal seams."""
        result = image.copy()
        iterator = tqdm(range(n), desc="Removing horizontal seams") if show_progress else range(n)

        for _ in iterator:
            energy = self._compute_energy(result)
            seam = self._find_horizontal_seam(energy)
            result = self._remove_horizontal_seam(result, seam)

        return result

    def _add_vertical_seams(
        self, image: np.ndarray, n: int, show_progress: bool = True
    ) -> np.ndarray:
        """Add n vertical seams."""
        result = image.copy()
        seams_to_add = []

        # First find all seams
        iterator = tqdm(range(n), desc="Finding seams to add") if show_progress else range(n)
        temp_image = image.copy()
        for _ in iterator:
            energy = self._compute_energy(temp_image)
            seam = self._find_vertical_seam(energy)
            seams_to_add.append(seam)
            temp_image = self._remove_vertical_seam(temp_image, seam)

        # Add seams in reverse order (lowest energy first)
        iterator = (
            tqdm(reversed(seams_to_add), desc="Adding vertical seams")
            if show_progress
            else reversed(seams_to_add)
        )
        for seam in iterator:
            result = self._add_vertical_seam(result, seam)

        return result

    def _add_horizontal_seams(
        self, image: np.ndarray, n: int, show_progress: bool = True
    ) -> np.ndarray:
        """Add n horizontal seams."""
        result = image.copy()
        seams_to_add = []

        iterator = tqdm(range(n), desc="Finding seams to add") if show_progress else range(n)
        temp_image = image.copy()
        for _ in iterator:
            energy = self._compute_energy(temp_image)
            seam = self._find_horizontal_seam(energy)
            seams_to_add.append(seam)
            temp_image = self._remove_horizontal_seam(temp_image, seam)

        iterator = (
            tqdm(reversed(seams_to_add), desc="Adding horizontal seams")
            if show_progress
            else reversed(seams_to_add)
        )
        for seam in iterator:
            result = self._add_horizontal_seam(result, seam)

        return result

    def visualize_energy(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Visualize energy map for debugging."""
        img = self._to_numpy(image)
        energy = self._compute_energy(img)

        # Colorize energy map
        from matplotlib import colormaps

        cmap = colormaps.get_cmap("hot")
        colored = cmap(energy)[:, :, :3]

        return (colored * 255).astype(np.uint8)

    def visualize_seams(
        self,
        image: Union[np.ndarray, Image.Image],
        n_seams: int = 10,
        direction: Literal["vertical", "horizontal"] = "vertical",
    ) -> np.ndarray:
        """Visualize which seams would be removed.

        Returns image with seams highlighted in red.
        """
        img = self._to_numpy(image).copy()
        result = img.copy()

        temp_image = img.copy()
        temp_masks = (self.keep_mask, self.remove_mask)

        for _ in range(n_seams):
            energy = self._compute_energy(temp_image)

            if direction == "vertical":
                seam = self._find_vertical_seam(energy)
                # Mark seam in red
                for i, j in enumerate(seam):
                    result[i, j] = [255, 0, 0]
                temp_image = self._remove_vertical_seam(temp_image, seam)
            else:
                seam = self._find_horizontal_seam(energy)
                for j, i in enumerate(seam):
                    result[i, j] = [255, 0, 0]
                temp_image = self._remove_horizontal_seam(temp_image, seam)

        # Restore masks
        self.keep_mask, self.remove_mask = temp_masks

        return result.astype(np.uint8)
