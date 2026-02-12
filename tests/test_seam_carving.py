"""Tests for seam carving."""

import numpy as np
import pytest

from semantic_compress.seam_carving import SeamCarver


class TestSeamCarver:
    """Tests for seam carving implementation."""

    def test_vertical_seam_removal(self):
        """Test removing a vertical seam reduces width by 1."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        carver = SeamCarver()

        result = carver.resize(img, target_width=99, show_progress=False)

        assert result.shape == (100, 99, 3)

    def test_horizontal_seam_removal(self):
        """Test removing a horizontal seam reduces height by 1."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        carver = SeamCarver()

        result = carver.resize(img, target_height=99, show_progress=False)

        assert result.shape == (99, 100, 3)

    def test_scale_resize(self):
        """Test resizing by scale factor."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        carver = SeamCarver()

        result = carver.resize(img, scale_width=0.5, scale_height=0.5, show_progress=False)

        assert result.shape == (50, 50, 3)

    def test_preserves_content(self):
        """Test that high-energy content is preserved."""
        # Create image with a bright square (high energy)
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[30:70, 30:70] = 1.0  # Bright square in center

        carver = SeamCarver()
        result = carver.resize(img, target_width=50, show_progress=False)

        # The bright region should still exist
        assert result[:, :, 0].max() > 0.8

    def test_grayscale_image(self):
        """Test seam carving on grayscale images."""
        img = np.random.rand(100, 100).astype(np.float32)
        carver = SeamCarver()

        result = carver.resize(img, target_width=90, show_progress=False)

        assert result.shape == (100, 90)

    def test_seam_addition(self):
        """Test adding seams to expand image."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        carver = SeamCarver()

        result = carver.resize(img, target_width=110, show_progress=False)

        assert result.shape == (100, 110, 3)
