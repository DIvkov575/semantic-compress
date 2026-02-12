"""Tests for energy functions."""

import numpy as np
import pytest

from semantic_compress.energy import (
    DeepEnergyFunction,
    GradientEnergyFunction,
    HybridEnergyFunction,
    SaliencyEnergyFunction,
)


class TestGradientEnergy:
    """Tests for gradient-based energy function."""

    def test_computes_energy_map(self):
        """Test that gradient energy produces a valid energy map."""
        # Create test image with edges
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[40:60, :] = 1.0  # Horizontal edge

        energy_fn = GradientEnergyFunction()
        energy = energy_fn.compute(img)

        assert energy.shape == (100, 100)
        assert energy.min() >= 0
        assert energy.max() <= 1

    def test_detects_edges(self):
        """Test that edges have higher energy than flat regions."""
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[50:51, :] = 1.0  # Sharp edge

        energy_fn = GradientEnergyFunction()
        energy = energy_fn.compute(img)

        # Edge should have high energy
        edge_energy = energy[50, 50]
        flat_energy = energy[10, 10]
        assert edge_energy > flat_energy


class TestSaliencyEnergy:
    """Tests for saliency-based energy function."""

    def test_spectral_saliency(self):
        """Test spectral saliency computation."""
        img = np.random.rand(100, 100, 3).astype(np.float32)

        energy_fn = SaliencyEnergyFunction(method="spectral")
        energy = energy_fn.compute(img)

        assert energy.shape == (100, 100)
        assert energy.min() >= 0
        assert energy.max() <= 1


class TestDeepEnergy:
    """Tests for deep learning-based energy function."""

    def test_vgg16_energy(self):
        """Test VGG16-based energy computation."""
        pytest.importorskip("torch")

        img = np.random.rand(224, 224, 3).astype(np.float32)

        energy_fn = DeepEnergyFunction(model_name="vgg16")
        energy = energy_fn.compute(img)

        assert energy.shape == (224, 224)
        assert energy.min() >= 0
        assert energy.max() <= 1

    def test_resnet_energy(self):
        """Test ResNet-based energy computation."""
        pytest.importorskip("torch")

        img = np.random.rand(224, 224, 3).astype(np.float32)

        energy_fn = DeepEnergyFunction(model_name="resnet18")
        energy = energy_fn.compute(img)

        assert energy.shape == (224, 224)


class TestHybridEnergy:
    """Tests for hybrid energy function."""

    def test_combines_energies(self):
        """Test that hybrid energy combines multiple functions."""
        img = np.random.rand(100, 100, 3).astype(np.float32)

        energies = [
            GradientEnergyFunction(),
            SaliencyEnergyFunction(),
        ]
        weights = [0.5, 0.5]

        energy_fn = HybridEnergyFunction(energies=energies, weights=weights)
        energy = energy_fn.compute(img)

        assert energy.shape == (100, 100)
        assert energy.min() >= 0
        assert energy.max() <= 1
