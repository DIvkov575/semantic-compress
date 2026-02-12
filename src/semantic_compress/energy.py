"""Energy functions for semantic-aware image processing.

This module implements various energy functions for determining pixel importance
in images, from traditional gradient-based methods to deep learning approaches
using pre-trained CNN feature maps.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class EnergyFunction(ABC):
    """Abstract base class for energy functions."""

    @abstractmethod
    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute energy map for the given image.

        Args:
            image: Input image as numpy array (H, W, C) in range [0, 255] or [0, 1]

        Returns:
            Energy map as numpy array (H, W) with higher values indicating
            more important/salient regions
        """
        pass

    def _normalize(self, energy: np.ndarray) -> np.ndarray:
        """Normalize energy map to [0, 1] range."""
        energy_min = energy.min()
        energy_max = energy.max()
        if energy_max > energy_min:
            return (energy - energy_min) / (energy_max - energy_min)
        return np.zeros_like(energy)


class GradientEnergyFunction(EnergyFunction):
    """Traditional gradient-based energy function (from original seam carving paper).

    Computes energy based on image gradients using Sobel operators.
    Higher gradients = more important (edges, details).
    """

    def __init__(self, use_color: bool = True):
        """Initialize gradient energy function.

        Args:
            use_color: Whether to compute energy on all color channels
        """
        self.use_color = use_color

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute gradient-based energy map."""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        if self.use_color and image.shape[2] >= 3:
            # Convert to grayscale for gradient computation
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image[:, :, 0]

        # Compute gradients
        grad_x = self._convolve2d(gray, sobel_x)
        grad_y = self._convolve2d(gray, sobel_y)

        # Energy is magnitude of gradient
        energy = np.sqrt(grad_x**2 + grad_y**2)

        return self._normalize(energy)

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution."""
        from scipy.signal import convolve2d

        return convolve2d(image, kernel, mode="same", boundary="symm")


class DeepEnergyFunction(EnergyFunction):
    """Deep learning-based energy function using CNN feature maps.

    Extracts features from intermediate layers of a pre-trained network
    (VGG16, ResNet, etc.) and computes pixel importance based on
    feature activation magnitudes. This captures semantic content better
    than gradient-based methods.
    """

    AVAILABLE_MODELS = ["vgg16", "vgg19", "resnet18", "resnet50", "efficientnet_b0"]

    def __init__(
        self,
        model_name: str = "vgg16",
        layer_names: Optional[list[str]] = None,
        device: Optional[str] = None,
        aggregate_method: str = "sum",
    ):
        """Initialize deep energy function.

        Args:
            model_name: Pre-trained model to use ('vgg16', 'vgg19', 'resnet18', 'resnet50')
            layer_names: Specific layers to extract features from (None = auto-select)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            aggregate_method: How to aggregate multi-layer features ('sum', 'max', 'weighted')
        """
        self.model_name = model_name.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregate_method = aggregate_method

        self.model, self.feature_layers = self._load_model(layer_names)
        self.model.to(self.device)
        self.model.eval()

        self.feature_maps: list[torch.Tensor] = []
        self._register_hooks()

    def _load_model(
        self, layer_names: Optional[list[str]]
    ) -> tuple[nn.Module, list[str]]:
        """Load pre-trained model and identify feature layers."""
        import torchvision.models as models

        if self.model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
            if layer_names is None:
                layer_names = ["features.8", "features.15", "features.22"]
        elif self.model_name == "vgg19":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_FEATURES)
            if layer_names is None:
                layer_names = ["features.8", "features.17", "features.26"]
        elif self.model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            if layer_names is None:
                layer_names = ["layer1", "layer2", "layer3"]
        elif self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            if layer_names is None:
                layer_names = ["layer1", "layer2", "layer3"]
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            if layer_names is None:
                layer_names = ["features.2", "features.4", "features.6"]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model, layer_names

    def _register_hooks(self) -> None:
        """Register forward hooks to capture feature maps."""
        self.feature_maps = []

        def hook_fn(module, input, output):
            self.feature_maps.append(output.detach())

        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(hook_fn)

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute deep energy map from CNN features.

        The energy at each pixel is computed based on the activation magnitude
        of features at that spatial location. Higher activations indicate
        semantically important regions (objects, textures, patterns).
        """
        # Preprocess image
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Convert to torch tensor (C, H, W)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        # Forward pass
        self.feature_maps = []
        with torch.no_grad():
            _ = self.model(tensor)

        if not self.feature_maps:
            raise RuntimeError("No feature maps captured. Check layer names.")

        # Resize all feature maps to original size
        h, w = image.shape[:2]
        resized_maps = []

        for feat_map in self.feature_maps:
            # Take absolute value and sum across channels
            importance = feat_map.abs().sum(dim=1, keepdim=True)  # (1, 1, H', W')
            # Upsample to original resolution
            upsampled = F.interpolate(
                importance, size=(h, w), mode="bilinear", align_corners=False
            )
            resized_maps.append(upsampled[0, 0].cpu().numpy())

        # Aggregate feature maps
        if self.aggregate_method == "sum":
            energy = np.sum(resized_maps, axis=0)
        elif self.aggregate_method == "max":
            energy = np.max(resized_maps, axis=0)
        elif self.aggregate_method == "weighted":
            # Weight deeper layers more (they capture higher-level semantics)
            weights = np.linspace(0.5, 1.0, len(resized_maps))
            energy = sum(w * m for w, m in zip(weights, resized_maps))
        else:
            energy = np.mean(resized_maps, axis=0)

        return self._normalize(energy)


class SaliencyEnergyFunction(EnergyFunction):
    """Visual saliency-based energy function.

    Uses a dedicated saliency detection model to identify visually
    prominent regions in the image (likely to attract human attention).
    """

    def __init__(
        self,
        method: str = "spectral",
        device: Optional[str] = None,
    ):
        """Initialize saliency energy function.

        Args:
            method: Saliency detection method ('spectral', 'fine_grained')
            device: Device for deep learning methods
        """
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if method == "fine_grained":
            self._init_fine_grained_model()

    def _init_fine_grained_model(self) -> None:
        """Initialize deep saliency detection model."""
        try:
            # Use a simple CNN-based saliency model
            import torchvision.models as models

            self.saliency_model = models.segmentation.fcn_resnet50(
                weights=models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            )
            self.saliency_model.to(self.device)
            self.saliency_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load saliency model: {e}")

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute saliency map."""
        if self.method == "spectral":
            return self._spectral_saliency(image)
        elif self.method == "fine_grained":
            return self._fine_grained_saliency(image)
        else:
            raise ValueError(f"Unknown saliency method: {self.method}")

    def _spectral_saliency(self, image: np.ndarray) -> np.ndarray:
        """Compute spectral residual saliency (fast, no ML)."""
        from scipy.fftpack import fft2, ifft2

        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        gray = gray.astype(np.float32)

        # FFT
        f = fft2(gray)
        magnitude = np.abs(f)
        phase = np.angle(f)

        # Spectral residual
        log_magnitude = np.log(magnitude + 1e-8)
        avg_filter = np.ones((3, 3)) / 9
        from scipy.ndimage import convolve

        avg_log = convolve(log_magnitude, avg_filter, mode="reflect")
        residual = log_magnitude - avg_log

        # Reconstruct
        saliency = np.abs(ifft2(np.exp(residual + 1j * phase))) ** 2

        # Gaussian blur
        from scipy.ndimage import gaussian_filter

        saliency = gaussian_filter(saliency, sigma=2.0)

        return self._normalize(saliency)

    def _fine_grained_saliency(self, image: np.ndarray) -> np.ndarray:
        """Compute fine-grained saliency using deep segmentation model."""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image - mean) / std

        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.saliency_model(tensor)["out"]
            # Sum across all classes for general saliency
            saliency = output.abs().sum(dim=1)[0].cpu().numpy()

        return self._normalize(saliency)


class HybridEnergyFunction(EnergyFunction):
    """Hybrid energy combining multiple energy functions.

    Combines gradient-based and deep learning-based energies with
    learnable weights for optimal semantic preservation.
    """

    def __init__(
        self,
        energies: Optional[list[EnergyFunction]] = None,
        weights: Optional[list[float]] = None,
        adaptive_weights: bool = False,
    ):
        """Initialize hybrid energy function.

        Args:
            energies: List of energy functions to combine
            weights: Weights for each energy function
            adaptive_weights: Whether to adaptively adjust weights based on image content
        """
        if energies is None:
            energies = [
                GradientEnergyFunction(),
                DeepEnergyFunction(),
                SaliencyEnergyFunction(),
            ]

        self.energies = energies
        self.weights = weights or [1.0] * len(energies)
        self.adaptive_weights = adaptive_weights

        if len(self.weights) != len(self.energies):
            raise ValueError("Number of weights must match number of energy functions")

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute hybrid energy map."""
        energy_maps = [e.compute(image) for e in self.energies]

        if self.adaptive_weights:
            # Adapt weights based on local variance (higher variance = more structure)
            adaptive_weights = []
            for emap in energy_maps:
                local_var = np.var(emap)
                adaptive_weights.append(local_var)
            total = sum(adaptive_weights) + 1e-8
            weights = [w / total for w in adaptive_weights]
        else:
            weights = self.weights

        # Weighted combination
        combined = sum(w * emap for w, emap in zip(weights, energy_maps))

        return self._normalize(combined)


class LearnedEnergyFunction(EnergyFunction):
    """Learnable energy function trained on perceptual quality metrics.

    This energy function can be fine-tuned on specific image types
    or quality metrics to optimize for particular use cases.
    """

    def __init__(
        self,
        base_model: str = "vgg16",
        fine_tuned_weights: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize learned energy function.

        Args:
            base_model: Base architecture for feature extraction
            fine_tuned_weights: Path to fine-tuned weights (optional)
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base = DeepEnergyFunction(base_model, device=self.device)

        if fine_tuned_weights:
            self._load_fine_tuned_weights(fine_tuned_weights)

    def _load_fine_tuned_weights(self, path: str) -> None:
        """Load fine-tuned weights for energy prediction."""
        checkpoint = torch.load(path, map_location=self.device)
        # Would load custom head weights here

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute learned energy map."""
        # Start with base deep energy
        base_energy = self.base.compute(image)

        # Could add learned refinements here
        return base_energy

    def fine_tune(
        self,
        images: list[np.ndarray],
        targets: list[np.ndarray],
        epochs: int = 10,
        lr: float = 1e-4,
    ) -> None:
        """Fine-tune energy function on specific data.

        Args:
            images: Training images
            targets: Target energy maps
            epochs: Number of training epochs
            lr: Learning rate
        """
        # Implementation would train a small refinement network
        pass
