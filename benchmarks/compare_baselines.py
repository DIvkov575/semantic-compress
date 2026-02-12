#!/usr/bin/env python3
"""
Compare semantic-compress against baseline methods.

Baseline methods:
- Naive bilinear resizing
- JPEG compression
- Bicubic downsampling
- Lanczos resizing
"""

import time
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import io

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_compress import SemanticCompressor, DeepEnergyFunction, HybridEnergyFunction
from run_benchmarks import MetricsComputer, BenchmarkRunner


@dataclass
class ComparisonResult:
    """Result from comparing methods."""
    method_name: str
    compression_ratio: float
    ssim: float
    psnr: float
    lpips: float
    feature_similarity: float
    processing_time_ms: float
    file_size_bytes: int
    bpp: float  # Bits per pixel


class BaselineCompressor:
    """Baseline compression methods for comparison."""
    
    @staticmethod
    def resize_bilinear(image: np.ndarray, ratio: float) -> np.ndarray:
        """Simple bilinear resize."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * np.sqrt(ratio)), int(w * np.sqrt(ratio))
        pil_img = Image.fromarray(image.astype(np.uint8))
        resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        return np.array(resized)
    
    @staticmethod
    def resize_bicubic(image: np.ndarray, ratio: float) -> np.ndarray:
        """Bicubic resize."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * np.sqrt(ratio)), int(w * np.sqrt(ratio))
        pil_img = Image.fromarray(image.astype(np.uint8))
        resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        return np.array(resized)
    
    @staticmethod
    def resize_lanczos(image: np.ndarray, ratio: float) -> np.ndarray:
        """Lanczos resize (high quality)."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * np.sqrt(ratio)), int(w * np.sqrt(ratio))
        pil_img = Image.fromarray(image.astype(np.uint8))
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(resized)
    
    @staticmethod
    def jpeg_compress(image: np.ndarray, target_ratio: float) -> np.ndarray:
        """JPEG compression at quality level to achieve target ratio."""
        pil_img = Image.fromarray(image.astype(np.uint8))
        
        # Binary search for quality level
        original_size = image.shape[0] * image.shape[1] * 3
        target_size = int(original_size * target_ratio)
        
        low, high = 1, 95
        best_quality = 50
        best_result = None
        
        while low <= high:
            mid = (low + high) // 2
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=mid)
            size = buffer.tell()
            
            if size <= target_size:
                best_quality = mid
                buffer.seek(0)
                best_result = np.array(Image.open(buffer))
                low = mid + 1
            else:
                high = mid - 1
        
        if best_result is None:
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=1)
            buffer.seek(0)
            best_result = np.array(Image.open(buffer))
        
        return best_result
    
    @staticmethod
    def webp_compress(image: np.ndarray, target_ratio: float) -> np.ndarray:
        """WebP compression."""
        pil_img = Image.fromarray(image.astype(np.uint8))
        
        original_size = image.shape[0] * image.shape[1] * 3
        target_size = int(original_size * target_ratio)
        
        # Try different quality levels
        for quality in range(95, 0, -5):
            buffer = io.BytesIO()
            pil_img.save(buffer, format="WEBP", quality=quality, method=6)
            if buffer.tell() <= target_size:
                buffer.seek(0)
                return np.array(Image.open(buffer))
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format="WEBP", quality=1)
        buffer.seek(0)
        return np.array(Image.open(buffer))


class ComparisonRunner:
    """Run comparison between semantic-compress and baselines."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = MetricsComputer()
        self.baseline = BaselineCompressor()
    
    def compare_methods(
        self,
        image: np.ndarray,
        target_ratio: float
    ) -> List[ComparisonResult]:
        """Compare all methods on a single image."""
        results = []
        h, w = image.shape[:2]
        original_pixels = h * w
        
        methods = {
            "Bilinear": lambda img, r: self.baseline.resize_bilinear(img, r),
            "Bicubic": lambda img, r: self.baseline.resize_bicubic(img, r),
            "Lanczos": lambda img, r: self.baseline.resize_lanczos(img, r),
            "JPEG": lambda img, r: self.baseline.jpeg_compress(img, r),
            "WebP": lambda img, r: self.baseline.webp_compress(img, r),
            "Semantic-Deep": lambda img, r: self._semantic_compress(img, r, "deep"),
            "Semantic-Hybrid": lambda img, r: self._semantic_compress(img, r, "hybrid"),
        }
        
        for name, method in methods.items():
            try:
                start = time.perf_counter()
                compressed = method(image, target_ratio)
                elapsed = (time.perf_counter() - start) * 1000
                
                # Compute actual ratio
                if compressed.shape[0] * compressed.shape[1] != original_pixels:
                    actual_ratio = (compressed.shape[0] * compressed.shape[1]) / original_pixels
                else:
                    actual_ratio = target_ratio
                
                # Compute file size
                pil_img = Image.fromarray(compressed.astype(np.uint8))
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                file_size = buffer.tell()
                
                # Bits per pixel
                bpp = (file_size * 8) / (compressed.shape[0] * compressed.shape[1])
                
                # Compute metrics
                metrics_dict = self.metrics.compute_all(image, compressed)
                
                result = ComparisonResult(
                    method_name=name,
                    compression_ratio=actual_ratio,
                    ssim=metrics_dict["ssim"],
                    psnr=metrics_dict["psnr"],
                    lpips=metrics_dict.get("lpips", -1),
                    feature_similarity=metrics_dict["feature_similarity"],
                    processing_time_ms=elapsed,
                    file_size_bytes=file_size,
                    bpp=bpp
                )
                results.append(result)
            except Exception as e:
                print(f"  {name} failed: {e}")
        
        return results
    
    def _semantic_compress(
        self,
        image: np.ndarray,
        ratio: float,
        method: str
    ) -> np.ndarray:
        """Compress using semantic-compress."""
        if method == "deep":
            energy_fn = DeepEnergyFunction(model_name="vgg16")
        else:
            energy_fn = HybridEnergyFunction(model_name="vgg16")
        
        compressor = SemanticCompressor(
            energy_function=energy_fn,
            target_ratio=ratio,
            max_iterations=50
        )
        result = compressor.compress(image)
        return result.compressed_image
    
    def run_comparison(
        self,
        image_sizes: List[Tuple[int, int]] = None,
        target_ratios: List[float] = None
    ) -> Dict:
        """Run full comparison."""
        if image_sizes is None:
            image_sizes = [(512, 512)]
        if target_ratios is None:
            target_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        
        print("Running baseline comparison...")
        print(f"Image sizes: {image_sizes}")
        print(f"Target ratios: {target_ratios}")
        print()
        
        # Generate test image
        h, w = image_sizes[0]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        y, x = np.ogrid[:h, :w]
        img[:, :, 0] = (np.sin(x / 30) * 127 + 128).astype(np.uint8)
        img[:, :, 1] = (np.cos(y / 30) * 127 + 128).astype(np.uint8)
        img[:, :, 2] = ((x + y) / 2 % 256).astype(np.uint8)
        
        all_results = []
        
        for ratio in target_ratios:
            print(f"Testing ratio {ratio:.2f}...")
            results = self.compare_methods(img, ratio)
            all_results.extend(results)
        
        self._save_comparison(all_results)
        self._generate_comparison_report(all_results)
        
        return {"results": [r.__dict__ for r in all_results]}
    
    def _save_comparison(self, results: List[ComparisonResult]):
        """Save comparison results."""
        results_dict = {
            "comparison_type": "baseline",
            "results": [r.__dict__ for r in results]
        }
        
        output_file = self.output_dir / "comparison_results.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nComparison results saved to: {output_file}")
    
    def _generate_comparison_report(self, results: List[ComparisonResult]):
        """Generate comparison report."""
        report_file = self.output_dir / "comparison_report.md"
        
        with open(report_file, "w") as f:
            f.write("# Baseline Comparison Report\n\n")
            f.write("Comparison of semantic-compress against standard compression methods.\n\n")
            
            # Group by compression ratio
            ratios = sorted(set(r.compression_ratio for r in results))
            
            for ratio in ratios:
                ratio_results = [r for r in results if abs(r.compression_ratio - ratio) < 0.05]
                f.write(f"## Compression Ratio: {ratio:.2f}\n\n")
                
                f.write("| Method | SSIM | PSNR | LPIPS | Time (ms) | BPP |\n")
                f.write("|--------|------|------|-------|-----------|-----|\n")
                
                for r in sorted(ratio_results, key=lambda x: x.ssim, reverse=True):
                    lpips_str = f"{r.lpips:.4f}" if r.lpips >= 0 else "N/A"
                    f.write(f"| {r.method_name} | {r.ssim:.4f} | {r.psnr:.2f} | "
                           f"{lpips_str} | {r.processing_time_ms:.1f} | {r.bpp:.2f} |\n")
                
                f.write("\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            methods = set(r.method_name for r in results)
            for method in methods:
                method_results = [r for r in results if r.method_name == method]
                f.write(f"### {method}\n\n")
                f.write(f"- Average SSIM: {np.mean([r.ssim for r in method_results]):.4f}\n")
                f.write(f"- Average PSNR: {np.mean([r.psnr for r in method_results]):.2f} dB\n")
                f.write(f"- Average Time: {np.mean([r.processing_time_ms for r in method_results]):.2f} ms\n")
                f.write(f"- Average BPP: {np.mean([r.bpp for r in method_results]):.2f}\n\n")
        
        print(f"Comparison report saved to: {report_file}")


def main():
    runner = ComparisonRunner()
    runner.run_comparison()


if __name__ == "__main__":
    main()
