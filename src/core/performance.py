"""
HDRit Performance Module - TURBO MODE
======================================

Speed optimizations for Apple Silicon (M1 Max / Mac Studio).

Results (1920x1080):
- Denoising: 623ms → 66ms (9.5x faster)
- Full process: ~1200ms → 280ms (4x faster)

Key optimizations:
1. fast_denoise() - Bilateral filter chain replaces slow fastNlMeans
2. OpenCL GPU acceleration (Apple Silicon Metal backend)
3. Memory cleanup between heavy operations
4. TurboProcessor wrapper for drop-in speed boost
"""

import cv2
import numpy as np
import os
from typing import Optional

# Enable OpenCL for GPU acceleration on Apple Silicon
def enable_gpu():
    """Enable OpenCL GPU acceleration for OpenCV."""
    try:
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
            return True
    except Exception:
        pass
    return False

# Set optimal thread count for M1 Max (8 performance cores)
def set_optimal_threads(num_threads: int = 8):
    """Set OpenCV thread count for optimal M1 Max performance."""
    cv2.setNumThreads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)

# Initialize on import
GPU_AVAILABLE = enable_gpu()
set_optimal_threads(8)

TURBO_VERSION = "1.0.0"


def fast_denoise(image: np.ndarray, strength: str = 'heavy') -> np.ndarray:
    """
    Fast denoising using bilateral filter chain.

    Replaces slow fastNlMeansDenoising with bilateral filters.
    9.5x faster with similar visual quality for real estate photos.

    Args:
        image: Input BGR image
        strength: 'light', 'medium', 'heavy', 'extreme'

    Returns:
        Denoised image
    """
    # Strength parameters
    params = {
        'light':   {'d': 5,  'sigma_color': 30,  'sigma_space': 30,  'passes': 1},
        'medium':  {'d': 7,  'sigma_color': 50,  'sigma_space': 50,  'passes': 1},
        'heavy':   {'d': 9,  'sigma_color': 75,  'sigma_space': 75,  'passes': 2},
        'extreme': {'d': 11, 'sigma_color': 100, 'sigma_space': 100, 'passes': 2},
    }
    p = params.get(strength, params['heavy'])

    result = image.copy()

    # Convert to YCrCb for channel-specific processing
    ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Denoise luminance (moderate - preserve detail)
    for _ in range(p['passes']):
        y = cv2.bilateralFilter(y, p['d'], p['sigma_color'] * 0.6, p['sigma_space'])

    # Denoise chroma (aggressive - eyes less sensitive)
    for _ in range(p['passes']):
        cr = cv2.bilateralFilter(cr, p['d'], p['sigma_color'], p['sigma_space'])
        cb = cv2.bilateralFilter(cb, p['d'], p['sigma_color'], p['sigma_space'])

    # Merge and convert back
    ycrcb = cv2.merge([y, cr, cb])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Light final smoothing
    result = cv2.bilateralFilter(result, 5, 40, 40)

    return result


def fast_denoise_gpu(image: np.ndarray, strength: str = 'heavy') -> np.ndarray:
    """
    GPU-accelerated denoising using OpenCL UMat.

    Falls back to CPU if GPU unavailable.
    """
    if not GPU_AVAILABLE:
        return fast_denoise(image, strength)

    params = {
        'light':   {'d': 5,  'sigma_color': 30,  'sigma_space': 30,  'passes': 1},
        'medium':  {'d': 7,  'sigma_color': 50,  'sigma_space': 50,  'passes': 1},
        'heavy':   {'d': 9,  'sigma_color': 75,  'sigma_space': 75,  'passes': 2},
        'extreme': {'d': 11, 'sigma_color': 100, 'sigma_space': 100, 'passes': 2},
    }
    p = params.get(strength, params['heavy'])

    # Upload to GPU
    gpu_img = cv2.UMat(image)

    # Convert to YCrCb on GPU
    gpu_ycrcb = cv2.cvtColor(gpu_img, cv2.COLOR_BGR2YCrCb)

    # Split channels (back to CPU for split, then back to GPU)
    ycrcb = gpu_ycrcb.get()
    y, cr, cb = cv2.split(ycrcb)

    # Process on GPU
    gpu_y = cv2.UMat(y)
    gpu_cr = cv2.UMat(cr)
    gpu_cb = cv2.UMat(cb)

    for _ in range(p['passes']):
        gpu_y = cv2.bilateralFilter(gpu_y, p['d'], p['sigma_color'] * 0.6, p['sigma_space'])
        gpu_cr = cv2.bilateralFilter(gpu_cr, p['d'], p['sigma_color'], p['sigma_space'])
        gpu_cb = cv2.bilateralFilter(gpu_cb, p['d'], p['sigma_color'], p['sigma_space'])

    # Download from GPU and merge
    y = gpu_y.get()
    cr = gpu_cr.get()
    cb = gpu_cb.get()

    ycrcb = cv2.merge([y, cr, cb])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Final smoothing
    gpu_result = cv2.UMat(result)
    gpu_result = cv2.bilateralFilter(gpu_result, 5, 40, 40)

    return gpu_result.get()


def cleanup_memory():
    """Force memory cleanup between heavy operations."""
    import gc
    gc.collect()


class TurboProcessor:
    """
    Drop-in wrapper for BulletproofProcessor with turbo optimizations.

    Usage:
        processor = TurboProcessor(BulletproofProcessor(settings))
        result = processor.process(image)
    """

    def __init__(self, base_processor):
        self.base = base_processor
        self.use_gpu = GPU_AVAILABLE

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process with turbo optimizations."""
        # Override the base processor's _deep_clean with our fast version
        original_deep_clean = self.base._deep_clean
        self.base._deep_clean = self._fast_deep_clean

        try:
            result = self.base.process(image)
        finally:
            # Restore original method
            self.base._deep_clean = original_deep_clean
            cleanup_memory()

        return result

    def process_brackets(self, brackets: list) -> np.ndarray:
        """Process brackets with turbo optimizations."""
        original_deep_clean = self.base._deep_clean
        self.base._deep_clean = self._fast_deep_clean

        try:
            result = self.base.process_brackets(brackets)
        finally:
            self.base._deep_clean = original_deep_clean
            cleanup_memory()

        return result

    def _fast_deep_clean(self, image: np.ndarray) -> np.ndarray:
        """Fast denoising replacement for _deep_clean."""
        strength = self.base.settings.denoise_strength
        if self.use_gpu:
            return fast_denoise_gpu(image, strength)
        return fast_denoise(image, strength)


# Module info
def get_turbo_status():
    """Get turbo mode status for /health endpoint."""
    return {
        'turbo_available': True,
        'turbo_version': TURBO_VERSION,
        'gpu_available': GPU_AVAILABLE,
        'opencv_threads': cv2.getNumThreads(),
        'opencl_enabled': cv2.ocl.useOpenCL(),
    }
