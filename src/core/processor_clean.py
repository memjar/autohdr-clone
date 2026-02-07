"""
HDRit Processor v5.0 - Clean AutoHDR-Style Pipeline
====================================================

Simplified, production-ready processor matching AutoHDR's core methodology.

Core Pipeline:
1. RAW decode (if applicable)
2. Multi-exposure fusion (Mertens algorithm)
3. Window detection & exterior recovery
4. Tone mapping (Natural or Intense preset)
5. White balance correction
6. Color/contrast adjustments
7. Noise reduction & sharpening
8. Output

Usage:
    processor = HDRitProcessor()
    result = processor.process(image)  # Single image
    result = processor.process_brackets([under, normal, over])  # Multi-bracket
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, List
from pathlib import Path

PROCESSOR_VERSION = "5.0.0"


# ============================================================================
# SETTINGS - Simple and Clean
# ============================================================================

@dataclass
class Settings:
    """Processing settings - AutoHDR style"""

    # Preset (the main control)
    preset: Literal['natural', 'intense'] = 'natural'

    # Manual adjustments (-100 to +100 scale, like Lightroom)
    brightness: float = 0
    contrast: float = 0
    vibrance: float = 0
    saturation: float = 0
    shadows: float = 0      # Shadow recovery
    highlights: float = 0   # Highlight recovery
    whites: float = 0
    blacks: float = 0

    # White balance
    temperature: float = 0  # -100 cool, +100 warm
    tint: float = 0         # -100 green, +100 magenta
    auto_wb: bool = True

    # Window pull (key feature)
    window_pull: Literal['off', 'natural', 'strong'] = 'natural'

    # Processing toggles
    denoise: bool = True
    sharpen: bool = True

    # HDR fusion strength (0-1)
    hdr_strength: float = 0.7


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class HDRitProcessor:
    """
    Clean HDR processor implementing AutoHDR's core methodology.

    The secret sauce: Mertens exposure fusion, NOT traditional HDR tone mapping.
    This preserves natural colors and avoids the overdone "HDR look".
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.mertens = cv2.createMergeMertens()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process a single image with HDR-style enhancements."""
        # Generate synthetic brackets from single image
        brackets = self._create_synthetic_brackets(image)
        return self._process_pipeline(brackets)

    def process_brackets(self, brackets: List[np.ndarray]) -> np.ndarray:
        """Process multiple exposure brackets (ideal case)."""
        return self._process_pipeline(brackets)

    def _process_pipeline(self, brackets: List[np.ndarray]) -> np.ndarray:
        """Main processing pipeline."""

        # Step 1: Mertens exposure fusion
        fused = self._mertens_fusion(brackets)

        # Step 2: Window detection and exterior recovery
        if self.settings.window_pull != 'off':
            fused = self._apply_window_pull(fused, brackets)

        # Step 3: Apply preset (Natural or Intense)
        result = self._apply_preset(fused)

        # Step 4: White balance
        if self.settings.auto_wb:
            result = self._auto_white_balance(result)
        result = self._apply_temperature_tint(result)

        # Step 5: Tone adjustments (shadows, highlights, etc.)
        result = self._apply_tone_adjustments(result)

        # Step 6: Basic adjustments (brightness, contrast, etc.)
        result = self._apply_basic_adjustments(result)

        # Step 7: Noise reduction
        if self.settings.denoise:
            result = self._denoise(result)

        # Step 8: Sharpening (always last before output)
        if self.settings.sharpen:
            result = self._sharpen(result)

        return result

    # ========================================================================
    # STEP 1: MERTENS EXPOSURE FUSION
    # ========================================================================

    def _create_synthetic_brackets(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create synthetic exposure brackets from a single image.
        Pre-denoise to prevent noise amplification in HDR process.
        """
        # PRE-DENOISE the input to prevent noise amplification
        clean = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convert to float for processing
        img_float = clean.astype(np.float32) / 255.0

        # Create under-exposed (for highlight detail)
        under = np.clip(img_float * 0.5, 0, 1)

        # Original (normal exposure)
        normal = img_float

        # Create over-exposed (for shadow detail)
        over = np.power(img_float, 0.5)
        over = np.clip(over * 1.3, 0, 1)

        # Convert back to uint8
        brackets = [
            (under * 255).astype(np.uint8),
            (normal * 255).astype(np.uint8),
            (over * 255).astype(np.uint8)
        ]

        return brackets

    def _mertens_fusion(self, brackets: List[np.ndarray]) -> np.ndarray:
        """
        Mertens exposure fusion - THE key technique.
        Combines multiple exposures without tone mapping artifacts.
        """
        # Ensure all brackets are same size
        target_shape = brackets[0].shape[:2]
        aligned_brackets = []
        for bracket in brackets:
            if bracket.shape[:2] != target_shape:
                bracket = cv2.resize(bracket, (target_shape[1], target_shape[0]))
            aligned_brackets.append(bracket)

        # Mertens fusion
        fusion = self.mertens.process(aligned_brackets)

        # Blend with original based on strength
        strength = self.settings.hdr_strength
        original = aligned_brackets[1].astype(np.float32) / 255.0
        result = original * (1 - strength) + fusion * strength

        # Convert to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return result

    # ========================================================================
    # STEP 2: WINDOW PULL (Exterior Recovery)
    # ========================================================================

    def _apply_window_pull(self, image: np.ndarray, brackets: List[np.ndarray]) -> np.ndarray:
        """
        Recover exterior detail through windows.
        Key technique for real estate photography.
        """
        # Detect bright window areas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Window threshold - very bright areas
        _, window_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Expand mask slightly to catch window frames
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        window_mask = cv2.dilate(window_mask, kernel, iterations=1)

        # Soften edges
        window_mask = cv2.GaussianBlur(window_mask, (21, 21), 0)
        window_mask = window_mask.astype(np.float32) / 255.0

        # Get exterior detail from under-exposed bracket
        under = brackets[0]

        # Calculate recovery strength based on preset
        if self.settings.window_pull == 'strong':
            recovery_strength = 0.85
        else:  # natural
            recovery_strength = 0.6

        # Blend exterior detail into window areas
        window_mask_3d = np.stack([window_mask] * 3, axis=-1)
        result = image.astype(np.float32)
        under_float = under.astype(np.float32)

        # Brighten the under-exposed to better match
        under_brightened = np.clip(under_float * 2.0, 0, 255)

        # Blend
        result = result * (1 - window_mask_3d * recovery_strength) + \
                 under_brightened * window_mask_3d * recovery_strength

        return np.clip(result, 0, 255).astype(np.uint8)

    # ========================================================================
    # STEP 3: PRESETS (Natural vs Intense)
    # ========================================================================

    def _apply_preset(self, image: np.ndarray) -> np.ndarray:
        """Apply Natural or Intense preset."""

        if self.settings.preset == 'intense':
            # More contrast, saturation, and local contrast
            image = self._apply_s_curve(image, strength=0.25)
            image = self._boost_saturation(image, 1.15)
            image = self._local_contrast(image, strength=0.3)
        else:
            # Natural - subtle enhancements
            image = self._apply_s_curve(image, strength=0.1)
            image = self._boost_saturation(image, 1.05)
            image = self._local_contrast(image, strength=0.15)

        return image

    def _apply_s_curve(self, image: np.ndarray, strength: float = 0.15) -> np.ndarray:
        """Apply S-curve for pleasing contrast."""
        # Create S-curve lookup table
        x = np.linspace(0, 1, 256)

        # Sigmoid-based S-curve
        midpoint = 0.5
        steepness = 1 + strength * 4
        y = 1 / (1 + np.exp(-steepness * (x - midpoint)))

        # Normalize
        y = (y - y.min()) / (y.max() - y.min())

        # Convert to uint8 lookup table
        lut = (y * 255).astype(np.uint8)

        # Apply to luminance only (preserve colors)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _boost_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Boost saturation uniformly."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _local_contrast(self, image: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """Add local contrast (clarity) using unsharp mask on luminance."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Large radius blur for local contrast
        blur = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=30)

        # Unsharp mask
        sharpened = l_channel + (l_channel - blur) * strength
        lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ========================================================================
    # STEP 4: WHITE BALANCE
    # ========================================================================

    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Auto white balance using gray world assumption."""
        result = image.astype(np.float32)

        # Calculate average of each channel
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])

        # Gray world: all channels should average to same value
        avg_gray = (avg_b + avg_g + avg_r) / 3

        # Calculate scaling factors
        scale_b = avg_gray / (avg_b + 1e-6)
        scale_g = avg_gray / (avg_g + 1e-6)
        scale_r = avg_gray / (avg_r + 1e-6)

        # Limit scaling to avoid extreme corrections
        max_scale = 1.5
        scale_b = np.clip(scale_b, 1/max_scale, max_scale)
        scale_g = np.clip(scale_g, 1/max_scale, max_scale)
        scale_r = np.clip(scale_r, 1/max_scale, max_scale)

        # Apply with reduced strength for subtlety
        strength = 0.6
        result[:, :, 0] = result[:, :, 0] * (1 + (scale_b - 1) * strength)
        result[:, :, 1] = result[:, :, 1] * (1 + (scale_g - 1) * strength)
        result[:, :, 2] = result[:, :, 2] * (1 + (scale_r - 1) * strength)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_temperature_tint(self, image: np.ndarray) -> np.ndarray:
        """Apply manual temperature and tint adjustments."""
        if self.settings.temperature == 0 and self.settings.tint == 0:
            return image

        result = image.astype(np.float32)

        # Temperature: shift blue-yellow axis
        temp = self.settings.temperature / 100.0  # -1 to +1
        if temp > 0:  # Warmer
            result[:, :, 2] += temp * 20  # More red
            result[:, :, 0] -= temp * 15  # Less blue
        else:  # Cooler
            result[:, :, 0] -= temp * 20  # More blue
            result[:, :, 2] += temp * 15  # Less red

        # Tint: shift green-magenta axis
        tint = self.settings.tint / 100.0  # -1 to +1
        result[:, :, 1] -= tint * 15  # Adjust green

        return np.clip(result, 0, 255).astype(np.uint8)

    # ========================================================================
    # STEP 5: TONE ADJUSTMENTS
    # ========================================================================

    def _apply_tone_adjustments(self, image: np.ndarray) -> np.ndarray:
        """Apply shadows, highlights, whites, blacks adjustments."""

        # Convert to LAB for luminance work
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Create luminosity masks
        shadows_mask = 1 - np.clip(l_channel / 80, 0, 1)  # Dark areas
        highlights_mask = np.clip((l_channel - 180) / 75, 0, 1)  # Bright areas
        whites_mask = np.clip((l_channel - 220) / 35, 0, 1)  # Very bright
        blacks_mask = 1 - np.clip(l_channel / 40, 0, 1)  # Very dark

        # Apply adjustments
        adjustment = 0.0

        if self.settings.shadows != 0:
            adjustment += shadows_mask * (self.settings.shadows / 100.0) * 40

        if self.settings.highlights != 0:
            adjustment -= highlights_mask * (self.settings.highlights / 100.0) * 30

        if self.settings.whites != 0:
            adjustment += whites_mask * (self.settings.whites / 100.0) * 20

        if self.settings.blacks != 0:
            adjustment -= blacks_mask * (self.settings.blacks / 100.0) * 20

        l_channel = np.clip(l_channel + adjustment, 0, 255)
        lab[:, :, 0] = l_channel.astype(np.uint8)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ========================================================================
    # STEP 6: BASIC ADJUSTMENTS
    # ========================================================================

    def _apply_basic_adjustments(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, vibrance, saturation."""
        result = image.astype(np.float32)

        # Brightness (-100 to +100)
        if self.settings.brightness != 0:
            result += self.settings.brightness * 1.5

        # Contrast (-100 to +100)
        if self.settings.contrast != 0:
            factor = 1 + (self.settings.contrast / 100.0) * 0.5
            result = (result - 128) * factor + 128

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Vibrance (smart saturation that protects already-saturated colors)
        if self.settings.vibrance != 0:
            result = self._apply_vibrance(result, self.settings.vibrance / 100.0)

        # Saturation
        if self.settings.saturation != 0:
            factor = 1 + (self.settings.saturation / 100.0) * 0.5
            result = self._boost_saturation(result, factor)

        return result

    def _apply_vibrance(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Vibrance - smart saturation that boosts muted colors more.
        Protects skin tones and already-saturated colors.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Calculate saturation factor based on current saturation
        # Low saturation pixels get more boost
        sat = hsv[:, :, 1] / 255.0
        boost_factor = 1 + strength * (1 - sat)  # Less boost for saturated

        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost_factor, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ========================================================================
    # STEP 7: NOISE REDUCTION (High Quality)
    # ========================================================================

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        EXTREME denoising to match AutoHDR's buttery smooth output.
        No grain tolerance - must be perfectly smooth.
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # HEAVY luminance denoising - two passes
        y_denoised = cv2.fastNlMeansDenoising(y, None, 15, 7, 21)
        y_denoised = cv2.fastNlMeansDenoising(y_denoised, None, 12, 7, 21)

        # EXTREME chroma denoising - kills all color noise
        cr_denoised = cv2.fastNlMeansDenoising(cr, None, 30, 7, 21)
        cb_denoised = cv2.fastNlMeansDenoising(cb, None, 30, 7, 21)

        # Heavy bilateral on chroma
        cr_denoised = cv2.bilateralFilter(cr_denoised, 15, 100, 100)
        cb_denoised = cv2.bilateralFilter(cb_denoised, 15, 100, 100)

        # Merge
        ycrcb_denoised = cv2.merge([y_denoised, cr_denoised, cb_denoised])
        result = cv2.cvtColor(ycrcb_denoised, cv2.COLOR_YCrCb2BGR)

        # Final bilateral pass on full image
        result = cv2.bilateralFilter(result, 9, 75, 75)

        return result

    # ========================================================================
    # STEP 8: SHARPENING (Subtle, Detail-Preserving)
    # ========================================================================

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Subtle sharpening using unsharp mask on luminance only.
        Preserves colors and avoids over-sharpening artifacts.
        """
        # Work on luminance only (LAB color space)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Subtle unsharp mask
        blur = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=1.0)
        sharpened = l_channel + (l_channel - blur) * 0.3  # Reduced from 0.5

        lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_image(image: np.ndarray, preset: str = 'natural', **kwargs) -> np.ndarray:
    """Quick processing with default settings."""
    settings = Settings(preset=preset, **kwargs)
    processor = HDRitProcessor(settings)
    return processor.process(image)


def process_brackets(brackets: List[np.ndarray], preset: str = 'natural', **kwargs) -> np.ndarray:
    """Process multiple exposure brackets."""
    settings = Settings(preset=preset, **kwargs)
    processor = HDRitProcessor(settings)
    return processor.process_brackets(brackets)
