"""
AutoHDR Clone - Core Image Processor v2
=======================================

Enhanced HDR processing with LAB color space for professional quality.

Improvements over v1:
- LAB color space for luminance manipulation (no color shifts)
- Multi-scale local contrast (clarity at fine/medium/coarse frequencies)
- Smooth highlight rolloff (filmic, no hard clipping)
- Adaptive shadow recovery with noise prevention
- Midtone contrast enhancement (S-curve)
- Edge-aware detail enhancement

Full Pipeline:
1. HDR Tone Mapping (LAB-based)
2. Brightness/Contrast/Vibrance/White Balance
3. Perspective Correction
4. Sky Enhancement
5. Window Pull
6. Grass Enhancement
7. Sign Removal
8. Twilight Effect

Usage:
    processor = AutoHDRProcessor(settings)
    result = processor.process(image)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

# Processor version
PROCESSOR_VERSION = "2.1.0"  # Added edge-aware halo prevention + pro window pull


@dataclass
class ProcessingSettings:
    """Settings for AutoHDR processing"""
    # Scene type
    scene_type: Literal['interior', 'exterior'] = 'interior'

    # Adjustments (-2 to +2 scale)
    brightness: float = 0.0
    contrast: float = 0.0
    vibrance: float = 0.0
    white_balance: float = 0.0

    # Window enhancement
    window_pull_intensity: Literal['natural', 'medium', 'strong'] = 'natural'

    # Sky settings
    cloud_style: Literal['fluffy', 'dramatic', 'clear'] = 'fluffy'
    retain_original_sky: bool = False
    interior_clouds: bool = True
    exterior_clouds: bool = True

    # Twilight
    twilight_style: Optional[Literal['pink', 'blue', 'orange']] = None

    # Enhancements
    perspective_correction: bool = True
    grass_replacement: bool = False
    sign_removal: bool = False
    declutter: bool = False

    # Special effects
    fire_in_fireplace: bool = False
    tv_replacement: Optional[str] = None


class AutoHDRProcessor:
    """
    Main processor implementing AutoHDR's editing pipeline.
    """

    def __init__(self, settings: Optional[ProcessingSettings] = None):
        self.settings = settings or ProcessingSettings()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Execute full processing pipeline.

        Args:
            image: Input image (BGR, uint8)

        Returns:
            Processed image (BGR, uint8)
        """
        result = image.copy()

        # ==========================================
        # STAGE 1: HDR TONE MAPPING (CORE EFFECT)
        # ==========================================
        # Reduced from 0.7 to 0.5 for more natural results
        result = self.apply_hdr_effect(result, strength=0.5)

        # ==========================================
        # STAGE 2-5: ADJUSTMENTS
        # ==========================================
        if self.settings.brightness != 0:
            result = self.adjust_brightness(result, self.settings.brightness)

        if self.settings.contrast != 0:
            result = self.adjust_contrast(result, self.settings.contrast)

        if self.settings.vibrance != 0:
            result = self.adjust_vibrance(result, self.settings.vibrance)

        if self.settings.white_balance != 0:
            result = self.adjust_white_balance(result, self.settings.white_balance)

        # ==========================================
        # STAGE 6: PERSPECTIVE CORRECTION
        # ==========================================
        if self.settings.perspective_correction:
            result = self.correct_perspective(result)

        # ==========================================
        # STAGE 7: GRASS ENHANCEMENT
        # ==========================================
        if self.settings.grass_replacement:
            result = self.enhance_grass(result)

        # ==========================================
        # STAGE 8: SIGN REMOVAL
        # ==========================================
        if self.settings.sign_removal:
            result = self.remove_signs(result)

        # ==========================================
        # STAGE 9: DECLUTTER
        # ==========================================
        if self.settings.declutter:
            result = self.declutter(result)

        # ==========================================
        # STAGE 10: SKY ENHANCEMENT
        # ==========================================
        if self.settings.scene_type == 'exterior' and not self.settings.retain_original_sky:
            result = self.enhance_sky(result, self.settings.cloud_style)

        # ==========================================
        # STAGE 11: WINDOW ENHANCEMENT
        # ==========================================
        result = self.enhance_windows(result, self.settings.window_pull_intensity)

        # ==========================================
        # STAGE 12: TWILIGHT EFFECT
        # ==========================================
        if self.settings.twilight_style:
            result = self.apply_twilight(result, self.settings.twilight_style)

        return result

    # ==========================================
    # CORE EFFECTS
    # ==========================================

    def apply_hdr_effect(
        self,
        image: np.ndarray,
        strength: float = 0.5,           # Reduced from 0.7 for natural look
        shadow_recovery: float = 0.35,   # Reduced from 0.4
        highlight_compression: float = 0.25,  # Reduced from 0.3
        local_contrast: float = 0.25,    # Reduced from 0.35 (prevents halos)
        midtone_contrast: float = 0.10   # Reduced from 0.15
    ) -> np.ndarray:
        """
        Advanced HDR tone mapping using LAB color space.

        LAB separates luminance from color, preventing color shifts
        that occur when processing RGB directly.

        Args:
            image: Input BGR uint8 image
            strength: Overall effect strength (0-1)
            shadow_recovery: How much to lift shadows (0-1)
            highlight_compression: How much to compress highlights (0-1)
            local_contrast: Clarity/local contrast amount (0-1)
            midtone_contrast: Midtone punch (0-1)

        Returns:
            Processed BGR uint8 image
        """
        # Convert to LAB color space (better for luminance manipulation)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = cv2.split(lab)

        # Normalize L channel to 0-1
        L_norm = L / 255.0

        # STEP 1: Adaptive shadow recovery
        L_norm = self._adaptive_shadow_recovery(L_norm, shadow_recovery * strength)

        # STEP 2: Highlight compression with smooth rolloff
        L_norm = self._highlight_rolloff(L_norm, highlight_compression * strength)

        # STEP 3: Multi-scale local contrast (the "HDR look")
        L_norm = self._multiscale_local_contrast(L_norm, local_contrast * strength)

        # STEP 4: Midtone contrast (S-curve for punch)
        L_norm = self._midtone_contrast(L_norm, midtone_contrast * strength)

        # STEP 5: Edge-aware detail enhancement
        L_norm = self._edge_aware_sharpen(L_norm, strength=0.1 * strength)

        # Convert back
        L_out = np.clip(L_norm * 255, 0, 255).astype(np.float32)

        # Minimal saturation adjustment (too much looks unnatural)
        # Reduced from 0.05 to 0.02 for more natural colors
        A = A + (A - 128) * 0.02 * strength
        B = B + (B - 128) * 0.02 * strength

        lab_out = cv2.merge([L_out, A, B])
        result = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    def _adaptive_shadow_recovery(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        Lift shadows adaptively - darker areas get more boost.
        Uses soft knee curve to prevent noise amplification in very dark areas.
        """
        if amount <= 0:
            return L

        # Shadow mask: darker pixels get higher weight
        shadow_weight = np.power(1.0 - L, 2.5)

        # Soft knee: limit boost in very dark areas (prevents noise)
        soft_knee = np.where(L < 0.1, L / 0.1, 1.0)
        shadow_weight = shadow_weight * soft_knee

        # Apply shadow lift
        lift = shadow_weight * amount * 0.5
        result = L + lift

        return np.clip(result, 0, 1)

    def _highlight_rolloff(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        Compress highlights with smooth filmic rolloff (no hard clipping).
        """
        if amount <= 0:
            return L

        # Filmic highlight rolloff
        threshold = 0.7  # Start compression above this
        mask = L > threshold

        if not mask.any():
            return L

        result = L.copy()

        # Soft compression above threshold
        over = (L[mask] - threshold) / (1 - threshold)  # 0-1 range above threshold

        # Filmic curve: x / (x + 1) style compression
        compressed = over / (1 + over * amount * 2)

        # Map back to original range
        result[mask] = threshold + compressed * (1 - threshold) * (1 - amount * 0.3)

        return result

    def _multiscale_local_contrast(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        Edge-aware multi-scale local contrast enhancement (clarity).
        Combines fine, medium, and coarse detail at different frequencies.
        REDUCES contrast near edges to prevent halos.
        """
        if amount <= 0:
            return L

        # ==========================================
        # EDGE DETECTION FOR HALO PREVENTION
        # ==========================================
        # Detect edges where halos typically form
        L_uint8 = (L * 255).astype(np.uint8)
        edges = cv2.Canny(L_uint8, 30, 100)

        # Dilate edges to create protection zone
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge_zone = cv2.dilate(edges, kernel, iterations=2)

        # Smooth the edge mask for gradual falloff
        edge_mask = cv2.GaussianBlur(edge_zone.astype(np.float32), (21, 21), 0) / 255.0

        # Edge protection: reduce local contrast strength near edges
        # 1.0 = full strength (no edges), 0.3 = reduced strength (near edges)
        edge_protection = 1.0 - (edge_mask * 0.7)

        # ==========================================
        # MULTI-SCALE LOCAL CONTRAST
        # ==========================================
        # Fine detail (small radius) - texture (less affected by halos)
        fine = self._unsharp_mask(L, sigma=5, amount=amount * 0.25)

        # Medium detail (medium radius) - local structure
        medium = self._unsharp_mask(L, sigma=15, amount=amount * 0.4)

        # Coarse detail (large radius) - the "HDR look" (most prone to halos)
        # Apply edge protection mainly to coarse detail
        coarse_raw = self._unsharp_mask(L, sigma=40, amount=amount * 0.5)
        coarse = L + (coarse_raw - L) * edge_protection

        # Combine with reduced weights for more natural look
        result = L + (fine - L) * 0.25 + (medium - L) * 0.35 + (coarse - L) * 0.25

        return np.clip(result, 0, 1)

    def _unsharp_mask(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        """Apply unsharp mask for local contrast at specified scale."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        high_pass = img - blurred
        return img + high_pass * amount

    def _midtone_contrast(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        S-curve for midtone contrast enhancement.
        Adds "punch" to midtones without affecting shadows/highlights.
        """
        if amount <= 0:
            return L

        # S-curve using tanh - center around 0.5
        centered = L - 0.5

        # Tanh-based S-curve with adjustable strength
        curve_strength = 1.0 + amount * 3
        curved = np.tanh(centered * curve_strength) / np.tanh(0.5 * curve_strength)

        # Map back and blend with original
        result = 0.5 + curved * 0.5

        return L * (1 - amount) + result * amount

    def _edge_aware_sharpen(self, L: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """
        Edge-aware sharpening that doesn't create halos.
        Uses bilateral filter to preserve edges.
        """
        if strength <= 0:
            return L

        # Bilateral preserves edges while smoothing
        L_uint8 = (L * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(L_uint8, 9, 75, 75).astype(np.float32) / 255

        # High-pass = original - smoothed
        high_pass = L - smoothed

        # Add back with strength (edge-aware because bilateral preserved edges)
        result = L + high_pass * strength

        return np.clip(result, 0, 1)

    # ==========================================
    # ADJUSTMENTS (LAB-based for quality)
    # ==========================================

    def adjust_brightness(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        Brightness adjustment in LAB space.
        Adjusts luminance only, preserving color integrity.

        Args:
            value: -2 to +2 scale
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Scale value to L range (±25 for full range)
        lab[:, :, 0] += (value / 2.0) * 25
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def adjust_contrast(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        Contrast adjustment in LAB space.
        Adjusts luminance contrast only, preventing color shifts.

        Args:
            value: -2 to +2 scale
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        factor = 1.0 + (value / 4.0)
        L = (L - 128) * factor + 128
        lab[:, :, 0] = np.clip(L, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def adjust_vibrance(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        Vibrance adjustment in LAB space.
        Boosts less-saturated colors more than already-saturated ones.
        More natural than simple saturation boost.

        Args:
            value: -2 to +2 scale
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # A and B channels represent color
        A, B = lab[:, :, 1], lab[:, :, 2]

        # Current saturation (distance from neutral gray)
        sat = np.sqrt((A - 128) ** 2 + (B - 128) ** 2)
        max_sat = np.max(sat) + 1e-6

        # Vibrance: boost less-saturated colors more
        boost = 1.0 + (value / 10.0) * (1.0 - sat / max_sat)

        lab[:, :, 1] = 128 + (A - 128) * boost
        lab[:, :, 2] = 128 + (B - 128) * boost

        lab = np.clip(lab, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def adjust_white_balance(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        White balance adjustment in LAB space.
        Adjusts B channel for warmth (more accurate than RGB).

        Args:
            value: -2 (cooler/blue) to +2 (warmer/yellow)
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # B channel: negative = blue, positive = yellow
        # Warmer = more yellow (increase B)
        # Cooler = more blue (decrease B)
        lab[:, :, 2] += value * 10  # Subtle shift

        lab = np.clip(lab, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ==========================================
    # GEOMETRIC CORRECTIONS
    # ==========================================

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct vertical line distortion.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=image.shape[0] // 4,
                                maxLineGap=10)

        if lines is None or len(lines) < 2:
            return image

        # Find vertical lines (angle close to 90 degrees)
        vertical_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Vertical lines are close to ±90 degrees
                if 70 < abs(angle) < 110:
                    vertical_angles.append(angle)

        if not vertical_angles:
            return image

        # Calculate average deviation from vertical
        avg_angle = np.mean(vertical_angles)
        rotation_needed = 90 - abs(avg_angle) if avg_angle > 0 else -(90 - abs(avg_angle))

        # Only correct if deviation is significant but not too large
        if 0.5 < abs(rotation_needed) < 5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_needed, 1.0)
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return image

    # ==========================================
    # ENVIRONMENTAL ENHANCEMENTS
    # ==========================================

    def enhance_sky(self, image: np.ndarray, style: str = 'fluffy') -> np.ndarray:
        """Enhance sky region based on style."""
        sky_mask = self._detect_sky(image)

        if sky_mask.sum() < 1000:  # No significant sky detected
            return image

        # Enhance existing sky
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask_float = sky_mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_float] * 3, axis=-1)

        if style == 'fluffy':
            # Boost blue and brightness
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask_float * 0.3) + hsv[:, :, 1] * 1.2 * mask_float * 0.3
            hsv[:, :, 2] = hsv[:, :, 2] * (1 - mask_float * 0.2) + hsv[:, :, 2] * 1.1 * mask_float * 0.2
        elif style == 'dramatic':
            # Increase contrast in sky
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask_float * 0.4) + hsv[:, :, 1] * 1.4 * mask_float * 0.4
            hsv[:, :, 2] = hsv[:, :, 2] * (1 - mask_float * 0.1) + hsv[:, :, 2] * 0.95 * mask_float * 0.1

        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _detect_sky(self, image: np.ndarray) -> np.ndarray:
        """Detect sky regions using color analysis."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Blue sky
        lower_blue = np.array([90, 20, 100])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # White/gray sky (overcast)
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 40, 255])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        mask = cv2.bitwise_or(mask_blue, mask_gray)

        # Only consider upper portion of image
        h = image.shape[0]
        mask[int(h * 0.6):, :] = 0

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def enhance_windows(self, image: np.ndarray, intensity: str = 'natural') -> np.ndarray:
        """
        Professional Window Pull using luminosity masking.

        This is THE key technique that separates amateur from pro real estate photos.
        Balances interior exposure with exterior view through windows using
        luminosity-based blending with feathered edges.

        Args:
            image: Input BGR image
            intensity: 'natural' (subtle), 'medium', or 'strong'
        """
        # Intensity controls how much we pull down the windows
        intensity_map = {'natural': 0.4, 'medium': 0.6, 'strong': 0.8}
        pull_strength = intensity_map.get(intensity, 0.4)

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # STEP 1: DETECT WINDOW REGIONS
        # ==========================================
        # Windows are typically bright, rectangular regions

        # Multi-threshold detection for various window brightnesses
        _, very_bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        _, medium_bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine with weights (very bright areas are more likely windows)
        window_likelihood = (very_bright.astype(np.float32) * 1.0 +
                            bright.astype(np.float32) * 0.5 +
                            medium_bright.astype(np.float32) * 0.2) / 1.7

        # Find contours to identify rectangular regions
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        window_mask = np.zeros((h, w), dtype=np.float32)
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect = cw / max(ch, 1)
            img_area = h * w

            # Window characteristics:
            # - Rectangular (aspect 0.3 to 3.0)
            # - Reasonable size (not tiny specks, not huge)
            # - Contains bright pixels
            min_area = img_area * 0.001  # At least 0.1% of image
            max_area = img_area * 0.3    # At most 30% of image

            if min_area < area < max_area and 0.2 < aspect < 5.0:
                # Check if region is actually bright (window-like)
                region_brightness = np.mean(gray[y:y+ch, x:x+cw])
                if region_brightness > 170:
                    # Create soft rectangular mask
                    cv2.rectangle(window_mask, (x, y), (x + cw, y + ch), 1.0, -1)

        if window_mask.sum() < 100:
            return image

        # ==========================================
        # STEP 2: CREATE LUMINOSITY MASK
        # ==========================================
        # Luminosity mask: bright areas = 1, dark areas = 0
        luminosity = gray.astype(np.float32) / 255.0

        # Only affect the bright parts within window regions
        # This preserves window frames and interior elements
        window_luminosity_mask = window_mask * luminosity

        # ==========================================
        # STEP 3: FEATHER THE EDGES
        # ==========================================
        # Critical for natural blending - no hard edges
        # Use larger blur for softer transition
        feather_size = max(15, int(min(h, w) * 0.02))  # 2% of image size
        if feather_size % 2 == 0:
            feather_size += 1

        feathered_mask = cv2.GaussianBlur(window_luminosity_mask, (feather_size, feather_size), 0)

        # Boost the mask contrast slightly for more defined windows
        feathered_mask = np.clip(feathered_mask * 1.3, 0, 1)

        # ==========================================
        # STEP 4: APPLY WINDOW PULL (LUMINOSITY BLEND)
        # ==========================================
        # Convert to LAB for luminosity-only adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Calculate target luminosity for windows
        # Pull bright windows down toward a natural level (~180 in L channel)
        target_L = 180

        # Blend: reduce L in bright window areas
        # The brighter the pixel, the more we pull it down
        pull_amount = (L - target_L) * feathered_mask * pull_strength
        L_adjusted = L - pull_amount

        # ==========================================
        # STEP 5: RECOVER EXTERIOR DETAIL
        # ==========================================
        # For very bright (blown out) areas, add subtle detail recovery
        blown_out = (L > 245).astype(np.float32)
        blown_out = cv2.GaussianBlur(blown_out, (5, 5), 0) * feathered_mask

        # Add subtle texture/detail to blown areas
        # This simulates recovering exterior view
        local_variance = cv2.Laplacian(gray, cv2.CV_32F)
        local_variance = np.abs(local_variance)
        local_variance = cv2.GaussianBlur(local_variance, (5, 5), 0)
        detail_boost = local_variance * blown_out * 0.3

        L_adjusted = L_adjusted - detail_boost

        # ==========================================
        # STEP 6: PRESERVE COLOR INTEGRITY
        # ==========================================
        # Slight saturation boost in window areas to counter the darkening
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # Boost color slightly in pulled areas
        color_boost = 1.0 + feathered_mask * 0.1
        A = 128 + (A - 128) * color_boost
        B = 128 + (B - 128) * color_boost

        # Reconstruct
        lab[:, :, 0] = np.clip(L_adjusted, 0, 255)
        lab[:, :, 1] = np.clip(A, 0, 255)
        lab[:, :, 2] = np.clip(B, 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    def enhance_grass(self, image: np.ndarray) -> np.ndarray:
        """Make grass more vibrant green."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Detect green regions (grass)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv.astype(np.uint8), lower_green, upper_green)

        # Only consider lower portion (ground)
        h = image.shape[0]
        mask[:int(h * 0.3), :] = 0

        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (15, 15), 0)

        # Enhance green: boost saturation, shift hue toward vibrant green
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask_float) + np.clip(hsv[:, :, 1] * 1.3, 0, 255) * mask_float
        # Shift hue toward 60 (pure green)
        hsv[:, :, 0] = hsv[:, :, 0] * (1 - mask_float * 0.3) + 55 * mask_float * 0.3

        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def remove_signs(self, image: np.ndarray) -> np.ndarray:
        """Remove signs using inpainting."""
        # This is a simplified version - full implementation would use ML
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)

            # Sign-like: rectangular, text-sized
            if 1000 < area < 50000 and 0.5 < aspect < 4:
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        if mask.sum() == 0:
            return image

        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    def declutter(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle smoothing to reduce visual clutter."""
        # Bilateral filter preserves edges while smoothing
        return cv2.bilateralFilter(image, 9, 50, 50)

    # ==========================================
    # SPECIAL EFFECTS
    # ==========================================

    def apply_twilight(self, image: np.ndarray, style: str = 'pink') -> np.ndarray:
        """
        Apply realistic day-to-dusk twilight effect.

        Features:
        - Sky gradient from horizon warmth to upper cool
        - Window glow simulation (interior lights)
        - Overall dusk color grading
        - Subtle vignette for atmosphere

        Args:
            style: 'pink' (warm sunset), 'blue' (cool dusk), 'orange' (golden hour)
        """
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # Detect sky region for gradient
        sky_mask = self._detect_sky(image).astype(np.float32) / 255.0

        # Create vertical gradient (warmer at horizon, cooler up top)
        gradient = np.linspace(0.3, 1.0, h).reshape(-1, 1)
        gradient = np.tile(gradient, (1, w))

        # Style-specific color grading
        if style == 'pink':
            # Warm pink/magenta sunset
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)
            sky_tint[:, :, 2] = 40 * gradient * sky_mask  # Red
            sky_tint[:, :, 1] = 10 * gradient * sky_mask  # Green
            sky_tint[:, :, 0] = -20 * gradient * sky_mask  # Blue (reduce)

            # Global color shift
            result[:, :, 2] *= 1.15  # Red boost
            result[:, :, 1] *= 1.02  # Green slight
            result[:, :, 0] *= 0.82  # Blue reduce

        elif style == 'blue':
            # Cool blue hour
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)
            sky_tint[:, :, 0] = 30 * (1 - gradient) * sky_mask  # Blue
            sky_tint[:, :, 2] = -15 * (1 - gradient) * sky_mask  # Red reduce

            result[:, :, 0] *= 1.12  # Blue boost
            result[:, :, 2] *= 0.88  # Red reduce

        elif style == 'orange':
            # Golden hour warmth
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)
            sky_tint[:, :, 2] = 50 * gradient * sky_mask  # Red
            sky_tint[:, :, 1] = 25 * gradient * sky_mask  # Green
            sky_tint[:, :, 0] = -30 * gradient * sky_mask  # Blue reduce

            result[:, :, 2] *= 1.25  # Red boost
            result[:, :, 1] *= 1.08  # Green
            result[:, :, 0] *= 0.70  # Blue reduce
        else:
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)

        # Apply sky tint
        result += sky_tint

        # Darken for dusk (use LAB to preserve colors)
        lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] *= 0.88  # Reduce luminance
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        # Window glow - detect bright areas and add warm glow
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Expand bright areas to create glow
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        glow_mask = cv2.dilate(bright, kernel, iterations=2)
        glow_mask = cv2.GaussianBlur(glow_mask.astype(np.float32), (41, 41), 0) / 255.0

        # Warm glow color
        result[:, :, 2] += glow_mask * 45  # Red
        result[:, :, 1] += glow_mask * 30  # Green
        result[:, :, 0] += glow_mask * 5   # Slight blue

        # Subtle vignette for atmosphere
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (dist / max_dist) * 0.25
        vignette = vignette.astype(np.float32)

        for i in range(3):
            result[:, :, i] *= vignette

        return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================
# CLI
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='AutoHDR Clone Processor')
    parser.add_argument('--input', '-i', required=True, help='Input image')
    parser.add_argument('--output', '-o', required=True, help='Output image')
    parser.add_argument('--brightness', type=float, default=0, help='Brightness (-2 to +2)')
    parser.add_argument('--contrast', type=float, default=0, help='Contrast (-2 to +2)')
    parser.add_argument('--vibrance', type=float, default=0, help='Vibrance (-2 to +2)')
    parser.add_argument('--twilight', choices=['pink', 'blue', 'orange'], help='Twilight style')

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load {args.input}")
        return

    # Configure settings
    settings = ProcessingSettings(
        brightness=args.brightness,
        contrast=args.contrast,
        vibrance=args.vibrance,
        twilight_style=args.twilight
    )

    # Process
    processor = AutoHDRProcessor(settings)
    result = processor.process(image)

    # Save
    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
