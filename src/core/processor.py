"""
AutoHDR Clone - Core Image Processor
====================================

Implements the full AutoHDR processing pipeline:
1. HDR Tone Mapping (core effect)
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
        result = self.apply_hdr_effect(result, strength=0.7)

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

    def apply_hdr_effect(self, image: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """
        HDR Tone Mapping - Real Estate Style (v11)
        Based on research: CLAHE + bilateral filtering + improved shadow recovery
        """
        # ==========================================
        # STEP 1: CLAHE ON LUMINANCE (key for floor recovery)
        # ==========================================
        # Convert to LAB, apply CLAHE to L channel only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # CLAHE - aggressive for real estate (very bright shadows needed)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Blend: 75% CLAHE for brighter shadows
        l_channel = cv2.addWeighted(l_channel, 0.25, l_enhanced, 0.75, 0)
        lab[:, :, 0] = l_channel
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        img_float = result.astype(np.float32) / 255.0

        # ==========================================
        # STEP 2: COOL WHITE BALANCE
        # ==========================================
        img_float[:, :, 2] *= 0.96   # Red down
        img_float[:, :, 0] *= 1.025  # Blue up

        luminance = cv2.cvtColor(np.clip(img_float, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)

        # ==========================================
        # STEP 3: ADAPTIVE SHADOW LIFTING
        # ==========================================
        # Stronger lift in darker areas (floor fix)
        shadows = 1.0 - luminance
        shadow_lift = np.power(shadows, 1.2) * 0.45 * strength

        result = img_float.copy()
        for i in range(3):
            result[:, :, i] = result[:, :, i] + shadow_lift

        # ==========================================
        # STEP 4: AUTO LEVELS
        # ==========================================
        p_low = np.percentile(result, 0.2)
        p_high = np.percentile(result, 99.6)
        if p_high - p_low > 0.1:
            result = (result - p_low) / (p_high - p_low)

        # ==========================================
        # STEP 5: GAMMA + BRIGHTNESS (pushed harder)
        # ==========================================
        result = np.clip(result, 0, 1)
        result = np.power(result, 0.76)  # More aggressive gamma
        result = result * 1.14 + 0.025   # Stronger brightness push

        # ==========================================
        # STEP 6: BILATERAL FILTER (edge-aware smoothing)
        # ==========================================
        result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        # Bilateral preserves edges while smoothing tones
        result_uint8 = cv2.bilateralFilter(result_uint8, 5, 40, 40)
        result = result_uint8.astype(np.float32) / 255.0

        # ==========================================
        # STEP 7: LOCAL CONTRAST (clarity)
        # ==========================================
        result = self._local_contrast(result, strength=0.28)

        # ==========================================
        # STEP 8: WHITE PUSH
        # ==========================================
        result_lum = np.mean(result, axis=2)
        white_mask = np.clip((result_lum - 0.72) / 0.28, 0, 1)
        for i in range(3):
            result[:, :, i] = result[:, :, i] + white_mask * 0.07

        # ==========================================
        # STEP 9: COLOR BOOST (LAB + HSV)
        # ==========================================
        result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] = 128 + (lab[:, :, 1] - 128) * 1.30
        lab[:, :, 2] = 128 + (lab[:, :, 2] - 128) * 1.25
        lab = np.clip(lab, 0, 255)
        result_uint8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        result_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result_uint8

    def _local_contrast(self, img_float: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Apply local contrast enhancement (clarity)"""
        # Create blurred version
        blurred = cv2.GaussianBlur(img_float, (0, 0), 30)

        # High-pass = original - blurred
        high_pass = img_float - blurred

        # Add back with strength
        result = img_float + high_pass * strength

        return result

    # ==========================================
    # ADJUSTMENTS
    # ==========================================

    def adjust_brightness(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        Brightness adjustment.
        value: -2 to +2 scale
        """
        adjusted = image.astype(np.float32)
        # Scale value to pixel range (±50 pixels for full range)
        adjusted += (value / 2.0) * 50
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def adjust_contrast(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        Contrast adjustment.
        value: -2 to +2 scale
        Formula: output = (input - 128) * factor + 128
        """
        factor = 1.0 + (value / 4.0)
        adjusted = image.astype(np.float32)
        adjusted = (adjusted - 128) * factor + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def adjust_vibrance(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        Vibrance adjustment - boosts saturation intelligently.
        value: -2 to +2 scale
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Boost saturation, but less for already-saturated colors
        saturation = hsv[:, :, 1]
        # Lower saturation pixels get bigger boost
        boost_factor = 1.0 + (value / 10.0) * (1.0 - saturation / 255.0)
        hsv[:, :, 1] = saturation * boost_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_white_balance(self, image: np.ndarray, value: float) -> np.ndarray:
        """
        White balance adjustment.
        value: -2 (cooler/blue) to +2 (warmer/orange)
        """
        adjusted = image.astype(np.float32)

        if value > 0:  # Warmer
            adjusted[:, :, 2] *= (1.0 + value * 0.12)  # Red boost
            adjusted[:, :, 0] *= (1.0 - value * 0.06)  # Blue reduce
        else:  # Cooler
            adjusted[:, :, 0] *= (1.0 - value * 0.12)  # Blue boost
            adjusted[:, :, 2] *= (1.0 + value * 0.06)  # Red reduce

        return np.clip(adjusted, 0, 255).astype(np.uint8)

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
        Window pull - brighten window areas for balanced interior/exterior.
        """
        intensity_map = {'natural': 1.15, 'medium': 1.3, 'strong': 1.5}
        factor = intensity_map.get(intensity, 1.15)

        # Detect potential window regions (bright rectangles)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to find bright regions
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        window_mask = np.zeros_like(gray)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)

            # Window-like: rectangular, reasonable size
            if area > 500 and 0.3 < aspect < 3.0:
                cv2.rectangle(window_mask, (x, y), (x + w, y + h), 255, -1)

        if window_mask.sum() == 0:
            return image

        # Apply selective darkening to window regions (reveal exterior)
        mask_float = window_mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (21, 21), 0)

        adjusted = image.astype(np.float32)
        # Reduce brightness in overexposed windows to reveal detail
        for i in range(3):
            adjusted[:, :, i] = adjusted[:, :, i] * (1 - mask_float * (1 - 1/factor))

        return np.clip(adjusted, 0, 255).astype(np.uint8)

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
        """Apply day-to-dusk twilight effect."""
        adjusted = image.astype(np.float32)

        if style == 'pink':
            adjusted[:, :, 2] *= 1.25  # Red boost
            adjusted[:, :, 1] *= 1.05  # Green slight
            adjusted[:, :, 0] *= 0.75  # Blue reduce
        elif style == 'blue':
            adjusted[:, :, 0] *= 1.15  # Blue boost
            adjusted[:, :, 2] *= 0.85  # Red reduce
        elif style == 'orange':
            adjusted[:, :, 2] *= 1.35  # Red boost
            adjusted[:, :, 1] *= 1.1   # Green
            adjusted[:, :, 0] *= 0.65  # Blue reduce

        # Darken slightly for dusk effect
        adjusted *= 0.92

        # Add warm glow to bright areas (simulates interior lights)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_float = bright.astype(np.float32) / 255.0
        bright_float = cv2.GaussianBlur(bright_float, (31, 31), 0)

        # Add warm glow
        adjusted[:, :, 2] += bright_float * 30  # Red
        adjusted[:, :, 1] += bright_float * 20  # Green

        return np.clip(adjusted, 0, 255).astype(np.uint8)


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
