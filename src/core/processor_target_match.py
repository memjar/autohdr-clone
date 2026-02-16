#!/usr/bin/env python3
"""
HDRit Processor - Target Match Edition
======================================

GOAL: Match James's 6 target images EXACTLY.

Target Metrics (from local analysis):
- Brightness (L): 190.5 (range: 184.7 - 196.6)
- Highlights %: 56.7% (range: 45.3 - 69.9%)
- Shadows %: 0.3% (range: 0.1 - 0.6%)
- Saturation: 28.7 (range: 22.5 - 35.4)
- Warmth (R-B): 16.2 (range: 9.2 - 20.1) - WARM, not cool!

Key Fixes from Gap Analysis:
- Brightness: +48 (142 → 190)
- Highlights: +41% (16% → 57%)
- Shadows: -6% (6.2% → 0.3%)
- Warmth: +9 (7.4 → 16.2)
"""

import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

PROCESSOR_VERSION = "TARGET_MATCH_v1.0"

# TARGET METRICS from James's images
TARGET = {
    "brightness_L": 190.5,
    "highlights_pct": 56.7,
    "shadows_pct": 0.3,
    "saturation": 28.7,
    "warmth": 16.2,
}


@dataclass
class TargetMatchSettings:
    """Settings tuned to match target images."""
    # Brightness boost (L channel multiplier) - TUNED v2
    brightness_mult: float = 1.15  # 15% boost (tuned down from 20%)
    brightness_offset: float = 3  # Minimal offset

    # Shadow lifting - more aggressive to hit 0.3% shadows
    shadow_gamma: float = 0.6  # Gamma < 1 lifts shadows
    shadow_threshold: int = 80  # Pixels below this get lifted

    # Warmth (R-B adjustment) - TUNED
    red_boost: float = 1.02  # +2% red (was 4% - too warm)
    blue_reduce: float = 0.98  # -2% blue (was 4% - too warm)

    # Saturation
    saturation_target: float = 28.7


class TargetMatchProcessor:
    """
    Processor specifically tuned to match James's target images.
    """

    def __init__(self, settings: Optional[TargetMatchSettings] = None):
        self.settings = settings or TargetMatchSettings()
        self.mertens = cv2.createMergeMertens()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image to match target metrics."""
        print(f"[{PROCESSOR_VERSION}] Processing...")

        # Stage 1: Light denoise (don't over-smooth)
        clean = self._light_denoise(image)

        # Stage 2: HDR fusion for detail recovery
        hdr = self._simple_hdr(clean)

        # Stage 3: BRIGHTNESS & SHADOW FIX (main issue)
        bright = self._fix_brightness(hdr)

        # Stage 4: WARMTH FIX (targets are warm, not cool!)
        warm = self._fix_warmth(bright)

        # Stage 5: Saturation adjustment
        final = self._fix_saturation(warm)

        # Stage 6: Verify and report
        self._report_metrics(final)

        return final

    def _light_denoise(self, image: np.ndarray) -> np.ndarray:
        """Light denoise - preserve detail."""
        # Single pass, moderate strength
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y = cv2.fastNlMeansDenoising(y, None, 5, 7, 21)
        cr = cv2.fastNlMeansDenoising(cr, None, 10, 7, 21)
        cb = cv2.fastNlMeansDenoising(cb, None, 10, 7, 21)

        return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    def _simple_hdr(self, image: np.ndarray) -> np.ndarray:
        """Simple HDR fusion from single image."""
        img_f = image.astype(np.float32) / 255.0

        # Create brackets
        under = np.clip(img_f * 0.5, 0, 1)
        normal = img_f
        over = np.clip(np.power(img_f + 0.001, 0.5) * 1.2, 0, 1)

        brackets = [
            (under * 255).astype(np.uint8),
            (normal * 255).astype(np.uint8),
            (over * 255).astype(np.uint8),
        ]

        fusion = self.mertens.process(brackets)
        result = np.clip(fusion * 255, 0, 255).astype(np.uint8)

        # Blend 50% fusion with original for naturalness
        return cv2.addWeighted(image, 0.5, result, 0.5, 0)

    def _fix_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        FIX #1: ADAPTIVE Brightness and Shadow Lifting

        Automatically adjusts based on input brightness to hit target of L=190.5
        """
        s = self.settings
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)

        # ========================================
        # STEP 0: MEASURE INPUT & CALCULATE ADAPTIVE BOOST
        # ========================================
        input_brightness = np.mean(l)
        target_brightness = TARGET["brightness_L"]  # 190.5

        # Calculate how much we need to boost
        brightness_gap = target_brightness - input_brightness

        # Adaptive multiplier: more boost for darker inputs
        if input_brightness > 0:
            adaptive_mult = target_brightness / input_brightness
            # Clamp to reasonable range
            adaptive_mult = np.clip(adaptive_mult, 1.0, 1.6)
        else:
            adaptive_mult = 1.4

        print(f"   Input L: {input_brightness:.1f}, Target: {target_brightness}, Adaptive mult: {adaptive_mult:.2f}")

        # ========================================
        # STEP 1: Shadow lift (gamma curve)
        # ========================================
        l_norm = l / 255.0

        # Gamma lift on shadows - AGGRESSIVE
        shadow_mask = np.clip(1 - l_norm, 0, 1) ** 1.5  # Less steep falloff
        gamma_lifted = np.power(l_norm + 0.001, s.shadow_gamma)
        l_norm = l_norm * (1 - shadow_mask * 0.85) + gamma_lifted * shadow_mask * 0.85  # Stronger blend

        # ========================================
        # STEP 2: ADAPTIVE brightness boost
        # ========================================
        l_norm = l_norm * adaptive_mult

        # ========================================
        # STEP 3: Fine-tune to hit target (respect max)
        # ========================================
        current_mean = np.mean(l_norm) * 255
        max_brightness = 196.6  # Max from target range

        if current_mean > 0:
            # Don't overshoot max
            target_clamped = min(target_brightness, max_brightness - 3)
            fine_tune = target_clamped / current_mean
            fine_tune = np.clip(fine_tune, 0.85, 1.08)  # Tighter range
            l_norm = l_norm * fine_tune

        # ========================================
        # STEP 4: GENTLE HIGHLIGHT PUSH
        # Target: 57% pixels above L=200
        # ========================================
        # Gentle push to upper midtones only
        highlight_push = np.exp(-((l_norm - 0.72) ** 2) / 0.12) * 0.06  # Reduced
        l_norm = l_norm + highlight_push

        # ========================================
        # STEP 5: SHADOW CLEANUP
        # Ensure shadows_pct < 0.6% (pixels below L=50)
        # ========================================
        # Find remaining dark pixels and lift them
        l_255 = l_norm * 255
        dark_mask = l_255 < 50
        dark_pct = np.sum(dark_mask) / dark_mask.size * 100

        if dark_pct > 0.6:
            # Lift dark pixels above threshold
            l_255 = np.where(l_255 < 50, 50 + (l_255 / 50) * 20, l_255)
            l_norm = l_255 / 255.0

        # ========================================
        # STEP 6: Final clamp
        # ========================================
        l_final = np.clip(l_norm * 255, 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_final

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _fix_warmth(self, image: np.ndarray) -> np.ndarray:
        """
        FIX #2: Warmth Adjustment

        Current: R-B = 7.4 (too cool/blue)
        Target:  R-B = 16.2 (warm wood tones)

        Need to boost red and/or reduce blue.
        """
        s = self.settings
        b, g, r = cv2.split(image.astype(np.float32))

        # Boost red slightly
        r = r * s.red_boost

        # Reduce blue slightly
        b = b * s.blue_reduce

        # Clamp and merge
        r = np.clip(r, 0, 255)
        b = np.clip(b, 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)

    def _fix_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        FIX #3: Saturation to Target

        Current: varies
        Target: 28.7 (range 22.5-35.4)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Measure current saturation
        current_sat = np.mean(s)

        # Adjust to target
        if current_sat > 0:
            scale = self.settings.saturation_target / current_sat
            # Don't over-correct
            scale = np.clip(scale, 0.7, 1.5)
            s = s * scale

        s = np.clip(s, 0, 255)
        hsv = cv2.merge([h, s, v])

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _report_metrics(self, image: np.ndarray):
        """Report final metrics vs target."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        l = lab[:, :, 0]
        s = hsv[:, :, 1]
        b, g, r = cv2.split(image)

        metrics = {
            "brightness_L": np.mean(l),
            "highlights_pct": np.sum(l >= 200) / l.size * 100,
            "shadows_pct": np.sum(l <= 50) / l.size * 100,
            "saturation": np.mean(s),
            "warmth": np.mean(r.astype(float) - b.astype(float)),
        }

        print("\n📊 OUTPUT METRICS vs TARGET:")
        print("-" * 50)
        for key, val in metrics.items():
            target = TARGET[key]
            diff = val - target
            status = "✅" if abs(diff) < target * 0.15 else "❌"
            print(f"   {key:18s}: {val:6.1f} (target: {target:.1f}, diff: {diff:+.1f}) {status}")


def test_processor():
    """Test with a sample image."""
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python3 processor_target_match.py <image.jpg>")
        return

    input_path = sys.argv[1]
    output_path = "/tmp/result_target_match.jpg"

    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read: {input_path}")
        return

    processor = TargetMatchProcessor()
    result = processor.process(img)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Compare with: python3 ~/.axe/scripts/compare_to_targets.py {output_path}")


if __name__ == "__main__":
    test_processor()
