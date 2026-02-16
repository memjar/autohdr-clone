"""
HDRit Professional Processor v8.0
==================================

Production-grade real estate photo processor calibrated to
professional Lightroom editing values from industry analysis.

Pipeline:
1. QUALITY UPSCALE - Ensure minimum professional resolution
2. SCENE ANALYSIS - Detect noise, color temp, windows, brightness
3. DENOISE - Adaptive bilateral (gentle luma, aggressive chroma)
4. WHITE BALANCE - Scene-adaptive Shade of Gray
5. HDR FUSION - Conservative synthetic brackets + Mertens
6. WINDOW PULL - Detect and recover blown window areas
7. TONE MAP - Single-pass Lightroom-calibrated (no stacking)
8. COLOR - Conservative vibrance (+12%), minimal saturation (+5%)
9. CLARITY - Local contrast on luminance
10. SHARPEN - Luminance-only unsharp mask with edge masking
11. OUTPUT GUARANTEE - Minimum resolution check

Calibrated to:
- Exposure +0.5 to +1.0
- Highlights -50 to -90
- Shadows +40 to +80
- Clarity +15 to +25
- Vibrance +4 to +20
- Saturation 0 to +10
- Sharpening Amount 40, Radius 1.0, Masking 30
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple

PROCESSOR_VERSION = "8.0.0"  # Professional RE - Lightroom Calibrated


@dataclass
class BulletproofSettings:
    """Production settings calibrated to professional real estate editing."""

    # Quality preset
    preset: Literal['natural', 'professional', 'vivid'] = 'professional'

    # Denoise - adaptive based on detected noise, this sets the floor
    denoise_strength: Literal['light', 'medium', 'heavy', 'extreme'] = 'medium'

    # HDR fusion strength (0-1, maps to 50-80% range)
    hdr_strength: float = 0.65

    # Lightroom-calibrated tone values (slider scale, same as LR)
    exposure: float = 0.7          # LR +0.7 (range +0.5 to +1.0)
    highlights: float = -70.0      # LR -70 (range -50 to -90)
    shadows: float = 60.0          # LR +60 (range +40 to +80)
    whites: float = 15.0           # LR +15 (range +10 to +25)
    blacks: float = -25.0          # LR -25 (range -19 to -45)

    # Clarity and color
    clarity: float = 20.0          # LR +20 (range +15 to +25)
    vibrance: float = 12.0         # LR +12 (range +4 to +20)
    saturation: float = 5.0        # LR +5 (range 0 to +10)

    # Sharpening (Lightroom values)
    sharpen: bool = True
    sharpen_amount: float = 40.0   # LR Amount 40
    sharpen_radius: float = 1.0    # LR Radius 1.0
    sharpen_masking: float = 30.0  # LR Masking 30

    # Window pull (real estate specific)
    window_pull: bool = True
    window_pull_strength: float = 0.6

    # Upscaling
    upscale: bool = False
    upscale_factor: float = 1.5

    # Tile processing (for large images)
    tile_size: int = 1024
    tile_overlap: int = 64


class BulletproofProcessor:
    """
    Professional real estate photo processor v8.0.

    Calibrated to match professional Lightroom editing workflows.
    Single-pass tone mapping prevents the stacking artifacts of v7.
    """

    def __init__(self, settings: Optional[BulletproofSettings] = None):
        self.settings = settings or BulletproofSettings()
        self.mertens = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=0.0  # Prevent over-weighting of exposure
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process single image to professional real estate quality."""

        # STAGE 0: Quality upscale
        image = self._quality_upscale(image, min_dimension=2500)

        # STAGE 1: Scene analysis (used by later stages)
        scene = self._analyze_scene(image)

        # STAGE 2: Denoise (TurboProcessor can override _deep_clean)
        clean = self._deep_clean(image)

        # STAGE 3: White balance (scene-adaptive)
        balanced = self._white_balance(clean, scene)

        # STAGE 4: HDR fusion (synthetic brackets)
        hdr = self._hdr_fusion(balanced)

        # STAGE 5: Window pull (detect & recover blown areas)
        if self.settings.window_pull:
            hdr = self._window_pull(hdr, balanced, scene)

        # STAGE 6: Tone map (single-pass, LR-calibrated)
        toned = self._tone_map(hdr)

        # STAGE 7: Color (conservative vibrance + saturation)
        colored = self._color_grade(toned)

        # STAGE 8: Clarity (local contrast)
        detailed = self._apply_clarity(colored)

        # STAGE 9: Sharpen (luminance-only with masking)
        if self.settings.sharpen:
            detailed = self._sharpen(detailed)

        # STAGE 10: Output quality guarantee
        result = self._quality_guarantee(detailed)

        return result

    def process_brackets(self, brackets: List[np.ndarray]) -> np.ndarray:
        """
        Process actual bracket exposures using Mertens fusion.

        v8: Actually merges brackets (v7 just picked the brightest).
        Uses the darkest bracket for window recovery.
        """
        if len(brackets) < 2:
            return self.process(brackets[0])

        # Align brackets
        aligned = self._align_brackets(brackets)

        # Mertens fusion
        merged = self._mertens_fusion(aligned)

        # Scene analysis on merged result
        scene = self._analyze_scene(merged)

        # Window pull using darkest bracket (real data, not synthetic)
        if self.settings.window_pull:
            brightnesses = [float(np.mean(b)) for b in aligned]
            darkest = aligned[int(np.argmin(brightnesses))]
            merged = self._window_pull_from_bracket(merged, darkest, scene)

        # Tone + color pipeline (skip HDR fusion since we already merged)
        balanced = self._white_balance(merged, scene)
        toned = self._tone_map(balanced)
        colored = self._color_grade(toned)
        detailed = self._apply_clarity(colored)
        if self.settings.sharpen:
            detailed = self._sharpen(detailed)
        result = self._quality_guarantee(detailed)

        return result

    # =========================================================================
    # SCENE ANALYSIS
    # =========================================================================

    def _analyze_scene(self, image: np.ndarray) -> dict:
        """
        Analyze scene to adapt processing parameters.

        Returns:
            noise_level: estimated noise (0-100)
            color_temp: 'warm', 'neutral', or 'cool'
            brightness: mean brightness (0-255)
            dynamic_range: p99 - p1 of luminance
            window_mask: feathered mask of blown window areas
            is_interior: whether this appears to be an interior shot
            warmth: R-B channel difference
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Noise estimation from center patch
        center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
        noise_level = self._estimate_noise(center)

        # Brightness
        mean_brightness = float(np.mean(gray))

        # Dynamic range
        p1 = float(np.percentile(gray, 1))
        p99 = float(np.percentile(gray, 99))
        dynamic_range = p99 - p1

        # Color temperature
        b_mean = float(np.mean(image[:, :, 0]))
        r_mean = float(np.mean(image[:, :, 2]))
        warmth = r_mean - b_mean
        if warmth > 15:
            color_temp = 'warm'
        elif warmth < -5:
            color_temp = 'cool'
        else:
            color_temp = 'neutral'

        # Window detection
        window_mask = self._detect_windows(image, gray)

        # Interior detection
        window_fraction = np.sum(window_mask > 0) / (h * w)
        is_interior = window_fraction > 0.02

        return {
            'noise_level': noise_level,
            'color_temp': color_temp,
            'brightness': mean_brightness,
            'dynamic_range': dynamic_range,
            'window_mask': window_mask,
            'is_interior': is_interior,
            'warmth': warmth,
        }

    def _estimate_noise(self, gray_patch: np.ndarray) -> float:
        """
        Estimate noise using median absolute deviation of Laplacian.
        Returns 0-100 scale.
        """
        laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
        sigma = float(np.median(np.abs(laplacian))) / 0.6745
        return min(sigma, 100.0)

    def _detect_windows(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Detect blown-out window regions in interior photos.

        Filters out false positives (white ceilings, walls, floors) using:
        1. Brightness threshold (>210 gray or >245 any channel)
        2. Size bounds (0.5% - 25% of image)
        3. Aspect ratio (0.3 - 4.0)
        4. Ceiling/floor exclusion (wide bright bands at top/bottom of frame)
        5. Edge contrast (real windows have dark frames, ceilings don't)

        Returns a feathered mask for smooth blending.
        """
        h, w = gray.shape

        # Bright areas (210 threshold, raised from 200 to reduce wall false positives)
        _, bright_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

        # Blown-out areas (near 255 in any channel)
        max_channel = np.max(image, axis=2)
        _, blown_mask = cv2.threshold(max_channel, 245, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(bright_mask, blown_mask)

        # Clean up morphologically
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Filter contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        window_mask = np.zeros_like(gray)
        min_area = h * w * 0.005   # At least 0.5% of image
        max_area = h * w * 0.25    # No more than 25% (too big = ceiling/sky)
        ceiling_y = int(h * 0.12)  # Top 12% of frame
        floor_y = int(h * 0.88)    # Bottom 12% of frame
        border_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / (ch + 1e-6)

            if aspect < 0.3 or aspect > 4.0:
                continue

            # CEILING: wide bright band touching top of frame = ceiling, not window
            if y < ceiling_y and cw > w * 0.4:
                continue

            # FLOOR: wide bright band touching bottom of frame = floor
            if (y + ch) > floor_y and cw > w * 0.4:
                continue

            # EDGE CONTRAST: real windows have dark frames/surrounds.
            # Compare brightness inside the contour vs a border ring just outside.
            contour_fill = np.zeros_like(gray)
            cv2.drawContours(contour_fill, [contour], -1, 255, -1)

            dilated = cv2.dilate(contour_fill, border_kernel, iterations=1)
            border_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(contour_fill))

            inner_brightness = float(np.mean(gray[contour_fill > 0]))
            border_pixels = gray[border_ring > 0]

            if len(border_pixels) > 0:
                border_brightness = float(np.mean(border_pixels))
                edge_contrast = inner_brightness - border_brightness

                # Windows have dark frames creating >= 25 brightness difference.
                # Ceilings/walls blend smoothly into surrounding bright areas.
                if edge_contrast < 25:
                    continue

            cv2.drawContours(window_mask, [contour], -1, 255, -1)

        # Feather for smooth blending
        window_mask = cv2.GaussianBlur(window_mask, (31, 31), 0)

        return window_mask

    # =========================================================================
    # DENOISE
    # =========================================================================

    def _deep_clean(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive denoising based on detected noise level.

        Uses bilateral filter chain (fast) in YCrCb space:
        - Gentle on luminance (preserve detail)
        - Aggressive on chroma (eyes less sensitive to color noise)

        TurboProcessor overrides this method for even faster processing.
        """
        # Estimate noise from center of image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
        noise = self._estimate_noise(center)

        # Adapt filter parameters to actual noise level
        if noise < 5:
            d, sigma_c, sigma_s, passes = 5, 25, 25, 1
        elif noise < 15:
            d, sigma_c, sigma_s, passes = 5, 35, 35, 1
        elif noise < 30:
            d, sigma_c, sigma_s, passes = 7, 50, 50, 1
        else:
            d, sigma_c, sigma_s, passes = 9, 65, 65, 2

        # Apply settings multiplier
        strength_mult = {'light': 0.7, 'medium': 1.0, 'heavy': 1.3, 'extreme': 1.6}
        mult = strength_mult.get(self.settings.denoise_strength, 1.0)
        sigma_c = int(sigma_c * mult)
        sigma_s = int(sigma_s * mult)

        # Work in YCrCb for channel-specific processing
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Luminance: gentle (preserve detail)
        for _ in range(passes):
            y = cv2.bilateralFilter(y, d, int(sigma_c * 0.5), sigma_s)

        # Chroma: aggressive (color noise is less visible)
        for _ in range(passes):
            cr = cv2.bilateralFilter(cr, d, sigma_c, sigma_s)
            cb = cv2.bilateralFilter(cb, d, sigma_c, sigma_s)

        ycrcb = cv2.merge([y, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # =========================================================================
    # WHITE BALANCE
    # =========================================================================

    def _white_balance(self, image: np.ndarray, scene: dict) -> np.ndarray:
        """
        Scene-adaptive white balance.

        Base: Shade of Gray (p=6, optimal for RE).
        Then adapts based on scene:
        - Warm interiors: gentle cooling to neutralize
        - Cool scenes: slight warming
        - Neutral: no extra adjustment

        Unlike v7 which forced a fixed cool WB on everything.
        """
        result = self._shade_of_gray_wb(image, p=6)

        warmth = scene['warmth']
        is_interior = scene['is_interior']

        if is_interior and warmth > 20:
            # Warm interior (tungsten/incandescent) - cool it gently
            cool_shift = min((warmth - 10) * 0.15, 8.0)
            b, g, r = cv2.split(result.astype(np.float32))
            r = np.clip(r - cool_shift, 0, 255)
            b = np.clip(b + cool_shift * 0.5, 0, 255)
            result = cv2.merge([b, g, r]).astype(np.uint8)

        elif warmth < -10:
            # Too cool - warm slightly
            warm_shift = min(abs(warmth) * 0.1, 5.0)
            b, g, r = cv2.split(result.astype(np.float32))
            r = np.clip(r + warm_shift, 0, 255)
            b = np.clip(b - warm_shift * 0.5, 0, 255)
            result = cv2.merge([b, g, r]).astype(np.uint8)

        return result

    def _shade_of_gray_wb(self, image: np.ndarray, p: int = 6) -> np.ndarray:
        """
        Shade of Gray white balance (Minkowski norm p=6).
        Generalization of gray world - optimal for real estate at p=6.
        """
        result = image.astype(np.float32)
        b, g, r = cv2.split(result)

        b_power = np.power(b + 1e-6, p).mean() ** (1 / p)
        g_power = np.power(g + 1e-6, p).mean() ** (1 / p)
        r_power = np.power(r + 1e-6, p).mean() ** (1 / p)

        avg = (b_power + g_power + r_power) / 3

        scale_b = np.clip(avg / (b_power + 1e-6), 0.75, 1.4)
        scale_g = np.clip(avg / (g_power + 1e-6), 0.75, 1.4)
        scale_r = np.clip(avg / (r_power + 1e-6), 0.75, 1.4)

        # Apply at 55% strength (natural, not over-corrected)
        strength = 0.55
        result[:, :, 0] = b * (1 + (scale_b - 1) * strength)
        result[:, :, 1] = g * (1 + (scale_g - 1) * strength)
        result[:, :, 2] = r * (1 + (scale_r - 1) * strength)

        return np.clip(result, 0, 255).astype(np.uint8)

    # =========================================================================
    # HDR FUSION
    # =========================================================================

    def _hdr_fusion(self, image: np.ndarray) -> np.ndarray:
        """
        Create HDR effect from single image using synthetic brackets.

        v8: Conservative bracket ratios for natural results.
        Under=0.65x (v7 was 0.5x, too dark), Over=gamma 0.55 (v7 was 0.45*1.2, too bright).
        """
        brackets = self._create_synthetic_brackets(image)

        fusion = self.mertens.process(brackets)
        fusion = np.clip(fusion * 255, 0, 255)

        strength = self.settings.hdr_strength
        original = image.astype(np.float32)
        result = original * (1 - strength) + fusion * strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def _create_synthetic_brackets(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create synthetic exposure brackets from a single image.
        Conservative ratios for natural-looking HDR.
        """
        img_f = image.astype(np.float32) / 255.0

        # Under-exposed (highlight detail recovery)
        under = np.clip(img_f * 0.65, 0, 1)

        # Normal
        normal = img_f

        # Over-exposed (shadow detail) - gentle gamma lift
        over = np.power(np.clip(img_f, 0.001, 1), 0.55)
        over = np.clip(over, 0, 1)

        return [
            (under * 255).astype(np.uint8),
            (normal * 255).astype(np.uint8),
            (over * 255).astype(np.uint8)
        ]

    # =========================================================================
    # WINDOW PULL
    # =========================================================================

    def _window_pull(self, image: np.ndarray, original: np.ndarray,
                     scene: dict) -> np.ndarray:
        """
        Recover detail in blown-out window areas (single image mode).

        Creates a darkened version and blends it into detected window regions.
        This is the #1 differentiator between AI and human RE editors.
        """
        window_mask = scene['window_mask']

        if np.sum(window_mask) < 100:
            return image

        # Create darker version with lifted shadow detail
        dark = (image.astype(np.float32) * 0.45)
        dark_f = dark / 255.0
        dark_f = np.power(np.clip(dark_f, 0.001, 1), 0.7)
        dark = np.clip(dark_f * 255, 0, 255).astype(np.uint8)

        # Feathered blend into window areas
        mask_f = window_mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_f] * 3, axis=-1)

        strength = self.settings.window_pull_strength
        result = image.astype(np.float32) * (1 - mask_3ch * strength) + \
                 dark.astype(np.float32) * mask_3ch * strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def _window_pull_from_bracket(self, merged: np.ndarray,
                                  dark_bracket: np.ndarray,
                                  scene: dict) -> np.ndarray:
        """
        Window pull using actual dark bracket (better than synthetic).
        Used when real exposure brackets are available.
        """
        window_mask = scene['window_mask']

        if np.sum(window_mask) < 100:
            return merged

        if dark_bracket.shape[:2] != merged.shape[:2]:
            dark_bracket = cv2.resize(dark_bracket,
                                      (merged.shape[1], merged.shape[0]))

        mask_f = window_mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_f] * 3, axis=-1)

        # Slightly less strength for real brackets (they have real data)
        strength = self.settings.window_pull_strength * 0.8
        result = merged.astype(np.float32) * (1 - mask_3ch * strength) + \
                 dark_bracket.astype(np.float32) * mask_3ch * strength

        return np.clip(result, 0, 255).astype(np.uint8)

    # =========================================================================
    # TONE MAPPING (Single-pass, Lightroom-calibrated)
    # =========================================================================

    def _tone_map(self, image: np.ndarray) -> np.ndarray:
        """
        Single-pass tone mapping calibrated to Lightroom slider values.

        All adjustments happen in ONE pass on the L channel in LAB space.
        This prevents the stacking artifacts from v7 which had:
        shadow lift + deep shadow lift + S-curve + brightness*1.15 + contrast*1.08

        Adaptive exposure: Dark scenes (mean L < 120) receive progressively
        stronger exposure and shadow lift (up to 2.5x at L=30) so that
        severely underexposed interiors are properly recovered. Images with
        normal brightness (L >= 120) are unaffected.

        Lightroom equivalents applied:
        - Exposure +0.7: global L lift of ~17.5 (scaled adaptively for dark scenes)
        - Highlights -70: pull top 30% down by ~21
        - Shadows +60: lift bottom 35% by ~18 (scaled up to 1.8x for dark scenes)
        - Whites +15: extend white point by ~2.25
        - Blacks -25: deepen black point by ~3.75
        - Gentle S-curve (steepness 1.5, 15% blend)
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)

        # --- ADAPTIVE EXPOSURE for dark scenes ---
        # Normal images (mean L > 120) use settings as-is.
        # Dark images (mean L < 100) get progressively stronger lift.
        # This prevents underexposed interiors from staying too dark.
        mean_l = float(np.mean(l))
        if mean_l < 120:
            # Scale factor: 1.0 at L=120, up to 2.5 at L=30
            # Smooth ramp using inverse proportion
            adaptive_scale = np.clip(1.0 + (120.0 - mean_l) / 60.0, 1.0, 2.5)
        else:
            adaptive_scale = 1.0

        # --- EXPOSURE (global lift) ---
        # LR +1.0 ≈ +25 in L, so +0.7 ≈ +17.5
        exposure_lift = self.settings.exposure * 25.0 * adaptive_scale
        l = l + exposure_lift

        # --- HIGHLIGHTS (recover bright areas) ---
        # Affect pixels above L=170, pull down proportionally
        hl_thresh = 170.0
        hl_mask = np.clip((l - hl_thresh) / (255.0 - hl_thresh), 0, 1)
        hl_pull = (self.settings.highlights / 100.0) * 30.0  # -70 → -21
        l = l + hl_mask * hl_pull

        # --- SHADOWS (lift dark areas) ---
        # Affect pixels below L=90, lift proportionally
        sh_thresh = 90.0
        sh_mask = np.clip((sh_thresh - l) / sh_thresh, 0, 1)
        sh_lift = (self.settings.shadows / 100.0) * 30.0 * min(adaptive_scale, 1.8)  # +60 → +18, capped scale for shadows
        l = l + sh_mask * sh_lift

        # --- WHITES (extend white point) ---
        wh_thresh = 230.0
        wh_mask = np.clip((l - wh_thresh) / (255.0 - wh_thresh + 1e-6), 0, 1)
        wh_push = (self.settings.whites / 100.0) * 15.0  # +15 → +2.25
        l = l + wh_mask * wh_push

        # --- BLACKS (deepen black point) ---
        bk_thresh = 30.0
        bk_mask = np.clip((bk_thresh - l) / bk_thresh, 0, 1)
        bk_pull = (self.settings.blacks / 100.0) * 15.0  # -25 → -3.75
        l = l + bk_mask * bk_pull

        # --- GENTLE S-CURVE (natural contrast) ---
        # v7 used midpoint=0.45, steepness=2.5, 30% blend — too aggressive
        l_norm = np.clip(l / 255.0, 0, 1)
        midpoint = 0.5
        steepness = 1.5
        curved = 1.0 / (1.0 + np.exp(-steepness * (l_norm - midpoint)))
        c_min, c_max = curved.min(), curved.max()
        curved = (curved - c_min) / (c_max - c_min + 1e-6)

        # Blend 15% curve with 85% direct
        l_norm = l_norm * 0.85 + curved * 0.15

        lab[:, :, 0] = np.clip(l_norm * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # =========================================================================
    # COLOR GRADING
    # =========================================================================

    def _color_grade(self, image: np.ndarray) -> np.ndarray:
        """
        Conservative color grading calibrated to professional RE values.

        Vibrance +12 (factor 1.12, not v7's 1.45)
        Saturation +5 (factor 1.05, barely touched)
        No aggressive selective color boosting.
        """
        # Vibrance (boost muted colors, protect saturated ones)
        vibrance_factor = 1.0 + (self.settings.vibrance / 100.0)  # 1.12
        result = self._apply_vibrance(image, vibrance_factor)

        # Saturation (global, very gentle)
        if self.settings.saturation > 0:
            sat_factor = 1.0 + (self.settings.saturation / 100.0)  # 1.05
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result

    def _apply_vibrance(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Smart vibrance - boosts muted colors more, protects saturated."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        sat = hsv[:, :, 1] / 255.0
        boost = 1 + (factor - 1) * (1 - sat)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # =========================================================================
    # CLARITY (Local Contrast)
    # =========================================================================

    def _apply_clarity(self, image: np.ndarray) -> np.ndarray:
        """
        Local contrast on luminance channel only.
        LR Clarity +20 → amount 0.20 unsharp mask with large radius.
        """
        amount = self.settings.clarity / 100.0

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)

        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=25)
        clarity = l + (l - blur) * amount

        lab[:, :, 0] = np.clip(clarity, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # =========================================================================
    # SHARPENING
    # =========================================================================

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Luminance-only sharpening with edge-aware masking.

        LR Amount 40 → 0.40 unsharp mask
        LR Radius 1.0 → sigma 1.0
        LR Masking 30 → protect smooth areas (30th percentile threshold)

        Applied to L channel only to prevent color fringing.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)

        amount = self.settings.sharpen_amount / 100.0
        radius = self.settings.sharpen_radius

        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=radius)
        detail = l - blur

        # Edge-aware masking: only sharpen where there's significant detail
        if self.settings.sharpen_masking > 0:
            masking = self.settings.sharpen_masking / 100.0
            edge_strength = np.abs(detail)
            threshold = np.percentile(edge_strength, masking * 100)
            edge_max = np.max(edge_strength)
            mask = np.clip(
                (edge_strength - threshold) / (edge_max - threshold + 1e-6),
                0, 1
            )
            sharpened = l + detail * amount * mask
        else:
            sharpened = l + detail * amount

        lab[:, :, 0] = np.clip(sharpened, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _quality_upscale(self, image: np.ndarray,
                         min_dimension: int = 2500) -> np.ndarray:
        """Upscale small images to professional resolution."""
        h, w = image.shape[:2]
        max_side = max(h, w)

        if max_side < min_dimension:
            scale = min(min_dimension / max_side, 3.0)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h),
                               interpolation=cv2.INTER_LANCZOS4)
            print(f"   ↑ Quality upscale: {w}x{h} → {new_w}x{new_h}")

        return image

    def _quality_guarantee(self, image: np.ndarray,
                           min_output: int = 2000) -> np.ndarray:
        """Ensure final output meets minimum professional resolution."""
        h, w = image.shape[:2]
        max_side = max(h, w)

        if max_side < min_output:
            scale = min_output / max_side
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_LANCZOS4)

        return image

    def _align_brackets(self, brackets: List[np.ndarray]) -> List[np.ndarray]:
        """Align bracket images using MTB (fast, good for exposure brackets)."""
        if len(brackets) < 2:
            return brackets

        target_shape = brackets[0].shape[:2]
        aligned = []
        for b in brackets:
            if b.shape[:2] != target_shape:
                b = cv2.resize(b, (target_shape[1], target_shape[0]))
            aligned.append(b)

        try:
            align_mtb = cv2.createAlignMTB()
            align_mtb.process(aligned, aligned)
        except Exception:
            pass

        return aligned

    def _mertens_fusion(self, brackets: List[np.ndarray]) -> np.ndarray:
        """Mertens exposure fusion for actual brackets."""
        fusion = self.mertens.process(brackets)
        return np.clip(fusion * 255, 0, 255).astype(np.uint8)

    def _bright_fusion(self, brackets: List[np.ndarray]) -> np.ndarray:
        """
        Enhanced bracket fusion with shadow lift and highlight recovery.
        Maintained for compatibility with external code paths.
        """
        target_shape = brackets[0].shape[:2]
        aligned = []
        for b in brackets:
            if b.shape[:2] != target_shape:
                b = cv2.resize(b, (target_shape[1], target_shape[0]))
            aligned.append(b)

        brightness = [float(np.mean(img)) for img in aligned]
        sorted_indices = np.argsort(brightness)

        darkest = aligned[sorted_indices[0]].astype(np.float32)
        brightest = aligned[sorted_indices[-1]].astype(np.float32)

        mertens_result = self.mertens.process(aligned)
        result = np.clip(mertens_result * 255, 0, 255).astype(np.float32)

        # Shadow lift from bright bracket
        result_gray = cv2.cvtColor(
            result.astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        shadow_mask = np.clip((100 - result_gray) / 80, 0, 1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
        shadow_mask = np.stack([shadow_mask] * 3, axis=-1)
        result = result * (1 - shadow_mask * 0.35) + \
                 brightest * shadow_mask * 0.35

        # Highlight recovery from dark bracket
        bright_gray = cv2.cvtColor(
            brightest.astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        highlight_mask = np.clip((bright_gray - 240) / 15, 0, 1)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
        highlight_mask = np.stack([highlight_mask] * 3, axis=-1)
        result = result * (1 - highlight_mask * 0.45) + \
                 darkest * highlight_mask * 0.45

        return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_bulletproof(image: np.ndarray,
                        preset: str = 'professional') -> np.ndarray:
    """Quick professional processing."""
    settings = BulletproofSettings(preset=preset)
    processor = BulletproofProcessor(settings)
    return processor.process(image)
