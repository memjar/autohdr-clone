"""
AutoHDR Clone - Professional Processor v3.0
============================================

Implements AutoHDR's complete editing methodology based on technical analysis.

CORE PRINCIPLES (from AutoHDR analysis):
1. Multi-bracket HDR fusion (Mertens) - THE secret sauce
2. Flambient-quality tone mapping
3. Intelligent window pull with exterior recovery
4. Auto white balance correction
5. Enhanced sky replacement with cloud styles
6. Professional twilight conversion
7. Edge-aware processing (no halos)

Key Techniques:
- Exposure fusion from 3-5 brackets (not tone mapping)
- LAB color space for all luminance operations
- Luminosity masking for window pull
- Gradient-based sky detection and replacement
- Realistic twilight with window glow

Usage:
    # Single image processing
    processor = AutoHDRProProcessor(settings)
    result = processor.process(image)

    # Multi-bracket HDR fusion (recommended)
    result = processor.process_brackets([under, normal, over])

Version: 3.0.0
Quality Target: 95%+ AutoHDR match
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Tuple, Union
from pathlib import Path

PROCESSOR_VERSION = "4.0.0"  # Multi-scenario architecture with adaptive parameters


# ============================================================================
# SCENARIO ANALYSIS (AutoHDR's Secret: Specialized Processing Per Scenario)
# ============================================================================

@dataclass
class ZoneAnalysis:
    """Analysis of a single image zone (grid cell)"""
    x: int
    y: int
    width: int
    height: int
    avg_brightness: float
    color_temp: float  # Estimated Kelvin
    is_window: bool
    is_sky: bool
    is_interior: bool
    saturation: float


@dataclass
class ScenarioAnalysis:
    """Complete scenario analysis for adaptive processing"""
    # Detected scenarios
    scenarios: List[str]
    intensity: str  # 'natural' or 'intense'

    # Lighting analysis
    is_mixed_lighting: bool
    color_temp_variance: float
    dominant_color_temp: float
    has_tungsten: bool  # Warm indoor lighting
    has_daylight: bool  # Cool outdoor lighting

    # Dynamic range
    dynamic_range: float
    needs_shadow_lift: bool
    needs_highlight_compression: bool
    is_high_contrast: bool

    # Zone data
    zones: List[ZoneAnalysis]
    interior_zones: List[int]  # Indices of interior zones
    exterior_zones: List[int]  # Indices of exterior/window zones

    # Content detection
    has_windows: bool
    has_sky: bool
    has_grass: bool
    window_percentage: float
    sky_percentage: float

    # Adaptive parameters (computed based on analysis)
    adaptive_wb_strength: float
    adaptive_shadow_lift: float
    adaptive_highlight_compress: float
    adaptive_saturation_boost: float
    adaptive_contrast: float


# ============================================================================
# SETTINGS
# ============================================================================

@dataclass
class ProSettings:
    """Professional processing settings matching AutoHDR capabilities"""

    # Scene type
    scene_type: Literal['interior', 'exterior', 'auto'] = 'auto'

    # Output style (AutoHDR's two modes)
    output_style: Literal['natural', 'intense'] = 'natural'

    # Manual adjustments (-2 to +2 scale)
    brightness: float = 0.0
    contrast: float = 0.0
    vibrance: float = 0.0
    white_balance: float = 0.0  # -2 cool, +2 warm

    # Auto corrections
    auto_white_balance: bool = True
    auto_exposure: bool = True

    # Window enhancement (THE key technique)
    window_pull: Literal['off', 'natural', 'medium', 'strong'] = 'natural'
    recover_exterior: bool = True  # Try to recover detail through windows

    # Sky settings
    sky_mode: Literal['original', 'enhance', 'replace'] = 'enhance'
    cloud_style: Literal['fluffy', 'wispy', 'dramatic', 'clear'] = 'fluffy'
    interior_clouds: bool = True   # Clouds visible through windows
    exterior_clouds: bool = True   # Full sky replacement

    # Twilight conversion
    twilight: Optional[Literal['golden', 'blue', 'pink', 'orange']] = None

    # Special effects
    fire_in_fireplace: bool = False
    tv_replacement: Optional[str] = None  # Path to replacement image
    grass_enhancement: bool = False

    # Cleanup
    sign_removal: bool = False
    declutter: bool = False
    perspective_correction: bool = True

    # Quality settings
    hdr_strength: float = 0.7      # Overall HDR intensity
    local_contrast: float = 0.35   # Clarity/detail
    shadow_recovery: float = 0.4   # Shadow lift
    highlight_protection: float = 0.35  # Highlight compression

    # === OUTPUT & UPSCALING ===
    output_dpi: int = 72           # Default screen DPI
    upscale_for_print: bool = False  # Upscale to 300 DPI for print
    upscale_method: Literal['lanczos', 'cubic', 'super_res'] = 'lanczos'
    target_megapixels: Optional[float] = None  # Target MP (e.g., 24.0 for 24MP)

    # === NEW: Brightness Distribution ===
    brightness_equalization: bool = True   # Even out brightness across image
    equalization_strength: float = 0.5     # How aggressive (0-1)

    # CLAHE (local adaptive histogram equalization)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0          # Contrast limiting (1-4)
    clahe_grid_size: int = 8               # Tile grid size

    # Luminosity zone control (-1 to +1 for each zone)
    zone_shadows: float = 0.0              # Lift/lower shadows (Zone 1-3)
    zone_midtones: float = 0.0             # Adjust midtones (Zone 4-6)
    zone_highlights: float = 0.0           # Adjust highlights (Zone 7-9)

    # Advanced white balance
    wb_method: Literal['gray_world', 'white_patch', 'combined'] = 'combined'

    # === DENOISING (for clean, grain-free output) ===
    denoise: bool = True
    denoise_strength: float = 0.5          # 0-1, higher = more smoothing
    denoise_preserve_detail: bool = True   # Use edge-aware denoising

    # === ADVANCED: Dodge & Burn (from Lightroom research) ===
    auto_dodge_burn: bool = True           # Auto even out room lighting
    dodge_shadows: float = 0.3             # Lift dark areas (0-1)
    burn_highlights: float = 0.15          # Control bright spots (0-1)

    # === ADVANCED: 7-Zone Luminosity System (Kuyper extended) ===
    use_7_zone_system: bool = False        # Advanced zone control
    zone_blacks: float = 0.0               # Zone 1-2 (L < 25)
    zone_deep_shadows: float = 0.0         # Zone 2-3 (L 25-50)
    # zone_shadows already defined          # Zone 3-4 (L 50-100)
    # zone_midtones already defined         # Zone 5 (L 100-155)
    zone_bright_midtones: float = 0.0      # Zone 6 (L 155-200)
    # zone_highlights already defined       # Zone 7-8 (L 200-235)
    zone_whites: float = 0.0               # Zone 9 (L > 235)

    # === MULTI-SCENARIO ARCHITECTURE (AutoHDR's Secret Sauce) ===
    use_adaptive_processing: bool = True   # Enable scenario detection & adaptive params
    zone_grid_size: int = 4                # 4x4 grid for zone analysis
    adaptive_wb: bool = True               # Zone-aware white balance
    mixed_lighting_threshold: float = 500  # Color temp variance for mixed lighting


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class AutoHDRProProcessor:
    """
    Professional HDR processor implementing AutoHDR's methodology.

    The key insight from AutoHDR's approach:
    - They use EXPOSURE FUSION (Mertens), not traditional tone mapping
    - This preserves natural colors and avoids the "HDR look"
    - Combined with intelligent local adjustments for flambient quality
    """

    def __init__(self, settings: Optional[ProSettings] = None):
        self.settings = settings or ProSettings()
        self._scenario_analysis: Optional[ScenarioAnalysis] = None

    # ========================================================================
    # MULTI-SCENARIO DETECTION & ANALYSIS (AutoHDR's Architecture)
    # ========================================================================

    def _analyze_scenario(self, image: np.ndarray) -> ScenarioAnalysis:
        """
        Comprehensive scenario analysis - THE key to AutoHDR's success.

        Analyzes the image to determine:
        1. What type of scene (interior/exterior/mixed)
        2. Lighting conditions (mixed tungsten/daylight)
        3. Dynamic range requirements
        4. Content (windows, sky, grass)
        5. Optimal processing parameters

        This enables ADAPTIVE processing instead of one-size-fits-all.
        """
        h, w = image.shape[:2]
        grid_size = self.settings.zone_grid_size

        # ====== STEP 1: DIVIDE INTO ZONES ======
        zones = self._analyze_zones(image, grid_size)

        # ====== STEP 2: LIGHTING PROFILE ======
        color_temps = [z.color_temp for z in zones]
        color_temp_variance = np.var(color_temps)
        dominant_color_temp = np.median(color_temps)

        # Detect mixed lighting (high variance = tungsten + daylight)
        is_mixed_lighting = color_temp_variance > self.settings.mixed_lighting_threshold

        # Tungsten (warm): < 4000K, Daylight (cool): > 5000K
        has_tungsten = any(z.color_temp < 4000 for z in zones if not z.is_window)
        has_daylight = any(z.color_temp > 5000 for z in zones)

        # ====== STEP 3: DYNAMIC RANGE ANALYSIS ======
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # 5th and 95th percentile for robust range
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        p5 = np.searchsorted(cumsum, total * 0.05)
        p95 = np.searchsorted(cumsum, total * 0.95)

        dynamic_range = p95 - p5
        needs_shadow_lift = p5 < 50
        needs_highlight_compression = p95 > 220
        is_high_contrast = dynamic_range > 150

        # ====== STEP 4: CONTENT DETECTION ======
        interior_zones = [i for i, z in enumerate(zones) if z.is_interior and not z.is_window]
        exterior_zones = [i for i, z in enumerate(zones) if z.is_window or z.is_sky]

        has_windows = any(z.is_window for z in zones)
        has_sky = any(z.is_sky for z in zones)
        has_grass = self._detect_grass_presence(image)

        window_percentage = sum(1 for z in zones if z.is_window) / len(zones)
        sky_percentage = sum(1 for z in zones if z.is_sky) / len(zones)

        # ====== STEP 5: CLASSIFY SCENARIOS ======
        scenarios = []

        if is_mixed_lighting and has_windows:
            scenarios.append('INTERIOR_WITH_WINDOWS')
        elif not has_windows and not has_sky:
            scenarios.append('PURE_INTERIOR')

        if has_sky and sky_percentage > 0.2:
            scenarios.append('LANDSCAPE_EXTERIOR')

        if has_grass:
            scenarios.append('GRASS_ENHANCEMENT')

        if is_high_contrast:
            scenarios.append('HIGH_DYNAMIC_RANGE')

        if needs_shadow_lift and needs_highlight_compression:
            scenarios.append('EXTREME_CONTRAST')

        # Determine intensity based on analysis
        intensity = 'intense' if is_high_contrast or dynamic_range > 120 else 'natural'

        # ====== STEP 6: COMPUTE ADAPTIVE PARAMETERS ======
        # These replace fixed settings with image-specific values

        # White balance strength: higher for mixed lighting
        adaptive_wb_strength = 0.6
        if is_mixed_lighting:
            adaptive_wb_strength = 0.85 + (color_temp_variance / 5000) * 0.15
        adaptive_wb_strength = min(1.0, adaptive_wb_strength)

        # Shadow lift: based on how dark the shadows are
        adaptive_shadow_lift = 0.3
        if needs_shadow_lift:
            adaptive_shadow_lift = 0.4 + (50 - p5) / 100 * 0.2
        adaptive_shadow_lift = min(0.6, adaptive_shadow_lift)

        # Highlight compression: based on how bright highlights are
        adaptive_highlight_compress = 0.25
        if needs_highlight_compression:
            adaptive_highlight_compress = 0.3 + (p95 - 220) / 35 * 0.15
        adaptive_highlight_compress = min(0.5, adaptive_highlight_compress)

        # Saturation boost: lower saturation images get more boost
        avg_saturation = np.mean([z.saturation for z in zones])
        if avg_saturation < 80:
            adaptive_saturation_boost = 1.15 + (80 - avg_saturation) / 80 * 0.15
        else:
            adaptive_saturation_boost = 1.1
        adaptive_saturation_boost = min(1.3, adaptive_saturation_boost)

        # Contrast: based on dynamic range
        if is_high_contrast:
            adaptive_contrast = 1.0  # Already high, don't add more
        else:
            adaptive_contrast = 1.1 + (150 - dynamic_range) / 150 * 0.15
        adaptive_contrast = min(1.25, adaptive_contrast)

        return ScenarioAnalysis(
            scenarios=scenarios,
            intensity=intensity,
            is_mixed_lighting=is_mixed_lighting,
            color_temp_variance=color_temp_variance,
            dominant_color_temp=dominant_color_temp,
            has_tungsten=has_tungsten,
            has_daylight=has_daylight,
            dynamic_range=dynamic_range,
            needs_shadow_lift=needs_shadow_lift,
            needs_highlight_compression=needs_highlight_compression,
            is_high_contrast=is_high_contrast,
            zones=zones,
            interior_zones=interior_zones,
            exterior_zones=exterior_zones,
            has_windows=has_windows,
            has_sky=has_sky,
            has_grass=has_grass,
            window_percentage=window_percentage,
            sky_percentage=sky_percentage,
            adaptive_wb_strength=adaptive_wb_strength,
            adaptive_shadow_lift=adaptive_shadow_lift,
            adaptive_highlight_compress=adaptive_highlight_compress,
            adaptive_saturation_boost=adaptive_saturation_boost,
            adaptive_contrast=adaptive_contrast
        )

    def _analyze_zones(self, image: np.ndarray, grid_size: int) -> List[ZoneAnalysis]:
        """
        Divide image into zones and analyze each one.

        This enables zone-specific processing:
        - Different white balance for interior vs exterior
        - Different tone mapping per zone
        - Window detection per zone
        """
        h, w = image.shape[:2]
        zone_h = h // grid_size
        zone_w = w // grid_size

        zones = []

        for gy in range(grid_size):
            for gx in range(grid_size):
                y1 = gy * zone_h
                y2 = (gy + 1) * zone_h if gy < grid_size - 1 else h
                x1 = gx * zone_w
                x2 = (gx + 1) * zone_w if gx < grid_size - 1 else w

                region = image[y1:y2, x1:x2]

                # Analyze zone
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)

                # Color temperature estimation (McCamy's formula)
                color_temp = self._estimate_color_temperature(region)

                # HSV for saturation
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])

                # Window detection: very bright region
                is_window = avg_brightness > 200 and saturation < 60

                # Sky detection: bright, blue-ish, low saturation
                blue_ratio = np.mean(region[:, :, 0]) / (np.mean(region[:, :, 2]) + 1)
                is_sky = avg_brightness > 180 and blue_ratio > 1.1 and gy < grid_size // 2

                # Interior: darker, warmer
                is_interior = avg_brightness < 150 and color_temp < 5000

                zones.append(ZoneAnalysis(
                    x=x1, y=y1, width=x2-x1, height=y2-y1,
                    avg_brightness=avg_brightness,
                    color_temp=color_temp,
                    is_window=is_window,
                    is_sky=is_sky,
                    is_interior=is_interior,
                    saturation=saturation
                ))

        return zones

    def _estimate_color_temperature(self, region: np.ndarray) -> float:
        """
        Estimate color temperature in Kelvin using McCamy's formula.

        This is critical for detecting mixed lighting scenarios.
        - Tungsten: ~2700-3200K (warm, orange)
        - Halogen: ~3000-3500K (warm)
        - Daylight: ~5500-6500K (cool, blue)
        - Overcast: ~6500-7500K (cooler)
        """
        # Get average RGB
        avg_b = np.mean(region[:, :, 0])
        avg_g = np.mean(region[:, :, 1])
        avg_r = np.mean(region[:, :, 2])

        # Normalize
        max_channel = max(avg_r, avg_g, avg_b, 1)
        r = avg_r / max_channel
        g = avg_g / max_channel
        b = avg_b / max_channel

        # Prevent division by zero
        if r + g + b < 0.01:
            return 5500  # Default daylight

        # Calculate chromaticity coordinates
        x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / (r + g + b)

        # McCamy's formula for color temperature
        n = (x - 0.3320) / (0.1858 - x + 0.0001)
        color_temp = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

        # Clamp to reasonable range
        return max(2000, min(10000, color_temp))

    def _detect_grass_presence(self, image: np.ndarray) -> bool:
        """Detect if grass/lawn is present (lower portion of image)."""
        h, w = image.shape[:2]
        lower_region = image[int(h * 0.6):, :]

        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)

        # Green detection
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        green_percentage = np.sum(green_mask > 0) / green_mask.size
        return green_percentage > 0.15

    # ========================================================================
    # MAIN PROCESSING PIPELINES
    # ========================================================================

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image with full pipeline.

        For best results, use process_brackets() with multiple exposures.
        """
        # ====== STAGE 0: SCENARIO ANALYSIS (AutoHDR's Secret) ======
        if self.settings.use_adaptive_processing:
            self._scenario_analysis = self._analyze_scenario(image)
            # Use detected intensity if auto
            if self.settings.output_style == 'natural':
                # Let analysis determine if we need more intensity
                pass  # Keep user setting

        # Auto-detect scene type
        if self.settings.scene_type == 'auto':
            scene = self._detect_scene_type(image)
        else:
            scene = self.settings.scene_type

        result = image.copy()

        # ====== STAGE 1: AUTO CORRECTIONS ======
        if self.settings.auto_white_balance:
            result = self._auto_white_balance(result)

        # ====== STAGE 2: HDR TONE MAPPING ======
        # Use adaptive parameters if available
        if self._scenario_analysis and self.settings.use_adaptive_processing:
            strength = self.settings.hdr_strength
            # Adjust based on dynamic range analysis
            if self._scenario_analysis.needs_shadow_lift:
                strength *= 1.1
            if self._scenario_analysis.is_high_contrast:
                strength *= 1.15
        else:
            strength = self.settings.hdr_strength

        if self.settings.output_style == 'intense':
            strength *= 1.4  # More aggressive for intense mode

        result = self._apply_flambient_tone_mapping(result, strength)

        # Intense mode: additional contrast boost
        if self.settings.output_style == 'intense':
            result = self._boost_contrast_intense(result)

        # ====== STAGE 3: WINDOW PULL ======
        if self.settings.window_pull != 'off' and scene == 'interior':
            result = self._professional_window_pull(result)

        # ====== STAGE 4: SKY PROCESSING ======
        if self.settings.sky_mode != 'original':
            result = self._process_sky(result, scene)

        # ====== STAGE 5: BRIGHTNESS EQUALIZATION (NEW) ======
        if self.settings.brightness_equalization:
            result = self._equalize_brightness(result)

        # ====== STAGE 6: CLAHE (NEW) ======
        if self.settings.use_clahe:
            result = self._apply_clahe(result)

        # ====== STAGE 7: AUTO DODGE & BURN (NEW - from Lightroom research) ======
        if self.settings.auto_dodge_burn:
            result = self._auto_dodge_burn(result)

        # ====== STAGE 8: LUMINOSITY ZONE ADJUSTMENTS ======
        if self.settings.use_7_zone_system:
            result = self._apply_7_zone_adjustments(result)
        else:
            result = self._apply_zone_adjustments(result)

        # ====== STAGE 8: MANUAL ADJUSTMENTS ======
        result = self._apply_adjustments(result)

        # ====== STAGE 9: PERSPECTIVE CORRECTION ======
        if self.settings.perspective_correction:
            result = self._correct_perspective(result)

        # ====== STAGE 7: GRASS ENHANCEMENT ======
        if self.settings.grass_enhancement and scene == 'exterior':
            result = self._enhance_grass(result)

        # ====== STAGE 8: SPECIAL EFFECTS ======
        if self.settings.fire_in_fireplace:
            result = self._add_fireplace_fire(result)

        # ====== STAGE 9: CLEANUP ======
        if self.settings.sign_removal:
            result = self._remove_signs(result)
        if self.settings.declutter:
            result = self._declutter(result)

        # ====== STAGE 10: DENOISING (NEW) ======
        if self.settings.denoise:
            result = self._denoise_image(result)

        # ====== STAGE 11: TWILIGHT ======
        if self.settings.twilight:
            result = self._apply_twilight(result, self.settings.twilight)

        # ====== STAGE 12: UPSCALING FOR PRINT ======
        if self.settings.upscale_for_print:
            result = self._upscale_for_print(result)

        return result

    def process_brackets(
        self,
        brackets: List[np.ndarray],
        ev_spacing: float = 2.0
    ) -> np.ndarray:
        """
        Process multiple exposure brackets - THE key AutoHDR technique.

        This is how AutoHDR achieves flambient-quality results:
        - Mertens exposure fusion (not HDR tone mapping)
        - Preserves natural colors
        - No artifacts or halos
        - Professional window/sky handling

        Args:
            brackets: List of 3-5 images at different exposures
                     Order: underexposed -> normal -> overexposed
            ev_spacing: EV difference between brackets (typically 2.0)

        Returns:
            Fused and processed image
        """
        if len(brackets) < 2:
            return self.process(brackets[0])

        # ====== STEP 1: ALIGN BRACKETS ======
        # AutoHDR handles handheld - we do too
        aligned = self._align_brackets(brackets)

        # ====== STEP 2: MERTENS EXPOSURE FUSION ======
        # This is THE secret - not HDR tone mapping, but exposure fusion
        fused = self._mertens_fusion(aligned)

        # ====== STEP 3: EXTRACT WINDOW/SKY FROM DARK BRACKET ======
        # Use underexposed frame for window detail recovery
        window_detail = self._extract_window_detail(aligned[0], fused)

        # ====== STEP 4: BLEND WINDOW DETAIL ======
        result = self._blend_window_recovery(fused, window_detail)

        # ====== STEP 5: CONTINUE WITH STANDARD PIPELINE ======
        # Skip HDR tone mapping since fusion already did it
        original_strength = self.settings.hdr_strength
        self.settings.hdr_strength = 0.3  # Minimal additional processing

        result = self.process(result)

        self.settings.hdr_strength = original_strength

        return result

    # ========================================================================
    # CORE TECHNIQUE: MERTENS EXPOSURE FUSION
    # ========================================================================

    def _align_brackets(self, brackets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align multiple exposures for handheld shooting.
        Uses ECC (Enhanced Correlation Coefficient) alignment.
        """
        if len(brackets) < 2:
            return brackets

        # Use middle bracket as reference
        ref_idx = len(brackets) // 2
        reference = brackets[ref_idx]
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        aligned = []
        for i, img in enumerate(brackets):
            if i == ref_idx:
                aligned.append(img)
                continue

            # Convert to grayscale for alignment
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find transformation matrix
            try:
                # ECC alignment - handles exposure differences
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           100, 1e-6)

                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, gray, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, criteria
                )

                # Apply transformation
                h, w = reference.shape[:2]
                aligned_img = cv2.warpAffine(
                    img, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE
                )
                aligned.append(aligned_img)

            except cv2.error:
                # Alignment failed, use original
                aligned.append(img)

        return aligned

    def _mertens_fusion(self, brackets: List[np.ndarray]) -> np.ndarray:
        """
        Mertens exposure fusion - AutoHDR's core technique.

        Unlike HDR tone mapping, this:
        - Preserves natural colors (no color shifts)
        - Produces no halos
        - Maintains realistic contrast
        - Works directly in display color space

        The algorithm weights pixels by:
        - Contrast (well-defined details)
        - Saturation (colorful areas)
        - Well-exposedness (middle gray preferred)
        """
        # Create Mertens fusion object
        merge_mertens = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=1.0
        )

        # Perform fusion
        fusion = merge_mertens.process(brackets)

        # Convert from [0,1] float to [0,255] uint8
        # Apply slight contrast enhancement during conversion
        fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

        return fusion

    def _extract_window_detail(
        self,
        dark_bracket: np.ndarray,
        fused: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract window/exterior detail from underexposed bracket.

        The dark bracket contains detail in bright areas (windows, sky)
        that may be blown out in the fused result.
        """
        # Detect overexposed regions in fused image
        gray_fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)

        # Regions that are very bright in fused result
        _, bright_mask = cv2.threshold(gray_fused, 240, 255, cv2.THRESH_BINARY)

        # Regions that have detail in dark bracket
        gray_dark = cv2.cvtColor(dark_bracket, cv2.COLOR_BGR2GRAY)

        # Where dark bracket has usable exposure (not too dark)
        dark_usable = (gray_dark > 30) & (gray_dark < 200)

        # Combined mask: bright in fused AND has detail in dark
        recovery_mask = bright_mask.astype(bool) & dark_usable
        recovery_mask = recovery_mask.astype(np.uint8) * 255

        # Feather the mask for smooth blending
        recovery_mask = cv2.GaussianBlur(recovery_mask, (21, 21), 0)

        return dark_bracket, recovery_mask.astype(np.float32) / 255.0

    def _blend_window_recovery(
        self,
        fused: np.ndarray,
        window_detail: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Blend recovered window detail into fused image."""
        dark_bracket, mask = window_detail

        if mask.max() < 0.1:
            return fused

        # Expand mask to 3 channels
        mask_3d = np.stack([mask] * 3, axis=-1)

        # Blend: use dark bracket detail in bright areas
        result = fused.astype(np.float32) * (1 - mask_3d * 0.7) + \
                 dark_bracket.astype(np.float32) * (mask_3d * 0.7)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ========================================================================
    # FLAMBIENT-QUALITY TONE MAPPING
    # ========================================================================

    def _apply_flambient_tone_mapping(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply tone mapping that produces flambient-quality results.

        "Flambient" = Flash + Ambient blend, the gold standard for
        real estate photography. We simulate this look through:

        1. Balanced exposure across frame
        2. Natural shadow fill (like fill flash)
        3. Controlled highlights (like ambient exposure)
        4. Subtle local contrast (like professional lighting)
        """
        # Convert to LAB for luminance-only processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = cv2.split(lab)
        L_norm = L / 255.0

        # ====== SHADOW RECOVERY (simulates fill flash) ======
        # Use adaptive parameters if available
        if self._scenario_analysis and self.settings.use_adaptive_processing:
            shadow_amount = self._scenario_analysis.adaptive_shadow_lift * strength
        else:
            shadow_amount = self.settings.shadow_recovery * strength
        L_norm = self._adaptive_shadow_lift(L_norm, shadow_amount)

        # ====== HIGHLIGHT PROTECTION ======
        if self._scenario_analysis and self.settings.use_adaptive_processing:
            highlight_amount = self._scenario_analysis.adaptive_highlight_compress * strength
        else:
            highlight_amount = self.settings.highlight_protection * strength
        L_norm = self._filmic_highlight_rolloff(L_norm, highlight_amount)

        # ====== LOCAL CONTRAST (simulates directional lighting) ======
        local_amount = self.settings.local_contrast * strength
        L_norm = self._edge_aware_local_contrast(L_norm, local_amount)

        # ====== MIDTONE ENHANCEMENT ======
        L_norm = self._midtone_punch(L_norm, 0.1 * strength)

        # ====== RECONSTRUCT ======
        L_out = np.clip(L_norm * 255, 0, 255).astype(np.float32)

        # Minimal saturation adjustment
        sat_boost = 0.03 * strength
        A = A + (A - 128) * sat_boost
        B = B + (B - 128) * sat_boost

        lab_out = cv2.merge([L_out, A, B])
        result = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    def _adaptive_shadow_lift(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        Lift shadows adaptively like fill flash.
        Darker areas get more boost, with noise prevention.
        """
        if amount <= 0:
            return L

        # Shadow weight: darker = more lift
        shadow_weight = np.power(1.0 - L, 2.5)

        # Soft knee: prevent noise in very dark areas
        soft_knee = np.where(L < 0.08, L / 0.08, 1.0)
        shadow_weight = shadow_weight * soft_knee

        # Apply lift
        lift = shadow_weight * amount * 0.5

        return np.clip(L + lift, 0, 1)

    def _filmic_highlight_rolloff(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        Compress highlights with smooth filmic curve.
        Prevents clipping while maintaining detail.
        """
        if amount <= 0:
            return L

        threshold = 0.65
        mask = L > threshold

        if not mask.any():
            return L

        result = L.copy()

        # Smooth compression above threshold
        over = (L[mask] - threshold) / (1 - threshold)
        compressed = over / (1 + over * amount * 2.5)
        result[mask] = threshold + compressed * (1 - threshold) * (1 - amount * 0.25)

        return result

    def _edge_aware_local_contrast(self, L: np.ndarray, amount: float) -> np.ndarray:
        """
        Multi-scale local contrast with edge protection.
        Creates depth and dimension without halos.
        """
        if amount <= 0:
            return L

        # ====== EDGE DETECTION FOR HALO PREVENTION ======
        L_uint8 = (L * 255).astype(np.uint8)
        edges = cv2.Canny(L_uint8, 25, 90)

        # Create protection zone around edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        edge_zone = cv2.dilate(edges, kernel, iterations=2)
        edge_mask = cv2.GaussianBlur(edge_zone.astype(np.float32), (25, 25), 0) / 255.0

        # Reduce contrast near edges (prevents halos)
        edge_protection = 1.0 - (edge_mask * 0.75)

        # ====== MULTI-SCALE LOCAL CONTRAST ======
        # Fine detail (texture)
        fine = self._unsharp(L, sigma=5, amount=amount * 0.3)

        # Medium detail (local structure)
        medium = self._unsharp(L, sigma=18, amount=amount * 0.45)

        # Coarse detail (overall depth) - apply edge protection
        coarse_raw = self._unsharp(L, sigma=45, amount=amount * 0.55)
        coarse = L + (coarse_raw - L) * edge_protection

        # Combine scales
        result = L + (fine - L) * 0.3 + (medium - L) * 0.35 + (coarse - L) * 0.25

        return np.clip(result, 0, 1)

    def _unsharp(self, img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        """Unsharp mask at specified scale."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return img + (img - blurred) * amount

    def _midtone_punch(self, L: np.ndarray, amount: float) -> np.ndarray:
        """S-curve for midtone contrast."""
        if amount <= 0:
            return L

        centered = L - 0.5
        curve_strength = 1.0 + amount * 4
        curved = np.tanh(centered * curve_strength) / np.tanh(0.5 * curve_strength)
        result = 0.5 + curved * 0.5

        return L * (1 - amount) + result * amount

    def _boost_contrast_intense(self, image: np.ndarray) -> np.ndarray:
        """
        Additional contrast boost for 'intense' output style.
        Adds punch and drama without going overboard.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Boost L channel contrast
        L = lab[:, :, 0]
        L = (L - 128) * 1.15 + 128
        lab[:, :, 0] = np.clip(L, 0, 255)

        # Slight saturation boost
        lab[:, :, 1] = 128 + (lab[:, :, 1] - 128) * 1.08
        lab[:, :, 2] = 128 + (lab[:, :, 2] - 128) * 1.08
        lab = np.clip(lab, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ========================================================================
    # PROFESSIONAL WINDOW PULL
    # ========================================================================

    def _professional_window_pull(self, image: np.ndarray) -> np.ndarray:
        """
        Professional window pull using luminosity masking.

        This is THE technique that separates amateur from pro real estate photos.
        AutoHDR's window pull:
        1. Detects bright rectangular regions (windows)
        2. Creates luminosity mask (brighter = more effect)
        3. Feathers edges for natural blend
        4. Pulls down highlights while preserving some detail
        5. Recovers exterior view where possible
        """
        intensity_map = {
            'natural': 0.45,
            'medium': 0.65,
            'strong': 0.85
        }
        pull_strength = intensity_map.get(self.settings.window_pull, 0.45)

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ====== STEP 1: DETECT WINDOWS ======
        # Multi-threshold for various brightness levels
        _, very_bright = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        _, bright = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        _, medium = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        # Find rectangular contours (window-like shapes)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        window_mask = np.zeros((h, w), dtype=np.float32)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect = cw / max(ch, 1)
            img_area = h * w

            # Window characteristics
            min_area = img_area * 0.0008
            max_area = img_area * 0.35

            if min_area < area < max_area and 0.15 < aspect < 6.0:
                region_brightness = np.mean(gray[y:y+ch, x:x+cw])
                if region_brightness > 165:
                    cv2.rectangle(window_mask, (x, y), (x+cw, y+ch), 1.0, -1)

        if window_mask.sum() < 50:
            return image

        # ====== STEP 2: LUMINOSITY MASK ======
        luminosity = gray.astype(np.float32) / 255.0
        window_lum_mask = window_mask * luminosity

        # ====== STEP 3: FEATHER EDGES ======
        feather = max(17, int(min(h, w) * 0.025))
        if feather % 2 == 0:
            feather += 1
        feathered = cv2.GaussianBlur(window_lum_mask, (feather, feather), 0)
        feathered = np.clip(feathered * 1.4, 0, 1)

        # ====== STEP 4: APPLY PULL IN LAB ======
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Target brightness for windows
        target_L = 175

        # Pull bright areas toward target
        pull_amount = (L - target_L) * feathered * pull_strength
        L_adjusted = L - pull_amount

        # ====== STEP 5: RECOVER EXTERIOR DETAIL ======
        if self.settings.recover_exterior:
            blown = (L > 248).astype(np.float32)
            blown = cv2.GaussianBlur(blown, (7, 7), 0) * feathered

            # Add texture to blown areas
            texture = cv2.Laplacian(gray, cv2.CV_32F)
            texture = np.abs(texture)
            texture = cv2.GaussianBlur(texture, (5, 5), 0)
            L_adjusted = L_adjusted - texture * blown * 0.25

        # ====== STEP 6: PRESERVE COLOR ======
        A, B = lab[:, :, 1], lab[:, :, 2]
        color_boost = 1.0 + feathered * 0.12
        A = 128 + (A - 128) * color_boost
        B = 128 + (B - 128) * color_boost

        lab[:, :, 0] = np.clip(L_adjusted, 0, 255)
        lab[:, :, 1] = np.clip(A, 0, 255)
        lab[:, :, 2] = np.clip(B, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ========================================================================
    # AUTO WHITE BALANCE
    # ========================================================================

    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Automatic white balance correction.
        Uses zone-aware processing for mixed lighting scenarios.
        """
        # Use zone-aware WB if adaptive processing is enabled and we have analysis
        if (self.settings.use_adaptive_processing and
            self.settings.adaptive_wb and
            self._scenario_analysis is not None and
            self._scenario_analysis.is_mixed_lighting):
            return self._zone_aware_white_balance(image)

        # Standard gray world white balance
        return self._gray_world_white_balance(image)

    def _gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Standard gray world white balance."""
        img_float = image.astype(np.float32)

        avg_b = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_r = np.mean(img_float[:, :, 2])

        avg_gray = (avg_b + avg_g + avg_r) / 3

        scale_b = min(max(avg_gray / (avg_b + 1e-6), 0.7), 1.4)
        scale_g = min(max(avg_gray / (avg_g + 1e-6), 0.7), 1.4)
        scale_r = min(max(avg_gray / (avg_r + 1e-6), 0.7), 1.4)

        # Use adaptive strength if available
        strength = 0.6
        if self._scenario_analysis:
            strength = self._scenario_analysis.adaptive_wb_strength

        img_float[:, :, 0] = img_float[:, :, 0] * (1 + (scale_b - 1) * strength)
        img_float[:, :, 1] = img_float[:, :, 1] * (1 + (scale_g - 1) * strength)
        img_float[:, :, 2] = img_float[:, :, 2] * (1 + (scale_r - 1) * strength)

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def _zone_aware_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Zone-aware white balance for mixed lighting scenarios.

        This is CRITICAL for interior photos with windows:
        - Interior zones (warm tungsten lighting): Correct toward neutral
        - Exterior/window zones (cool daylight): Preserve natural color
        - Blend at boundaries for smooth transitions

        AutoHDR does this automatically - it's why their results look natural.
        """
        h, w = image.shape[:2]
        result = image.astype(np.float32)
        zones = self._scenario_analysis.zones
        grid_size = self.settings.zone_grid_size

        # Create correction mask
        correction_mask = np.zeros((h, w, 3), dtype=np.float32)
        weight_mask = np.zeros((h, w), dtype=np.float32)

        for zone in zones:
            # Calculate zone-specific correction
            region = image[zone.y:zone.y+zone.height, zone.x:zone.x+zone.width]

            # Get zone color averages
            avg_b = np.mean(region[:, :, 0])
            avg_g = np.mean(region[:, :, 1])
            avg_r = np.mean(region[:, :, 2])
            avg_gray = (avg_b + avg_g + avg_r) / 3

            # Calculate correction factors
            scale_b = avg_gray / (avg_b + 1e-6)
            scale_g = avg_gray / (avg_g + 1e-6)
            scale_r = avg_gray / (avg_r + 1e-6)

            # Clamp corrections
            scale_b = min(max(scale_b, 0.7), 1.4)
            scale_g = min(max(scale_g, 0.7), 1.4)
            scale_r = min(max(scale_r, 0.7), 1.4)

            # Zone-specific strength based on type
            if zone.is_window or zone.is_sky:
                # Exterior: preserve natural daylight color
                zone_strength = 0.3
            elif zone.is_interior and zone.color_temp < 4000:
                # Warm interior: stronger correction
                zone_strength = 0.85
            else:
                # Default
                zone_strength = 0.6

            # Apply correction to zone with feathering
            y1, y2 = zone.y, zone.y + zone.height
            x1, x2 = zone.x, zone.x + zone.width

            # Create feathered zone mask
            zone_mask = np.ones((zone.height, zone.width), dtype=np.float32)

            # Feather edges for smooth blending
            feather = max(10, min(zone.height, zone.width) // 4)
            if feather > 0:
                # Top edge
                zone_mask[:feather, :] *= np.linspace(0, 1, feather)[:, np.newaxis]
                # Bottom edge
                zone_mask[-feather:, :] *= np.linspace(1, 0, feather)[:, np.newaxis]
                # Left edge
                zone_mask[:, :feather] *= np.linspace(0, 1, feather)[np.newaxis, :]
                # Right edge
                zone_mask[:, -feather:] *= np.linspace(1, 0, feather)[np.newaxis, :]

            # Store corrections
            correction_mask[y1:y2, x1:x2, 0] += (scale_b - 1) * zone_strength * zone_mask[:, :, np.newaxis].squeeze()
            correction_mask[y1:y2, x1:x2, 1] += (scale_g - 1) * zone_strength * zone_mask[:, :, np.newaxis].squeeze()
            correction_mask[y1:y2, x1:x2, 2] += (scale_r - 1) * zone_strength * zone_mask[:, :, np.newaxis].squeeze()
            weight_mask[y1:y2, x1:x2] += zone_mask

        # Normalize by weight
        weight_mask = np.maximum(weight_mask, 1e-6)
        for c in range(3):
            correction_mask[:, :, c] /= weight_mask

        # Apply correction
        result[:, :, 0] *= (1 + correction_mask[:, :, 0])
        result[:, :, 1] *= (1 + correction_mask[:, :, 1])
        result[:, :, 2] *= (1 + correction_mask[:, :, 2])

        return np.clip(result, 0, 255).astype(np.uint8)

    # ========================================================================
    # SKY PROCESSING
    # ========================================================================

    def _process_sky(self, image: np.ndarray, scene: str) -> np.ndarray:
        """Process sky based on mode and cloud style."""
        sky_mask = self._detect_sky_region(image)

        if sky_mask.max() < 0.1:
            return image

        if self.settings.sky_mode == 'enhance':
            return self._enhance_sky(image, sky_mask)
        elif self.settings.sky_mode == 'replace':
            return self._replace_sky(image, sky_mask)

        return image

    def _detect_sky_region(self, image: np.ndarray) -> np.ndarray:
        """Detect sky using color and position analysis."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Blue sky detection
        lower_blue = np.array([85, 15, 100])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # White/overcast sky
        lower_white = np.array([0, 0, 175])
        upper_white = np.array([180, 45, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Combine
        sky_mask = cv2.bitwise_or(mask_blue, mask_white)

        # Weight by vertical position (sky is usually up)
        position_weight = np.linspace(1.0, 0.0, h).reshape(-1, 1)
        position_weight = np.tile(position_weight, (1, w))
        position_weight = (position_weight ** 0.5 * 255).astype(np.uint8)

        sky_mask = cv2.bitwise_and(sky_mask, position_weight)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

        # Smooth edges
        sky_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (15, 15), 0)

        return sky_mask / 255.0

    def _enhance_sky(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhance existing sky color and contrast."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        mask_3d = mask[:, :, np.newaxis]

        style = self.settings.cloud_style

        if style == 'fluffy':
            # Boost saturation and slight brightness
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + mask * 0.25)
            hsv[:, :, 2] = hsv[:, :, 2] * (1 + mask * 0.08)
        elif style == 'dramatic':
            # More saturation, slight darkening
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + mask * 0.4)
            hsv[:, :, 2] = hsv[:, :, 2] * (1 - mask * 0.08)
        elif style == 'wispy':
            # Subtle enhancement
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + mask * 0.15)
            hsv[:, :, 2] = hsv[:, :, 2] * (1 + mask * 0.05)
        # 'clear' - minimal processing

        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _replace_sky(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Replace sky with generated gradient."""
        h, w = image.shape[:2]

        # Create gradient sky based on style
        sky = np.zeros((h, w, 3), dtype=np.float32)

        # Vertical gradient
        gradient = np.linspace(0, 1, h).reshape(-1, 1)
        gradient = np.tile(gradient, (1, w))

        if self.settings.cloud_style == 'fluffy':
            # Light blue gradient
            sky[:, :, 0] = 230 - gradient * 50  # B
            sky[:, :, 1] = 180 - gradient * 40  # G
            sky[:, :, 2] = 140 - gradient * 30  # R
        elif self.settings.cloud_style == 'dramatic':
            # Deeper blue
            sky[:, :, 0] = 200 - gradient * 80
            sky[:, :, 1] = 140 - gradient * 60
            sky[:, :, 2] = 100 - gradient * 40
        elif self.settings.cloud_style == 'wispy':
            # Pale blue
            sky[:, :, 0] = 240 - gradient * 30
            sky[:, :, 1] = 210 - gradient * 30
            sky[:, :, 2] = 180 - gradient * 20
        else:  # clear
            sky[:, :, 0] = 245 - gradient * 40
            sky[:, :, 1] = 220 - gradient * 30
            sky[:, :, 2] = 200 - gradient * 20

        # Blend
        mask_3d = mask[:, :, np.newaxis]
        result = image.astype(np.float32) * (1 - mask_3d) + sky * mask_3d

        return np.clip(result, 0, 255).astype(np.uint8)

    # ========================================================================
    # TWILIGHT CONVERSION
    # ========================================================================

    def _apply_twilight(self, image: np.ndarray, style: str) -> np.ndarray:
        """
        Convert day to twilight/dusk.

        AutoHDR's twilight is "more realistic than a human" - we aim for that.
        Key elements:
        1. Sky gradient (warm horizon, cool zenith)
        2. Window glow (interior lights)
        3. Overall dusk color grade
        4. Atmospheric vignette
        """
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # Detect sky
        sky_mask = self._detect_sky_region(image)

        # Vertical gradient for sky
        gradient = np.linspace(0.2, 1.0, h).reshape(-1, 1)
        gradient = np.tile(gradient, (1, w)).astype(np.float32)

        # Style-specific color grading
        if style == 'golden' or style == 'orange':
            # Warm golden hour
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)
            sky_tint[:, :, 2] = 55 * gradient * sky_mask  # R
            sky_tint[:, :, 1] = 30 * gradient * sky_mask  # G
            sky_tint[:, :, 0] = -25 * gradient * sky_mask  # B

            result[:, :, 2] *= 1.22
            result[:, :, 1] *= 1.06
            result[:, :, 0] *= 0.72

        elif style == 'pink':
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)
            sky_tint[:, :, 2] = 45 * gradient * sky_mask
            sky_tint[:, :, 1] = 15 * gradient * sky_mask
            sky_tint[:, :, 0] = -15 * gradient * sky_mask

            result[:, :, 2] *= 1.18
            result[:, :, 1] *= 1.02
            result[:, :, 0] *= 0.80

        elif style == 'blue':
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)
            sky_tint[:, :, 0] = 35 * (1 - gradient) * sky_mask
            sky_tint[:, :, 2] = -18 * (1 - gradient) * sky_mask

            result[:, :, 0] *= 1.15
            result[:, :, 2] *= 0.85
        else:
            sky_tint = np.zeros((h, w, 3), dtype=np.float32)

        result += sky_tint

        # Darken for dusk
        lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8),
                          cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] *= 0.85
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        # Window glow
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        glow = cv2.dilate(bright, kernel, iterations=3)
        glow = cv2.GaussianBlur(glow.astype(np.float32), (51, 51), 0) / 255.0

        # Warm glow
        result[:, :, 2] += glow * 50
        result[:, :, 1] += glow * 35
        result[:, :, 0] += glow * 8

        # Vignette
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = (1 - (dist / max_dist) * 0.28).astype(np.float32)

        for i in range(3):
            result[:, :, i] *= vignette

        return np.clip(result, 0, 255).astype(np.uint8)

    # ========================================================================
    # SPECIAL EFFECTS
    # ========================================================================

    def _add_fireplace_fire(self, image: np.ndarray) -> np.ndarray:
        """
        Detect fireplace and add realistic fire effect.
        """
        # Detect dark rectangular regions in lower half (potential fireplaces)
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for dark regions (empty fireplace)
        _, dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Only in lower 2/3 of image
        dark[:int(h * 0.33), :] = 0

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = image.copy()

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect = cw / max(ch, 1)

            # Fireplace-like: wider than tall, reasonable size
            if 0.8 < aspect < 3.0 and 5000 < area < 100000:
                # Add fire glow
                fire_mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(fire_mask, (x + cw//2, y + ch//2),
                           (cw//2, ch//2), 0, 0, 360, 1.0, -1)
                fire_mask = cv2.GaussianBlur(fire_mask, (31, 31), 0)

                # Orange/yellow fire colors
                result[:, :, 2] = np.clip(
                    result[:, :, 2] + fire_mask * 80, 0, 255
                ).astype(np.uint8)
                result[:, :, 1] = np.clip(
                    result[:, :, 1] + fire_mask * 50, 0, 255
                ).astype(np.uint8)
                result[:, :, 0] = np.clip(
                    result[:, :, 0] + fire_mask * 10, 0, 255
                ).astype(np.uint8)

        return result

    # ========================================================================
    # BRIGHTNESS EQUALIZATION (Lightroom-inspired)
    # ========================================================================

    def _equalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Spread brightness evenly across the image.

        Based on Lightroom's approach:
        - Identify dark regions that need lifting
        - Identify bright regions that need control
        - Blend to create even illumination
        - Preserve local contrast and detail

        This simulates professional lighting or flash fill.
        """
        strength = self.settings.equalization_strength

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Calculate local brightness (large blur = regional brightness)
        local_brightness = cv2.GaussianBlur(L, (0, 0), 50)

        # Calculate global target brightness
        global_mean = np.mean(L)
        target = max(global_mean, 130)  # Aim for at least medium brightness

        # Calculate how much each region deviates from target
        deviation = target - local_brightness

        # Create adjustment map (lift dark areas, slightly reduce bright areas)
        # Dark areas (below target): positive adjustment
        # Bright areas (above target): smaller negative adjustment
        adjustment = np.where(
            deviation > 0,
            deviation * strength * 0.7,    # Lift shadows
            deviation * strength * 0.2     # Gentle highlight control
        )

        # Apply with protection for very dark/bright areas
        # Soft knee to prevent noise amplification in shadows
        shadow_protection = np.clip(L / 30, 0, 1)  # 0 at black, 1 at L=30+
        highlight_protection = np.clip((255 - L) / 30, 0, 1)

        adjustment = adjustment * shadow_protection * highlight_protection

        # Apply adjustment
        L_adjusted = L + adjustment

        # Preserve original contrast by blending
        L_final = L * (1 - strength * 0.5) + L_adjusted * (strength * 0.5 + 0.5)

        lab[:, :, 0] = np.clip(L_final, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        This is THE technique for spreading brightness locally:
        - Divides image into tiles
        - Equalizes histogram in each tile
        - Interpolates for smooth transitions
        - Clip limit prevents over-amplification

        Result: Even brightness distribution while preserving local contrast.
        """
        # Convert to LAB (apply CLAHE only to L channel)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.settings.clahe_clip_limit,
            tileGridSize=(self.settings.clahe_grid_size,
                         self.settings.clahe_grid_size)
        )

        # Apply CLAHE
        L_clahe = clahe.apply(L)

        # Blend with original for more natural result
        # Full CLAHE can look artificial
        blend = 0.4  # 40% CLAHE, 60% original
        L_blended = (L.astype(np.float32) * (1 - blend) +
                    L_clahe.astype(np.float32) * blend)

        lab[:, :, 0] = np.clip(L_blended, 0, 255).astype(np.uint8)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _auto_dodge_burn(self, image: np.ndarray) -> np.ndarray:
        """
        Automatic dodge & burn for even room lighting.

        Dodge = lighten dark areas (like fill flash)
        Burn = darken overly bright areas (like exposure control)

        This technique is fundamental to professional interior photography,
        creating the "well-lit room" look without actual lighting setup.
        """
        dodge_strength = self.settings.dodge_shadows
        burn_strength = self.settings.burn_highlights

        if dodge_strength <= 0 and burn_strength <= 0:
            return image

        # Convert to LAB for luminance-only processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Calculate local brightness map (regional lighting analysis)
        local_brightness = cv2.GaussianBlur(L, (0, 0), 40)

        # Global statistics
        global_mean = np.mean(L)
        target_brightness = max(global_mean, 140)  # Aim for well-lit room

        # ====== DODGE: Lift dark areas ======
        if dodge_strength > 0:
            # Identify regions darker than target
            dark_deviation = target_brightness - local_brightness
            dark_regions = np.maximum(dark_deviation, 0)

            # Create dodge mask with smooth falloff
            # More lift for darker areas, less for moderately dark
            dodge_intensity = np.power(dark_regions / 100, 0.8) * dodge_strength

            # Shadow protection: don't amplify noise in very dark areas
            shadow_knee = np.clip(L / 25, 0, 1)
            dodge_intensity *= shadow_knee

            # Apply dodge (lift shadows)
            dodge_amount = dodge_intensity * 40  # Max ~40 L units lift
            L = L + dodge_amount

        # ====== BURN: Control bright spots ======
        if burn_strength > 0:
            # Identify regions brighter than target
            bright_deviation = local_brightness - target_brightness
            bright_regions = np.maximum(bright_deviation, 0)

            # Create burn mask - gentle control, not aggressive
            burn_intensity = np.power(bright_regions / 80, 0.7) * burn_strength

            # Highlight protection: preserve some brightness
            highlight_knee = np.clip((255 - L) / 50, 0, 1)
            burn_intensity *= highlight_knee

            # Apply burn (reduce highlights)
            burn_amount = burn_intensity * 25  # Max ~25 L units reduction
            L = L - burn_amount

        lab[:, :, 0] = np.clip(L, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_7_zone_adjustments(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 7-zone luminosity adjustments (extended Kuyper technique).

        Seven zones for precise control:
        - Zone 1-2: Blacks (L < 25)
        - Zone 2-3: Deep Shadows (L 25-50)
        - Zone 3-4: Shadows (L 50-100)
        - Zone 5: Midtones (L 100-155)
        - Zone 6: Bright Midtones (L 155-200)
        - Zone 7-8: Highlights (L 200-235)
        - Zone 9: Whites (L > 235)

        This provides much finer control than the basic 3-zone system,
        allowing professional-level tonal adjustments.
        """
        # Check if any adjustments needed
        has_adjustments = (
            self.settings.zone_blacks != 0 or
            self.settings.zone_deep_shadows != 0 or
            self.settings.zone_shadows != 0 or
            self.settings.zone_midtones != 0 or
            self.settings.zone_bright_midtones != 0 or
            self.settings.zone_highlights != 0 or
            self.settings.zone_whites != 0
        )

        if not has_adjustments:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Create masks for each zone using Gaussian falloff
        # This creates smooth transitions between zones

        # Zone 1-2: Blacks (center=12, width=15)
        blacks_mask = self._create_zone_mask(L, center=12, width=15)

        # Zone 2-3: Deep Shadows (center=37, width=18)
        deep_shadows_mask = self._create_zone_mask(L, center=37, width=18)

        # Zone 3-4: Shadows (center=75, width=30)
        shadows_mask = self._create_zone_mask(L, center=75, width=30)

        # Zone 5: Midtones (center=127, width=35)
        midtones_mask = self._create_zone_mask(L, center=127, width=35)

        # Zone 6: Bright Midtones (center=177, width=28)
        bright_mids_mask = self._create_zone_mask(L, center=177, width=28)

        # Zone 7-8: Highlights (center=217, width=22)
        highlights_mask = self._create_zone_mask(L, center=217, width=22)

        # Zone 9: Whites (center=245, width=12)
        whites_mask = self._create_zone_mask(L, center=245, width=12)

        # Apply adjustments to each zone
        # Scale: -1 to +1 maps to -25 to +25 L units
        scale = 25

        L_adjusted = L.copy()

        if self.settings.zone_blacks != 0:
            L_adjusted += blacks_mask * self.settings.zone_blacks * scale

        if self.settings.zone_deep_shadows != 0:
            L_adjusted += deep_shadows_mask * self.settings.zone_deep_shadows * scale

        if self.settings.zone_shadows != 0:
            L_adjusted += shadows_mask * self.settings.zone_shadows * scale

        if self.settings.zone_midtones != 0:
            L_adjusted += midtones_mask * self.settings.zone_midtones * scale

        if self.settings.zone_bright_midtones != 0:
            L_adjusted += bright_mids_mask * self.settings.zone_bright_midtones * scale

        if self.settings.zone_highlights != 0:
            L_adjusted += highlights_mask * self.settings.zone_highlights * scale

        if self.settings.zone_whites != 0:
            L_adjusted += whites_mask * self.settings.zone_whites * scale

        lab[:, :, 0] = np.clip(L_adjusted, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _apply_zone_adjustments(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Lightroom-style luminosity zone adjustments.

        Five zones (like Lightroom):
        - Blacks (Zone 1-2): L < 25
        - Shadows (Zone 3-4): L 25-75
        - Midtones (Zone 5): L 75-180
        - Highlights (Zone 6-7): L 180-230
        - Whites (Zone 8-9): L > 230

        This allows separate control of shadows/midtones/highlights.
        """
        # Check if any zone adjustments needed
        if (self.settings.zone_shadows == 0 and
            self.settings.zone_midtones == 0 and
            self.settings.zone_highlights == 0):
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Create luminosity masks for each zone
        # Using smooth transitions (feathered masks)

        # Shadows mask (peaks at L=50, fades at edges)
        shadows_mask = self._create_zone_mask(L, center=50, width=60)

        # Midtones mask (peaks at L=128)
        midtones_mask = self._create_zone_mask(L, center=128, width=80)

        # Highlights mask (peaks at L=200)
        highlights_mask = self._create_zone_mask(L, center=200, width=50)

        # Apply adjustments to each zone
        # Scale: -1 to +1 maps to -30 to +30 L units
        scale = 30

        L_adjusted = L.copy()

        if self.settings.zone_shadows != 0:
            L_adjusted += shadows_mask * self.settings.zone_shadows * scale

        if self.settings.zone_midtones != 0:
            L_adjusted += midtones_mask * self.settings.zone_midtones * scale

        if self.settings.zone_highlights != 0:
            L_adjusted += highlights_mask * self.settings.zone_highlights * scale

        lab[:, :, 0] = np.clip(L_adjusted, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _create_zone_mask(
        self,
        L: np.ndarray,
        center: float,
        width: float
    ) -> np.ndarray:
        """
        Create a smooth luminosity mask centered at a specific L value.

        Uses Gaussian falloff for smooth transitions between zones.
        """
        # Gaussian mask centered at 'center' with 'width' sigma
        mask = np.exp(-((L - center) ** 2) / (2 * width ** 2))
        return mask

    def _advanced_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced white balance using combined methods.

        Methods:
        1. Gray World: Assumes average color should be neutral gray
        2. White Patch: Finds brightest region, assumes it's white
        3. Combined: Weighted blend of both methods

        This produces more accurate results than gray world alone.
        """
        method = self.settings.wb_method

        if method == 'gray_world':
            return self._auto_white_balance(image)

        elif method == 'white_patch':
            return self._white_patch_wb(image)

        else:  # combined
            # Apply both methods with weights
            gray_world = self._auto_white_balance(image)
            white_patch = self._white_patch_wb(image)

            # Blend: 60% gray world, 40% white patch
            result = cv2.addWeighted(gray_world, 0.6, white_patch, 0.4, 0)
            return result

    def _white_patch_wb(self, image: np.ndarray) -> np.ndarray:
        """
        White Patch white balance algorithm.

        Finds the brightest region and assumes it should be white.
        Scales other colors accordingly.
        """
        img_float = image.astype(np.float32)

        # Find the brightest pixels (top 0.5%)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 99.5)
        bright_mask = gray >= threshold

        if bright_mask.sum() < 100:
            return image

        # Get average color of bright region
        bright_b = np.mean(img_float[:, :, 0][bright_mask])
        bright_g = np.mean(img_float[:, :, 1][bright_mask])
        bright_r = np.mean(img_float[:, :, 2][bright_mask])

        # Scale to make bright region white
        max_val = max(bright_b, bright_g, bright_r)

        if max_val < 100:
            return image

        scale_b = min(255 / (bright_b + 1), 1.5)
        scale_g = min(255 / (bright_g + 1), 1.5)
        scale_r = min(255 / (bright_r + 1), 1.5)

        # Apply with reduced strength
        strength = 0.5
        img_float[:, :, 0] *= 1 + (scale_b - 1) * strength
        img_float[:, :, 1] *= 1 + (scale_g - 1) * strength
        img_float[:, :, 2] *= 1 + (scale_r - 1) * strength

        return np.clip(img_float, 0, 255).astype(np.uint8)

    # ========================================================================
    # ADJUSTMENTS
    # ========================================================================

    def _apply_adjustments(self, image: np.ndarray) -> np.ndarray:
        """Apply manual brightness/contrast/vibrance/WB adjustments."""
        result = image

        if self.settings.brightness != 0:
            result = self._adjust_brightness(result, self.settings.brightness)

        if self.settings.contrast != 0:
            result = self._adjust_contrast(result, self.settings.contrast)

        if self.settings.vibrance != 0:
            result = self._adjust_vibrance(result, self.settings.vibrance)

        if self.settings.white_balance != 0:
            result = self._adjust_white_balance(result, self.settings.white_balance)

        return result

    def _adjust_brightness(self, image: np.ndarray, value: float) -> np.ndarray:
        """Brightness in LAB space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] += (value / 2.0) * 28
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _adjust_contrast(self, image: np.ndarray, value: float) -> np.ndarray:
        """Contrast in LAB space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        factor = 1.0 + (value / 3.5)
        lab[:, :, 0] = (lab[:, :, 0] - 128) * factor + 128
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _adjust_vibrance(self, image: np.ndarray, value: float) -> np.ndarray:
        """Vibrance - boost less saturated colors more."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        A, B = lab[:, :, 1], lab[:, :, 2]

        sat = np.sqrt((A - 128) ** 2 + (B - 128) ** 2)
        max_sat = np.max(sat) + 1e-6

        # Use adaptive saturation boost if available
        if self._scenario_analysis and self.settings.use_adaptive_processing and value == 0:
            # Apply adaptive boost automatically
            adaptive_boost = self._scenario_analysis.adaptive_saturation_boost
            boost = adaptive_boost * (1.0 - sat / max_sat * 0.3)
        else:
            boost = 1.0 + (value / 8.0) * (1.0 - sat / max_sat)

        lab[:, :, 1] = 128 + (A - 128) * boost
        lab[:, :, 2] = 128 + (B - 128) * boost
        lab = np.clip(lab, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _adjust_white_balance(self, image: np.ndarray, value: float) -> np.ndarray:
        """White balance in LAB space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] += value * 12
        lab = np.clip(lab, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _detect_scene_type(self, image: np.ndarray) -> str:
        """Auto-detect if scene is interior or exterior."""
        h, w = image.shape[:2]

        # Check upper portion for sky
        upper = image[:int(h * 0.35), :]
        hsv_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

        # Blue sky detection
        lower_blue = np.array([85, 20, 100])
        upper_blue = np.array([135, 255, 255])
        sky_pixels = cv2.inRange(hsv_upper, lower_blue, upper_blue)

        sky_ratio = np.sum(sky_pixels > 0) / sky_pixels.size

        # If significant sky visible, likely exterior
        if sky_ratio > 0.15:
            return 'exterior'

        return 'interior'

    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct vertical line distortion."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=image.shape[0] // 4,
                                maxLineGap=10)

        if lines is None or len(lines) < 2:
            return image

        vertical_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if 70 < abs(angle) < 110:
                    vertical_angles.append(angle)

        if not vertical_angles:
            return image

        avg_angle = np.mean(vertical_angles)
        rotation = 90 - abs(avg_angle) if avg_angle > 0 else -(90 - abs(avg_angle))

        if 0.5 < abs(rotation) < 4:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return image

    def _enhance_grass(self, image: np.ndarray) -> np.ndarray:
        """Make grass more vibrant."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        lower_green = np.array([30, 35, 35])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv.astype(np.uint8), lower_green, upper_green)

        h = image.shape[0]
        mask[:int(h * 0.35), :] = 0

        mask_float = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0) / 255.0

        hsv[:, :, 1] = hsv[:, :, 1] * (1 + mask_float * 0.35)
        hsv[:, :, 0] = hsv[:, :, 0] * (1 - mask_float * 0.08) + 55 * mask_float * 0.08

        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _remove_signs(self, image: np.ndarray) -> np.ndarray:
        """Remove signs using inpainting."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)

            if 800 < area < 60000 and 0.4 < aspect < 4.5:
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        if mask.sum() == 0:
            return image

        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    def _declutter(self, image: np.ndarray) -> np.ndarray:
        """Subtle smoothing to reduce visual clutter."""
        return cv2.bilateralFilter(image, 9, 55, 55)

    def _upscale_for_print(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image to print quality (300 DPI).

        AutoHDR offers print-quality upscaling for professional materials.
        This uses high-quality interpolation to increase resolution while
        maintaining sharpness and detail.

        Methods:
        - lanczos: Best for photographic content (default)
        - cubic: Good balance of speed and quality
        - super_res: AI-enhanced (requires additional model, falls back to lanczos)
        """
        if not self.settings.upscale_for_print:
            return image

        h, w = image.shape[:2]
        current_pixels = h * w

        # Calculate scale factor
        if self.settings.target_megapixels:
            # Scale to target megapixels
            target_pixels = self.settings.target_megapixels * 1_000_000
            scale = np.sqrt(target_pixels / current_pixels)
        else:
            # Scale based on DPI ratio (72 DPI to 300 DPI = 4.17x)
            scale = 300 / max(self.settings.output_dpi, 72)

        if scale <= 1.0:
            return image  # Already at or above target

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Select interpolation method
        method = self.settings.upscale_method

        if method == 'lanczos':
            # Lanczos interpolation - best for photos
            result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        elif method == 'cubic':
            # Bicubic interpolation - faster, still good
            result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Default to lanczos (super_res would require ML model)
            result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Apply subtle sharpening to compensate for upscaling softness
        if scale > 1.5:
            # Unsharp mask for large upscales
            blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
            result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

        return result

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Professional denoising for clean, grain-free output.

        Uses multiple techniques:
        1. Non-local means denoising (best quality)
        2. Bilateral filter for edge preservation
        3. Selective denoising (more in shadows where noise is visible)

        This removes the grain/noise that can appear after HDR processing,
        especially in shadow areas that were lifted.
        """
        strength = self.settings.denoise_strength

        if strength <= 0:
            return image

        # Calculate denoising parameters based on strength
        # h parameter for Non-local means (higher = more denoising)
        h_luminance = 3 + strength * 7  # Range: 3-10
        h_color = 3 + strength * 7

        if self.settings.denoise_preserve_detail:
            # Method 1: Non-local means (best quality, slower)
            # Works in LAB space for better results
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h_luminance,  # h (luminance strength)
                h_color,      # hColor (color strength)
                7,            # templateWindowSize
                21            # searchWindowSize
            )

            # Blend based on luminance - denoise shadows more than highlights
            # Noise is more visible in shadows
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            # Shadow mask: more denoising in dark areas
            shadow_weight = np.power(1.0 - gray, 1.5)
            shadow_weight = cv2.GaussianBlur(shadow_weight, (15, 15), 0)

            # Highlight areas: less denoising to preserve detail
            highlight_weight = 1.0 - shadow_weight * 0.5

            # Create blend mask (3 channel)
            blend_mask = np.stack([
                0.3 + shadow_weight * 0.5,  # More denoising in shadows
                0.3 + shadow_weight * 0.5,
                0.3 + shadow_weight * 0.5
            ], axis=-1)

            # Blend original with denoised
            result = (image.astype(np.float32) * (1 - blend_mask) +
                     denoised.astype(np.float32) * blend_mask)

            result = np.clip(result, 0, 255).astype(np.uint8)

        else:
            # Method 2: Simple bilateral filter (faster, still good)
            d = 9  # Diameter
            sigma_color = 50 + strength * 50  # 50-100
            sigma_space = 50 + strength * 50

            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        # Optional: Light sharpening to recover any lost detail
        if self.settings.denoise_preserve_detail and strength > 0.3:
            # Very subtle unsharp mask
            blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
            result = cv2.addWeighted(result, 1.1, blurred, -0.1, 0)

        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_single(
    image: np.ndarray,
    style: str = 'natural',
    **kwargs
) -> np.ndarray:
    """Quick single-image processing."""
    settings = ProSettings(output_style=style, **kwargs)
    processor = AutoHDRProProcessor(settings)
    return processor.process(image)


def process_brackets(
    brackets: List[np.ndarray],
    style: str = 'natural',
    **kwargs
) -> np.ndarray:
    """Process exposure brackets (recommended for best quality)."""
    settings = ProSettings(output_style=style, **kwargs)
    processor = AutoHDRProProcessor(settings)
    return processor.process_brackets(brackets)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='AutoHDR Clone Pro Processor v3')
    parser.add_argument('--input', '-i', required=True, nargs='+',
                       help='Input image(s). Multiple for bracket fusion.')
    parser.add_argument('--output', '-o', required=True, help='Output image')
    parser.add_argument('--style', choices=['natural', 'intense'], default='natural')
    parser.add_argument('--window-pull', choices=['off', 'natural', 'medium', 'strong'],
                       default='natural')
    parser.add_argument('--twilight', choices=['golden', 'blue', 'pink', 'orange'])
    parser.add_argument('--sky', choices=['original', 'enhance', 'replace'],
                       default='enhance')
    parser.add_argument('--brightness', type=float, default=0)
    parser.add_argument('--contrast', type=float, default=0)
    parser.add_argument('--vibrance', type=float, default=0)

    args = parser.parse_args()

    # Load image(s)
    images = [cv2.imread(p) for p in args.input]
    images = [img for img in images if img is not None]

    if not images:
        print("Error: No valid images found")
        return

    print(f"Loaded {len(images)} image(s)")

    # Configure
    settings = ProSettings(
        output_style=args.style,
        window_pull=args.window_pull,
        twilight=args.twilight,
        sky_mode=args.sky,
        brightness=args.brightness,
        contrast=args.contrast,
        vibrance=args.vibrance
    )

    processor = AutoHDRProProcessor(settings)

    # Process
    if len(images) > 1:
        print("Using bracket fusion...")
        result = processor.process_brackets(images)
    else:
        result = processor.process(images[0])

    # Save
    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
