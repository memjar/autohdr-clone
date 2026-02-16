"""
HDRit Smart Processor - Phase 3
================================

Room-aware, lens-corrected HDR processing pipeline.
Combines all phases into intelligent auto-processing.

Pipeline:
1. EXIF Analysis → Lens profile lookup
2. Room Classification → Editing profile selection
3. Lens Correction → Distortion/vignette fix
4. HDR Processing → Bulletproof processor with room settings
5. Output Polish → Room-specific adjustments

This is the production-grade processor for HDRit.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from pathlib import Path

from processor_bulletproof import BulletproofProcessor, BulletproofSettings
from room_classifier import RoomClassifier, RoomType, RoomProfile, classify_room
from lens_profiles import LensCorrector, LensProfile, LENS_DATABASE, DEFAULT_PROFILE


SMART_PROCESSOR_VERSION = "1.0.0"


@dataclass
class SmartProcessingResult:
    """Result from smart processing with metadata."""
    image: np.ndarray
    room_type: RoomType
    room_confidence: float
    lens_profile: LensProfile
    processing_time_ms: float
    settings_used: Dict


class SmartProcessor:
    """
    Intelligent HDR processor that adapts to each image.

    Features:
    - Auto room detection → optimized editing profile
    - Auto lens correction → from EXIF or database
    - Bulletproof HDR → zero grain, professional output
    - Room-specific polish → perfect for each space type
    """

    def __init__(self, auto_room: bool = True, auto_lens: bool = True):
        """
        Initialize smart processor.

        Args:
            auto_room: Enable automatic room detection
            auto_lens: Enable automatic lens correction
        """
        self.auto_room = auto_room
        self.auto_lens = auto_lens
        self.room_classifier = RoomClassifier()
        self.lens_corrector = None

    def process(
        self,
        image: np.ndarray,
        exif: Optional[Dict] = None,
        room_override: Optional[RoomType] = None,
        lens_override: Optional[str] = None,
    ) -> SmartProcessingResult:
        """
        Process image with smart detection.

        Args:
            image: Input BGR image
            exif: Optional EXIF data dict
            room_override: Force specific room type
            lens_override: Force specific lens profile key

        Returns:
            SmartProcessingResult with processed image and metadata
        """
        import time
        start = time.time()

        # =====================================================
        # STEP 1: Room Classification
        # =====================================================
        if room_override:
            room_type = room_override
            room_confidence = 1.0
            room_profile = self.room_classifier.get_profile(room_type)
        elif self.auto_room:
            room_type, room_profile, room_confidence = classify_room(image)
        else:
            room_type = RoomType.LIVING_ROOM
            room_confidence = 0.0
            room_profile = self.room_classifier.get_profile(room_type)

        # =====================================================
        # STEP 2: Lens Profile Selection
        # =====================================================
        if lens_override:
            lens_profile = LENS_DATABASE.get(lens_override, DEFAULT_PROFILE)
        elif exif and self.auto_lens:
            self.lens_corrector = LensCorrector.from_exif(exif)
            lens_profile = self.lens_corrector.profile
        else:
            lens_profile = DEFAULT_PROFILE
            self.lens_corrector = LensCorrector(lens_profile)

        # =====================================================
        # STEP 3: Lens Correction
        # =====================================================
        if self.auto_lens and self.lens_corrector:
            corrected = self.lens_corrector.correct(image)
        else:
            corrected = image

        # =====================================================
        # STEP 4: HDR Processing with Room-Adapted Settings
        # =====================================================
        bp_settings = self._create_settings_from_profile(room_profile)
        processor = BulletproofProcessor(bp_settings)
        hdr_result = processor.process(corrected)

        # =====================================================
        # STEP 5: Room-Specific Polish
        # =====================================================
        polished = self._room_polish(hdr_result, room_type, room_profile)

        processing_time = (time.time() - start) * 1000

        return SmartProcessingResult(
            image=polished,
            room_type=room_type,
            room_confidence=room_confidence,
            lens_profile=lens_profile,
            processing_time_ms=processing_time,
            settings_used={
                'brightness': room_profile.brightness_adjust,
                'saturation': room_profile.saturation_adjust,
                'clarity': room_profile.clarity_adjust,
                'color_temp': room_profile.color_temp,
            }
        )

    def _create_settings_from_profile(self, profile: RoomProfile) -> BulletproofSettings:
        """Convert room profile to bulletproof settings."""
        # Map room profile adjustments to processor settings
        return BulletproofSettings(
            preset='professional',
            denoise_strength='heavy',
            hdr_strength=0.6,
            sharpen=False,  # Keep soft like AutoHDR
            exposure=0.7 + profile.brightness_adjust * 0.3,
            clarity=15.0 + profile.clarity_adjust * 10.0,
        )

    def _room_polish(
        self,
        image: np.ndarray,
        room_type: RoomType,
        profile: RoomProfile
    ) -> np.ndarray:
        """Apply room-specific finishing touches."""
        result = image.copy()

        # Color temperature adjustment
        result = self._adjust_color_temp(result, profile.color_temp)

        # Saturation boost for exteriors
        if room_type in [RoomType.FRONT_EXTERIOR, RoomType.BACK_EXTERIOR,
                         RoomType.POOL, RoomType.LANDSCAPE]:
            result = self._boost_saturation(result, profile.saturation_adjust)

        # Extra shadow lift for dark rooms
        if room_type in [RoomType.BASEMENT, RoomType.GARAGE, RoomType.WINE_CELLAR]:
            result = self._lift_shadows(result, profile.shadow_lift + 0.2)

        return result

    def _adjust_color_temp(self, image: np.ndarray, target_kelvin: int) -> np.ndarray:
        """
        Adjust white balance to target color temperature.

        Lower Kelvin = warmer (orange)
        Higher Kelvin = cooler (blue)
        """
        # Reference: 5500K is neutral daylight
        neutral = 5500
        shift = (target_kelvin - neutral) / 2000  # -1 to +1 range

        result = image.astype(np.float32)

        if shift < 0:
            # Warmer - boost red, reduce blue
            result[:, :, 2] *= 1 + abs(shift) * 0.15  # Red
            result[:, :, 0] *= 1 - abs(shift) * 0.10  # Blue
        else:
            # Cooler - boost blue, reduce red
            result[:, :, 0] *= 1 + shift * 0.15  # Blue
            result[:, :, 2] *= 1 - shift * 0.10  # Red

        return np.clip(result, 0, 255).astype(np.uint8)

    def _boost_saturation(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Boost color saturation."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1 + amount
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _lift_shadows(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Lift shadows without blowing highlights."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Shadow mask (darker areas get more lift)
        shadow_mask = 1.0 - np.clip(l_channel / 150, 0, 1)
        l_channel += shadow_mask * amount * 40

        lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def smart_process(
    image: np.ndarray,
    exif: Optional[Dict] = None
) -> np.ndarray:
    """
    Quick smart processing with auto-detection.

    Args:
        image: Input BGR image
        exif: Optional EXIF data

    Returns:
        Processed image
    """
    processor = SmartProcessor()
    result = processor.process(image, exif=exif)
    return result.image


def process_file(
    input_path: str,
    output_path: Optional[str] = None,
    auto_room: bool = True,
    auto_lens: bool = True,
) -> SmartProcessingResult:
    """
    Process image file with smart detection.

    Args:
        input_path: Path to input image
        output_path: Optional output path (default: adds _hdr suffix)
        auto_room: Enable room detection
        auto_lens: Enable lens correction

    Returns:
        SmartProcessingResult
    """
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot load image: {input_path}")

    # TODO: Extract EXIF from file
    exif = None

    # Process
    processor = SmartProcessor(auto_room=auto_room, auto_lens=auto_lens)
    result = processor.process(image, exif=exif)

    # Save output
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_hdr{p.suffix}")

    cv2.imwrite(output_path, result.image)

    return result


if __name__ == "__main__":
    # Test with sample image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_processor.py <image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"Processing: {input_path}")

    result = process_file(input_path)

    print(f"\nResults:")
    print(f"  Room Type: {result.room_type.value}")
    print(f"  Confidence: {result.room_confidence:.0%}")
    print(f"  Lens: {result.lens_profile.name}")
    print(f"  Time: {result.processing_time_ms:.0f}ms")
