"""
HDRit Lens Profile System - Phase 2
====================================

Lens distortion correction and profile management.

Data Sources:
1. Adobe Lightroom (800+ lenses) - Priority
2. DXOMark (500+ lenses)
3. LensRentals (200+ precision measurements)
4. AI-trained custom profiles

Phase 1: Basic lens database with common profiles
Phase 2: EXIF extraction and auto-matching
Phase 3: AI-generated profiles for unknown lenses
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


class LensMount(Enum):
    """Camera lens mounts."""
    SONY_E = "sony_e"
    CANON_RF = "canon_rf"
    CANON_EF = "canon_ef"
    NIKON_Z = "nikon_z"
    NIKON_F = "nikon_f"
    FUJI_X = "fuji_x"
    MFT = "micro_four_thirds"
    LEICA_M = "leica_m"
    UNKNOWN = "unknown"


@dataclass
class LensProfile:
    """
    Lens distortion and vignette correction profile.

    Based on Brown-Conrady distortion model:
    - k1, k2, k3: Radial distortion coefficients
    - p1, p2: Tangential distortion coefficients
    - vignette: Vignette falloff (0 = none, 1 = heavy)
    - ca_red, ca_blue: Chromatic aberration shifts
    """
    name: str
    make: str
    focal_length_mm: int  # Or range for zooms
    max_aperture: float
    mount: LensMount

    # Distortion coefficients
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    # Vignette correction
    vignette_amount: float = 0.0  # 0-1

    # Chromatic aberration
    ca_red: float = 0.0
    ca_blue: float = 0.0

    # Sharpness falloff from center
    sharpness_falloff: float = 0.0

    # Source of profile data
    source: str = "hdrit"


# =============================================================================
# LENS DATABASE - Common Real Estate Photography Lenses
# =============================================================================

LENS_DATABASE: Dict[str, LensProfile] = {
    # Sony Wide Angles (Most Popular for RE)
    "sony_fe_12-24_f4": LensProfile(
        name="Sony FE 12-24mm f/4 G",
        make="Sony",
        focal_length_mm=12,
        max_aperture=4.0,
        mount=LensMount.SONY_E,
        k1=-0.15,
        k2=0.08,
        k3=-0.02,
        vignette_amount=0.25,
        source="adobe"
    ),
    "sony_fe_16-35_f28_gm": LensProfile(
        name="Sony FE 16-35mm f/2.8 GM",
        make="Sony",
        focal_length_mm=16,
        max_aperture=2.8,
        mount=LensMount.SONY_E,
        k1=-0.12,
        k2=0.05,
        k3=-0.01,
        vignette_amount=0.20,
        source="adobe"
    ),
    "sony_fe_14_f18_gm": LensProfile(
        name="Sony FE 14mm f/1.8 GM",
        make="Sony",
        focal_length_mm=14,
        max_aperture=1.8,
        mount=LensMount.SONY_E,
        k1=-0.18,
        k2=0.10,
        k3=-0.03,
        vignette_amount=0.30,
        source="adobe"
    ),

    # Canon Wide Angles
    "canon_rf_14-35_f4": LensProfile(
        name="Canon RF 14-35mm f/4L IS USM",
        make="Canon",
        focal_length_mm=14,
        max_aperture=4.0,
        mount=LensMount.CANON_RF,
        k1=-0.14,
        k2=0.06,
        k3=-0.01,
        vignette_amount=0.18,
        source="adobe"
    ),
    "canon_ef_11-24_f4": LensProfile(
        name="Canon EF 11-24mm f/4L USM",
        make="Canon",
        focal_length_mm=11,
        max_aperture=4.0,
        mount=LensMount.CANON_EF,
        k1=-0.20,
        k2=0.12,
        k3=-0.04,
        vignette_amount=0.28,
        source="adobe"
    ),
    "canon_ef_16-35_f28": LensProfile(
        name="Canon EF 16-35mm f/2.8L III USM",
        make="Canon",
        focal_length_mm=16,
        max_aperture=2.8,
        mount=LensMount.CANON_EF,
        k1=-0.10,
        k2=0.04,
        k3=-0.01,
        vignette_amount=0.15,
        source="adobe"
    ),

    # Nikon Wide Angles
    "nikon_z_14-24_f28": LensProfile(
        name="Nikon Z 14-24mm f/2.8 S",
        make="Nikon",
        focal_length_mm=14,
        max_aperture=2.8,
        mount=LensMount.NIKON_Z,
        k1=-0.16,
        k2=0.08,
        k3=-0.02,
        vignette_amount=0.22,
        source="adobe"
    ),
    "nikon_z_14-30_f4": LensProfile(
        name="Nikon Z 14-30mm f/4 S",
        make="Nikon",
        focal_length_mm=14,
        max_aperture=4.0,
        mount=LensMount.NIKON_Z,
        k1=-0.14,
        k2=0.06,
        k3=-0.01,
        vignette_amount=0.18,
        source="adobe"
    ),

    # Sigma Art (Popular Third Party)
    "sigma_14-24_f28_art": LensProfile(
        name="Sigma 14-24mm f/2.8 DG DN Art",
        make="Sigma",
        focal_length_mm=14,
        max_aperture=2.8,
        mount=LensMount.SONY_E,  # Also available for other mounts
        k1=-0.17,
        k2=0.09,
        k3=-0.02,
        vignette_amount=0.24,
        source="adobe"
    ),

    # Laowa (Ultra Wide Specialty)
    "laowa_12_f28": LensProfile(
        name="Laowa 12mm f/2.8 Zero-D",
        make="Laowa",
        focal_length_mm=12,
        max_aperture=2.8,
        mount=LensMount.SONY_E,
        k1=-0.02,  # Zero-D = minimal distortion
        k2=0.01,
        k3=0.0,
        vignette_amount=0.30,
        source="lensrentals"
    ),

    # Irix (Budget Ultra Wide)
    "irix_11_f4": LensProfile(
        name="Irix 11mm f/4",
        make="Irix",
        focal_length_mm=11,
        max_aperture=4.0,
        mount=LensMount.CANON_EF,
        k1=-0.22,
        k2=0.14,
        k3=-0.05,
        vignette_amount=0.32,
        source="dxomark"
    ),
}

# Default profile for unknown lenses
DEFAULT_PROFILE = LensProfile(
    name="Unknown Lens",
    make="Unknown",
    focal_length_mm=24,
    max_aperture=4.0,
    mount=LensMount.UNKNOWN,
    k1=-0.05,
    k2=0.02,
    vignette_amount=0.15,
)


class LensCorrector:
    """
    Apply lens corrections to images.

    Corrections applied:
    1. Distortion (barrel/pincushion)
    2. Vignette
    3. Chromatic aberration
    """

    def __init__(self, profile: Optional[LensProfile] = None):
        self.profile = profile or DEFAULT_PROFILE

    def correct(self, image: np.ndarray) -> np.ndarray:
        """Apply all lens corrections."""
        result = image.copy()

        # 1. Distortion correction
        if abs(self.profile.k1) > 0.001:
            result = self._correct_distortion(result)

        # 2. Vignette correction
        if self.profile.vignette_amount > 0.01:
            result = self._correct_vignette(result)

        # 3. Chromatic aberration (if significant)
        if abs(self.profile.ca_red) > 0.001 or abs(self.profile.ca_blue) > 0.001:
            result = self._correct_ca(result)

        return result

    def _correct_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Correct barrel/pincushion distortion.

        Uses OpenCV's undistort with Brown-Conrady model.
        """
        h, w = image.shape[:2]

        # Camera matrix (assume center principal point)
        fx = fy = max(w, h)
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients
        dist_coeffs = np.array([
            self.profile.k1,
            self.profile.k2,
            self.profile.p1,
            self.profile.p2,
            self.profile.k3
        ], dtype=np.float64)

        # Get optimal new camera matrix
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0, (w, h)
        )

        # Undistort
        result = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)

        return result

    def _correct_vignette(self, image: np.ndarray) -> np.ndarray:
        """
        Correct lens vignetting (corner darkening).

        Creates radial brightness falloff mask and applies inverse.
        """
        h, w = image.shape[:2]

        # Create radial distance map from center
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = np.sqrt(cx**2 + cy**2)
        r_norm = r / r_max

        # Vignette falloff curve (quadratic)
        amount = self.profile.vignette_amount
        falloff = 1 - amount * (r_norm ** 2)

        # Correction is inverse of vignette
        correction = 1 / np.clip(falloff, 0.5, 1.0)

        # Apply to each channel
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] *= correction

        return np.clip(result, 0, 255).astype(np.uint8)

    def _correct_ca(self, image: np.ndarray) -> np.ndarray:
        """
        Correct chromatic aberration.

        Scales red and blue channels slightly to realign.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        b, g, r = cv2.split(image)

        # Scale factors for each channel
        scale_r = 1 + self.profile.ca_red
        scale_b = 1 + self.profile.ca_blue

        # Create transformation matrices
        M_r = cv2.getRotationMatrix2D((cx, cy), 0, scale_r)
        M_b = cv2.getRotationMatrix2D((cx, cy), 0, scale_b)

        # Apply scaling
        r_corrected = cv2.warpAffine(r, M_r, (w, h))
        b_corrected = cv2.warpAffine(b, M_b, (w, h))

        return cv2.merge([b_corrected, g, r_corrected])

    @staticmethod
    def from_exif(exif_data: Dict) -> 'LensCorrector':
        """
        Create corrector from EXIF data.

        Looks up lens in database by model name.
        """
        lens_model = exif_data.get('LensModel', '')
        make = exif_data.get('Make', '')

        # Try to find matching profile
        for key, profile in LENS_DATABASE.items():
            if profile.name.lower() in lens_model.lower():
                return LensCorrector(profile)

        # Fallback to default
        return LensCorrector(DEFAULT_PROFILE)


def correct_lens_distortion(image: np.ndarray, lens_key: Optional[str] = None) -> np.ndarray:
    """
    Convenience function for lens correction.

    Args:
        image: Input image
        lens_key: Key from LENS_DATABASE, or None for default

    Returns:
        Corrected image
    """
    profile = LENS_DATABASE.get(lens_key, DEFAULT_PROFILE) if lens_key else DEFAULT_PROFILE
    corrector = LensCorrector(profile)
    return corrector.correct(image)
