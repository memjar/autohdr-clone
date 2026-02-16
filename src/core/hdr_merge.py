"""
HDR Merge Module
================
Merge bracketed exposures into a single HDR image.

Supports:
- Traditional Debevec merging (OpenCV)
- Mertens fusion (no exposure data needed)
- Neural HDR (coming soon)

Usage:
    python -m src.core.hdr_merge --input ./brackets/ --output ./result.jpg
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HDRConfig:
    """Configuration for HDR processing."""
    # Alignment
    align_images: bool = True
    align_method: Literal["ecc", "mtb", "orb"] = "mtb"

    # Merge method
    merge_method: Literal["debevec", "robertson", "mertens"] = "mertens"

    # Tone mapping
    tone_map_method: Literal["reinhard", "drago", "mantiuk", "none"] = "reinhard"

    # Tone mapping params
    gamma: float = 2.2
    saturation: float = 1.0
    contrast: float = 1.0

    # Output
    output_bit_depth: int = 8


class HDRMerger:
    """
    Merge multiple exposure brackets into HDR.

    Example:
        merger = HDRMerger()
        result = merger.merge_brackets(
            images=[img1, img2, img3],
            exposures=[1/250, 1/60, 1/15]  # Optional for Mertens
        )
    """

    def __init__(self, config: Optional[HDRConfig] = None):
        self.config = config or HDRConfig()

    def load_images(self, paths: List[Path]) -> List[np.ndarray]:
        """Load images from file paths."""
        images = []
        for path in sorted(paths):
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            images.append(img)
            logger.info(f"Loaded: {path.name} ({img.shape})")
        return images

    def align_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Align bracketed images to handle camera movement."""
        if not self.config.align_images or len(images) < 2:
            return images

        logger.info(f"Aligning {len(images)} images using {self.config.align_method}")

        if self.config.align_method == "mtb":
            # Median Threshold Bitmap - fast and works well for brackets
            alignMTB = cv2.createAlignMTB()
            aligned = images.copy()
            alignMTB.process(aligned, aligned)
            return aligned

        elif self.config.align_method == "ecc":
            # Enhanced Correlation Coefficient - more accurate but slower
            return self._align_ecc(images)

        elif self.config.align_method == "orb":
            # Feature-based alignment using ORB
            return self._align_orb(images)

        return images

    def _align_ecc(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Align using ECC algorithm."""
        reference = images[len(images) // 2]  # Use middle exposure as reference
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        aligned = []
        for i, img in enumerate(images):
            if i == len(images) // 2:
                aligned.append(img)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find transformation
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

            try:
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                )
                aligned_img = cv2.warpAffine(
                    img, warp_matrix, (img.shape[1], img.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                aligned.append(aligned_img)
            except cv2.error:
                logger.warning(f"ECC alignment failed for image {i}, using original")
                aligned.append(img)

        return aligned

    def _align_orb(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Align using ORB feature matching."""
        reference = images[len(images) // 2]
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(500)
        ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        aligned = []
        for i, img in enumerate(images):
            if i == len(images) // 2:
                aligned.append(img)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)

            if desc is None or len(kp) < 4:
                aligned.append(img)
                continue

            matches = bf.match(ref_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            if len(matches) < 4:
                aligned.append(img)
                continue

            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None:
                aligned_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
                aligned.append(aligned_img)
            else:
                aligned.append(img)

        return aligned

    def merge_debevec(
        self,
        images: List[np.ndarray],
        exposures: List[float]
    ) -> np.ndarray:
        """
        Merge using Debevec method (requires exposure times).
        Returns HDR radiance map.
        """
        logger.info("Merging with Debevec method")

        exposures_arr = np.array(exposures, dtype=np.float32)

        # Estimate camera response function
        calibrate = cv2.createCalibrateDebevec()
        response = calibrate.process(images, exposures_arr)

        # Merge to HDR
        merge = cv2.createMergeDebevec()
        hdr = merge.process(images, exposures_arr, response)

        return hdr

    def merge_robertson(
        self,
        images: List[np.ndarray],
        exposures: List[float]
    ) -> np.ndarray:
        """
        Merge using Robertson method (iterative).
        """
        logger.info("Merging with Robertson method")

        exposures_arr = np.array(exposures, dtype=np.float32)

        calibrate = cv2.createCalibrateRobertson()
        response = calibrate.process(images, exposures_arr)

        merge = cv2.createMergeRobertson()
        hdr = merge.process(images, exposures_arr, response)

        return hdr

    def merge_mertens(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Merge using Mertens exposure fusion.
        Does NOT require exposure times - works directly on LDR images.
        Good for real estate where you just want balanced exposure.
        """
        logger.info("Merging with Mertens fusion (no exposure data needed)")

        merge = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=1.0
        )
        fusion = merge.process(images)

        # Mertens returns float [0,1], convert to uint8
        fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

        return fusion

    def tone_map(self, hdr: np.ndarray) -> np.ndarray:
        """Apply tone mapping to HDR radiance map."""
        method = self.config.tone_map_method

        if method == "none":
            # Simple linear scaling
            result = np.clip(hdr * 255, 0, 255).astype(np.uint8)

        elif method == "reinhard":
            logger.info("Tone mapping: Reinhard")
            tonemap = cv2.createTonemapReinhard(
                gamma=self.config.gamma,
                intensity=0.0,
                light_adapt=0.8,
                color_adapt=0.0
            )
            ldr = tonemap.process(hdr)
            result = np.clip(ldr * 255, 0, 255).astype(np.uint8)

        elif method == "drago":
            logger.info("Tone mapping: Drago")
            tonemap = cv2.createTonemapDrago(
                gamma=self.config.gamma,
                saturation=self.config.saturation,
                bias=0.85
            )
            ldr = tonemap.process(hdr)
            result = np.clip(ldr * 255, 0, 255).astype(np.uint8)

        elif method == "mantiuk":
            logger.info("Tone mapping: Mantiuk")
            tonemap = cv2.createTonemapMantiuk(
                gamma=self.config.gamma,
                scale=0.85,
                saturation=self.config.saturation
            )
            ldr = tonemap.process(hdr)
            result = np.clip(ldr * 255, 0, 255).astype(np.uint8)

        else:
            raise ValueError(f"Unknown tone mapping method: {method}")

        return result

    def merge_brackets(
        self,
        images: List[np.ndarray],
        exposures: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Main entry point: merge bracket images into final HDR result.

        Args:
            images: List of bracket images (BGR, uint8)
            exposures: Optional exposure times in seconds. If None, uses Mertens.

        Returns:
            Tone-mapped result (BGR, uint8)
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images for HDR merge")

        # Align
        aligned = self.align_images(images)

        # Merge
        method = self.config.merge_method

        if method == "mertens":
            # Mertens doesn't need exposure data
            result = self.merge_mertens(aligned)

        elif method == "debevec":
            if exposures is None:
                raise ValueError("Debevec method requires exposure times")
            hdr = self.merge_debevec(aligned, exposures)
            result = self.tone_map(hdr)

        elif method == "robertson":
            if exposures is None:
                raise ValueError("Robertson method requires exposure times")
            hdr = self.merge_robertson(aligned, exposures)
            result = self.tone_map(hdr)

        else:
            raise ValueError(f"Unknown merge method: {method}")

        return result


# ============================================
# SMART GROUPING & QUALITY-AWARE BLENDING
# ============================================

@dataclass
class SceneGroup:
    """A group of related images (same scene, different exposures or duplicates)."""
    scene_id: str
    images: List[np.ndarray]
    group_type: str  # "bracket" or "duplicate"
    raw_bytes_list: List[bytes]  # Original file bytes for EXIF extraction


def extract_exif(file_bytes: bytes) -> dict:
    """Extract relevant EXIF data from image bytes."""
    try:
        from PIL import Image
        from PIL.ExifTags import Base as ExifBase
        import io as _io

        img = Image.open(_io.BytesIO(file_bytes))
        exif_data = img.getexif()
        if not exif_data:
            return {}

        result = {}
        # DateTimeOriginal
        dto = exif_data.get(ExifBase.DateTimeOriginal) or exif_data.get(ExifBase.DateTime)
        if dto:
            from datetime import datetime
            try:
                result['timestamp'] = datetime.strptime(str(dto), '%Y:%m:%d %H:%M:%S')
            except (ValueError, TypeError):
                pass

        # Exposure time (shutter speed in seconds)
        et = exif_data.get(ExifBase.ExposureTime)
        if et:
            result['exposure_time'] = float(et)

        # F-number
        fn = exif_data.get(ExifBase.FNumber)
        if fn:
            result['fnumber'] = float(fn)

        # Focal length
        fl = exif_data.get(ExifBase.FocalLength)
        if fl:
            result['focal_length'] = float(fl)

        # ISO
        iso = exif_data.get(ExifBase.ISOSpeedRatings)
        if iso:
            result['iso'] = int(iso)

        return result
    except Exception:
        return {}


def compute_sharpness(image: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def group_by_scene(
    images: List[np.ndarray],
    file_bytes_list: List[bytes],
    time_threshold_sec: float = 30.0,
    ev_threshold: float = 1.0,
) -> List[SceneGroup]:
    """
    Group uploaded images by scene using EXIF data.

    Algorithm:
    1. Extract EXIF timestamps and exposure settings from each image
    2. Sort by timestamp
    3. Cluster by time proximity (< time_threshold_sec = same scene)
    4. Within cluster: if exposure values vary > ev_threshold, it's brackets;
       otherwise, pick the sharpest as "best" and mark as duplicates

    Args:
        images: List of BGR images (np.ndarray)
        file_bytes_list: Original file bytes for EXIF extraction
        time_threshold_sec: Max seconds between photos to be same scene
        ev_threshold: Min EV difference to classify as bracket set

    Returns:
        List of SceneGroup objects
    """
    import math

    # Extract EXIF for each image
    exif_list = [extract_exif(fb) for fb in file_bytes_list]

    # Build index with EXIF data
    indexed = []
    for i, (img, exif) in enumerate(zip(images, exif_list)):
        indexed.append({
            'idx': i,
            'image': img,
            'exif': exif,
            'bytes': file_bytes_list[i],
            'timestamp': exif.get('timestamp'),
            'exposure_time': exif.get('exposure_time'),
            'fnumber': exif.get('fnumber'),
            'focal_length': exif.get('focal_length'),
        })

    # Sort by timestamp if available, otherwise keep original order
    has_timestamps = any(item['timestamp'] is not None for item in indexed)
    if has_timestamps:
        # Items without timestamps go to the end
        with_ts = [x for x in indexed if x['timestamp'] is not None]
        without_ts = [x for x in indexed if x['timestamp'] is None]
        with_ts.sort(key=lambda x: x['timestamp'])
        indexed = with_ts + without_ts

    # Cluster by time proximity
    clusters = []
    current_cluster = [indexed[0]] if indexed else []

    for item in indexed[1:]:
        prev = current_cluster[-1]
        if (item['timestamp'] is not None and prev['timestamp'] is not None):
            gap = abs((item['timestamp'] - prev['timestamp']).total_seconds())
            if gap <= time_threshold_sec:
                current_cluster.append(item)
                continue
        elif not has_timestamps:
            # No timestamps at all — treat all as one group
            current_cluster.append(item)
            continue

        # New cluster
        clusters.append(current_cluster)
        current_cluster = [item]

    if current_cluster:
        clusters.append(current_cluster)

    # Classify each cluster as bracket or duplicate
    groups = []
    for ci, cluster in enumerate(clusters):
        if len(cluster) == 1:
            # Single image — not a group, but return it for consistency
            groups.append(SceneGroup(
                scene_id=f"scene_{ci+1}",
                images=[cluster[0]['image']],
                group_type="single",
                raw_bytes_list=[cluster[0]['bytes']],
            ))
            continue

        # Check if exposure values vary enough for brackets
        ev_values = []
        for item in cluster:
            et = item['exposure_time']
            fn = item['fnumber']
            iso = item['exif'].get('iso', 100)
            if et and fn:
                # Compute EV: EV = log2(f^2 / t) - log2(ISO/100)
                ev = math.log2((fn ** 2) / et) - math.log2(iso / 100)
                ev_values.append(ev)

        if len(ev_values) >= 2:
            ev_range = max(ev_values) - min(ev_values)
            group_type = "bracket" if ev_range >= ev_threshold else "duplicate"
        else:
            # Can't determine from EXIF — check brightness as fallback
            brightnesses = []
            for item in cluster:
                gray = cv2.cvtColor(item['image'], cv2.COLOR_BGR2GRAY)
                brightnesses.append(np.mean(gray))
            brightness_range = max(brightnesses) - min(brightnesses)
            group_type = "bracket" if brightness_range > 30 else "duplicate"

        groups.append(SceneGroup(
            scene_id=f"scene_{ci+1}",
            images=[item['image'] for item in cluster],
            group_type=group_type,
            raw_bytes_list=[item['bytes'] for item in cluster],
        ))

    return groups


def pick_sharpest(images: List[np.ndarray]) -> np.ndarray:
    """Pick the sharpest image from a list of duplicates."""
    if len(images) == 1:
        return images[0]
    sharpness_scores = [compute_sharpness(img) for img in images]
    best_idx = int(np.argmax(sharpness_scores))
    logger.info(f"Picked sharpest image (index {best_idx}, score {sharpness_scores[best_idx]:.1f})")
    return images[best_idx]


def merge_with_sharpness_weights(images: List[np.ndarray]) -> np.ndarray:
    """
    Mertens fusion with additional sharpness weighting.

    For each image, computes a sharpness map (Laplacian variance per region).
    The sharpest regions from each image get higher weight in the fusion,
    naturally selecting the "best elements" from each photo.
    """
    if len(images) < 2:
        return images[0] if images else np.zeros((1, 1, 3), dtype=np.uint8)

    # Compute per-image sharpness maps (downsample for speed)
    sharpness_maps = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Laplacian gives local sharpness
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.abs(lap)
        # Smooth to get regional sharpness (not pixel-level noise)
        sharpness = cv2.GaussianBlur(sharpness, (31, 31), 0)
        sharpness_maps.append(sharpness)

    # Normalize sharpness maps to weights that sum to 1
    total = sum(sharpness_maps)
    total = np.maximum(total, 1e-10)  # Avoid division by zero

    # Apply Mertens fusion first (handles exposure/contrast/saturation)
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )
    mertens_result = merge.process(images)
    mertens_result = np.clip(mertens_result * 255, 0, 255).astype(np.uint8)

    # Blend using sharpness weights as a refinement
    # Use sharpness to select detail from the best source per region
    sharpness_blend = np.zeros_like(images[0], dtype=np.float64)
    for img, smap in zip(images, sharpness_maps):
        weight = (smap / total)[:, :, np.newaxis]
        sharpness_blend += img.astype(np.float64) * weight
    sharpness_blend = np.clip(sharpness_blend, 0, 255).astype(np.uint8)

    # Final: 70% Mertens (good exposure balance) + 30% sharpness blend (best detail)
    result = cv2.addWeighted(mertens_result, 0.7, sharpness_blend, 0.3, 0)

    return result


def estimate_exposures(images: List[np.ndarray]) -> List[float]:
    """
    Estimate relative exposures from image brightness.
    Useful when EXIF data is unavailable.
    """
    exposures = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        # Crude estimation: brighter = longer exposure
        # Normalize around middle brightness
        exposure = mean_brightness / 128.0
        exposures.append(exposure)

    # Normalize to reasonable range
    min_exp = min(exposures)
    exposures = [e / min_exp for e in exposures]

    return exposures


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Merge HDR brackets")
    parser.add_argument("--input", "-i", required=True, help="Input directory with brackets")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--method", "-m", default="mertens",
                        choices=["mertens", "debevec", "robertson"],
                        help="Merge method")
    parser.add_argument("--tonemap", "-t", default="reinhard",
                        choices=["reinhard", "drago", "mantiuk", "none"],
                        help="Tone mapping method")
    parser.add_argument("--no-align", action="store_true", help="Skip alignment")

    args = parser.parse_args()

    # Find images
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG")) + \
                      list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
    else:
        image_paths = [input_path]

    if len(image_paths) < 2:
        print(f"Error: Need at least 2 images, found {len(image_paths)}")
        return

    print(f"Found {len(image_paths)} images")

    # Configure
    config = HDRConfig(
        align_images=not args.no_align,
        merge_method=args.method,
        tone_map_method=args.tonemap
    )

    # Process
    merger = HDRMerger(config)
    images = merger.load_images(image_paths)

    # Estimate exposures if needed
    exposures = None
    if args.method in ["debevec", "robertson"]:
        exposures = estimate_exposures(images)
        print(f"Estimated exposures: {exposures}")

    result = merger.merge_brackets(images, exposures)

    # Save
    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
