"""
Test processor v3 with real ARW files
"""
import sys
sys.path.insert(0, '/tmp/autohdr-clone/src')

import cv2
import numpy as np
import rawpy
from pathlib import Path
from core.processor_v3 import AutoHDRProProcessor, ProSettings

# ARW files path
ARW_DIR = Path("/Users/home/Downloads/drive-downlo 001 ad-20260205T191553Z-1-")

def load_arw(path: Path) -> np.ndarray:
    """Load ARW file and convert to BGR"""
    print(f"Loading: {path.name}")
    with rawpy.imread(str(path)) as raw:
        # Post-process with good defaults
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=8
        )
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def main():
    print("=" * 60)
    print("Processing Real ARW Brackets with Processor v3.0")
    print("=" * 60)

    # Load the 3 brackets
    arw_files = sorted(ARW_DIR.glob("*.ARW"))
    print(f"\nFound {len(arw_files)} ARW files:")
    for f in arw_files:
        print(f"  - {f.name}")

    # Load all brackets
    brackets = [load_arw(f) for f in arw_files]
    print(f"\nBracket shapes: {[b.shape for b in brackets]}")

    # Check exposure levels
    for i, b in enumerate(brackets):
        mean_brightness = np.mean(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
        print(f"  Bracket {i+1} mean brightness: {mean_brightness:.1f}")

    # ============================================
    # TEST 1: Single image processing (middle bracket)
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 1: Single Image Processing (middle bracket)")
    print("=" * 60)

    settings_single = ProSettings(
        output_style='natural',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.6,
        local_contrast=0.3,
        shadow_recovery=0.35
    )
    processor = AutoHDRProProcessor(settings_single)

    result_single = processor.process(brackets[1])  # Middle bracket
    cv2.imwrite("/tmp/result_single.jpg", result_single, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("Saved: /tmp/result_single.jpg")

    # ============================================
    # TEST 2: Bracket fusion (THE AutoHDR way)
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 2: Bracket Fusion (Mertens)")
    print("=" * 60)

    settings_fusion = ProSettings(
        output_style='natural',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.5,  # Lower since fusion already does tone mapping
        local_contrast=0.25,
        shadow_recovery=0.3
    )
    processor_fusion = AutoHDRProProcessor(settings_fusion)

    result_fusion = processor_fusion.process_brackets(brackets)
    cv2.imwrite("/tmp/result_fusion.jpg", result_fusion, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("Saved: /tmp/result_fusion.jpg")

    # ============================================
    # TEST 3: Intense mode
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 3: Intense Mode")
    print("=" * 60)

    settings_intense = ProSettings(
        output_style='intense',
        auto_white_balance=True,
        window_pull='medium',
        hdr_strength=0.65,
        local_contrast=0.35,
        shadow_recovery=0.4
    )
    processor_intense = AutoHDRProProcessor(settings_intense)

    result_intense = processor_intense.process_brackets(brackets)
    cv2.imwrite("/tmp/result_intense.jpg", result_intense, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("Saved: /tmp/result_intense.jpg")

    # ============================================
    # TEST 4: Optimized for this specific image
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 4: Optimized Settings")
    print("=" * 60)

    # Based on the target image analysis:
    # - Good shadow recovery (under desks visible)
    # - Vibrant but natural colors (blues pop)
    # - No window blowout
    # - Slight contrast boost

    settings_optimized = ProSettings(
        output_style='natural',
        scene_type='interior',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.38,
        highlight_protection=0.32,
        vibrance=0.3,  # Boost colors slightly
        contrast=0.15,  # Slight contrast
        brightness=0.05  # Tiny brightness boost
    )
    processor_optimized = AutoHDRProProcessor(settings_optimized)

    result_optimized = processor_optimized.process_brackets(brackets)
    cv2.imwrite("/tmp/result_optimized.jpg", result_optimized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("Saved: /tmp/result_optimized.jpg")

    # ============================================
    # QUALITY METRICS
    # ============================================
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)

    for name, result in [
        ("Single", result_single),
        ("Fusion", result_fusion),
        ("Intense", result_intense),
        ("Optimized", result_optimized)
    ]:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        # Dynamic range
        dr = gray.max() - gray.min()

        # Shadow detail (how much detail in dark areas)
        shadow_detail = np.std(gray[gray < 80])

        # Highlight detail (how much detail in bright areas)
        highlight_detail = np.std(gray[gray > 200]) if (gray > 200).any() else 0

        # Color vibrancy
        saturation = np.mean(hsv[:, :, 1])

        # Overall contrast
        contrast = np.std(gray)

        print(f"\n{name}:")
        print(f"  Dynamic range: {dr}")
        print(f"  Shadow detail: {shadow_detail:.1f}")
        print(f"  Highlight detail: {highlight_detail:.1f}")
        print(f"  Saturation: {saturation:.1f}")
        print(f"  Contrast: {contrast:.1f}")

    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("/tmp/result_single.jpg    - Single image processing")
    print("/tmp/result_fusion.jpg    - Bracket fusion (recommended)")
    print("/tmp/result_intense.jpg   - Intense mode")
    print("/tmp/result_optimized.jpg - Optimized for this scene")
    print("\nCompare these to the target image!")

if __name__ == '__main__':
    main()
