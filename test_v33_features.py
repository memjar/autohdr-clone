"""
Test v3.3.0 features: Auto Dodge/Burn + 7-Zone Luminosity System
"""
import sys
sys.path.insert(0, '/tmp/autohdr-clone/src')

import cv2
import numpy as np
import rawpy
from pathlib import Path
from core.processor_v3 import AutoHDRProProcessor, ProSettings, PROCESSOR_VERSION

# ARW files path
ARW_DIR = Path("/Users/home/Downloads/drive-downlo 001 ad-20260205T191553Z-1-")

def load_arw(path: Path) -> np.ndarray:
    """Load ARW file and convert to BGR"""
    print(f"Loading: {path.name}")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=8
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def analyze_image(name: str, img: np.ndarray):
    """Analyze image quality metrics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Overall stats
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Zone analysis (7-zone)
    blacks = L[L < 25]
    deep_shadows = L[(L >= 25) & (L < 50)]
    shadows = L[(L >= 50) & (L < 100)]
    midtones = L[(L >= 100) & (L < 155)]
    bright_mids = L[(L >= 155) & (L < 200)]
    highlights = L[(L >= 200) & (L < 235)]
    whites = L[L >= 235]

    print(f"\n{name}:")
    print(f"  Mean brightness: {mean_brightness:.1f}")
    print(f"  Std (contrast): {std_brightness:.1f}")
    print(f"  Zone distribution:")
    print(f"    Blacks (L<25):       {len(blacks):>8} px ({len(blacks)/L.size*100:>5.1f}%)")
    print(f"    Deep shadows (25-50):{len(deep_shadows):>8} px ({len(deep_shadows)/L.size*100:>5.1f}%)")
    print(f"    Shadows (50-100):    {len(shadows):>8} px ({len(shadows)/L.size*100:>5.1f}%)")
    print(f"    Midtones (100-155):  {len(midtones):>8} px ({len(midtones)/L.size*100:>5.1f}%)")
    print(f"    Bright mids (155-200):{len(bright_mids):>7} px ({len(bright_mids)/L.size*100:>5.1f}%)")
    print(f"    Highlights (200-235):{len(highlights):>8} px ({len(highlights)/L.size*100:>5.1f}%)")
    print(f"    Whites (L>=235):     {len(whites):>8} px ({len(whites)/L.size*100:>5.1f}%)")

def main():
    print("=" * 60)
    print(f"Testing AutoHDR Clone Processor v{PROCESSOR_VERSION}")
    print("New Features: Auto Dodge/Burn + 7-Zone Luminosity System")
    print("=" * 60)

    # Load brackets
    arw_files = sorted(ARW_DIR.glob("*.ARW"))
    if not arw_files:
        print("No ARW files found!")
        return

    brackets = [load_arw(f) for f in arw_files]
    print(f"\nLoaded {len(brackets)} brackets")

    # ============================================
    # TEST 1: Standard processing (baseline)
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 1: Standard Processing (no dodge/burn, 3-zone)")
    print("=" * 60)

    settings_standard = ProSettings(
        output_style='natural',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.38,
        auto_dodge_burn=False,  # Disabled
        use_7_zone_system=False,  # Use basic 3-zone
        denoise=True,
        denoise_strength=0.5
    )

    processor_std = AutoHDRProProcessor(settings_standard)
    result_standard = processor_std.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v33_standard.jpg", result_standard, [cv2.IMWRITE_JPEG_QUALITY, 95])
    analyze_image("Standard", result_standard)

    # ============================================
    # TEST 2: With Auto Dodge/Burn
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 2: With Auto Dodge/Burn")
    print("=" * 60)

    settings_dodge_burn = ProSettings(
        output_style='natural',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.38,
        auto_dodge_burn=True,  # Enabled!
        dodge_shadows=0.35,
        burn_highlights=0.2,
        use_7_zone_system=False,
        denoise=True,
        denoise_strength=0.5
    )

    processor_db = AutoHDRProProcessor(settings_dodge_burn)
    result_dodge_burn = processor_db.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v33_dodge_burn.jpg", result_dodge_burn, [cv2.IMWRITE_JPEG_QUALITY, 95])
    analyze_image("Dodge/Burn", result_dodge_burn)

    # ============================================
    # TEST 3: With 7-Zone System
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 3: With 7-Zone Luminosity System")
    print("=" * 60)

    settings_7zone = ProSettings(
        output_style='natural',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.38,
        auto_dodge_burn=False,
        use_7_zone_system=True,  # Enabled!
        zone_blacks=0.2,         # Slight lift to blacks
        zone_deep_shadows=0.35,  # Lift deep shadows
        zone_shadows=0.25,       # Lift shadows
        zone_midtones=0.1,       # Slight midtone boost
        zone_bright_midtones=0.0,
        zone_highlights=-0.1,    # Slight highlight control
        zone_whites=-0.15,       # Control whites
        denoise=True,
        denoise_strength=0.5
    )

    processor_7z = AutoHDRProProcessor(settings_7zone)
    result_7zone = processor_7z.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v33_7zone.jpg", result_7zone, [cv2.IMWRITE_JPEG_QUALITY, 95])
    analyze_image("7-Zone", result_7zone)

    # ============================================
    # TEST 4: FULL COMBO (Dodge/Burn + 7-Zone)
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 4: FULL COMBO (Dodge/Burn + 7-Zone)")
    print("=" * 60)

    settings_full = ProSettings(
        output_style='natural',
        auto_white_balance=True,
        window_pull='natural',
        hdr_strength=0.55,
        local_contrast=0.28,
        shadow_recovery=0.4,
        # Auto dodge/burn for even lighting
        auto_dodge_burn=True,
        dodge_shadows=0.3,
        burn_highlights=0.15,
        # 7-zone for fine control
        use_7_zone_system=True,
        zone_blacks=0.15,
        zone_deep_shadows=0.25,
        zone_shadows=0.15,
        zone_midtones=0.05,
        zone_bright_midtones=0.0,
        zone_highlights=-0.1,
        zone_whites=-0.1,
        # Denoising
        denoise=True,
        denoise_strength=0.5
    )

    processor_full = AutoHDRProProcessor(settings_full)
    result_full = processor_full.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v33_full.jpg", result_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
    analyze_image("Full Combo", result_full)

    # ============================================
    # COMPARISON SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("/tmp/result_v33_standard.jpg   - Baseline (no new features)")
    print("/tmp/result_v33_dodge_burn.jpg - With Auto Dodge/Burn")
    print("/tmp/result_v33_7zone.jpg      - With 7-Zone System")
    print("/tmp/result_v33_full.jpg       - Full Combo (RECOMMENDED)")
    print("\nCompare these to the target image!")

if __name__ == '__main__':
    main()
