"""
Test v4.0.0 Multi-Scenario Architecture
Demonstrates adaptive processing based on image analysis
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

def print_scenario_analysis(analysis):
    """Print detailed scenario analysis"""
    print("\n" + "=" * 60)
    print("SCENARIO ANALYSIS (AutoHDR's Secret Sauce)")
    print("=" * 60)

    print(f"\n  Detected Scenarios: {analysis.scenarios}")
    print(f"  Processing Intensity: {analysis.intensity}")

    print(f"\n  LIGHTING PROFILE:")
    print(f"    Mixed Lighting: {analysis.is_mixed_lighting}")
    print(f"    Color Temp Variance: {analysis.color_temp_variance:.1f}")
    print(f"    Dominant Color Temp: {analysis.dominant_color_temp:.0f}K")
    print(f"    Has Tungsten (warm): {analysis.has_tungsten}")
    print(f"    Has Daylight (cool): {analysis.has_daylight}")

    print(f"\n  DYNAMIC RANGE:")
    print(f"    Range: {analysis.dynamic_range:.0f}")
    print(f"    Needs Shadow Lift: {analysis.needs_shadow_lift}")
    print(f"    Needs Highlight Compress: {analysis.needs_highlight_compression}")
    print(f"    High Contrast: {analysis.is_high_contrast}")

    print(f"\n  CONTENT DETECTION:")
    print(f"    Has Windows: {analysis.has_windows} ({analysis.window_percentage*100:.1f}%)")
    print(f"    Has Sky: {analysis.has_sky} ({analysis.sky_percentage*100:.1f}%)")
    print(f"    Has Grass: {analysis.has_grass}")
    print(f"    Interior Zones: {len(analysis.interior_zones)}")
    print(f"    Exterior Zones: {len(analysis.exterior_zones)}")

    print(f"\n  ADAPTIVE PARAMETERS (computed from analysis):")
    print(f"    WB Strength: {analysis.adaptive_wb_strength:.2f}")
    print(f"    Shadow Lift: {analysis.adaptive_shadow_lift:.2f}")
    print(f"    Highlight Compress: {analysis.adaptive_highlight_compress:.2f}")
    print(f"    Saturation Boost: {analysis.adaptive_saturation_boost:.2f}x")
    print(f"    Contrast: {analysis.adaptive_contrast:.2f}x")

    # Zone breakdown
    print(f"\n  ZONE ANALYSIS ({len(analysis.zones)} zones):")
    for i, zone in enumerate(analysis.zones):
        zone_type = []
        if zone.is_window: zone_type.append("WINDOW")
        if zone.is_sky: zone_type.append("SKY")
        if zone.is_interior: zone_type.append("INTERIOR")
        if not zone_type: zone_type.append("neutral")

        print(f"    Zone {i:2d}: Brightness={zone.avg_brightness:5.1f}, "
              f"ColorTemp={zone.color_temp:5.0f}K, "
              f"Type={', '.join(zone_type)}")

def main():
    print("=" * 60)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION}")
    print("Multi-Scenario Architecture Test")
    print("=" * 60)

    # Load brackets
    arw_files = sorted(ARW_DIR.glob("*.ARW"))
    if not arw_files:
        print("No ARW files found!")
        return

    brackets = [load_arw(f) for f in arw_files]
    print(f"\nLoaded {len(brackets)} brackets")

    # Use middle bracket for analysis demo
    test_image = brackets[1]

    # ============================================
    # SCENARIO ANALYSIS DEMO
    # ============================================
    print("\n" + "=" * 60)
    print("Analyzing middle bracket...")
    print("=" * 60)

    # Create processor with adaptive processing enabled
    settings = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        zone_grid_size=4,
        adaptive_wb=True
    )
    processor = AutoHDRProProcessor(settings)

    # Run analysis
    analysis = processor._analyze_scenario(test_image)
    print_scenario_analysis(analysis)

    # ============================================
    # COMPARISON: Adaptive vs Fixed Parameters
    # ============================================
    print("\n" + "=" * 60)
    print("PROCESSING COMPARISON")
    print("=" * 60)

    # Process with FIXED parameters (old way)
    settings_fixed = ProSettings(
        output_style='natural',
        use_adaptive_processing=False,  # DISABLED
        hdr_strength=0.55,
        shadow_recovery=0.38,
        highlight_protection=0.32,
        auto_dodge_burn=True,
        denoise=True
    )
    processor_fixed = AutoHDRProProcessor(settings_fixed)

    print("\n  Processing with FIXED parameters...")
    result_fixed = processor_fixed.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v40_fixed.jpg", result_fixed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Process with ADAPTIVE parameters (new way)
    settings_adaptive = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,  # ENABLED
        adaptive_wb=True,
        auto_dodge_burn=True,
        denoise=True
    )
    processor_adaptive = AutoHDRProProcessor(settings_adaptive)

    print("  Processing with ADAPTIVE parameters...")
    result_adaptive = processor_adaptive.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v40_adaptive.jpg", result_adaptive, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ============================================
    # QUALITY METRICS COMPARISON
    # ============================================
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print("=" * 60)

    for name, result in [("Fixed Params", result_fixed), ("Adaptive Params", result_adaptive)]:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]

        # Calculate metrics
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)

        # Zone distribution
        blacks = np.sum(L < 25) / L.size * 100
        shadows = np.sum((L >= 25) & (L < 100)) / L.size * 100
        midtones = np.sum((L >= 100) & (L < 200)) / L.size * 100
        highlights = np.sum(L >= 200) / L.size * 100

        print(f"\n  {name}:")
        print(f"    Mean Brightness: {mean_brightness:.1f}")
        print(f"    Contrast (std): {std_contrast:.1f}")
        print(f"    Blacks (<25): {blacks:.1f}%")
        print(f"    Shadows (25-100): {shadows:.1f}%")
        print(f"    Midtones (100-200): {midtones:.1f}%")
        print(f"    Highlights (>200): {highlights:.1f}%")

    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("/tmp/result_v40_fixed.jpg    - Fixed parameters (old way)")
    print("/tmp/result_v40_adaptive.jpg - Adaptive parameters (AutoHDR way)")
    print("\nThe adaptive version should have better-balanced exposure")
    print("and more natural mixed-lighting handling!")

if __name__ == '__main__':
    main()
