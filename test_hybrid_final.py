"""
AutoHDR Clone v4.5.0 - Hybrid Final Test
Combines: Hollywood brightness + Old settings saturation/blacks
"""
import sys
sys.path.insert(0, '/tmp/autohdr-clone/src')

import cv2
import numpy as np
import rawpy
from pathlib import Path
from core.processor_v3 import AutoHDRProProcessor, ProSettings, PROCESSOR_VERSION

ARW_DIR = Path("/Users/home/Downloads/drive-downlo 001 ad-20260205T191553Z-1-")
TARGET_PATH = Path("/Users/home/Downloads/processed_hdr_1770354016632.jpg")

def load_arw(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def calculate_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    return {
        'mean_brightness': np.mean(gray),
        'contrast': np.std(gray),
        'saturation': np.mean(hsv[:, :, 1]),
        'blacks_pct': np.sum(L < 25) / L.size * 100,
    }

def compare_to_target(result, target):
    if result.shape != target.shape:
        target = cv2.resize(target, (result.shape[1], result.shape[0]))
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_e = np.sqrt(np.sum((result_lab - target_lab)**2, axis=2))
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result_norm = (result_gray - np.mean(result_gray)) / (np.std(result_gray) + 1e-6)
    target_norm = (target_gray - np.mean(target_gray)) / (np.std(target_gray) + 1e-6)
    correlation = np.mean(result_norm * target_norm)
    hist_result = cv2.calcHist([result_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_target = cv2.calcHist([target_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(hist_result, hist_target, cv2.HISTCMP_CORREL)
    return {'delta_e': np.mean(delta_e), 'correlation': correlation, 'hist_corr': hist_corr,
            'score': correlation * 50 + hist_corr * 50}

def main():
    print("=" * 70)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - HYBRID FINAL TEST")
    print("Combining: Hollywood brightness + saturation/blacks preservation")
    print("=" * 70)

    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))
    target_metrics = calculate_metrics(target)

    print(f"\nTARGET: B={target_metrics['mean_brightness']:.1f}, S={target_metrics['saturation']:.1f}, Blk={target_metrics['blacks_pct']:.1f}%")

    results = []

    # HYBRID 1: Hollywood + high vibrance + reduced processing
    print("\n" + "-" * 50)
    print("HYBRID 1: Hollywood + vibrance=0.65 (from old best)")

    settings_h1 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.25,  # Reduced
        two_tier_local_strength=0.15,   # Reduced
        use_histogram_params=True,
        hdr_strength=0.32,  # From old best
        local_contrast=0.15,
        shadow_recovery=0.12,  # From old best
        highlight_protection=0.5,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.15,
        vibrance=0.65,  # From old best
        brightness=-0.18,  # Adjusted for Hollywood lifting
        denoise=True,
        denoise_strength=0.45,
        use_hollywood_grading=True,
        use_color_wheels=True,
        shadow_color_shift=(0.015, 0.008),
        use_hollywood_s_curve=True,
        hollywood_shadow_lift=0.04,  # Reduced
        hollywood_midtone_contrast=1.06,  # Reduced
        hollywood_highlight_compress=0.03,
        lut_style='none',
    )

    processor = AutoHDRProProcessor(settings_h1)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Hybrid 1", metrics, comp, result, settings_h1))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")

    # HYBRID 2: Minimal Hollywood, focus on saturation
    print("\n" + "-" * 50)
    print("HYBRID 2: Minimal Hollywood + saturation focus")

    settings_h2 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.2,
        two_tier_local_strength=0.12,
        use_histogram_params=True,
        hdr_strength=0.35,
        local_contrast=0.18,
        shadow_recovery=0.15,
        highlight_protection=0.45,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.3,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.18,
        vibrance=0.7,
        brightness=-0.12,
        contrast=0.08,
        denoise=True,
        denoise_strength=0.45,
        use_hollywood_grading=True,
        use_color_wheels=True,
        use_hollywood_s_curve=False,  # Off
        lut_style='golden_hour',
        lut_intensity=0.2,
    )

    processor = AutoHDRProProcessor(settings_h2)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Hybrid 2", metrics, comp, result, settings_h2))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")

    # HYBRID 3: Balanced approach
    print("\n" + "-" * 50)
    print("HYBRID 3: Balanced (targeting all metrics)")

    settings_h3 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.22,
        two_tier_local_strength=0.15,
        use_histogram_params=True,
        hdr_strength=0.38,
        local_contrast=0.2,
        shadow_recovery=0.14,
        highlight_protection=0.42,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.35,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.16,
        vibrance=0.68,
        brightness=-0.1,
        contrast=0.05,
        denoise=True,
        denoise_strength=0.45,
        use_hollywood_grading=True,
        use_color_wheels=True,
        shadow_color_shift=(0.01, 0.005),
        use_hollywood_s_curve=True,
        hollywood_shadow_lift=0.035,
        hollywood_midtone_contrast=1.05,
        hollywood_highlight_compress=0.025,
        lut_style='professional_clean',
        lut_intensity=0.15,
    )

    processor = AutoHDRProProcessor(settings_h3)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Hybrid 3", metrics, comp, result, settings_h3))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")

    # HYBRID 4: Maximum saturation preservation
    print("\n" + "-" * 50)
    print("HYBRID 4: Maximum saturation + blacks")

    settings_h4 = ProSettings(
        output_style='natural',
        use_adaptive_processing=False,
        auto_white_balance=True,
        use_two_tier_tone_mapping=False,  # Off
        use_histogram_params=False,
        hdr_strength=0.32,
        local_contrast=0.15,
        shadow_recovery=0.1,
        highlight_protection=0.55,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.12,
        vibrance=0.65,
        brightness=-0.2,
        denoise=True,
        denoise_strength=0.45,
        use_hollywood_grading=True,
        use_color_wheels=True,
        shadow_color_shift=(0.02, 0.01),
        use_hollywood_s_curve=False,
        lut_style='golden_hour',
        lut_intensity=0.25,
    )

    processor = AutoHDRProProcessor(settings_h4)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Hybrid 4", metrics, comp, result, settings_h4))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}% | Score: {comp['score']:.1f}%")

    # RESULTS
    print("\n" + "=" * 70)
    print("RESULTS (sorted by score)")
    print("=" * 70)

    results.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (name, metrics, comp, _, _) in enumerate(results):
        b_diff = abs(metrics['mean_brightness'] - 106.8)
        s_diff = abs(metrics['saturation'] - 50.4)
        blk_diff = abs(metrics['blacks_pct'] - 10.0)
        print(f"\n  {i+1}. {name}: {comp['score']:.1f}%")
        print(f"     B={metrics['mean_brightness']:.1f} ({b_diff:.1f} off)")
        print(f"     S={metrics['saturation']:.1f} ({s_diff:.1f} off)")
        print(f"     Blk={metrics['blacks_pct']:.1f}% ({blk_diff:.1f} off)")

    # Save best result
    best_name, best_metrics, best_comp, best_result, best_settings = results[0]
    cv2.imwrite("/tmp/autohdr_v45_BEST.jpg", best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("\n" + "=" * 70)
    print(f"BEST: {best_name} - {best_comp['score']:.1f}%")
    print(f"Saved to: /tmp/autohdr_v45_BEST.jpg")
    print("=" * 70)

    # Also save all for visual comparison
    for name, _, _, result, _ in results:
        safe_name = name.replace(" ", "_").lower()
        cv2.imwrite(f"/tmp/autohdr_{safe_name}.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("\nAll outputs saved to /tmp/autohdr_*.jpg")

if __name__ == '__main__':
    main()
