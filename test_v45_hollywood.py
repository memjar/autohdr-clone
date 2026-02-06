"""
AutoHDR Clone v4.5.0 - Hollywood Color Grading Test
Tests: Color Wheels, Film S-Curve, LUT Styles
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
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - HOLLYWOOD COLOR GRADING TEST")
    print("=" * 70)

    # Load
    print("\nLoading brackets...")
    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))
    target_metrics = calculate_metrics(target)

    print(f"\nTARGET: B={target_metrics['mean_brightness']:.1f}, S={target_metrics['saturation']:.1f}, Blk={target_metrics['blacks_pct']:.1f}%")

    results = []

    # ============================================
    # TEST 1: Previous best (67.5%) + Hollywood defaults
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 1: Previous Best + Hollywood Color Wheels")

    settings_1 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        use_histogram_params=True,
        hdr_strength=0.35,
        local_contrast=0.18,
        shadow_recovery=0.12,
        highlight_protection=0.5,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.2,
        vibrance=0.6,
        brightness=-0.25,
        denoise=True,
        denoise_strength=0.45,
        # Hollywood grading
        use_hollywood_grading=True,
        use_color_wheels=True,
        shadow_color_shift=(0.02, 0.01),  # Warm shadows
        midtone_color_shift=(0.0, 0.0),
        highlight_color_shift=(-0.01, 0.02),  # Cool highlights
        use_hollywood_s_curve=False,
        lut_style='none',
    )

    processor_1 = AutoHDRProProcessor(settings_1)
    result_1 = processor_1.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v45_test1.jpg", result_1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_1 = calculate_metrics(result_1)
    comp_1 = compare_to_target(result_1, target)
    results.append(("Color Wheels", metrics_1, comp_1))
    print(f"  B={metrics_1['mean_brightness']:.1f}, S={metrics_1['saturation']:.1f}, Blk={metrics_1['blacks_pct']:.1f}% | Score: {comp_1['score']:.1f}%")

    # ============================================
    # TEST 2: Hollywood S-Curve (film-style)
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 2: Hollywood Film S-Curve")

    settings_2 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        use_histogram_params=True,
        hdr_strength=0.35,
        local_contrast=0.18,
        shadow_recovery=0.12,
        highlight_protection=0.5,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=False,  # Disable basic S-curve
        vibrance=0.6,
        brightness=-0.2,
        denoise=True,
        denoise_strength=0.45,
        # Hollywood grading
        use_hollywood_grading=True,
        use_color_wheels=True,
        use_hollywood_s_curve=True,  # Enable Hollywood S-curve
        hollywood_shadow_lift=0.08,
        hollywood_midtone_contrast=1.12,
        hollywood_highlight_compress=0.06,
        lut_style='none',
    )

    processor_2 = AutoHDRProProcessor(settings_2)
    result_2 = processor_2.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v45_test2.jpg", result_2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_2 = calculate_metrics(result_2)
    comp_2 = compare_to_target(result_2, target)
    results.append(("Hollywood S-Curve", metrics_2, comp_2))
    print(f"  B={metrics_2['mean_brightness']:.1f}, S={metrics_2['saturation']:.1f}, Blk={metrics_2['blacks_pct']:.1f}% | Score: {comp_2['score']:.1f}%")

    # ============================================
    # TEST 3: Professional Clean LUT
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 3: Professional Clean LUT")

    settings_3 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        use_histogram_params=True,
        hdr_strength=0.38,
        local_contrast=0.2,
        shadow_recovery=0.15,
        highlight_protection=0.45,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.3,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.18,
        vibrance=0.55,
        brightness=-0.2,
        denoise=True,
        denoise_strength=0.45,
        # Hollywood grading
        use_hollywood_grading=True,
        use_color_wheels=True,
        use_hollywood_s_curve=False,
        lut_style='professional_clean',
        lut_intensity=0.4,
    )

    processor_3 = AutoHDRProProcessor(settings_3)
    result_3 = processor_3.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v45_test3.jpg", result_3, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_3 = calculate_metrics(result_3)
    comp_3 = compare_to_target(result_3, target)
    results.append(("Professional Clean LUT", metrics_3, comp_3))
    print(f"  B={metrics_3['mean_brightness']:.1f}, S={metrics_3['saturation']:.1f}, Blk={metrics_3['blacks_pct']:.1f}% | Score: {comp_3['score']:.1f}%")

    # ============================================
    # TEST 4: Golden Hour LUT
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 4: Golden Hour LUT")

    settings_4 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        use_histogram_params=True,
        hdr_strength=0.38,
        local_contrast=0.2,
        shadow_recovery=0.15,
        highlight_protection=0.45,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.3,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.18,
        vibrance=0.5,
        brightness=-0.22,
        denoise=True,
        denoise_strength=0.45,
        # Hollywood grading
        use_hollywood_grading=True,
        use_color_wheels=False,  # Let LUT handle color
        use_hollywood_s_curve=False,
        lut_style='golden_hour',
        lut_intensity=0.35,
    )

    processor_4 = AutoHDRProProcessor(settings_4)
    result_4 = processor_4.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v45_test4.jpg", result_4, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_4 = calculate_metrics(result_4)
    comp_4 = compare_to_target(result_4, target)
    results.append(("Golden Hour LUT", metrics_4, comp_4))
    print(f"  B={metrics_4['mean_brightness']:.1f}, S={metrics_4['saturation']:.1f}, Blk={metrics_4['blacks_pct']:.1f}% | Score: {comp_4['score']:.1f}%")

    # ============================================
    # TEST 5: Full Hollywood Pipeline
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 5: Full Hollywood Pipeline (wheels + S-curve + LUT)")

    settings_5 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.3,
        two_tier_local_strength=0.2,
        use_histogram_params=True,
        hdr_strength=0.35,
        local_contrast=0.18,
        shadow_recovery=0.12,
        highlight_protection=0.5,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=False,  # Use Hollywood version instead
        vibrance=0.55,
        brightness=-0.22,
        denoise=True,
        denoise_strength=0.45,
        # Full Hollywood pipeline
        use_hollywood_grading=True,
        use_color_wheels=True,
        shadow_color_shift=(0.015, 0.008),
        midtone_color_shift=(0.0, 0.0),
        highlight_color_shift=(-0.008, 0.015),
        use_hollywood_s_curve=True,
        hollywood_shadow_lift=0.06,
        hollywood_midtone_contrast=1.1,
        hollywood_highlight_compress=0.05,
        lut_style='professional_clean',
        lut_intensity=0.3,
    )

    processor_5 = AutoHDRProProcessor(settings_5)
    result_5 = processor_5.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v45_test5.jpg", result_5, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_5 = calculate_metrics(result_5)
    comp_5 = compare_to_target(result_5, target)
    results.append(("Full Hollywood", metrics_5, comp_5))
    print(f"  B={metrics_5['mean_brightness']:.1f}, S={metrics_5['saturation']:.1f}, Blk={metrics_5['blacks_pct']:.1f}% | Score: {comp_5['score']:.1f}%")

    # ============================================
    # BASELINE: Previous Best (67.5%)
    # ============================================
    print("\n" + "-" * 50)
    print("BASELINE: Previous Best (no Hollywood)")

    settings_bl = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        use_histogram_params=True,
        hdr_strength=0.35,
        local_contrast=0.18,
        shadow_recovery=0.12,
        highlight_protection=0.5,
        brightness_equalization=False,
        use_clahe=True,
        clahe_clip_limit=1.2,
        auto_dodge_burn=False,
        apply_s_curve=True,
        s_curve_strength=0.2,
        vibrance=0.6,
        brightness=-0.25,
        denoise=True,
        denoise_strength=0.45,
        use_hollywood_grading=False,  # OFF
    )

    processor_bl = AutoHDRProProcessor(settings_bl)
    result_bl = processor_bl.process_brackets(brackets)
    cv2.imwrite("/tmp/result_v45_baseline.jpg", result_bl, [cv2.IMWRITE_JPEG_QUALITY, 95])
    metrics_bl = calculate_metrics(result_bl)
    comp_bl = compare_to_target(result_bl, target)
    results.append(("Baseline (67.5%)", metrics_bl, comp_bl))
    print(f"  B={metrics_bl['mean_brightness']:.1f}, S={metrics_bl['saturation']:.1f}, Blk={metrics_bl['blacks_pct']:.1f}% | Score: {comp_bl['score']:.1f}%")

    # ============================================
    # RESULTS
    # ============================================
    print("\n" + "=" * 70)
    print("RESULTS RANKING")
    print("=" * 70)

    results.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (name, metrics, comp) in enumerate(results):
        improvement = comp['score'] - 67.5
        imp_str = f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
        print(f"  {i+1}. {name}: {comp['score']:.1f}% ({imp_str})")
        print(f"     B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Blk={metrics['blacks_pct']:.1f}%")

    best_name, best_metrics, best_comp = results[0]
    print("\n" + "=" * 70)
    print(f"BEST: {best_name} - {best_comp['score']:.1f}%")
    print("=" * 70)

if __name__ == '__main__':
    main()
