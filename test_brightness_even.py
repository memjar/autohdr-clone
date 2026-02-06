"""
AutoHDR Clone v4.6.0 - Even Brightness Test
Focus: Match target's even brightness distribution
Issue: Our output is dark in background, uneven lighting
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

    # Brightness evenness: compare quadrants
    h, w = gray.shape
    q1 = np.mean(gray[:h//2, :w//2])  # Top-left
    q2 = np.mean(gray[:h//2, w//2:])  # Top-right
    q3 = np.mean(gray[h//2:, :w//2])  # Bottom-left
    q4 = np.mean(gray[h//2:, w//2:])  # Bottom-right
    evenness = 1 - np.std([q1, q2, q3, q4]) / np.mean([q1, q2, q3, q4])

    return {
        'mean_brightness': np.mean(gray),
        'saturation': np.mean(hsv[:, :, 1]),
        'blacks_pct': np.sum(L < 25) / L.size * 100,
        'evenness': evenness,
        'quadrants': (q1, q2, q3, q4)
    }

def compare_to_target(result, target):
    if result.shape != target.shape:
        target = cv2.resize(target, (result.shape[1], result.shape[0]))
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result_norm = (result_gray - np.mean(result_gray)) / (np.std(result_gray) + 1e-6)
    target_norm = (target_gray - np.mean(target_gray)) / (np.std(target_gray) + 1e-6)
    correlation = np.mean(result_norm * target_norm)
    hist_result = cv2.calcHist([result_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_target = cv2.calcHist([target_gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(hist_result, hist_target, cv2.HISTCMP_CORREL)
    return {'correlation': correlation, 'hist_corr': hist_corr,
            'score': correlation * 50 + hist_corr * 50}

def main():
    print("=" * 70)
    print(f"AutoHDR Clone v{PROCESSOR_VERSION} - EVEN BRIGHTNESS TEST")
    print("Focus: Match target's even brightness distribution")
    print("=" * 70)

    brackets = [load_arw(f) for f in sorted(ARW_DIR.glob("*.ARW"))]
    target = cv2.imread(str(TARGET_PATH))
    target_metrics = calculate_metrics(target)

    print(f"\nTARGET METRICS:")
    print(f"  Brightness: {target_metrics['mean_brightness']:.1f}")
    print(f"  Saturation: {target_metrics['saturation']:.1f}")
    print(f"  Evenness: {target_metrics['evenness']:.3f}")
    print(f"  Quadrants: TL={target_metrics['quadrants'][0]:.0f}, TR={target_metrics['quadrants'][1]:.0f}, BL={target_metrics['quadrants'][2]:.0f}, BR={target_metrics['quadrants'][3]:.0f}")

    results = []

    # ============================================
    # TEST 1: Aggressive CLAHE (key technique)
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 1: Aggressive CLAHE (clipLimit=2.5, grid=10x10)")

    settings_1 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.3,
        two_tier_local_strength=0.2,
        hdr_strength=0.45,
        local_contrast=0.25,
        shadow_recovery=0.35,  # Strong shadow lift
        highlight_protection=0.4,
        brightness_equalization=True,
        equalization_strength=0.5,  # Strong equalization
        use_clahe=True,
        clahe_clip_limit=2.5,  # More aggressive
        clahe_grid_size=10,    # Larger tiles
        auto_dodge_burn=True,  # Enable dodge/burn
        dodge_shadows=0.4,     # Lift shadows strongly
        burn_highlights=0.1,
        apply_s_curve=True,
        s_curve_strength=0.15,
        vibrance=0.55,
        brightness=0.0,  # Don't manually darken
        denoise=True,
        denoise_strength=0.4,
        use_perceptual_processing=True,
        use_hollywood_grading=True,
        use_color_wheels=True,
        use_hollywood_s_curve=False,
        lut_style='none',
    )

    processor = AutoHDRProProcessor(settings_1)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Aggressive CLAHE", metrics, comp, result))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Even={metrics['evenness']:.3f} | Score: {comp['score']:.1f}%")
    print(f"  Quadrants: TL={metrics['quadrants'][0]:.0f}, TR={metrics['quadrants'][1]:.0f}, BL={metrics['quadrants'][2]:.0f}, BR={metrics['quadrants'][3]:.0f}")

    # ============================================
    # TEST 2: Maximum shadow lift
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 2: Maximum Shadow Lift + Dodge/Burn")

    settings_2 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.35,
        two_tier_local_strength=0.25,
        hdr_strength=0.5,
        local_contrast=0.28,
        shadow_recovery=0.5,  # Maximum shadow lift
        highlight_protection=0.35,
        brightness_equalization=True,
        equalization_strength=0.6,
        use_clahe=True,
        clahe_clip_limit=3.0,  # Very aggressive
        clahe_grid_size=8,
        auto_dodge_burn=True,
        dodge_shadows=0.5,  # Maximum dodge
        burn_highlights=0.15,
        apply_s_curve=True,
        s_curve_strength=0.12,
        vibrance=0.5,
        brightness=0.05,  # Slight overall lift
        denoise=True,
        denoise_strength=0.45,
        use_perceptual_processing=True,
        use_hollywood_grading=False,
    )

    processor = AutoHDRProProcessor(settings_2)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Max Shadow Lift", metrics, comp, result))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Even={metrics['evenness']:.3f} | Score: {comp['score']:.1f}%")
    print(f"  Quadrants: TL={metrics['quadrants'][0]:.0f}, TR={metrics['quadrants'][1]:.0f}, BL={metrics['quadrants'][2]:.0f}, BR={metrics['quadrants'][3]:.0f}")

    # ============================================
    # TEST 3: Target brightness matching
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 3: Target Brightness + Equalization")

    settings_3 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.3,
        two_tier_local_strength=0.2,
        hdr_strength=0.48,
        local_contrast=0.25,
        shadow_recovery=0.4,
        highlight_protection=0.38,
        brightness_equalization=True,
        equalization_strength=0.55,
        use_clahe=True,
        clahe_clip_limit=2.8,
        clahe_grid_size=10,
        auto_dodge_burn=True,
        dodge_shadows=0.45,
        burn_highlights=0.12,
        apply_s_curve=True,
        s_curve_strength=0.15,
        vibrance=0.58,
        brightness=0.08,  # Lift to match target
        contrast=0.05,
        denoise=True,
        denoise_strength=0.4,
        use_perceptual_processing=True,
        use_csf_contrast=True,
        csf_mid_boost=0.12,
        use_hollywood_grading=True,
        use_color_wheels=True,
        lut_style='professional_clean',
        lut_intensity=0.15,
    )

    processor = AutoHDRProProcessor(settings_3)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Target Match", metrics, comp, result))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Even={metrics['evenness']:.3f} | Score: {comp['score']:.1f}%")
    print(f"  Quadrants: TL={metrics['quadrants'][0]:.0f}, TR={metrics['quadrants'][1]:.0f}, BL={metrics['quadrants'][2]:.0f}, BR={metrics['quadrants'][3]:.0f}")

    # ============================================
    # TEST 4: Extreme equalization
    # ============================================
    print("\n" + "-" * 50)
    print("TEST 4: Extreme Equalization")

    settings_4 = ProSettings(
        output_style='natural',
        use_adaptive_processing=True,
        auto_white_balance=True,
        use_two_tier_tone_mapping=True,
        two_tier_global_strength=0.4,
        two_tier_local_strength=0.3,
        hdr_strength=0.55,
        local_contrast=0.3,
        shadow_recovery=0.55,
        highlight_protection=0.3,
        brightness_equalization=True,
        equalization_strength=0.7,  # Very strong
        use_clahe=True,
        clahe_clip_limit=3.5,  # Extreme
        clahe_grid_size=12,
        auto_dodge_burn=True,
        dodge_shadows=0.55,
        burn_highlights=0.2,
        apply_s_curve=True,
        s_curve_strength=0.1,
        vibrance=0.5,
        brightness=0.1,
        denoise=True,
        denoise_strength=0.45,
        use_perceptual_processing=False,
        use_hollywood_grading=False,
    )

    processor = AutoHDRProProcessor(settings_4)
    result = processor.process_brackets(brackets)
    metrics = calculate_metrics(result)
    comp = compare_to_target(result, target)
    results.append(("Extreme Equal", metrics, comp, result))
    print(f"  B={metrics['mean_brightness']:.1f}, S={metrics['saturation']:.1f}, Even={metrics['evenness']:.3f} | Score: {comp['score']:.1f}%")
    print(f"  Quadrants: TL={metrics['quadrants'][0]:.0f}, TR={metrics['quadrants'][1]:.0f}, BL={metrics['quadrants'][2]:.0f}, BR={metrics['quadrants'][3]:.0f}")

    # RESULTS
    print("\n" + "=" * 70)
    print("RESULTS (sorted by score)")
    print("=" * 70)

    results.sort(key=lambda x: x[2]['score'], reverse=True)
    for i, (name, metrics, comp, _) in enumerate(results):
        print(f"\n  {i+1}. {name}: {comp['score']:.1f}%")
        print(f"     B={metrics['mean_brightness']:.1f} (target {target_metrics['mean_brightness']:.1f})")
        print(f"     S={metrics['saturation']:.1f} (target {target_metrics['saturation']:.1f})")
        print(f"     Evenness={metrics['evenness']:.3f} (target {target_metrics['evenness']:.3f})")

    # Save best
    best_name, best_metrics, best_comp, best_result = results[0]
    cv2.imwrite("/tmp/autohdr_even_BEST.jpg", best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite("/Users/home/Desktop/autohdr_even_BEST.jpg", best_result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save all for comparison
    for name, _, _, result in results:
        safe_name = name.replace(" ", "_").lower()
        cv2.imwrite(f"/Users/home/Desktop/autohdr_{safe_name}.jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n\nBEST: {best_name} - {best_comp['score']:.1f}%")
    print("All outputs saved to Desktop")

if __name__ == '__main__':
    main()
